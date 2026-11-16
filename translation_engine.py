"""Text translation engine.

Adapters:
  indic-translate-mlx     bodhan-ai/indic-translate (or any mlx-lm chat model)   MLX on Apple Silicon
  indic-translate-torch   bodhan-ai/indic-translate (Gemma 4 E4B fine-tune)      PyTorch MPS / CUDA / CPU

Indic Translate is a single-purpose model: one user turn holding one instruction and one source
text, no system message, greedy decoding. Its quality degrades silently when the prompt drifts
from that contract, so every request is built here by `build_prompt`:

    Translate the following text into {Target}[ (Roman)| (code-mixed)]:\n\n{source}
    Transliterate the following text into {Target}[ (Roman)]:\n\n{source}

The source language is never named (the model card asks for this; the model detects it).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

from model_manager import ModelError

logger = logging.getLogger("translation_engine")

# code -> (English name, prompt name, script). Prompt names are the strings the model was trained on
# (model card, "Supported languages"); for multi-script languages they also select the script.
LANGUAGES: Dict[str, Tuple[str, str, str]] = {
    "en": ("English", "English", "Latin"),
    "as": ("Assamese", "Assamese", "Bengali-Assamese"),
    "bn": ("Bengali", "Bengali", "Bengali"),
    "brx": ("Bodo", "Bodo", "Devanagari"),
    "doi": ("Dogri", "Dogri", "Devanagari"),
    "gom": ("Konkani", "Konkani", "Devanagari"),
    "gu": ("Gujarati", "Gujarati", "Gujarati"),
    "hi": ("Hindi", "Hindi", "Devanagari"),
    "kn": ("Kannada", "Kannada", "Kannada"),
    "ks": ("Kashmiri", "Kashmiri (Perso-Arabic script)", "Perso-Arabic"),
    "mai": ("Maithili", "Maithili", "Devanagari"),
    "ml": ("Malayalam", "Malayalam", "Malayalam"),
    "mr": ("Marathi", "Marathi", "Devanagari"),
    "mni": ("Manipuri", "Manipuri (Meitei script)", "Meetei Mayek"),
    "mni_Beng": ("Manipuri", "Manipuri (Bengali script)", "Bengali"),
    "ne": ("Nepali", "Nepali", "Devanagari"),
    "or": ("Odia", "Odia", "Odia"),
    "pa": ("Punjabi", "Punjabi", "Gurmukhi"),
    "sa": ("Sanskrit", "Sanskrit", "Devanagari"),
    "sat": ("Santali", "Santali", "Ol Chiki"),
    "sd": ("Sindhi", "Sindhi (Devanagari script)", "Devanagari"),
    "sd_Arab": ("Sindhi", "Sindhi (Perso-Arabic script)", "Perso-Arabic"),
    "ta": ("Tamil", "Tamil", "Tamil"),
    "te": ("Telugu", "Telugu", "Telugu"),
    "ur": ("Urdu", "Urdu", "Perso-Arabic"),
}

# Other spellings accepted for `target_language`: FLORES-200 codes, ISO 639-2/3, common names.
_ALIASES = {
    "eng_latn": "en", "eng": "en", "asm_beng": "as", "asm": "as", "ben_beng": "bn", "ben": "bn",
    "bangla": "bn", "brx_deva": "brx", "doi_deva": "doi", "gom_deva": "gom", "kok": "gom",
    "guj_gujr": "gu", "guj": "gu", "hin_deva": "hi", "hin": "hi", "kan_knda": "kn", "kan": "kn",
    "kas_arab": "ks", "kas": "ks", "mai_deva": "mai", "mal_mlym": "ml", "mal": "ml",
    "mar_deva": "mr", "mar": "mr", "mni_mtei": "mni", "meitei": "mni", "mni_beng": "mni_Beng",
    "npi_deva": "ne", "npi": "ne", "nep": "ne", "ory_orya": "or", "ory": "or", "odi": "or", "oriya": "or",
    "pan_guru": "pa", "pan": "pa", "san_deva": "sa", "san": "sa", "sat_olck": "sat",
    "snd_deva": "sd", "snd": "sd", "snd_arab": "sd_Arab", "tam_taml": "ta", "tam": "ta",
    "tel_telu": "te", "tel": "te", "urd_arab": "ur", "urd": "ur",
}
for _code, (_name, _prompt, _script) in LANGUAGES.items():
    _ALIASES.setdefault(_code.lower(), _code)
    _ALIASES.setdefault(_prompt.lower(), _code)
    _ALIASES.setdefault(_name.lower(), _code)     # bare name -> the card's default script

TASKS = ("translate", "transliterate")
SCRIPTS = ("native", "roman", "code-mixed")
_QUALIFIER_RE = re.compile(r"\s*\((roman|romanized|romanised|latin|code[- ]?mixed)\)\s*$", re.I)


def resolve_language(value: Optional[str]) -> str:
    """Map a code, FLORES code or name to a LANGUAGES key."""
    key = (value or "").strip()
    if not key:
        raise ModelError("target_language is required (e.g. 'hi', 'Telugu', 'tam_Taml').", param="target_language")
    code = _ALIASES.get(key.lower()) or _ALIASES.get(key.lower().replace("-", "_"))
    if code is None:
        raise ModelError(f"Unsupported target_language '{value}'. See GET /v1/translations/languages.",
                         param="target_language", code="unsupported_language")
    return code


def normalize_script(script: Optional[str]) -> str:
    s = (script or "native").strip().lower().replace("_", "-")
    if s in ("roman", "romanized", "romanised", "latin"):
        return "roman"
    if s in ("code-mixed", "codemixed", "mixed", "code mixed"):
        return "code-mixed"
    if s in ("native", ""):
        return "native"
    raise ModelError(f"script must be one of {', '.join(SCRIPTS)}.", param="script")


def target_spec(target_language: str, task: str = "translate", script: Optional[str] = None) -> dict:
    """Resolve the target into {code, name, prompt_name, script, task}. A '(Roman)' / '(code-mixed)'
    suffix on the language name is accepted as an alternative to `script`."""
    task = (task or "translate").strip().lower()
    if task not in TASKS:
        raise ModelError(f"task must be one of {', '.join(TASKS)}.", param="task")
    raw = (target_language or "").strip()
    m = _QUALIFIER_RE.search(raw)
    if m:
        raw = raw[:m.start()]
        suffix = normalize_script(m.group(1))
        if script and normalize_script(script) not in ("native", suffix):
            raise ModelError("target_language suffix and script disagree.", param="script")
        script = suffix
    script = normalize_script(script)
    code = resolve_language(raw)
    name, prompt_name, native_script = LANGUAGES[code]
    if code == "en" and script != "native":
        raise ModelError("English has no Romanized or code-mixed form; use script=native.", param="script")
    if task == "transliterate":
        if code == "en":
            raise ModelError("Transliteration targets an Indic language (native or Roman script).", param="target_language")
        if script == "code-mixed":
            raise ModelError("Transliteration supports script=native or roman, not code-mixed.", param="script")
    if script == "roman":
        prompt_name += " (Roman)"
    elif script == "code-mixed":
        prompt_name += " (code-mixed)"
    return {"code": code, "name": name, "prompt_name": prompt_name, "task": task,
            "script": script, "output_script": "Latin" if script == "roman" else native_script}


def build_prompt(text: str, target: dict) -> str:
    verb = "Transliterate" if target["task"] == "transliterate" else "Translate"
    return f"{verb} the following text into {target['prompt_name']}:\n\n{text.strip()}"


_CONTRACT_RE = re.compile(r"^(Translate|Transliterate) the following text into (.+?):\n\n", re.S)


def parse_prompt(prompt: str) -> Optional[dict]:
    """If a chat message already follows the prompt contract, return its target (for headers/logging)."""
    m = _CONTRACT_RE.match(prompt or "")
    if not m:
        return None
    try:
        return target_spec(m.group(2), m.group(1).lower())
    except ModelError:
        return None


def default_max_tokens(source_tokens: int) -> int:
    """Indic scripts tokenise ~1.5-2x English (model card); leave generous room, cap at the 8k document budget."""
    return max(256, min(8192, int(source_tokens * 2.5) + 64))


def catalog() -> dict:
    return {
        "languages": {code: {"name": n, "prompt_name": p, "script": s} for code, (n, p, s) in LANGUAGES.items()},
        "tasks": list(TASKS),
        "scripts": list(SCRIPTS),
    }


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------
class TranslationAdapter:
    name = "base"
    engine = "base"
    context_length = 32768     # validated context of indic-translate

    def load(self, local_dir: str, spec: dict, token: Optional[str]) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            setattr(self, attr, None)

    def encode_chat(self, user_content: str) -> List[int]:
        """Render one user turn with the packaged chat template: no system turn and thinking explicitly
        off, giving exactly `<bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n` (model card). Some
        Gemma 4 templates (and mlx-lm's wrapper) default thinking on, which injects a system turn."""
        messages = [{"role": "user", "content": user_content}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def stream(self, prompt_ids: List[int], max_tokens: int, temperature: float, top_p: float,
               cancel: Optional[threading.Event]) -> Iterator[Tuple[str, Optional[str]]]:
        """Yield (text_delta, finish_reason); finish_reason is set on the last item only."""
        raise NotImplementedError


_STOP_STRINGS = ("<turn|>", "<eos>", "<end_of_turn>", "<bos>", "<pad>")
_THOUGHT_OPEN, _THOUGHT_CLOSE = "<|channel>", "<channel|>"


def _strip_stops(text: str) -> str:
    for s in _STOP_STRINGS:
        text = text.replace(s, "")
    return text


class _ThoughtFilter:
    """Drops a leading `<|channel>thought ... <channel|>` block from streamed output."""

    def __init__(self):
        self.buf, self.mode = "", "start"       # start -> (thinking ->) pass

    def feed(self, piece: str) -> str:
        if self.mode == "pass":
            return piece
        self.buf += piece
        if self.mode == "start":
            head = self.buf.lstrip()
            if head.startswith(_THOUGHT_OPEN):
                self.mode = "thinking"
            elif _THOUGHT_OPEN.startswith(head):
                return ""                       # could still become the opening tag
            else:
                self.mode, out, self.buf = "pass", self.buf, ""
                return out
        cut = self.buf.find(_THOUGHT_CLOSE)
        if cut < 0:
            return ""
        out = self.buf[cut + len(_THOUGHT_CLOSE):].lstrip()
        self.mode, self.buf = "pass", ""
        return out

    def flush(self) -> str:
        out = self.buf if self.mode == "start" else ""      # an unfinished thought is dropped
        self.buf = ""
        return out


class IndicTranslateMLX(TranslationAdapter):
    """mlx-lm. Loads the original Hugging Face Gemma 4 checkpoint directly (vision/audio towers are
    dropped by mlx-lm's sanitizer); `quantization` 4bit/8bit quantizes at load time."""
    name = "indic-translate-mlx"
    engine = "mlx"

    def load(self, local_dir, spec, token):
        try:
            from mlx_lm import load
        except ImportError:
            raise ModelError("mlx-lm is not installed. Run: pip install mlx-lm", status=501, code="backend_unavailable")
        self.model, self.tokenizer, config = load(local_dir, return_config=True)
        bits = {"4bit": 4, "8bit": 8}.get(spec.get("quantization") or "")
        if bits and not config.get("quantization"):
            import mlx.core as mx
            from mlx_lm.utils import quantize_model
            self.model, _ = quantize_model(self.model, config, group_size=64, bits=bits)
            mx.eval(self.model.parameters())
            logger.info("Quantized %s to %d-bit at load", spec["id"], bits)
        for stop in ("<turn|>", "<end_of_turn>"):
            try:
                if stop in self.tokenizer.get_vocab():
                    self.tokenizer.add_eos_token(stop)
            except Exception:
                pass

    def stream(self, prompt_ids, max_tokens, temperature, top_p, cancel):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=float(temperature or 0.0), top_p=float(top_p) if top_p and top_p < 1.0 else 0.0)
        finish = "length"
        for resp in stream_generate(self.model, self.tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            if resp.finish_reason:
                finish = resp.finish_reason
            text = _strip_stops(resp.text)
            if text:
                yield text, None
        yield "", finish


class IndicTranslateTorch(TranslationAdapter):
    """Reference path from the model card: transformers AutoModelForImageTextToText in bfloat16."""
    name = "indic-translate-torch"
    engine = "pytorch"

    def load(self, local_dir, spec, token):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(local_dir)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = AutoModelForImageTextToText.from_pretrained(
            local_dir, dtype=dtype, attn_implementation="sdpa").to(self.device).eval()

    def stream(self, prompt_ids, max_tokens, temperature, top_p, cancel):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(cancel is not None and cancel.is_set())

        # Special tokens are kept so a thinking block stays recognisable; stop tokens are stripped below.
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        kwargs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_tokens,
                      streamer=streamer, use_cache=True, stopping_criteria=StoppingCriteriaList([_Cancel()]))
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p or 1.0))
        else:
            kwargs.update(do_sample=False)
        result: dict = {}

        def run():
            try:
                with torch.inference_mode():
                    result["out"] = self.model.generate(**kwargs)
            except Exception as e:          # surface in the consumer thread
                result["error"] = e
                streamer.end()

        worker = threading.Thread(target=run, daemon=True, name="translate-generate")
        worker.start()
        for piece in streamer:
            piece = _strip_stops(piece)
            if piece:
                yield piece, None
        worker.join()
        if "error" in result:
            raise result["error"]
        n_new = result["out"].shape[-1] - len(prompt_ids) if "out" in result else 0
        if cancel is not None and cancel.is_set():
            finish = "cancelled"
        else:
            finish = "length" if n_new >= max_tokens else "stop"
        yield "", finish


ADAPTERS = {cls.name: cls for cls in (IndicTranslateMLX, IndicTranslateTorch)}


def create_adapter(spec: dict) -> TranslationAdapter:
    cls = ADAPTERS.get(spec.get("adapter") or "")
    if cls is None:
        raise ModelError(f"Unknown translation adapter '{spec.get('adapter')}' for model '{spec['id']}'.",
                         status=501, code="adapter_not_found")
    return cls()


def catalog_for(spec: dict) -> dict:
    return catalog()


# ----------------------------------------------------------------------------
# Generation (model-agnostic; runs on the GPU worker thread)
# ----------------------------------------------------------------------------
def prepare(adapter: TranslationAdapter, user_content: str, max_tokens: Optional[int],
            source_text: Optional[str] = None) -> Tuple[List[int], int]:
    """Tokenise the request and pick max_tokens. Returns (prompt_ids, max_tokens)."""
    ids = adapter.encode_chat(user_content)
    limit = adapter.context_length
    if len(ids) >= limit:
        raise ModelError(f"Input is {len(ids)} tokens; the model's context is {limit}. Split the document.",
                         param="input", code="context_length_exceeded")
    if not max_tokens:
        max_tokens = default_max_tokens(adapter.count_tokens(source_text if source_text is not None else user_content))
    return ids, max(1, min(int(max_tokens), limit - len(ids)))


def generate(adapter: TranslationAdapter, prompt_ids: List[int], max_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0, cancel: Optional[threading.Event] = None, on_delta=None) -> dict:
    """Run one generation. `on_delta(text)` receives streamed pieces. Returns text, finish_reason, usage, stats."""
    t0 = time.time()
    ttft, parts, finish = None, [], "stop"
    thoughts = _ThoughtFilter()

    def push(delta: str) -> None:
        nonlocal ttft
        if not delta:
            return
        if ttft is None:
            ttft = time.time() - t0
        parts.append(delta)
        if on_delta is not None:
            on_delta(delta)

    for delta, reason in adapter.stream(prompt_ids, max_tokens, temperature, top_p, cancel):
        push(thoughts.feed(delta))
        if reason:
            finish = reason
    push(thoughts.flush())
    text = "".join(parts).strip()
    elapsed = time.time() - t0
    completion_tokens = adapter.count_tokens(text) if text else 0
    return {
        "text": text,
        "finish_reason": finish,
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": completion_tokens,
                  "total_tokens": len(prompt_ids) + completion_tokens},
        "stats": {"engine": adapter.engine, "total_s": round(elapsed, 3), "ttft_s": round(ttft or elapsed, 3),
                  "tokens_per_s": round(completion_tokens / max(elapsed - (ttft or 0), 1e-3), 1)},
    }
