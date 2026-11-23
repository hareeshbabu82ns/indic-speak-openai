"""Text-to-Speech engine.

Adapters wrap one model family each behind a common interface; `synthesize` and
`stream_synthesis` add the model-agnostic parts (sentence chunking, silence
trimming, speed, 24 kHz output).

Adapters:
  indic-speak-mlx    iniquitous/indic-speak-mlx-bf16   MLX LM + SNAC/Vocos, real-time streaming
  indic-speak-torch  bodhan-ai/indic-speak             PyTorch (MPS / CUDA / CPU)
  kokoro             hexgrad/Kokoro-82M                optional `kokoro` package
  transformers-tts   any HF text-to-speech pipeline    MMS-TTS / VITS, Bark, ...
  remote-openai-tts  another OpenAI-compatible server  proxied over HTTP (e.g. Vāgdhenu)
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from audio_utils import TTS_SR, change_speed, resample
from model_manager import ModelError

logger = logging.getLogger("tts_engine")

SR = TTS_SR

# ----------------------------------------------------------------------------
# Indic-Speak catalog (44 first-party voices, 22 languages)
# ----------------------------------------------------------------------------
VOICE_CATALOG = {
    "Prastuti": {"gender": "Female", "lang": "as", "lang_name": "Assamese", "desc": "Warm mid register with animated inflection"},
    "Ankur": {"gender": "Male", "lang": "as", "lang_name": "Assamese", "desc": "Deep voice with resonant chest tone"},
    "Ishita": {"gender": "Female", "lang": "bn", "lang_name": "Bengali", "desc": "Bright and melodious tone"},
    "Sourav": {"gender": "Male", "lang": "bn", "lang_name": "Bengali", "desc": "Confident, clear articulation"},
    "Gwrbw": {"gender": "Female", "lang": "brx", "lang_name": "Bodo", "desc": "Gentle, natural flow"},
    "Sansuma": {"gender": "Male", "lang": "brx", "lang_name": "Bodo", "desc": "Steady, warm cadence"},
    "Preeti": {"gender": "Female", "lang": "doi", "lang_name": "Dogri", "desc": "Sweet, expressive storytelling tone"},
    "Sham": {"gender": "Male", "lang": "doi", "lang_name": "Dogri", "desc": "Rich, grounded baritone"},
    "Dhara": {"gender": "Female", "lang": "gu", "lang_name": "Gujarati", "desc": "Lively, rhythmic delivery"},
    "Parth": {"gender": "Male", "lang": "gu", "lang_name": "Gujarati", "desc": "Clear, friendly register"},
    "Kavya": {"gender": "Female", "lang": "hi", "lang_name": "Hindi", "desc": "Clear alto, versatile and expressive"},
    "Amit": {"gender": "Male", "lang": "hi", "lang_name": "Hindi", "desc": "Deep, confident, engaging voice"},
    "Deepika": {"gender": "Female", "lang": "kn", "lang_name": "Kannada", "desc": "Crisp, melodious pronunciation"},
    "Adarsh": {"gender": "Male", "lang": "kn", "lang_name": "Kannada", "desc": "Resonant, authoritative male tone"},
    "Zoon": {"gender": "Female", "lang": "ks", "lang_name": "Kashmiri", "desc": "Soft, lyrical cadence"},
    "Ishfaq": {"gender": "Male", "lang": "ks", "lang_name": "Kashmiri", "desc": "Deep, expressive regional tone"},
    "Anjali": {"gender": "Female", "lang": "kok", "lang_name": "Konkani", "desc": "Gentle, rhythmic inflection"},
    "Sandeep": {"gender": "Male", "lang": "kok", "lang_name": "Konkani", "desc": "Warm, conversational register"},
    "Vaidehi": {"gender": "Female", "lang": "mai", "lang_name": "Maithili", "desc": "Soft, classic literary tone"},
    "Madhukar": {"gender": "Male", "lang": "mai", "lang_name": "Maithili", "desc": "Grounded, traditional cadence"},
    "Lakshmi": {"gender": "Female", "lang": "ml", "lang_name": "Malayalam", "desc": "Fluid, expressive delivery"},
    "Kiran": {"gender": "Male", "lang": "ml", "lang_name": "Malayalam", "desc": "Crisp, energetic voice"},
    "Thoibi": {"gender": "Female", "lang": "mni", "lang_name": "Manipuri", "desc": "Bright, clear soprano"},
    "Chaoba": {"gender": "Male", "lang": "mni", "lang_name": "Manipuri", "desc": "Steady, warm baritone"},
    "Anagha": {"gender": "Female", "lang": "mr", "lang_name": "Marathi", "desc": "Clear, confident narration"},
    "Chinmay": {"gender": "Male", "lang": "mr", "lang_name": "Marathi", "desc": "Deep, engaging articulation"},
    "Srijana": {"gender": "Female", "lang": "ne", "lang_name": "Nepali", "desc": "Warm, melodious inflection"},
    "Sagar": {"gender": "Male", "lang": "ne", "lang_name": "Nepali", "desc": "Strong, natural tone"},
    "Itishree": {"gender": "Female", "lang": "or", "lang_name": "Odia", "desc": "Gentle, rhythmic flow"},
    "Akash": {"gender": "Male", "lang": "or", "lang_name": "Odia", "desc": "Resonant, confident delivery"},
    "Kaur": {"gender": "Female", "lang": "pa", "lang_name": "Punjabi", "desc": "Energetic, clear and bright"},
    "Manpreet": {"gender": "Male", "lang": "pa", "lang_name": "Punjabi", "desc": "Bold, robust male tone"},
    "Bharati": {"gender": "Female", "lang": "sa", "lang_name": "Sanskrit", "desc": "Pristine classical diction"},
    "Aryaman": {"gender": "Male", "lang": "sa", "lang_name": "Sanskrit", "desc": "Solemn, resonant chanting tone"},
    "Phulmani": {"gender": "Female", "lang": "sat", "lang_name": "Santali", "desc": "Gentle and authentic flow"},
    "Sibu": {"gender": "Male", "lang": "sat", "lang_name": "Santali", "desc": "Warm, grounded cadence"},
    "Moomal": {"gender": "Female", "lang": "sd", "lang_name": "Sindhi", "desc": "Bright, balanced register"},
    "Rano": {"gender": "Male", "lang": "sd", "lang_name": "Sindhi", "desc": "Quick, animated, low-mid register"},
    "Anitha": {"gender": "Female", "lang": "ta", "lang_name": "Tamil", "desc": "Warm alto, quick, composed authority"},
    "Arun": {"gender": "Male", "lang": "ta", "lang_name": "Tamil", "desc": "Very deep, commanding, dramatic presence"},
    "Sravani": {"gender": "Female", "lang": "te", "lang_name": "Telugu", "desc": "Bright register, vivid and engaging"},
    "Vamsi": {"gender": "Male", "lang": "te", "lang_name": "Telugu", "desc": "Rich low register, warm gravitas"},
    "Saba": {"gender": "Female", "lang": "ur", "lang_name": "Urdu", "desc": "Velvet alto, literary polish"},
    "Zaid": {"gender": "Male", "lang": "ur", "lang_name": "Urdu", "desc": "Deep, refined, formal weight"},
}

OPENAI_VOICE_MAP = {
    "alloy": "Amit", "echo": "Arun", "fable": "Deepika",
    "onyx": "Vamsi", "nova": "Kavya", "shimmer": "Sravani",
}

# Style names accepted by the API. NEUTRAL means "no style tag".
AVAILABLE_STYLES = [
    "NEUTRAL", "HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST",
    "news", "AIR style news", "TV style news", "advertisements", "children's stories",
    "educational lecture", "Customer Care", "single person narration audiobook",
]
# The checkpoint was trained on these exact tag spellings (see the repo's voices.md).
_STYLE_TAGS = {s.lower(): s for s in AVAILABLE_STYLES[7:]}
_STYLE_TAGS.update({e: e for e in ("happy", "sad", "anger", "fear", "surprise", "disgust")})


def normalize_style(style: Optional[str]) -> str:
    """API style name -> the tag the model was trained with ('' = no style block)."""
    s = (style or "").strip()
    if not s or s.lower() in ("neutral", "none", "default"):
        return ""
    return _STYLE_TAGS.get(s.lower(), s)


# ----------------------------------------------------------------------------
# Text chunking
# ----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[।॥.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:،—])\s+")
MAX_SENTENCE_CHARS = 220   # ~18 s of speech; a single generation is capped near 25 s


def split_text(text: str, max_chars: int = MAX_SENTENCE_CHARS) -> List[str]:
    """Split text into sentence-sized chunks the models can synthesise without truncating.

    Sentences are cut at ।.!? and newlines; a sentence longer than `max_chars` is cut
    again at clause punctuation, then at spaces.
    """
    out: List[str] = []
    for sent in (p.strip() for p in _SENT_SPLIT.split(text.strip())):
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        current = ""
        for clause in _CLAUSE_SPLIT.split(sent):
            while len(clause) > max_chars:             # no punctuation at all: cut at a space
                cut = clause.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                if current:
                    out.append(current)
                    current = ""
                out.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            if current and len(current) + 1 + len(clause) > max_chars:
                out.append(current)
                current = clause
            else:
                current = f"{current} {clause}".strip()
        if current:
            out.append(current)
    return out


def trim_silence(wav: np.ndarray, sr: int = SR, thr: float = 0.01, keep_s: float = 0.15) -> np.ndarray:
    """Cut leading/trailing near-silence, keeping `keep_s` of padding."""
    frame = sr // 50
    n = len(wav) // frame * frame
    if n == 0:
        return wav
    loud = np.abs(wav[:n]).reshape(-1, frame).max(1) > thr
    if not loud.any():
        return wav[: int(keep_s * sr)]
    pad = int(keep_s * sr)
    i0 = max(0, int(np.argmax(loud)) * frame - pad)
    i1 = min(len(wav), (len(loud) - int(np.argmax(loud[::-1]))) * frame + pad)
    return wav[i0:i1]


def _token_cap(sentence: str) -> int:
    """~98 audio tokens per second, ~12 chars per second; allow 2x plus a 3 s floor."""
    return min(int((2 * len(sentence) / 12 + 3) * 98), 2520)


def _import_from(directory: str, module: str):
    """Import a helper module that ships inside a model repo."""
    import importlib
    cached = sys.modules.get(module)
    if cached is not None and not str(getattr(cached, "__file__", "")).startswith(directory):
        del sys.modules[module]        # a same-named module from another repo
    if directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)
    return importlib.import_module(module)


def _torch_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------
class TTSAdapter:
    name = "base"
    engine = "base"
    sample_rate = SR
    default_voice = "default"
    native_speed = False        # adapter applies `speed` itself
    chunk_text = True           # False: adapter does its own long-text handling

    @classmethod
    def catalog(cls) -> dict:
        return {"voices": {"default": {"gender": "", "lang": "", "lang_name": "", "desc": "Model default voice"}},
                "openai_aliases": {}, "styles": [], "default_voice": "default"}

    def load(self, local_dir: str, spec: dict, token: Optional[str]) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            setattr(self, attr, None)

    def resolve_voice(self, voice: Optional[str]) -> str:
        cat = self.catalog()
        name = (voice or "").strip()
        if not name:
            return cat["default_voice"]
        low = name.lower()
        if low in cat["openai_aliases"]:
            return cat["openai_aliases"][low]
        for v in cat["voices"]:
            if v.lower() == low:
                return v
        logger.warning("Voice '%s' is not available for %s; using '%s'", voice, self.name, cat["default_voice"])
        return cat["default_voice"]

    def synth_sentence(self, text: str, voice: str, style: str, params: dict) -> Tuple[np.ndarray, dict]:
        """One utterance -> (float32 mono @ self.sample_rate, stats)."""
        raise NotImplementedError

    def stream_sentence(self, text: str, voice: str, style: str, params: dict) -> Iterator[np.ndarray]:
        wav, _ = self.synth_sentence(text, voice, style, params)
        step = self.sample_rate // 2
        for i in range(0, len(wav), step):
            yield wav[i:i + step]


class _IndicSpeakBase(TTSAdapter):
    @classmethod
    def catalog(cls) -> dict:
        return {"voices": VOICE_CATALOG, "openai_aliases": OPENAI_VOICE_MAP,
                "styles": AVAILABLE_STYLES, "default_voice": "Amit"}


class IndicSpeakMLX(_IndicSpeakBase):
    name = "indic-speak-mlx"
    engine = "mlx"

    def load(self, local_dir, spec, token):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            raise ModelError("MLX is not installed; indic-speak-mlx needs Apple Silicon. Use indic-speak-torch instead.",
                             status=501, code="backend_unavailable")
        _import_from(local_dir, "inference")
        module = _import_from(local_dir, "tts_mlx")
        self.tts = module.MLXTTS(local_dir, decode_device=_torch_device())

    def _kwargs(self, text, params):
        kw = {"temperature": float(params.get("temperature", 0.6)), "max_tokens": _token_cap(text)}
        if params.get("seed") is not None:
            kw["seed"] = int(params["seed"])
        return kw

    def synth_sentence(self, text, voice, style, params):
        wav, stats = self.tts(text, speaker=voice, style=style, **self._kwargs(text, params))
        return wav, {"tokens": stats.get("tokens", 0), "ttfa": stats.get("ttft"), "hit_cap": stats.get("hit_cap", False)}

    def stream_sentence(self, text, voice, style, params):
        for chunk, _ in self.tts.stream(text, speaker=voice, style=style, **self._kwargs(text, params)):
            if len(chunk):
                yield chunk


class IndicSpeakTorch(_IndicSpeakBase):
    name = "indic-speak-torch"
    engine = "pytorch"

    def load(self, local_dir, spec, token):
        module = _import_from(local_dir, "inference")
        self.device = _torch_device()
        self.tts = module.TTS(local_dir, device=self.device)

    def synth_sentence(self, text, voice, style, params):
        kw = {"temperature": float(params.get("temperature", 0.6)), "max_new_tokens": _token_cap(text)}
        if params.get("seed") is not None:
            kw["seed"] = int(params["seed"])
        wav = self.tts(text=text, speaker=voice, style=style, **kw)
        return wav, {}


KOKORO_VOICES = {
    "af_heart": ("Female", "en-us"), "af_bella": ("Female", "en-us"), "af_nicole": ("Female", "en-us"),
    "af_sarah": ("Female", "en-us"), "af_sky": ("Female", "en-us"), "af_alloy": ("Female", "en-us"),
    "af_nova": ("Female", "en-us"), "af_aoede": ("Female", "en-us"), "af_kore": ("Female", "en-us"),
    "am_adam": ("Male", "en-us"), "am_michael": ("Male", "en-us"), "am_echo": ("Male", "en-us"),
    "am_onyx": ("Male", "en-us"), "am_fenrir": ("Male", "en-us"), "am_puck": ("Male", "en-us"),
    "bf_emma": ("Female", "en-gb"), "bf_isabella": ("Female", "en-gb"), "bf_alice": ("Female", "en-gb"),
    "bm_george": ("Male", "en-gb"), "bm_lewis": ("Male", "en-gb"), "bm_fable": ("Male", "en-gb"),
    "hf_alpha": ("Female", "hi"), "hf_beta": ("Female", "hi"), "hm_omega": ("Male", "hi"), "hm_psi": ("Male", "hi"),
}
_KOKORO_LANG_NAMES = {"en-us": "English (US)", "en-gb": "English (UK)", "hi": "Hindi"}


class KokoroTTS(TTSAdapter):
    name = "kokoro"
    engine = "kokoro"
    default_voice = "af_heart"
    native_speed = True
    chunk_text = False

    @classmethod
    def catalog(cls) -> dict:
        voices = {v: {"gender": g, "lang": lang, "lang_name": _KOKORO_LANG_NAMES[lang], "desc": ""}
                  for v, (g, lang) in KOKORO_VOICES.items()}
        aliases = {"alloy": "af_alloy", "echo": "am_echo", "fable": "bm_fable",
                   "onyx": "am_onyx", "nova": "af_nova", "shimmer": "af_bella"}
        return {"voices": voices, "openai_aliases": aliases, "styles": [], "default_voice": "af_heart"}

    def load(self, local_dir, spec, token):
        try:
            from kokoro import KModel, KPipeline
        except ImportError:
            raise ModelError("The 'kokoro' package is not installed. Run: pip install kokoro",
                             status=501, code="backend_unavailable")
        import os
        self.repo_id = spec["repo_id"]
        weights = next((f for f in sorted(os.listdir(local_dir)) if f.endswith(".pth")), None)
        if weights is None:
            raise ModelError(f"No .pth checkpoint found in {spec['repo_id']}.", status=500, code="model_load_failed")
        self.device = _torch_device()
        self.model = KModel(repo_id=self.repo_id, config=os.path.join(local_dir, "config.json"),
                            model=os.path.join(local_dir, weights)).to(self.device).eval()
        self._pipeline_cls = KPipeline
        self.pipelines: Dict[str, object] = {}

    def resolve_voice(self, voice):
        name = (voice or "").strip()
        if re.fullmatch(r"[a-z]{2}_[a-z0-9]+", name):   # any voice file shipped in the repo
            return name
        return super().resolve_voice(voice)

    def _pipeline(self, voice: str):
        code = voice[0]
        if code not in self.pipelines:
            self.pipelines[code] = self._pipeline_cls(lang_code=code, repo_id=self.repo_id, model=self.model)
        return self.pipelines[code]

    def stream_sentence(self, text, voice, style, params):
        speed = float(params.get("speed", 1.0))
        for result in self._pipeline(voice)(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            if result.audio is not None:
                yield result.audio.detach().float().cpu().numpy()

    def synth_sentence(self, text, voice, style, params):
        parts = list(self.stream_sentence(text, voice, style, params))
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), {}


class TransformersTTS(TTSAdapter):
    """Generic Hugging Face `text-to-speech` pipeline (VITS/MMS, Bark, SpeechT5-style models)."""
    name = "transformers-tts"
    engine = "transformers"

    def load(self, local_dir, spec, token):
        from transformers import pipeline
        self.device = _torch_device()
        try:
            self.pipe = pipeline("text-to-speech", model=local_dir, device=self.device)
            self.pipe("ok")                       # some architectures lack MPS kernels; fail here, not mid-request
        except Exception as e:
            if self.device == "cpu":
                raise
            logger.warning("%s failed on %s (%s); falling back to CPU", spec["repo_id"], self.device, e)
            self.device = "cpu"
            self.pipe = pipeline("text-to-speech", model=local_dir, device="cpu")
        self.sample_rate = int(getattr(self.pipe, "sampling_rate", None) or SR)

    def synth_sentence(self, text, voice, style, params):
        out = self.pipe(text)
        self.sample_rate = int(out["sampling_rate"])
        return np.asarray(out["audio"], dtype=np.float32).reshape(-1), {}


class RemoteOpenAITTS(TTSAdapter):
    """Proxy to another OpenAI-compatible TTS server (`POST {remote_url}/audio/speech`).

    Registry fields: `remote_url` (base URL ending in /v1), `remote_model` (the model name
    upstream; defaults to the id), optional `timeout_s` and `language_name` for the voice list.
    Text is sent whole: the upstream server does its own segmentation, and splitting at
    sentence marks would break verses apart before meter detection.
    """
    name = "remote-openai-tts"
    engine = "remote"
    native_speed = True
    chunk_text = False
    _voice_cache: Dict[str, Tuple[float, dict]] = {}
    VOICE_CACHE_S = 60.0

    @classmethod
    def catalog_for_spec(cls, spec: dict) -> dict:
        url = (spec.get("remote_url") or "").rstrip("/")
        hit = cls._voice_cache.get(url)
        if hit and time.time() - hit[0] < cls.VOICE_CACHE_S:
            return hit[1]
        lang_name = spec.get("language_name") or "Remote"
        fallback = (spec.get("default_params") or {}).get("voice") or "default"
        names, default, note = [fallback], fallback, ""
        try:
            import requests
            r = requests.get(f"{url}/audio/voices", timeout=3)
            r.raise_for_status()
            data = r.json()
            raw = data.get("voices") or []
            if isinstance(raw, dict):
                raw = list(raw)
            names = [v if isinstance(v, str) else v.get("id") or v.get("name") for v in raw] or names
            default = data.get("default") or names[0]
            note = data.get("note") or ""
        except Exception as e:
            logger.warning("Could not fetch voices from %s: %s", url, e)
            cat = {"voices": {fallback: {"gender": "", "lang": "", "lang_name": lang_name, "desc": "Upstream unreachable"}},
                   "openai_aliases": {}, "styles": [], "default_voice": fallback}
            return cat                                   # not cached: retry next time
        voices = {n: {"gender": "", "lang": "", "lang_name": lang_name, "desc": note if n == default else ""}
                  for n in names if n}
        cat = {"voices": voices, "openai_aliases": {}, "styles": [], "default_voice": default}
        cls._voice_cache[url] = (time.time(), cat)
        return cat

    def load(self, local_dir, spec, token):
        import requests
        self.spec = spec
        self.url = (spec.get("remote_url") or "").rstrip("/")
        if not self.url:
            raise ModelError(f"Model '{spec['id']}' has no remote_url.", status=500, code="model_load_failed")
        self.remote_model = spec.get("remote_model") or spec["id"]
        self.timeout = float(spec.get("timeout_s") or 300)
        self.session = requests.Session()
        try:
            self.session.get(f"{self.url}/models", timeout=5).raise_for_status()
        except Exception as e:
            raise ModelError(f"Remote TTS server for '{spec['id']}' at {self.url} is unreachable: {e}",
                             status=502, code="upstream_unavailable")

    def resolve_voice(self, voice):
        cat = self.catalog_for_spec(self.spec)
        name = (voice or "").strip()
        if not name:
            return cat["default_voice"]
        for v in cat["voices"]:
            if v.lower() == name.lower():
                return v
        return name          # unknown names (e.g. OpenAI voices) are the upstream server's call

    def synth_sentence(self, text, voice, style, params):
        import io
        import soundfile as sf
        body = {"model": self.remote_model, "input": text, "voice": voice,
                "response_format": "wav", "speed": float(params.get("speed", 1.0))}
        if params.get("seed") is not None:
            body["seed"] = int(params["seed"])
        if params.get("instructions"):
            body["instructions"] = params["instructions"]
        try:
            r = self.session.post(f"{self.url}/audio/speech", json=body, timeout=self.timeout)
        except Exception as e:
            raise ModelError(f"Remote TTS request failed: {e}", status=502, code="upstream_unavailable")
        if r.status_code != 200:
            raise ModelError(f"Remote TTS returned {r.status_code}: {r.text[:300]}",
                             status=502 if r.status_code >= 500 else 400, code="upstream_error")
        wav, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
        self.sample_rate = int(sr)
        return wav.mean(axis=1), {}


ADAPTERS = {cls.name: cls for cls in (IndicSpeakMLX, IndicSpeakTorch, KokoroTTS, TransformersTTS, RemoteOpenAITTS)}


def create_adapter(spec: dict) -> TTSAdapter:
    cls = ADAPTERS.get(spec.get("adapter") or "")
    if cls is None:
        raise ModelError(f"Unknown TTS adapter '{spec.get('adapter')}' for model '{spec['id']}'.",
                         status=501, code="adapter_not_found")
    return cls()


def catalog_for(spec: dict) -> dict:
    cls = ADAPTERS.get(spec.get("adapter") or "", TTSAdapter)
    if hasattr(cls, "catalog_for_spec"):
        return cls.catalog_for_spec(spec)
    return cls.catalog()


# ----------------------------------------------------------------------------
# Synthesis (model-agnostic)
# ----------------------------------------------------------------------------
GAP_S = 0.25


def _to_output_rate(adapter: TTSAdapter, wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return resample(wav, adapter.sample_rate, SR) if adapter.sample_rate != SR else wav


def synthesize(adapter: TTSAdapter, text: str, voice: Optional[str] = None, style: Optional[str] = None,
               speed: float = 1.0, params: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Full synthesis. Returns (float32 mono @ 24 kHz, stats)."""
    params = dict(params or {})
    voice_name = adapter.resolve_voice(voice)
    style_tag = normalize_style(style)
    params["speed"] = speed
    sentences = split_text(text) if adapter.chunk_text else [text.strip()]
    if not sentences or not sentences[0]:
        raise ModelError("Input text is empty.", param="input")

    t0 = time.time()
    gap = np.zeros(int(GAP_S * SR), np.float32)
    parts: List[np.ndarray] = []
    tokens, ttfa, truncated = 0, None, 0
    for sent in sentences:
        wav, st = adapter.synth_sentence(sent, voice_name, style_tag, params)
        if ttfa is None:
            ttfa = time.time() - t0
        tokens += int(st.get("tokens") or 0)
        truncated += 1 if st.get("hit_cap") else 0
        wav = _to_output_rate(adapter, wav)
        if adapter.chunk_text:
            wav = trim_silence(wav)
        parts += [wav, gap]
    wav = np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)
    if not adapter.native_speed:
        wav = change_speed(wav, speed)
    total = time.time() - t0
    audio_s = len(wav) / SR
    stats = {
        "engine": adapter.engine, "voice": voice_name, "style": style_tag or "NEUTRAL",
        "sentences": len(sentences), "total_s": round(total, 3), "audio_s": round(audio_s, 3),
        "rtf": round(total / max(audio_s, 1e-4), 3), "ttfa_s": round(ttfa or total, 3),
    }
    if tokens:
        stats["tokens"] = tokens
        stats["tok_per_s"] = round(tokens / max(total, 1e-6), 1)
    if truncated:
        stats["truncated_sentences"] = truncated
    return wav, stats


def stream_synthesis(adapter: TTSAdapter, text: str, voice: Optional[str] = None, style: Optional[str] = None,
                     speed: float = 1.0, params: Optional[dict] = None,
                     cancel: Optional[threading.Event] = None) -> Iterator[np.ndarray]:
    """Yield float32 mono @ 24 kHz chunks as they are generated.

    At speed 1.0 chunks are emitted as the model produces them. Other speeds are
    time-stretched one sentence at a time, so latency becomes sentence-granular.
    """
    params = dict(params or {})
    voice_name = adapter.resolve_voice(voice)
    style_tag = normalize_style(style)
    params["speed"] = speed
    sentences = split_text(text) if adapter.chunk_text else [text.strip()]
    if not sentences or not sentences[0]:
        raise ModelError("Input text is empty.", param="input")
    stretch = (not adapter.native_speed) and abs(speed - 1.0) >= 0.02
    gap = np.zeros(int(GAP_S * SR / (speed if stretch else 1.0)), np.float32)

    for i, sent in enumerate(sentences):
        pending: List[np.ndarray] = []
        for chunk in adapter.stream_sentence(sent, voice_name, style_tag, params):
            if cancel is not None and cancel.is_set():
                return
            chunk = _to_output_rate(adapter, chunk)
            if stretch:
                pending.append(chunk)
            elif len(chunk):
                yield chunk
        if pending:
            yield change_speed(np.concatenate(pending), speed)
        if i < len(sentences) - 1:
            yield gap
