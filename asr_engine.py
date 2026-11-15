"""Automatic Speech Recognition engine.

Adapters:
  indic-transcribe   bodhan-ai/indic-transcribe-core   Indic Canary (MPS / CUDA / CPU)
  whisper-mlx        mlx-community/whisper-*           mlx-whisper on Apple Silicon
  transformers-asr   any HF ASR pipeline               Whisper (PyTorch), wav2vec2, ...

Every adapter takes mono float32 audio at 16 kHz and returns
`{"language", "text", "segments": [{"id", "seek", "start", "end", "text"}]}`.
"""
from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional

import numpy as np

from audio_utils import ASR_SR, format_srt, format_vtt
from model_manager import ModelError
from transliterate import romanise

logger = logging.getLogger("asr_engine")

TARGET_SR = ASR_SR

# Languages of bodhan-ai/indic-transcribe-core (26 Indic + English) plus auto-detect.
LANGUAGES_CATALOG = {
    "auto": "Auto-detect",
    "as": "Assamese", "bgc": "Haryanvi", "bhb": "Bhilali", "bho": "Bhojpuri", "bn": "Bengali",
    "brx": "Bodo", "doi": "Dogri", "en": "English", "gu": "Gujarati", "hi": "Hindi",
    "hne": "Chhattisgarhi", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}
MODES = ["native", "mixed", "romanised"]
RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


def normalize_mode(mode: Optional[str]) -> str:
    m = (mode or "native").strip().lower()
    if m in ("mixed", "itn"):
        return "mixed"
    if m.startswith("roman"):
        return "romanised"
    return "native"


def normalize_language(language: Optional[str]) -> Optional[str]:
    lang = (language or "").strip().lower()
    return None if lang in ("", "auto", "none", "detect") else lang


def _torch_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _segment(idx: int, start: float, end: float, text: str) -> dict:
    return {"id": idx, "seek": int(start * TARGET_SR), "start": round(float(start), 3),
            "end": round(float(end), 3), "text": text.strip()}


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------
class ASRAdapter:
    name = "base"
    engine = "base"
    supports_translation = False

    @classmethod
    def catalog(cls) -> dict:
        return {"languages": {"auto": "Auto-detect"}, "modes": ["native"]}

    def load(self, local_dir: str, spec: dict, token: Optional[str]) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            setattr(self, attr, None)

    def transcribe(self, wav: np.ndarray, language: Optional[str], mode: str, task: str,
                   temperature: float, prompt: Optional[str], chunk_seconds: float) -> dict:
        raise NotImplementedError


class IndicTranscribe(ASRAdapter):
    name = "indic-transcribe"
    engine = "pytorch"

    @classmethod
    def catalog(cls) -> dict:
        return {"languages": LANGUAGES_CATALOG, "modes": MODES}

    def load(self, local_dir, spec, token):
        import importlib
        if local_dir in sys.path:
            sys.path.remove(local_dir)
        sys.path.insert(0, local_dir)
        self._module = importlib.import_module("indic_transcribe")
        self._long = importlib.import_module("long_form")
        self.device = _torch_device()
        self.asr = self._module.IndicTranscribe.from_pretrained(local_dir, device=self.device)

    def transcribe(self, wav, language, mode, task, temperature, prompt, chunk_seconds):
        import torch
        audio = torch.from_numpy(np.ascontiguousarray(wav))
        lang = language if language in LANGUAGES_CATALOG and language != "auto" else None
        if language and lang is None:
            logger.warning("Language '%s' is not supported by indic-transcribe; auto-detecting.", language)
        if task == "translate":
            mode = "romanised"          # this checkpoint transliterates; it has no translation head
        if lang is None:
            lang = self._long.identify_long(self.asr, audio)

        gen_kw = {"max_new_tokens": 512}
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=float(temperature))

        # The checkpoint trains on <=30 s clips; longer single passes collapse into
        # repetition, so cut at pauses and decode each piece (guarded against repeats).
        hard_max = max(5.0, min(float(chunk_seconds or 30.0), 30.0))
        cuts = self._long.split_on_silence(audio, sr=TARGET_SR, target=hard_max * 0.8, hard_max=hard_max)
        segments: List[dict] = []
        for a, b in cuts:
            piece = audio[a:b]
            if piece.numel() < 1600:    # < 0.1 s
                continue
            text = self._long._decode_guarded(self.asr, piece, lang, mode, dict(gen_kw)).strip()
            if text:
                segments.append(_segment(len(segments), a / TARGET_SR, b / TARGET_SR, text))
        # This checkpoint emits native script whatever the prompt asks for (see its model
        # card), so romanised output is finished here by rule-based ISO 15919 transliteration.
        delivered = "native"
        if mode == "romanised":
            for seg in segments:
                seg["text"], changed = romanise(seg["text"])
                delivered = "romanised" if changed else delivered
        return {"language": lang, "text": " ".join(s["text"] for s in segments), "segments": segments,
                "mode": delivered}


class WhisperMLX(ASRAdapter):
    name = "whisper-mlx"
    engine = "mlx"
    supports_translation = True

    @classmethod
    def catalog(cls) -> dict:
        try:
            from mlx_whisper.tokenizer import LANGUAGES
            langs = {"auto": "Auto-detect", **{k: v.title() for k, v in sorted(LANGUAGES.items(), key=lambda kv: kv[1])}}
        except Exception:
            langs = {"auto": "Auto-detect", "en": "English", "hi": "Hindi"}
        return {"languages": langs, "modes": ["native"]}

    def load(self, local_dir, spec, token):
        try:
            import mlx.core as mx
            import mlx_whisper  # noqa: F401
            from mlx_whisper.load_models import load_model
        except ImportError:
            raise ModelError("mlx-whisper is not installed. Run: pip install mlx-whisper",
                             status=501, code="backend_unavailable")
        self.path = local_dir
        self.model = load_model(local_dir, dtype=mx.float16)

    def unload(self):
        try:
            holder = sys.modules["mlx_whisper.transcribe"].ModelHolder
            if holder.model is self.model:
                holder.model, holder.model_path = None, None
        except Exception:
            pass
        super().unload()

    def transcribe(self, wav, language, mode, task, temperature, prompt, chunk_seconds):
        import mlx_whisper
        holder = sys.modules["mlx_whisper.transcribe"].ModelHolder
        holder.model, holder.model_path = self.model, self.path     # reuse our resident weights
        kw = {}
        if temperature and temperature > 0:
            kw["temperature"] = float(temperature)
        result = mlx_whisper.transcribe(
            np.ascontiguousarray(wav, dtype=np.float32), path_or_hf_repo=self.path, language=language,
            task="translate" if task == "translate" else "transcribe", initial_prompt=prompt or None,
            verbose=None, **kw)
        segments = [_segment(i, s["start"], s["end"], s["text"]) for i, s in enumerate(result.get("segments", []))
                    if s.get("text", "").strip()]
        return {"language": result.get("language") or language, "text": result.get("text", "").strip(), "segments": segments}


class TransformersASR(ASRAdapter):
    """Generic Hugging Face ASR pipeline. Whisper checkpoints get language ID and translation."""
    name = "transformers-asr"
    engine = "transformers"

    @classmethod
    def catalog(cls) -> dict:
        try:
            from transformers.models.whisper.tokenization_whisper import LANGUAGES
            langs = {"auto": "Auto-detect", **{k: v.title() for k, v in sorted(LANGUAGES.items(), key=lambda kv: kv[1])}}
        except Exception:
            langs = {"auto": "Auto-detect"}
        return {"languages": langs, "modes": ["native"]}

    def load(self, local_dir, spec, token):
        import torch
        from transformers import pipeline
        self.device = _torch_device()
        dtype = torch.float16 if self.device in ("mps", "cuda") and spec.get("quantization") in ("fp16", "bf16") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=local_dir, device=self.device, dtype=dtype)
        self.is_whisper = self.pipe.model.config.model_type == "whisper"
        self.supports_translation = self.is_whisper

    def _detect_language(self, wav: np.ndarray) -> Optional[str]:
        import torch
        try:
            feats = self.pipe.feature_extractor(wav[: 30 * TARGET_SR], sampling_rate=TARGET_SR, return_tensors="pt").input_features
            feats = feats.to(self.pipe.model.device, dtype=self.pipe.model.dtype)
            with torch.no_grad():
                token_id = self.pipe.model.detect_language(feats)[0].item()
            return self.pipe.tokenizer.decode([token_id]).strip("<|>") or None
        except Exception as e:
            logger.warning("Whisper language detection failed: %s", e)
            return None

    def transcribe(self, wav, language, mode, task, temperature, prompt, chunk_seconds):
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        duration = len(wav) / TARGET_SR
        call_kw: Dict[str, object] = {}
        if duration > 30.0:
            call_kw["chunk_length_s"] = 30
        if self.is_whisper:
            if language is None:
                language = self._detect_language(wav)
            gen_kw: Dict[str, object] = {"task": "translate" if task == "translate" else "transcribe"}
            if language:
                gen_kw["language"] = language
            if temperature and temperature > 0:
                gen_kw.update(do_sample=True, temperature=float(temperature))
            call_kw.update(return_timestamps=True, generate_kwargs=gen_kw)
        out = self.pipe({"raw": wav, "sampling_rate": TARGET_SR}, **call_kw)
        text = (out.get("text") or "").strip()
        segments: List[dict] = []
        for chunk in out.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            piece = (chunk.get("text") or "").strip()
            if piece:
                start = float(start) if start is not None else (segments[-1]["end"] if segments else 0.0)
                segments.append(_segment(len(segments), start, float(end) if end is not None else duration, piece))
        if not segments and text:
            segments = [_segment(0, 0.0, duration, text)]
        return {"language": language or "unknown", "text": text, "segments": segments}


ADAPTERS = {cls.name: cls for cls in (IndicTranscribe, WhisperMLX, TransformersASR)}


def create_adapter(spec: dict) -> ASRAdapter:
    cls = ADAPTERS.get(spec.get("adapter") or "")
    if cls is None:
        raise ModelError(f"Unknown ASR adapter '{spec.get('adapter')}' for model '{spec['id']}'.",
                         status=501, code="adapter_not_found")
    return cls()


def catalog_for(spec: dict) -> dict:
    cls = ADAPTERS.get(spec.get("adapter") or "", ASRAdapter)
    return cls.catalog()


# ----------------------------------------------------------------------------
# Transcription + output formatting (model-agnostic)
# ----------------------------------------------------------------------------
def transcribe(adapter: ASRAdapter, wav: np.ndarray, language: Optional[str] = None, mode: str = "native",
               task: str = "transcribe", temperature: float = 0.0, prompt: Optional[str] = None,
               chunk_seconds: float = 30.0) -> dict:
    """Run a transcription. Returns the verbose record (task, language, duration, text, segments)."""
    duration = len(wav) / TARGET_SR
    if duration < 0.1:
        raise ModelError("Audio is too short to transcribe (under 0.1 s).", param="file")
    result = adapter.transcribe(wav, normalize_language(language), normalize_mode(mode), task,
                                float(temperature or 0.0), prompt, chunk_seconds)
    return {
        "task": task,
        "language": result["language"],
        "duration": round(duration, 3),
        "text": result["text"],
        "segments": result["segments"],
        "mode": "english" if task == "translate" and adapter.supports_translation else result.get("mode", "native"),
    }


def format_transcript(record: dict, response_format: str):
    """Render a verbose record as (body, media_type) in an OpenAI response format."""
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return record["text"], "text/plain; charset=utf-8"
    if fmt == "srt":
        return format_srt(record["segments"]), "application/x-subrip; charset=utf-8"
    if fmt == "vtt":
        return format_vtt(record["segments"]), "text/vtt; charset=utf-8"
    if fmt == "verbose_json":
        return {k: record[k] for k in ("task", "language", "duration", "text", "segments")}, "application/json"
    return {"text": record["text"]}, "application/json"
