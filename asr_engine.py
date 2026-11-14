"""Indic-Transcribe Automatic Speech Recognition (ASR) Engine
Supports 27 Indic languages + English, auto language detection, short and long-form transcription,
timestamped segments, and OpenAI Whisper API compatibility (/v1/audio/transcriptions & translations).
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download

logger = logging.getLogger("asr_engine")
logging.basicConfig(level=logging.INFO)

TARGET_SR = 16_000

# Supported languages in Indic-Transcribe
LANGUAGES_CATALOG = {
    "auto": "Auto-detect",
    "as": "Assamese",
    "bgc": "Haryanvi",
    "bhb": "Bhilali",
    "bho": "Bhojpuri",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hne": "Chhattisgarhi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

MODES = ["native", "mixed", "romanised"]


def seconds_to_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def format_srt(segments: List[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}")
        lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines).strip()


def format_vtt(segments: List[dict]) -> str:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}")
        lines.append(f"{seconds_to_vtt_time(seg['start'])} --> {seconds_to_vtt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines).strip()


class ASREngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transcriptions_dir = self.data_dir / "transcriptions"
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)

        self.hf_token = os.getenv("HF_TOKEN")
        self.asr = None
        self.split_on_silence_fn = None
        self.identify_long_fn = None
        self.decode_guarded_fn = None
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Preload Indic-Transcribe checkpoint into memory."""
        if self.asr is not None:
            return
        logger.info("Loading Indic-Transcribe-Core model (bodhan-ai/indic-transcribe-core)...")
        asr_dir = snapshot_download("bodhan-ai/indic-transcribe-core", token=self.hf_token)
        sys.path.insert(0, str(asr_dir))

        from indic_transcribe import IndicTranscribe
        from long_form import split_on_silence, identify_long, _decode_guarded

        self.asr = IndicTranscribe.from_pretrained(asr_dir, device=self.device)
        self.split_on_silence_fn = split_on_silence
        self.identify_long_fn = identify_long
        self.decode_guarded_fn = _decode_guarded
        logger.info(f"Indic-Transcribe loaded on device {self.device}.")

    def load_audio(self, audio_source: Union[str, bytes, io.IOBase]) -> torch.Tensor:
        """Load any audio input into a mono 16kHz float32 torch.Tensor."""
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            with open(audio_source, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(audio_source, bytes):
            raw_bytes = audio_source
        elif hasattr(audio_source, "read"):
            raw_bytes = audio_source.read()
        else:
            raise ValueError("Unsupported audio source type")

        # 1. Try reading with soundfile
        try:
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
            wav = torch.as_tensor(data).mean(dim=1)
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            return wav
        except Exception:
            pass

        # 2. Fallback to ffmpeg pipe for WebM/Opus/AAC/M4A/MP3
        try:
            cmd = [
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(TARGET_SR),
                "pipe:1"
            ]
            proc = subprocess.run(cmd, input=raw_bytes, capture_output=True, check=True)
            wav_np = np.frombuffer(proc.stdout, dtype=np.float32)
            return torch.from_numpy(wav_np)
        except Exception as e:
            logger.error(f"FFmpeg decoding failed: {e}")
            raise ValueError(f"Could not decode audio with SoundFile or FFmpeg: {e}")

    def transcribe(
        self,
        audio_source: Union[str, bytes, io.IOBase],
        language: Optional[str] = None,
        mode: str = "native",
        response_format: str = "json",
        temperature: float = 0.0,
        save_file: bool = True,
        filename: Optional[str] = None,
    ) -> Union[dict, str]:
        """Transcribe audio with segment timestamps and return requested OpenAI format."""
        if self.asr is None:
            self.load_model()

        wav = self.load_audio(audio_source)
        total_samples = wav.shape[-1]
        duration_s = total_samples / TARGET_SR

        # Validate / normalize language
        lang = language.strip().lower() if language else None
        if lang in ("auto", "none", "", "detect"):
            lang = None

        if lang and lang not in LANGUAGES_CATALOG:
            logger.warning(f"Language '{lang}' is not in catalog. Falling back to auto-detect.")
            lang = None

        # Mode normalization
        norm_mode = mode.lower()
        if norm_mode not in MODES:
            if norm_mode in ("itn", "mixed"):
                norm_mode = "mixed"
            elif "roman" in norm_mode:
                norm_mode = "romanised"
            else:
                norm_mode = "native"

        # Determine language if auto-detect requested
        detected_lang = lang
        if detected_lang is None:
            detected_lang = self.identify_long_fn(self.asr, wav)
            logger.info(f"Auto-detected language: {detected_lang}")

        # Split into segments for long audio or for fine-grained timestamping
        # If <= 30 seconds and standard json/text format requested without timestamps, direct call
        segments = []
        is_short = duration_s <= 30.0

        if is_short and response_format in ("json", "text"):
            text = self.asr.transcribe(wav, lang=detected_lang, mode=norm_mode)
            full_text = text.strip() if isinstance(text, str) else text[0].strip()
            segments.append({
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": round(duration_s, 3),
                "text": full_text
            })
        else:
            # Cut at pauses (target 15-25s, max 30s)
            cuts = self.split_on_silence_fn(wav, sr=TARGET_SR, target=20.0, hard_max=30.0)
            text_parts = []
            for idx, (a, b) in enumerate(cuts):
                piece = wav[a:b]
                if piece.numel() < 1600:  # Skip < 0.1s
                    continue
                start_sec = round(a / TARGET_SR, 3)
                end_sec = round(b / TARGET_SR, 3)
                piece_text = self.decode_guarded_fn(
                    self.asr, piece, detected_lang, norm_mode, {"max_new_tokens": 512}
                ).strip()
                if piece_text:
                    text_parts.append(piece_text)
                    segments.append({
                        "id": idx,
                        "seek": a,
                        "start": start_sec,
                        "end": end_sec,
                        "text": piece_text
                    })
            full_text = " ".join(text_parts)

        # Save record if requested
        file_id = f"asr_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if save_file:
            try:
                record = {
                    "id": file_id,
                    "filename": filename or f"{file_id}.wav",
                    "language": detected_lang,
                    "mode": norm_mode,
                    "duration": round(duration_s, 3),
                    "text": full_text,
                    "segments": segments,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(self.transcriptions_dir / f"{file_id}.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not save transcription record: {e}")

        # Format output according to OpenAI response_format
        fmt = response_format.lower()
        if fmt == "text":
            return full_text
        elif fmt == "srt":
            return format_srt(segments)
        elif fmt == "vtt":
            return format_vtt(segments)
        elif fmt == "verbose_json":
            return {
                "task": "transcribe",
                "language": detected_lang,
                "duration": round(duration_s, 3),
                "text": full_text,
                "segments": segments
            }
        else:  # default "json"
            return {"text": full_text}

    def list_saved_transcriptions(self) -> list[dict]:
        """List all saved transcriptions."""
        items = []
        for meta_file in sorted(self.transcriptions_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    items.append(json.load(f))
            except Exception:
                continue
        return items
