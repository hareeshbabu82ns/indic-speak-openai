"""Indic-Speak Text-to-Speech Engine
Supports both MLX (Apple Silicon accelerated, real-time streaming) and PyTorch (MPS/CPU)
Provides OpenAI speech API compatibility (/v1/audio/speech)
"""
from __future__ import annotations

import io
import json
import logging
import os
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger("tts_engine")
logging.basicConfig(level=logging.INFO)

SR = 24_000

# Voice catalog from bodhan-ai/indic-speak
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

# Standard OpenAI voice aliases mapping
OPENAI_VOICE_MAP = {
    "alloy": "Amit",
    "echo": "Arun",
    "fable": "Deepika",
    "onyx": "Vamsi",
    "nova": "Kavya",
    "shimmer": "Sravani",
}

# Available styles / emotions
AVAILABLE_STYLES = [
    "NEUTRAL",
    "HAPPY",
    "SAD",
    "ANGER",
    "FEAR",
    "SURPRISE",
    "DISGUST",
    "news",
    "AIR style news",
    "TV style news",
    "advertisements",
    "children's stories",
    "educational lecture",
    "Customer Care",
    "single person narration audiobook",
]

def make_streaming_wav_header(sample_rate: int = SR, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Generate a 44-byte WAV header configured for indefinite streaming (0xFFFFFFFF length fields)."""
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,  # Streaming file size
        b"WAVE",
        b"fmt ",
        16,          # Subchunk1Size
        1,           # AudioFormat (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF   # Streaming data size
    )


class TTSEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.generated_dir = self.data_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.hf_token = os.getenv("HF_TOKEN")
        self.mlx_tts = None
        self.torch_tts = None
        self.speak_long_fn = None
        self.stream_long_fn = None

        self._has_mlx = False
        try:
            import mlx.core as mx
            self._has_mlx = True
        except ImportError:
            self._has_mlx = False

    def load_models(self):
        """Preload the fastest available TTS model (MLX on Mac or PyTorch)."""
        logger.info("Initializing TTS engine...")
        if self._has_mlx:
            try:
                logger.info("Loading MLX TTS model (iniquitous/indic-speak-mlx-bf16)...")
                tts_mlx_dir = snapshot_download("iniquitous/indic-speak-mlx-bf16", token=self.hf_token)
                sys.path.insert(0, str(tts_mlx_dir))
                from tts_mlx import MLXTTS, speak_long, stream_long

                self.mlx_tts = MLXTTS(tts_mlx_dir)
                self.speak_long_fn = speak_long
                self.stream_long_fn = stream_long
                logger.info("MLX TTS model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load MLX TTS: {e}. Falling back to PyTorch TTS.")

        if self.mlx_tts is None:
            self._load_torch_tts()

    def _load_torch_tts(self):
        """Loads PyTorch Indic-Speak TTS model."""
        if self.torch_tts is not None:
            return
        logger.info("Loading PyTorch TTS model (bodhan-ai/indic-speak)...")
        tts_dir = snapshot_download("bodhan-ai/indic-speak", token=self.hf_token)
        sys.path.insert(0, str(tts_dir))
        from inference import TTS

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_tts = TTS(tts_dir, device=device)
        logger.info(f"PyTorch TTS model loaded on {device}.")

    def resolve_voice(self, voice: Optional[str]) -> str:
        """Map OpenAI voice name or case-insensitive voice name to a standard Indic voice."""
        if not voice:
            return "Amit"

        voice_clean = voice.strip()
        voice_lower = voice_clean.lower()

        # Check OpenAI mapping first
        if voice_lower in OPENAI_VOICE_MAP:
            return OPENAI_VOICE_MAP[voice_lower]

        # Check exact catalog match
        for catalog_voice in VOICE_CATALOG:
            if catalog_voice.lower() == voice_lower:
                return catalog_voice

        # Default fallback
        logger.warning(f"Voice '{voice}' not found, falling back to 'Amit'")
        return "Amit"

    def adjust_speed(self, wav: np.ndarray, speed: float) -> np.ndarray:
        """Adjust playback speed by resampling (speed > 1 speeds up, speed < 1 slows down)."""
        if abs(speed - 1.0) < 0.02:
            return wav
        speed = max(0.25, min(speed, 4.0))
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        orig_sr = int(SR * speed)
        import torchaudio
        resampled = torchaudio.functional.resample(wav_tensor, orig_sr, SR)
        return resampled.squeeze(0).numpy()

    def synthesize(
        self,
        text: str,
        voice: str = "Amit",
        style: str = "NEUTRAL",
        speed: float = 1.0,
        model_name: str = "indic-speak-mlx",
    ) -> Tuple[np.ndarray, dict]:
        """Synthesize text to float32 numpy audio @ 24kHz. Returns (wav, stats)."""
        resolved_voice = self.resolve_voice(voice)
        use_torch = "torch" in model_name.lower() or (self.mlx_tts is None)

        t0 = time.time()
        if use_torch:
            self._load_torch_tts()
            wav = self.torch_tts(text=text, speaker=resolved_voice, style=style, temperature=0.6)
            total_time = time.time() - t0
            stats = {
                "total_s": total_time,
                "audio_s": len(wav) / SR,
                "rtf": total_time / max(len(wav) / SR, 1e-4),
                "engine": "pytorch"
            }
        else:
            if "\n" in text or len(text) > 80:
                wav, stats_list = self.speak_long_fn(
                    self.mlx_tts, text, speaker=resolved_voice, style=style, temperature=0.6
                )
            else:
                wav, single_stats = self.mlx_tts(
                    text, speaker=resolved_voice, style=style, temperature=0.6
                )
                stats_list = [single_stats]

            total_time = time.time() - t0
            stats = {
                "total_s": total_time,
                "audio_s": len(wav) / SR,
                "rtf": total_time / max(len(wav) / SR, 1e-4),
                "engine": "mlx"
            }

        if abs(speed - 1.0) >= 0.02:
            wav = self.adjust_speed(wav, speed)
            stats["audio_s"] = len(wav) / SR

        return wav, stats

    def stream_audio(
        self,
        text: str,
        voice: str = "Amit",
        style: str = "NEUTRAL",
        response_format: str = "mp3",
        speed: float = 1.0,
        save_file: bool = True,
        file_id: Optional[str] = None
    ) -> Generator[bytes, None, None]:
        """Stream generated audio in real-time. Yields chunk bytes."""
        resolved_voice = self.resolve_voice(voice)
        format_lower = response_format.lower()

        accumulated_chunks = []

        if self.mlx_tts is not None and self.stream_long_fn is not None:
            # Yield WAV header if WAV stream requested
            if format_lower == "wav":
                yield make_streaming_wav_header(sample_rate=SR, num_channels=1, bits_per_sample=16)

            for chunk, stats in self.stream_long_fn(
                self.mlx_tts, text, speaker=resolved_voice, style=style, temperature=0.6
            ):
                if len(chunk) == 0:
                    continue
                if abs(speed - 1.0) >= 0.02:
                    chunk = self.adjust_speed(chunk, speed)

                if save_file:
                    accumulated_chunks.append(chunk)

                # Format conversion
                chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                if format_lower in ("pcm", "wav"):
                    yield chunk_int16.tobytes()
                elif format_lower == "mp3":
                    # Write short MP3 buffer
                    buf = io.BytesIO()
                    sf.write(buf, chunk, SR, format="MP3")
                    yield buf.getvalue()
                else:
                    yield chunk_int16.tobytes()

        else:
            # Fallback if streaming generator unavailable: synthesize utterance and stream in slices
            wav, _ = self.synthesize(text, voice=resolved_voice, style=style, speed=speed)
            if save_file:
                accumulated_chunks.append(wav)

            if format_lower == "wav":
                yield make_streaming_wav_header(sample_rate=SR, num_channels=1, bits_per_sample=16)

            chunk_len = 2048 * 4
            for i in range(0, len(wav), chunk_len):
                piece = wav[i:i + chunk_len]
                piece_int16 = (np.clip(piece, -1.0, 1.0) * 32767).astype(np.int16)
                if format_lower in ("pcm", "wav"):
                    yield piece_int16.tobytes()
                elif format_lower == "mp3":
                    buf = io.BytesIO()
                    sf.write(buf, piece, SR, format="MP3")
                    yield buf.getvalue()
                else:
                    yield piece_int16.tobytes()

        # Save complete streamed audio to disk in background if requested
        if save_file and accumulated_chunks:
            try:
                full_wav = np.concatenate(accumulated_chunks)
                fid = file_id or f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                self.save_audio(fid, full_wav, format=format_lower, metadata={
                    "text": text,
                    "voice": resolved_voice,
                    "style": style,
                    "speed": speed,
                    "streamed": True
                })
            except Exception as e:
                logger.error(f"Failed to save streamed audio: {e}")

    def encode_audio(self, wav: np.ndarray, format: str = "mp3") -> bytes:
        """Encode float32 waveform array to target audio bytes."""
        fmt = format.lower()
        if fmt == "pcm":
            return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        buf = io.BytesIO()
        sf_format = "WAV"
        if fmt in ("mp3", "mpeg"):
            sf_format = "MP3"
        elif fmt == "flac":
            sf_format = "FLAC"
        elif fmt in ("ogg", "opus"):
            sf_format = "OGG"
        elif fmt == "wav":
            sf_format = "WAV"

        sf.write(buf, wav, SR, format=sf_format)
        return buf.getvalue()

    def save_audio(self, file_id: str, wav: np.ndarray, format: str = "mp3", metadata: Optional[dict] = None) -> dict:
        """Save synthesized audio and its metadata to the server filesystem."""
        fmt = format.lower()
        ext = "raw" if fmt == "pcm" else fmt
        filename = f"{file_id}.{ext}"
        filepath = self.generated_dir / filename

        audio_bytes = self.encode_audio(wav, format=fmt)
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        duration = len(wav) / SR
        meta = {
            "id": file_id,
            "filename": filename,
            "path": str(filepath),
            "format": fmt,
            "duration": round(duration, 3),
            "size_bytes": len(audio_bytes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {})
        }

        meta_path = self.generated_dir / f"{file_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return meta

    def list_saved_files(self) -> list[dict]:
        """List all saved audio files."""
        files = []
        for meta_file in sorted(self.generated_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    files.append(json.load(f))
            except Exception:
                continue
        return files
