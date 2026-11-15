"""Shared audio helpers: decoding, encoding, streaming headers, time-stretch, subtitles."""
from __future__ import annotations

import io
import logging
import shutil
import struct
import subprocess
from typing import List

import numpy as np
import soundfile as sf

logger = logging.getLogger("audio_utils")

TTS_SR = 24_000
ASR_SR = 16_000

# response_format -> (media type, file extension, libsndfile format)
AUDIO_FORMATS = {
    "mp3": ("audio/mpeg", "mp3", "MP3"),
    "wav": ("audio/wav", "wav", "WAV"),
    "flac": ("audio/flac", "flac", "FLAC"),
    "pcm": ("audio/pcm", "raw", None),
    "opus": ("audio/ogg", "ogg", "OGG"),
    "ogg": ("audio/ogg", "ogg", "OGG"),
}
STREAMABLE_FORMATS = ("wav", "pcm", "mp3")

_FFMPEG = shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return _FFMPEG is not None


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float32 audio (torchaudio sinc interpolation)."""
    if orig_sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    import torch
    import torchaudio

    out = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)).float(), orig_sr, target_sr)
    return out.numpy()


def decode_audio(raw: bytes, target_sr: int = ASR_SR) -> np.ndarray:
    """Decode any audio container to mono float32 at `target_sr`.

    libsndfile handles WAV/FLAC/OGG/MP3; everything else (M4A, WEBM, AAC...) is piped
    through ffmpeg.
    """
    if not raw:
        raise ValueError("Audio payload is empty")
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return resample(data.mean(axis=1), sr, target_sr)
    except Exception:
        pass

    if not _FFMPEG:
        raise ValueError("Could not decode audio with SoundFile and ffmpeg is not installed")
    cmd = [_FFMPEG, "-nostdin", "-loglevel", "error", "-i", "pipe:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1"]
    proc = subprocess.run(cmd, input=raw, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "ignore").strip()[-300:]
        raise ValueError(f"Could not decode audio (unsupported or corrupt file): {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
def float_to_pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def encode_audio(wav: np.ndarray, fmt: str = "mp3", sr: int = TTS_SR) -> bytes:
    """Encode a float32 mono waveform to the requested container."""
    fmt = fmt.lower()
    if fmt not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format '{fmt}'")
    if fmt == "pcm":
        return float_to_pcm16(wav)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format=AUDIO_FORMATS[fmt][2])
    return buf.getvalue()


def streaming_wav_header(sr: int = TTS_SR, channels: int = 1, bits: int = 16) -> bytes:
    """44-byte WAV header with 0xFFFFFFFF lengths, for streams of unknown duration."""
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1, channels, sr,
        sr * block_align, block_align, bits, b"data", 0xFFFFFFFF,
    )


# ----------------------------------------------------------------------------
# Speed
# ----------------------------------------------------------------------------
def _atempo_chain(speed: float) -> str:
    """ffmpeg's atempo accepts 0.5..2.0 per instance; chain instances beyond that."""
    parts, remaining = [], speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def change_speed(wav: np.ndarray, speed: float, sr: int = TTS_SR) -> np.ndarray:
    """Change playback speed (0.25x-4x).

    Uses ffmpeg's pitch-preserving `atempo` time-stretch when ffmpeg is present;
    otherwise falls back to plain resampling (which also shifts pitch).
    """
    speed = max(0.25, min(float(speed), 4.0))
    if abs(speed - 1.0) < 0.02 or len(wav) == 0:
        return wav
    if _FFMPEG:
        cmd = [_FFMPEG, "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
               "-i", "pipe:0", "-filter:a", _atempo_chain(speed), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
        proc = subprocess.run(cmd, input=wav.astype("<f4").tobytes(), capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
        logger.warning("ffmpeg atempo failed, falling back to resampling: %s", proc.stderr[-200:])
    return resample(wav, int(sr * speed), sr)


# ----------------------------------------------------------------------------
# Subtitles
# ----------------------------------------------------------------------------
def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def format_srt(segments: List[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(f"{i}\n{_timestamp(seg['start'], ',')} --> {_timestamp(seg['end'], ',')}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)


def format_vtt(segments: List[dict]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(f"{_timestamp(seg['start'], '.')} --> {_timestamp(seg['end'], '.')}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)


# ----------------------------------------------------------------------------
# Streaming MP3
# ----------------------------------------------------------------------------
class Mp3StreamEncoder:
    """Continuous MP3 encoding for streamed synthesis.

    Encoding each audio chunk as its own MP3 file leaves encoder padding at every
    seam (audible clicks), so chunks are fed through one long-lived ffmpeg process.
    Without ffmpeg, chunks are encoded individually as a fallback.
    """

    def __init__(self, on_data, sr: int = TTS_SR):
        import threading
        self._on_data, self._sr = on_data, sr
        self._proc = None
        if _FFMPEG:
            self._proc = subprocess.Popen(
                [_FFMPEG, "-nostdin", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                 "-codec:a", "libmp3lame", "-b:a", "96k", "-flush_packets", "1", "-f", "mp3", "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._reader = threading.Thread(target=self._pump, daemon=True)
            self._reader.start()

    def _pump(self):
        while True:
            data = self._proc.stdout.read1(8192)
            if not data:
                break
            self._on_data(data)

    def write(self, wav: np.ndarray) -> None:
        if self._proc is None:
            self._on_data(encode_audio(wav, "mp3", self._sr))
            return
        try:
            self._proc.stdin.write(float_to_pcm16(wav))
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._reader.join(timeout=10)
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
