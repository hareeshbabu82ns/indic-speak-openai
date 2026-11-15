"""Universal AI Serving Platform - Phase 1: Text-to-Speech and Speech-to-Text.

OpenAI-compatible FastAPI server with dynamic model lifecycle management
(register, download, load, unload, delete) for Hugging Face TTS and ASR models,
accelerated on Apple Silicon (MLX / PyTorch MPS) with CPU fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Query, Request, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

import asr_engine  # noqa: E402
import tts_engine  # noqa: E402
from audio_utils import (ASR_SR, AUDIO_FORMATS, STREAMABLE_FORMATS, TTS_SR, Mp3StreamEncoder, decode_audio,  # noqa: E402
                         encode_audio, float_to_pcm16, format_srt, format_vtt, has_ffmpeg, streaming_wav_header)
from model_manager import MODALITIES, SUPPORTED_MODALITIES, ModelError, ModelManager  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("server")
for _noisy in ("httpx", "httpx2", "httpcore"):      # one line per Hub request otherwise
    logging.getLogger(_noisy).setLevel(logging.WARNING)

VERSION = "2.0.0"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
GENERATED_DIR = DATA_DIR / "generated"
TRANSCRIPTS_DIR = DATA_DIR / "transcriptions"
UPLOADS_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
for _d in (GENERATED_DIR, GENERATED_DIR / "images", TRANSCRIPTS_DIR, UPLOADS_DIR, DATA_DIR / "cache"):
    _d.mkdir(parents=True, exist_ok=True)

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

manager = ModelManager(os.getenv("MODELS_CONFIG", str(BASE_DIR / "config" / "models.json")))
manager.register_factory("tts", tts_engine.create_adapter)
manager.register_factory("asr", asr_engine.create_adapter)


# ----------------------------------------------------------------------------
# App + lifecycle
# ----------------------------------------------------------------------------
async def _preload(names: List[str]) -> None:
    for name in names:
        targets = [manager.defaults.get(m) for m in SUPPORTED_MODALITIES] if name == "defaults" else [name]
        for target in filter(None, targets):
            try:
                await manager.ensure_loaded(target)
            except Exception as e:
                logger.error("Preload of '%s' failed: %s", target, getattr(e, "message", e))


async def _idle_reaper() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            unloaded = await manager.unload_idle()
            if unloaded:
                logger.info("Unloaded idle models: %s", ", ".join(unloaded))
        except Exception as e:
            logger.warning("Idle unload failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload = [p.strip() for p in os.getenv("PRELOAD_MODELS", "").split(",") if p.strip()]
    background = [asyncio.create_task(_idle_reaper())]
    if preload:
        logger.info("Preloading in the background: %s", ", ".join(preload))
        background.append(asyncio.create_task(_preload(preload)))
    logger.info("Server ready. %d models registered; defaults: %s", len(manager.models), manager.defaults)
    yield
    for task in background:
        task.cancel()
    manager.shutdown()


app = FastAPI(
    title="Universal AI Serving Platform (Phase 1: TTS + ASR)",
    description="OpenAI-compatible speech synthesis and transcription with Hugging Face model lifecycle management.",
    version=VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-File-Id", "X-File-Url", "X-Audio-Duration", "X-Generation-Time",
                    "X-RTF", "X-Model", "X-Voice", "X-Sample-Rate", "X-Language", "X-Script-Mode"],
)


def _error(status: int, message: str, code: str = "invalid_request_error", param: Optional[str] = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind, "param": param, "code": code}})


@app.exception_handler(ModelError)
async def _model_error(_: Request, exc: ModelError):
    return _error(exc.status, exc.message, exc.code, exc.param)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, str(exc.detail), "not_found" if exc.status_code == 404 else "http_error")


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = [str(p) for p in first.get("loc", []) if p not in ("body", "query", "path")]
    param = ".".join(loc) or None
    return _error(400, f"{param + ': ' if param else ''}{first.get('msg', 'Invalid request')}", "invalid_request_error", param)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return _error(500, f"{type(exc).__name__}: {exc}", "internal_error")


# ----------------------------------------------------------------------------
# Saved files (TTS audio, ASR transcripts + their uploads)
# ----------------------------------------------------------------------------
def _new_file_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _check_file_id(file_id: str) -> str:
    if not FILE_ID_RE.fullmatch(file_id):
        raise ModelError("Invalid file id.", status=400, code="invalid_file_id", param="file_id")
    return file_id


def _save_tts(file_id: str, audio: bytes, fmt: str, duration: float, meta: dict) -> dict:
    ext = AUDIO_FORMATS[fmt][1]
    path = GENERATED_DIR / f"{file_id}.{ext}"
    path.write_bytes(audio)
    record = {
        "id": file_id, "type": "tts", "filename": path.name, "format": fmt,
        "duration": round(duration, 3), "size_bytes": len(audio),
        "created": int(time.time()), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **meta,
    }
    (GENERATED_DIR / f"{file_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _save_transcript(file_id: str, record: dict, model_id: str, filename: Optional[str],
                     upload: Optional[bytes], stats: dict) -> None:
    saved = {
        "id": file_id, "type": "asr", "filename": filename or f"{file_id}.wav", "model": model_id,
        "created": int(time.time()), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats, **record,
    }
    if upload is not None:
        ext = Path(filename or "").suffix.lower()
        ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext or "") else ".wav"
        (UPLOADS_DIR / f"{file_id}{ext}").write_bytes(upload)
        saved["audio_file"] = f"{file_id}{ext}"
        saved["size_bytes"] = len(upload)
    (TRANSCRIPTS_DIR / f"{file_id}.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tts_audio_path(file_id: str) -> Optional[Path]:
    for p in GENERATED_DIR.glob(f"{file_id}.*"):
        if p.suffix != ".json" and p.stem == file_id:
            return p
    return None


def _upload_path(file_id: str) -> Optional[Path]:
    for p in UPLOADS_DIR.glob(f"{file_id}.*"):
        if p.stem == file_id:
            return p
    return None


_MEDIA_BY_EXT = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
                 ".raw": "audio/pcm", ".m4a": "audio/mp4", ".webm": "audio/webm", ".mp4": "audio/mp4"}


def _list_files() -> List[dict]:
    items: List[dict] = []
    for meta_path in GENERATED_DIR.glob("*.json"):
        rec = _read_json(meta_path)
        if not rec or not _tts_audio_path(meta_path.stem):
            continue
        rec.setdefault("id", meta_path.stem)
        rec["type"] = "tts"
        rec.pop("path", None)
        rec["audio_url"] = f"/v1/audio/files/{rec['id']}"
        items.append(rec)
    for meta_path in TRANSCRIPTS_DIR.glob("*.json"):
        rec = _read_json(meta_path)
        if not rec:
            continue
        rec.setdefault("id", meta_path.stem)
        rec["type"] = "asr"
        rec["audio_url"] = f"/v1/audio/files/{rec['id']}?audio=true" if _upload_path(rec["id"]) else None
        items.append(rec)
    for rec in items:
        rec["url"] = f"/v1/audio/files/{rec['id']}"
        rec["download_url"] = f"/v1/audio/files/{rec['id']}/download"
    items.sort(key=lambda r: (r.get("created") or 0, r.get("created_at", "")), reverse=True)
    return items


# ----------------------------------------------------------------------------
# Text-to-Speech
# ----------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="TTS model id or alias (indic-speak-mlx, indic-speak-torch, kokoro-82m, tts-1, ...). Defaults to the default TTS model.")
    input: str = Field(..., description="Text to synthesise")
    voice: Optional[str] = Field(default=None, description="Voice name (e.g. Sravani, Amit) or OpenAI alias (alloy, echo, fable, onyx, nova, shimmer)")
    response_format: Optional[str] = Field(default="mp3", description="mp3, wav, flac, pcm (16-bit LE) or opus")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed, 0.25-4.0")
    style: Optional[str] = Field(default="NEUTRAL", description="Delivery style / emotion tag")
    stream: Optional[bool] = Field(default=False, description="Stream audio chunks as they are generated")
    save: Optional[bool] = Field(default=True, description="Keep the audio on the server for later download")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature (model default 0.6)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible sampling")
    instructions: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; used as style when `style` is unset")


@app.post("/v1/audio/speech", tags=["Audio"])
async def audio_speech(req: SpeechRequest):
    """OpenAI-compatible speech synthesis."""
    fmt = (req.response_format or "mp3").lower()
    if fmt not in AUDIO_FORMATS:
        raise ModelError(f"response_format must be one of {', '.join(AUDIO_FORMATS)}.", param="response_format")
    text = (req.input or "").strip()
    if not text:
        raise ModelError("input must not be empty.", param="input")
    if len(text) > MAX_INPUT_CHARS:
        raise ModelError(f"input is {len(text)} characters; the limit is {MAX_INPUT_CHARS}.", param="input")
    if req.stream and fmt not in STREAMABLE_FORMATS:
        raise ModelError(f"Streaming supports {', '.join(STREAMABLE_FORMATS)}; got '{fmt}'.", param="response_format")

    spec = await manager.ensure_loaded(req.model, "tts")
    defaults = spec.get("default_params", {})
    voice = req.voice or defaults.get("voice")
    style = req.style
    if style in (None, "", "NEUTRAL") and req.instructions in tts_engine.AVAILABLE_STYLES:
        style = req.instructions
    speed = float(req.speed or defaults.get("speed") or 1.0)
    params: Dict[str, Any] = {"temperature": req.temperature if req.temperature is not None else defaults.get("temperature", 0.6)}
    if req.seed is not None:
        params["seed"] = req.seed
    save = req.save is not False
    file_id = _new_file_id("tts")
    media_type, ext, _ = AUDIO_FORMATS[fmt]
    base_meta = {"text": text, "model": spec["id"], "speed": speed}
    headers = {"X-Model": spec["id"], "X-Sample-Rate": str(TTS_SR)}
    if save:
        headers.update({"X-File-Id": file_id, "X-File-Url": f"/v1/audio/files/{file_id}"})

    if req.stream:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        def emit(data: Optional[bytes]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, data)

        def job() -> None:
            import numpy as np
            chunks, t0, ttfa = [], time.time(), None
            encoder = Mp3StreamEncoder(emit) if fmt == "mp3" else None
            adapter = None
            try:
                adapter = manager.adapter(spec["id"])
                for chunk in tts_engine.stream_synthesis(adapter, text, voice, style, speed, params, cancel):
                    if ttfa is None:
                        ttfa = time.time() - t0
                    chunks.append(chunk)
                    if encoder is not None:
                        encoder.write(chunk)
                    else:
                        emit(float_to_pcm16(chunk))
            except Exception:
                logger.exception("Streaming synthesis failed")
            finally:
                if encoder is not None:
                    encoder.close()
                if save and chunks and adapter is not None:
                    try:
                        wav = np.concatenate(chunks)
                        total, audio_s = time.time() - t0, len(wav) / TTS_SR
                        _save_tts(file_id, encode_audio(wav, fmt), fmt, audio_s, {
                            **base_meta, "voice": adapter.resolve_voice(voice),
                            "style": tts_engine.normalize_style(style) or "NEUTRAL", "streamed": True,
                            "stats": {"engine": adapter.engine, "total_s": round(total, 3), "audio_s": round(audio_s, 3),
                                      "rtf": round(total / max(audio_s, 1e-4), 3), "ttfa_s": round(ttfa or total, 3)},
                        })
                    except Exception:
                        logger.exception("Could not save streamed audio")
                emit(None)

        manager.submit(job)

        async def body():
            try:
                if fmt == "wav":
                    yield streaming_wav_header(TTS_SR)
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                cancel.set()        # client went away: stop generating

        headers["Content-Disposition"] = f'inline; filename="{file_id}.{ext}"'
        return StreamingResponse(body(), media_type=media_type, headers=headers)

    def job():
        adapter = manager.adapter(spec["id"])
        wav, stats = tts_engine.synthesize(adapter, text, voice, style, speed, params)
        audio = encode_audio(wav, fmt)
        if save:
            _save_tts(file_id, audio, fmt, stats["audio_s"],
                      {**base_meta, "voice": stats["voice"], "style": stats["style"], "streamed": False, "stats": stats})
        return audio, stats

    audio, stats = await manager.run(job)
    headers.update({
        "Content-Disposition": f'attachment; filename="{file_id}.{ext}"',
        "X-Audio-Duration": str(stats["audio_s"]), "X-Generation-Time": str(stats["total_s"]),
        "X-RTF": str(stats["rtf"]), "X-Voice": stats["voice"],
    })
    return Response(content=audio, media_type=media_type, headers=headers)


@app.get("/v1/audio/voices", tags=["Audio"])
async def list_voices(model: Optional[str] = Query(None, description="TTS model id; defaults to the default TTS model")):
    """Voice catalog. Top-level fields describe `model`; `models` holds every registered TTS model."""
    per_model = {}
    for spec in list(manager.models.values()):
        if spec["modality"] == "tts":
            cat = tts_engine.catalog_for(spec)
            per_model[spec["id"]] = {"voices": cat["voices"], "openai_aliases": cat["openai_aliases"],
                                     "styles": cat["styles"], "default_voice": cat["default_voice"]}
    if not per_model:
        return {"model": None, "voices": {}, "openai_aliases": {}, "styles": [], "by_language": {}, "models": {}}
    spec = manager.resolve(model, "tts")
    current = per_model[spec["id"]]
    by_language: Dict[str, Dict[str, List[str]]] = {}
    for name, v in current["voices"].items():
        by_language.setdefault(v.get("lang_name") or "Default", {}).setdefault(v.get("gender") or "Voice", []).append(name)
    return {"model": spec["id"], **current, "count": len(current["voices"]), "by_language": by_language, "models": per_model}


# ----------------------------------------------------------------------------
# Speech-to-Text
# ----------------------------------------------------------------------------
async def _transcribe(file: UploadFile, model: Optional[str], language: Optional[str], prompt: Optional[str],
                      response_format: str, temperature: float, mode: str, save: bool, task: str, chunk_seconds: float):
    fmt = (response_format or "json").lower()
    if fmt not in asr_engine.RESPONSE_FORMATS:
        raise ModelError(f"response_format must be one of {', '.join(asr_engine.RESPONSE_FORMATS)}.", param="response_format")
    content = await file.read()
    if not content:
        raise ModelError("Uploaded file is empty.", param="file")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ModelError(f"Upload exceeds {MAX_UPLOAD_MB} MB.", status=413, code="file_too_large", param="file")

    spec = await manager.ensure_loaded(model, "asr")
    defaults = spec.get("default_params", {})
    language = language if language is not None else defaults.get("language")
    mode = asr_engine.normalize_mode(mode or defaults.get("mode"))
    try:
        wav = await asyncio.get_running_loop().run_in_executor(None, decode_audio, content, ASR_SR)
    except ValueError as e:
        raise ModelError(str(e), param="file", code="invalid_audio")

    def job():
        t0 = time.time()
        adapter = manager.adapter(spec["id"])
        record = asr_engine.transcribe(adapter, wav, language, mode, task, temperature, prompt, chunk_seconds)
        elapsed = time.time() - t0
        return record, {"engine": adapter.engine, "total_s": round(elapsed, 3),
                        "rtf": round(elapsed / max(record["duration"], 1e-4), 3)}

    record, stats = await manager.run(job)
    headers = {"X-Model": spec["id"], "X-Language": str(record["language"]), "X-Generation-Time": str(stats["total_s"]),
               "X-Script-Mode": record["mode"]}
    if save:
        file_id = _new_file_id("asr")
        _save_transcript(file_id, record, spec["id"], file.filename, content, stats)
        headers.update({"X-File-Id": file_id, "X-File-Url": f"/v1/audio/files/{file_id}"})
    body, media_type = asr_engine.format_transcript(record, fmt)
    if isinstance(body, str):
        return PlainTextResponse(body, media_type=media_type, headers=headers)
    return JSONResponse(body, headers=headers)


@app.post("/v1/audio/transcriptions", tags=["Audio"])
async def audio_transcriptions(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, webm, flac)"),
    model: Optional[str] = Form(None, description="ASR model id or alias (indic-transcribe, whisper-large-v3, whisper-1, ...)"),
    language: Optional[str] = Form(None, description="ISO-639 language code, or 'auto' / omitted for detection"),
    prompt: Optional[str] = Form(None, description="Optional context prompt (Whisper models)"),
    response_format: str = Form("json", description="json, text, verbose_json, srt or vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    mode: str = Form("native", description="Indic script mode: native, mixed (itn) or romanised"),
    save: bool = Form(True, description="Keep the upload and transcript on the server"),
    chunk_seconds: float = Form(30.0, ge=5.0, le=30.0, description="Max segment length; lower gives finer subtitle timestamps at a small accuracy cost"),
):
    """OpenAI-compatible transcription."""
    return await _transcribe(file, model, language, prompt, response_format, temperature, mode, save, "transcribe", chunk_seconds)


@app.post("/v1/audio/translations", tags=["Audio"])
async def audio_translations(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    save: bool = Form(True),
):
    """OpenAI-compatible translation. Whisper models translate to English; Indic-Transcribe
    returns a romanised (Latin script) transliteration."""
    return await _transcribe(file, model, None, prompt, response_format, temperature, "romanised", save, "translate", 30.0)


@app.get("/v1/audio/languages", tags=["Audio"])
async def list_languages(model: Optional[str] = Query(None, description="ASR model id; defaults to the default ASR model")):
    """Supported ASR languages and script modes."""
    per_model = {s["id"]: asr_engine.catalog_for(s) for s in list(manager.models.values()) if s["modality"] == "asr"}
    if not per_model:
        return {"model": None, "languages": {}, "modes": [], "models": {}}
    spec = manager.resolve(model, "asr")
    current = per_model[spec["id"]]
    return {"model": spec["id"], "languages": current["languages"], "modes": current["modes"],
            "count": len(current["languages"]), "models": per_model}


# ----------------------------------------------------------------------------
# Saved files
# ----------------------------------------------------------------------------
@app.get("/v1/audio/files", tags=["Files"])
async def list_files(type: Optional[str] = Query(None, description="Filter: tts or asr"), limit: int = Query(500, ge=1, le=5000)):
    """Saved TTS audio and ASR transcripts, newest first."""
    items = _list_files()
    if type:
        items = [f for f in items if f["type"] == type.lower()]
    return items[:limit]


@app.get("/v1/audio/files/{file_id}", tags=["Files"])
async def get_file(file_id: str, audio: bool = Query(False, description="For transcripts: return the uploaded audio instead of the JSON record")):
    """Stream a saved TTS file, or return a saved transcript record."""
    _check_file_id(file_id)
    path = _tts_audio_path(file_id)
    if path:
        return FileResponse(path, media_type=_MEDIA_BY_EXT.get(path.suffix, "application/octet-stream"))
    record = _read_json(TRANSCRIPTS_DIR / f"{file_id}.json")
    if record is not None:
        if audio:
            upload = _upload_path(file_id)
            if not upload:
                raise ModelError("No audio was stored for this transcript.", status=404, code="file_not_found")
            return FileResponse(upload, media_type=_MEDIA_BY_EXT.get(upload.suffix, "application/octet-stream"))
        return record
    raise ModelError(f"File '{file_id}' not found.", status=404, code="file_not_found")


@app.get("/v1/audio/files/{file_id}/download", tags=["Files"])
async def download_file(file_id: str, format: str = Query("json", description="For transcripts: json, txt, srt or vtt")):
    """Download a saved file as an attachment."""
    _check_file_id(file_id)
    path = _tts_audio_path(file_id)
    if path:
        return FileResponse(path, filename=path.name, media_type=_MEDIA_BY_EXT.get(path.suffix, "application/octet-stream"))
    record = _read_json(TRANSCRIPTS_DIR / f"{file_id}.json")
    if record is None:
        raise ModelError(f"File '{file_id}' not found.", status=404, code="file_not_found")
    fmt = format.lower()
    if fmt in ("txt", "text"):
        body, ext = record.get("text", ""), "txt"
    elif fmt == "srt":
        body, ext = format_srt(record.get("segments", [])), "srt"
    elif fmt == "vtt":
        body, ext = format_vtt(record.get("segments", [])), "vtt"
    else:
        body, ext = json.dumps(record, indent=2, ensure_ascii=False), "json"
    return Response(body, media_type="application/json" if ext == "json" else "text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{file_id}.{ext}"'})


@app.delete("/v1/audio/files/{file_id}", tags=["Files"])
async def delete_file(file_id: str):
    """Delete a saved file, its metadata and (for transcripts) the stored upload."""
    _check_file_id(file_id)
    deleted = False
    for directory in (GENERATED_DIR, TRANSCRIPTS_DIR, UPLOADS_DIR):
        for p in directory.glob(f"{file_id}.*"):
            if p.stem == file_id and p.is_file():
                p.unlink(missing_ok=True)
                deleted = True
    if not deleted:
        raise ModelError(f"File '{file_id}' not found.", status=404, code="file_not_found")
    return {"id": file_id, "object": "file", "deleted": True, "status": "deleted"}


# ----------------------------------------------------------------------------
# Models: OpenAI listing + lifecycle management
# ----------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    repo_id: str = Field(..., description="Hugging Face repository id, e.g. mlx-community/whisper-large-v3-mlx")
    modality: str = Field(..., description=f"One of: {', '.join(MODALITIES)}")
    id: Optional[str] = Field(default=None, description="Model alias used in API calls; derived from repo_id when omitted")
    backend: Optional[str] = Field(default="auto", description="mlx, transformers, diffusers or auto")
    quantization: Optional[str] = Field(default=None, description="4bit, 8bit, bf16, fp16, fp32 or default")
    adapter: Optional[str] = Field(default=None, description="Engine adapter; auto-detected when omitted")
    context_length: Optional[int] = None
    chat_template: Optional[str] = None
    default_params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    overwrite: Optional[bool] = False


class SettingsRequest(BaseModel):
    hf_token: Optional[str] = Field(default=None, description="Hugging Face token; held in memory only")
    max_loaded_models: Optional[int] = Field(default=None, ge=1, le=16)
    min_free_memory_gb: Optional[float] = Field(default=None, ge=0)
    idle_unload_minutes: Optional[float] = Field(default=None, ge=0)
    auto_download: Optional[bool] = None


@app.get("/v1/models", tags=["Models"])
async def list_models(modality: Optional[str] = Query(None), include_aliases: bool = Query(True)):
    """OpenAI model list, extended with modality, backend, repo_id, status and size_mb."""
    data = [m for m in manager.list_models() if not modality or m["modality"] == modality]
    if include_aliases:
        by_id = {m["id"]: m for m in data}
        for alias, target in manager.alias_table().items():
            if target in by_id and alias not in by_id:
                t = by_id[target]
                data.append({
                    "id": alias, "object": "model", "created": t["created"], "owned_by": "openai-compatible",
                    "alias_of": target, "repo_id": t["repo_id"], "modality": t["modality"], "backend": t["backend"],
                    "status": t["status"], "size_mb": t["size_mb"],
                })
    return {"object": "list", "data": data}


@app.post("/v1/models/register", tags=["Models"])
async def register_model(req: RegisterRequest):
    """Add a Hugging Face model to the registry (persisted to config/models.json)."""
    payload = req.model_dump(exclude_none=True)
    try:    # best effort: remember the remote size so status and progress are accurate
        payload["remote_size_bytes"] = await asyncio.get_running_loop().run_in_executor(
            None, manager._remote_size, req.repo_id.strip())
    except Exception:
        pass
    spec = manager.register(payload)
    return manager.public_model(spec)


@app.get("/v1/models/recommended", tags=["Models"])
async def recommended_models():
    """Tested model packs for one-click registration."""
    return {"object": "list", "data": manager.recommended()}


@app.get("/v1/models/tasks", tags=["Models"])
async def list_tasks():
    return {"object": "list", "data": manager.list_tasks()}


@app.get("/v1/models/tasks/{task_id}", tags=["Models"])
async def get_task(task_id: str):
    """Download / load progress: status is in_progress, completed or failed."""
    return manager.get_task(task_id)


@app.get("/v1/models/{model_id}", tags=["Models"])
async def get_model(model_id: str):
    return manager.public_model(manager.resolve(model_id))


@app.post("/v1/models/{model_id}/download", tags=["Models"])
async def download_model(model_id: str):
    """Start a background download from the Hugging Face Hub."""
    task = manager.start_download(model_id)
    return {"task_id": task["task_id"], "model_id": task["model_id"],
            "status": "downloading" if task["status"] == "in_progress" else task["status"]}


@app.post("/v1/models/{model_id}/load", tags=["Models"])
async def load_model(model_id: str):
    """Load weights into memory (downloads first when auto_download is on)."""
    t0 = time.time()
    spec = await manager.ensure_loaded(model_id)
    return {**manager.public_model(spec), "load_seconds": round(time.time() - t0, 2)}


@app.post("/v1/models/{model_id}/unload", tags=["Models"])
async def unload_model(model_id: str):
    """Free the model's memory and clear MLX / MPS caches."""
    spec = manager.resolve(model_id)
    was_loaded = await manager.unload(spec["id"])
    return {**manager.public_model(spec), "unloaded": was_loaded}


@app.post("/v1/models/{model_id}/default", tags=["Models"])
async def set_default_model(model_id: str):
    """Make this model the default for its modality (and the target of tts-1 / whisper-1)."""
    spec = manager.set_default(model_id)
    return {**manager.public_model(spec), "defaults": dict(manager.defaults)}


@app.delete("/v1/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str, delete_files: bool = Query(False, description="Also remove downloaded weights from disk")):
    """De-register a model; optionally delete its cached weights."""
    spec = manager.resolve(model_id)
    await manager.unload(spec["id"])
    return await asyncio.get_running_loop().run_in_executor(None, manager.unregister, spec["id"], delete_files)


@app.get("/v1/hf/inspect", tags=["Models"])
async def hf_inspect(repo_id: str = Query(..., description="Hugging Face repository id")):
    """Look up a repository on the Hub and suggest modality, backend and quantization."""
    return await asyncio.get_running_loop().run_in_executor(None, manager.inspect_repo, repo_id)


# ----------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------
@app.get("/v1/system/status", tags=["System"])
async def system_status():
    """Unified memory, MLX / MPS allocation, loaded models, disk cache and device capability."""
    status = await asyncio.get_running_loop().run_in_executor(None, manager.system_status)
    status.update(version=VERSION, phase=1, supported_modalities=list(SUPPORTED_MODALITIES), ffmpeg=has_ffmpeg())
    return status


@app.get("/v1/system/settings", tags=["System"])
async def get_settings():
    return manager.public_settings()


@app.post("/v1/system/settings", tags=["System"])
async def update_settings(req: SettingsRequest):
    return manager.update_settings(req.model_dump(exclude_unset=True))


@app.get("/health", tags=["System"])
async def health():
    loaded = manager.loaded_ids()
    adapters = {manager.models[m].get("adapter") for m in loaded if m in manager.models}
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"
    return {
        "status": "ok", "version": VERSION, "phase": 1, "device": device,
        "models": {
            "tts_mlx": "indic-speak-mlx" in adapters,
            "tts_torch": "indic-speak-torch" in adapters,
            "asr": any(manager.models[m]["modality"] == "asr" for m in loaded if m in manager.models),
        },
        "loaded_models": loaded,
        "defaults": dict(manager.defaults),
        "sample_rates": {"tts": TTS_SR, "asr": ASR_SR},
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.head("/", include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@app.head("/ui", include_in_schema=False)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})
    return HTMLResponse("<h2>API is running. UI not found at static/index.html</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=False)
