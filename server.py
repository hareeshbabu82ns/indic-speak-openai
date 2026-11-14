"""FastAPI OpenAI-Compatible TTS & ASR Server
Implements /v1/audio/speech, /v1/audio/transcriptions, /v1/audio/translations, and /v1/models
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_engine import ASREngine, LANGUAGES_CATALOG, MODES
from tts_engine import TTSEngine, AVAILABLE_STYLES, OPENAI_VOICE_MAP, VOICE_CATALOG

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

# Global engine instances
tts_engine = TTSEngine(data_dir="data")
asr_engine = ASREngine(data_dir="data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload TTS and ASR models on startup."""
    logger.info("Initializing models...")
    try:
        tts_engine.load_models()
    except Exception as e:
        logger.error(f"Failed to preload TTS engine: {e}")

    try:
        asr_engine.load_model()
    except Exception as e:
        logger.error(f"Failed to preload ASR engine: {e}")

    logger.info("Server startup complete. Ready to receive requests.")
    yield
    logger.info("Shutting down server...")


app = FastAPI(
    title="Indic-Speak & Transcribe OpenAI-Compatible API",
    description="OpenAI-compatible Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) API with streaming and file storage.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web apps and agents
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-File-Id", "X-File-Url", "X-Audio-Duration"]
)

# Static files directory
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
if (STATIC_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


class SpeechRequest(BaseModel):
    model: str = Field(default="indic-speak-mlx", description="TTS model id (indic-speak-mlx, indic-speak-torch, tts-1, tts-1-hd)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="Amit", description="Voice name (e.g. Sravani, Amit, Kavya) or OpenAI alias (alloy, echo, etc.)")
    response_format: Optional[str] = Field(default="mp3", description="Audio format (mp3, wav, flac, pcm, ogg)")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speed of generated speech")
    style: Optional[str] = Field(default="NEUTRAL", description="Delivery emotion or style tag")
    stream: Optional[bool] = Field(default=False, description="Stream audio chunks directly to client")
    save: Optional[bool] = Field(default=True, description="Save generated audio file on server for download")


# -----------------------------------------------------------------------------
# 1. OpenAI Compatible TTS (/v1/audio/speech)
# -----------------------------------------------------------------------------
@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    """OpenAI-compatible speech endpoint. Synthesizes text into audio."""
    fmt = (req.response_format or "mp3").lower()
    media_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
        "ogg": "audio/ogg",
        "opus": "audio/ogg"
    }
    media_type = media_map.get(fmt, "audio/mpeg")
    file_id = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    if req.stream:
        # Stream chunked audio output
        logger.info(f"Streaming speech for voice '{req.voice}' (format: {fmt})")
        stream_gen = tts_engine.stream_audio(
            text=req.input,
            voice=req.voice,
            style=req.style or "NEUTRAL",
            response_format=fmt,
            speed=req.speed or 1.0,
            save_file=req.save if req.save is not None else True,
            file_id=file_id
        )
        return StreamingResponse(
            stream_gen,
            media_type=media_type,
            headers={
                "X-File-Id": file_id,
                "X-File-Url": f"/v1/audio/files/{file_id}",
                "Content-Disposition": f'inline; filename="{file_id}.{fmt}"'
            }
        )

    # Full synthesis
    logger.info(f"Synthesizing speech for voice '{req.voice}' (model: {req.model}, format: {fmt})")
    try:
        wav, stats = tts_engine.synthesize(
            text=req.input,
            voice=req.voice,
            style=req.style or "NEUTRAL",
            speed=req.speed or 1.0,
            model_name=req.model
        )
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

    encoded_bytes = tts_engine.encode_audio(wav, format=fmt)

    meta = {}
    if req.save:
        meta = tts_engine.save_audio(
            file_id=file_id,
            wav=wav,
            format=fmt,
            metadata={
                "type": "tts",
                "text": req.input,
                "voice": tts_engine.resolve_voice(req.voice),
                "style": req.style or "NEUTRAL",
                "model": req.model,
                "speed": req.speed or 1.0,
                "stats": stats
            }
        )

    filename = meta.get("filename", f"{file_id}.{fmt}")
    return Response(
        content=encoded_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-File-Id": file_id,
            "X-File-Url": f"/v1/audio/files/{file_id}",
            "X-Audio-Duration": str(stats.get("audio_s", round(len(wav) / 24000, 3)))
        }
    )


# -----------------------------------------------------------------------------
# 2. OpenAI Compatible ASR (/v1/audio/transcriptions & /v1/audio/translations)
# -----------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(..., description="The audio file object (wav, mp3, m4a, ogg, webm, flac)"),
    model: str = Form("indic-transcribe", description="ASR model name (indic-transcribe, whisper-1)"),
    language: Optional[str] = Form(None, description="ISO-639-1 language code or auto"),
    prompt: Optional[str] = Form(None, description="Optional prompt / context"),
    response_format: str = Form("json", description="Output format: json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, description="Sampling temperature"),
    timestamp_granularities: Optional[str] = Form(None, description="Timestamp granularity (segment/word)"),
    mode: str = Form("native", description="Output mode: native, mixed/itn, romanised"),
    save: bool = Form(True, description="Save audio upload and transcript record")
):
    """OpenAI-compatible transcription endpoint. Transcribes speech audio to text."""
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save audio upload to disk if requested
        if save:
            upload_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            ext = Path(file.filename or "audio.wav").suffix or ".wav"
            upload_path = Path("data/uploads") / f"{upload_id}{ext}"
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(content)

        result = asr_engine.transcribe(
            audio_source=content,
            language=language,
            mode=mode,
            response_format=response_format,
            temperature=temperature,
            save_file=save,
            filename=file.filename
        )

        fmt = response_format.lower()
        if fmt in ("text", "srt", "vtt"):
            media = "text/plain" if fmt != "vtt" else "text/vtt"
            return PlainTextResponse(content=str(result), media_type=media)
        elif fmt == "verbose_json":
            return JSONResponse(content=result)
        else:
            # Standard JSON {"text": "..."}
            return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile = File(...),
    model: str = Form("indic-transcribe"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """OpenAI-compatible translation endpoint. Transcribes and romanizes Indic speech."""
    return await audio_transcriptions(
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        mode="romanised",
        save=True
    )


# -----------------------------------------------------------------------------
# 3. Model & Voice Catalogs (/v1/models, /v1/audio/voices)
# -----------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """List available OpenAI-compatible models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "indic-speak-mlx",
                "object": "model",
                "created": 1700000000,
                "owned_by": "indic-speak",
                "capabilities": ["text-to-speech", "streaming"],
                "description": "Apple Silicon MLX 4-bit accelerated text-to-speech with 44 Indic voices."
            },
            {
                "id": "indic-speak-torch",
                "object": "model",
                "created": 1700000000,
                "owned_by": "bodhan-ai",
                "capabilities": ["text-to-speech"],
                "description": "PyTorch MPS/CPU text-to-speech model."
            },
            {
                "id": "tts-1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai-compatible",
                "capabilities": ["text-to-speech"],
                "description": "OpenAI TTS standard model alias."
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai-compatible",
                "capabilities": ["text-to-speech"],
                "description": "OpenAI TTS-1-HD model alias."
            },
            {
                "id": "indic-transcribe",
                "object": "model",
                "created": 1700000000,
                "owned_by": "bodhan-ai",
                "capabilities": ["speech-to-text", "transcription", "language-identification"],
                "description": "Indic Transcribe Canary model supporting 27 Indic languages + English."
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai-compatible",
                "capabilities": ["speech-to-text"],
                "description": "OpenAI Whisper model alias."
            }
        ]
    }


@app.get("/v1/audio/voices")
async def list_voices():
    """List all supported voices with descriptions, languages, and OpenAI aliases."""
    return {
        "voices": VOICE_CATALOG,
        "openai_aliases": OPENAI_VOICE_MAP,
        "styles": AVAILABLE_STYLES
    }


@app.get("/v1/audio/languages")
async def list_languages():
    """List all supported ASR languages."""
    return {
        "languages": LANGUAGES_CATALOG,
        "modes": MODES
    }


# -----------------------------------------------------------------------------
# 4. File Management & Download Endpoints
# -----------------------------------------------------------------------------
@app.get("/v1/audio/files")
async def list_files():
    """List all saved TTS and ASR files with metadata."""
    tts_files = tts_engine.list_saved_files()
    asr_files = asr_engine.list_saved_transcriptions()

    all_files = []
    for f in tts_files:
        f["type"] = "tts"
        all_files.append(f)
    for f in asr_files:
        f["type"] = "asr"
        all_files.append(f)

    # Sort descending by creation
    all_files.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return all_files


@app.get("/v1/audio/files/{file_id}")
async def get_file(file_id: str):
    """Stream or preview a saved audio file or transcript JSON."""
    # Check TTS generated files
    tts_dir = Path("data/generated")
    for f in tts_dir.glob(f"{file_id}.*"):
        if f.suffix != ".json":
            return FileResponse(f, media_type="audio/" + f.suffix.lstrip("."))

    # Check ASR records
    asr_file = Path("data/transcriptions") / f"{file_id}.json"
    if asr_file.exists():
        with open(asr_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/v1/audio/files/{file_id}/download")
async def download_file(file_id: str):
    """Download a saved audio or transcript file as an attachment."""
    tts_dir = Path("data/generated")
    for f in tts_dir.glob(f"{file_id}.*"):
        if f.suffix != ".json":
            return FileResponse(f, filename=f.name, headers={"Content-Disposition": f'attachment; filename="{f.name}"'})

    asr_file = Path("data/transcriptions") / f"{file_id}.json"
    if asr_file.exists():
        return FileResponse(asr_file, filename=f"{file_id}.json", headers={"Content-Disposition": f'attachment; filename="{file_id}.json"'})

    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/v1/audio/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a saved file and its metadata."""
    deleted = False
    for d in (Path("data/generated"), Path("data/transcriptions"), Path("data/uploads")):
        for f in d.glob(f"{file_id}*"):
            f.unlink(missing_ok=True)
            deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "deleted", "id": file_id}


# -----------------------------------------------------------------------------
# 5. UI and Health Check Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.head("/")
@app.get("/ui", response_class=HTMLResponse)
@app.head("/ui")
async def serve_ui():
    """Serve the web testing UI."""
    html_file = Path("static/index.html")
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Indic-Speak & Transcribe API is running. UI not found in static/index.html</h2>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models": {
            "tts_mlx": tts_engine.mlx_tts is not None,
            "tts_torch": tts_engine.torch_tts is not None,
            "asr": asr_engine.asr is not None
        },
        "device": asr_engine.device,
        "sample_rates": {
            "tts": 24000,
            "asr": 16000
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False)
