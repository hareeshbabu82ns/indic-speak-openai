Build a production-ready, OpenAI-compatible Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) REST API server using FastAPI, accelerated for Apple Silicon (MLX and PyTorch MPS) with CPU fallback, based on `bodhan-ai/indic-speak`, `iniquitous/indic-speak-mlx-bf16`, and `bodhan-ai/indic-transcribe-core`. The project must include a self-contained web testing UI, a bootstrap startup script (`start.sh`), and an automated test suite (`test_api.py`).

### 1. Core Architecture & Model Engines

1. **Text-to-Speech (TTS) Engine (`tts_engine.py`)**:
   - Primary backend: `iniquitous/indic-speak-mlx-bf16` (MLX 4-bit/BF16 accelerated for Apple Silicon GPU with SNAC/Vocos decode).
   - Secondary/fallback backend: `bodhan-ai/indic-speak` using PyTorch (`mps`/`cuda`/`cpu`).
   - Catalog: Support all 44 first-party voices across 22 Indic languages (Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu) and English.
   - Standard OpenAI voice mapping: Map `alloy` -> `Amit`, `echo` -> `Arun`, `fable` -> `Deepika`, `onyx` -> `Vamsi`, `nova` -> `Kavya`, `shimmer` -> `Sravani`.
   - Voice style tags: Support delivery styles including `NEUTRAL`, `HAPPY`, `SAD`, `ANGER`, `FEAR`, `SURPRISE`, `news`, `AIR style news`, `TV style news`, `advertisements`, `children's stories`, `educational lecture`, `Customer Care`, and `single person narration audiobook`.
   - Long-text handling: Use sentence-chunked synthesis (`speak_long`) to prevent 4-bit model truncation on multi-sentence inputs.
   - Speed adjustment: Allow playback speed adjustment from 0.25x to 4.0x via resampling.
   - Formats: Support `mp3`, `wav` (24 kHz mono), `flac`, and `pcm` (raw 16-bit signed integer LE).
   - Delivery modes:
     - Non-streaming: Full waveform synthesis, saving to `data/generated/{file_id}.{format}` with metadata JSON, returning audio binary with `X-File-Id`, `X-File-Url`, and `Content-Disposition`.
     - Real-time streaming (`stream=true`): Yield chunked audio bytes via HTTP chunked transfer. For `wav`, prepend an indefinite 44-byte WAV streaming header (`0xFFFFFFFF` data length). In the background, accumulate and persist the generated stream so it can be downloaded afterward.

2. **Automatic Speech Recognition (ASR) Engine (`asr_engine.py`)**:
   - Backend: `bodhan-ai/indic-transcribe-core` (Indic Canary architecture on `mps`/`cuda`/`cpu`).
   - Languages: Support all 27 Indic languages + English, plus automatic language detection (`language=None` or `"auto"`).
   - Long-form audio: Automatically segment audio exceeding 30 seconds at pauses/silence (`split_on_silence`) to prevent model repetition collapse.
   - Audio ingestion: Robustly accept any audio input (WAV, MP3, M4A, OGG, WEBM, FLAC) via SoundFile with FFmpeg piping fallback, resampling to 16 kHz mono float32.
   - Output formats: `json` (`{"text": "..."}`), `text` (plain string), `verbose_json` (`{"task": "transcribe", "language": "...", "duration": ..., "text": "...", "segments": [{"id": ..., "start": ..., "end": ..., "text": ...}]}`), `srt` (SubRip), and `vtt` (WebVTT).
   - Script modes: `native` (default), `mixed`/`itn` (Latin loanwords and digits), and `romanised` (transliteration).
   - Persistence: Save incoming audio to `data/uploads/` and transcript metadata to `data/transcriptions/{file_id}.json`.

---

### 2. OpenAI-Compatible API Server (`server.py`)

Implement a FastAPI application with lifespan model preloading and CORS middleware supporting the following endpoints:

- `POST /v1/audio/speech`:
  - Request body (JSON): `model` (e.g. `indic-speak-mlx`, `indic-speak-torch`, `tts-1`), `input` (text), `voice` (Indic voice name or OpenAI alias), `response_format` (`mp3`/`wav`/`flac`/`pcm`), `speed` (float, default 1.0), `style` (optional string, default `NEUTRAL`), `stream` (bool, default false), `save` (bool, default true).
  - Returns audio stream with appropriate `Content-Type`, `X-File-Id`, `X-File-Url`, and `Content-Disposition`.
- `POST /v1/audio/transcriptions`:
  - Multipart form upload: `file` (UploadFile), `model` (`indic-transcribe`, `whisper-1`), `language` (optional ISO-639-1 code), `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `mode` (`native`, `mixed`, `romanised`), `temperature`, `save` (bool).
  - Returns transcript in the requested format.
- `POST /v1/audio/translations`:
  - Multipart form upload, transcribes and outputs romanised transliteration.
- `GET /v1/models`: Standard OpenAI list format containing `indic-speak-mlx`, `indic-speak-torch`, `indic-transcribe`, `tts-1`, `tts-1-hd`, `whisper-1`.
- `GET /v1/audio/voices`: Returns catalog of all 44 voices with gender, language, and OpenAI voice aliases.
- `GET /v1/audio/languages`: Returns all 28 supported ASR languages and script modes.
- `GET /v1/audio/files`: Returns list of all saved TTS and ASR files with metadata.
- `GET /v1/audio/files/{file_id}`: Stream or view a saved file.
- `GET /v1/audio/files/{file_id}/download`: Download file with `Content-Disposition: attachment`.
- `DELETE /v1/audio/files/{file_id}`: Delete a saved file and its metadata.
- `GET /` and `GET /ui`: Serve the web testing interface (`static/index.html`).
- `GET /health`: Health status endpoint reporting model readiness and device status.

---

### 3. Web Testing UI (`static/index.html`)

Create a self-contained, responsive single-page web UI (dark theme, no mandatory external CDN dependencies) accessible at `http://localhost:8000/`:

- **Tab 1: Text-to-Speech (TTS)**:
  - Textarea with language presets (Telugu, Hindi, Tamil, Bengali, Sanskrit, English).
  - Voice dropdown (44 voices grouped by language and gender + 6 OpenAI aliases).
  - Emotion / style selector, format selector (MP3, WAV, FLAC, PCM), speed slider.
  - Action buttons: "Generate Audio (Save & Download)" and "Stream Directly".
  - Audio player with scrubber, metadata stats (duration, generation time, Real-Time Factor), and a direct "Download Audio File" button.
  - Live cURL command preview.
- **Tab 2: Speech-to-Text (ASR)**:
  - In-browser microphone recorder with live pulse animation, recording timer, and preview player.
  - Drag-and-drop file upload zone supporting WAV, MP3, M4A, OGG, WEBM, FLAC.
  - Language dropdown (Auto-detect + 27 Indic languages + English), script mode selector, format selector (`verbose_json`, `json`, `text`, `srt`, `vtt`).
  - Result card displaying detected language badge, copy text button, transcript download (.txt, .srt, .json), and an interactive segment list with clickable timestamps.
- **Tab 3: Saved Files & Downloads**:
  - History table listing all generated TTS files and ASR transcripts with inline audio playback and download actions.
- **Tab 4: OpenAI API Code**:
  - Copyable integration snippets for official `openai` Python SDK and `curl`.

---

### 4. Automated Startup Script (`start.sh`)

Write a POSIX-compliant, executable `start.sh` script:

1. Detects whether `.venv` exists in the project root; if not, creates one using `python3.12` or `python3`.
2. Activates `.venv`.
3. Checks and installs all missing packages from `requirements.txt`.
4. Creates necessary directories (`data/generated`, `data/transcriptions`, `data/uploads`, `static`).
5. Prints a console banner displaying URLs for Web UI, OpenAPI Swagger docs, and API endpoints.
6. Launches `uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} "$@"`.

---

### 5. Automated Verification Suite (`test_api.py`)

Write a Python test script using `requests` and `soundfile` that verifies:

1. `GET /health` and `GET /v1/models` return 200 with model IDs present.
2. `GET /v1/audio/voices` returns all 44 voices and aliases.
3. `POST /v1/audio/speech` (non-streaming): generates valid 24 kHz audio, returns headers `X-File-Id` and `X-File-Url`, and successfully downloads the file via `/v1/audio/files/{file_id}/download`.
4. `POST /v1/audio/speech` (streaming): validates HTTP chunked streaming and chunk reception.
5. `POST /v1/audio/transcriptions`: tests transcription on both short audio and long audio, auto language detection, segment timestamp verification (`verbose_json`), and `srt` formatting.
6. `GET /v1/audio/files`: validates that generated files appear in the history list.
