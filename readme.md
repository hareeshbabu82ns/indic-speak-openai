# Indic-Speak & Transcribe: OpenAI-Compatible Speech API

An OpenAI-compatible **Text-to-Speech (TTS)** and **Automatic Speech Recognition (ASR)** API server powered by `bodhan-ai/indic-speak`, `iniquitous/indic-speak-mlx-bf16`, and `bodhan-ai/indic-transcribe-core`.

---

## ⚡ Quick Start

Start the server using `start.sh`. It automatically creates the virtual environment if not present, activates it, verifies dependencies from `requirements.txt`, and launches the API server:

```bash
chmod +x start.sh
./start.sh
```

- **Web UI Tester:** [http://localhost:8000/](http://localhost:8000/)
- **Interactive OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🎧 Features

- **OpenAI-Compatible Speech API (`POST /v1/audio/speech`)**:
  - Direct drop-in replacement for OpenAI TTS.
  - Supports non-streaming synthesis with automatic file saving and downloading (`X-File-Id`, `X-File-Url`).
  - Supports **real-time audio streaming** (`"stream": true`) directly to clients with chunk transfer encoding.
  - Formats: `mp3`, `wav`, `flac`, `pcm` (s16le raw).
  - 44 first-party voices across 22 Indic languages + English.
  - Seamless aliases for standard OpenAI voices (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`).
  - Accelerated with **Apple Silicon MLX** GPU and **PyTorch MPS**.

- **OpenAI-Compatible Transcription API (`POST /v1/audio/transcriptions` & `/v1/audio/translations`)**:
  - Direct drop-in replacement for OpenAI Whisper.
  - Automatic language detection + support for 27 Indic languages and English.
  - Supports both short audio and long-form chunked audio without degradation.
  - Formats: `json`, `verbose_json` (with segment timestamps), `text`, `srt` (SubRip), `vtt` (WebVTT).
  - Script modes: `native` (default), `mixed`/`itn` (Latin loanwords & digits), `romanised` (transliteration).

- **Modern Web UI**:
  - Interactive browser testing for both TTS and ASR.
  - In-browser voice recording (microphone) and file upload.
  - Audio playback with waveform, speed control, download buttons, and segment inspection.
  - File Explorer for browsing, listening, and downloading previously generated/transcribed audio files.

---

## 💻 Python OpenAI SDK Usage

```python
from openai import OpenAI

# Connect to local Indic-Speak server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 1. Text-to-Speech (TTS)
response = client.audio.speech.create(
    model="indic-speak-mlx",     # or "indic-speak-torch", "tts-1"
    voice="Sravani",             # or "Amit", "Kavya", "alloy", "nova"
    input="కరాగ్రే వసతే లక్ష్మీః కరమధ్యే సరస్వతీ",
    response_format="mp3"
)
response.stream_to_file("speech.mp3")

# 2. Speech-to-Text (ASR)
with open("speech.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="indic-transcribe",
        file=audio_file,
        language="te"  # or None for auto-detect
    )
print(transcript.text)
```

---

## 📡 Key Endpoints

| Method | Endpoint                             | Description                                       |
| ------ | ------------------------------------ | ------------------------------------------------- |
| `POST` | `/v1/audio/speech`                   | OpenAI-compatible TTS (streaming & non-streaming) |
| `POST` | `/v1/audio/transcriptions`           | OpenAI-compatible ASR transcription               |
| `POST` | `/v1/audio/translations`             | OpenAI-compatible transliteration/translation     |
| `GET`  | `/v1/models`                         | List available models                             |
| `GET`  | `/v1/audio/voices`                   | Catalog of 44 voices & styles                     |
| `GET`  | `/v1/audio/languages`                | Supported Indic languages                         |
| `GET`  | `/v1/audio/files`                    | List saved audio and transcript files             |
| `GET`  | `/v1/audio/files/{file_id}`          | Play or retrieve saved file                       |
| `GET`  | `/v1/audio/files/{file_id}/download` | Download saved file                               |
| `GET`  | `/`                                  | Web UI Testing Interface                          |
