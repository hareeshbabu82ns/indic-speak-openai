# AI Serving Platform: OpenAI-Compatible Speech API (Phase 1)

An OpenAI-compatible serving platform for Hugging Face models on Apple Silicon (MLX and PyTorch MPS, with CPU fallback). Phase 1 serves **Text-to-Speech** and **Speech-to-Text**, with full model lifecycle management (register, download, load, unload, delete) through a REST API and a web dashboard. The full plan and later phases are in [docs/app-prompt.md](docs/app-prompt.md).

## Quick start

```bash
./start.sh
```

`start.sh` creates `.venv`, installs `requirements.txt` when it changes, seeds `config/models.json` and starts the server.

- Dashboard: <http://localhost:8000/>
- OpenAPI docs: <http://localhost:8000/docs>

```bash
./start.sh --port 9000 --host 127.0.0.1 --preload defaults --hf-token hf_xxx
```

Models load on first request. `--preload defaults` loads the default TTS and ASR models at startup. `ffmpeg` is needed for M4A/WEBM uploads, MP3 streaming and pitch-preserving speed changes.

## Models

| id | Repository | Type | Backend |
| --- | --- | --- | --- |
| `indic-speak-mlx` (default TTS) | `iniquitous/indic-speak-mlx-bf16` | TTS | MLX |
| `indic-speak-torch` | `bodhan-ai/indic-speak` | TTS | PyTorch |
| `kokoro-82m` | `hexgrad/Kokoro-82M` | TTS | PyTorch |
| `indic-transcribe` (default ASR) | `bodhan-ai/indic-transcribe-core` | ASR | PyTorch |
| `whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` | ASR | MLX |

`tts-1`, `tts-1-hd` and `whisper-1` point at the default model of their type; change it with "Set default" in the dashboard. Any other Hugging Face TTS or ASR repository can be added from the Models tab or with `POST /v1/models/register`.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="tts-1", voice="Sravani", input="నమస్కారం", response_format="mp3",
) as response:
    response.stream_to_file("speech.mp3")

with open("speech.mp3", "rb") as audio:
    print(client.audio.transcriptions.create(model="whisper-1", file=audio).text)
```

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/speech` | TTS: `mp3`, `wav`, `flac`, `pcm`; `stream`, `speed`, `style`, `save` |
| `POST` | `/v1/audio/transcriptions` | ASR: `json`, `text`, `verbose_json`, `srt`, `vtt`; `language`, `mode` |
| `POST` | `/v1/audio/translations` | English (Whisper) or romanised (Indic-Transcribe) transcript |
| `GET` | `/v1/audio/voices`, `/v1/audio/languages` | Voice and language catalogs per model |
| `GET` `DELETE` | `/v1/audio/files`, `/v1/audio/files/{id}`, `/v1/audio/files/{id}/download` | Saved audio and transcripts |
| `GET` | `/v1/models`, `/v1/models/{id}` | Models with modality, backend, status, size |
| `POST` | `/v1/models/register` | Add a Hugging Face model to the registry |
| `POST` | `/v1/models/{id}/download`, `/load`, `/unload`, `/default` | Lifecycle actions |
| `GET` | `/v1/models/tasks/{task_id}` | Download progress |
| `DELETE` | `/v1/models/{id}?delete_files=true` | De-register, optionally delete weights |
| `GET` | `/v1/hf/inspect?repo_id=` | Detect modality, backend, size of a Hub repository |
| `GET` | `/v1/system/status`, `/v1/system/settings`, `/health` | Memory, loaded models, limits |

## Notes

- `indic-transcribe-core` emits native script only. `mode=romanised` and `/v1/audio/translations` return an ISO 15919 transliteration; `mode=mixed` returns native script. The delivered mode is in the `X-Script-Mode` header.
- At most 3 models stay in memory by default; the least recently used one is unloaded first. Change this under Models > Memory & download settings.
- There is no authentication. Use `--host 127.0.0.1` on shared networks.
- Recording from the microphone on another device needs HTTPS (browsers only expose the microphone on `https://` or `http://localhost`). Run `./start.sh --https` and open `https://<this-machine-ip>:8000/`; it creates a self-signed certificate in `data/certs/`, so accept the browser warning once per device and use `curl -k` / `verify=False` from API clients.

## Tests

```bash
python test_api.py
```

Runs against a live server; `--skip-lifecycle` skips the model download test.
