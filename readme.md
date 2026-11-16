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

Models load on first request. `--preload defaults` loads the default TTS and ASR models at startup; add `--preload indic-translate` for the translation model. `ffmpeg` is needed for M4A/WEBM uploads, MP3 streaming and pitch-preserving speed changes.

## Running in the background

`./start.sh` (or `./start.sh run`) runs in the foreground and stops with Ctrl+C. To keep the server running after you close the terminal, use the `start` command and manage it with the others:

```bash
./start.sh start --preload defaults     # start in the background (same options as a foreground run)
./start.sh status                       # PID, uptime, memory, URL, health, loaded models
./start.sh logs                         # last 100 log lines; -n 500 for more, -f to follow
./start.sh restart                      # stop, then start again with the previous options
./start.sh restart --port 9000          # restart with new options (these become the saved ones)
./start.sh stop                         # graceful stop; SIGKILL after 30 s (STOP_TIMEOUT=60 to change)
```

| Command | What it does |
| --- | --- |
| `run` (default) | Bootstraps and runs in the foreground |
| `start` | Bootstraps, then runs uvicorn detached and waits up to 60 s (`START_TIMEOUT`) for `/health`. If the server exits during startup, the last log lines are printed and the command fails |
| `stop` | Sends SIGTERM so uvicorn shuts down cleanly, and sends SIGKILL after `STOP_TIMEOUT` seconds |
| `restart [options]` | `stop` + `start`. With no options it reuses the ones from the last `start` |
| `status` | Exit code `0` when running and healthy, `1` when the process is up but not responding yet, `3` when stopped, so it can be used in scripts |
| `logs [-f] [-n N]` | Shows or follows `data/logs/server.log` |

`-d` / `--daemon` is a shorthand for `start` (`./start.sh -d --port 9000`). `start` and `run` refuse to launch while a background server is already running.

Files (all under the git-ignored `data/` directory):

- `data/logs/server.log`: server output. It is rotated to `server.log.1` at startup once it passes 10 MB.
- `data/run/server.pid`, `server.state`, `server.args`: PID, address and the options `restart` reuses.

The Hugging Face token is never written to disk, so `restart` does not remember `--hf-token`. For gated models, export `HF_TOKEN` in the shell before `start`/`restart`, run `hf auth login` once (the server picks up the stored login), or set the token in the dashboard after each start.

To start the server at login, add `cd /path/to/indic-speak && ./start.sh start --skip-install` to a login item or a launchd agent.

## Models

| id | Repository | Type | Backend |
| --- | --- | --- | --- |
| `indic-speak-mlx` (default TTS) | `iniquitous/indic-speak-mlx-bf16` | TTS | MLX |
| `indic-speak-torch` | `bodhan-ai/indic-speak` | TTS | PyTorch |
| `kokoro-82m` | `hexgrad/Kokoro-82M` | TTS | PyTorch |
| `indic-transcribe` (default ASR) | `bodhan-ai/indic-transcribe-core` | ASR | PyTorch |
| `whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` | ASR | MLX |
| `indic-translate` (default translation) | `bodhan-ai/indic-translate` | Translation | MLX |
| `indic-translate-torch` | `bodhan-ai/indic-translate` | Translation | PyTorch |

`tts-1`, `tts-1-hd` and `whisper-1` point at the default model of their type; change it with "Set default" in the dashboard. Any other Hugging Face TTS, ASR or translation repository can be added from the Models tab or with `POST /v1/models/register`. Recommended extras: `indic-translate-mlx-8bit` (same weights quantized to 8-bit at load, about half the memory) and `gemma-4-e4b-it-4bit` (the ungated 5 GB base model, a lower-quality stand-in for trying the API).

### Text translation (Indic Translate)

[`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate) is a Gemma 4 E4B fine-tune that translates between English and the 22 scheduled Indian languages in both directions, takes whole documents up to 32k tokens while keeping Markdown, LaTeX and code structure, and also does Romanized output, transliteration and code-mixed output. The repository is **gated**: accept the license at the link above with the Hugging Face account whose token the server uses (`HF_TOKEN` or the dashboard setting), then download about 16 GB. On Apple Silicon it runs on MLX straight from the original checkpoint (no conversion).

The model follows one fixed prompt format and loses quality silently when it drifts. The server always builds that prompt itself (`Translate the following text into Hindi:\n\n<text>`, no system turn, thinking off, source language never named), so clients only send the text and a target.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="indic-translate",
    messages=[{"role": "user", "content": "The committee approved the proposal after a long debate."}],
    extra_body={"target_language": "hi"},          # also: task="transliterate", script="roman" | "code-mixed"
)
print(resp.choices[0].message.content)
```

- `target_language` takes a code (`hi`, `te`, `mni_Beng`, `sd_Arab`), a FLORES code (`tam_Taml`) or a name (`Telugu`, `Hindi (Roman)`). See `GET /v1/translations/languages`.
- Instead of `target_language`, a user message already written as `Translate the following text into <Language>:\n\n<text>` is passed through unchanged. Any other message without a target is rejected with `missing_target_language`.
- Only the last user message is translated; system messages are ignored, because the model is single-turn.
- `stream=true` returns standard `chat.completion.chunk` SSE events (with `stream_options.include_usage`).
- Greedy decoding (`temperature` 0) is the default. When `max_tokens` is omitted it is sized from the input (2.5 times the input tokens, up to 8192), since Indic scripts need 1.5 to 2 times the tokens of English.

`POST /v1/translations` is a structured alternative: `{"input": "..." | [...], "target_language": "ta", "task": "translate", "script": "native"}` returns `{"object": "translation", "text", "data": [{"index", "text", "finish_reason"}], "usage"}`, and with `stream=true` it sends `translation.chunk` events with `delta` text per `index`.

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
| `POST` | `/v1/chat/completions` | OpenAI chat completions served by translation models; `target_language`, `task`, `script`, `stream` |
| `POST` | `/v1/translations` | Structured text translation / transliteration, single or batch, optional SSE streaming |
| `GET` | `/v1/translations/languages` | Translation target languages, prompt names, scripts and tasks |
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

Runs against a live server; `--skip-lifecycle` skips the model download test. The translation group runs inference when a translation model is on disk (`--translation-model ID` picks one); `--only translation,system` runs selected groups.
