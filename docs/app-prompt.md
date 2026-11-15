Build a production-ready, OpenAI-compatible Universal AI Serving Platform and REST API server using FastAPI, accelerated for Apple Silicon (MLX and PyTorch MPS) with CPU fallback. The platform dynamically loads, serves, and manages open-source models directly from Hugging Face by their repository ID (`repo_id`). It supports all major model modalities:

1. **Chat & Text Generation (LLM)** (`/v1/chat/completions`, `/v1/completions`)
2. **Vision-Language & Multimodal (VLM)** (`/v1/chat/completions` with images)
3. **Text-to-Speech (TTS)** (`/v1/audio/speech`)
4. **Speech-to-Text / Automatic Speech Recognition (ASR)** (`/v1/audio/transcriptions`, `/v1/audio/translations`)
5. **Text Embeddings** (`/v1/embeddings`)
6. **Cross-Encoder Reranking** (`/v1/rerank`)
7. **Image Generation (Text-to-Image)** (`/v1/images/generations`)

The platform includes pre-configured first-class support for `iniquitous/indic-speak-mlx-bf16`, `bodhan-ai/indic-speak`, and `bodhan-ai/indic-transcribe-core`, alongside popular Hugging Face models (e.g. Qwen2.5, DeepSeek-R1 Distill, Llama-3.2, Whisper, Kokoro, BGE, FLUX/SD). The platform provides complete model lifecycle management (register, download, load into unified memory, unload, delete) via both an OpenAI-extended REST API and a self-contained, dark-themed Web Management Dashboard and Testing Playground (`static/index.html`), accompanied by an automated startup script (`start.sh`) and an automated test suite.

---

### 0. Delivery Phases & Cross-Cutting Design

The platform ships in phases. Every phase adds one engine module, registers its adapter factory with the Model Manager and adds a playground tab; the manager, registry, lifecycle API and dashboard shell built in phase 1 do not change.

| Phase | Scope | Engine module | Status |
| --- | --- | --- | --- |
| **1** | Text-to-Speech, Speech-to-Text, Model Manager, lifecycle API, dashboard (Models, TTS, ASR, Files, API Code tabs), `start.sh`, `test_api.py` | `tts_engine.py`, `asr_engine.py`, `model_manager.py`, `audio_utils.py`, `transliterate.py` | **Implemented** |
| 2 | Chat / completions (LLM), vision-language (VLM), SSE streaming, reasoning blocks, tool calling | `llm_engine.py` | Planned |
| 3 | Embeddings and cross-encoder reranking | `embedding_engine.py` | Planned |
| 4 | Image generation | `image_engine.py` | Planned |
| 5 | Hardening: API-key auth, request queue limits, Prometheus metrics, download cancel/resume | `server.py` | Planned |

Models of a later-phase modality can already be registered, inspected and downloaded in phase 1; loading them returns `501 modality_not_supported`.

#### 0.1 Engine adapter contract

Each engine module exposes `create_adapter(spec) -> adapter` and `catalog_for(spec) -> dict` (static metadata such as voices or languages, available without loading weights). An adapter implements `load(local_dir, spec, hf_token)`, `unload()` and its modality's inference methods. The registry entry's `adapter` field selects the class; it is auto-detected from the repository (`guess_adapter`) and can be overridden at registration.

| Modality | Adapter | Serves |
| --- | --- | --- |
| tts | `indic-speak-mlx` | `iniquitous/indic-speak-mlx-bf16` (MLX LM + SNAC/Vocos, chunk-level streaming) |
| tts | `indic-speak-torch` | `bodhan-ai/indic-speak` (PyTorch MPS / CUDA / CPU) |
| tts | `kokoro` | `hexgrad/Kokoro-82M` via the `kokoro` package |
| tts | `transformers-tts` | Any Hugging Face `text-to-speech` pipeline (MMS / VITS, Bark, ...) |
| asr | `indic-transcribe` | `bodhan-ai/indic-transcribe-core` |
| asr | `whisper-mlx` | `mlx-community/whisper-*` via `mlx-whisper` |
| asr | `transformers-asr` | Any Hugging Face ASR pipeline (Whisper on PyTorch, wav2vec2, ...) |

#### 0.2 Concurrency model

All loads, unloads and inference run on **one dedicated worker thread** owned by the Model Manager. MLX and MPS are not safe to drive from several threads, and one GPU gains nothing from parallel jobs. The asyncio loop stays free for status polling, downloads, file serving and audio decoding. Streaming synthesis runs on the worker and hands chunks to the response through an `asyncio.Queue`; a client disconnect sets a cancel flag that stops generation.

#### 0.3 Model resolution, defaults and OpenAI aliases

A request's `model` is resolved in this order: registry `id`, per-model `aliases`, global alias table, `repo_id`. The global alias table maps OpenAI names to the **current default model of a modality** (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` -> default TTS; `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` -> default ASR). "Set as Default" in the dashboard (`POST /v1/models/{id}/default`) therefore retargets every unmodified OpenAI client. An omitted `model` also uses the default. Initial defaults: `indic-speak-mlx` on Apple Silicon (`indic-speak-torch` elsewhere) and `indic-transcribe`.

#### 0.4 Lifecycle state machine

`registered` -> `downloading` -> `cached` -> `loaded` -> (`unload` or eviction) -> `cached`. A model is `cached` only when every repository it needs (`repo_id` plus `extra_repos`, e.g. the SNAC codec) is on disk with no partial blobs. With `auto_download` on (default), the first inference or load request for a `registered` model starts or joins a download task and waits for it.

Before a load, the manager evicts least-recently-used models until both limits hold: at most `max_loaded_models` resident (default 3) and at least `min_free_memory_gb` free after the incoming model's estimated footprint (default 4 GB). `idle_unload_minutes` (default 0 = off) unloads models that have not served a request recently. Unloading drops the adapter, runs `gc.collect()`, `mx.clear_cache()` and `torch.mps.empty_cache()`.

#### 0.5 Errors, limits and security

- Every error uses the OpenAI envelope `{"error": {"message", "type", "param", "code"}}`; validation errors are `400`, unknown models `404 model_not_found`, wrong modality `400 model_modality_mismatch`, registry conflicts `409`, oversized uploads `413`.
- Limits: `MAX_INPUT_CHARS` (default 20000) for TTS input, `MAX_UPLOAD_MB` (default 500) for audio uploads.
- File ids are validated against `^[A-Za-z0-9_-]{1,80}$` and matched by exact stem, so they cannot escape the data directories.
- `HF_TOKEN` is read from the environment / `.env`; a token entered in the dashboard is held in memory only and never written to disk or returned by the API.
- There is no authentication before phase 5. Bind to `127.0.0.1` (`./start.sh --host 127.0.0.1`) on untrusted networks, since the lifecycle endpoints can download and delete model weights.

#### 0.6 Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Bind address |
| `HF_TOKEN` | unset | Gated / private repositories |
| `PRELOAD_MODELS` | unset | Comma-separated model ids to load at startup; `defaults` = default TTS + ASR |
| `MODELS_CONFIG` | `config/models.json` | Registry file |
| `DATA_DIR` | `data` | Generated audio, uploads, transcripts |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | Weights cache (shared with other Hugging Face tools) |
| `MAX_INPUT_CHARS`, `MAX_UPLOAD_MB` | `20000`, `500` | Request limits |
| `LOG_LEVEL` | `INFO` | Logging |

---

### 1. Core Architecture & Dynamic Multi-Model Engines

The server architecture is modular, decoupled, and orchestrated by a central **Model Manager** (`model_manager.py`). Models are loaded into Apple Silicon unified memory on demand (the default) or preloaded at startup (`--preload`), with dynamic memory management that frees VRAM/RAM (`mx.clear_cache()` and `torch.mps.empty_cache()`) when swapping or unloading models.

#### 1.1 Model Registry & Lifecycle Manager (`model_manager.py` & `config/models.json`)

- **Model Registry Schema**:
  - `id`: Unique OpenAI model identifier/alias (e.g. `qwen2.5-7b`, `deepseek-r1-7b`, `whisper-large-v3`, `indic-speak-mlx`, `bge-small-en`, `flux-schnell`).
  - `repo_id`: Hugging Face repository ID (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`, `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`, `mlx-community/whisper-large-v3-mlx`, `hexgrad/Kokoro-82M`, `BAAI/bge-small-en-v1.5`, `iniquitous/indic-speak-mlx-bf16`, `bodhan-ai/indic-transcribe-core`).
  - `modality`: One of `llm` (Chat/Text), `vlm` (Vision-Language), `tts` (Text-to-Speech), `asr` (Speech-to-Text), `embedding` (Text Embeddings), `rerank` (Cross-Encoder Reranker), or `image_gen` (Diffusion/Image).
  - `backend`: `mlx` (preferred for Apple Silicon), `transformers` (PyTorch MPS/CPU), `diffusers`, or `auto`.
  - `quantization` / `dtype`: `4bit`, `8bit`, `bf16`, `fp16`, `fp32`, or `default`.
  - `context_length`: Maximum sequence length (tokens).
  - `chat_template`: Jinja2 template or tokenizer default.
  - `default_params`: Modality-specific defaults (e.g. temperature, max_tokens, speed, voice, language, steps).
  - `status`: `registered` (configured), `downloading`, `cached` (downloaded to disk, not in memory) or `loaded` (active in memory). Derived at request time, never stored.
  - `adapter`: Engine adapter that serves the model (section 0.1); auto-detected, overridable.
  - `aliases`: Extra names accepted in the `model` field.
  - `extra_repos`: Additional repositories the model needs (e.g. `hubertsiuzdak/snac_24khz`), downloaded and sized with it.
  - `ignore_patterns`: Files to skip when downloading (e.g. `nemo/*` halves the Indic-Transcribe download).
  - `description`, `remote_size_bytes`, `created`.
- **Registry file layout** (`config/models.json`, written atomically): `{"version", "defaults": {"tts", "asr"}, "aliases": {"tts-1": "@tts", ...}, "settings": {"max_loaded_models", "min_free_memory_gb", "idle_unload_minutes", "auto_download"}, "models": [...]}`. A corrupt file is moved aside and the defaults are re-seeded.
- **Hugging Face Hub Integration**:
  - Query HF Hub metadata (`pipeline_tag`, `model_type`, `architectures`, file tree, and size) via `huggingface_hub` to auto-detect modality, default backend, and required dependencies.
  - Background downloads with real-time progress (percent, bytes, speed, ETA) using `huggingface_hub.snapshot_download`; bytes are counted through the hub's own progress hooks, which stay accurate with the Xet transfer backend.
  - Support for private/gated repositories using `HF_TOKEN` from environment or UI settings.
  - Local cache management: Inspect disk size (`~/.cache/huggingface/hub` or `data/cache`) and purge uninstalled models.
- **Apple Silicon Memory Management**:
  - Live tracking of Apple Silicon unified memory (Total, Active, Wired, Free, MLX Cache limit).
  - Load-on-demand: Automatically load the requested model if not currently resident in memory.
  - Configurable memory limits and automatic LRU eviction of idle models to avoid macOS out-of-memory kernel panics or swap thrashing.
- **Pre-Configured Out-of-the-Box Models**:
  - **Chat/LLM**: `mlx-community/Qwen2.5-7B-Instruct-4bit` (alias `qwen2.5-7b`), `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` (alias `deepseek-r1-7b`), `mlx-community/Llama-3.2-3B-Instruct-4bit` (alias `llama-3.2-3b`).
  - **Vision-Language (VLM)**: `mlx-community/Qwen2-VL-7B-Instruct-4bit` (alias `qwen2-vl-7b`).
  - **TTS**: `iniquitous/indic-speak-mlx-bf16` (id `indic-speak-mlx`, default), `bodhan-ai/indic-speak` (id `indic-speak-torch`), `hexgrad/Kokoro-82M` (id `kokoro-82m`). `tts-1` / `tts-1-hd` follow the default TTS model.
  - **ASR**: `bodhan-ai/indic-transcribe-core` (id `indic-transcribe`, default), `mlx-community/whisper-large-v3-mlx` (id `whisper-large-v3`). `whisper-1` follows the default ASR model.
  - **Recommended one-click extras**: `openai/whisper-large-v3-turbo` (PyTorch), `mlx-community/whisper-tiny`, `facebook/mms-tts-hin`.
  - **Embeddings**: `BAAI/bge-small-en-v1.5` (alias `text-embedding-3-small`, `bge-small`).
  - **Reranker**: `BAAI/bge-reranker-base` (alias `bge-reranker`).
  - **Image Generation**: `black-forest-labs/FLUX.1-schnell` via `mflux` or `stabilityai/stable-diffusion-xl-base-1.0` via `diffusers` (MPS).

---

#### 1.2 Chat & Vision-Language Engine (`llm_engine.py`)

- **Backends**:
  - `mlx-lm` for Apple Silicon-optimized quantized LLMs (4-bit / 8-bit / BF16).
  - Hugging Face `transformers` with PyTorch `mps`/`cpu` for generic Hugging Face CausalLM models.
  - `mlx-vlm` or `transformers` for multimodal Vision-Language models.
- **Capabilities**:
  - **OpenAI Parity**: Implements `/v1/chat/completions` and `/v1/completions`.
  - **Streaming SSE**: Full Server-Sent Events streaming (`stream=true`) yielding `chat.completion.chunk` tokens in real-time.
  - **Reasoning Models**: Native handling of thinking/reasoning models (DeepSeek-R1, Qwen-Thinking) parsing `<think>...</think>` tags cleanly for UI presentation and API delivery.
  - **Vision / Multimodal**: Ingests image URLs and base64-encoded images in `messages[].content` (`{"type": "image_url", "image_url": {"url": "..."}}`) for VLM models.
  - **Tool & Function Calling**: Supports `tools` and `tool_choice` schemas, formatting function definitions into prompt context and parsing structured JSON function call arguments.
  - **Jinja2 Chat Templates**: Automatically loads and applies the model's official tokenizer chat template (`tokenizer.apply_chat_template`), respecting system instructions, multi-turn history, and roles (`system`, `user`, `assistant`, `tool`).
  - **Generation Parameters**: `temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `presence_penalty`, `stop` sequences.
  - **Token Metrics**: Returns accurate token usage counters (`prompt_tokens`, `completion_tokens`, `total_tokens`) and generation speed (tokens per second).

---

#### 1.3 Text-to-Speech (TTS) Engine (`tts_engine.py`)

- **Backends**:
  - `indic-speak`: `iniquitous/indic-speak-mlx-bf16` (MLX SNAC/Vocos) and `bodhan-ai/indic-speak` (PyTorch MPS/CPU).
  - Generic Hugging Face TTS adapter: Support for models such as `hexgrad/Kokoro-82M`, Bark, and ChatTTS.
- **Catalog & Voices**:
  - First-party catalog of 44 voices across 22 Indic languages and English.
  - OpenAI voice alias mapping: `alloy` -> `Amit`, `echo` -> `Arun`, `fable` -> `Deepika`, `onyx` -> `Vamsi`, `nova` -> `Kavya`, `shimmer` -> `Sravani`.
  - Delivery style tags: `NEUTRAL`, `HAPPY`, `SAD`, `ANGER`, `FEAR`, `SURPRISE`, `DISGUST`, `news`, `AIR style news`, `TV style news`, `advertisements`, `children's stories`, `educational lecture`, `Customer Care`, `single person narration audiobook`. Names are matched case-insensitively and mapped to the tag spelling the checkpoint was trained on (`voices.md`): `NEUTRAL` sends no style block, emotions are lower-cased.
  - Unknown voices fall back to the model's default voice; the voice actually used is returned in `X-Voice`.
- **Synthesis & Streaming**:
  - Sentence-chunked synthesis (`speak_long` behaviour: per-sentence generation with a duration-based token cap, silence trimming, 0.25 s gaps) to prevent truncation on multi-sentence text. Sentences over ~220 characters are split again at clause punctuation, then at spaces, since one generation is capped near 25 s of audio.
  - Speed adjustment (0.25x to 4.0x) by pitch-preserving time-stretch (ffmpeg `atempo`), falling back to resampling when ffmpeg is absent. Models with native speed control (Kokoro) use it directly.
  - Optional `temperature` and `seed` request fields for reproducible sampling.
  - Formats: `mp3`, `wav` (24 kHz mono), `flac`, and `pcm` (raw 16-bit signed integer LE).
  - Delivery modes:
    - Non-streaming: Full synthesis saved to `data/generated/{file_id}.{format}` with response headers `X-File-Id`, `X-File-Url`, and metadata JSON.
    - Streaming (`stream=true`): Chunked transfer encoding for `wav` (indefinite 44-byte header, `0xFFFFFFFF`), `pcm` and `mp3` (one continuous ffmpeg encoder, so chunk seams carry no encoder padding); other formats return `400`. Audio is accumulated and persisted under the `X-File-Id` announced in the response headers. At speeds other than 1.0 the stream is time-stretched per sentence.
    - All output is 24 kHz mono; adapters with another native rate are resampled.

---

#### 1.4 Automatic Speech Recognition (ASR) Engine (`asr_engine.py`)

- **Backends**:
  - `mlx-whisper` / Hugging Face Whisper: Ultra-fast Apple Silicon transcription for `openai/whisper-*` and community MLX checkpoints.
  - `bodhan-ai/indic-transcribe-core`: Indic Canary architecture on MPS/CUDA/CPU for 26 Indic languages + English (27 trained languages; the catalog adds `auto`).
  - Generic Hugging Face ASR pipelines (`transformers-asr`), with Whisper language identification and translation.
- **Capabilities**:
  - Ingestion: WAV, MP3, M4A, OGG, WEBM, FLAC via SoundFile with FFmpeg piping fallback, resampled to 16 kHz mono float32.
  - Automatic language detection (`language=None` or `"auto"`) and ISO-639-1 language routing.
  - Long-form audio processing: Automatic pause/silence segmentation (`split_on_silence`) for audio > 30s to prevent repetition collapse or hallucinations; each piece is decoded with the repository's repetition guard. The optional `chunk_seconds` form field (5-30, default 30) lowers the maximum segment length for finer subtitle timestamps at a small accuracy cost. Language is identified once per file, not per segment.
  - Output formats: `json` (`{"text": "..."}`), `text` (plain string), `verbose_json` (timestamps and segment breakdown `[{"id": 0, "start": 0.0, "end": 2.5, "text": "..."}]`), `srt` (SubRip subtitles), and `vtt` (WebVTT).
  - Script modes (Indic models): `native` (default), `mixed`/`itn` and `romanised`. The published `indic-transcribe-core` checkpoint emits native script only, so `romanised` is completed after decoding by rule-based ISO 15919 transliteration (`transliterate.py`; Brahmic scripts, Perso-Arabic text passes through unchanged) and `mixed` currently returns native script. The mode actually delivered is reported in the `X-Script-Mode` response header.
  - Translation (`/v1/audio/translations`): Whisper models translate to English; Indic-Transcribe returns the romanised transcript.
  - Persistence: Raw uploads stored as `data/uploads/{file_id}.{ext}` and transcript records in `data/transcriptions/{file_id}.json`, sharing one id so history can replay the audio next to its transcript.

---

#### 1.5 Text Embeddings & Cross-Encoder Reranking Engine (`embedding_engine.py`)

- **Backends**: Hugging Face `sentence-transformers` / `transformers` and MLX embedding architectures.
- **Embeddings Endpoint (`/v1/embeddings`)**:
  - Accepts single string or array of strings (`input`).
  - Batching support, L2 vector normalization, customizable pooling (`mean`, `cls`).
  - Returns OpenAI-compatible vector array with token usage statistics.
- **Cross-Encoder Reranking Endpoint (`/v1/rerank`)**:
  - Accepts `query` and a list of `documents` (strings or objects).
  - Computes cross-encoder relevance scores (using models like `bge-reranker-large`).
  - Returns sorted list with document indices, scores, and optional text payload.

---

#### 1.6 Image Generation Engine (`image_engine.py`)

- **Backends**: `mflux` (native MLX acceleration for FLUX.1 schnell/dev on Apple Silicon) or PyTorch `diffusers` (MPS accelerated for Stable Diffusion / SDXL).
- **Endpoint (`/v1/images/generations`)**:
  - Accepts `prompt`, `model`, `n` (number of images), `size` (e.g. `1024x1024`, `512x512`), `response_format` (`url` or `b64_json`), and optional `negative_prompt` or `num_inference_steps`.
  - Saves generated images to `data/generated/images/{file_id}.png` and returns OpenAI-standard response.

---

### 2. OpenAI-Compatible & Model Management REST API (`server.py`)

Implement a FastAPI application with lifespan model preloading, CORS middleware, background task workers, and the following REST endpoints:

#### OpenAI Standard Inference Endpoints

- `POST /v1/chat/completions`:
  - Request body (JSON): `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stream`, `stop`, `tools`, `tool_choice`, `presence_penalty`, `frequency_penalty`. Supports image URLs / base64 in messages for VLMs.
  - Returns non-streaming JSON or SSE streaming `data: {"choices": [{"delta": {"content": "..."}}]}\n\n`.
- `POST /v1/completions`: OpenAI legacy text completion endpoint.
- `POST /v1/embeddings`:
  - Request body (JSON): `model`, `input` (string or array of strings), `encoding_format` (`float`).
  - Returns vector array: `{"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}], "usage": {...}}`.
- `POST /v1/rerank`:
  - Request body (JSON): `model`, `query`, `documents` (list of strings), `top_n` (optional int).
  - Returns ranked list: `{"results": [{"index": 0, "relevance_score": 0.94, "document": "..."}], "model": "..."}`.
- `POST /v1/images/generations`:
  - Request body (JSON): `model`, `prompt`, `size`, `n`, `response_format`, `steps`.
  - Returns image data URLs or base64 JSON payload.
- `POST /v1/audio/speech`:
  - Request body (JSON): `model`, `input`, `voice`, `response_format` (`mp3`/`wav`/`flac`/`pcm`), `speed` (0.25 to 4.0), `style` (default `NEUTRAL`), `stream` (bool), `save` (bool).
  - Optional: `temperature`, `seed`. Omitting `model` uses the default TTS model.
  - Returns binary audio stream with `Content-Type`, `X-File-Id`, `X-File-Url`, and `Content-Disposition`, plus `X-Model`, `X-Voice`, `X-Audio-Duration`, `X-Generation-Time` and `X-RTF`.
- `POST /v1/audio/transcriptions`:
  - Multipart form: `file` (UploadFile), `model`, `language`, `prompt`, `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `mode` (`native`, `mixed`, `romanised`), `temperature`, `save`, `chunk_seconds`.
  - Returns transcript in the requested format, with `X-Model`, `X-Language`, `X-Script-Mode`, `X-Generation-Time` and (when saved) `X-File-Id` / `X-File-Url`.
- `POST /v1/audio/translations`:
  - Multipart form upload, transcribes and outputs translated/romanised text.
- `GET /v1/models`: Standard OpenAI format listing all registered models with extended fields: `modality`, `backend`, `repo_id`, `status` (`loaded`/`cached`/`unloaded`), and `size_mb`.
- `GET /v1/models/{model_id}`: Retrieve metadata and status for a specific model.

#### Dynamic Model Management Endpoints

- `POST /v1/models/register`:
  - Request body (JSON): `id` (alias), `repo_id` (HF ID), `modality` (`llm`, `vlm`, `tts`, `asr`, `embedding`, `rerank`, `image_gen`), `backend` (`mlx`, `transformers`, `diffusers`), `quantization`, `context_length`, `default_params`.
  - Registers the model in `config/models.json` dynamically without server restart.
- `POST /v1/models/{model_id}/download`:
  - Triggers asynchronous background download of weights from Hugging Face Hub.
  - Returns `{"task_id": "...", "status": "downloading"}`.
- `GET /v1/models/tasks/{task_id}`:
  - Returns download or loading progress percentage, speed, and status (`in_progress`, `completed`, `failed`).
- `POST /v1/models/{model_id}/load`:
  - Loads weights into memory (Apple Silicon GPU unified memory via MLX or MPS).
- `POST /v1/models/{model_id}/unload`:
  - Frees memory occupied by the model and invokes memory cache-clearing routines.
- `DELETE /v1/models/{model_id}`:
  - Query param `delete_files=true|false`. De-registers model and optionally removes downloaded weights from disk.
- `GET /v1/hf/inspect`:
  - Query param `repo_id`. Queries Hugging Face API to retrieve model card tags, pipeline type, estimated size, and auto-suggests optimal modality, backend, and quantization.
- `GET /v1/system/status`:
  - Reports Apple Silicon unified memory (Total, Used, Free), MLX cache, MPS allocation, loaded models, disk cache usage, and device capability.
- `GET /v1/models/recommended`: Tested model packs for one-click registration, flagged when already registered.
- `GET /v1/models/tasks`: All download / load tasks, newest first.
- `POST /v1/models/{model_id}/default`: Make the model the default for its modality (retargets `tts-1` / `whisper-1`).
- `GET` / `POST /v1/system/settings`: `max_loaded_models`, `min_free_memory_gb`, `idle_unload_minutes`, `auto_download`, and a write-only `hf_token`.

#### Auxiliary & File Endpoints

- `GET /v1/audio/voices`: Catalog of voices grouped by model, language, gender, and OpenAI aliases.
- `GET /v1/audio/languages`: Supported ASR languages and script modes.
- `GET /v1/audio/files`: List saved TTS audio and ASR transcripts.
- `GET /v1/audio/files/{file_id}`: Stream saved TTS audio, or return a transcript record (`?audio=true` returns the transcript's uploaded audio).
- `GET /v1/audio/files/{file_id}/download`: Download saved file with attachment disposition (`?format=json|txt|srt|vtt` for transcripts).
- `DELETE /v1/audio/files/{file_id}`: Delete file and its metadata.
- `GET /` and `GET /ui`: Serve the management dashboard and testing web interface (`static/index.html`).
- `GET /health`: Health status endpoint.

---

### 3. Web Management Dashboard & Multi-Modality Playground (`static/index.html`)

Create a self-contained, responsive, dark-themed single-page web UI (no mandatory external CDN dependencies) accessible at `http://localhost:8000/`:

**Phasing:** phase 1 ships Tab 1 (Model Manager), Tab 3 (TTS), Tab 4 (ASR), Tab 7 (Saved Files) and Tab 8 (API Code). Tab 2 arrives with phase 2, Tab 5 with phase 3 and Tab 6 with phase 4. Phase 1 additions: a memory and download settings panel on Tab 1, live streamed playback through Web Audio on Tab 3 (with time-to-first-audio), and a task selector (transcribe / translate) and segment-length selector on Tab 4.

#### Tab 1: Model Manager & Hugging Face Hub Explorer (Central Management)

- **System Resource Monitor Bar**:
  - Live visual meters for Unified Memory / RAM (Total, Used, Free), MLX cache memory, Hugging Face disk cache size, and count of actively loaded models.
- **Quick-Add Recommended Model Packs**:
  - One-click cards to register tested popular models:
    - _LLM / Chat_: `Qwen2.5-7B-Instruct-4bit`, `DeepSeek-R1-Distill-Qwen-7B-4bit`, `Llama-3.2-3B-Instruct-4bit`
    - _VLM_: `Qwen2-VL-7B-Instruct-4bit`
    - _ASR_: `whisper-large-v3-mlx`, `indic-transcribe-core`
    - _TTS_: `indic-speak-mlx-bf16`, `indic-speak-torch`, `kokoro-82m`
    - _Embeddings & Rerank_: `bge-small-en-v1.5`, `bge-reranker-base`
    - _Image Gen_: `flux-schnell` / `stable-diffusion-xl`
- **Add Model from Hugging Face**:
  - Input field for Hugging Face Repo ID (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`, `hexgrad/Kokoro-82M`).
  - "Inspect Hugging Face" button: auto-queries `/v1/hf/inspect` to detect modality (`LLM`, `VLM`, `TTS`, `ASR`, `Embedding`, `Rerank`, `Image Gen`), file size, and recommended backend.
  - Configurable form: Model Alias (`id`), Modality selector, Backend (`MLX` vs `PyTorch MPS/CPU`), Quantization (`4bit`, `8bit`, `bf16`, `fp16`), and optional default parameters.
  - "Register & Download" button with real-time download progress bar and status badge.
- **Model Registry Table & Action Cards**:
  - Displays all configured models with badges for Modality, Backend, Disk Size, and Memory State (`Loaded` [green], `Cached` [blue], `Downloading` [yellow], `Unloaded` [gray]).
  - Interactive actions: **Load to Memory**, **Unload**, **Download**, **Delete**, **Set as Default**, and **Open in Playground**.

#### Tab 2: Chat & LLM / VLM Playground (`/v1/chat/completions`)

- Model selector dropdown (lists all active LLM and VLM models).
- Parameter panel: System prompt textarea, Temperature slider (0.0 - 2.0), Top_p slider, Max Tokens slider, and Streaming toggle.
- Conversational chat UI:
  - Multi-turn message history with distinct user and assistant speech bubbles.
  - Multimodal input: Drag-and-drop image upload or URL input for Vision models.
  - Streaming token output animation with smooth auto-scrolling.
  - Collapsible `<think>` reasoning block for DeepSeek-R1 / thinking models.
  - Formatted markdown, syntax-highlighted code blocks, and "Copy Code" buttons.
  - Performance telemetry: Tokens per second (tok/s), time-to-first-token (TTFT), and total tokens.

#### Tab 3: Text-to-Speech (TTS) Playground (`/v1/audio/speech`)

- Model selector (Indic-Speak MLX, Indic-Speak PyTorch, Kokoro, etc.).
- Dynamic voice selector: Populated based on selected model (44 Indic voices with gender and language grouping for Indic-Speak; Kokoro voice options when selected; 6 OpenAI aliases).
- Multilingual text prompt presets (Hindi, Telugu, Tamil, Bengali, Sanskrit, English) and custom textarea.
- Emotion / delivery style selector, audio format selector (`mp3`, `wav`, `flac`, `pcm`), speed slider (0.25x - 4.0x).
- Action buttons: "Generate & Download" and "Stream Directly".
- Waveform audio player, duration / RTF (Real-Time Factor) stats, and direct download button.
- Live cURL and Python OpenAI SDK command preview.

#### Tab 4: Speech-to-Text (ASR) Playground (`/v1/audio/transcriptions`)

- Model selector (Whisper MLX, Indic-Transcribe, Whisper PyTorch).
- Audio input: In-browser microphone recorder with live pulse animation/timer, and drag-and-drop audio file upload zone (WAV, MP3, M4A, OGG, WEBM, FLAC).
- Controls: Language dropdown (Auto-detect + 27 Indic languages + English), Script mode selector (`native`, `mixed`/`itn`, `romanised`), Response format selector (`verbose_json`, `json`, `text`, `srt`, `vtt`).
- Result card: Detected language badge, transcript text view, interactive timestamped segments with click-to-seek audio playback, and export buttons (`.txt`, `.srt`, `.json`).

#### Tab 5: Text Embeddings & Reranker Playground (`/v1/embeddings` & `/v1/rerank`)

- Model selector (lists all loaded embedding and reranker models).
- Sub-view A (Semantic Similarity): Multi-input text fields to test semantic similarity with a live cosine similarity score matrix.
- Sub-view B (Document Reranker): Query input field and multi-line candidate documents input; calculates cross-encoder relevance scores and outputs sorted ranking.
- Vector inspector: Display vector dimension, L2 norm, and raw float array preview.

#### Tab 6: Image Generation Playground (`/v1/images/generations`)

- Model selector (FLUX.1 schnell MLX, SDXL, etc.).
- Prompt and negative prompt text fields.
- Aspect ratio / Resolution selector (`1024x1024`, `1024x768`, `768x1024`, `512x512`), Steps slider.
- "Generate Image" button with loading spinner.
- Image viewer with full-resolution zoom, generation metadata, and direct download button.

#### Tab 7: Saved Files & History

- Filterable and searchable history table of all generated TTS audio files, ASR transcripts, and generated images.
- Columns: File ID, Modality (TTS/ASR/Image), Model, Timestamp, Duration/Size, Actions.
- Inline audio player, view transcript modal, image lightbox, direct download button, and delete action.

#### Tab 8: OpenAI SDK Integration Code

- Dynamically generated, copy-pasteable code snippets for Python (`openai` library), TypeScript/JavaScript, and `curl`, updating in real time to match the currently selected model and parameters in each playground tab.

---

### 4. Automated Startup Script (`start.sh`)

Write a POSIX-compliant, executable `start.sh` script that:

1. Detects Python environment; creates a `.venv` virtual environment using `python3.12` or `python3` if not already present.
2. Activates `.venv`.
3. Checks and installs/upgrades dependencies from `requirements.txt` (re-installed only when the file changes; `--upgrade` forces a refresh, `--skip-install` skips the check). Phase 1 installs the Core API, PyTorch & Hugging Face, Audio and MLX (`mlx`, `mlx-lm`, `mlx-whisper`) groups plus `psutil`, `kokoro` and `indic-transliteration`; `mlx-vlm`, `sentence-transformers`, `diffusers` and `mflux` are added with their phases:
   - Core API: `fastapi>=0.110.0`, `uvicorn[standard]>=0.28.0`, `pydantic>=2.0.0`, `python-multipart>=0.0.9`, `requests>=2.31.0`, `python-dotenv`.
   - Apple Silicon & MLX: `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-whisper`, `mflux` (on macOS Apple Silicon).
   - PyTorch & Hugging Face: `torch`, `torchaudio`, `transformers>=4.40.0`, `accelerate`, `sentencepiece`, `huggingface_hub`, `sentence-transformers`, `diffusers`.
   - Audio processing: `soundfile`, `snac`, `numpy`, `scipy`.
4. Initializes project directory layout:
   - `config/` (seeds default `models.json` if not present)
   - `data/cache/` (HF weights cache)
   - `data/generated/` (TTS audio files)
   - `data/generated/images/` (Generated images)
   - `data/transcriptions/` (ASR transcripts)
   - `data/uploads/` (uploaded audio files)
   - `static/` (Web UI)
5. Detects Apple Silicon hardware (M1/M2/M3/M4, unified memory size) and displays a terminal banner with URLs for the Web Management UI, OpenAPI Docs, and endpoint summary.
6. Supports CLI arguments (`--host`, `--port`, `--preload <model_id>`, `--hf-token <token>`).
7. Executes `uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} "$@"`.

---

### 5. Automated Verification Suite (`test_api.py`)

Write a comprehensive Python test script using `requests` and `soundfile` that verifies:

**Phasing:** phase 1 covers groups 1, 2, 4, 5 and 8 (groups 3, 6 and 7 arrive with phases 2-4). The lifecycle group registers a 75 MB Whisper checkpoint, downloads, loads, transcribes, unloads and de-registers it. The suite also checks error envelopes, speed, every audio format, long-form segmentation, romanised translation output and file-id validation, and deletes every file and model it creates.

1. **System & Health Checks**:
   - `GET /health` and `GET /v1/system/status` return 200 with system memory and device readiness.
   - `GET /v1/models` returns the catalog of pre-configured and registered models with modality attributes.
2. **Model Management Lifecycle**:
   - `POST /v1/models/register`: Register a new test model configuration (e.g. lightweight embedding or text model).
   - `GET /v1/models/{model_id}`: Verify model metadata is retrievable.
   - `POST /v1/models/{model_id}/load` and `POST /v1/models/{model_id}/unload`: Test memory loading and unloading cycle.
3. **LLM Chat Completions (`POST /v1/chat/completions`)**:
   - Non-streaming chat completion returns standard OpenAI JSON format with choices and token usage.
   - Streaming chat completion validates Server-Sent Events (SSE) `data: {"choices": ...}` streaming chunks and termination with `[DONE]`.
   - Reasoning model parsing (verifying `<think>` blocks are handled).
4. **Text-to-Speech (`POST /v1/audio/speech`)**:
   - Non-streaming synthesis generates valid audio (24 kHz), returns headers `X-File-Id` and `X-File-Url`, and verifies audio integrity via `/v1/audio/files/{file_id}/download`.
   - Streaming synthesis (`stream=true`) validates HTTP chunked transfer and WAV header/chunk reception.
   - Voice catalog check via `GET /v1/audio/voices` verifying all 44 voices and OpenAI aliases.
5. **Speech-to-Text (`POST /v1/audio/transcriptions`)**:
   - Transcription with auto-language detection.
   - Response format validation for `json`, `text`, `verbose_json` (segment timestamps), and `srt`.
6. **Text Embeddings & Reranking (`POST /v1/embeddings` & `POST /v1/rerank`)**:
   - Verifies batch embedding creation, dimensionality, and float vector response structure.
   - Verifies reranking response ordering and relevance scores.
7. **Image Generation (`POST /v1/images/generations`)**:
   - Verifies image generation endpoint returns valid URL or base64 data.
8. **File Management**:
   - `GET /v1/audio/files` lists generated files; `DELETE /v1/audio/files/{file_id}` successfully deletes files.
