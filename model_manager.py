"""Model registry and lifecycle manager.

Owns `config/models.json` (the registry), Hugging Face downloads with progress
tracking, loading/unloading of models into Apple Silicon unified memory, LRU
eviction and system/memory reporting.

Threading model: every load, unload and inference call runs on ONE dedicated
worker thread (`ModelManager.run`). MLX and MPS are not safe to drive from
several threads at once, and a single GPU gains nothing from parallel jobs, so
requests are serialised there while the asyncio loop stays free for status
polling, downloads and file serving.
"""
from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger("model_manager")

MODALITIES = ("llm", "vlm", "tts", "asr", "translation", "embedding", "rerank", "image_gen")
SUPPORTED_MODALITIES = ("tts", "asr", "translation")
BACKENDS = ("mlx", "transformers", "diffusers", "auto")
QUANTIZATIONS = ("4bit", "8bit", "bf16", "fp16", "fp32", "default")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
GB = 1024 ** 3
MB = 1024 ** 2

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

PIPELINE_TO_MODALITY = {
    "text-to-speech": "tts",
    "text-to-audio": "tts",
    "automatic-speech-recognition": "asr",
    "translation": "translation",
    "text-generation": "llm",
    "text2text-generation": "llm",
    "image-text-to-text": "vlm",
    "visual-question-answering": "vlm",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "text-ranking": "rerank",
    "text-to-image": "image_gen",
}

# Catalog of tested models. `default: True` entries are seeded into a fresh registry;
# the rest are offered as one-click "recommended" cards in the dashboard.
MODEL_CATALOG: List[dict] = [
    {
        "id": "indic-speak-mlx", "repo_id": "iniquitous/indic-speak-mlx-bf16", "modality": "tts",
        "backend": "mlx", "quantization": "bf16", "adapter": "indic-speak-mlx",
        "extra_repos": ["hubertsiuzdak/snac_24khz"], "default": True, "apple_silicon_only": True,
        "description": "Indic-Speak on MLX (Apple Silicon GPU). 44 voices, 22 Indic languages + English, real-time streaming.",
        "default_params": {"voice": "Amit", "style": "NEUTRAL", "temperature": 0.6, "speed": 1.0},
    },
    {
        "id": "indic-speak-torch", "repo_id": "bodhan-ai/indic-speak", "modality": "tts",
        "backend": "transformers", "quantization": "bf16", "adapter": "indic-speak-torch",
        "extra_repos": ["hubertsiuzdak/snac_24khz"], "ignore_patterns": ["*.png"], "default": True,
        "description": "Indic-Speak reference checkpoint on PyTorch (MPS / CUDA / CPU).",
        "default_params": {"voice": "Amit", "style": "NEUTRAL", "temperature": 0.6, "speed": 1.0},
    },
    {
        "id": "kokoro-82m", "repo_id": "hexgrad/Kokoro-82M", "modality": "tts",
        "backend": "transformers", "quantization": "fp32", "adapter": "kokoro", "default": True,
        "description": "Kokoro 82M: small, fast English/Hindi TTS (needs `pip install kokoro`).",
        "default_params": {"voice": "af_heart", "speed": 1.0},
    },
    {
        "id": "indic-transcribe", "repo_id": "bodhan-ai/indic-transcribe-core", "modality": "asr",
        "backend": "transformers", "quantization": "fp32", "adapter": "indic-transcribe",
        "ignore_patterns": ["nemo/*", "*.png"], "default": True,
        "description": "Indic Canary ASR: 26 Indic languages + English with language ID. Native script; romanised via transliteration.",
        "default_params": {"language": "auto", "mode": "native"},
    },
    {
        "id": "whisper-large-v3", "repo_id": "mlx-community/whisper-large-v3-mlx", "modality": "asr",
        "backend": "mlx", "quantization": "fp16", "adapter": "whisper-mlx", "default": True,
        "apple_silicon_only": True,
        "description": "OpenAI Whisper large-v3 on MLX: 99 languages, translation to English.",
        "default_params": {"language": "auto"},
    },
    {
        "id": "whisper-large-v3-turbo", "repo_id": "openai/whisper-large-v3-turbo", "modality": "asr",
        "backend": "transformers", "quantization": "fp16", "adapter": "transformers-asr",
        "description": "Whisper large-v3-turbo on PyTorch (MPS / CUDA / CPU).",
        "default_params": {"language": "auto"},
    },
    {
        "id": "whisper-tiny-mlx", "repo_id": "mlx-community/whisper-tiny", "modality": "asr",
        "backend": "mlx", "quantization": "fp16", "adapter": "whisper-mlx", "apple_silicon_only": True,
        "description": "Whisper tiny on MLX: 75 MB smoke-test model.",
        "default_params": {"language": "auto"},
    },
    {
        "id": "mms-tts-hin", "repo_id": "facebook/mms-tts-hin", "modality": "tts",
        "backend": "transformers", "quantization": "fp32", "adapter": "transformers-tts",
        "description": "Meta MMS VITS Hindi: 145 MB single-voice TTS via the generic transformers adapter.",
        "default_params": {},
    },
    {
        "id": "indic-translate", "repo_id": "bodhan-ai/indic-translate", "modality": "translation",
        "backend": "mlx", "quantization": "bf16", "adapter": "indic-translate-mlx", "default": True,
        "apple_silicon_only": True, "ignore_patterns": ["assets/*", "benchmarks/*", "*.png"],
        "description": "Bodhan Indic Translate (Gemma 4 E4B) on MLX: English <-> 22 Indian languages, documents up to 32k tokens, "
                       "Romanized, code-mixed and transliteration. Gated: accept the license on huggingface.co first.",
        "default_params": {"temperature": 0.0, "top_p": 1.0},
    },
    {
        "id": "indic-translate-torch", "repo_id": "bodhan-ai/indic-translate", "modality": "translation",
        "backend": "transformers", "quantization": "bf16", "adapter": "indic-translate-torch", "default": True,
        "ignore_patterns": ["assets/*", "benchmarks/*", "*.png"],
        "description": "Bodhan Indic Translate reference checkpoint on PyTorch (MPS / CUDA / CPU). Gated.",
        "default_params": {"temperature": 0.0, "top_p": 1.0},
    },
    {
        "id": "indic-translate-mlx-8bit", "repo_id": "bodhan-ai/indic-translate", "modality": "translation",
        "backend": "mlx", "quantization": "8bit", "adapter": "indic-translate-mlx", "apple_silicon_only": True,
        "ignore_patterns": ["assets/*", "benchmarks/*", "*.png"],
        "description": "Indic Translate quantized to 8-bit at load time on MLX: about half the memory of bf16. Gated.",
        "default_params": {"temperature": 0.0, "top_p": 1.0},
    },
    {
        "id": "gemma-4-e4b-it-4bit", "repo_id": "mlx-community/gemma-4-e4b-it-4bit", "modality": "translation",
        "backend": "mlx", "quantization": "4bit", "adapter": "indic-translate-mlx", "apple_silicon_only": True,
        "description": "Base Gemma 4 E4B instruct, 4-bit MLX (5 GB, ungated). The base model Indic Translate was tuned from; "
                       "a lower-quality stand-in served through the same prompt format.",
        "default_params": {"temperature": 0.0, "top_p": 1.0},
    },
]

# OpenAI model names -> "@<modality>" (the current default model of that modality).
DEFAULT_ALIASES = {
    "tts-1": "@tts", "tts-1-hd": "@tts", "gpt-4o-mini-tts": "@tts",
    "whisper-1": "@asr", "gpt-4o-transcribe": "@asr", "gpt-4o-mini-transcribe": "@asr",
}

DEFAULT_SETTINGS = {
    "max_loaded_models": 3,       # LRU-evict beyond this many resident models
    "min_free_memory_gb": 4.0,    # evict idle models until a load leaves this much free
    "idle_unload_minutes": 0,     # 0 = never unload idle models automatically
    "auto_download": True,        # download missing weights on first use
}


class ModelError(Exception):
    """Error carrying an HTTP status and an OpenAI-style error code."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error", param: Optional[str] = None):
        super().__init__(message)
        self.message, self.status, self.code, self.param = message, status, code, param


# ----------------------------------------------------------------------------
# Hugging Face helpers
# ----------------------------------------------------------------------------
def hf_cache_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def _repo_cache_folder(repo_id: str) -> Path:
    return hf_cache_dir() / ("models--" + repo_id.replace("/", "--"))


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _dir_bytes(Path(entry.path))
            except OSError:
                continue
    except OSError:
        pass
    return total


def _byte_counting_tqdm(counter: dict):
    """A tqdm class that adds the bytes snapshot_download writes to disk into `counter`.

    snapshot_download aggregates every file into a "Reconstructing" bar (unit B); bars may
    be disabled when stderr is not a terminal, so bytes are counted here, not read from tqdm.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _Bar(hf_tqdm):
        def __init__(self, *args, **kwargs):
            self._counts = kwargs.get("unit") == "B" and "ownloading" not in str(kwargs.get("desc", ""))
            if self._counts and kwargs.get("initial"):
                counter["bytes"] += int(kwargs["initial"])
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self._counts and n:
                counter["bytes"] += int(n)
            return super().update(n)

    return _Bar


def guess_quantization(repo_id: str) -> str:
    name = repo_id.lower()
    for pattern, quant in ((r"4[-_]?bit|q4", "4bit"), (r"8[-_]?bit|q8", "8bit"), (r"bf16", "bf16"),
                           (r"fp16|f16|half", "fp16"), (r"fp32|f32", "fp32")):
        if re.search(pattern, name):
            return quant
    return "default"


def guess_adapter(repo_id: str, modality: str, backend: str) -> Optional[str]:
    name = repo_id.lower()
    if modality == "tts":
        if "indic-speak" in name:
            return "indic-speak-mlx" if backend == "mlx" else "indic-speak-torch"
        if "kokoro" in name:
            return "kokoro"
        return "transformers-tts" if backend != "mlx" else None
    if modality == "asr":
        if "indic-transcribe" in name:
            return "indic-transcribe"
        if backend == "mlx":
            return "whisper-mlx" if "whisper" in name else None
        return "transformers-asr"
    if modality == "translation":
        return "indic-translate-mlx" if backend == "mlx" else "indic-translate-torch"
    return None


def slugify_model_id(repo_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.split("/")[-1]).strip("-.").lower()
    return slug[:64] or "model"


# ----------------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------------
class ModelManager:
    def __init__(self, config_path: str = "config/models.json"):
        self.config_path = Path(config_path)
        self.hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
        self._lock = threading.RLock()
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._factories: Dict[str, Callable[[dict], Any]] = {}
        self._loaded: Dict[str, dict] = {}       # model_id -> {adapter, loaded_at, last_used, memory_bytes}
        self._tasks: Dict[str, dict] = {}
        self._download_futures: Dict[str, threading.Event] = {}
        self._cache_index: Dict[str, dict] = {}
        self._cache_index_at = 0.0
        self._cache_total = 0
        self.models: Dict[str, dict] = {}
        self.defaults: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._load_config()

    # -- registry persistence ------------------------------------------------
    def _load_config(self) -> None:
        data = None
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as e:
                backup = self.config_path.with_suffix(".json.corrupt")
                self.config_path.replace(backup)
                logger.error("Could not parse %s (%s); moved to %s and re-seeding defaults", self.config_path, e, backup)
        if data is None:
            data = self._seed_config()
        self.models = {m["id"]: m for m in data.get("models", [])}
        self.defaults = dict(data.get("defaults", {}))
        # Registries written before a modality was supported get that modality's default models once.
        seeded = data.get("seeded_modalities") or ["tts", "asr"]
        added = False
        for modality in SUPPORTED_MODALITIES:
            if modality in seeded:
                continue
            for entry in MODEL_CATALOG:
                if (entry["modality"] == modality and entry.get("default") and entry["id"] not in self.models
                        and not (entry.get("apple_silicon_only") and not IS_APPLE_SILICON)):
                    self.models[entry["id"]] = self._clean_entry(entry)
                    added = True
        self.seeded_modalities = sorted(set(seeded) | set(SUPPORTED_MODALITIES))
        self.aliases = {**DEFAULT_ALIASES, **data.get("aliases", {})}
        self.settings = {**DEFAULT_SETTINGS, **data.get("settings", {})}
        for modality in SUPPORTED_MODALITIES:      # make sure each default still points somewhere
            if self.defaults.get(modality) not in self.models:
                candidates = [m["id"] for m in self.models.values() if m["modality"] == modality]
                if candidates:
                    self.defaults[modality] = candidates[0]
                else:
                    self.defaults.pop(modality, None)
        if added or not self.config_path.exists() or seeded != self.seeded_modalities:
            self._save_config()

    def _seed_config(self) -> dict:
        models = []
        for entry in MODEL_CATALOG:
            if not entry.get("default"):
                continue
            if entry.get("apple_silicon_only") and not IS_APPLE_SILICON:
                continue
            models.append(self._clean_entry(entry))
        ids = {m["id"] for m in models}
        return {
            "version": 1,
            "defaults": {
                "tts": "indic-speak-mlx" if "indic-speak-mlx" in ids else "indic-speak-torch",
                "asr": "indic-transcribe",
                "translation": "indic-translate" if "indic-translate" in ids else "indic-translate-torch",
            },
            "seeded_modalities": list(SUPPORTED_MODALITIES),
            "aliases": dict(DEFAULT_ALIASES),
            "settings": dict(DEFAULT_SETTINGS),
            "models": models,
        }

    @staticmethod
    def _clean_entry(entry: dict) -> dict:
        out = {k: copy.deepcopy(v) for k, v in entry.items() if k not in ("default", "apple_silicon_only")}
        out.setdefault("aliases", [])
        out.setdefault("default_params", {})
        out.setdefault("context_length", None)
        out.setdefault("chat_template", None)
        out.setdefault("created", int(time.time()))
        return out

    def _save_config(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "defaults": self.defaults,
                "seeded_modalities": getattr(self, "seeded_modalities", list(SUPPORTED_MODALITIES)),
                "aliases": self.aliases,
                "settings": self.settings,
                "models": list(self.models.values()),
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(self.config_path)

    # -- lookup --------------------------------------------------------------
    def register_factory(self, modality: str, factory: Callable[[dict], Any]) -> None:
        self._factories[modality] = factory

    def resolve(self, model: Optional[str], modality: Optional[str] = None) -> dict:
        """Map a request's `model` (id, alias, OpenAI name or repo id) to a registry entry."""
        with self._lock:
            name = (model or "").strip()
            spec = None
            if not name and modality:
                name = "@" + modality
            target = self.aliases.get(name, name)
            if target.startswith("@"):
                target = self.defaults.get(target[1:], "")
            if target in self.models:
                spec = self.models[target]
            else:
                lowered = target.lower()
                for m in self.models.values():
                    if lowered == m["id"].lower() or lowered == m["repo_id"].lower() or lowered in (a.lower() for a in m.get("aliases", [])):
                        spec = m
                        break
            if spec is None:
                raise ModelError(f"The model '{model}' does not exist. Register it via POST /v1/models/register.",
                                 status=404, code="model_not_found", param="model")
            if modality and spec["modality"] != modality:
                raise ModelError(f"Model '{spec['id']}' is a {spec['modality']} model and cannot serve {modality} requests.",
                                 status=400, code="model_modality_mismatch", param="model")
            return spec

    def alias_table(self) -> Dict[str, str]:
        """OpenAI alias -> concrete model id, with '@modality' targets resolved."""
        out = {}
        for alias, target in self.aliases.items():
            if target.startswith("@"):
                target = self.defaults.get(target[1:], "")
            if target in self.models:
                out[alias] = target
        return out

    # -- cache inspection ----------------------------------------------------
    def _scan_cache(self, force: bool = False) -> Dict[str, dict]:
        with self._lock:
            if not force and time.time() - self._cache_index_at < 5.0:
                return self._cache_index
            index: Dict[str, dict] = {}
            total = 0
            try:
                from huggingface_hub import scan_cache_dir
                info = scan_cache_dir()
                total = info.size_on_disk
                for repo in info.repos:
                    if repo.repo_type != "model":
                        continue
                    revisions = sorted(repo.revisions, key=lambda r: r.last_modified or 0, reverse=True)
                    blobs = Path(repo.repo_path) / "blobs"
                    index[repo.repo_id] = {
                        "size": repo.size_on_disk,
                        "path": str(revisions[0].snapshot_path) if revisions else None,
                        "incomplete": any(blobs.glob("*.incomplete")) if blobs.is_dir() else False,
                    }
            except Exception as e:  # CacheNotFound on a fresh machine
                logger.debug("HF cache scan: %s", e)
            self._cache_index, self._cache_index_at, self._cache_total = index, time.time(), total
            return index

    def _repos(self, spec: dict) -> List[str]:
        return [spec["repo_id"], *spec.get("extra_repos", [])]

    def is_cached(self, spec: dict) -> bool:
        index = self._scan_cache()
        for i, repo in enumerate(self._repos(spec)):
            entry = index.get(repo)
            if not entry or not entry["path"] or entry["incomplete"]:
                return False
            expected = spec.get("remote_size_bytes") if i == 0 else None
            if expected and entry["size"] < 0.98 * expected:
                return False
        return True

    def status_of(self, spec: dict) -> str:
        with self._lock:
            if any(t["model_id"] == spec["id"] and t["type"] == "download" and t["status"] == "in_progress"
                   for t in self._tasks.values()):
                return "downloading"
            if spec["id"] in self._loaded:
                return "loaded"
        return "cached" if self.is_cached(spec) else "registered"

    def public_model(self, spec: dict) -> dict:
        index = self._scan_cache()
        size = sum(index.get(r, {}).get("size", 0) for r in self._repos(spec))
        status = self.status_of(spec)
        loaded = self._loaded.get(spec["id"])
        active_task = next((t["task_id"] for t in self._tasks.values()
                            if t["model_id"] == spec["id"] and t["status"] == "in_progress"), None)
        return {
            "id": spec["id"],
            "object": "model",
            "created": spec.get("created", 0),
            "owned_by": spec["repo_id"].split("/")[0],
            "repo_id": spec["repo_id"],
            "modality": spec["modality"],
            "backend": spec.get("backend", "auto"),
            "adapter": spec.get("adapter"),
            "quantization": spec.get("quantization", "default"),
            "context_length": spec.get("context_length"),
            "description": spec.get("description", ""),
            "aliases": sorted(set(spec.get("aliases", [])) | {a for a, t in self.alias_table().items() if t == spec["id"]}),
            "default_params": spec.get("default_params", {}),
            "status": status,
            "supported": spec["modality"] in SUPPORTED_MODALITIES and bool(spec.get("adapter")),
            "is_default": self.defaults.get(spec["modality"]) == spec["id"],
            "size_mb": round(size / MB, 1) if size else (round(spec["remote_size_bytes"] / MB, 1) if spec.get("remote_size_bytes") else None),
            "size_on_disk_mb": round(size / MB, 1),
            "memory_mb": round(loaded["memory_bytes"] / MB, 1) if loaded else None,
            "loaded_at": loaded["loaded_at"] if loaded else None,
            "last_used": loaded["last_used"] if loaded else None,
            "task_id": active_task,
        }

    def list_models(self) -> List[dict]:
        with self._lock:
            specs = list(self.models.values())
        return [self.public_model(s) for s in specs]

    # -- registration --------------------------------------------------------
    def register(self, payload: dict) -> dict:
        repo_id = (payload.get("repo_id") or "").strip().strip("/")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
            raise ModelError("repo_id must look like 'organisation/model-name'.", param="repo_id")
        modality = payload.get("modality")
        if modality not in MODALITIES:
            raise ModelError(f"modality must be one of {', '.join(MODALITIES)}.", param="modality")
        backend = payload.get("backend") or "auto"
        if backend not in BACKENDS:
            raise ModelError(f"backend must be one of {', '.join(BACKENDS)}.", param="backend")
        if backend == "auto":
            backend = "mlx" if ("mlx" in repo_id.lower() and IS_APPLE_SILICON) else "transformers"
        model_id = (payload.get("id") or slugify_model_id(repo_id)).strip()
        if not MODEL_ID_RE.fullmatch(model_id):
            raise ModelError("id may only contain letters, digits, '.', '_' and '-' (max 64 chars).", param="id")
        if model_id in ("register", "tasks", "recommended"):
            raise ModelError(f"'{model_id}' is a reserved name.", param="id")
        with self._lock:
            if model_id in self.models and not payload.get("overwrite"):
                raise ModelError(f"A model with id '{model_id}' is already registered.", status=409,
                                 code="model_exists", param="id")
            if model_id in self._loaded:
                raise ModelError(f"Unload '{model_id}' before re-registering it.", status=409, code="model_loaded")
            catalog = next((c for c in MODEL_CATALOG if c["repo_id"].lower() == repo_id.lower()), {})
            spec = {
                "id": model_id,
                "repo_id": repo_id,
                "modality": modality,
                "backend": backend,
                "quantization": payload.get("quantization") or guess_quantization(repo_id),
                "adapter": payload.get("adapter") or guess_adapter(repo_id, modality, backend),
                "context_length": payload.get("context_length"),
                "chat_template": payload.get("chat_template"),
                "default_params": payload.get("default_params") or copy.deepcopy(catalog.get("default_params", {})),
                "description": payload.get("description") or catalog.get("description", ""),
                "aliases": [a for a in (payload.get("aliases") or []) if isinstance(a, str) and a],
                "extra_repos": payload.get("extra_repos") or list(catalog.get("extra_repos", [])),
                "ignore_patterns": payload.get("ignore_patterns") or list(catalog.get("ignore_patterns", [])),
                "remote_size_bytes": payload.get("remote_size_bytes"),
                "created": int(time.time()),
            }
            self.models[model_id] = spec
            self.defaults.setdefault(modality, model_id)
            self._save_config()
        return spec

    def set_default(self, model_id: str) -> dict:
        spec = self.resolve(model_id)
        with self._lock:
            self.defaults[spec["modality"]] = spec["id"]
            self._save_config()
        return spec

    def unregister(self, model_id: str, delete_files: bool = False) -> dict:
        spec = self.resolve(model_id)
        if self.status_of(spec) == "downloading":
            raise ModelError("A download is in progress for this model; wait for it to finish.", status=409, code="model_busy")
        freed = 0
        with self._lock:
            del self.models[spec["id"]]
            if self.defaults.get(spec["modality"]) == spec["id"]:
                remaining = [m["id"] for m in self.models.values() if m["modality"] == spec["modality"]]
                if remaining:
                    self.defaults[spec["modality"]] = remaining[0]
                else:
                    self.defaults.pop(spec["modality"], None)
            self._save_config()
            still_used = {r for m in self.models.values() for r in self._repos(m)}
        if delete_files:
            freed = self._delete_cached([r for r in self._repos(spec) if r not in still_used])
        return {"id": spec["id"], "object": "model", "deleted": True, "files_deleted": delete_files, "freed_mb": round(freed / MB, 1)}

    def _delete_cached(self, repos: List[str]) -> int:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            hashes = [rev.commit_hash for repo in info.repos if repo.repo_id in repos for rev in repo.revisions]
            if not hashes:
                return 0
            strategy = info.delete_revisions(*hashes)
            freed = strategy.expected_freed_size
            strategy.execute()
            self._scan_cache(force=True)
            return freed
        except Exception as e:
            logger.warning("Could not delete cached files for %s: %s", repos, e)
            return 0

    # -- Hugging Face Hub ----------------------------------------------------
    def _remote_size(self, repo_id: str, ignore_patterns: Optional[List[str]] = None) -> int:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import filter_repo_objects
        info = HfApi(token=self.hf_token).model_info(repo_id, files_metadata=True)
        siblings = list(filter_repo_objects(info.siblings or [], ignore_patterns=ignore_patterns or None,
                                            key=lambda s: s.rfilename))
        return sum(s.size or 0 for s in siblings)

    def inspect_repo(self, repo_id: str) -> dict:
        """Query the Hub and suggest how to register `repo_id`."""
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
        repo_id = repo_id.strip().strip("/")
        try:
            info = HfApi(token=self.hf_token).model_info(repo_id, files_metadata=True)
        except RepositoryNotFoundError:
            raise ModelError(f"Repository '{repo_id}' was not found on the Hugging Face Hub (or is private; set HF_TOKEN).",
                             status=404, code="repo_not_found", param="repo_id")
        except GatedRepoError:
            raise ModelError(f"Repository '{repo_id}' is gated; accept its terms on huggingface.co and set HF_TOKEN.",
                             status=403, code="repo_gated", param="repo_id")
        except HfHubHTTPError as e:
            raise ModelError(f"Hugging Face Hub request failed: {e}", status=502, code="hub_error")
        except Exception as e:
            raise ModelError(f"Could not reach the Hugging Face Hub: {e}", status=502, code="hub_error")

        tags = [t.lower() for t in (info.tags or [])]
        files = [s.rfilename for s in (info.siblings or [])]
        name = repo_id.lower()
        modality = PIPELINE_TO_MODALITY.get(info.pipeline_tag or "")
        if modality is None:
            if any(k in name for k in ("whisper", "transcribe", "asr", "wav2vec", "canary", "parakeet")):
                modality = "asr"
            elif any(k in name for k in ("translat", "indictrans", "nllb", "madlad")):
                modality = "translation"
            elif any(k in name for k in ("tts", "speak", "kokoro", "bark", "parler", "vits")):
                modality = "tts"
            elif any(k in name for k in ("rerank",)):
                modality = "rerank"
            elif any(k in name for k in ("embed", "bge-", "e5-", "minilm")):
                modality = "embedding"
            elif any(k in name for k in ("flux", "stable-diffusion", "sdxl")):
                modality = "image_gen"
            else:
                modality = "llm"
        if (info.library_name or "").lower() == "mlx" or "mlx" in tags or name.startswith("mlx-community/") or "-mlx" in name:
            backend = "mlx"
        elif (info.library_name or "").lower() == "diffusers" or "diffusers" in tags:
            backend = "diffusers"
        else:
            backend = "transformers"
        catalog = next((c for c in MODEL_CATALOG if c["repo_id"].lower() == name
                        and not (c.get("apple_silicon_only") and not IS_APPLE_SILICON)), None)
        if catalog:
            backend = catalog["backend"]
        adapter = (catalog or {}).get("adapter") or guess_adapter(repo_id, modality, backend)
        size = sum(s.size or 0 for s in (info.siblings or []))
        supported = modality in SUPPORTED_MODALITIES and adapter is not None
        notes = []
        if modality not in SUPPORTED_MODALITIES:
            notes.append(f"'{modality}' models can be registered and downloaded now; serving them arrives in a later phase.")
        elif adapter is None:
            notes.append("No engine adapter matches this repository yet, so it cannot be loaded.")
        if backend == "mlx" and not IS_APPLE_SILICON:
            notes.append("MLX weights need Apple Silicon.")
        if info.gated and not self.hf_token:
            notes.append("Gated repository: set HF_TOKEN before downloading.")
        with self._lock:
            registered_as = [m["id"] for m in self.models.values() if m["repo_id"].lower() == name]
        return {
            "repo_id": info.id or repo_id,
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "tags": (info.tags or [])[:40],
            "license": next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None),
            "downloads": info.downloads,
            "likes": info.likes,
            "gated": bool(info.gated),
            "private": bool(info.private),
            "last_modified": info.last_modified.isoformat() if info.last_modified else None,
            "num_files": len(files),
            "size_bytes": size,
            "size_mb": round(size / MB, 1),
            "cached": bool(self._scan_cache().get(info.id or repo_id)),
            "registered_as": registered_as,
            "suggested": {
                "id": (catalog or {}).get("id") or slugify_model_id(repo_id),
                "modality": modality,
                "backend": backend,
                "quantization": (catalog or {}).get("quantization") or guess_quantization(repo_id),
                "adapter": adapter,
                "default_params": (catalog or {}).get("default_params", {}),
            },
            "supported": supported,
            "notes": notes,
        }

    def recommended(self) -> List[dict]:
        with self._lock:
            registered = {m["repo_id"].lower(): m["id"] for m in self.models.values()}
        out = []
        for entry in MODEL_CATALOG:
            if entry.get("apple_silicon_only") and not IS_APPLE_SILICON:
                continue
            out.append({
                "id": entry["id"], "repo_id": entry["repo_id"], "modality": entry["modality"],
                "backend": entry["backend"], "quantization": entry["quantization"],
                "description": entry["description"],
                "registered_as": registered.get(entry["repo_id"].lower()),
            })
        return out

    # -- downloads -----------------------------------------------------------
    def get_task(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ModelError(f"Task '{task_id}' not found.", status=404, code="task_not_found")
            return dict(task)

    def list_tasks(self) -> List[dict]:
        with self._lock:
            return sorted((dict(t) for t in self._tasks.values()), key=lambda t: t["started_at"], reverse=True)

    def _new_task(self, model_id: str, kind: str) -> dict:
        task = {
            "task_id": f"task_{uuid.uuid4().hex[:12]}", "model_id": model_id, "type": kind,
            "status": "in_progress", "progress": 0.0, "downloaded_bytes": 0, "total_bytes": 0,
            "speed_bps": 0.0, "eta_seconds": None, "error": None,
            "started_at": time.time(), "finished_at": None,
        }
        with self._lock:
            self._tasks[task["task_id"]] = task
            if len(self._tasks) > 200:      # keep the task log bounded
                for old in sorted(self._tasks.values(), key=lambda t: t["started_at"])[:50]:
                    if old["status"] != "in_progress":
                        self._tasks.pop(old["task_id"], None)
        return task

    def start_download(self, model_id: str) -> dict:
        """Kick off (or join) a background download. Returns the task record."""
        spec = self.resolve(model_id)
        with self._lock:
            for t in self._tasks.values():
                if t["model_id"] == spec["id"] and t["type"] == "download" and t["status"] == "in_progress":
                    return dict(t)
            task = self._new_task(spec["id"], "download")
            done = threading.Event()
            self._download_futures[task["task_id"]] = done
        threading.Thread(target=self._download_worker, args=(task, spec, done), daemon=True,
                         name=f"download-{spec['id']}").start()
        return dict(task)

    def _download_worker(self, task: dict, spec: dict, done: threading.Event) -> None:
        from huggingface_hub import snapshot_download
        repos = self._repos(spec)
        stop = threading.Event()
        try:
            totals = {}
            for i, repo in enumerate(repos):
                try:
                    totals[repo] = self._remote_size(repo, spec.get("ignore_patterns") if i == 0 else None)
                except Exception as e:
                    logger.warning("Could not read size of %s: %s", repo, e)
                    totals[repo] = 0
            total = sum(totals.values())
            task["total_bytes"] = total
            if totals.get(spec["repo_id"]):
                spec["remote_size_bytes"] = totals[spec["repo_id"]]

            counted = {"bytes": 0}      # bytes written to disk, reported by huggingface_hub itself

            def monitor():
                last_bytes, last_t, best = None, time.time(), 0
                while not stop.wait(0.5):
                    # Fallback for hub versions without byte bars: size of the cache folders.
                    on_disk = sum(_dir_bytes(_repo_cache_folder(r) / "blobs") for r in repos)
                    best = have = max(best, counted["bytes"] or on_disk)
                    now = time.time()
                    if last_bytes is not None and now > last_t:
                        inst = max(have - last_bytes, 0) / (now - last_t)
                        task["speed_bps"] = round(0.6 * task["speed_bps"] + 0.4 * inst, 1)
                    last_bytes, last_t = have, now
                    task["downloaded_bytes"] = min(have, total) if total else have
                    if total:
                        task["progress"] = round(min(99.9, 100.0 * have / total), 1)
                        remaining = max(total - have, 0)
                        task["eta_seconds"] = round(remaining / task["speed_bps"]) if task["speed_bps"] > 1 else None

            threading.Thread(target=monitor, daemon=True).start()
            for i, repo in enumerate(repos):
                snapshot_download(repo, token=self.hf_token, tqdm_class=_byte_counting_tqdm(counted),
                                  ignore_patterns=(spec.get("ignore_patterns") or None) if i == 0 else None)
            task.update(status="completed", progress=100.0, downloaded_bytes=total or task["downloaded_bytes"],
                        eta_seconds=0, speed_bps=0.0)
            with self._lock:
                self._save_config()
        except Exception as e:
            logger.error("Download of %s failed: %s", spec["id"], e)
            task.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
        finally:
            stop.set()
            task["finished_at"] = time.time()
            self._scan_cache(force=True)
            done.set()

    async def wait_for_download(self, task_id: str) -> dict:
        done = self._download_futures.get(task_id)
        if done is not None:
            await asyncio.get_running_loop().run_in_executor(None, done.wait)
        return self.get_task(task_id)

    # -- load / unload -------------------------------------------------------
    async def run(self, fn: Callable, *args):
        """Run `fn` on the single GPU worker thread."""
        return await asyncio.get_running_loop().run_in_executor(self._gpu, fn, *args)

    def submit(self, fn: Callable, *args):
        return self._gpu.submit(fn, *args)

    async def ensure_loaded(self, model: Optional[str], modality: Optional[str] = None) -> dict:
        """Resolve, download if needed, and load a model. Returns its registry entry."""
        spec = self.resolve(model, modality)
        if spec["id"] in self._loaded:
            return spec
        if spec["modality"] not in SUPPORTED_MODALITIES:
            raise ModelError(f"Serving '{spec['modality']}' models is not implemented yet "
                             f"(supported: {', '.join(SUPPORTED_MODALITIES)}).",
                             status=501, code="modality_not_supported")
        if not spec.get("adapter"):
            raise ModelError(f"No engine adapter is available for '{spec['repo_id']}'.", status=501, code="adapter_not_found")
        if self.status_of(spec) != "cached":
            running = self.status_of(spec) == "downloading"
            if not running and not self.settings.get("auto_download", True):
                raise ModelError(f"Model '{spec['id']}' is not downloaded. POST /v1/models/{spec['id']}/download first.",
                                 status=409, code="model_not_downloaded")
            task = self.start_download(spec["id"])
            task = await self.wait_for_download(task["task_id"])
            if task["status"] != "completed":
                raise ModelError(f"Download of '{spec['id']}' failed: {task.get('error')}", status=502, code="download_failed")
        await self.run(self._load_sync, spec)
        return spec

    def _device_memory(self) -> int:
        used = 0
        try:
            import mlx.core as mx
            used += mx.get_active_memory()
        except Exception:
            pass
        try:
            import torch
            if torch.backends.mps.is_available():
                # not driver_allocated_memory(): that is process-wide Metal and would count MLX twice
                used += torch.mps.current_allocated_memory()
        except Exception:
            pass
        return used

    def _load_sync(self, spec: dict) -> None:
        """Runs on the GPU thread."""
        if spec["id"] in self._loaded:
            return
        factory = self._factories.get(spec["modality"])
        if factory is None:
            raise ModelError(f"No engine registered for modality '{spec['modality']}'.", status=501, code="modality_not_supported")
        from huggingface_hub import snapshot_download
        try:
            local_dir = snapshot_download(spec["repo_id"], local_files_only=True, token=self.hf_token)
        except Exception as e:
            raise ModelError(f"Weights for '{spec['id']}' are not on disk: {e}", status=409, code="model_not_downloaded")

        self._evict_for(spec)
        task = self._new_task(spec["id"], "load")
        before, t0 = self._device_memory(), time.time()
        try:
            adapter = factory(spec)
            adapter.load(local_dir, spec, self.hf_token)
        except ModelError as e:
            task.update(status="failed", error=e.message, finished_at=time.time())
            self.clear_device_caches()
            raise
        except Exception as e:
            logger.exception("Loading %s failed", spec["id"])
            task.update(status="failed", error=f"{type(e).__name__}: {e}"[:500], finished_at=time.time())
            self.clear_device_caches()
            raise ModelError(f"Loading '{spec['id']}' failed: {type(e).__name__}: {e}", status=500, code="model_load_failed")
        now = time.time()
        with self._lock:
            self._loaded[spec["id"]] = {
                "adapter": adapter, "loaded_at": now, "last_used": now,
                "memory_bytes": max(self._device_memory() - before, 0), "load_seconds": round(now - t0, 2),
            }
        task.update(status="completed", progress=100.0, finished_at=now)
        logger.info("Loaded %s in %.1fs", spec["id"], now - t0)

    def _evict_for(self, incoming: dict) -> None:
        """LRU-evict resident models to respect the count and free-memory limits."""
        max_models = max(1, int(self.settings.get("max_loaded_models", 3)))
        min_free = float(self.settings.get("min_free_memory_gb", 4.0)) * GB
        index = self._scan_cache()
        estimate = int(1.15 * sum(index.get(r, {}).get("size", 0) for r in self._repos(incoming)))

        def lru():
            with self._lock:
                return min(self._loaded, key=lambda k: self._loaded[k]["last_used"]) if self._loaded else None

        while len(self._loaded) >= max_models:
            victim = lru()
            logger.info("Evicting %s (max_loaded_models=%d)", victim, max_models)
            self._unload_sync(victim)
        while self._loaded and psutil.virtual_memory().available - estimate < min_free:
            victim = lru()
            logger.info("Evicting %s to free memory for %s", victim, incoming["id"])
            self._unload_sync(victim)

    def _unload_sync(self, model_id: str) -> bool:
        with self._lock:
            entry = self._loaded.pop(model_id, None)
        if entry is None:
            return False
        try:
            entry["adapter"].unload()
        except Exception as e:
            logger.warning("Adapter unload for %s raised: %s", model_id, e)
        del entry
        self.clear_device_caches()
        logger.info("Unloaded %s", model_id)
        return True

    @staticmethod
    def clear_device_caches() -> None:
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def unload(self, model_id: str) -> bool:
        spec = self.resolve(model_id)
        return await self.run(self._unload_sync, spec["id"])

    def adapter(self, model_id: str):
        """Fetch a loaded adapter and mark it as recently used. Call on the GPU thread."""
        entry = self._loaded.get(model_id)
        if entry is None:
            raise ModelError(f"Model '{model_id}' was unloaded before the request ran; retry.", status=503, code="model_unloaded")
        entry["last_used"] = time.time()
        return entry["adapter"]

    def loaded_ids(self) -> List[str]:
        with self._lock:
            return list(self._loaded)

    async def unload_idle(self) -> List[str]:
        minutes = float(self.settings.get("idle_unload_minutes", 0) or 0)
        if minutes <= 0:
            return []
        cutoff = time.time() - minutes * 60
        with self._lock:
            idle = [mid for mid, e in self._loaded.items() if e["last_used"] < cutoff]
        for mid in idle:
            await self.run(self._unload_sync, mid)
        return idle

    def shutdown(self) -> None:
        for mid in self.loaded_ids():
            try:
                self._gpu.submit(self._unload_sync, mid).result(timeout=30)
            except Exception:
                pass
        self._gpu.shutdown(wait=False, cancel_futures=True)

    # -- settings ------------------------------------------------------------
    def update_settings(self, changes: dict) -> dict:
        with self._lock:
            if "hf_token" in changes:
                self.hf_token = (changes["hf_token"] or "").strip() or None   # memory only, never persisted
            if changes.get("max_loaded_models") is not None:
                self.settings["max_loaded_models"] = max(1, min(int(changes["max_loaded_models"]), 16))
            if changes.get("min_free_memory_gb") is not None:
                self.settings["min_free_memory_gb"] = max(0.0, float(changes["min_free_memory_gb"]))
            if changes.get("idle_unload_minutes") is not None:
                self.settings["idle_unload_minutes"] = max(0.0, float(changes["idle_unload_minutes"]))
            if changes.get("auto_download") is not None:
                self.settings["auto_download"] = bool(changes["auto_download"])
            self._save_config()
        return self.public_settings()

    def public_settings(self) -> dict:
        return {**self.settings, "hf_token_set": bool(self.hf_token), "defaults": dict(self.defaults),
                "aliases": self.alias_table()}

    # -- system status -------------------------------------------------------
    _chip: Optional[str] = None

    @classmethod
    def chip_name(cls) -> str:
        if cls._chip is None:
            chip = platform.processor() or platform.machine()
            if platform.system() == "Darwin":
                try:
                    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                          text=True, timeout=2).stdout.strip() or chip
                except Exception:
                    pass
            cls._chip = chip
        return cls._chip

    def system_status(self) -> dict:
        vm = psutil.virtual_memory()
        memory = {
            "total_gb": round(vm.total / GB, 2),
            "used_gb": round((vm.total - vm.available) / GB, 2),
            "free_gb": round(vm.available / GB, 2),
            "percent": vm.percent,
            "active_gb": round(getattr(vm, "active", 0) / GB, 2),
            "wired_gb": round(getattr(vm, "wired", 0) / GB, 2),
            "process_rss_gb": round(psutil.Process().memory_info().rss / GB, 2),
        }
        mlx_info: Dict[str, Any] = {"available": False}
        try:
            import mlx.core as mx
            mlx_info = {
                "available": True,
                "active_gb": round(mx.get_active_memory() / GB, 3),
                "cache_gb": round(mx.get_cache_memory() / GB, 3),
                "peak_gb": round(mx.get_peak_memory() / GB, 3),
            }
        except Exception:
            pass
        mps_info: Dict[str, Any] = {"available": False}
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                mps_info = {
                    "available": True,
                    "allocated_gb": round(torch.mps.current_allocated_memory() / GB, 3),
                    "driver_allocated_gb": round(torch.mps.driver_allocated_memory() / GB, 3),
                }
            elif torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        self._scan_cache()
        disk = psutil.disk_usage(str(hf_cache_dir() if hf_cache_dir().exists() else Path.home()))
        with self._lock:
            loaded = [{
                "id": mid, "modality": self.models[mid]["modality"] if mid in self.models else None,
                "memory_mb": round(e["memory_bytes"] / MB, 1), "loaded_at": e["loaded_at"],
                "last_used": e["last_used"], "load_seconds": e["load_seconds"],
            } for mid, e in self._loaded.items()]
            active_tasks = [dict(t) for t in self._tasks.values() if t["status"] == "in_progress"]
            n_models = len(self.models)
        return {
            "status": "ok",
            "device": {
                "chip": self.chip_name(), "platform": platform.platform(), "apple_silicon": IS_APPLE_SILICON,
                "accelerator": device, "mlx": mlx_info["available"], "mps": mps_info["available"],
                "cpu_count": psutil.cpu_count(), "python": platform.python_version(),
            },
            "memory": memory,
            "mlx": mlx_info,
            "mps": mps_info,
            "disk": {
                "hf_cache_dir": str(hf_cache_dir()),
                "hf_cache_gb": round(self._cache_total / GB, 2),
                "disk_free_gb": round(disk.free / GB, 1),
                "disk_total_gb": round(disk.total / GB, 1),
            },
            "models": {"registered": n_models, "loaded": len(loaded)},
            "loaded_models": loaded,
            "active_tasks": active_tasks,
            "settings": self.public_settings(),
        }
