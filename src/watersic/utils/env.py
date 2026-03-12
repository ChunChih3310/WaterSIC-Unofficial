from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .path_guard import repo_path


def load_repo_env() -> None:
    load_dotenv(repo_path(".env"), override=False)


def configure_repo_caches() -> dict[str, str]:
    cache_map = {
        "HF_HOME": str(repo_path("outputs", "hf_cache")),
        "HUGGINGFACE_HUB_CACHE": str(repo_path("outputs", "hf_cache", "hub")),
        "TRANSFORMERS_CACHE": str(repo_path("outputs", "hf_cache", "transformers")),
        "PIP_CACHE_DIR": str(repo_path(".cache", "pip")),
    }
    for key, value in cache_map.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, value)
    return cache_map
