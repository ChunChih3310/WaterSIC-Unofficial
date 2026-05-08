from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .path_guard import ensure_parent_dir, ensure_within_repo


def load_yaml(path: str | Path) -> dict[str, Any]:
    with ensure_within_repo(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    with ensure_within_repo(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return target
