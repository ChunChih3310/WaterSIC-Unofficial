from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .device import DeviceSelection, pick_idle_gpu
from .env import configure_repo_caches, load_repo_env
from .io import load_yaml, save_json
from .logging import configure_logging
from .path_guard import repo_path
from .seed import seed_everything


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path(), text=True).strip()
    except Exception:
        return "unknown"


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def prepare_runtime(*, log_name: str, debug: bool, seed: int) -> tuple[dict, object]:
    load_repo_env()
    caches = configure_repo_caches()
    seed_everything(seed)
    logger = configure_logging(log_name, debug=debug)
    logger.info("Configured repo-local caches: %s", caches)
    return caches, logger


def select_runtime_device(device_config: dict | None, logger) -> DeviceSelection:
    override = (device_config or {}).get("override")
    if override:
        if override == "cpu":
            selection = DeviceSelection(torch_device="cpu", cuda_visible_devices=None, reason="manual_cpu_override")
        elif override.startswith("cuda"):
            selection = DeviceSelection(torch_device=override, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), reason="manual_cuda_override")
        else:
            raise ValueError(f"Unsupported device override: {override}")
    else:
        selection = pick_idle_gpu(device_config=device_config, logger=logger)
    logger.info("Device selection: %s", selection)
    return selection


def load_config(path: str | Path) -> dict:
    return load_yaml(path)


def write_report_bundle(run_name: str, report_dict: dict, markdown: str) -> tuple[Path, Path]:
    json_path = repo_path("outputs", "reports", f"{run_name}.json")
    md_path = repo_path("outputs", "reports", f"{run_name}.md")
    save_json(json_path, report_dict)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def resolve_torch_device(selection: DeviceSelection) -> torch.device:
    return torch.device(selection.torch_device)
