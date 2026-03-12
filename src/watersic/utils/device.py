from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DeviceSelection:
    torch_device: str
    cuda_visible_devices: str | None
    reason: str
    gpu_index: int | None = None
    gpu_name: str | None = None


def _parse_nvidia_smi(output: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        index, name, memory_used, memory_total, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used": float(memory_used),
                "memory_total": float(memory_total),
                "utilization": float(utilization),
            }
        )
    return rows


def _idle_score(row: dict[str, float | int | str]) -> float:
    memory_frac = float(row["memory_used"]) / max(float(row["memory_total"]), 1.0)
    util_frac = float(row["utilization"]) / 100.0
    return memory_frac + util_frac


def pick_idle_gpu(candidates: Iterable[int] | None = None) -> DeviceSelection:
    if not torch.cuda.is_available():
        return DeviceSelection(torch_device="cpu", cuda_visible_devices=None, reason="cuda_unavailable")

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visible:
        return DeviceSelection(torch_device="cuda", cuda_visible_devices=existing_visible, reason="cuda_visible_devices_respected")

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(query, check=True, capture_output=True, text=True)
    rows = _parse_nvidia_smi(result.stdout)
    if candidates is not None:
        allowed = set(candidates)
        rows = [row for row in rows if int(row["index"]) in allowed]
    if not rows:
        return DeviceSelection(torch_device="cpu", cuda_visible_devices=None, reason="no_visible_gpus")

    best = min(rows, key=_idle_score)
    selected_index = int(best["index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)
    return DeviceSelection(
        torch_device="cuda",
        cuda_visible_devices=str(selected_index),
        reason="idle_gpu_selected",
        gpu_index=selected_index,
        gpu_name=str(best["name"]),
    )
