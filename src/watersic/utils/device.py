from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class DeviceSelection:
    torch_device: str
    cuda_visible_devices: str | None
    reason: str
    gpu_index: int | None = None
    gpu_name: str | None = None
    warning: str | None = None
    ranking: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoGpuPolicy:
    min_free_memory_gib: float = 24.0
    max_used_memory_gib: float = 2.0

    @property
    def min_free_memory_mib(self) -> float:
        return self.min_free_memory_gib * 1024.0

    @property
    def max_used_memory_mib(self) -> float:
        return self.max_used_memory_gib * 1024.0


def _parse_float(value: str) -> float:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        raise ValueError(f"Cannot parse numeric value from {value!r}")
    return float(value)


def _parse_int(value: str) -> int:
    return int(_parse_float(value))


def _parse_nvidia_smi(output: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, name, memory_used, memory_free, memory_total, utilization = parts
        rows.append(
            {
                "index": _parse_int(index),
                "uuid": uuid,
                "name": name,
                "memory_used": _parse_float(memory_used),
                "memory_free": _parse_float(memory_free),
                "memory_total": _parse_float(memory_total),
                "utilization": _parse_float(utilization),
            }
        )
    return rows


def _parse_compute_processes(output: str) -> dict[str, dict[str, float | int]]:
    processes: dict[str, dict[str, float | int]] = {}
    stripped = output.strip()
    if not stripped or stripped == "No running processes found":
        return processes
    for line in stripped.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected nvidia-smi process row: {line!r}")
        gpu_uuid, _pid, used_memory = parts
        stats = processes.setdefault(gpu_uuid, {"process_count": 0, "process_memory": 0.0})
        stats["process_count"] = int(stats["process_count"]) + 1
        if used_memory not in {"", "N/A", "[Not Supported]"}:
            stats["process_memory"] = float(stats["process_memory"]) + _parse_float(used_memory)
    return processes


def _build_policy(device_config: dict[str, Any] | None) -> AutoGpuPolicy:
    config = device_config or {}
    return AutoGpuPolicy(
        min_free_memory_gib=float(config.get("min_free_memory_gib", 24.0)),
        max_used_memory_gib=float(config.get("max_used_memory_gib", 2.0)),
    )


def _ranked_gpu_candidates(
    rows: list[dict[str, float | int | str]],
    *,
    process_stats: dict[str, dict[str, float | int]] | None,
    policy: AutoGpuPolicy,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    process_query_available = process_stats is not None
    for row in rows:
        used_mib = float(row["memory_used"])
        free_mib = float(row["memory_free"])
        process_count: int | None
        process_memory_mib: float | None
        if process_query_available:
            gpu_process_stats = process_stats.get(str(row["uuid"]), {})
            process_count = int(gpu_process_stats.get("process_count", 0))
            process_memory_mib = float(gpu_process_stats.get("process_memory", 0.0))
        else:
            process_count = None
            process_memory_mib = None

        # Ranking policy is intentionally conservative:
        # 1. confirmed zero-process GPUs outrank GPUs with unknown or active processes
        # 2. within the same process class, prefer lower used memory and higher free memory
        # 3. use utilization only as the final tie-breaker
        zero_process_rank = 0 if process_count == 0 else 1 if process_count is None else 2
        ranking_key = (
            zero_process_rank,
            process_count if process_count is not None else 10**6,
            used_mib,
            -free_mib,
            float(row["utilization"]),
            int(row["index"]),
        )
        is_idle = (
            process_count == 0
            and used_mib <= policy.max_used_memory_mib
            and free_mib >= policy.min_free_memory_mib
        )
        ranked.append(
            {
                **row,
                "process_count": process_count,
                "process_memory": process_memory_mib,
                "is_idle": is_idle,
                "ranking_key": ranking_key,
            }
        )
    return sorted(ranked, key=lambda row: row["ranking_key"])


def _format_candidate_summary(row: dict[str, Any], *, rank: int, policy: AutoGpuPolicy) -> str:
    process_count = row["process_count"]
    process_text = "unknown" if process_count is None else str(process_count)
    process_mem = row["process_memory"]
    process_mem_text = "unknown" if process_mem is None else f"{process_mem:.0f}MiB"
    idle_text = "idle" if row["is_idle"] else "busy_or_threshold_failed"
    return (
        f"rank={rank} gpu={row['index']} name={row['name']} "
        f"processes={process_text} process_mem={process_mem_text} "
        f"used={float(row['memory_used']):.0f}MiB free={float(row['memory_free']):.0f}MiB "
        f"util={float(row['utilization']):.0f}% idle_class={idle_text} "
        f"thresholds(max_used={policy.max_used_memory_mib:.0f}MiB,min_free={policy.min_free_memory_mib:.0f}MiB)"
    )


def pick_idle_gpu(
    candidates: Iterable[int] | None = None,
    *,
    device_config: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> DeviceSelection:
    if not torch.cuda.is_available():
        return DeviceSelection(torch_device="cpu", cuda_visible_devices=None, reason="cuda_unavailable")

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visible:
        return DeviceSelection(torch_device="cuda", cuda_visible_devices=existing_visible, reason="cuda_visible_devices_respected")

    policy = _build_policy(device_config)
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(gpu_query, check=True, capture_output=True, text=True)
        rows = _parse_nvidia_smi(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        warning = f"Falling back to CPU because GPU auto-selection query failed: {exc}"
        if logger is not None:
            logger.warning(warning)
        return DeviceSelection(
            torch_device="cpu",
            cuda_visible_devices=None,
            reason="gpu_query_failed",
            warning=warning,
        )

    if candidates is not None:
        allowed = set(candidates)
        rows = [row for row in rows if int(row["index"]) in allowed]
    if not rows:
        return DeviceSelection(torch_device="cpu", cuda_visible_devices=None, reason="no_visible_gpus")

    process_query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    process_stats: dict[str, dict[str, float | int]] | None
    process_warning: str | None = None
    try:
        process_result = subprocess.run(process_query, check=True, capture_output=True, text=True)
        process_stats = _parse_compute_processes(process_result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        process_stats = None
        process_warning = f"GPU process visibility unavailable; ranking falls back to memory-first selection: {exc}"

    ranked = _ranked_gpu_candidates(rows, process_stats=process_stats, policy=policy)
    ranking = tuple(
        _format_candidate_summary(row, rank=rank, policy=policy)
        for rank, row in enumerate(ranked, start=1)
    )
    if logger is not None:
        logger.info(
            "GPU auto-selection policy: prefer zero compute processes, then lowest used memory / highest free memory, then utilization."
        )
        for summary in ranking:
            logger.info("GPU candidate %s", summary)
        if process_warning:
            logger.warning(process_warning)

    best = ranked[0]
    warning: str | None = process_warning
    if not any(bool(row["is_idle"]) for row in ranked):
        warning = (
            "No GPU met the idle thresholds; selected the least-bad device by zero-process, used-memory, free-memory, and utilization ranking."
            if warning is None
            else warning
            + " No GPU met the idle thresholds; selected the least-bad device by zero-process, used-memory, free-memory, and utilization ranking."
        )
        if logger is not None:
            logger.warning(warning)

    selected_index = int(best["index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)
    return DeviceSelection(
        torch_device="cuda",
        cuda_visible_devices=str(selected_index),
        reason="idle_gpu_selected" if warning is None else "least_bad_gpu_selected",
        gpu_index=selected_index,
        gpu_name=str(best["name"]),
        warning=warning,
        ranking=ranking,
    )
