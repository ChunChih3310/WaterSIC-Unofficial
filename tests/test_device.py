from __future__ import annotations

import logging
import os
import subprocess

import pytest

from watersic.utils.device import pick_idle_gpu


def _mock_nvidia_smi(monkeypatch, *, gpu_output: str, process_output: str = "") -> None:
    def _run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        query = " ".join(cmd)
        if "--query-gpu=" in query:
            return subprocess.CompletedProcess(cmd, 0, stdout=gpu_output, stderr="")
        if "--query-compute-apps=" in query:
            return subprocess.CompletedProcess(cmd, 0, stdout=process_output, stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _run)


def test_pick_idle_gpu_prefers_zero_process_gpu(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _mock_nvidia_smi(
        monkeypatch,
        gpu_output=(
            "0, GPU-busy, NVIDIA RTX A6000, 12288, 36852, 49140, 2\n"
            "1, GPU-idle, NVIDIA RTX A6000, 256, 48884, 49140, 11\n"
        ),
        process_output=(
            "GPU-busy, 1001, 8192\n"
            "GPU-busy, 1002, 4096\n"
        ),
    )
    observed_visible: list[str | None] = []

    def _is_available() -> bool:
        observed_visible.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return True

    monkeypatch.setattr("torch.cuda.is_available", _is_available)

    selection = pick_idle_gpu()

    assert selection.gpu_index == 1
    assert selection.cuda_visible_devices == "1"
    assert selection.torch_device == "cuda"
    assert selection.logical_device_index == 0
    assert selection.reason == "idle_gpu_selected"
    assert any("gpu=1" in line and "processes=0" in line for line in selection.ranking)
    assert observed_visible == ["1"]


def test_pick_idle_gpu_prefers_more_free_memory_when_process_free(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _mock_nvidia_smi(
        monkeypatch,
        gpu_output=(
            "0, GPU-more-free, NVIDIA RTX A6000, 1024, 44000, 45024, 25\n"
            "1, GPU-less-free, NVIDIA RTX A6000, 1024, 32000, 33024, 5\n"
        ),
        process_output="",
    )

    selection = pick_idle_gpu()

    assert selection.gpu_index == 0
    assert selection.reason == "idle_gpu_selected"
    assert selection.warning is None


def test_pick_idle_gpu_raises_when_all_gpus_busy_by_default(monkeypatch, caplog) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _mock_nvidia_smi(
        monkeypatch,
        gpu_output=(
            "0, GPU-one-proc, NVIDIA RTX A6000, 8192, 40948, 49140, 5\n"
            "1, GPU-two-proc, NVIDIA RTX A6000, 4096, 45044, 49140, 0\n"
        ),
        process_output=(
            "GPU-one-proc, 2001, 8192\n"
            "GPU-two-proc, 2002, 2048\n"
            "GPU-two-proc, 2003, 2048\n"
        ),
    )
    logger = logging.getLogger("watersic.test_device")

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError) as exc_info:
            pick_idle_gpu(logger=logger)
    message = str(exc_info.value)

    assert "refusing to auto-select a busy GPU" in message
    assert "No GPU met the idle thresholds" in caplog.text


def test_pick_idle_gpu_warns_and_chooses_least_bad_with_explicit_override(monkeypatch, caplog) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _mock_nvidia_smi(
        monkeypatch,
        gpu_output=(
            "0, GPU-one-proc, NVIDIA RTX A6000, 8192, 40948, 49140, 5\n"
            "1, GPU-two-proc, NVIDIA RTX A6000, 4096, 45044, 49140, 0\n"
        ),
        process_output=(
            "GPU-one-proc, 2001, 8192\n"
            "GPU-two-proc, 2002, 2048\n"
            "GPU-two-proc, 2003, 2048\n"
        ),
    )
    logger = logging.getLogger("watersic.test_device")

    with caplog.at_level(logging.INFO):
        selection = pick_idle_gpu(device_config={"allow_busy_fallback": True}, logger=logger)

    assert selection.gpu_index == 0
    assert selection.reason == "least_bad_gpu_selected"
    assert selection.warning is not None
    assert "No GPU met the idle thresholds" in selection.warning
    assert "No GPU met the idle thresholds" in caplog.text


def test_pick_idle_gpu_respects_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("nvidia-smi should not run when CUDA_VISIBLE_DEVICES is already set")

    monkeypatch.setattr(subprocess, "run", _should_not_run)

    selection = pick_idle_gpu()

    assert selection.reason == "cuda_visible_devices_respected"
    assert selection.cuda_visible_devices == "6"
    assert selection.torch_device == "cuda"
    assert selection.gpu_index == 6
