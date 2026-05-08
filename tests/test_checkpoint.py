from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from watersic.quant.checkpoint import CheckpointConfig, CheckpointManager, StagePlanEntry
from watersic.quant.watersic_layer import LayerStatistics
from watersic.report.schema import LayerReport
from watersic.stats.covariance import SecondMomentAccumulator
from watersic.utils.path_guard import repo_path


class _FakeMLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = _FakeMLP(hidden_dim)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_dim)])


class _FakeModel(nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.model = _FakeBackbone(hidden_dim)


def _checkpoint_root(name: str) -> Path:
    return repo_path("outputs", "test_checkpoints", name)


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _build_manager(*, run_name: str, checkpoint_dir: str, git_commit: str = "abc123") -> tuple[CheckpointManager, _FakeModel, StagePlanEntry]:
    model = _FakeModel().eval()
    module_path = "model.layers.0.mlp.gate_proj"
    stage = StagePlanEntry(
        layer_index=0,
        stage_index=0,
        stage_kinds=("gate_proj",),
        module_paths=(module_path,),
        module_num_weights={module_path: int(model.get_submodule(module_path).weight.numel())},
    )
    manager = CheckpointManager(
        run_name=run_name,
        config=CheckpointConfig(enabled=True, dir=checkpoint_dir, resume="auto", strict_git_commit=True),
        model_config={"model_id": "fake/model", "model_revision": "main", "tokenizer_id": "fake/model"},
        quant_config={
            "run_name": run_name,
            "target_global_bitwidth": 3.0,
            "reference_stats": True,
            "calibration": {"split": "train", "sequence_length": 8, "num_sequences": 4, "batch_size": 1},
            "_config_path": "configs/test_quant.yaml",
        },
        eval_config={
            "split": "test",
            "sequence_length": 8,
            "_config_path": "configs/test_eval.yaml",
        },
        report_metadata={
            "timestamp": "2026-04-11T00:00:00Z",
            "git_commit": git_commit,
            "environment_name": "watersic",
            "model_id": "fake/model",
            "model_revision": "main",
            "tokenizer_id": "fake/model",
            "tokenizer_revision": "main",
            "quant_config_path": "configs/test_quant.yaml",
            "eval_config_path": "configs/test_eval.yaml",
            "calibration_split": "train",
            "sequence_length": 8,
            "calibration_sequences": 4,
            "notes": [],
        },
        total_weights=int(model.get_submodule(module_path).weight.numel()),
        stage_plan=[stage],
    )
    return manager, model, stage


def test_second_moment_accumulator_state_dict_roundtrip() -> None:
    acc = SecondMomentAccumulator(3)
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=torch.float32)
    acc.update(x)

    restored = SecondMomentAccumulator.from_state_dict(acc.state_dict())

    torch.testing.assert_close(restored.finalize(), acc.finalize())
    assert restored.weight_sum == acc.weight_sum


def test_checkpoint_collection_and_stage_stats_roundtrip() -> None:
    name = f"pytest_ckpt_{uuid.uuid4().hex[:8]}"
    root = _checkpoint_root(name)
    _cleanup(root)
    try:
        manager, _model, stage = _build_manager(run_name=name, checkpoint_dir="outputs/test_checkpoints")
        manager.initialize_or_validate()

        acc = SecondMomentAccumulator(2)
        acc.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
        accumulators = {"model.layers.0.mlp.gate_proj": {"sigma_x": acc}}
        manager.save_collection_state(stage, accumulators=accumulators, next_batch_index=3)

        restored_state = manager.load_collection_state(stage)
        assert restored_state is not None
        restored_accumulators, next_batch = restored_state
        assert next_batch == 3
        torch.testing.assert_close(
            restored_accumulators["model.layers.0.mlp.gate_proj"]["sigma_x"].finalize(),
            acc.finalize(),
        )

        stats = {
            "model.layers.0.mlp.gate_proj": LayerStatistics(
                sigma_x=torch.eye(2, dtype=torch.float64),
                sigma_x_hat=2 * torch.eye(2, dtype=torch.float64),
                sigma_x_x_hat=3 * torch.eye(2, dtype=torch.float64),
                variances=torch.tensor([2.0, 2.0], dtype=torch.float64),
            )
        }
        manager.save_stage_stats(stage, stats)
        restored_stats = manager.load_stage_stats(stage)
        assert restored_stats is not None
        torch.testing.assert_close(restored_stats["model.layers.0.mlp.gate_proj"].sigma_x, stats["model.layers.0.mlp.gate_proj"].sigma_x)
        torch.testing.assert_close(
            restored_stats["model.layers.0.mlp.gate_proj"].sigma_x_hat,
            stats["model.layers.0.mlp.gate_proj"].sigma_x_hat,
        )
    finally:
        _cleanup(root)


def test_checkpoint_committed_stage_replay_restores_weights_and_budget() -> None:
    name = f"pytest_ckpt_{uuid.uuid4().hex[:8]}"
    root = _checkpoint_root(name)
    _cleanup(root)
    try:
        manager, model, stage = _build_manager(run_name=name, checkpoint_dir="outputs/test_checkpoints")
        manager.initialize_or_validate()

        module_path = stage.module_paths[0]
        target_weight = torch.arange(model.get_submodule(module_path).weight.numel(), dtype=torch.float32).view_as(
            model.get_submodule(module_path).weight
        )
        model.get_submodule(module_path).weight.data.copy_(target_weight)
        manager.save_committed_stage(
            stage,
            model=model,
            stage_layer_reports=[
                LayerReport(
                    name=module_path,
                    kind="gate_proj",
                    target_bitwidth=3.0,
                    achieved_bitwidth=2.9,
                    raw_bitwidth=4.0,
                    entropy_bitwidth=2.7,
                    huffman_bitwidth=2.8,
                    huffman_shortest_symbol_length_bits=1,
                    huffman_longest_symbol_length_bits=6,
                    side_information_bitwidth=0.2,
                    weighted_error=0.01,
                    applied_damping=1e-4,
                )
            ],
            stage_layer_results={
                module_path: {
                    "kind": "gate_proj",
                    "layer_index": 0,
                    "target_rate": 3.0,
                    "achieved_rate": 2.9,
                    "entropy_rate": 2.7,
                    "huffman_rate": 2.8,
                    "huffman_shortest_symbol_length_bits": 1,
                    "huffman_longest_symbol_length_bits": 6,
                    "reference_stats_effective": False,
                    "relative_weight_mse": 0.01,
                }
            },
            pre_stage_remaining_budget=48.0,
            post_stage_remaining_budget=1.6,
            pre_stage_remaining_weights=16,
            post_stage_remaining_weights=0,
            quantization_anomalies=["none"],
        )

        fresh_manager, fresh_model, _ = _build_manager(run_name=name, checkpoint_dir="outputs/test_checkpoints")
        fresh_manager.initialize_or_validate()
        restored = fresh_manager.restore_committed_progress(fresh_model)

        assert restored.resumed is True
        assert restored.restored_stage_count == 1
        assert restored.remaining_budget == pytest.approx(1.6)
        assert restored.remaining_weights == 0
        assert len(restored.layer_reports) == 1
        assert module_path in restored.layer_results
        torch.testing.assert_close(fresh_model.get_submodule(module_path).weight.detach().cpu(), target_weight)
    finally:
        _cleanup(root)


def test_checkpoint_manifest_rejects_git_mismatch_when_strict() -> None:
    name = f"pytest_ckpt_{uuid.uuid4().hex[:8]}"
    root = _checkpoint_root(name)
    _cleanup(root)
    try:
        manager, _model, _stage = _build_manager(run_name=name, checkpoint_dir="outputs/test_checkpoints", git_commit="commit_a")
        manager.initialize_or_validate()

        mismatch_manager, _model_b, _stage_b = _build_manager(
            run_name=name,
            checkpoint_dir="outputs/test_checkpoints",
            git_commit="commit_b",
        )
        with pytest.raises(RuntimeError, match="git commit mismatch"):
            mismatch_manager.initialize_or_validate()
    finally:
        _cleanup(root)
