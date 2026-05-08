from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from watersic.quant.watersic_layer import LayerStatistics
from watersic.report.schema import LayerReport
from watersic.stats.covariance import SecondMomentAccumulator
from watersic.utils.io import load_json
from watersic.utils.path_guard import ensure_parent_dir, ensure_within_repo, repo_path


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool = False
    dir: str = "outputs/checkpoints"
    resume: str = "auto"
    strict_git_commit: bool = True
    keep_after_completion: bool = True
    save_collection_every_batches: int = 32


@dataclass(frozen=True)
class StagePlanEntry:
    layer_index: int
    stage_index: int
    stage_kinds: tuple[str, ...]
    module_paths: tuple[str, ...]
    module_num_weights: dict[str, int]

    @property
    def stage_id(self) -> str:
        kinds = "-".join(self.stage_kinds)
        return f"layer_{self.layer_index:04d}_stage_{self.stage_index:02d}__{kinds}"

    @property
    def total_stage_weights(self) -> int:
        return int(sum(int(value) for value in self.module_num_weights.values()))

    def to_identity_payload(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "layer_index": self.layer_index,
            "stage_index": self.stage_index,
            "stage_kinds": list(self.stage_kinds),
            "module_paths": list(self.module_paths),
            "module_num_weights": {key: int(value) for key, value in self.module_num_weights.items()},
        }


@dataclass
class ResumeState:
    committed_stage_ids: set[str]
    layer_reports: list[LayerReport]
    layer_results: dict[str, dict[str, Any]]
    remaining_budget: float
    remaining_weights: int
    quantization_anomalies: list[str]
    resumed: bool
    restored_stage_count: int


def _canonicalize_config(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            cleaned[key] = _canonicalize_config(value)
        elif isinstance(value, list):
            cleaned[key] = [_canonicalize_config(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _config_digest(payload: dict[str, Any]) -> str:
    canonical = _canonicalize_config(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent_dir(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)
    return target


def _atomic_torch_save(path: str | Path, payload: Any) -> Path:
    target = ensure_parent_dir(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def _atomic_safetensors_save(path: str | Path, payload: dict[str, torch.Tensor]) -> Path:
    target = ensure_parent_dir(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    save_file(payload, str(tmp))
    os.replace(tmp, target)
    return target


def _serialize_layer_statistics(stats_map: dict[str, LayerStatistics]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key, stats in stats_map.items():
        payload[key] = {
            "sigma_x": stats.sigma_x.cpu(),
            "sigma_x_hat": None if stats.sigma_x_hat is None else stats.sigma_x_hat.cpu(),
            "sigma_x_x_hat": None if stats.sigma_x_x_hat is None else stats.sigma_x_x_hat.cpu(),
            "sigma_x_weighted": None if stats.sigma_x_weighted is None else stats.sigma_x_weighted.cpu(),
            "sigma_x_hat_weighted": None if stats.sigma_x_hat_weighted is None else stats.sigma_x_hat_weighted.cpu(),
            "sigma_x_x_hat_weighted": None if stats.sigma_x_x_hat_weighted is None else stats.sigma_x_x_hat_weighted.cpu(),
            "sigma_delta_x_hat": None if stats.sigma_delta_x_hat is None else stats.sigma_delta_x_hat.cpu(),
            "variances": None if stats.variances is None else stats.variances.cpu(),
        }
    return payload


def _deserialize_layer_statistics(payload: dict[str, dict[str, Any]]) -> dict[str, LayerStatistics]:
    stats_map: dict[str, LayerStatistics] = {}
    for key, value in payload.items():
        stats_map[key] = LayerStatistics(
            sigma_x=value["sigma_x"],
            sigma_x_hat=value.get("sigma_x_hat"),
            sigma_x_x_hat=value.get("sigma_x_x_hat"),
            sigma_x_weighted=value.get("sigma_x_weighted"),
            sigma_x_hat_weighted=value.get("sigma_x_hat_weighted"),
            sigma_x_x_hat_weighted=value.get("sigma_x_x_hat_weighted"),
            sigma_delta_x_hat=value.get("sigma_delta_x_hat"),
            variances=value.get("variances"),
        )
    return stats_map


def _serialize_accumulators(accumulators: dict[str, dict[str, SecondMomentAccumulator]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        module_path: {name: accumulator.state_dict() for name, accumulator in module_accumulators.items()}
        for module_path, module_accumulators in accumulators.items()
    }


def _deserialize_accumulators(payload: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, SecondMomentAccumulator]]:
    return {
        module_path: {
            name: SecondMomentAccumulator.from_state_dict(accumulator_state)
            for name, accumulator_state in module_accumulators.items()
        }
        for module_path, module_accumulators in payload.items()
    }


class CheckpointManager:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        run_name: str,
        config: CheckpointConfig,
        model_config: dict[str, Any],
        quant_config: dict[str, Any],
        eval_config: dict[str, Any],
        report_metadata: dict[str, Any],
        total_weights: int,
        stage_plan: list[StagePlanEntry],
    ) -> None:
        self.run_name = run_name
        self.config = config
        self.model_config = model_config
        self.quant_config = quant_config
        self.eval_config = eval_config
        self.report_metadata = report_metadata
        self.total_weights = int(total_weights)
        self.stage_plan = stage_plan
        self.stage_plan_by_id = {entry.stage_id: entry for entry in stage_plan}
        self.root = repo_path(config.dir, run_name)
        self.manifest_path = self.root / "manifest.json"
        self.stages_root = self.root / "stages"
        self._manifest: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def initialize_or_validate(self) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            if self.config.resume == "never":
                raise RuntimeError(
                    f"Checkpoint directory already exists for run {self.run_name}; resume=never refuses to reuse {self.manifest_path}"
                )
            manifest = load_json(self.manifest_path)
            self._validate_manifest(manifest)
            self._manifest = manifest
            return

        if self.config.resume == "require":
            raise RuntimeError(f"resume=require but no checkpoint manifest exists at {self.manifest_path}")

        manifest = self._build_manifest()
        _atomic_write_json(self.manifest_path, manifest)
        self._manifest = manifest

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_name": self.run_name,
            "git_commit": self.report_metadata["git_commit"],
            "model_id": self.report_metadata["model_id"],
            "model_revision": self.report_metadata.get("model_revision"),
            "tokenizer_id": self.report_metadata["tokenizer_id"],
            "tokenizer_revision": self.report_metadata.get("tokenizer_revision"),
            "model_config_digest": _config_digest(self.model_config),
            "quant_config_path": self.quant_config.get("_config_path", ""),
            "quant_config_digest": _config_digest(self.quant_config),
            "eval_config_path": self.eval_config.get("_config_path", ""),
            "eval_config_digest": _config_digest(self.eval_config),
            "sequence_length": int(self.report_metadata["sequence_length"]),
            "calibration_sequences": int(self.report_metadata["calibration_sequences"]),
            "calibration_split": self.report_metadata.get("calibration_split", "train"),
            "target_global_bitwidth": float(self.quant_config["target_global_bitwidth"]),
            "total_weights": self.total_weights,
            "stage_plan": [entry.to_identity_payload() for entry in self.stage_plan],
            "stage_plan_digest": self._stage_plan_digest(self.stage_plan),
            "completed_stage_ids": [],
            "state": "initialized",
        }

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        if int(manifest.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported checkpoint schema version in {self.manifest_path}")
        if manifest.get("run_name") != self.run_name:
            raise RuntimeError(f"Checkpoint run_name mismatch in {self.manifest_path}")
        if manifest.get("model_id") != self.report_metadata["model_id"]:
            raise RuntimeError("Checkpoint model_id mismatch")
        if manifest.get("model_revision") != self.report_metadata.get("model_revision"):
            raise RuntimeError("Checkpoint model_revision mismatch")
        if manifest.get("tokenizer_id") != self.report_metadata["tokenizer_id"]:
            raise RuntimeError("Checkpoint tokenizer_id mismatch")
        if manifest.get("tokenizer_revision") != self.report_metadata.get("tokenizer_revision"):
            raise RuntimeError("Checkpoint tokenizer_revision mismatch")
        if manifest.get("model_config_digest") != _config_digest(self.model_config):
            raise RuntimeError("Checkpoint model config digest mismatch")
        if manifest.get("quant_config_digest") != _config_digest(self.quant_config):
            raise RuntimeError("Checkpoint quant config digest mismatch")
        if manifest.get("eval_config_digest") != _config_digest(self.eval_config):
            raise RuntimeError("Checkpoint eval config digest mismatch")
        if int(manifest.get("sequence_length", -1)) != int(self.report_metadata["sequence_length"]):
            raise RuntimeError("Checkpoint sequence length mismatch")
        if int(manifest.get("calibration_sequences", -1)) != int(self.report_metadata["calibration_sequences"]):
            raise RuntimeError("Checkpoint calibration sequence count mismatch")
        if manifest.get("calibration_split") != self.report_metadata.get("calibration_split", "train"):
            raise RuntimeError("Checkpoint calibration split mismatch")
        if self.config.strict_git_commit and manifest.get("git_commit") != self.report_metadata["git_commit"]:
            raise RuntimeError("Checkpoint git commit mismatch under strict_git_commit=true")
        if int(manifest.get("total_weights", -1)) != self.total_weights:
            raise RuntimeError("Checkpoint total_weights mismatch")
        if manifest.get("stage_plan_digest") != self._stage_plan_digest(self.stage_plan):
            raise RuntimeError("Checkpoint stage plan mismatch")

    @staticmethod
    def _stage_plan_digest(stage_plan: list[StagePlanEntry]) -> str:
        payload = [entry.to_identity_payload() for entry in stage_plan]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _stage_dir(self, stage: StagePlanEntry) -> Path:
        return self.stages_root / stage.stage_id

    def _stage_identity_payload(self, stage: StagePlanEntry) -> dict[str, Any]:
        return {
            **stage.to_identity_payload(),
            "run_name": self.run_name,
            "model_id": self.report_metadata["model_id"],
            "model_revision": self.report_metadata.get("model_revision"),
            "tokenizer_id": self.report_metadata["tokenizer_id"],
            "tokenizer_revision": self.report_metadata.get("tokenizer_revision"),
            "quant_config_digest": _config_digest(self.quant_config),
            "eval_config_digest": _config_digest(self.eval_config),
            "sequence_length": int(self.report_metadata["sequence_length"]),
            "calibration_sequences": int(self.report_metadata["calibration_sequences"]),
            "calibration_split": self.report_metadata.get("calibration_split", "train"),
            "reference_stats": bool(self.quant_config.get("reference_stats", False)),
            "calibration_batch_size": int(self.quant_config.get("calibration", {}).get("batch_size", 1)),
        }

    def _write_stage_identity(self, stage: StagePlanEntry) -> None:
        _atomic_write_json(self._stage_dir(stage) / "stage_identity.json", self._stage_identity_payload(stage))

    def _validate_stage_identity(self, stage: StagePlanEntry) -> None:
        identity_path = self._stage_dir(stage) / "stage_identity.json"
        if not identity_path.exists():
            raise RuntimeError(f"Missing stage identity file for checkpoint stage {stage.stage_id}")
        payload = load_json(identity_path)
        expected = self._stage_identity_payload(stage)
        if payload != expected:
            raise RuntimeError(f"Checkpoint stage identity mismatch for {stage.stage_id}")

    def restore_committed_progress(self, model) -> ResumeState:
        if not self.enabled:
            return ResumeState(
                committed_stage_ids=set(),
                layer_reports=[],
                layer_results={},
                remaining_budget=float(self.quant_config["target_global_bitwidth"]) * self.total_weights,
                remaining_weights=self.total_weights,
                quantization_anomalies=[],
                resumed=False,
                restored_stage_count=0,
            )

        manifest = self._manifest if self._manifest is not None else load_json(self.manifest_path)
        committed_stage_ids = set(str(item) for item in manifest.get("completed_stage_ids", []))
        layer_reports: list[LayerReport] = []
        layer_results: dict[str, dict[str, Any]] = {}
        quantization_anomalies: list[str] = []
        remaining_budget = float(self.quant_config["target_global_bitwidth"]) * self.total_weights
        remaining_weights = self.total_weights
        restored_stage_count = 0

        for stage in self.stage_plan:
            if stage.stage_id not in committed_stage_ids:
                break
            stage_dir = self._stage_dir(stage)
            commit_path = stage_dir / "commit.json"
            if not commit_path.exists():
                raise RuntimeError(f"Checkpoint manifest claims committed stage {stage.stage_id}, but {commit_path} is missing")
            self._validate_stage_identity(stage)
            weight_path = stage_dir / "weights.safetensors"
            stage_result_path = stage_dir / "stage_result.json"
            if not weight_path.exists() or not stage_result_path.exists():
                raise RuntimeError(f"Checkpoint stage {stage.stage_id} is missing durable payload files")
            stage_weights = load_file(str(weight_path), device="cpu")
            for module_path in stage.module_paths:
                if module_path not in stage_weights:
                    raise RuntimeError(f"Checkpoint stage {stage.stage_id} is missing weight tensor for {module_path}")
                module = model.get_submodule(module_path)
                module.weight.data.copy_(stage_weights[module_path].to(module.weight.device, dtype=module.weight.dtype))
            stage_result = load_json(stage_result_path)
            for payload in stage_result["layer_reports"]:
                layer_reports.append(LayerReport(**payload))
            for module_path in stage.module_paths:
                layer_results[module_path] = stage_result["layer_results"][module_path]
            quantization_anomalies.extend(stage_result.get("quantization_anomalies", []))
            remaining_budget = float(stage_result["post_stage_remaining_budget"])
            remaining_weights = int(stage_result["post_stage_remaining_weights"])
            restored_stage_count += 1

        resumed = restored_stage_count > 0
        return ResumeState(
            committed_stage_ids=committed_stage_ids,
            layer_reports=layer_reports,
            layer_results=layer_results,
            remaining_budget=remaining_budget,
            remaining_weights=remaining_weights,
            quantization_anomalies=quantization_anomalies,
            resumed=resumed,
            restored_stage_count=restored_stage_count,
        )

    def load_collection_state(self, stage: StagePlanEntry) -> tuple[dict[str, dict[str, SecondMomentAccumulator]], int] | None:
        if not self.enabled:
            return None
        stage_dir = self._stage_dir(stage)
        progress_path = stage_dir / "collection_progress.json"
        accumulators_path = stage_dir / "accumulators.pt"
        if not progress_path.exists() or not accumulators_path.exists():
            return None
        self._validate_stage_identity(stage)
        progress = load_json(progress_path)
        payload = torch.load(accumulators_path, map_location="cpu", weights_only=False)
        return _deserialize_accumulators(payload["accumulators"]), int(progress["next_batch_index"])

    def save_collection_state(
        self,
        stage: StagePlanEntry,
        *,
        accumulators: dict[str, dict[str, SecondMomentAccumulator]],
        next_batch_index: int,
    ) -> None:
        if not self.enabled:
            return
        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._write_stage_identity(stage)
        _atomic_torch_save(
            stage_dir / "accumulators.pt",
            {
                "accumulators": _serialize_accumulators(accumulators),
            },
        )
        _atomic_write_json(
            stage_dir / "collection_progress.json",
            {
                "stage_id": stage.stage_id,
                "next_batch_index": int(next_batch_index),
            },
        )
        self._update_manifest(state="collecting", current_stage_id=stage.stage_id)

    def load_stage_stats(self, stage: StagePlanEntry) -> dict[str, LayerStatistics] | None:
        if not self.enabled:
            return None
        stage_dir = self._stage_dir(stage)
        stats_path = stage_dir / "stats.pt"
        ready_path = stage_dir / "stage_ready.json"
        if not stats_path.exists() or not ready_path.exists():
            return None
        self._validate_stage_identity(stage)
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        return _deserialize_layer_statistics(payload["stats"])

    def save_stage_stats(self, stage: StagePlanEntry, stats_map: dict[str, LayerStatistics]) -> None:
        if not self.enabled:
            return
        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._write_stage_identity(stage)
        _atomic_torch_save(
            stage_dir / "stats.pt",
            {
                "stats": _serialize_layer_statistics(stats_map),
            },
        )
        _atomic_write_json(stage_dir / "stage_ready.json", {"stage_id": stage.stage_id, "state": "stats_ready"})
        self._update_manifest(state="stats_ready", current_stage_id=stage.stage_id)

    def save_committed_stage(
        self,
        stage: StagePlanEntry,
        *,
        model,
        stage_layer_reports: list[LayerReport],
        stage_layer_results: dict[str, dict[str, Any]],
        pre_stage_remaining_budget: float,
        post_stage_remaining_budget: float,
        pre_stage_remaining_weights: int,
        post_stage_remaining_weights: int,
        quantization_anomalies: list[str],
    ) -> None:
        if not self.enabled:
            return
        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._write_stage_identity(stage)
        tensors = {
            module_path: model.get_submodule(module_path).weight.detach().cpu().contiguous()
            for module_path in stage.module_paths
        }
        _atomic_safetensors_save(stage_dir / "weights.safetensors", tensors)
        _atomic_write_json(
            stage_dir / "stage_result.json",
            {
                "stage_id": stage.stage_id,
                "layer_reports": [asdict(report) for report in stage_layer_reports],
                "layer_results": stage_layer_results,
                "pre_stage_remaining_budget": float(pre_stage_remaining_budget),
                "post_stage_remaining_budget": float(post_stage_remaining_budget),
                "pre_stage_remaining_weights": int(pre_stage_remaining_weights),
                "post_stage_remaining_weights": int(post_stage_remaining_weights),
                "quantization_anomalies": list(quantization_anomalies),
            },
        )
        _atomic_write_json(stage_dir / "commit.json", {"stage_id": stage.stage_id, "state": "stage_committed"})

        manifest = self._manifest if self._manifest is not None else load_json(self.manifest_path)
        committed_stage_ids = list(manifest.get("completed_stage_ids", []))
        if stage.stage_id not in committed_stage_ids:
            committed_stage_ids.append(stage.stage_id)
        self._update_manifest(
            completed_stage_ids=committed_stage_ids,
            state="stage_committed",
            current_stage_id=stage.stage_id,
        )

    def mark_completed(self) -> None:
        if not self.enabled:
            return
        self._update_manifest(state="completed", current_stage_id=None)

    def _update_manifest(self, **updates: Any) -> None:
        if not self.enabled:
            return
        manifest = dict(self._manifest if self._manifest is not None else load_json(self.manifest_path))
        manifest.update(updates)
        _atomic_write_json(self.manifest_path, manifest)
        self._manifest = manifest
