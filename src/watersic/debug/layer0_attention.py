from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from watersic.data.calibration import CalibrationConfig, load_calibration_dataset
from watersic.data.wikitext2 import load_wikitext2_blocks
from watersic.eval.runner import run_wikitext2_perplexity
from watersic.models.hooks import ModuleInputCollector
from watersic.models.registry import LinearModuleSpec, iter_linear_module_specs, load_model_and_tokenizer
from watersic.quant.watersic_layer import (
    LayerQuantizationConfig,
    LayerStatistics,
    PreparedLayerProblem,
    prepare_layer_problem,
    quantize_linear_layer,
)
from watersic.quant.watersic_model import collect_layer_statistics
from watersic.utils.io import save_json
from watersic.utils.path_guard import repo_path
from watersic.utils.runtime import git_commit_hash, utc_timestamp


ATTENTION_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class AttentionDebugStage:
    name: str
    label: str
    spacing_strategy: str
    use_lmmse: bool
    use_activation_drift: bool
    use_residual_correction: bool
    use_attention_weighting: bool
    use_adaptive_mixing: bool
    optimize_mixing: bool
    epsilon_qr: float
    epsilon_aw: float
    max_rescaler_iters: int


def _stage_from_dict(payload: dict[str, Any]) -> AttentionDebugStage:
    return AttentionDebugStage(
        name=str(payload["name"]),
        label=str(payload["label"]),
        spacing_strategy=str(payload.get("spacing_strategy", "watersic")),
        use_lmmse=bool(payload.get("use_lmmse", True)),
        use_activation_drift=bool(payload.get("use_activation_drift", True)),
        use_residual_correction=bool(payload.get("use_residual_correction", True)),
        use_attention_weighting=bool(payload.get("use_attention_weighting", True)),
        use_adaptive_mixing=bool(payload.get("use_adaptive_mixing", True)),
        optimize_mixing=bool(payload.get("optimize_mixing", False)),
        epsilon_qr=float(payload.get("epsilon_qr", 0.0)),
        epsilon_aw=float(payload.get("epsilon_aw", 0.0)),
        max_rescaler_iters=int(payload.get("max_rescaler_iters", 0)),
    )


def _attention_specs(model) -> dict[str, LinearModuleSpec]:
    specs = {
        spec.kind: spec
        for spec in iter_linear_module_specs(model)
        if spec.layer_index == 0 and spec.kind in ATTENTION_ORDER
    }
    missing = [kind for kind in ATTENTION_ORDER if kind not in specs]
    if missing:
        raise ValueError(f"Missing layer-0 attention modules: {missing}")
    return specs


def _copy_module_weight(dst_module, src_module) -> None:
    dst_module.weight.data.copy_(src_module.weight.data)
    if getattr(dst_module, "bias", None) is not None and getattr(src_module, "bias", None) is not None:
        dst_module.bias.data.copy_(src_module.bias.data)


def restore_attention_block_weights(model, reference_model, specs: dict[str, LinearModuleSpec]) -> None:
    for spec in specs.values():
        _copy_module_weight(model.get_submodule(spec.full_path), reference_model.get_submodule(spec.full_path))


def _maybe_empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _layer_config_from_stage(base_quant: dict[str, Any], stage: AttentionDebugStage) -> LayerQuantizationConfig:
    return LayerQuantizationConfig(
        target_rate=float(base_quant["target_bitwidth"]),
        damping=float(base_quant.get("damping", 1e-4)),
        binary_search_iterations=int(base_quant.get("binary_search_iterations", 30)),
        row_sample_fraction=float(base_quant.get("row_sample_fraction", 0.1)),
        golden_section_iterations=int(base_quant.get("golden_section_iterations", 15)),
        dead_feature_tau=float(base_quant.get("dead_feature_tau", 1e-3)),
        epsilon_qr=stage.epsilon_qr,
        epsilon_aw=stage.epsilon_aw,
        max_rescaler_iters=stage.max_rescaler_iters,
        rescaler_ridge=float(base_quant.get("rescaler_ridge", 1e-8)),
        seed=int(base_quant.get("seed", 0)),
        use_lmmse=stage.use_lmmse,
        use_activation_drift=stage.use_activation_drift,
        use_residual_correction=stage.use_residual_correction,
        residual_scale=float(base_quant.get("residual_scale", 1.0)),
        use_attention_weighting=stage.use_attention_weighting,
        use_adaptive_mixing=stage.use_adaptive_mixing,
        spacing_strategy=stage.spacing_strategy,
    )


def _relative_mse_components(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    numerator = float((cand - ref).square().sum().item())
    denominator = float(ref.square().sum().item())
    return numerator, denominator


@torch.no_grad()
def _collect_module_relative_output_mse(
    model,
    reference_model,
    dataset,
    specs: dict[str, LinearModuleSpec],
    *,
    device: torch.device,
    reference_device: torch.device,
) -> dict[str, Any]:
    module_paths = [specs[kind].full_path for kind in ATTENTION_ORDER]
    numerators = {path: 0.0 for path in module_paths}
    denominators = {path: 0.0 for path in module_paths}
    for batch in dataset.batches(1):
        with ModuleInputCollector(model, module_paths=module_paths, collect_outputs=True) as q_store:
            model(input_ids=batch.to(device), use_cache=False)
        with ModuleInputCollector(reference_model, module_paths=module_paths, collect_outputs=True) as ref_store:
            reference_model(input_ids=batch.to(reference_device), use_cache=False)
        for path in module_paths:
            num, den = _relative_mse_components(ref_store.outputs[path], q_store.outputs[path])
            numerators[path] += num
            denominators[path] += den
    by_kind = {
        kind: numerators[specs[kind].full_path] / max(denominators[specs[kind].full_path], 1e-12)
        for kind in ATTENTION_ORDER
    }
    total_num = sum(numerators.values())
    total_den = sum(denominators.values())
    return {
        "by_kind": by_kind,
        "aggregate": total_num / max(total_den, 1e-12),
    }


@torch.no_grad()
def _module_input_relative_mse(
    model,
    reference_model,
    dataset,
    module_path: str,
    *,
    device: torch.device,
    reference_device: torch.device,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for batch in dataset.batches(1):
        with ModuleInputCollector(model, module_paths=[module_path]) as q_store:
            model(input_ids=batch.to(device), use_cache=False)
        with ModuleInputCollector(reference_model, module_paths=[module_path]) as ref_store:
            reference_model(input_ids=batch.to(reference_device), use_cache=False)
        num, den = _relative_mse_components(ref_store.inputs[module_path], q_store.inputs[module_path])
        numerator += num
        denominator += den
    return numerator / max(denominator, 1e-12)


def _spectral_audit(moment: torch.Tensor) -> dict[str, float]:
    sym = 0.5 * (moment.to(torch.float64) + moment.to(torch.float64).transpose(0, 1))
    eigvals = torch.linalg.eigvalsh(sym.cpu())
    max_eig = float(eigvals.max().item())
    min_eig = float(eigvals.min().item())
    clipped_min = max(abs(min_eig), 1e-12)
    return {
        "fro_norm": float(torch.linalg.matrix_norm(sym, ord="fro").item()),
        "trace": float(torch.trace(sym).item()),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": max_eig / clipped_min,
    }


def _cross_audit(moment: torch.Tensor) -> dict[str, float]:
    mat = moment.to(torch.float64).cpu()
    return {
        "fro_norm": float(torch.linalg.matrix_norm(mat, ord="fro").item()),
        "max_abs": float(mat.abs().max().item()),
    }


def _problem_audit(problem: PreparedLayerProblem, stats: LayerStatistics, config: LayerQuantizationConfig, *, kind: str) -> dict[str, Any]:
    dead = problem.dead_features
    reference_delta_norm = 0.0
    if stats.sigma_x_hat is not None:
        reference_delta_norm = float((stats.sigma_x - stats.sigma_x_hat).to(torch.float64).norm().item())
    return {
        "kind": kind,
        "reference_stats": {
            "activation_drift_enabled": config.use_activation_drift,
            "reference_stats_effective": config.use_activation_drift and stats.sigma_x_hat is not None and reference_delta_norm > 0.0,
            "reference_stats_bypassed": not config.use_activation_drift,
            "sigma_x_minus_sigma_x_hat_fro_norm": reference_delta_norm,
        },
        "covariances": {
            "sigma_x": _spectral_audit(problem.sigma_x),
            "sigma_x_hat": _spectral_audit(problem.sigma_x_hat),
            "sigma_cross": _cross_audit(problem.sigma_cross),
        },
        "damping": {
            "configured": float(config.damping),
            "applied": float(problem.applied_damping),
            "placement": "Applied to the final selected Sigma_Xhat moment immediately before Cholesky factorization.",
        },
        "dead_features": {
            "original_dim": int(dead.keep_mask.numel()),
            "kept_dim": int(dead.keep_indices.numel()),
            "pruned_dim": int((~dead.keep_mask).sum().item()),
            "threshold": float(dead.threshold),
            "keep_mask": [bool(x) for x in dead.keep_mask.tolist()],
            "pruned_indices": [int(x) for x in dead.dead_indices.tolist()],
        },
        "y_construction": {
            "formula": (
                "Y = (W Sigma_cross + Sigma_delta_x_hat_if_enabled) (L^T)^(-1)"
                if problem.compensation_matrix is not None
                else "Y = W Sigma_cross (L^T)^(-1)"
            ),
            "target_cross_fro_norm": float(torch.linalg.matrix_norm(problem.target_cross, ord="fro").item()),
            "target_y_fro_norm": float(torch.linalg.matrix_norm(problem.target_y, ord="fro").item()),
            "target_y_max_abs": float(problem.target_y.abs().max().item()),
        },
    }


def _quantization_audit(result, config: LayerQuantizationConfig) -> dict[str, Any]:
    columns = [
        {
            "index": column.index,
            "spacing": column.spacing,
            "quant_step": column.quant_step,
            "gamma": column.gamma,
            "raw_mean_abs_error": column.raw_mean_abs_error,
            "recursive_update_max_abs_error": column.recursive_update_max_abs_error,
        }
        for column in result.search.quantization.columns
    ]
    return {
        "alpha_i_formula": "alpha_i = c / L_ii" if config.spacing_strategy == "watersic" else "alpha_i = c",
        "gamma_i_formula": "<Y_i, q_i> / (||q_i||^2 + ridge)" if config.use_lmmse else "disabled (gamma_i = 1)",
        "spacing_strategy": config.spacing_strategy,
        "selected_c": float(result.search.selected_c),
        "search_bounds": [float(x) for x in result.search.bounds],
        "weighted_error": float(result.search.quantization.weighted_error),
        "spacings_summary": {
            "min": float(result.search.quantization.spacings.min().item()),
            "max": float(result.search.quantization.spacings.max().item()),
        },
        "gammas_summary": {
            "min": float(result.search.quantization.gammas.min().item()),
            "max": float(result.search.quantization.gammas.max().item()),
        },
        "recursive_update_max_abs_error": max(column["recursive_update_max_abs_error"] for column in columns) if columns else 0.0,
        "columns": columns,
    }


def _module_summary(kind: str, spec: LinearModuleSpec, stage: AttentionDebugStage, config: LayerQuantizationConfig, stats: LayerStatistics, problem: PreparedLayerProblem, result) -> dict[str, Any]:
    weight = result.quantized_weight.to(torch.float64)
    return {
        "name": spec.full_path,
        "kind": kind,
        "stage": stage.label,
        "bitrate": {
            "target": float(config.target_rate),
            "achieved": float(result.bitrate.final_effective_average_bitwidth),
            "raw": float(result.bitrate.raw_average_bitwidth),
            "entropy": float(result.bitrate.entropy_average_bitwidth),
            "huffman": float(result.bitrate.huffman_average_bitwidth),
            "side_information": float(result.bitrate.side_information_average_bitwidth),
        },
        "problem": _problem_audit(problem, stats, config, kind=kind),
        "quantization": _quantization_audit(result, config),
        "rescalers": {
            "enabled": bool(config.max_rescaler_iters > 0),
            "iterations": int(result.rescalers.iterations),
            "initial_objective": float(result.rescalers.initial_objective),
            "final_objective": float(result.rescalers.final_objective),
        },
        "weight_norms": {
            "quantized_fro_norm": float(torch.linalg.matrix_norm(weight, ord="fro").item()),
        },
    }


def _module_weight_relative_mse(reference_model, model, spec: LinearModuleSpec) -> float:
    reference = reference_model.get_submodule(spec.full_path).weight.detach().to(torch.float64).cpu()
    current = model.get_submodule(spec.full_path).weight.detach().to(torch.float64).cpu()
    num, den = _relative_mse_components(reference, current)
    return num / max(den, 1e-12)


def _golden_section_search(
    objective: Callable[[float], float],
    *,
    iterations: int,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float, list[dict[str, float]]]:
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a = float(lower)
    b = float(upper)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = objective(c)
    fd = objective(d)
    history = [
        {"x": c, "objective": fc},
        {"x": d, "objective": fd},
    ]
    for _ in range(iterations):
        if fc <= fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = objective(c)
            history.append({"x": c, "objective": fc})
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = objective(d)
            history.append({"x": d, "objective": fd})
    if fc <= fd:
        return c, fc, history
    return d, fd, history


def _quantize_qkv_only(
    model,
    reference_model,
    specs: dict[str, LinearModuleSpec],
    stats_map: dict[str, LayerStatistics],
    config: LayerQuantizationConfig,
) -> None:
    restore_attention_block_weights(model, reference_model, specs)
    for kind in ATTENTION_ORDER[:3]:
        spec = specs[kind]
        module = model.get_submodule(spec.full_path)
        result = quantize_linear_layer(module.weight.detach().cpu(), stats_map[spec.full_path], config, kind=kind)
        module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))


def _optimize_mixing(
    model,
    reference_model,
    specs: dict[str, LinearModuleSpec],
    qkv_stats: dict[str, LayerStatistics],
    calibration_dataset,
    config: LayerQuantizationConfig,
    *,
    device: torch.device,
    reference_device: torch.device,
    logger,
) -> tuple[float, float, dict[str, Any]]:
    if not config.use_attention_weighting or not config.use_adaptive_mixing:
        return config.epsilon_qr, config.epsilon_aw, {"enabled": False}

    o_path = specs["o_proj"].full_path
    search_iterations = config.golden_section_iterations

    def objective_for(epsilon_qr: float, epsilon_aw: float) -> float:
        candidate_config = replace(config, epsilon_qr=epsilon_qr, epsilon_aw=epsilon_aw)
        _quantize_qkv_only(model, reference_model, specs, qkv_stats, candidate_config)
        mse = _module_input_relative_mse(
            model,
            reference_model,
            calibration_dataset,
            o_path,
            device=device,
            reference_device=reference_device,
        )
        _maybe_empty_cuda_cache()
        return mse

    best_qr, best_qr_objective, qr_history = _golden_section_search(
        lambda value: objective_for(value, config.epsilon_aw),
        iterations=search_iterations,
    )
    logger.info("Selected epsilon_qr=%.6f with wo-input relative MSE %.6e", best_qr, best_qr_objective)

    best_aw, best_aw_objective, aw_history = _golden_section_search(
        lambda value: objective_for(best_qr, value),
        iterations=search_iterations,
    )
    logger.info("Selected epsilon_aw=%.6f with wo-input relative MSE %.6e", best_aw, best_aw_objective)
    restore_attention_block_weights(model, reference_model, specs)
    return best_qr, best_aw, {
        "enabled": True,
        "epsilon_qr": best_qr,
        "epsilon_aw": best_aw,
        "epsilon_qr_objective": best_qr_objective,
        "epsilon_aw_objective": best_aw_objective,
        "epsilon_qr_history": qr_history,
        "epsilon_aw_history": aw_history,
        "objective": "Relative MSE at the layer-0 o_proj input after quantizing q_proj/k_proj/v_proj.",
    }


def _stage_output_dir(run_name: str) -> Path:
    return repo_path("outputs", "reports", run_name)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Layer-0 Attention Debug Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Model: `{report['model_id']}`",
        f"- Baseline small-eval PPL: `{report['baseline_small_eval_ppl']:.4f}`",
        "",
        "| Stage | Small-Eval PPL | Block Rel MSE | q | k | v | o | Entropy bw | Huffman bw |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in report["stages"]:
        rel = stage["relative_mse"]
        lines.append(
            f"| {stage['label']} | {stage['small_eval_ppl']:.4f} | {rel['aggregate']:.6e} | "
            f"{rel['by_kind']['q_proj']:.6e} | {rel['by_kind']['k_proj']:.6e} | {rel['by_kind']['v_proj']:.6e} | {rel['by_kind']['o_proj']:.6e} | "
            f"{stage['bitrate']['entropy']:.4f} | {stage['bitrate']['huffman']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_layer0_attention_debug(
    model_config: dict[str, Any],
    debug_config: dict[str, Any],
    *,
    device: torch.device,
    logger,
    reference_device: torch.device | None = None,
) -> dict[str, Any]:
    run_name = str(debug_config["run_name"])
    stage_dir = _stage_output_dir(run_name)

    working_model, tokenizer = load_model_and_tokenizer(model_config)
    working_model.to(device)
    working_model.eval()

    if reference_device is None:
        reference_device = device
    reference_model, _ = load_model_and_tokenizer(model_config)
    reference_model.to(reference_device)
    reference_model.eval()

    calibration_dataset = load_calibration_dataset(
        tokenizer,
        CalibrationConfig(
            split=str(debug_config.get("calibration", {}).get("split", "train")),
            sequence_length=int(debug_config.get("calibration", {}).get("sequence_length", 2048)),
            num_sequences=int(debug_config.get("calibration", {}).get("num_sequences", 4)),
            batch_size=int(debug_config.get("calibration", {}).get("batch_size", 1)),
        ),
    )
    probe_cfg = debug_config.get("probe_eval", {})
    probe_dataset = load_wikitext2_blocks(
        tokenizer,
        split=str(probe_cfg.get("split", "test")),
        sequence_length=int(probe_cfg.get("sequence_length", 2048)),
        limit_sequences=int(probe_cfg.get("num_sequences", 4)),
    )

    specs = _attention_specs(working_model)
    logger.info("Running baseline small-eval perplexity for layer-0 attention debug")
    baseline_small_eval = run_wikitext2_perplexity(
        reference_model,
        probe_dataset,
        device=str(reference_device),
        batch_size=int(probe_cfg.get("batch_size", 1)),
    ).perplexity

    report: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "run_name": run_name,
        "model_id": model_config["model_id"],
        "model_revision": model_config.get("model_revision"),
        "baseline_small_eval_ppl": baseline_small_eval,
        "stages": [],
    }

    base_quant = debug_config["quant"]
    stage_specs = [_stage_from_dict(stage_payload) for stage_payload in debug_config["stages"]]

    for stage in stage_specs:
        logger.info("Starting debug stage %s", stage.label)
        restore_attention_block_weights(working_model, reference_model, specs)
        _maybe_empty_cuda_cache()

        qkv_specs = [specs["q_proj"], specs["k_proj"], specs["v_proj"]]
        qkv_stats = collect_layer_statistics(
            working_model,
            calibration_dataset,
            qkv_specs,
            device=device,
            logger=logger,
            calibration_batch_size=int(debug_config.get("calibration", {}).get("batch_size", 1)),
            reference_model=reference_model,
            reference_device=reference_device,
        )

        layer_config = _layer_config_from_stage(base_quant, stage)
        mixing_audit = {"enabled": False}
        if stage.optimize_mixing:
            epsilon_qr, epsilon_aw, mixing_audit = _optimize_mixing(
                working_model,
                reference_model,
                specs,
                qkv_stats,
                calibration_dataset,
                layer_config,
                device=device,
                reference_device=reference_device,
                logger=logger,
            )
            layer_config = replace(layer_config, epsilon_qr=epsilon_qr, epsilon_aw=epsilon_aw)
            restore_attention_block_weights(working_model, reference_model, specs)

        module_results: dict[str, Any] = {}
        for kind in ATTENTION_ORDER[:3]:
            spec = specs[kind]
            module = working_model.get_submodule(spec.full_path)
            stats = qkv_stats[spec.full_path]
            problem = prepare_layer_problem(module.weight.detach().cpu(), stats, layer_config, kind=kind)
            result = quantize_linear_layer(module.weight.detach().cpu(), stats, layer_config, kind=kind)
            module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
            module_results[kind] = _module_summary(kind, spec, stage, layer_config, stats, problem, result)
            module_results[kind]["weight_relative_mse"] = _module_weight_relative_mse(reference_model, working_model, spec)

        o_stats = collect_layer_statistics(
            working_model,
            calibration_dataset,
            [specs["o_proj"]],
            device=device,
            logger=logger,
            calibration_batch_size=int(debug_config.get("calibration", {}).get("batch_size", 1)),
            reference_model=reference_model,
            reference_device=reference_device,
        )
        o_spec = specs["o_proj"]
        o_module = working_model.get_submodule(o_spec.full_path)
        o_problem = prepare_layer_problem(o_module.weight.detach().cpu(), o_stats[o_spec.full_path], layer_config, kind="o_proj")
        o_result = quantize_linear_layer(o_module.weight.detach().cpu(), o_stats[o_spec.full_path], layer_config, kind="o_proj")
        o_module.weight.data.copy_(o_result.quantized_weight.to(o_module.weight.device, dtype=o_module.weight.dtype))
        module_results["o_proj"] = _module_summary("o_proj", o_spec, stage, layer_config, o_stats[o_spec.full_path], o_problem, o_result)
        module_results["o_proj"]["weight_relative_mse"] = _module_weight_relative_mse(reference_model, working_model, o_spec)

        relative_mse = _collect_module_relative_output_mse(
            working_model,
            reference_model,
            probe_dataset,
            specs,
            device=device,
            reference_device=reference_device,
        )
        small_eval_ppl = run_wikitext2_perplexity(
            working_model,
            probe_dataset,
            device=str(device),
            batch_size=int(probe_cfg.get("batch_size", 1)),
        ).perplexity

        total_weights = sum(working_model.get_submodule(specs[kind].full_path).weight.numel() for kind in ATTENTION_ORDER)
        entropy = sum(
            module_results[kind]["bitrate"]["entropy"] * working_model.get_submodule(specs[kind].full_path).weight.numel()
            for kind in ATTENTION_ORDER
        ) / max(total_weights, 1)
        huffman = sum(
            module_results[kind]["bitrate"]["huffman"] * working_model.get_submodule(specs[kind].full_path).weight.numel()
            for kind in ATTENTION_ORDER
        ) / max(total_weights, 1)

        stage_result = {
            "name": stage.name,
            "label": stage.label,
            "config": asdict(stage),
            "effective_layer_config": asdict(layer_config),
            "mixing_audit": mixing_audit,
            "small_eval_ppl": float(small_eval_ppl),
            "relative_mse": relative_mse,
            "bitrate": {
                "entropy": float(entropy),
                "huffman": float(huffman),
            },
            "modules": module_results,
        }
        save_json(stage_dir / f"{stage.name}.json", stage_result)
        save_json(stage_dir / f"{stage.name}_config.json", {"model_config": model_config, "debug_config": debug_config, "stage": stage_result["effective_layer_config"]})
        report["stages"].append(
            {
                "name": stage_result["name"],
                "label": stage_result["label"],
                "config": stage_result["config"],
                "effective_layer_config": stage_result["effective_layer_config"],
                "mixing_audit": stage_result["mixing_audit"],
                "small_eval_ppl": stage_result["small_eval_ppl"],
                "relative_mse": stage_result["relative_mse"],
                "bitrate": stage_result["bitrate"],
            }
        )
        logger.info("Completed debug stage %s with small-eval PPL %.4f", stage.label, small_eval_ppl)
        _maybe_empty_cuda_cache()

    report["summary_markdown_path"] = str(stage_dir / "summary.md")
    save_json(stage_dir / "summary.json", report)
    (stage_dir / "summary.md").write_text(_render_markdown(report), encoding="utf-8")
    return report
