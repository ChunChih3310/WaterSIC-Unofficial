from __future__ import annotations

import gc
from typing import Any

import torch

from watersic.data.calibration import CalibrationConfig, load_calibration_dataset
from watersic.data.wikitext2 import load_wikitext2_blocks
from watersic.eval.runner import run_wikitext2_perplexity
from watersic.models.hooks import ModuleInputCollector
from watersic.models.registry import LinearModuleSpec, group_specs_by_layer, iter_linear_module_specs, load_model_and_tokenizer
from watersic.quant.watersic_layer import LayerQuantizationConfig, LayerStatistics, prepare_layer_problem, quantize_linear_layer
from watersic.quant.watersic_model import _layer_stage_specs, collect_layer_statistics
from watersic.stats.covariance import SecondMomentAccumulator, damped_second_moment
from watersic.stats.residual import is_residual_target, legacy_residual_compensation_matrix
from watersic.utils.io import save_json
from watersic.utils.path_guard import repo_path
from watersic.utils.runtime import git_commit_hash, utc_timestamp


def _maybe_empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _copy_module_weight(dst_module, src_module) -> None:
    dst_module.weight.data.copy_(src_module.weight.data)
    if getattr(dst_module, "bias", None) is not None and getattr(src_module, "bias", None) is not None:
        dst_module.bias.data.copy_(src_module.bias.data)


def _relative_weight_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    numerator = float((cand - ref).square().sum().item())
    denominator = float(ref.square().sum().item())
    return numerator, denominator


def _module_weight_relative_mse(reference_model, model, spec: LinearModuleSpec) -> float:
    reference = reference_model.get_submodule(spec.full_path).weight.detach().to(torch.float64).cpu()
    current = model.get_submodule(spec.full_path).weight.detach().to(torch.float64).cpu()
    num, den = _relative_weight_error(reference, current)
    return num / max(den, 1e-12)


def _ordered_selected_specs(model, *, max_layers: int | None, max_modules: int | None) -> list[LinearModuleSpec]:
    specs = iter_linear_module_specs(model)
    grouped = group_specs_by_layer(specs)
    layer_indices = sorted(grouped)
    if max_layers is not None:
        layer_indices = layer_indices[:max_layers]

    ordered: list[LinearModuleSpec] = []
    for layer_index in layer_indices:
        for stage_specs in _layer_stage_specs(grouped[layer_index]):
            ordered.extend(stage_specs)
    if max_modules is not None:
        ordered = ordered[:max_modules]
    return ordered


def _build_layer_config(layer_cfg: dict[str, Any], *, target_rate: float, residual_scale: float, use_residual_correction: bool) -> LayerQuantizationConfig:
    return LayerQuantizationConfig(
        target_rate=target_rate,
        damping=float(layer_cfg.get("damping", 1e-4)),
        binary_search_iterations=int(layer_cfg.get("binary_search_iterations", 30)),
        row_sample_fraction=float(layer_cfg.get("row_sample_fraction", 0.1)),
        golden_section_iterations=int(layer_cfg.get("golden_section_iterations", 15)),
        dead_feature_tau=float(layer_cfg.get("dead_feature_tau", 1e-3)),
        epsilon_qr=float(layer_cfg.get("epsilon_qr", 0.0)),
        epsilon_aw=float(layer_cfg.get("epsilon_aw", 0.0)),
        max_rescaler_iters=int(layer_cfg.get("max_rescaler_iters", 0)),
        rescaler_ridge=float(layer_cfg.get("rescaler_ridge", 1e-8)),
        seed=int(layer_cfg.get("seed", 0)),
        use_lmmse=bool(layer_cfg.get("use_lmmse", True)),
        use_activation_drift=bool(layer_cfg.get("use_activation_drift", True)),
        use_residual_correction=use_residual_correction,
        residual_scale=float(residual_scale),
        use_attention_weighting=bool(layer_cfg.get("use_attention_weighting", True)),
        use_adaptive_mixing=bool(layer_cfg.get("use_adaptive_mixing", True)),
        spacing_strategy=str(layer_cfg.get("spacing_strategy", "watersic")),
    )


def _prefix_target_budget(working_model, ordered_specs: list[LinearModuleSpec], target_global_bitwidth: float) -> tuple[float, float]:
    total_weights = sum(working_model.get_submodule(spec.full_path).weight.numel() for spec in ordered_specs)
    return target_global_bitwidth * total_weights, float(total_weights)


def _quantize_prefix(
    working_model,
    reference_model,
    calibration_dataset,
    ordered_specs: list[LinearModuleSpec],
    *,
    prefix_layer_cfg: dict[str, Any],
    target_global_bitwidth: float,
    device: torch.device,
    reference_device: torch.device,
    calibration_batch_size: int,
    logger,
) -> tuple[LinearModuleSpec, float, list[dict[str, Any]]]:
    if not ordered_specs:
        raise ValueError("No ordered specs selected for residual debug")
    target_spec = ordered_specs[-1]
    prefix_specs = ordered_specs[:-1]
    remaining_budget, remaining_weights = _prefix_target_budget(working_model, ordered_specs, target_global_bitwidth)
    prefix_results: list[dict[str, Any]] = []

    grouped_prefix = group_specs_by_layer(prefix_specs)
    for layer_index in sorted(grouped_prefix):
        layer_specs = grouped_prefix[layer_index]
        for stage_specs in _layer_stage_specs(layer_specs):
            logger.info("Residual debug prefix collect layer %d stage %s", layer_index, [spec.kind for spec in stage_specs])
            stats_map = collect_layer_statistics(
                working_model,
                calibration_dataset,
                stage_specs,
                device=device,
                logger=logger,
                calibration_batch_size=calibration_batch_size,
                reference_model=reference_model,
                reference_device=reference_device,
            )
            for spec in stage_specs:
                module = working_model.get_submodule(spec.full_path)
                original_weight = module.weight.detach().cpu().to(torch.float64)
                num_weights = module.weight.numel()
                target_rate = remaining_budget / max(remaining_weights, 1.0)
                layer_config = _build_layer_config(
                    prefix_layer_cfg,
                    target_rate=target_rate,
                    residual_scale=float(prefix_layer_cfg.get("residual_scale", 1.0)),
                    use_residual_correction=bool(prefix_layer_cfg.get("use_residual_correction", False)),
                )
                result = quantize_linear_layer(original_weight, stats_map[spec.full_path], layer_config, kind=spec.kind)
                module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
                remaining_budget -= result.bitrate.final_effective_average_bitwidth * num_weights
                remaining_weights -= num_weights
                prefix_results.append(
                    {
                        "name": spec.full_path,
                        "kind": spec.kind,
                        "target_rate": target_rate,
                        "achieved_rate": result.bitrate.final_effective_average_bitwidth,
                        "reference_delta_norm": float(
                            (stats_map[spec.full_path].sigma_x - stats_map[spec.full_path].sigma_x_hat).to(torch.float64).norm().item()
                        ),
                        "relative_weight_mse": _module_weight_relative_mse(reference_model, working_model, spec),
                    }
                )
                logger.info(
                    "Residual debug prefix quantized %s at %.4f bits (achieved %.4f)",
                    spec.full_path,
                    target_rate,
                    result.bitrate.final_effective_average_bitwidth,
                )
                _maybe_empty_cuda_cache()

    target_rate = remaining_budget / max(remaining_weights, 1.0)
    return target_spec, float(target_rate), prefix_results


def _manual_collect_o_proj_statistics(
    model,
    reference_model,
    calibration_dataset,
    target_spec: LinearModuleSpec,
    *,
    device: torch.device,
    reference_device: torch.device,
    calibration_batch_size: int,
) -> tuple[LayerStatistics, dict[str, Any]]:
    module_path = target_spec.full_path
    layer_path = target_spec.layer_path
    module = model.get_submodule(module_path)
    hidden_dim = model.get_submodule(layer_path).input_layernorm.weight.numel()

    acc_sigma_x = SecondMomentAccumulator(module.in_features)
    acc_sigma_x_hat = SecondMomentAccumulator(module.in_features)
    acc_sigma_cross = SecondMomentAccumulator(module.in_features, module.in_features)
    acc_sigma_delta = SecondMomentAccumulator(hidden_dim, module.in_features)
    first_batch_audit: dict[str, Any] = {}

    for batch_index, batch in enumerate(calibration_dataset.batches(calibration_batch_size)):
        batch = batch.to(device)
        with torch.no_grad():
            with ModuleInputCollector(model, module_paths=[module_path], layer_paths=[layer_path]) as q_store:
                model(input_ids=batch, use_cache=False, output_attentions=False)
            with ModuleInputCollector(reference_model, module_paths=[module_path], layer_paths=[layer_path]) as ref_store:
                reference_model(input_ids=batch.to(reference_device), use_cache=False, output_attentions=False)

        q_input = q_store.inputs[module_path]
        ref_input = ref_store.inputs[module_path]
        q_layer_input = q_store.layer_inputs[layer_path]
        ref_layer_input = ref_store.layer_inputs[layer_path]
        delta = ref_layer_input - q_layer_input

        acc_sigma_x.update(ref_input)
        acc_sigma_x_hat.update(q_input)
        acc_sigma_cross.update(ref_input, q_input)
        acc_sigma_delta.update(delta, q_input)

        if batch_index == 0:
            first_batch_audit = {
                "delta_definition": "Delta = R - R_hat = reference_layer_input - quantized_layer_input",
                "layer_input_delta_fro_norm": float(torch.linalg.matrix_norm(delta.to(torch.float64), ord="fro").item()),
                "o_proj_input_delta_fro_norm": float(
                    torch.linalg.matrix_norm((ref_input - q_input).to(torch.float64), ord="fro").item()
                ),
            }

    manual_stats = LayerStatistics(
        sigma_x=acc_sigma_x.finalize(),
        sigma_x_hat=acc_sigma_x_hat.finalize(),
        sigma_x_x_hat=acc_sigma_cross.finalize(),
        sigma_delta_x_hat=acc_sigma_delta.finalize(),
        variances=torch.diag(acc_sigma_x_hat.finalize()),
    )
    return manual_stats, first_batch_audit


def _timing_audit(
    working_model,
    reference_model,
    ordered_specs: list[LinearModuleSpec],
    target_spec: LinearModuleSpec,
    official_stats: LayerStatistics,
    manual_stats: LayerStatistics,
    first_batch_audit: dict[str, Any],
) -> dict[str, Any]:
    qkv_specs = [spec for spec in ordered_specs if spec.layer_index == target_spec.layer_index and spec.kind in {"q_proj", "k_proj", "v_proj"}]
    qkv_weight_mse = {spec.kind: _module_weight_relative_mse(reference_model, working_model, spec) for spec in qkv_specs}
    target_weight_mse = _module_weight_relative_mse(reference_model, working_model, target_spec)
    sigma_delta_diff = (manual_stats.sigma_delta_x_hat - official_stats.sigma_delta_x_hat).to(torch.float64)
    sigma_delta_wrong_sign = (manual_stats.sigma_delta_x_hat + official_stats.sigma_delta_x_hat).to(torch.float64)
    sigma_x_hat_diff = (manual_stats.sigma_x_hat - official_stats.sigma_x_hat).to(torch.float64)

    return {
        "stage_timing": "same-layer, post-QKV, pre-o_proj",
        "residual_target_kind_check": bool(is_residual_target(target_spec.kind)),
        "qkv_weight_relative_mse_before_o_proj": qkv_weight_mse,
        "o_proj_weight_relative_mse_before_quantization": target_weight_mse,
        "manual_sigma_delta_match_fro_norm": float(torch.linalg.matrix_norm(sigma_delta_diff, ord="fro").item()),
        "manual_sigma_delta_wrong_sign_fro_norm": float(torch.linalg.matrix_norm(sigma_delta_wrong_sign, ord="fro").item()),
        "manual_sigma_x_hat_match_fro_norm": float(torch.linalg.matrix_norm(sigma_x_hat_diff, ord="fro").item()),
        **first_batch_audit,
    }


def _exact_condition_number(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.to(torch.float64).cpu())
    return float((singular_values.max() / torch.clamp(singular_values.min(), min=1e-12)).item())


def _legacy_formula_audit(problem, reduced_sigma_delta: torch.Tensor, *, legacy_ridge: float) -> dict[str, Any]:
    legacy_term = legacy_residual_compensation_matrix(problem.sigma_x_hat, reduced_sigma_delta, ridge=legacy_ridge)
    if legacy_term is None:
        return {"available": False}
    legacy_target_cross = problem.base_target_cross + legacy_term
    legacy_target_y = torch.linalg.solve_triangular(
        problem.cholesky,
        legacy_target_cross.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    return {
        "available": True,
        "legacy_residual_term_fro_norm": float(torch.linalg.matrix_norm(legacy_term, ord="fro").item()),
        "legacy_residual_to_base_ratio": float(
            torch.linalg.matrix_norm(legacy_term, ord="fro").item()
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        ),
        "legacy_target_cross_fro_norm": float(torch.linalg.matrix_norm(legacy_target_cross, ord="fro").item()),
        "legacy_target_y_fro_norm": float(torch.linalg.matrix_norm(legacy_target_y, ord="fro").item()),
        "legacy_target_y_max_abs": float(legacy_target_y.abs().max().item()),
    }


def _residual_point(
    working_model,
    reference_model,
    target_spec: LinearModuleSpec,
    official_stats: LayerStatistics,
    target_rate: float,
    probe_dataset,
    *,
    layer_cfg: dict[str, Any],
    residual_scale: float,
    device: torch.device,
    reference_device: torch.device,
    legacy_ridge: float,
) -> dict[str, Any]:
    target_module = working_model.get_submodule(target_spec.full_path)
    reference_module = reference_model.get_submodule(target_spec.full_path)
    _copy_module_weight(target_module, reference_module)
    weight = target_module.weight.detach().cpu().to(torch.float64)
    layer_config = _build_layer_config(
        layer_cfg,
        target_rate=target_rate,
        residual_scale=residual_scale,
        use_residual_correction=True,
    )
    problem = prepare_layer_problem(weight, official_stats, layer_config, kind=target_spec.kind)
    reduced_sigma_delta = official_stats.sigma_delta_x_hat.to(torch.float64)[:, problem.dead_features.keep_indices]
    damped_h = damped_second_moment(problem.sigma_x_hat, problem.applied_damping)
    result = quantize_linear_layer(weight, official_stats, layer_config, kind=target_spec.kind)
    target_module.weight.data.copy_(result.quantized_weight.to(target_module.weight.device, dtype=target_module.weight.dtype))
    quantized_eval = run_wikitext2_perplexity(
        working_model,
        probe_dataset,
        device=str(device),
        batch_size=1,
    )
    rel_num, rel_den = _relative_weight_error(weight, result.quantized_weight)
    return {
        "residual_scale": residual_scale,
        "small_eval_ppl": quantized_eval.perplexity,
        "relative_weight_mse": rel_num / max(rel_den, 1e-12),
        "max_abs_weight_error": float((result.quantized_weight.to(torch.float64) - weight).abs().max().item()),
        "base_cross_fro_norm": float(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item()),
        "sigma_delta_x_hat_fro_norm": float(torch.linalg.matrix_norm(reduced_sigma_delta, ord="fro").item()),
        "combined_cross_fro_norm": float(torch.linalg.matrix_norm(problem.target_cross, ord="fro").item()),
        "residual_to_base_ratio": float(
            torch.linalg.matrix_norm(reduced_sigma_delta, ord="fro").item()
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        ),
        "scaled_residual_to_base_ratio": float(
            (residual_scale * torch.linalg.matrix_norm(reduced_sigma_delta, ord="fro").item())
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        ),
        "combined_to_base_ratio": float(
            torch.linalg.matrix_norm(problem.target_cross, ord="fro").item()
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        ),
        "sigma_x_hat_condition_number": _exact_condition_number(problem.sigma_x_hat),
        "h_after_damping_condition_number": _exact_condition_number(damped_h),
        "dead_features": {
            "before": int(official_stats.sigma_x_hat.shape[0]),
            "after": int(problem.dead_features.keep_indices.numel()),
            "pruned": int((~problem.dead_features.keep_mask).sum().item()),
        },
        "cholesky_diag_min": float(torch.diag(problem.cholesky).min().item()),
        "cholesky_diag_max": float(torch.diag(problem.cholesky).max().item()),
        "target_y": {
            "fro_norm": float(torch.linalg.matrix_norm(problem.target_y, ord="fro").item()),
            "min": float(problem.target_y.min().item()),
            "max": float(problem.target_y.max().item()),
            "max_abs": float(problem.target_y.abs().max().item()),
        },
        "quantization": {
            "alpha_min": float(result.diagnostics["alpha_min"]),
            "alpha_max": float(result.diagnostics["alpha_max"]),
            "gamma_min": float(result.diagnostics["gamma_min"]),
            "gamma_max": float(result.diagnostics["gamma_max"]),
            "entropy_rate": float(result.bitrate.entropy_average_bitwidth),
            "huffman_rate": float(result.bitrate.huffman_average_bitwidth),
        },
        "legacy_formula_audit": _legacy_formula_audit(problem, reduced_sigma_delta, legacy_ridge=legacy_ridge),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    config_audit = report["config_audit"]
    timing = report["timing_audit"]
    lines = [
        "# Layer1 o_proj Residual Debug Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Model: `{report['model_id']}`",
        f"- Target module: `{report['target']['name']}`",
        f"- Baseline small-eval PPL: `{report['baseline_small_eval_ppl']:.4f}`",
        f"- Target rate at o_proj: `{report['target']['target_rate']:.4f}`",
        f"- Reference stats requested: `{config_audit['reference_stats_requested']}`",
        f"- Reference stats used: `{config_audit['reference_stats_used']}`",
        f"- Rescalers enabled: `{config_audit['rescalers_enabled']}`",
        "",
        "## Config Audit",
        "",
        f"- Calibration: `{config_audit['calibration_split']}` / `{config_audit['calibration_num_sequences']}` sequences / len `{config_audit['calibration_sequence_length']}` / batch `{config_audit['calibration_batch_size']}`",
        f"- Probe eval: `{config_audit['probe_split']}` / `{config_audit['probe_num_sequences']}` sequences / len `{config_audit['probe_sequence_length']}` / batch `{config_audit['probe_batch_size']}`",
        f"- Residual scales: `{config_audit['residual_scales']}`",
        f"- Legacy ridge for audit: `{config_audit['legacy_ridge']}`",
        "",
        "## Timing Audit",
        "",
        f"- Stage timing: `{timing['stage_timing']}`",
        f"- Delta definition: `{timing['delta_definition']}`",
        f"- Residual target kind check: `{timing['residual_target_kind_check']}`",
        f"- qkv weight MSE before o_proj: `{timing['qkv_weight_relative_mse_before_o_proj']}`",
        f"- o_proj weight MSE before quantization: `{timing['o_proj_weight_relative_mse_before_quantization']:.6e}`",
        f"- Manual Sigma_Delta match error: `{timing['manual_sigma_delta_match_fro_norm']:.6e}`",
        f"- Wrong-sign mismatch error: `{timing['manual_sigma_delta_wrong_sign_fro_norm']:.6e}`",
        f"- Manual Sigma_Xhat match error: `{timing['manual_sigma_x_hat_match_fro_norm']:.6e}`",
        f"- Layer-input delta ||R-Rhat||_F (first batch): `{timing['layer_input_delta_fro_norm']:.6e}`",
        f"- o_proj-input delta ||X-Xhat||_F (first batch): `{timing['o_proj_input_delta_fro_norm']:.6e}`",
        "",
        "## Sweep",
        "",
        "| Residual Scale | Small-Eval PPL | Rel Weight MSE | ||WΣ|| | ||ΣΔ|| | ||sum|| | ||ΣΔ||/||WΣ|| | H cond | max |Y| | legacy max |Y| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in report["residual_scale_sweep"]:
        legacy_max_abs = point["legacy_formula_audit"]["legacy_target_y_max_abs"] if point["legacy_formula_audit"]["available"] else 0.0
        lines.append(
            f"| {point['residual_scale']:.2f} | {point['small_eval_ppl']:.4f} | {point['relative_weight_mse']:.6e} | "
            f"{point['base_cross_fro_norm']:.6e} | {point['sigma_delta_x_hat_fro_norm']:.6e} | "
            f"{point['combined_cross_fro_norm']:.6e} | {point['scaled_residual_to_base_ratio']:.6e} | "
            f"{point['h_after_damping_condition_number']:.6e} | {point['target_y']['max_abs']:.6e} | {legacy_max_abs:.6e} |"
        )
    return "\n".join(lines) + "\n"


def run_o_proj_residual_debug(
    model_config: dict[str, Any],
    debug_config: dict[str, Any],
    *,
    device: torch.device,
    logger,
    reference_device: torch.device | None = None,
) -> dict[str, Any]:
    run_name = str(debug_config["run_name"])
    output_dir = repo_path("outputs", "reports", run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_model, tokenizer = load_model_and_tokenizer(model_config)
    working_model.to(device)
    working_model.eval()
    if reference_device is None:
        reference_device = device
    reference_model, _ = load_model_and_tokenizer(model_config)
    reference_model.to(reference_device)
    reference_model.eval()

    calibration_cfg = debug_config.get("calibration", {})
    calibration_dataset = load_calibration_dataset(
        tokenizer,
        CalibrationConfig(
            split=str(calibration_cfg.get("split", "train")),
            sequence_length=int(calibration_cfg.get("sequence_length", 2048)),
            num_sequences=int(calibration_cfg.get("num_sequences", 6)),
            batch_size=int(calibration_cfg.get("batch_size", 1)),
        ),
    )
    probe_cfg = debug_config.get("probe_eval", {})
    probe_dataset = load_wikitext2_blocks(
        tokenizer,
        split=str(probe_cfg.get("split", "test")),
        sequence_length=int(probe_cfg.get("sequence_length", 2048)),
        limit_sequences=int(probe_cfg.get("num_sequences", 8)),
    )
    baseline_small_eval = run_wikitext2_perplexity(
        reference_model,
        probe_dataset,
        device=str(reference_device),
        batch_size=int(probe_cfg.get("batch_size", 1)),
    ).perplexity

    prefix_cfg = debug_config["prefix_quant"]
    ordered_specs = _ordered_selected_specs(
        working_model,
        max_layers=prefix_cfg.get("max_layers"),
        max_modules=prefix_cfg.get("max_modules"),
    )
    target_cfg = debug_config["target"]
    if ordered_specs[-1].layer_index != int(target_cfg["layer_index"]) or ordered_specs[-1].kind != str(target_cfg["kind"]):
        raise ValueError(f"Configured target does not match ordered selection tail: {ordered_specs[-1]}")

    target_spec, target_rate, prefix_results = _quantize_prefix(
        working_model,
        reference_model,
        calibration_dataset,
        ordered_specs,
        prefix_layer_cfg=prefix_cfg["layer"],
        target_global_bitwidth=float(prefix_cfg["target_global_bitwidth"]),
        device=device,
        reference_device=reference_device,
        calibration_batch_size=int(calibration_cfg.get("batch_size", 1)),
        logger=logger,
    )

    official_stats = collect_layer_statistics(
        working_model,
        calibration_dataset,
        [target_spec],
        device=device,
        logger=logger,
        calibration_batch_size=int(calibration_cfg.get("batch_size", 1)),
        reference_model=reference_model,
        reference_device=reference_device,
    )[target_spec.full_path]
    manual_stats, first_batch_audit = _manual_collect_o_proj_statistics(
        working_model,
        reference_model,
        calibration_dataset,
        target_spec,
        device=device,
        reference_device=reference_device,
        calibration_batch_size=int(calibration_cfg.get("batch_size", 1)),
    )
    timing_audit = _timing_audit(working_model, reference_model, ordered_specs, target_spec, official_stats, manual_stats, first_batch_audit)

    sweep_results: list[dict[str, Any]] = []
    for residual_scale in [float(value) for value in debug_config.get("residual_scales", [0.0, 0.25, 0.5, 0.75, 1.0])]:
        logger.info("Residual debug sweep at residual_scale=%.2f", residual_scale)
        point = _residual_point(
            working_model,
            reference_model,
            target_spec,
            official_stats,
            target_rate,
            probe_dataset,
            layer_cfg=prefix_cfg["layer"],
            residual_scale=residual_scale,
            device=device,
            reference_device=reference_device,
            legacy_ridge=float(debug_config.get("legacy_ridge", 1e-6)),
        )
        sweep_results.append(point)
        save_json(output_dir / f"scale_{residual_scale:.2f}.json", point)
        _copy_module_weight(working_model.get_submodule(target_spec.full_path), reference_model.get_submodule(target_spec.full_path))
        _maybe_empty_cuda_cache()

    report = {
        "timestamp": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "run_name": run_name,
        "model_id": model_config["model_id"],
        "baseline_small_eval_ppl": baseline_small_eval,
        "config_audit": {
            "reference_stats_requested": bool(debug_config.get("reference_stats", True)),
            "reference_stats_used": bool(debug_config.get("reference_stats", True) and reference_model is not None),
            "reference_device": str(reference_device),
            "rescalers_enabled": bool(prefix_cfg["layer"].get("max_rescaler_iters", 0) > 0),
            "calibration_split": str(calibration_cfg.get("split", "train")),
            "calibration_sequence_length": int(calibration_cfg.get("sequence_length", 2048)),
            "calibration_num_sequences": int(calibration_cfg.get("num_sequences", 6)),
            "calibration_batch_size": int(calibration_cfg.get("batch_size", 1)),
            "probe_split": str(probe_cfg.get("split", "test")),
            "probe_sequence_length": int(probe_cfg.get("sequence_length", 2048)),
            "probe_num_sequences": int(probe_cfg.get("num_sequences", 8)),
            "probe_batch_size": int(probe_cfg.get("batch_size", 1)),
            "residual_scales": [float(value) for value in debug_config.get("residual_scales", [0.0, 0.25, 0.5, 0.75, 1.0])],
            "legacy_ridge": float(debug_config.get("legacy_ridge", 1e-6)),
        },
        "target": {
            "name": target_spec.full_path,
            "layer_index": target_spec.layer_index,
            "kind": target_spec.kind,
            "target_rate": target_rate,
        },
        "timing_audit": timing_audit,
        "prefix_results": prefix_results,
        "residual_scale_sweep": sweep_results,
        "official_stats_audit": {
            "sigma_delta_x_hat_fro_norm": float(torch.linalg.matrix_norm(official_stats.sigma_delta_x_hat, ord="fro").item()),
            "sigma_x_hat_fro_norm": float(torch.linalg.matrix_norm(official_stats.sigma_x_hat, ord="fro").item()),
            "sigma_x_x_hat_fro_norm": float(torch.linalg.matrix_norm(official_stats.sigma_x_x_hat, ord="fro").item()),
        },
        "manual_stats_audit": {
            "sigma_delta_x_hat_fro_norm": float(torch.linalg.matrix_norm(manual_stats.sigma_delta_x_hat, ord="fro").item()),
            "sigma_x_hat_fro_norm": float(torch.linalg.matrix_norm(manual_stats.sigma_x_hat, ord="fro").item()),
            "sigma_x_x_hat_fro_norm": float(torch.linalg.matrix_norm(manual_stats.sigma_x_x_hat, ord="fro").item()),
        },
    }
    save_json(output_dir / "summary.json", report)
    markdown = _render_markdown(report)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return {
        "summary_json_path": output_dir / "summary.json",
        "summary_markdown_path": output_dir / "summary.md",
    }
