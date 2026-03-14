from __future__ import annotations

import gc
import math
import time
from dataclasses import replace
from typing import Any, Callable

import torch

from watersic.models.registry import LinearModuleSpec

from .watersic_layer import (
    LayerQuantizationConfig,
    LayerStatistics,
    prepare_layer_problem,
    quantize_linear_layer,
    quantize_prepared_layer_problem,
)


class _EarlyStopModuleInputCapture(RuntimeError):
    pass


def _copy_module_weight(dst_module, src_module) -> None:
    dst_module.weight.data.copy_(src_module.weight.data)
    if getattr(dst_module, "bias", None) is not None and getattr(src_module, "bias", None) is not None:
        dst_module.bias.data.copy_(src_module.bias.data)


def _restore_module_weights(model, weight_backups: dict[str, torch.Tensor], specs: list[LinearModuleSpec]) -> None:
    for spec in specs:
        module = model.get_submodule(spec.full_path)
        module.weight.data.copy_(weight_backups[spec.full_path].to(module.weight.device, dtype=module.weight.dtype))


def _maybe_empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _relative_mse_components(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    numerator = float((cand - ref).square().sum().item())
    denominator = float(ref.square().sum().item())
    return numerator, denominator


@torch.no_grad()
def _collect_module_inputs(
    model,
    calibration_batches: list[torch.Tensor],
    module_path: str,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    collected: list[torch.Tensor] = []
    module = model.get_submodule(module_path)
    for batch in calibration_batches:
        captured: torch.Tensor | None = None

        def hook(_module, args):
            nonlocal captured
            captured = args[0].detach().to(torch.float32).cpu()
            raise _EarlyStopModuleInputCapture

        handle = module.register_forward_pre_hook(hook)
        try:
            try:
                model(input_ids=batch.to(device), use_cache=False)
            except _EarlyStopModuleInputCapture:
                pass
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError(f"Failed to capture inputs for {module_path}")
        collected.append(captured)
    return collected


@torch.no_grad()
def _module_input_relative_mse(
    model,
    calibration_batches: list[torch.Tensor],
    reference_inputs: list[torch.Tensor],
    module_path: str,
    *,
    device: torch.device,
) -> float:
    numerator = 0.0
    denominator = 0.0
    candidate_inputs = _collect_module_inputs(
        model,
        calibration_batches,
        module_path,
        device=device,
    )
    for reference_input, candidate_input in zip(reference_inputs, candidate_inputs, strict=True):
        num, den = _relative_mse_components(reference_input, candidate_input)
        numerator += num
        denominator += den
    return numerator / max(denominator, 1e-12)


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


def _quantize_qkv_stage_candidate(
    model,
    qkv_specs: list[LinearModuleSpec],
    stats_map: dict[str, LayerStatistics],
    config: LayerQuantizationConfig,
    *,
    stage_remaining_budget: float,
    stage_remaining_weights: int,
    selected_cs: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    achieved: list[dict[str, float]] = []
    total_weights = 0
    total_effective_bits = 0.0
    total_entropy_bits = 0.0
    total_huffman_bits = 0.0
    remaining_budget = float(stage_remaining_budget)
    remaining_weights = int(stage_remaining_weights)
    calibrated_cs: dict[str, float] = {}
    stage_mode = "fixed_c_reuse" if selected_cs is not None else "per_matrix_rate_search"
    for spec in qkv_specs:
        module = model.get_submodule(spec.full_path)
        original_weight = module.weight.detach().cpu()
        target_rate = remaining_budget / max(remaining_weights, 1)
        candidate_config = replace(config, target_rate=target_rate)
        if selected_cs is None:
            result = quantize_linear_layer(
                original_weight,
                stats_map[spec.full_path],
                candidate_config,
                kind=spec.kind,
            )
        else:
            problem = prepare_layer_problem(
                original_weight,
                stats_map[spec.full_path],
                candidate_config,
                kind=spec.kind,
            )
            result = quantize_prepared_layer_problem(
                original_weight,
                problem,
                candidate_config,
                kind=spec.kind,
                selected_c=float(selected_cs[spec.full_path]),
                reference_stats_available=bool(
                    stats_map[spec.full_path].sigma_x_hat is not None
                    and stats_map[spec.full_path].sigma_x_x_hat is not None
                ),
            )
        module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
        num_weights = model.get_submodule(spec.full_path).weight.numel()
        total_weights += num_weights
        total_effective_bits += result.bitrate.final_effective_average_bitwidth * num_weights
        total_entropy_bits += result.bitrate.entropy_average_bitwidth * num_weights
        total_huffman_bits += result.bitrate.huffman_average_bitwidth * num_weights
        calibrated_cs[spec.full_path] = float(result.search.selected_c)
        achieved.append(
            {
                "name": spec.full_path,
                "kind": spec.kind,
                "target_rate": target_rate,
                "selected_c": float(result.search.selected_c),
                "achieved_rate": float(result.bitrate.final_effective_average_bitwidth),
                "entropy_rate": float(result.bitrate.entropy_average_bitwidth),
                "huffman_rate": float(result.bitrate.huffman_average_bitwidth),
            }
        )
        remaining_budget -= result.bitrate.final_effective_average_bitwidth * num_weights
        remaining_weights -= num_weights
    stage_target_rate = float(stage_remaining_budget) / max(int(stage_remaining_weights), 1)
    return calibrated_cs, {
        "target_rate": stage_target_rate,
        "selected_c_by_name": calibrated_cs,
        "search_mode": stage_mode,
        "achieved_rates": achieved,
        "achieved_stage_effective_rate": total_effective_bits / max(total_weights, 1),
        "achieved_stage_entropy_rate": total_entropy_bits / max(total_weights, 1),
        "achieved_stage_huffman_rate": total_huffman_bits / max(total_weights, 1),
    }


def optimize_attention_stage_mixing(
    model,
    reference_model,
    qkv_specs: list[LinearModuleSpec],
    o_proj_spec: LinearModuleSpec,
    stats_map: dict[str, LayerStatistics],
    calibration_dataset,
    config: LayerQuantizationConfig,
    *,
    stage_remaining_budget: float,
    stage_remaining_weights: int,
    calibration_batch_size: int,
    device: torch.device,
    reference_device: torch.device,
    logger,
) -> tuple[float, float, dict[str, Any]]:
    if (
        reference_model is None
        or not config.use_attention_weighting
        or not config.use_adaptive_mixing
        or not config.optimize_adaptive_mixing
    ):
        return config.epsilon_qr, config.epsilon_aw, {"enabled": False}

    weight_backups = {
        spec.full_path: model.get_submodule(spec.full_path).weight.detach().cpu().clone()
        for spec in qkv_specs
    }
    search_iterations = config.golden_section_iterations
    o_proj_path = o_proj_spec.full_path
    calibration_batches = [batch.cpu() for batch in calibration_dataset.batches(calibration_batch_size)]
    reference_inputs = _collect_module_inputs(
        reference_model,
        calibration_batches,
        o_proj_path,
        device=reference_device,
    )
    timing = {
        "objective_evaluations": 0,
        "quantization_seconds_total": 0.0,
        "forward_seconds_total": 0.0,
    }

    def objective_for(
        epsilon_qr: float,
        epsilon_aw: float,
        *,
        selected_cs: dict[str, float] | None,
    ) -> tuple[float, dict[str, Any], dict[str, float]]:
        _restore_module_weights(model, weight_backups, qkv_specs)
        candidate_config = replace(config, epsilon_qr=epsilon_qr, epsilon_aw=epsilon_aw)
        quant_start = time.perf_counter()
        candidate_cs, stage_summary = _quantize_qkv_stage_candidate(
            model,
            qkv_specs,
            stats_map,
            candidate_config,
            stage_remaining_budget=stage_remaining_budget,
            stage_remaining_weights=stage_remaining_weights,
            selected_cs=selected_cs,
        )
        quant_seconds = time.perf_counter() - quant_start
        forward_start = time.perf_counter()
        mse = _module_input_relative_mse(
            model,
            calibration_batches,
            reference_inputs,
            o_proj_path,
            device=device,
        )
        forward_seconds = time.perf_counter() - forward_start
        timing["objective_evaluations"] += 1
        timing["quantization_seconds_total"] += quant_seconds
        timing["forward_seconds_total"] += forward_seconds
        _maybe_empty_cuda_cache()
        return mse, stage_summary, candidate_cs

    initial_start = time.perf_counter()
    initial_objective, initial_stage_summary, initial_cs = objective_for(0.0, 0.0, selected_cs=None)
    initial_seconds = time.perf_counter() - initial_start
    logger.info(
        "Initial QKV rate-calibration at epsilon_qr=0 epsilon_aw=0 reached wo-input relative MSE %.6e",
        initial_objective,
    )

    def qr_objective(value: float) -> float:
        objective, _, _ = objective_for(value, 0.0, selected_cs=initial_cs)
        return objective

    qr_search_start = time.perf_counter()
    best_qr, best_qr_objective, qr_history = _golden_section_search(
        qr_objective,
        iterations=search_iterations,
    )
    qr_search_seconds = time.perf_counter() - qr_search_start
    logger.info("Selected epsilon_qr=%.6f with wo-input relative MSE %.6e", best_qr, best_qr_objective)

    def aw_objective(value: float) -> float:
        objective, _, _ = objective_for(best_qr, value, selected_cs=initial_cs)
        return objective

    aw_search_start = time.perf_counter()
    best_aw, best_aw_objective, aw_history = _golden_section_search(
        aw_objective,
        iterations=search_iterations,
    )
    aw_search_seconds = time.perf_counter() - aw_search_start
    logger.info("Selected epsilon_aw=%.6f with wo-input relative MSE %.6e", best_aw, best_aw_objective)

    final_objective, final_stage_summary, _ = objective_for(best_qr, best_aw, selected_cs=initial_cs)
    _restore_module_weights(model, weight_backups, qkv_specs)
    return best_qr, best_aw, {
        "enabled": True,
        "initial_candidate": {
            "epsilon_qr": 0.0,
            "epsilon_aw": 0.0,
            "wo_input_relative_mse": float(initial_objective),
            "selected_c_by_name": initial_cs,
            "quantization_mode": initial_stage_summary["search_mode"],
            "quantization_seconds": float(initial_seconds),
            "achieved_rates": initial_stage_summary["achieved_rates"],
            "achieved_stage_effective_rate": float(initial_stage_summary["achieved_stage_effective_rate"]),
            "achieved_stage_entropy_rate": float(initial_stage_summary["achieved_stage_entropy_rate"]),
            "achieved_stage_huffman_rate": float(initial_stage_summary["achieved_stage_huffman_rate"]),
        },
        "epsilon_qr": float(best_qr),
        "epsilon_aw": float(best_aw),
        "epsilon_qr_objective": float(best_qr_objective),
        "epsilon_aw_objective": float(best_aw_objective),
        "final_wo_input_relative_mse": float(final_objective),
        "fixed_c_used_during_search": initial_cs,
        "final_achieved_rates": final_stage_summary["achieved_rates"],
        "final_stage_effective_rate": float(final_stage_summary["achieved_stage_effective_rate"]),
        "final_stage_entropy_rate": float(final_stage_summary["achieved_stage_entropy_rate"]),
        "final_stage_huffman_rate": float(final_stage_summary["achieved_stage_huffman_rate"]),
        "epsilon_qr_history": qr_history,
        "epsilon_aw_history": aw_history,
        "objective": "Relative MSE at the o_proj input over the calibration set after quantizing q_proj/k_proj/v_proj.",
        "timing": {
            "initial_rate_calibration_seconds": float(initial_seconds),
            "initial_candidate_total_seconds": float(initial_seconds),
            "epsilon_qr_search_seconds": float(qr_search_seconds),
            "epsilon_aw_search_seconds": float(aw_search_seconds),
            "objective_evaluations": int(timing["objective_evaluations"]),
            "objective_quantization_seconds_total": float(timing["quantization_seconds_total"]),
            "objective_forward_seconds_total": float(timing["forward_seconds_total"]),
        },
        "paper_alignment": {
            "rate_calibration": "Binary-search c at epsilon_qr=0 and epsilon_aw=0 before mixing search.",
            "drift_mixing": "Golden-section search epsilon_qr in [0,1] with epsilon_aw fixed at 0 while reusing the step-1 q/k/v scales.",
            "attention_weighting": "Golden-section search epsilon_aw in [0,1] with epsilon_qr fixed at epsilon_qr* while reusing the step-1 q/k/v scales.",
            "final_quantization": "Final q/k/v quantization reruns binary search over c with selected epsilon_qr* and epsilon_aw*.",
        },
    }
