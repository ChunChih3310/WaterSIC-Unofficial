from __future__ import annotations

import gc
import math
from dataclasses import replace
from typing import Any, Callable

import torch

from watersic.models.hooks import ModuleInputCollector
from watersic.models.registry import LinearModuleSpec

from .watersic_layer import LayerQuantizationConfig, LayerStatistics, quantize_linear_layer


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
def _module_input_relative_mse(
    model,
    reference_model,
    dataset,
    module_path: str,
    *,
    device: torch.device,
    reference_device: torch.device,
    calibration_batch_size: int,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for batch in dataset.batches(calibration_batch_size):
        with ModuleInputCollector(model, module_paths=[module_path]) as q_store:
            model(input_ids=batch.to(device), use_cache=False)
        with ModuleInputCollector(reference_model, module_paths=[module_path]) as ref_store:
            reference_model(input_ids=batch.to(reference_device), use_cache=False)
        num, den = _relative_mse_components(ref_store.inputs[module_path], q_store.inputs[module_path])
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
) -> list[dict[str, float]]:
    remaining_budget = float(stage_remaining_budget)
    remaining_weights = int(stage_remaining_weights)
    achieved: list[dict[str, float]] = []
    for spec in qkv_specs:
        module = model.get_submodule(spec.full_path)
        target_rate = remaining_budget / max(remaining_weights, 1)
        candidate_config = replace(config, target_rate=target_rate)
        result = quantize_linear_layer(
            module.weight.detach().cpu(),
            stats_map[spec.full_path],
            candidate_config,
            kind=spec.kind,
        )
        module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
        num_weights = module.weight.numel()
        remaining_budget -= result.bitrate.final_effective_average_bitwidth * num_weights
        remaining_weights -= num_weights
        achieved.append(
            {
                "name": spec.full_path,
                "kind": spec.kind,
                "target_rate": target_rate,
                "achieved_rate": float(result.bitrate.final_effective_average_bitwidth),
                "entropy_rate": float(result.bitrate.entropy_average_bitwidth),
                "huffman_rate": float(result.bitrate.huffman_average_bitwidth),
            }
        )
    return achieved


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

    def objective_for(epsilon_qr: float, epsilon_aw: float) -> tuple[float, list[dict[str, float]]]:
        _restore_module_weights(model, weight_backups, qkv_specs)
        candidate_config = replace(config, epsilon_qr=epsilon_qr, epsilon_aw=epsilon_aw)
        achieved = _quantize_qkv_stage_candidate(
            model,
            qkv_specs,
            stats_map,
            candidate_config,
            stage_remaining_budget=stage_remaining_budget,
            stage_remaining_weights=stage_remaining_weights,
        )
        mse = _module_input_relative_mse(
            model,
            reference_model,
            calibration_dataset,
            o_proj_path,
            device=device,
            reference_device=reference_device,
            calibration_batch_size=calibration_batch_size,
        )
        _maybe_empty_cuda_cache()
        return mse, achieved

    initial_objective, initial_achieved = objective_for(0.0, 0.0)
    logger.info(
        "Initial QKV rate-calibration at epsilon_qr=0 epsilon_aw=0 reached wo-input relative MSE %.6e",
        initial_objective,
    )

    def qr_objective(value: float) -> float:
        objective, _ = objective_for(value, 0.0)
        return objective

    best_qr, best_qr_objective, qr_history = _golden_section_search(
        qr_objective,
        iterations=search_iterations,
    )
    logger.info("Selected epsilon_qr=%.6f with wo-input relative MSE %.6e", best_qr, best_qr_objective)

    def aw_objective(value: float) -> float:
        objective, _ = objective_for(best_qr, value)
        return objective

    best_aw, best_aw_objective, aw_history = _golden_section_search(
        aw_objective,
        iterations=search_iterations,
    )
    logger.info("Selected epsilon_aw=%.6f with wo-input relative MSE %.6e", best_aw, best_aw_objective)

    final_objective, final_achieved = objective_for(best_qr, best_aw)
    _restore_module_weights(model, weight_backups, qkv_specs)
    return best_qr, best_aw, {
        "enabled": True,
        "initial_candidate": {
            "epsilon_qr": 0.0,
            "epsilon_aw": 0.0,
            "wo_input_relative_mse": float(initial_objective),
            "achieved_rates": initial_achieved,
        },
        "epsilon_qr": float(best_qr),
        "epsilon_aw": float(best_aw),
        "epsilon_qr_objective": float(best_qr_objective),
        "epsilon_aw_objective": float(best_aw_objective),
        "final_wo_input_relative_mse": float(final_objective),
        "final_achieved_rates": final_achieved,
        "epsilon_qr_history": qr_history,
        "epsilon_aw_history": aw_history,
        "objective": "Relative MSE at the o_proj input over the calibration set after quantizing q_proj/k_proj/v_proj.",
        "paper_alignment": {
            "rate_calibration": "Binary-search c at epsilon_qr=0 and epsilon_aw=0 before mixing search.",
            "drift_mixing": "Golden-section search epsilon_qr in [0,1] with epsilon_aw fixed at 0.",
            "attention_weighting": "Golden-section search epsilon_aw in [0,1] with epsilon_qr fixed at epsilon_qr*.",
            "final_quantization": "Final q/k/v quantization reruns binary search over c with selected epsilon_qr* and epsilon_aw*.",
        },
    }
