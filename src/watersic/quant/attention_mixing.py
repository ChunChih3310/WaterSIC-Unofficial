from __future__ import annotations

import gc
import math
import time
from dataclasses import replace
from typing import Any, Callable

import torch

from watersic.models.registry import LinearModuleSpec

from .rate_search import binary_search_c
from .watersic_layer import (
    ObjectiveQuantizationResult,
    LayerQuantizationConfig,
    LayerStatistics,
    prepare_layer_problem,
    quantize_linear_layer,
    quantize_prepared_layer_problem,
    quantize_prepared_layer_problem_objective,
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
        backup = weight_backups[spec.full_path]
        if backup.device == module.weight.device and backup.dtype == module.weight.dtype:
            module.weight.data.copy_(backup)
        else:
            module.weight.data.copy_(backup.to(module.weight.device, dtype=module.weight.dtype))


def _finalize_cuda_cleanup(*, uses_cuda: bool) -> None:
    gc.collect()
    if uses_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _relative_mse_components(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    numerator = float((cand - ref).square().sum().item())
    denominator = float(ref.square().sum().item())
    return numerator, denominator


def _copy_batch_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    copy_non_blocking = device.type == "cuda" and batch.device.type == "cpu" and batch.is_pinned()
    return batch.to(device, non_blocking=copy_non_blocking)


@torch.inference_mode()
def _capture_module_input_for_batch(
    model,
    batch: torch.Tensor,
    module_path: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    module = model.get_submodule(module_path)
    captured: torch.Tensor | None = None

    def hook(_module, args):
        nonlocal captured
        captured = args[0].detach().to(torch.float32).cpu()
        raise _EarlyStopModuleInputCapture

    handle = module.register_forward_pre_hook(hook)
    try:
        try:
            model(input_ids=_copy_batch_to_device(batch, device), use_cache=False)
        except _EarlyStopModuleInputCapture:
            pass
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError(f"Failed to capture inputs for {module_path}")
    return captured


@torch.inference_mode()
def _collect_module_inputs(
    model,
    calibration_batches: list[torch.Tensor],
    module_path: str,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    collected: list[torch.Tensor] = []
    for batch in calibration_batches:
        collected.append(_capture_module_input_for_batch(model, batch, module_path, device=device))
    return collected


@torch.inference_mode()
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
    for batch, reference_input in zip(calibration_batches, reference_inputs, strict=True):
        candidate_input = _capture_module_input_for_batch(
            model,
            batch,
            module_path,
            device=device,
        )
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
    modules_by_name: dict[str, torch.nn.Module] | None = None,
    weight_cpu_backups: dict[str, torch.Tensor] | None = None,
    objective_only: bool = False,
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
        module = modules_by_name[spec.full_path] if modules_by_name is not None else model.get_submodule(spec.full_path)
        original_weight = (
            weight_cpu_backups[spec.full_path]
            if weight_cpu_backups is not None
            else module.weight.detach().cpu()
        )
        target_rate = remaining_budget / max(remaining_weights, 1)
        candidate_config = replace(config, target_rate=target_rate)
        objective_result: ObjectiveQuantizationResult | None = None
        if selected_cs is None:
            if objective_only:
                problem = prepare_layer_problem(
                    original_weight,
                    stats_map[spec.full_path],
                    candidate_config,
                    kind=spec.kind,
                )
                search = binary_search_c(
                    problem.target_y,
                    problem.cholesky,
                    candidate_config.target_rate,
                    side_information_bits=problem.side_information_bits,
                    num_iterations=candidate_config.binary_search_iterations,
                    row_sample_fraction=candidate_config.row_sample_fraction,
                    seed=candidate_config.seed,
                    use_lmmse=candidate_config.use_lmmse,
                    spacing_strategy=candidate_config.spacing_strategy,
                    collect_quantization_details=False,
                )
                objective_result = quantize_prepared_layer_problem_objective(
                    original_weight,
                    problem,
                    candidate_config,
                    search=search,
                )
            else:
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
            if objective_only:
                objective_result = quantize_prepared_layer_problem_objective(
                    original_weight,
                    problem,
                    candidate_config,
                    selected_c=float(selected_cs[spec.full_path]),
                )
            else:
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
        quantized_weight = (
            objective_result.quantized_weight if objective_result is not None else result.quantized_weight
        )
        module.weight.data.copy_(quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
        num_weights = module.weight.numel()
        total_weights += num_weights
        achieved_rate = objective_result.effective_rate if objective_result is not None else result.bitrate.final_effective_average_bitwidth
        entropy_rate = objective_result.entropy_rate if objective_result is not None else result.bitrate.entropy_average_bitwidth
        huffman_rate = objective_result.huffman_rate if objective_result is not None else result.bitrate.huffman_average_bitwidth
        selected_c_value = objective_result.selected_c if objective_result is not None else float(result.search.selected_c)
        total_effective_bits += achieved_rate * num_weights
        total_entropy_bits += entropy_rate * num_weights
        total_huffman_bits += huffman_rate * num_weights
        calibrated_cs[spec.full_path] = selected_c_value
        if not objective_only:
            achieved.append(
                {
                    "name": spec.full_path,
                    "kind": spec.kind,
                    "target_rate": target_rate,
                    "selected_c": selected_c_value,
                    "achieved_rate": achieved_rate,
                    "entropy_rate": entropy_rate,
                    "huffman_rate": huffman_rate,
                }
            )
        remaining_budget -= achieved_rate * num_weights
        remaining_weights -= num_weights
    stage_target_rate = float(stage_remaining_budget) / max(int(stage_remaining_weights), 1)
    return calibrated_cs, {
        "target_rate": stage_target_rate,
        "selected_c_by_name": calibrated_cs,
        "search_mode": stage_mode,
        "achieved_rates": achieved if not objective_only else [],
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

    modules_by_name = {
        spec.full_path: model.get_submodule(spec.full_path)
        for spec in qkv_specs
    }
    weight_backups = {
        spec.full_path: model.get_submodule(spec.full_path).weight.detach().clone()
        for spec in qkv_specs
    }
    weight_cpu_backups = {
        spec.full_path: modules_by_name[spec.full_path].weight.detach().cpu().clone()
        for spec in qkv_specs
    }
    search_iterations = config.golden_section_iterations
    o_proj_path = o_proj_spec.full_path
    calibration_batches = [batch.cpu() for batch in calibration_dataset.batches(calibration_batch_size)]
    uses_cuda = device.type == "cuda" or reference_device.type == "cuda"
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

    try:
        def objective_for(
            epsilon_qr: float,
            epsilon_aw: float,
            *,
            selected_cs: dict[str, float] | None,
            collect_summary: bool,
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
                modules_by_name=modules_by_name,
                weight_cpu_backups=weight_cpu_backups,
                objective_only=not collect_summary,
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
            return mse, stage_summary, candidate_cs

        initial_start = time.perf_counter()
        initial_objective, initial_stage_summary, initial_cs = objective_for(
            0.0,
            0.0,
            selected_cs=None,
            collect_summary=True,
        )
        initial_seconds = time.perf_counter() - initial_start
        logger.info(
            "Initial QKV rate-calibration at epsilon_qr=0 epsilon_aw=0 reached wo-input relative MSE %.6e",
            initial_objective,
        )

        def qr_objective(value: float) -> float:
            objective, _, _ = objective_for(value, 0.0, selected_cs=initial_cs, collect_summary=False)
            return objective

        qr_search_start = time.perf_counter()
        best_qr, best_qr_objective, qr_history = _golden_section_search(
            qr_objective,
            iterations=search_iterations,
        )
        qr_search_seconds = time.perf_counter() - qr_search_start
        logger.info("Selected epsilon_qr=%.6f with wo-input relative MSE %.6e", best_qr, best_qr_objective)

        def aw_objective(value: float) -> float:
            objective, _, _ = objective_for(best_qr, value, selected_cs=initial_cs, collect_summary=False)
            return objective

        aw_search_start = time.perf_counter()
        best_aw, best_aw_objective, aw_history = _golden_section_search(
            aw_objective,
            iterations=search_iterations,
        )
        aw_search_seconds = time.perf_counter() - aw_search_start
        logger.info("Selected epsilon_aw=%.6f with wo-input relative MSE %.6e", best_aw, best_aw_objective)

        final_objective, final_stage_summary, _ = objective_for(
            best_qr,
            best_aw,
            selected_cs=initial_cs,
            collect_summary=True,
        )
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
    finally:
        _restore_module_weights(model, weight_backups, qkv_specs)
        del reference_inputs
        del calibration_batches
        del weight_backups
        del weight_cpu_backups
        del modules_by_name
        _finalize_cuda_cleanup(uses_cuda=uses_cuda)
