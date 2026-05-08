from __future__ import annotations

from dataclasses import dataclass

import torch

from watersic.eval.metrics import BitrateMetrics, estimate_bitrate_metrics, estimate_effective_average_bitwidth

from .zsic import ZSICResult, zsic_quantize, zsic_quantize_for_objective, zsic_quantize_for_rate_search


@dataclass(frozen=True)
class SearchStep:
    c: float
    rate: float


@dataclass(frozen=True)
class RateSearchResult:
    target_rate: float
    selected_c: float
    achieved_rate: float
    bounds: tuple[float, float]
    history: list[SearchStep]
    bitrate: BitrateMetrics
    quantization: ZSICResult


def _sample_rows(target_y: torch.Tensor, fraction: float, seed: int = 0) -> torch.Tensor:
    if fraction >= 1.0:
        return target_y
    generator = torch.Generator(device=target_y.device)
    generator.manual_seed(seed)
    num_rows = max(1, int(round(target_y.shape[0] * fraction)))
    indices = torch.randperm(target_y.shape[0], generator=generator, device=target_y.device)[:num_rows]
    return target_y.index_select(0, indices)


def quantize_at_c(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    side_information_bits: float = 0.0,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
    collect_details: bool = True,
) -> tuple[BitrateMetrics, ZSICResult]:
    if collect_details:
        result = zsic_quantize(target_y, cholesky, c, use_lmmse=use_lmmse, spacing_strategy=spacing_strategy)
    else:
        result = zsic_quantize_for_objective(target_y, cholesky, c, use_lmmse=use_lmmse, spacing_strategy=spacing_strategy)
    metrics = estimate_bitrate_metrics(result.quantized_ints, side_information_bits=side_information_bits)
    return metrics, result


def _evaluate(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    side_information_bits: float = 0.0,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
) -> tuple[float, BitrateMetrics]:
    metrics, _ = quantize_at_c(
        target_y,
        cholesky,
        c,
        side_information_bits=side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
    return metrics.final_effective_average_bitwidth, metrics


def _evaluate_rate_only(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    side_information_bits: float = 0.0,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
) -> float:
    quantized_ints = zsic_quantize_for_rate_search(
        target_y,
        cholesky,
        c,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
    return estimate_effective_average_bitwidth(quantized_ints, side_information_bits=side_information_bits)


def binary_search_c(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    target_rate: float,
    *,
    side_information_bits: float = 0.0,
    num_iterations: int = 30,
    row_sample_fraction: float = 0.1,
    seed: int = 0,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
    collect_quantization_details: bool = True,
) -> RateSearchResult:
    sampled_target = _sample_rows(target_y, row_sample_fraction, seed=seed).cpu()
    sampled_cholesky = cholesky.cpu()
    history: list[SearchStep] = []

    low = 1e-6
    high = max(float(target_y.std().item()), 1e-3)

    high_rate = _evaluate_rate_only(
        sampled_target,
        sampled_cholesky,
        high,
        side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
    while high_rate > target_rate:
        high *= 2.0
        high_rate = _evaluate_rate_only(
            sampled_target,
            sampled_cholesky,
            high,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )
        history.append(SearchStep(c=high, rate=high_rate))
        if high > 1e6:
            break

    low_rate = _evaluate_rate_only(
        sampled_target,
        sampled_cholesky,
        low,
        side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
    while low_rate < target_rate:
        low /= 2.0
        low_rate = _evaluate_rate_only(
            sampled_target,
            sampled_cholesky,
            low,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )
        history.append(SearchStep(c=low, rate=low_rate))
        if low < 1e-12:
            break

    for _ in range(num_iterations):
        mid = 0.5 * (low + high)
        mid_rate = _evaluate_rate_only(
            sampled_target,
            sampled_cholesky,
            mid,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )
        history.append(SearchStep(c=mid, rate=mid_rate))
        if mid_rate > target_rate:
            low = mid
        else:
            high = mid

    selected_c = high
    bitrate, quantization = quantize_at_c(
        target_y.cpu(),
        cholesky.cpu(),
        selected_c,
        side_information_bits=side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
        collect_details=collect_quantization_details,
    )
    return RateSearchResult(
        target_rate=target_rate,
        selected_c=selected_c,
        achieved_rate=bitrate.final_effective_average_bitwidth,
        bounds=(low, high),
        history=history,
        bitrate=bitrate,
        quantization=quantization,
    )
