from __future__ import annotations

from dataclasses import dataclass

import torch

from watersic.eval.metrics import BitrateMetrics
from watersic.stats.attention_weighting import is_attention_weighted_target
from watersic.stats.covariance import stable_cholesky
from watersic.stats.dead_features import DeadFeatureReport, detect_dead_features, expand_columns, prune_columns
from watersic.stats.drift import symmetrize
from watersic.stats.residual import is_residual_target, residual_compensation_matrix

from .rate_search import RateSearchResult, binary_search_c
from .rescalers import RescalerResult, optimize_diagonal_rescalers


@dataclass(frozen=True)
class LayerQuantizationConfig:
    target_rate: float
    damping: float = 1e-4
    binary_search_iterations: int = 30
    row_sample_fraction: float = 0.1
    golden_section_iterations: int = 15
    dead_feature_tau: float = 1e-3
    epsilon_qr: float = 1.0
    epsilon_aw: float = 1.0
    max_rescaler_iters: int = 8
    rescaler_ridge: float = 1e-8
    seed: int = 0


@dataclass(frozen=True)
class LayerStatistics:
    sigma_x: torch.Tensor
    sigma_x_hat: torch.Tensor | None = None
    sigma_x_x_hat: torch.Tensor | None = None
    sigma_x_weighted: torch.Tensor | None = None
    sigma_x_hat_weighted: torch.Tensor | None = None
    sigma_x_x_hat_weighted: torch.Tensor | None = None
    sigma_delta_x_hat: torch.Tensor | None = None
    variances: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerQuantizationResult:
    quantized_weight: torch.Tensor
    integer_codes: torch.Tensor
    spacings: torch.Tensor
    lmmse_gammas: torch.Tensor
    row_scale: torch.Tensor
    column_scale: torch.Tensor
    dead_features: DeadFeatureReport
    search: RateSearchResult
    rescalers: RescalerResult
    bitrate: BitrateMetrics
    compensation_matrix: torch.Tensor | None
    selected_moment: torch.Tensor
    applied_damping: float


def _metadata_overhead_bits(num_rows: int, num_cols: int, dead_report: DeadFeatureReport) -> float:
    scale_bits = 16 * (num_rows + num_cols)
    mask_bits = dead_report.keep_mask.numel()
    return float(scale_bits + mask_bits)


def _final_statistics(stats: LayerStatistics, kind: str, epsilon_qr: float, epsilon_aw: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_x = stats.sigma_x.to(torch.float64)
    sigma_x_hat = (stats.sigma_x_hat if stats.sigma_x_hat is not None else stats.sigma_x).to(torch.float64)
    sigma_x_x_hat = (stats.sigma_x_x_hat if stats.sigma_x_x_hat is not None else stats.sigma_x).to(torch.float64)

    if not is_attention_weighted_target(kind):
        return symmetrize(sigma_x), symmetrize(sigma_x_hat), sigma_x_x_hat

    sigma_w_x = (stats.sigma_x_weighted if stats.sigma_x_weighted is not None else sigma_x).to(torch.float64)
    sigma_w_x_hat = (stats.sigma_x_hat_weighted if stats.sigma_x_hat_weighted is not None else sigma_x_hat).to(torch.float64)
    sigma_w_cross = (stats.sigma_x_x_hat_weighted if stats.sigma_x_x_hat_weighted is not None else sigma_x_x_hat).to(torch.float64)

    eps_qr = float(max(0.0, min(1.0, epsilon_qr)))
    eps_aw = float(max(0.0, min(1.0, epsilon_aw)))
    sigma_x_final = (1.0 - eps_aw) * sigma_w_x + eps_aw * sigma_x
    sigma_x_hat_final = (1.0 - eps_aw) * ((1.0 - eps_qr) * sigma_w_x_hat + eps_qr * sigma_w_x) + eps_aw * (
        (1.0 - eps_qr) * sigma_x_hat + eps_qr * sigma_x
    )
    sigma_cross_final = (1.0 - eps_aw) * ((1.0 - eps_qr) * sigma_w_cross + eps_qr * sigma_w_x) + eps_aw * (
        (1.0 - eps_qr) * sigma_x_x_hat + eps_qr * sigma_x
    )
    return symmetrize(sigma_x_final), symmetrize(sigma_x_hat_final), sigma_cross_final


def quantize_linear_layer(
    weight: torch.Tensor,
    stats: LayerStatistics,
    config: LayerQuantizationConfig,
    *,
    kind: str,
) -> LayerQuantizationResult:
    variances = stats.variances
    if variances is None:
        variances = torch.diag(stats.sigma_x_hat if stats.sigma_x_hat is not None else stats.sigma_x)
    dead_report = detect_dead_features(variances, tau=config.dead_feature_tau)

    reduced_weight = prune_columns(weight, dead_report).to(torch.float64)
    final_sigma_x, final_sigma_x_hat, final_sigma_cross = _final_statistics(stats, kind, config.epsilon_qr, config.epsilon_aw)
    sigma_x = final_sigma_x[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_x_hat = final_sigma_x_hat[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_cross = final_sigma_cross[dead_report.keep_indices][:, dead_report.keep_indices]

    compensation = None
    if is_residual_target(kind) and sigma_x_hat is not None and stats.sigma_delta_x_hat is not None:
        sigma_delta = stats.sigma_delta_x_hat.to(torch.float64)[:, dead_report.keep_indices]
        compensation = residual_compensation_matrix(sigma_x_hat, sigma_delta)
    target_cross = reduced_weight @ sigma_cross
    if compensation is not None:
        target_cross = target_cross + compensation.to(target_cross.dtype)

    cholesky, applied_damping = stable_cholesky(sigma_x_hat, config.damping)
    target_y = torch.linalg.solve_triangular(
        cholesky.transpose(0, 1),
        target_cross.transpose(0, 1),
        upper=True,
    ).transpose(0, 1)
    side_bits = _metadata_overhead_bits(reduced_weight.shape[0], reduced_weight.shape[1], dead_report)
    search = binary_search_c(
        target_y.cpu(),
        cholesky.cpu(),
        config.target_rate,
        side_information_bits=side_bits,
        num_iterations=config.binary_search_iterations,
        row_sample_fraction=config.row_sample_fraction,
        seed=config.seed,
    )

    rescalers = optimize_diagonal_rescalers(
        search.quantization.preliminary_weight.cpu(),
        target_cross.cpu(),
        sigma_x_hat.cpu(),
        initial_column_scale=search.quantization.gammas.cpu(),
        max_iters=config.max_rescaler_iters,
        ridge=config.rescaler_ridge,
    )

    reduced_quantized = rescalers.reconstructed.to(weight.dtype)
    quantized_weight = expand_columns(reduced_quantized, dead_report).to(weight.dtype)
    integer_codes = expand_columns(search.quantization.quantized_ints.to(torch.int64), dead_report).to(torch.int64)
    spacings = expand_columns(search.quantization.spacings[None, :], dead_report).squeeze(0).to(weight.dtype)
    lmmse_gammas = expand_columns(search.quantization.gammas[None, :], dead_report).squeeze(0).to(weight.dtype)
    column_scale = expand_columns(rescalers.column_scale[None, :], dead_report).squeeze(0).to(weight.dtype)

    return LayerQuantizationResult(
        quantized_weight=quantized_weight,
        integer_codes=integer_codes,
        spacings=spacings,
        lmmse_gammas=lmmse_gammas,
        row_scale=rescalers.row_scale.to(weight.dtype),
        column_scale=column_scale,
        dead_features=dead_report,
        search=search,
        rescalers=rescalers,
        bitrate=search.bitrate,
        compensation_matrix=compensation,
        selected_moment=sigma_x_hat,
        applied_damping=applied_damping,
    )
