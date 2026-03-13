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
from .rescalers import RescalerResult, disabled_rescalers, optimize_diagonal_rescalers


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
    max_rescaler_iters: int = 0
    rescaler_ridge: float = 1e-8
    seed: int = 0
    use_lmmse: bool = True
    use_activation_drift: bool = True
    use_residual_correction: bool = True
    use_attention_weighting: bool = True
    use_adaptive_mixing: bool = True
    spacing_strategy: str = "watersic"


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


@dataclass(frozen=True)
class PreparedLayerProblem:
    reduced_weight: torch.Tensor
    sigma_x: torch.Tensor
    sigma_x_hat: torch.Tensor
    sigma_cross: torch.Tensor
    target_cross: torch.Tensor
    target_y: torch.Tensor
    cholesky: torch.Tensor
    applied_damping: float
    side_information_bits: float
    dead_features: DeadFeatureReport
    compensation_matrix: torch.Tensor | None


def _metadata_overhead_bits(num_rows: int, num_cols: int, dead_report: DeadFeatureReport) -> float:
    scale_bits = 16 * (num_rows + num_cols)
    mask_bits = dead_report.keep_mask.numel()
    return float(scale_bits + mask_bits)


def select_final_statistics(
    stats: LayerStatistics,
    kind: str,
    config: LayerQuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_x = stats.sigma_x.to(torch.float64)
    if config.use_activation_drift:
        sigma_x_hat = (stats.sigma_x_hat if stats.sigma_x_hat is not None else stats.sigma_x).to(torch.float64)
        sigma_x_x_hat = (stats.sigma_x_x_hat if stats.sigma_x_x_hat is not None else stats.sigma_x).to(torch.float64)
    else:
        sigma_x_hat = sigma_x
        sigma_x_x_hat = sigma_x

    if not config.use_attention_weighting or not is_attention_weighted_target(kind):
        return symmetrize(sigma_x), symmetrize(sigma_x_hat), sigma_x_x_hat

    sigma_w_x = (stats.sigma_x_weighted if stats.sigma_x_weighted is not None else sigma_x).to(torch.float64)
    sigma_w_x_hat = (stats.sigma_x_hat_weighted if stats.sigma_x_hat_weighted is not None else sigma_x_hat).to(torch.float64)
    sigma_w_cross = (stats.sigma_x_x_hat_weighted if stats.sigma_x_x_hat_weighted is not None else sigma_x_x_hat).to(torch.float64)

    if not config.use_adaptive_mixing:
        return symmetrize(sigma_w_x), symmetrize(sigma_w_x_hat), sigma_w_cross

    eps_qr = float(max(0.0, min(1.0, config.epsilon_qr)))
    eps_aw = float(max(0.0, min(1.0, config.epsilon_aw)))
    sigma_x_final = (1.0 - eps_aw) * sigma_w_x + eps_aw * sigma_x
    sigma_x_hat_final = (1.0 - eps_aw) * ((1.0 - eps_qr) * sigma_w_x_hat + eps_qr * sigma_w_x) + eps_aw * (
        (1.0 - eps_qr) * sigma_x_hat + eps_qr * sigma_x
    )
    sigma_cross_final = (1.0 - eps_aw) * ((1.0 - eps_qr) * sigma_w_cross + eps_qr * sigma_w_x) + eps_aw * (
        (1.0 - eps_qr) * sigma_x_x_hat + eps_qr * sigma_x
    )
    return symmetrize(sigma_x_final), symmetrize(sigma_x_hat_final), sigma_cross_final


def prepare_layer_problem(
    weight: torch.Tensor,
    stats: LayerStatistics,
    config: LayerQuantizationConfig,
    *,
    kind: str,
) -> PreparedLayerProblem:
    final_sigma_x, final_sigma_x_hat, final_sigma_cross = select_final_statistics(stats, kind, config)
    dead_variances = torch.diag(final_sigma_x_hat if final_sigma_x_hat is not None else final_sigma_x)
    dead_report = detect_dead_features(dead_variances, tau=config.dead_feature_tau)

    reduced_weight = prune_columns(weight, dead_report).to(torch.float64)
    sigma_x = final_sigma_x[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_x_hat = final_sigma_x_hat[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_cross = final_sigma_cross[dead_report.keep_indices][:, dead_report.keep_indices]

    compensation = None
    if config.use_residual_correction and is_residual_target(kind) and stats.sigma_delta_x_hat is not None:
        sigma_delta = stats.sigma_delta_x_hat.to(torch.float64)[:, dead_report.keep_indices]
        compensation = residual_compensation_matrix(sigma_x_hat, sigma_delta)
    target_cross = reduced_weight @ sigma_cross
    if compensation is not None:
        target_cross = target_cross + compensation.to(target_cross.dtype)

    cholesky, applied_damping = stable_cholesky(sigma_x_hat, config.damping)
    target_y = torch.linalg.solve_triangular(
        cholesky,
        target_cross.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    side_bits = _metadata_overhead_bits(reduced_weight.shape[0], reduced_weight.shape[1], dead_report)
    return PreparedLayerProblem(
        reduced_weight=reduced_weight,
        sigma_x=sigma_x,
        sigma_x_hat=sigma_x_hat,
        sigma_cross=sigma_cross,
        target_cross=target_cross,
        target_y=target_y,
        cholesky=cholesky,
        applied_damping=applied_damping,
        side_information_bits=side_bits,
        dead_features=dead_report,
        compensation_matrix=compensation,
    )


def quantize_linear_layer(
    weight: torch.Tensor,
    stats: LayerStatistics,
    config: LayerQuantizationConfig,
    *,
    kind: str,
) -> LayerQuantizationResult:
    problem = prepare_layer_problem(weight, stats, config, kind=kind)
    search = binary_search_c(
        problem.target_y.cpu(),
        problem.cholesky.cpu(),
        config.target_rate,
        side_information_bits=problem.side_information_bits,
        num_iterations=config.binary_search_iterations,
        row_sample_fraction=config.row_sample_fraction,
        seed=config.seed,
        use_lmmse=config.use_lmmse,
        spacing_strategy=config.spacing_strategy,
    )

    if config.max_rescaler_iters > 0:
        rescalers = optimize_diagonal_rescalers(
            search.quantization.preliminary_weight.cpu(),
            problem.target_cross.cpu(),
            problem.sigma_x_hat.cpu(),
            initial_column_scale=search.quantization.gammas.cpu(),
            max_iters=config.max_rescaler_iters,
            ridge=config.rescaler_ridge,
        )
    else:
        rescalers = disabled_rescalers(
            search.quantization.preliminary_weight.cpu(),
            problem.target_cross.cpu(),
            problem.sigma_x_hat.cpu(),
            initial_column_scale=search.quantization.gammas.cpu(),
        )

    reduced_quantized = rescalers.reconstructed.to(weight.dtype)
    quantized_weight = expand_columns(reduced_quantized, problem.dead_features).to(weight.dtype)
    integer_codes = expand_columns(search.quantization.quantized_ints.to(torch.int64), problem.dead_features).to(torch.int64)
    spacings = expand_columns(search.quantization.spacings[None, :], problem.dead_features).squeeze(0).to(weight.dtype)
    lmmse_gammas = expand_columns(search.quantization.gammas[None, :], problem.dead_features).squeeze(0).to(weight.dtype)
    column_scale = expand_columns(rescalers.column_scale[None, :], problem.dead_features).squeeze(0).to(weight.dtype)

    return LayerQuantizationResult(
        quantized_weight=quantized_weight,
        integer_codes=integer_codes,
        spacings=spacings,
        lmmse_gammas=lmmse_gammas,
        row_scale=rescalers.row_scale.to(weight.dtype),
        column_scale=column_scale,
        dead_features=problem.dead_features,
        search=search,
        rescalers=rescalers,
        bitrate=search.bitrate,
        compensation_matrix=problem.compensation_matrix,
        selected_moment=problem.sigma_x_hat,
        applied_damping=problem.applied_damping,
    )
