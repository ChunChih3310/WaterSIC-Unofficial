from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    residual_scale: float = 1.0
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
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PreparedLayerProblem:
    reduced_weight: torch.Tensor
    sigma_x: torch.Tensor
    sigma_x_hat: torch.Tensor
    sigma_cross: torch.Tensor
    base_target_cross: torch.Tensor
    residual_term: torch.Tensor | None
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


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values encountered in {name}")


def _cholesky_diag_condition_proxy(cholesky: torch.Tensor) -> float:
    diag = torch.diag(cholesky).to(torch.float64).abs()
    if diag.numel() == 0:
        return 1.0
    min_diag = float(torch.clamp(diag.min(), min=1e-12).item())
    max_diag = float(diag.max().item())
    return (max_diag / min_diag) ** 2


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
    if dead_report.keep_indices.numel() == 0:
        raise ValueError("All features were pruned as dead; refusing to quantize an empty reduced problem")
    sigma_x = final_sigma_x[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_x_hat = final_sigma_x_hat[dead_report.keep_indices][:, dead_report.keep_indices]
    sigma_cross = final_sigma_cross[dead_report.keep_indices][:, dead_report.keep_indices]
    _assert_finite_tensor("sigma_x", sigma_x)
    _assert_finite_tensor("sigma_x_hat", sigma_x_hat)
    _assert_finite_tensor("sigma_cross", sigma_cross)

    base_target_cross = reduced_weight @ sigma_cross
    compensation = None
    if config.use_residual_correction and is_residual_target(kind) and stats.sigma_delta_x_hat is not None:
        sigma_delta = stats.sigma_delta_x_hat.to(torch.float64)[:, dead_report.keep_indices]
        compensation = residual_compensation_matrix(sigma_delta, scale=config.residual_scale)
        _assert_finite_tensor("sigma_delta_x_hat", sigma_delta)
        _assert_finite_tensor("compensation_matrix", compensation)
    target_cross = base_target_cross
    if compensation is not None:
        target_cross = target_cross + compensation.to(target_cross.dtype)
    _assert_finite_tensor("target_cross", target_cross)

    cholesky, applied_damping = stable_cholesky(sigma_x_hat, config.damping)
    _assert_finite_tensor("cholesky", cholesky)
    if not bool((torch.diag(cholesky) > 0).all().item()):
        raise ValueError("Cholesky diagonal must remain strictly positive")
    target_y = torch.linalg.solve_triangular(
        cholesky,
        target_cross.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    _assert_finite_tensor("target_y", target_y)
    side_bits = _metadata_overhead_bits(reduced_weight.shape[0], reduced_weight.shape[1], dead_report)
    return PreparedLayerProblem(
        reduced_weight=reduced_weight,
        sigma_x=sigma_x,
        sigma_x_hat=sigma_x_hat,
        sigma_cross=sigma_cross,
        base_target_cross=base_target_cross,
        residual_term=compensation,
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
    _assert_finite_tensor("search_spacings", search.quantization.spacings)
    _assert_finite_tensor("search_gammas", search.quantization.gammas)
    _assert_finite_tensor("search_reconstructed", search.quantization.reconstructed)
    if not bool((search.quantization.spacings > 0).all().item()):
        raise ValueError("Column spacings must remain strictly positive")

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
    _assert_finite_tensor("rescaler_reconstructed", rescalers.reconstructed)
    _assert_finite_tensor("rescaler_row_scale", rescalers.row_scale)
    _assert_finite_tensor("rescaler_column_scale", rescalers.column_scale)

    reduced_quantized = rescalers.reconstructed.to(weight.dtype)
    quantized_weight = expand_columns(reduced_quantized, problem.dead_features).to(weight.dtype)
    integer_codes = expand_columns(search.quantization.quantized_ints.to(torch.int64), problem.dead_features).to(torch.int64)
    spacings = expand_columns(search.quantization.spacings[None, :], problem.dead_features).squeeze(0).to(weight.dtype)
    lmmse_gammas = expand_columns(search.quantization.gammas[None, :], problem.dead_features).squeeze(0).to(weight.dtype)
    column_scale = expand_columns(rescalers.column_scale[None, :], problem.dead_features).squeeze(0).to(weight.dtype)
    _assert_finite_tensor("quantized_weight", quantized_weight.to(torch.float64))
    _assert_finite_tensor("expanded_spacings", spacings.to(torch.float64))
    _assert_finite_tensor("expanded_lmmse_gammas", lmmse_gammas.to(torch.float64))
    diagnostics = {
        "all_finite": True,
        "sigma_x_fro_norm": float(torch.linalg.matrix_norm(problem.sigma_x, ord="fro").item()),
        "sigma_x_hat_fro_norm": float(torch.linalg.matrix_norm(problem.sigma_x_hat, ord="fro").item()),
        "sigma_cross_fro_norm": float(torch.linalg.matrix_norm(problem.sigma_cross, ord="fro").item()),
        "base_target_cross_fro_norm": float(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item()),
        "residual_term_fro_norm": float(
            torch.linalg.matrix_norm(problem.residual_term, ord="fro").item()
            if problem.residual_term is not None
            else 0.0
        ),
        "target_cross_fro_norm": float(torch.linalg.matrix_norm(problem.target_cross, ord="fro").item()),
        "target_y_fro_norm": float(torch.linalg.matrix_norm(problem.target_y, ord="fro").item()),
        "target_y_min": float(problem.target_y.min().item()),
        "target_y_max": float(problem.target_y.max().item()),
        "target_y_max_abs": float(problem.target_y.abs().max().item()),
        "cholesky_diag_min": float(torch.diag(problem.cholesky).min().item()),
        "cholesky_diag_max": float(torch.diag(problem.cholesky).max().item()),
        "sigma_x_hat_condition_proxy": _cholesky_diag_condition_proxy(problem.cholesky),
        "alpha_min": float(search.quantization.spacings.min().item()),
        "alpha_max": float(search.quantization.spacings.max().item()),
        "gamma_min": float(search.quantization.gammas.min().item()),
        "gamma_max": float(search.quantization.gammas.max().item()),
        "recursive_update_max_abs_error": float(
            max((column.recursive_update_max_abs_error for column in search.quantization.columns), default=0.0)
        ),
        "residual_correction_applied": bool(problem.compensation_matrix is not None),
        "residual_scale": float(config.residual_scale),
        "residual_to_base_ratio": float(
            torch.linalg.matrix_norm(problem.residual_term, ord="fro").item()
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        )
        if problem.residual_term is not None
        else 0.0,
        "combined_to_base_ratio": float(
            torch.linalg.matrix_norm(problem.target_cross, ord="fro").item()
            / max(torch.linalg.matrix_norm(problem.base_target_cross, ord="fro").item(), 1e-12)
        ),
        "attention_weighting_requested": bool(config.use_attention_weighting),
        "attention_weighting_applied": bool(config.use_attention_weighting and is_attention_weighted_target(kind)),
        "reference_stats_available": bool(stats.sigma_x_hat is not None and stats.sigma_x_x_hat is not None),
    }

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
        diagnostics=diagnostics,
    )
