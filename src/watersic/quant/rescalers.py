from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RescalerResult:
    row_scale: torch.Tensor
    column_scale: torch.Tensor
    reconstructed: torch.Tensor
    initial_objective: float
    final_objective: float
    iterations: int


@dataclass(frozen=True)
class PreparedRescalerProblem:
    target_cross: torch.Tensor
    sigma_x_hat: torch.Tensor
    eye: torch.Tensor


def _objective(weight0: torch.Tensor, target_cross: torch.Tensor, sigma_x_hat: torch.Tensor, row_scale: torch.Tensor, column_scale: torch.Tensor) -> float:
    approx = (weight0 * row_scale[:, None]) * column_scale[None, :]
    quad = torch.trace(approx @ sigma_x_hat @ approx.transpose(0, 1))
    linear = 2.0 * torch.trace(target_cross @ approx.transpose(0, 1))
    return float((quad - linear).item())


def _renormalize(row_scale: torch.Tensor, column_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    factor = row_scale.numel() / torch.clamp(row_scale.sum(), min=1e-12)
    return row_scale * factor, column_scale / factor


def prepare_rescaler_problem(target_cross: torch.Tensor, sigma_x_hat: torch.Tensor) -> PreparedRescalerProblem:
    target_cross_f64 = target_cross.to(torch.float64)
    sigma_x_hat_f64 = sigma_x_hat.to(torch.float64)
    return PreparedRescalerProblem(
        target_cross=target_cross_f64,
        sigma_x_hat=sigma_x_hat_f64,
        eye=torch.eye(sigma_x_hat_f64.shape[0], dtype=torch.float64, device=sigma_x_hat_f64.device),
    )


def _optimize_diagonal_rescalers_prepared(
    weight0: torch.Tensor,
    prepared: PreparedRescalerProblem,
    *,
    initial_column_scale: torch.Tensor | None = None,
    max_iters: int = 12,
    ridge: float = 1e-8,
) -> RescalerResult:
    w0 = weight0.to(torch.float64)
    target_cross = prepared.target_cross
    sigma_x_hat = prepared.sigma_x_hat

    row_scale = torch.ones(w0.shape[0], dtype=torch.float64, device=w0.device)
    column_scale = (
        torch.ones(w0.shape[1], dtype=torch.float64, device=w0.device)
        if initial_column_scale is None
        else initial_column_scale.to(torch.float64).clone()
    )
    row_scale, column_scale = _renormalize(row_scale, column_scale)

    initial = _objective(w0, target_cross, sigma_x_hat, row_scale, column_scale)
    eye = prepared.eye

    for _ in range(max_iters):
        w0_gamma = w0 * column_scale[None, :]
        f8 = w0_gamma @ sigma_x_hat @ w0_gamma.transpose(0, 1)
        f7t = target_cross @ w0_gamma.transpose(0, 1)
        row_scale = torch.diag(f7t) / (torch.diag(f8) + ridge)
        row_scale, column_scale = _renormalize(row_scale, column_scale)

        weighted_w0 = w0 * row_scale[:, None]
        f2 = weighted_w0.transpose(0, 1) @ weighted_w0
        f3 = f2 * sigma_x_hat + ridge * eye
        f4 = (w0.transpose(0, 1) * row_scale[None, :]) @ target_cross
        rhs = torch.diag(f4)
        column_scale = torch.linalg.solve(f3, rhs)
        row_scale, column_scale = _renormalize(row_scale, column_scale)

    reconstructed = (w0 * row_scale[:, None]) * column_scale[None, :]
    return RescalerResult(
        row_scale=row_scale.to(weight0.dtype),
        column_scale=column_scale.to(weight0.dtype),
        reconstructed=reconstructed.to(weight0.dtype),
        initial_objective=initial,
        final_objective=_objective(w0, target_cross, sigma_x_hat, row_scale, column_scale),
        iterations=max_iters,
    )


def _disabled_rescalers_prepared(
    weight0: torch.Tensor,
    prepared: PreparedRescalerProblem,
    *,
    initial_column_scale: torch.Tensor | None = None,
) -> RescalerResult:
    w0 = weight0.to(torch.float64)
    row_scale = torch.ones(w0.shape[0], dtype=torch.float64, device=w0.device)
    column_scale = (
        torch.ones(w0.shape[1], dtype=torch.float64, device=w0.device)
        if initial_column_scale is None
        else initial_column_scale.to(torch.float64).clone()
    )
    reconstructed = (w0 * row_scale[:, None]) * column_scale[None, :]
    objective = _objective(
        w0,
        prepared.target_cross,
        prepared.sigma_x_hat,
        row_scale,
        column_scale,
    )
    return RescalerResult(
        row_scale=row_scale.to(weight0.dtype),
        column_scale=column_scale.to(weight0.dtype),
        reconstructed=reconstructed.to(weight0.dtype),
        initial_objective=objective,
        final_objective=objective,
        iterations=0,
    )


def optimize_diagonal_rescalers(
    weight0: torch.Tensor,
    target_cross: torch.Tensor,
    sigma_x_hat: torch.Tensor,
    *,
    initial_column_scale: torch.Tensor | None = None,
    max_iters: int = 12,
    ridge: float = 1e-8,
) -> RescalerResult:
    return _optimize_diagonal_rescalers_prepared(
        weight0,
        prepare_rescaler_problem(target_cross, sigma_x_hat),
        initial_column_scale=initial_column_scale,
        max_iters=max_iters,
        ridge=ridge,
    )


def disabled_rescalers(
    weight0: torch.Tensor,
    target_cross: torch.Tensor,
    sigma_x_hat: torch.Tensor,
    *,
    initial_column_scale: torch.Tensor | None = None,
) -> RescalerResult:
    return _disabled_rescalers_prepared(
        weight0,
        prepare_rescaler_problem(target_cross, sigma_x_hat),
        initial_column_scale=initial_column_scale,
    )
