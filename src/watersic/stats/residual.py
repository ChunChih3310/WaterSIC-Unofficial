from __future__ import annotations

import torch


RESIDUAL_TARGET_KINDS = {"o_proj", "down_proj"}


def is_residual_target(kind: str) -> bool:
    return kind in RESIDUAL_TARGET_KINDS


def residual_compensation_matrix(
    sigma_x_hat: torch.Tensor,
    sigma_delta_x_hat: torch.Tensor | None,
    *,
    ridge: float = 1e-6,
) -> torch.Tensor | None:
    if sigma_delta_x_hat is None:
        return None
    sigma_x_hat = sigma_x_hat.to(torch.float64)
    sigma_delta_x_hat = sigma_delta_x_hat.to(torch.float64)
    solve_matrix = sigma_x_hat + ridge * torch.eye(sigma_x_hat.shape[0], dtype=sigma_x_hat.dtype, device=sigma_x_hat.device)
    compensation_t = torch.linalg.solve(solve_matrix, sigma_delta_x_hat.transpose(0, 1))
    return compensation_t.transpose(0, 1)
