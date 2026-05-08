from __future__ import annotations

import torch


def symmetrize(moment: torch.Tensor) -> torch.Tensor:
    return 0.5 * (moment + moment.transpose(0, 1))


def mix_activation_moments(
    sigma_x: torch.Tensor,
    sigma_x_hat: torch.Tensor | None,
    *,
    epsilon_qr: float,
) -> torch.Tensor:
    if sigma_x_hat is None:
        return symmetrize(sigma_x)
    eps = float(max(0.0, min(1.0, epsilon_qr)))
    mixed = (1.0 - eps) * sigma_x.to(torch.float64) + eps * sigma_x_hat.to(torch.float64)
    return symmetrize(mixed)
