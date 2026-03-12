from __future__ import annotations

from dataclasses import dataclass

import torch

from .lmmse import lmmse_shrinkage


@dataclass(frozen=True)
class ZSICColumnState:
    index: int
    spacing: float
    gamma: float
    raw_mean_abs_error: float


@dataclass(frozen=True)
class ZSICResult:
    quantized_ints: torch.Tensor
    preliminary_weight: torch.Tensor
    reconstructed: torch.Tensor
    spacings: torch.Tensor
    gammas: torch.Tensor
    weighted_error: float
    columns: list[ZSICColumnState]


def column_spacings_from_cholesky(cholesky: torch.Tensor, c: float, min_spacing: float = 1e-8) -> torch.Tensor:
    diag = torch.diag(cholesky).to(torch.float64)
    spacings = c / torch.clamp(diag, min=1e-12)
    return torch.clamp(spacings, min=min_spacing)


def weighted_quantization_error(target_y: torch.Tensor, reconstructed_weight: torch.Tensor, cholesky: torch.Tensor) -> float:
    error = target_y.to(torch.float64) - reconstructed_weight.to(torch.float64) @ cholesky.to(torch.float64)
    return float(error.square().mean().item())


def zsic_quantize(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    use_lmmse: bool = True,
) -> ZSICResult:
    if target_y.ndim != 2:
        raise ValueError(f"Expected a 2D transformed target matrix, got shape {tuple(target_y.shape)}")
    if cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1]:
        raise ValueError("Expected a square lower-triangular Cholesky factor")
    if cholesky.shape[0] != target_y.shape[1]:
        raise ValueError("Cholesky factor width must match weight input dimension")

    work = target_y.to(torch.float64).clone()
    chol = cholesky.to(torch.float64)
    spacings = column_spacings_from_cholesky(chol, c)
    quantized_ints = torch.zeros_like(work, dtype=torch.int64)
    preliminary_weight = torch.zeros_like(work, dtype=torch.float64)
    gammas = torch.ones(target_y.shape[1], dtype=torch.float64, device=work.device)
    columns: list[ZSICColumnState] = []

    for idx in range(target_y.shape[1] - 1, -1, -1):
        source = work[:, idx]
        ints = torch.round(source / c).to(torch.int64)
        raw_quantized = ints.to(torch.float64) * c
        gamma = lmmse_shrinkage(source, raw_quantized) if use_lmmse else torch.tensor(1.0, dtype=torch.float64, device=work.device)
        spacing = spacings[idx]

        quantized_ints[:, idx] = ints
        preliminary_weight[:, idx] = ints.to(torch.float64) * spacing
        gammas[idx] = gamma
        columns.append(
            ZSICColumnState(
                index=idx,
                spacing=float(spacing.item()),
                gamma=float(gamma.item()),
                raw_mean_abs_error=float((source - raw_quantized).abs().mean().item()),
            )
        )

        work[:, : idx + 1] -= gamma * c * ints.to(torch.float64)[:, None] * chol[idx : idx + 1, : idx + 1]

    columns.reverse()
    reconstructed = preliminary_weight * gammas[None, :]
    return ZSICResult(
        quantized_ints=quantized_ints,
        preliminary_weight=preliminary_weight.to(target_y.dtype),
        reconstructed=reconstructed.to(target_y.dtype),
        spacings=spacings.to(target_y.dtype),
        gammas=gammas.to(target_y.dtype),
        weighted_error=weighted_quantization_error(target_y, reconstructed, chol),
        columns=columns,
    )
