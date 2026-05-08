from __future__ import annotations

from dataclasses import dataclass

import torch

from .lmmse import lmmse_shrinkage


@dataclass(frozen=True)
class ZSICColumnState:
    index: int
    spacing: float
    quant_step: float
    gamma: float
    raw_mean_abs_error: float
    recursive_update_max_abs_error: float


@dataclass(frozen=True)
class ZSICResult:
    quantized_ints: torch.Tensor
    preliminary_weight: torch.Tensor
    reconstructed: torch.Tensor
    spacings: torch.Tensor
    gammas: torch.Tensor
    weighted_error: float
    columns: list[ZSICColumnState]


def _validate_zsic_inputs(target_y: torch.Tensor, cholesky: torch.Tensor) -> None:
    if target_y.ndim != 2:
        raise ValueError(f"Expected a 2D transformed target matrix, got shape {tuple(target_y.shape)}")
    if cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1]:
        raise ValueError("Expected a square lower-triangular Cholesky factor")
    if cholesky.shape[0] != target_y.shape[1]:
        raise ValueError("Cholesky factor width must match weight input dimension")


def _prepare_zsic_buffers(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    spacing_strategy: str = "watersic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_zsic_inputs(target_y, cholesky)
    work = target_y.to(torch.float64).clone()
    chol = cholesky.to(torch.float64)
    spacings = column_spacings_from_cholesky(chol, c, spacing_strategy=spacing_strategy)
    return work, chol, spacings


def _zsic_quantize_search_impl(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
) -> torch.Tensor:
    work, chol, spacings = _prepare_zsic_buffers(target_y, cholesky, c, spacing_strategy=spacing_strategy)
    quantized_ints = torch.zeros_like(work, dtype=torch.int64)

    for idx in range(target_y.shape[1] - 1, -1, -1):
        source = work[:, idx]
        spacing = spacings[idx]
        quant_step = spacing * chol[idx, idx]
        ints_float = torch.round(source / quant_step)
        quantized_ints[:, idx] = ints_float.to(torch.int64)
        if use_lmmse:
            raw_quantized = ints_float * quant_step
            gamma = lmmse_shrinkage(source, raw_quantized)
        else:
            gamma = torch.tensor(1.0, dtype=torch.float64, device=work.device)
        work[:, : idx + 1].addmm_(
            ints_float[:, None],
            chol[idx : idx + 1, : idx + 1],
            beta=1.0,
            alpha=-float((gamma * spacing).item()),
        )
    return quantized_ints


def _zsic_quantize_result_impl(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
    collect_columns: bool,
    collect_weighted_error: bool,
) -> ZSICResult:
    work, chol, spacings = _prepare_zsic_buffers(target_y, cholesky, c, spacing_strategy=spacing_strategy)
    quantized_ints = torch.zeros_like(work, dtype=torch.int64)
    preliminary_weight = torch.zeros_like(work, dtype=torch.float64)
    gammas = torch.ones(target_y.shape[1], dtype=torch.float64, device=work.device)
    columns: list[ZSICColumnState] = []

    for idx in range(target_y.shape[1] - 1, -1, -1):
        work_prefix = work[:, : idx + 1]
        source = work[:, idx].clone()
        spacing = spacings[idx]
        quant_step = spacing * chol[idx, idx]
        ints_float = torch.round(source / quant_step)
        quantized_ints[:, idx] = ints_float.to(torch.int64)
        raw_quantized = ints_float * quant_step
        gamma = lmmse_shrinkage(source, raw_quantized) if use_lmmse else torch.tensor(1.0, dtype=torch.float64, device=work.device)
        preliminary_weight[:, idx] = ints_float * spacing
        gammas[idx] = gamma

        if collect_columns:
            prefix_before = work_prefix.clone()
            update = gamma * spacing * ints_float[:, None] * chol[idx : idx + 1, : idx + 1]
            work_prefix.sub_(update)
            columns.append(
                ZSICColumnState(
                    index=idx,
                    spacing=float(spacing.item()),
                    quant_step=float(quant_step.item()),
                    gamma=float(gamma.item()),
                    raw_mean_abs_error=float((source - raw_quantized).abs().mean().item()),
                    recursive_update_max_abs_error=float((prefix_before - work_prefix - update).abs().max().item()),
                )
            )
            continue

        work_prefix.addmm_(
            ints_float[:, None],
            chol[idx : idx + 1, : idx + 1],
            beta=1.0,
            alpha=-float((gamma * spacing).item()),
        )

    columns.reverse()
    reconstructed = preliminary_weight * gammas[None, :]
    return ZSICResult(
        quantized_ints=quantized_ints,
        preliminary_weight=preliminary_weight.to(target_y.dtype),
        reconstructed=reconstructed.to(target_y.dtype),
        spacings=spacings.to(target_y.dtype),
        gammas=gammas.to(target_y.dtype),
        weighted_error=weighted_quantization_error(target_y, reconstructed, chol) if collect_weighted_error else 0.0,
        columns=columns,
    )


def column_spacings_from_cholesky(
    cholesky: torch.Tensor,
    c: float,
    *,
    spacing_strategy: str = "watersic",
    min_spacing: float = 1e-8,
) -> torch.Tensor:
    diag = torch.diag(cholesky).to(torch.float64)
    if spacing_strategy == "watersic":
        spacings = c / torch.clamp(diag, min=1e-12)
    elif spacing_strategy == "uniform_alpha":
        spacings = torch.full_like(diag, float(c))
    else:
        raise ValueError(f"Unsupported spacing strategy: {spacing_strategy}")
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
    spacing_strategy: str = "watersic",
) -> ZSICResult:
    return _zsic_quantize_result_impl(
        target_y,
        cholesky,
        c,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
        collect_columns=True,
        collect_weighted_error=True,
    )


def zsic_quantize_for_objective(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
) -> ZSICResult:
    return _zsic_quantize_result_impl(
        target_y,
        cholesky,
        c,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
        collect_columns=False,
        collect_weighted_error=False,
    )


def zsic_quantize_for_rate_search(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    c: float,
    *,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
) -> torch.Tensor:
    return _zsic_quantize_search_impl(
        target_y,
        cholesky,
        c,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
