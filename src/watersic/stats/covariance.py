from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SecondMomentAccumulator:
    dim_in: int
    dim_out: int | None = None
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        self.dim_out = self.dim_in if self.dim_out is None else self.dim_out
        self._sum = torch.zeros((self.dim_in, self.dim_out), dtype=self.dtype)
        self._weight_sum = torch.tensor(0.0, dtype=self.dtype)

    @property
    def weight_sum(self) -> float:
        return float(self._weight_sum.item())

    def update(self, x: torch.Tensor, y: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> None:
        x_mat = x.reshape(-1, x.shape[-1]).to(self.dtype)
        y_mat = x_mat if y is None else y.reshape(-1, y.shape[-1]).to(self.dtype)
        if weights is None:
            self._sum += x_mat.transpose(0, 1) @ y_mat
            self._weight_sum += x_mat.shape[0]
            return

        flat_weights = weights.reshape(-1).to(self.dtype)
        weighted_x = x_mat * flat_weights[:, None]
        self._sum += weighted_x.transpose(0, 1) @ y_mat
        self._weight_sum += flat_weights.sum()

    def finalize(self, eps: float = 1e-12) -> torch.Tensor:
        denom = max(self.weight_sum, eps)
        return self._sum / denom


def damped_second_moment(moment: torch.Tensor, damping: float) -> torch.Tensor:
    eye = torch.eye(moment.shape[0], device=moment.device, dtype=moment.dtype)
    diag_scale = torch.trace(moment) / max(moment.shape[0], 1)
    return moment + damping * diag_scale * eye


def stable_cholesky(moment: torch.Tensor, damping: float, max_retries: int = 8) -> tuple[torch.Tensor, float]:
    current = damping
    for _ in range(max_retries):
        try:
            damped = damped_second_moment(moment, current)
            return torch.linalg.cholesky(damped), current
        except RuntimeError:
            current *= 10.0
    raise RuntimeError("Unable to compute Cholesky factor even after repeated damping increases")
