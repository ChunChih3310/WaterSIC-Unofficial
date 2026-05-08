from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeadFeatureReport:
    variances: torch.Tensor
    threshold: float
    keep_mask: torch.Tensor

    @property
    def keep_indices(self) -> torch.Tensor:
        return torch.nonzero(self.keep_mask, as_tuple=False).flatten()

    @property
    def dead_indices(self) -> torch.Tensor:
        return torch.nonzero(~self.keep_mask, as_tuple=False).flatten()


def feature_variances(x: torch.Tensor) -> torch.Tensor:
    x_mat = x.reshape(-1, x.shape[-1]).to(torch.float64)
    centered = x_mat - x_mat.mean(dim=0, keepdim=True)
    return centered.square().mean(dim=0)


def detect_dead_features(variances: torch.Tensor, tau: float = 1e-3) -> DeadFeatureReport:
    variances = variances.to(torch.float64)
    median_variance = torch.median(variances)
    threshold = float((median_variance * tau).item())
    keep_mask = variances > threshold
    return DeadFeatureReport(variances=variances, threshold=threshold, keep_mask=keep_mask)


def prune_columns(weight: torch.Tensor, report: DeadFeatureReport) -> torch.Tensor:
    return weight[:, report.keep_indices]


def expand_columns(reduced_weight: torch.Tensor, report: DeadFeatureReport) -> torch.Tensor:
    full = torch.zeros(
        (reduced_weight.shape[0], report.keep_mask.numel()),
        dtype=reduced_weight.dtype,
        device=reduced_weight.device,
    )
    full[:, report.keep_indices] = reduced_weight
    return full
