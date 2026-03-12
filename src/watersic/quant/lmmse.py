from __future__ import annotations

import torch


def lmmse_shrinkage(source: torch.Tensor, quantized: torch.Tensor, ridge: float = 1e-12) -> torch.Tensor:
    source_vec = source.reshape(-1).to(torch.float64)
    quant_vec = quantized.reshape(-1).to(torch.float64)
    denom = torch.dot(quant_vec, quant_vec)
    if float(denom.item()) <= ridge:
        return torch.tensor(1.0, dtype=torch.float64, device=source.device)
    alpha = torch.dot(source_vec, quant_vec) / (denom + ridge)
    return torch.clamp(alpha, min=0.0, max=1.0)
