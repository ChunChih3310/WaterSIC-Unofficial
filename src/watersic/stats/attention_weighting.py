from __future__ import annotations

import torch


QKV_TARGET_KINDS = {"q_proj", "k_proj", "v_proj"}


def is_attention_weighted_target(kind: str) -> bool:
    return kind in QKV_TARGET_KINDS


def token_importance_from_attention(attention_probs: torch.Tensor) -> torch.Tensor:
    if attention_probs.ndim != 4:
        raise ValueError(f"Expected attention probs with shape [batch, heads, query, key], got {tuple(attention_probs.shape)}")
    # Received attention mass is the most stable token-importance proxy available from the attention matrix.
    return attention_probs.to(torch.float64).mean(dim=1).mean(dim=1)


def weighted_second_moment(x: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    x_mat = x.reshape(-1, x.shape[-1]).to(torch.float64)
    weights = token_weights.reshape(-1).to(torch.float64)
    weighted_x = x_mat * weights[:, None]
    denom = torch.clamp(weights.sum(), min=1e-12)
    return weighted_x.transpose(0, 1) @ x_mat / denom
