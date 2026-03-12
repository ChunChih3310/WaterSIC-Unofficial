from __future__ import annotations

import torch


QKV_TARGET_KINDS = {"q_proj", "k_proj", "v_proj"}


def is_attention_weighted_target(kind: str) -> bool:
    return kind in QKV_TARGET_KINDS


def token_importance_from_attention(attention_probs: torch.Tensor) -> torch.Tensor:
    if attention_probs.ndim != 4:
        raise ValueError(f"Expected attention probs with shape [batch, heads, query, key], got {tuple(attention_probs.shape)}")
    probs = attention_probs.to(torch.float64)
    batch, num_heads, seq_len, _ = probs.shape
    weights = torch.zeros((batch, seq_len), dtype=torch.float64, device=probs.device)
    for key_index in range(seq_len):
        # Paper equation (19): average over heads and valid causal query positions i >= j.
        weights[:, key_index] = probs[:, :, key_index:, key_index].mean(dim=(1, 2))
    return weights


def weighted_second_moment(x: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    x_mat = x.reshape(-1, x.shape[-1]).to(torch.float64)
    weights = token_weights.reshape(-1).to(torch.float64)
    weighted_x = x_mat * weights[:, None]
    denom = torch.clamp(weights.sum(), min=1e-12)
    return weighted_x.transpose(0, 1) @ x_mat / denom
