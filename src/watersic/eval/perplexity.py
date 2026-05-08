from __future__ import annotations

import math

import torch

from watersic.data.wikitext2 import TokenBlockDataset


def _copy_batch_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    copy_non_blocking = device.type == "cuda" and batch.device.type == "cpu" and batch.is_pinned()
    return batch.to(device, non_blocking=copy_non_blocking)


@torch.inference_mode()
def evaluate_perplexity(
    model,
    dataset: TokenBlockDataset,
    *,
    device: torch.device,
    batch_size: int = 1,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataset.batches(batch_size):
        input_ids = _copy_batch_to_device(batch, device)
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = outputs.loss.detach().float().item()
        tokens = input_ids.numel() - input_ids.shape[0]
        total_loss += loss * tokens
        total_tokens += tokens
    return math.exp(total_loss / max(total_tokens, 1))
