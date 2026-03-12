from __future__ import annotations

from dataclasses import dataclass

import torch

from .perplexity import evaluate_perplexity


@dataclass(frozen=True)
class EvalResult:
    perplexity: float
    dataset_split: str
    sequence_length: int
    batch_size: int


def run_wikitext2_perplexity(model, dataset, *, device: str, batch_size: int) -> EvalResult:
    ppl = evaluate_perplexity(model, dataset, device=torch.device(device), batch_size=batch_size)
    return EvalResult(
        perplexity=ppl,
        dataset_split=dataset.split,
        sequence_length=dataset.sequence_length,
        batch_size=batch_size,
    )
