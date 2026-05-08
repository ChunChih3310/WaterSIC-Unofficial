from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from watersic.data.wikitext2 import TokenBlockDataset
from watersic.eval.perplexity import evaluate_perplexity


class _FixedLossModel:
    def __init__(self, losses: list[float]) -> None:
        self._losses = iter(losses)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, *, input_ids: torch.Tensor, labels: torch.Tensor, use_cache: bool = False):
        assert use_cache is False
        assert torch.equal(input_ids, labels)
        return SimpleNamespace(loss=torch.tensor(next(self._losses), dtype=torch.float32))


def test_evaluate_perplexity_matches_weighted_token_average() -> None:
    dataset = TokenBlockDataset(
        input_ids=torch.arange(24, dtype=torch.long).view(3, 8),
        split="test",
        sequence_length=8,
    )
    model = _FixedLossModel([0.2, 0.4, 0.6])

    ppl = evaluate_perplexity(
        model,
        dataset,
        device=torch.device("cpu"),
        batch_size=1,
    )

    expected_loss = ((0.2 * 7) + (0.4 * 7) + (0.6 * 7)) / (7 + 7 + 7)
    assert model.eval_called is True
    assert math.isclose(ppl, math.exp(expected_loss), rel_tol=0.0, abs_tol=2e-8)
