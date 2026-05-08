import torch

from watersic.stats.attention_weighting import token_importance_from_attention


def test_attention_weighting_matches_causal_average() -> None:
    attention = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    attention[0, 0] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.4, 0.6, 0.0],
            [0.1, 0.2, 0.7],
        ]
    )
    weights = token_importance_from_attention(attention)
    expected = torch.tensor([[0.5, 0.4, 0.7]], dtype=torch.float64)
    assert torch.allclose(weights, expected)
