import torch

from watersic.quant.lmmse import lmmse_shrinkage


def test_lmmse_handles_zero_quantized_norm() -> None:
    source = torch.tensor([0.0, 0.0, 0.0])
    quantized = torch.tensor([0.0, 0.0, 0.0])
    alpha = lmmse_shrinkage(source, quantized)
    assert torch.isfinite(alpha)
    assert alpha.item() == 1.0


def test_lmmse_is_bounded() -> None:
    source = torch.tensor([1.0, -1.0, 0.5])
    quantized = torch.tensor([1.2, -0.8, 0.4])
    alpha = lmmse_shrinkage(source, quantized)
    assert 0.0 <= alpha.item() <= 1.0
