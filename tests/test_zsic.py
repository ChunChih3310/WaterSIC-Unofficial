import torch

from watersic.quant.zsic import zsic_quantize


def test_zsic_outputs_integer_codes_and_expected_shapes() -> None:
    weight = torch.tensor([[0.2, -0.3], [0.7, 1.1]], dtype=torch.float32)
    moment = torch.tensor([[2.0, 0.1], [0.1, 1.5]], dtype=torch.float64)
    chol = torch.linalg.cholesky(moment)
    result = zsic_quantize(weight, chol, c=0.25)
    assert result.quantized_ints.shape == weight.shape
    assert result.reconstructed.shape == weight.shape
    assert result.quantized_ints.dtype == torch.int64


def test_zsic_is_deterministic() -> None:
    weight = torch.randn(4, 4, dtype=torch.float32)
    chol = torch.linalg.cholesky(torch.eye(4, dtype=torch.float64))
    result_a = zsic_quantize(weight, chol, c=0.4)
    result_b = zsic_quantize(weight, chol, c=0.4)
    assert torch.equal(result_a.quantized_ints, result_b.quantized_ints)
    assert torch.allclose(result_a.reconstructed, result_b.reconstructed)
