import torch

from watersic.quant.zsic import zsic_quantize, zsic_quantize_for_objective, zsic_quantize_for_rate_search


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


def test_zsic_watersic_recursive_update_uses_alpha_not_global_c() -> None:
    target_y = torch.tensor([[1.2, -0.6]], dtype=torch.float64)
    chol = torch.tensor([[2.0, 0.0], [4.0, 3.0]], dtype=torch.float64)
    result = zsic_quantize(target_y, chol, c=0.5, use_lmmse=False, spacing_strategy="watersic")

    assert result.quantized_ints.tolist() == [[4, -1]]
    assert result.columns[1].index == 1
    assert result.columns[1].quant_step == 0.5


def test_zsic_uniform_alpha_changes_rounding_step() -> None:
    target_y = torch.tensor([[1.2, -0.6]], dtype=torch.float64)
    chol = torch.tensor([[2.0, 0.0], [4.0, 3.0]], dtype=torch.float64)
    result = zsic_quantize(target_y, chol, c=0.5, use_lmmse=False, spacing_strategy="uniform_alpha")

    assert result.quantized_ints.tolist() == [[1, 0]]
    assert result.columns[1].quant_step == 1.5


def test_zsic_rate_search_path_matches_full_quantized_ints() -> None:
    torch.manual_seed(0)
    target_y = torch.randn(8, 5, dtype=torch.float64)
    base = torch.randn(5, 5, dtype=torch.float64)
    chol = torch.linalg.cholesky(base.T @ base + 0.5 * torch.eye(5, dtype=torch.float64))
    c = 0.37

    full = zsic_quantize(target_y, chol, c=c)
    search_ints = zsic_quantize_for_rate_search(target_y, chol, c=c)

    assert torch.equal(full.quantized_ints, search_ints)


def test_zsic_objective_path_matches_full_result_tensors() -> None:
    torch.manual_seed(0)
    target_y = torch.randn(8, 5, dtype=torch.float64)
    base = torch.randn(5, 5, dtype=torch.float64)
    chol = torch.linalg.cholesky(base.T @ base + 0.5 * torch.eye(5, dtype=torch.float64))
    c = 0.37

    full = zsic_quantize(target_y, chol, c=c)
    objective = zsic_quantize_for_objective(target_y, chol, c=c)

    assert torch.equal(full.quantized_ints, objective.quantized_ints)
    assert torch.equal(full.preliminary_weight, objective.preliminary_weight)
    assert torch.equal(full.reconstructed, objective.reconstructed)
    assert torch.equal(full.spacings, objective.spacings)
    assert torch.equal(full.gammas, objective.gammas)
    assert objective.columns == []
    assert objective.weighted_error == 0.0
