import torch

from watersic.quant.rate_search import _sample_rows, binary_search_c, quantize_at_c


def _reference_binary_search_c(
    target_y: torch.Tensor,
    cholesky: torch.Tensor,
    target_rate: float,
    *,
    side_information_bits: float = 0.0,
    num_iterations: int = 30,
    row_sample_fraction: float = 0.1,
    seed: int = 0,
    use_lmmse: bool = True,
    spacing_strategy: str = "watersic",
):
    sampled_target = _sample_rows(target_y, row_sample_fraction, seed=seed).cpu()
    sampled_cholesky = cholesky.cpu()
    history: list[tuple[float, float]] = []

    low = 1e-6
    high = max(float(target_y.std().item()), 1e-3)

    high_rate = quantize_at_c(
        sampled_target,
        sampled_cholesky,
        high,
        side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )[0].final_effective_average_bitwidth
    while high_rate > target_rate:
        high *= 2.0
        high_rate = quantize_at_c(
            sampled_target,
            sampled_cholesky,
            high,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )[0].final_effective_average_bitwidth
        history.append((high, high_rate))
        if high > 1e6:
            break

    low_rate = quantize_at_c(
        sampled_target,
        sampled_cholesky,
        low,
        side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )[0].final_effective_average_bitwidth
    while low_rate < target_rate:
        low /= 2.0
        low_rate = quantize_at_c(
            sampled_target,
            sampled_cholesky,
            low,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )[0].final_effective_average_bitwidth
        history.append((low, low_rate))
        if low < 1e-12:
            break

    for _ in range(num_iterations):
        mid = 0.5 * (low + high)
        mid_rate = quantize_at_c(
            sampled_target,
            sampled_cholesky,
            mid,
            side_information_bits,
            use_lmmse=use_lmmse,
            spacing_strategy=spacing_strategy,
        )[0].final_effective_average_bitwidth
        history.append((mid, mid_rate))
        if mid_rate > target_rate:
            low = mid
        else:
            high = mid

    selected_c = high
    bitrate, quantization = quantize_at_c(
        target_y.cpu(),
        cholesky.cpu(),
        selected_c,
        side_information_bits=side_information_bits,
        use_lmmse=use_lmmse,
        spacing_strategy=spacing_strategy,
    )
    return {
        "selected_c": selected_c,
        "achieved_rate": bitrate.final_effective_average_bitwidth,
        "bounds": (low, high),
        "history": history,
        "quantization": quantization,
    }


def test_rate_search_moves_toward_target_rate() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 8)
    moment = torch.randn(8, 8, dtype=torch.float64)
    moment = moment.transpose(0, 1) @ moment + 0.1 * torch.eye(8, dtype=torch.float64)
    chol = torch.linalg.cholesky(moment)

    low_rate = binary_search_c(weight, chol, target_rate=1.5, num_iterations=8, row_sample_fraction=0.25)
    high_rate = binary_search_c(weight, chol, target_rate=4.0, num_iterations=8, row_sample_fraction=0.25)

    assert low_rate.achieved_rate <= high_rate.achieved_rate
    assert low_rate.selected_c >= high_rate.selected_c


def test_rate_search_matches_full_metric_reference_search_exactly() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 6, dtype=torch.float64)
    moment = torch.randn(6, 6, dtype=torch.float64)
    moment = moment.transpose(0, 1) @ moment + 0.1 * torch.eye(6, dtype=torch.float64)
    chol = torch.linalg.cholesky(moment)

    result = binary_search_c(weight, chol, target_rate=3.0, num_iterations=8, row_sample_fraction=0.25, seed=7)
    reference = _reference_binary_search_c(weight, chol, target_rate=3.0, num_iterations=8, row_sample_fraction=0.25, seed=7)

    assert result.selected_c == reference["selected_c"]
    assert result.achieved_rate == reference["achieved_rate"]
    assert result.bounds == reference["bounds"]
    assert [(step.c, step.rate) for step in result.history] == reference["history"]
    assert torch.equal(result.quantization.quantized_ints, reference["quantization"].quantized_ints)


def test_rate_search_minimal_quantization_matches_full_result() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 6, dtype=torch.float64)
    moment = torch.randn(6, 6, dtype=torch.float64)
    moment = moment.transpose(0, 1) @ moment + 0.1 * torch.eye(6, dtype=torch.float64)
    chol = torch.linalg.cholesky(moment)

    full = binary_search_c(weight, chol, target_rate=3.0, num_iterations=8, row_sample_fraction=0.25, seed=7)
    minimal = binary_search_c(
        weight,
        chol,
        target_rate=3.0,
        num_iterations=8,
        row_sample_fraction=0.25,
        seed=7,
        collect_quantization_details=False,
    )

    assert minimal.selected_c == full.selected_c
    assert minimal.achieved_rate == full.achieved_rate
    assert minimal.bounds == full.bounds
    assert [(step.c, step.rate) for step in minimal.history] == [(step.c, step.rate) for step in full.history]
    assert torch.equal(minimal.quantization.quantized_ints, full.quantization.quantized_ints)
    assert torch.equal(minimal.quantization.preliminary_weight, full.quantization.preliminary_weight)
    assert torch.equal(minimal.quantization.reconstructed, full.quantization.reconstructed)
    assert torch.equal(minimal.quantization.spacings, full.quantization.spacings)
    assert torch.equal(minimal.quantization.gammas, full.quantization.gammas)
    assert minimal.quantization.columns == []
    assert minimal.quantization.weighted_error == 0.0


def test_quantize_at_c_minimal_quantization_matches_full_result() -> None:
    torch.manual_seed(0)
    target_y = torch.randn(16, 5, dtype=torch.float64)
    base = torch.randn(5, 5, dtype=torch.float64)
    chol = torch.linalg.cholesky(base.T @ base + 0.5 * torch.eye(5, dtype=torch.float64))

    full_metrics, full_result = quantize_at_c(target_y, chol, c=0.25, side_information_bits=7.0)
    minimal_metrics, minimal_result = quantize_at_c(target_y, chol, c=0.25, side_information_bits=7.0, collect_details=False)

    assert minimal_metrics == full_metrics
    assert torch.equal(minimal_result.quantized_ints, full_result.quantized_ints)
    assert torch.equal(minimal_result.preliminary_weight, full_result.preliminary_weight)
    assert torch.equal(minimal_result.reconstructed, full_result.reconstructed)
    assert torch.equal(minimal_result.spacings, full_result.spacings)
    assert torch.equal(minimal_result.gammas, full_result.gammas)
