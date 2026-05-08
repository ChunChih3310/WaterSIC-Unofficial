import torch

from watersic.quant.rate_search import binary_search_c
from watersic.quant.watersic_layer import (
    LayerQuantizationConfig,
    LayerStatistics,
    prepare_layer_problem,
    quantize_linear_layer,
    quantize_prepared_layer_problem_objective,
)


def test_prepare_layer_problem_recovers_plain_watersic_target_y() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    chol = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float64)
    sigma_x = chol @ chol.transpose(0, 1)
    stats = LayerStatistics(sigma_x=sigma_x)
    config = LayerQuantizationConfig(
        target_rate=4.0,
        damping=0.0,
        dead_feature_tau=0.0,
        use_activation_drift=False,
        use_residual_correction=False,
        use_attention_weighting=False,
        use_adaptive_mixing=False,
    )

    problem = prepare_layer_problem(weight, stats, config, kind="q_proj")

    assert torch.allclose(problem.target_y, weight @ chol, atol=1e-8, rtol=1e-8)


def test_objective_quantization_matches_full_quantization_result() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 4, dtype=torch.float64)
    sigma = torch.randn(4, 4, dtype=torch.float64)
    sigma = sigma @ sigma.transpose(0, 1) + 1e-3 * torch.eye(4, dtype=torch.float64)
    stats = LayerStatistics(
        sigma_x=sigma,
        sigma_x_hat=sigma,
        sigma_x_x_hat=sigma,
    )
    config = LayerQuantizationConfig(
        target_rate=3.0,
        damping=1e-4,
        dead_feature_tau=0.0,
        use_activation_drift=True,
        use_residual_correction=False,
        use_attention_weighting=False,
        use_adaptive_mixing=False,
        max_rescaler_iters=4,
    )

    full_result = quantize_linear_layer(weight, stats, config, kind="q_proj")
    problem = prepare_layer_problem(weight, stats, config, kind="q_proj")
    objective_result = quantize_prepared_layer_problem_objective(
        weight,
        problem,
        config,
        search=full_result.search,
    )

    torch.testing.assert_close(objective_result.quantized_weight.to(torch.float64), full_result.quantized_weight.to(torch.float64))
    assert objective_result.selected_c == float(full_result.search.selected_c)
    assert objective_result.effective_rate == float(full_result.bitrate.final_effective_average_bitwidth)
    assert objective_result.entropy_rate == float(full_result.bitrate.entropy_average_bitwidth)
    assert objective_result.huffman_rate == float(full_result.bitrate.huffman_average_bitwidth)


def test_objective_quantization_matches_minimal_search_result() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 4, dtype=torch.float64)
    sigma = torch.randn(4, 4, dtype=torch.float64)
    sigma = sigma @ sigma.transpose(0, 1) + 1e-3 * torch.eye(4, dtype=torch.float64)
    stats = LayerStatistics(
        sigma_x=sigma,
        sigma_x_hat=sigma,
        sigma_x_x_hat=sigma,
    )
    config = LayerQuantizationConfig(
        target_rate=3.0,
        damping=1e-4,
        dead_feature_tau=0.0,
        use_activation_drift=True,
        use_residual_correction=False,
        use_attention_weighting=False,
        use_adaptive_mixing=False,
        max_rescaler_iters=4,
    )

    full_result = quantize_linear_layer(weight, stats, config, kind="q_proj")
    problem = prepare_layer_problem(weight, stats, config, kind="q_proj")
    search = binary_search_c(
        problem.target_y,
        problem.cholesky,
        config.target_rate,
        side_information_bits=problem.side_information_bits,
        num_iterations=config.binary_search_iterations,
        row_sample_fraction=config.row_sample_fraction,
        seed=config.seed,
        use_lmmse=config.use_lmmse,
        spacing_strategy=config.spacing_strategy,
        collect_quantization_details=False,
    )

    objective_result = quantize_prepared_layer_problem_objective(
        weight,
        problem,
        config,
        search=search,
    )

    torch.testing.assert_close(objective_result.quantized_weight.to(torch.float64), full_result.quantized_weight.to(torch.float64))
    assert objective_result.selected_c == float(full_result.search.selected_c)
    assert objective_result.effective_rate == float(full_result.bitrate.final_effective_average_bitwidth)
    assert objective_result.entropy_rate == float(full_result.bitrate.entropy_average_bitwidth)
    assert objective_result.huffman_rate == float(full_result.bitrate.huffman_average_bitwidth)
