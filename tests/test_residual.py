import torch

from watersic.quant.watersic_layer import LayerQuantizationConfig, LayerStatistics, prepare_layer_problem
from watersic.stats.residual import residual_compensation_matrix


def test_residual_compensation_matrix_returns_scaled_delta() -> None:
    sigma_delta = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)

    compensation = residual_compensation_matrix(sigma_delta, scale=0.25)

    assert compensation is not None
    assert torch.allclose(compensation, 0.25 * sigma_delta)


def test_zero_delta_residual_path_matches_no_residual_path() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
    sigma = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    zero_delta = torch.zeros((weight.shape[0], weight.shape[1]), dtype=torch.float64)
    stats = LayerStatistics(
        sigma_x=sigma,
        sigma_x_hat=sigma,
        sigma_x_x_hat=sigma,
        sigma_delta_x_hat=zero_delta,
    )

    no_residual = prepare_layer_problem(
        weight,
        stats,
        LayerQuantizationConfig(
            target_rate=4.0,
            damping=0.0,
            dead_feature_tau=0.0,
            use_activation_drift=True,
            use_residual_correction=False,
            use_attention_weighting=False,
            use_adaptive_mixing=False,
        ),
        kind="o_proj",
    )
    with_zero_delta = prepare_layer_problem(
        weight,
        stats,
        LayerQuantizationConfig(
            target_rate=4.0,
            damping=0.0,
            dead_feature_tau=0.0,
            use_activation_drift=True,
            use_residual_correction=True,
            use_attention_weighting=False,
            use_adaptive_mixing=False,
            residual_scale=1.0,
        ),
        kind="o_proj",
    )

    assert torch.allclose(with_zero_delta.base_target_cross, no_residual.base_target_cross)
    assert with_zero_delta.residual_term is not None
    assert torch.allclose(with_zero_delta.residual_term, torch.zeros_like(with_zero_delta.residual_term))
    assert torch.allclose(with_zero_delta.target_cross, no_residual.target_cross)
    assert torch.allclose(with_zero_delta.target_y, no_residual.target_y)
