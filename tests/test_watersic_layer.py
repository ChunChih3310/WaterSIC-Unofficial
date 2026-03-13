import torch

from watersic.quant.watersic_layer import LayerQuantizationConfig, LayerStatistics, prepare_layer_problem


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
