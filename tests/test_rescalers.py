import torch

from watersic.quant.rescalers import optimize_diagonal_rescalers


def test_rescaler_optimization_does_not_worsen_objective() -> None:
    torch.manual_seed(0)
    target = torch.randn(16, 8)
    weight0 = target + 0.05 * torch.randn(16, 8)
    sigma_x_hat = torch.eye(8, dtype=torch.float64)
    target_cross = target.to(torch.float64) @ sigma_x_hat
    result = optimize_diagonal_rescalers(weight0, target_cross, sigma_x_hat, max_iters=6)
    assert result.final_objective <= result.initial_objective + 1e-9
