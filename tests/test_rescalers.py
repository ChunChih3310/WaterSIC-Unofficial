import torch

from watersic.quant.rescalers import (
    _disabled_rescalers_prepared,
    _optimize_diagonal_rescalers_prepared,
    disabled_rescalers,
    optimize_diagonal_rescalers,
    prepare_rescaler_problem,
)


def test_rescaler_optimization_does_not_worsen_objective() -> None:
    torch.manual_seed(0)
    target = torch.randn(16, 8)
    weight0 = target + 0.05 * torch.randn(16, 8)
    sigma_x_hat = torch.eye(8, dtype=torch.float64)
    target_cross = target.to(torch.float64) @ sigma_x_hat
    result = optimize_diagonal_rescalers(weight0, target_cross, sigma_x_hat, max_iters=6)
    assert result.final_objective <= result.initial_objective + 1e-9


def test_prepared_rescaler_helpers_match_public_paths() -> None:
    torch.manual_seed(0)
    target = torch.randn(8, 4)
    weight0 = target + 0.05 * torch.randn(8, 4)
    sigma_x_hat = torch.eye(4, dtype=torch.float64)
    target_cross = target.to(torch.float64) @ sigma_x_hat
    initial_column_scale = torch.rand(4, dtype=torch.float64) + 0.5
    prepared = prepare_rescaler_problem(target_cross, sigma_x_hat)

    public_enabled = optimize_diagonal_rescalers(
        weight0,
        target_cross,
        sigma_x_hat,
        initial_column_scale=initial_column_scale,
        max_iters=4,
    )
    prepared_enabled = _optimize_diagonal_rescalers_prepared(
        weight0,
        prepared,
        initial_column_scale=initial_column_scale,
        max_iters=4,
    )
    torch.testing.assert_close(public_enabled.row_scale, prepared_enabled.row_scale)
    torch.testing.assert_close(public_enabled.column_scale, prepared_enabled.column_scale)
    torch.testing.assert_close(public_enabled.reconstructed, prepared_enabled.reconstructed)
    assert public_enabled.initial_objective == prepared_enabled.initial_objective
    assert public_enabled.final_objective == prepared_enabled.final_objective

    public_disabled = disabled_rescalers(
        weight0,
        target_cross,
        sigma_x_hat,
        initial_column_scale=initial_column_scale,
    )
    prepared_disabled = _disabled_rescalers_prepared(
        weight0,
        prepared,
        initial_column_scale=initial_column_scale,
    )
    torch.testing.assert_close(public_disabled.row_scale, prepared_disabled.row_scale)
    torch.testing.assert_close(public_disabled.column_scale, prepared_disabled.column_scale)
    torch.testing.assert_close(public_disabled.reconstructed, prepared_disabled.reconstructed)
    assert public_disabled.initial_objective == prepared_disabled.initial_objective
    assert public_disabled.final_objective == prepared_disabled.final_objective
