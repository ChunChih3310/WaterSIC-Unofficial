import torch

from watersic.quant.rate_search import binary_search_c


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
