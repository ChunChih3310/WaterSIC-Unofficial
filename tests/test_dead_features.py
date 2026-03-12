import torch

from watersic.stats.dead_features import DeadFeatureReport, detect_dead_features, expand_columns, prune_columns


def test_dead_features_are_removed_and_reinserted() -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    variances = torch.tensor([1.0, 1e-6, 2.0])
    report = detect_dead_features(variances, tau=1e-3)
    reduced = prune_columns(weight, report)
    restored = expand_columns(reduced, report)
    assert report.keep_indices.tolist() == [0, 2]
    assert torch.allclose(restored[:, 1], torch.zeros(2))
    assert torch.allclose(restored[:, [0, 2]], weight[:, [0, 2]])
