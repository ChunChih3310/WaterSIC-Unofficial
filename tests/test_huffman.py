import math

import torch

from watersic.eval.metrics import estimate_bitrate_metrics
from watersic.utils.huffman import canonical_huffman_report, empirical_entropy


def test_huffman_average_code_length_matches_simple_case() -> None:
    symbols = [0, 0, 0, 1, 1, 2]
    report = canonical_huffman_report(symbols)
    assert math.isclose(report.average_code_length_bits, 1.5, rel_tol=0.0, abs_tol=1e-6)
    assert report.shortest_code_length_bits == 1
    assert report.longest_code_length_bits == 2
    assert report.code_lengths == {0: 1, 1: 2, 2: 2}


def test_empirical_entropy_single_symbol_is_zero() -> None:
    assert empirical_entropy([7, 7, 7, 7]) == 0.0


def test_huffman_report_handles_empty_and_single_symbol_edge_cases() -> None:
    empty = canonical_huffman_report([])
    assert empty.average_code_length_bits == 0.0
    assert empty.shortest_code_length_bits == 0
    assert empty.longest_code_length_bits == 0

    single = canonical_huffman_report([7, 7, 7, 7])
    assert single.average_code_length_bits == 1.0
    assert single.shortest_code_length_bits == 1
    assert single.longest_code_length_bits == 1


def test_bitrate_metrics_include_huffman_length_bounds() -> None:
    metrics = estimate_bitrate_metrics(torch.tensor([[0, 0, 0], [1, 1, 2]], dtype=torch.int64))
    assert math.isclose(metrics.huffman_average_bitwidth, 1.5, rel_tol=0.0, abs_tol=1e-6)
    assert metrics.huffman_shortest_symbol_length_bits == 1
    assert metrics.huffman_longest_symbol_length_bits == 2
