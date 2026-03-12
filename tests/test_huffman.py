import math

from watersic.utils.huffman import canonical_huffman_report, empirical_entropy


def test_huffman_average_code_length_matches_simple_case() -> None:
    symbols = [0, 0, 0, 1, 1, 2]
    report = canonical_huffman_report(symbols)
    assert math.isclose(report.average_code_length_bits, 1.5, rel_tol=0.0, abs_tol=1e-6)
    assert report.code_lengths == {0: 1, 1: 2, 2: 2}


def test_empirical_entropy_single_symbol_is_zero() -> None:
    assert empirical_entropy([7, 7, 7, 7]) == 0.0
