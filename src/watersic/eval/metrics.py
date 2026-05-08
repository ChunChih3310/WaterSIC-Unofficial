from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from watersic.utils.huffman import canonical_huffman_report, empirical_entropy


@dataclass(frozen=True)
class BitrateMetrics:
    num_weights: int
    raw_average_bitwidth: float
    entropy_average_bitwidth: float
    huffman_average_bitwidth: float
    huffman_shortest_symbol_length_bits: int
    huffman_longest_symbol_length_bits: int
    side_information_average_bitwidth: float
    final_effective_average_bitwidth: float


def estimate_effective_average_bitwidth(symbols: torch.Tensor, *, side_information_bits: float = 0.0) -> float:
    flat = symbols.reshape(-1).to(torch.int64).cpu()
    flat_list = [int(x) for x in flat.tolist()]
    num_weights = len(flat_list)
    entropy_avg = empirical_entropy(flat_list)
    side_avg = side_information_bits / max(num_weights, 1)
    return entropy_avg + side_avg


def _fixed_width_for_signed_integers(symbols: torch.Tensor) -> float:
    if symbols.numel() == 0:
        return 0.0
    max_abs = int(symbols.abs().max().item())
    if max_abs == 0:
        return 1.0
    return 1.0 + math.ceil(math.log2(max_abs + 1))


def estimate_bitrate_metrics(symbols: torch.Tensor, *, side_information_bits: float = 0.0) -> BitrateMetrics:
    flat = symbols.reshape(-1).to(torch.int64).cpu()
    flat_list = [int(x) for x in flat.tolist()]
    num_weights = len(flat_list)
    raw_avg = _fixed_width_for_signed_integers(flat)
    entropy_avg = empirical_entropy(flat_list)
    huffman = canonical_huffman_report(flat_list)
    side_avg = side_information_bits / max(num_weights, 1)
    return BitrateMetrics(
        num_weights=num_weights,
        raw_average_bitwidth=raw_avg,
        entropy_average_bitwidth=entropy_avg,
        huffman_average_bitwidth=huffman.average_code_length_bits,
        huffman_shortest_symbol_length_bits=huffman.shortest_code_length_bits,
        huffman_longest_symbol_length_bits=huffman.longest_code_length_bits,
        side_information_average_bitwidth=side_avg,
        final_effective_average_bitwidth=entropy_avg + side_avg,
    )
