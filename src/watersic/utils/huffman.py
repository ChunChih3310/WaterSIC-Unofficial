from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HuffmanReport:
    num_symbols: int
    entropy_bits: float
    average_code_length_bits: float
    code_lengths: dict[int, int]


class _Node:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq: int, symbol: int | None = None, left: "_Node | None" = None, right: "_Node | None" = None) -> None:
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right


def empirical_entropy(symbols: Iterable[int]) -> float:
    counts = Counter(symbols)
    return entropy_from_counts(counts)


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def canonical_huffman_report(symbols: Iterable[int]) -> HuffmanReport:
    counts = Counter(symbols)
    total = sum(counts.values())
    if total == 0:
        return HuffmanReport(num_symbols=0, entropy_bits=0.0, average_code_length_bits=0.0, code_lengths={})
    if len(counts) == 1:
        only_symbol = next(iter(counts))
        return HuffmanReport(num_symbols=1, entropy_bits=0.0, average_code_length_bits=1.0, code_lengths={int(only_symbol): 1})

    heap: list[tuple[int, int, _Node]] = []
    for order, (symbol, freq) in enumerate(sorted(counts.items())):
        heapq.heappush(heap, (freq, order, _Node(freq=freq, symbol=int(symbol))))

    next_order = len(heap)
    while len(heap) > 1:
        freq_a, _, left = heapq.heappop(heap)
        freq_b, _, right = heapq.heappop(heap)
        parent = _Node(freq=freq_a + freq_b, left=left, right=right)
        heapq.heappush(heap, (parent.freq, next_order, parent))
        next_order += 1

    (_, _, root) = heap[0]
    code_lengths: dict[int, int] = {}

    def visit(node: _Node, depth: int) -> None:
        if node.symbol is not None:
            code_lengths[node.symbol] = max(depth, 1)
            return
        assert node.left is not None and node.right is not None
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(root, 0)
    average_code_length = sum(counts[symbol] * code_lengths[int(symbol)] for symbol in counts) / total
    return HuffmanReport(
        num_symbols=len(counts),
        entropy_bits=entropy_from_counts(counts),
        average_code_length_bits=average_code_length,
        code_lengths=code_lengths,
    )
