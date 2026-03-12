from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LayerReport:
    name: str
    kind: str
    target_bitwidth: float
    achieved_bitwidth: float
    raw_bitwidth: float
    entropy_bitwidth: float
    huffman_bitwidth: float
    side_information_bitwidth: float
    weighted_error: float
    applied_damping: float


@dataclass
class RunReport:
    timestamp: str
    git_commit: str
    environment_name: str
    device: str
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    quant_config_path: str
    eval_config_path: str | None
    sequence_length: int
    calibration_sequences: int
    target_global_bitwidth: float
    achieved_global_bitwidth: float
    raw_average_bitwidth: float
    entropy_average_bitwidth: float
    huffman_average_bitwidth: float
    side_information_overhead: float
    perplexity: float | None = None
    notes: list[str] = field(default_factory=list)
    layers: list[LayerReport] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
