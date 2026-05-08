from __future__ import annotations

from dataclasses import dataclass

from .wikitext2 import TokenBlockDataset


@dataclass(frozen=True)
class CalibrationConfig:
    split: str = "train"
    sequence_length: int = 2048
    num_sequences: int = 32
    batch_size: int = 1


def load_calibration_dataset(tokenizer, config: CalibrationConfig) -> TokenBlockDataset:
    from .wikitext2 import load_wikitext2_blocks

    return load_wikitext2_blocks(
        tokenizer,
        split=config.split,
        sequence_length=config.sequence_length,
        limit_sequences=config.num_sequences,
    )
