from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenBlockDataset:
    input_ids: torch.Tensor
    split: str
    sequence_length: int

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            yield self.input_ids[start : start + batch_size]


def load_wikitext2_blocks(
    tokenizer,
    *,
    split: str,
    sequence_length: int,
    limit_sequences: int | None = None,
) -> TokenBlockDataset:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    total_tokens = (encoded.numel() // sequence_length) * sequence_length
    blocks = encoded[:total_tokens].view(-1, sequence_length)
    if limit_sequences is not None:
        blocks = blocks[:limit_sequences]
    return TokenBlockDataset(input_ids=blocks.contiguous(), split=split, sequence_length=sequence_length)
