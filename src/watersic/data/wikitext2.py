from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from watersic.utils.path_guard import repo_path


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


def _tokenizer_cache_identity(tokenizer) -> dict[str, object]:
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
        "class_name": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }


def _wikitext2_cache_path(tokenizer, *, split: str, sequence_length: int) -> Path:
    identity = _tokenizer_cache_identity(tokenizer)
    payload = {
        "dataset": "wikitext-2-raw-v1",
        "split": split,
        "sequence_length": sequence_length,
        "tokenizer": identity,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return repo_path("outputs", "stats", "wikitext2_cache", f"{split}_seq{sequence_length}_{digest}.pt")


def load_wikitext2_blocks(
    tokenizer,
    *,
    split: str,
    sequence_length: int,
    limit_sequences: int | None = None,
) -> TokenBlockDataset:
    from datasets import load_dataset

    cache_path = _wikitext2_cache_path(tokenizer, split=split, sequence_length=sequence_length)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        blocks = payload["input_ids"].to(torch.long)
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(torch.long)
        total_tokens = (encoded.numel() // sequence_length) * sequence_length
        blocks = encoded[:total_tokens].view(-1, sequence_length).contiguous()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": blocks,
                "split": split,
                "sequence_length": sequence_length,
                "tokenizer": _tokenizer_cache_identity(tokenizer),
            },
            cache_path,
        )
    if limit_sequences is not None:
        blocks = blocks[:limit_sequences]
    return TokenBlockDataset(input_ids=blocks.contiguous(), split=split, sequence_length=sequence_length)
