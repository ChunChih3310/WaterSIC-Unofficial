from __future__ import annotations

from dataclasses import dataclass

import torch

from watersic.data.wikitext2 import _wikitext2_cache_path, load_wikitext2_blocks


@dataclass
class _DummyTokenizer:
    name_or_path: str = "dummy-tokenizer"
    is_fast: bool = True
    vocab_size: int = 256
    model_max_length: int = 4096
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert not add_special_tokens
        token_ids = torch.arange(24, dtype=torch.long).unsqueeze(0)
        return {"input_ids": token_ids}


def test_wikitext2_blocks_uses_repo_local_cache(monkeypatch) -> None:
    tokenizer = _DummyTokenizer()
    cache_path = _wikitext2_cache_path(tokenizer, split="train", sequence_length=8)
    if cache_path.exists():
        cache_path.unlink()

    calls = {"count": 0}

    def fake_load_dataset(name: str, subset: str, *, split: str):
        calls["count"] += 1
        assert name == "wikitext"
        assert subset == "wikitext-2-raw-v1"
        assert split == "train"
        return [{"text": "alpha"}, {"text": "beta"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    first = load_wikitext2_blocks(tokenizer, split="train", sequence_length=8)
    assert calls["count"] == 1
    assert cache_path.exists()
    assert tuple(first.input_ids.shape) == (3, 8)

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("cache was not reused")

    monkeypatch.setattr("datasets.load_dataset", fail_load_dataset)
    second = load_wikitext2_blocks(tokenizer, split="train", sequence_length=8, limit_sequences=2)
    assert calls["count"] == 1
    assert tuple(second.input_ids.shape) == (2, 8)
    assert torch.equal(second.input_ids, first.input_ids[:2])

    cache_path.unlink()
