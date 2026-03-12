from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_model(model_id: str, *, revision: str | None = None, dtype: str = "auto", device_map: str | None = None):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype if dtype != "auto" else None,
        device_map=device_map,
        trust_remote_code=True,
    )


def load_qwen_tokenizer(model_id: str, *, revision: str | None = None):
    return AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
