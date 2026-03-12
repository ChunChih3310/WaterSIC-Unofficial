from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LinearModuleSpec:
    layer_index: int
    layer_path: str
    module_path: str
    kind: str

    @property
    def full_path(self) -> str:
        return f"{self.layer_path}.{self.module_path}"


TARGET_ORDER = (
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
)


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported model structure for {type(model).__name__}")


def iter_linear_module_specs(model: nn.Module) -> list[LinearModuleSpec]:
    layers = get_transformer_layers(model)
    specs: list[LinearModuleSpec] = []
    for layer_index in range(len(layers)):
        layer_path = f"model.layers.{layer_index}"
        layer = model.get_submodule(layer_path)
        for relative_path, kind in TARGET_ORDER:
            module = layer.get_submodule(relative_path)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {layer_path}.{relative_path}, got {type(module).__name__}")
            specs.append(LinearModuleSpec(layer_index=layer_index, layer_path=layer_path, module_path=relative_path, kind=kind))
    return specs


def group_specs_by_layer(specs: list[LinearModuleSpec]) -> dict[int, list[LinearModuleSpec]]:
    grouped: dict[int, list[LinearModuleSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.layer_index, []).append(spec)
    return grouped


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name in (None, "auto"):
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(model_config: dict, *, device_map: str | None = None):
    model_kwargs = {
        "revision": model_config.get("model_revision"),
        "torch_dtype": resolve_torch_dtype(model_config.get("dtype", "auto")),
        "device_map": device_map,
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "attn_implementation": model_config.get("attn_implementation"),
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    tokenizer_kwargs = {
        "revision": model_config.get("tokenizer_revision", model_config.get("model_revision")),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    model = AutoModelForCausalLM.from_pretrained(model_config["model_id"], **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.get("tokenizer_id", model_config["model_id"]), **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
