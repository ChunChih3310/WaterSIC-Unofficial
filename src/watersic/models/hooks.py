from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class HookStore:
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    layer_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()
        self.layer_inputs.clear()


class ModuleInputCollector:
    def __init__(
        self,
        model: nn.Module,
        module_paths: list[str],
        layer_paths: list[str] | None = None,
        *,
        collect_outputs: bool = False,
    ) -> None:
        self.model = model
        self.module_paths = module_paths
        self.layer_paths = layer_paths or []
        self.collect_outputs = collect_outputs
        self.store = HookStore()
        self._stack = ExitStack()

    def __enter__(self) -> HookStore:
        self.store.clear()
        for path in self.module_paths:
            module = self.model.get_submodule(path)
            handle = module.register_forward_pre_hook(self._make_input_hook(path))
            self._stack.callback(handle.remove)
            if self.collect_outputs:
                handle = module.register_forward_hook(self._make_output_hook(path))
                self._stack.callback(handle.remove)
        for path in self.layer_paths:
            layer = self.model.get_submodule(path)
            handle = layer.register_forward_pre_hook(self._make_layer_hook(path))
            self._stack.callback(handle.remove)
        return self.store

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stack.close()

    def _make_input_hook(self, path: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            self.store.inputs[path] = args[0].detach().to(torch.float32).cpu()

        return hook

    def _make_output_hook(self, path: str):
        def hook(_module: nn.Module, _args: tuple[torch.Tensor, ...], output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.store.outputs[path] = tensor.detach().to(torch.float32).cpu()

        return hook

    def _make_layer_hook(self, path: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            self.store.layer_inputs[path] = args[0].detach().to(torch.float32).cpu()

        return hook
