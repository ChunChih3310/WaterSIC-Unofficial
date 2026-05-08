from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from watersic.models.registry import LinearModuleSpec
from watersic.quant.attention_mixing import (
    _collect_module_inputs,
    _module_input_relative_mse,
    optimize_attention_stage_mixing,
)
from watersic.quant.watersic_layer import LayerQuantizationConfig, LayerStatistics


class _CalibrationDataset:
    def __init__(self, batches: list[torch.Tensor]) -> None:
        self._batches = batches

    def batches(self, batch_size: int):
        assert batch_size == 1
        yield from self._batches


class _TinySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _TinyLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _TinySelfAttention(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(self.input_layernorm(x))


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(hidden_dim)])


class _TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.model = _TinyBackbone(hidden_dim)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


def _stats(hidden_dim: int) -> LayerStatistics:
    sigma = torch.eye(hidden_dim, dtype=torch.float64)
    return LayerStatistics(
        sigma_x=sigma,
        sigma_x_hat=sigma,
        sigma_x_x_hat=sigma,
        sigma_x_weighted=sigma,
        sigma_x_hat_weighted=sigma,
        sigma_x_x_hat_weighted=sigma,
        variances=torch.diag(sigma),
    )


def test_optimize_attention_stage_mixing_cleans_once_after_search(monkeypatch) -> None:
    torch.manual_seed(0)
    model = _TinyModel().eval()
    reference_model = _TinyModel().eval()
    hidden_dim = model.model.layers[0].self_attn.q_proj.in_features
    qkv_specs = [
        LinearModuleSpec(0, "model.layers.0", "self_attn.q_proj", "q_proj"),
        LinearModuleSpec(0, "model.layers.0", "self_attn.k_proj", "k_proj"),
        LinearModuleSpec(0, "model.layers.0", "self_attn.v_proj", "v_proj"),
    ]
    o_proj_spec = LinearModuleSpec(0, "model.layers.0", "self_attn.o_proj", "o_proj")
    stats_map = {spec.full_path: _stats(hidden_dim) for spec in qkv_specs}
    dataset = _CalibrationDataset([torch.tensor([[1, 2, 3]], dtype=torch.long)])

    monkeypatch.setattr(
        "watersic.quant.attention_mixing._collect_module_inputs",
        lambda *args, **kwargs: [torch.ones((1, 3, hidden_dim), dtype=torch.float32)],
    )
    monkeypatch.setattr(
        "watersic.quant.attention_mixing._module_input_relative_mse",
        lambda *args, **kwargs: 0.25,
    )

    def fake_quantize(*args, selected_cs=None, **kwargs):
        stage_summary = {
            "search_mode": "fixed_c_reuse" if selected_cs is not None else "per_matrix_rate_search",
            "achieved_rates": [],
            "achieved_stage_effective_rate": 0.0,
            "achieved_stage_entropy_rate": 0.0,
            "achieved_stage_huffman_rate": 0.0,
        }
        candidate_cs = {spec.full_path: 0.5 for spec in qkv_specs}
        return candidate_cs, stage_summary

    monkeypatch.setattr("watersic.quant.attention_mixing._quantize_qkv_stage_candidate", fake_quantize)

    gc_calls = {"count": 0}
    empty_cache_calls = {"count": 0}
    monkeypatch.setattr(
        "watersic.quant.attention_mixing.gc.collect",
        lambda: gc_calls.__setitem__("count", gc_calls["count"] + 1),
    )
    monkeypatch.setattr(
        "watersic.quant.attention_mixing.torch.cuda.empty_cache",
        lambda: empty_cache_calls.__setitem__("count", empty_cache_calls["count"] + 1),
    )

    _, _, audit = optimize_attention_stage_mixing(
        model,
        reference_model,
        qkv_specs,
        o_proj_spec,
        stats_map,
        dataset,
        LayerQuantizationConfig(
            target_rate=3.0,
            use_attention_weighting=True,
            use_adaptive_mixing=True,
            optimize_adaptive_mixing=True,
            golden_section_iterations=2,
        ),
        stage_remaining_budget=100.0,
        stage_remaining_weights=100,
        calibration_batch_size=1,
        device=torch.device("cpu"),
        reference_device=torch.device("cpu"),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert audit["enabled"] is True
    assert audit["timing"]["objective_evaluations"] > 1
    assert gc_calls["count"] == 1
    assert empty_cache_calls["count"] == 0


def test_module_input_relative_mse_matches_materialized_inputs() -> None:
    torch.manual_seed(0)
    reference_model = _TinyModel().eval()
    candidate_model = _TinyModel().eval()
    with torch.no_grad():
        candidate_model.model.layers[0].self_attn.q_proj.weight.mul_(1.1)

    calibration_batches = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6]], dtype=torch.long),
    ]
    module_path = "model.layers.0.self_attn.o_proj"
    device = torch.device("cpu")

    reference_inputs = _collect_module_inputs(
        reference_model,
        calibration_batches,
        module_path,
        device=device,
    )
    candidate_inputs = _collect_module_inputs(
        candidate_model,
        calibration_batches,
        module_path,
        device=device,
    )
    manual_numerator = 0.0
    manual_denominator = 0.0
    for reference_input, candidate_input in zip(reference_inputs, candidate_inputs, strict=True):
        reference_fp64 = reference_input.to(torch.float64)
        candidate_fp64 = candidate_input.to(torch.float64)
        manual_numerator += float((candidate_fp64 - reference_fp64).square().sum().item())
        manual_denominator += float(reference_fp64.square().sum().item())
    manual_mse = manual_numerator / max(manual_denominator, 1e-12)

    streamed_mse = _module_input_relative_mse(
        candidate_model,
        calibration_batches,
        reference_inputs,
        module_path,
        device=device,
    )

    assert streamed_mse == manual_mse
