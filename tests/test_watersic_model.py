from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from watersic.models.registry import LinearModuleSpec
from watersic.stats.attention_weighting import token_importance_from_attention, weighted_second_moment
from watersic.quant.watersic_model import collect_layer_statistics


class _CalibrationDataset:
    def __init__(self, batches: list[torch.Tensor]) -> None:
        self._batches = batches

    def batches(self, batch_size: int):
        assert batch_size == 1
        yield from self._batches


class _FakeMLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = _FakeMLP(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp.gate_proj(self.input_layernorm(x))


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_dim)])


class _FakeModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.model = _FakeBackbone(hidden_dim)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, output_attentions: bool = False):
        del use_cache
        del output_attentions
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(attentions=None)


class _FakeSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, *, output_attentions: bool = False):
        hidden = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        if not output_attentions:
            return hidden
        seq_len = x.shape[1]
        attention = torch.tril(torch.ones((x.shape[0], 1, seq_len, seq_len), dtype=x.dtype, device=x.device))
        attention = attention / attention.sum(dim=-1, keepdim=True)
        return hidden, attention


class _CountingLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.calls = 0
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = _FakeMLP(hidden_dim)

    def forward(self, x: torch.Tensor, *, output_attentions: bool = False):
        del output_attentions
        self.calls += 1
        return self.mlp.gate_proj(self.input_layernorm(x))


class _FakeAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _FakeSelfAttention(hidden_dim)

    def forward(self, x: torch.Tensor, *, output_attentions: bool = False):
        normed = self.input_layernorm(x)
        return self.self_attn(normed, output_attentions=output_attentions)


class _FakeAttentionBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeAttentionLayer(hidden_dim), _CountingLayer(hidden_dim)])


class _FakeAttentionModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.model = _FakeAttentionBackbone(hidden_dim)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, output_attentions: bool = False):
        del use_cache
        x = self.embed(input_ids)
        for layer in self.model.layers:
            layer_output = layer(x, output_attentions=output_attentions)
            x = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return SimpleNamespace(attentions=None)


def test_collect_layer_statistics_matches_manual_second_moments_and_cleans_once(monkeypatch) -> None:
    torch.manual_seed(0)
    model = _FakeModel().eval()
    batches = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6]], dtype=torch.long),
    ]
    dataset = _CalibrationDataset(batches)
    spec = LinearModuleSpec(
        layer_index=0,
        layer_path="model.layers.0",
        module_path="mlp.gate_proj",
        kind="gate_proj",
    )

    gc_calls = {"count": 0}
    monkeypatch.setattr("watersic.quant.watersic_model.gc.collect", lambda: gc_calls.__setitem__("count", gc_calls["count"] + 1))

    stats = collect_layer_statistics(
        model,
        dataset,
        [spec],
        device=torch.device("cpu"),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        calibration_batch_size=1,
    )[spec.full_path]

    manual_inputs = []
    for batch in batches:
        x = model.embed(batch)
        gate_input = model.model.layers[0].input_layernorm(x).detach().to(torch.float64)
        manual_inputs.append(gate_input.reshape(-1, gate_input.shape[-1]))
    stacked = torch.cat(manual_inputs, dim=0)
    manual_second_moment = stacked.transpose(0, 1) @ stacked / stacked.shape[0]

    torch.testing.assert_close(stats.sigma_x.to(torch.float64), manual_second_moment)
    torch.testing.assert_close(stats.sigma_x_hat.to(torch.float64), manual_second_moment)
    torch.testing.assert_close(stats.sigma_x_x_hat.to(torch.float64), manual_second_moment)
    torch.testing.assert_close(stats.variances.to(torch.float64), torch.diag(manual_second_moment))
    assert gc_calls["count"] == 1


def test_collect_layer_statistics_stops_after_target_layer_and_preserves_attention_weighted_stats() -> None:
    torch.manual_seed(0)
    model = _FakeAttentionModel().eval()
    batches = [torch.tensor([[1, 2, 3]], dtype=torch.long)]
    dataset = _CalibrationDataset(batches)
    spec = LinearModuleSpec(
        layer_index=0,
        layer_path="model.layers.0",
        module_path="self_attn.q_proj",
        kind="q_proj",
    )

    stats = collect_layer_statistics(
        model,
        dataset,
        [spec],
        device=torch.device("cpu"),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        calibration_batch_size=1,
    )[spec.full_path]

    embedded = model.embed(batches[0])
    q_input = model.model.layers[0].input_layernorm(embedded).detach().to(torch.float64)
    seq_len = q_input.shape[1]
    attention = torch.tril(torch.ones((q_input.shape[0], 1, seq_len, seq_len), dtype=q_input.dtype))
    attention = attention / attention.sum(dim=-1, keepdim=True)
    token_weights = token_importance_from_attention(attention)
    manual_weighted = weighted_second_moment(q_input, token_weights)

    torch.testing.assert_close(stats.sigma_x.to(torch.float64), weighted_second_moment(q_input, torch.ones_like(token_weights)))
    torch.testing.assert_close(stats.sigma_x_weighted.to(torch.float64), manual_weighted)
    torch.testing.assert_close(stats.sigma_x_hat_weighted.to(torch.float64), manual_weighted)
    torch.testing.assert_close(stats.sigma_x_x_hat_weighted.to(torch.float64), manual_weighted)
    assert model.model.layers[1].calls == 0
