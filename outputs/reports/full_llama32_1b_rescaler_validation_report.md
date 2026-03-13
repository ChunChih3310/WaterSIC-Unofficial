# Full-Model Llama-3.2-1B Rescaler Validation

## Scope

- Model: `Llama-3.2-1B`
- Baseline reference run: `full_reftrue_norescaler`
- Validation run: `full_reftrue_rescaler`
- Calibration: WikiText-2 train, `8` chunks, sequence length `2048`
- Evaluation: full WikiText-2 test perplexity
- Shared stable path:
  - `reference_stats: true`
  - fixed residual correction enabled
  - staged same-layer stat refresh enabled
  - adaptive-mixing values still fixed from the validated layer-0 search
- Difference under test:
  - baseline: diagonal rescalers disabled
  - validation: diagonal rescalers enabled with `max_rescaler_iters: 4`

## Headline Result

Diagonal rescalers were stable on the full-model path and materially improved the benchmark:

- baseline quantized PPL: `16.8684`
- rescaler quantized PPL: `15.7029`
- absolute improvement: `-1.1654`

The bitrate stayed effectively unchanged:

| Metric | No Rescaler | Rescaler | Delta |
| --- | ---: | ---: | ---: |
| Effective bits | `2.9984` | `2.9984` | `+0.0000` |
| Entropy bits | `2.9865` | `2.9865` | `+0.0000` |
| Huffman bits | `3.0368` | `3.0368` | `+0.0000` |
| Side-info bits | `0.0119` | `0.0119` | `+0.0000` |
| Baseline PPL | `9.7041` | `9.7041` | `+0.0000` |
| Quantized PPL | `16.8684` | `15.7029` | `-1.1654` |

Paper comparison at `3.00` bits:

- paper PPL: `10.57`
- no-rescaler gap vs paper: `+6.2984`
- rescaler gap vs paper: `+5.1329`
- gap reduction from rescalers alone: `1.1654`

## Distortion Comparison

Per-kind mean relative weight MSE:

| Kind | No Rescaler | Rescaler | Delta |
| --- | ---: | ---: | ---: |
| `o_proj` | `0.3215` | `0.3170` | `-0.0046` |
| `down_proj` | `0.3034` | `0.3023` | `-0.0012` |
| `q_proj` | `0.0696` | `0.0696` | `-0.0001` |
| `k_proj` | `0.0979` | `0.0983` | `+0.0003` |
| `v_proj` | `0.0876` | `0.0876` | `+0.0000` |
| `gate_proj` | `0.0502` | `0.0502` | `-0.0001` |
| `up_proj` | `0.0500` | `0.0499` | `-0.0000` |

Specific before/after comparisons for the main error concentrations:

| Layer | Kind | No Rescaler | Rescaler | Delta |
| --- | --- | ---: | ---: | ---: |
| `model.layers.0.self_attn.o_proj` | `o_proj` | `0.4164` | `0.3936` | `-0.0229` |
| `model.layers.4.self_attn.o_proj` | `o_proj` | `0.4098` | `0.4032` | `-0.0066` |
| `model.layers.11.self_attn.o_proj` | `o_proj` | `0.4860` | `0.4763` | `-0.0097` |
| `model.layers.12.self_attn.o_proj` | `o_proj` | `0.4215` | `0.4178` | `-0.0036` |
| `model.layers.13.self_attn.o_proj` | `o_proj` | `0.4685` | `0.4596` | `-0.0090` |
| `model.layers.0.mlp.down_proj` | `down_proj` | `0.3644` | `0.3635` | `-0.0009` |
| `model.layers.6.mlp.down_proj` | `down_proj` | `0.3364` | `0.3335` | `-0.0028` |
| `model.layers.11.mlp.down_proj` | `down_proj` | `0.3166` | `0.3146` | `-0.0019` |

Worst layers by relative weight MSE:

| Rank | No Rescaler | Rescaler |
| --- | --- | --- |
| 1 | `model.layers.11.self_attn.o_proj` `0.4860` | `model.layers.11.self_attn.o_proj` `0.4763` |
| 2 | `model.layers.13.self_attn.o_proj` `0.4685` | `model.layers.13.self_attn.o_proj` `0.4596` |
| 3 | `model.layers.12.self_attn.o_proj` `0.4215` | `model.layers.12.self_attn.o_proj` `0.4178` |
| 4 | `model.layers.0.self_attn.o_proj` `0.4164` | `model.layers.4.self_attn.o_proj` `0.4032` |
| 5 | `model.layers.4.self_attn.o_proj` `0.4098` | `model.layers.0.self_attn.o_proj` `0.3936` |

## Reference-Stats Coverage Audit

The rescaler run again reported `109 / 112` matrices with effective reference stats. The exact three non-effective matrices were:

- `model.layers.0.self_attn.q_proj`
- `model.layers.0.self_attn.k_proj`
- `model.layers.0.self_attn.v_proj`

All three had `reference_stats_delta_norm = 0.0`. This is expected rather than a bug: they are the first QKV stage in the network, so no earlier quantized layer exists yet to create activation drift at their inputs.

## Runtime

- quantization runtime: `18417.68s`
- total runtime including evaluation: `18525.23s`
- peak memory: `18.65 GiB`
- quantization anomalies: none

## Decision

Diagonal rescalers are now validated on the stable full-model path.

- `o_proj` distortion improved.
- `down_proj` distortion improved slightly.
- final WikiText-2 PPL improved materially.

Decision: keep rescalers enabled for the next full-model quality-recovery run.

Follow-on run already started:

- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml`
- log: `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_20260314_033153.log`
