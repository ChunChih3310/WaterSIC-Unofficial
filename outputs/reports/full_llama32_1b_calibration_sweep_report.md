# Full-Model Llama-3.2-1B Calibration Sweep Report

Date: 2026-03-15

## Status

The first completed calibration-size upgrade is done on the current best validated path:

- model: `Llama-3.2-1B`
- pipeline: `reference_stats: true`, fixed residual correction, staged same-layer stat refresh, rescalers enabled, adaptive mixing disabled
- completed sweep points:
  - `8` chunks: completed
  - `16` chunks: completed
  - `32` chunks: not run yet

## Completed Points

| Run | Config | Calib Chunks | Effective Bits | Entropy Bits | Huffman Bits | Side Bits | Quantized PPL | Delta vs Paper |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8`-chunk anchor | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `8` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `15.7029` | `+5.1329` |
| `16`-chunk run | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml` | `16` | `2.9984` | `2.9865` | `3.0371` | `0.0119` | `12.4574` | `+1.8874` |

## Finished `16`-Chunk Run

- run: `llama32_1b_full_3p0bit_reftrue_rescaler_calib16`
- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml`
- report bundle:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.md`
- artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_calib16/`
- log:
  - `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_calib16_20260315_133857.log`

Core metrics:

- baseline WikiText-2 PPL: `9.7041`
- quantized WikiText-2 PPL: `12.4574`
- effective bits: `2.9984`
- entropy bits: `2.9865`
- Huffman bits: `3.0371`
- side-information overhead: `0.0119`
- total runtime: `22502.96s`
- quantization runtime: `22405.30s`
- peak GPU memory: `18.65 GiB`
- quantization anomalies: none

## Distortion Change From `8` To `16` Chunks

| Kind | `8` chunks | `16` chunks | Delta |
| --- | ---: | ---: | ---: |
| `o_proj` mean rel MSE | `0.3170` | `0.2382` | `-0.0788` |
| `down_proj` mean rel MSE | `0.3023` | `0.1709` | `-0.1313` |
| `q_proj` mean rel MSE | `0.0696` | `0.0603` | `-0.0093` |
| `k_proj` mean rel MSE | `0.0983` | `0.0854` | `-0.0128` |
| `v_proj` mean rel MSE | `0.0876` | `0.0756` | `-0.0120` |

Worst layers by relative weight MSE:

- `8` chunks:
  - `model.layers.11.self_attn.o_proj` `0.4763`
  - `model.layers.13.self_attn.o_proj` `0.4596`
  - `model.layers.12.self_attn.o_proj` `0.4178`
  - `model.layers.4.self_attn.o_proj` `0.4032`
  - `model.layers.0.self_attn.o_proj` `0.3936`
- `16` chunks:
  - `model.layers.13.self_attn.o_proj` `0.3427`
  - `model.layers.11.self_attn.o_proj` `0.3223`
  - `model.layers.0.self_attn.o_proj` `0.3108`
  - `model.layers.12.self_attn.o_proj` `0.3035`
  - `model.layers.4.self_attn.o_proj` `0.2935`

Interpretation:

1. Increasing calibration from `8` to `16` chunks materially improved the best validated path.
2. The biggest gains are still on the previously dominant residual-path kinds, especially `o_proj` and `down_proj`.
3. The dominant outliers remain late-layer `o_proj`, but their magnitudes are much lower.
4. The paper gap fell from `+5.1329` to `+1.8874`, which makes calibration look like the strongest current limiter on the validated rescaler-only path.

## Safe Runtime Change

This sweep uses the repo-local WikiText-2 token-block cache added in commit `6173f21`.

What changed:

- tokenized non-overlapping `2048`-token blocks are cached under `outputs/stats/wikitext2_cache/`

Why it is safe:

- the dataset split is unchanged
- the tokenizer call is unchanged
- concatenation and chunking semantics are unchanged
- later runs only reuse the exact saved block tensors

Observed effect:

- it removes repeated tokenization cost from follow-up calibration and evaluation launches
- it does not change the experiment itself

## Conclusion

The `16`-chunk run is the new best completed `Llama-3.2-1B` point in the repo:

- effective bits: `2.9984`
- quantized WikiText-2 PPL: `12.4574`

The next highest-signal move is to run `32` chunks on the same rescaler-only path before spending more time on adaptive-mixing changes.
