# Full-Model Llama-3.2-1B Calibration Sweep Report

Date: 2026-03-16

## Status

The mainline calibration sweep on the validated rescaler-only path now has three completed points:

- model: `Llama-3.2-1B`
- pipeline: `reference_stats: true`, fixed residual correction, staged same-layer stat refresh, rescalers enabled, adaptive mixing disabled
- completed sweep points:
  - `8` chunks
  - `16` chunks
  - `32` chunks

Huffman shortest/longest symbol lengths are now part of the reporting schema. These three historical runs predate integer-code serialization for exact backfill, so their code-length range is shown as `n/a`.

## Completed Points

| Run | Config | Calib Chunks | Effective Bits | Entropy Bits | Huffman Bits | Huff Min | Huff Max | Side Bits | Quantized PPL | Delta vs Paper | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8`-chunk anchor | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `8` | `2.9984` | `2.9865` | `3.0368` | `n/a` | `n/a` | `0.0119` | `15.7029` | `+5.1329` | `18525.23s` |
| `16`-chunk run | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml` | `16` | `2.9984` | `2.9865` | `3.0371` | `n/a` | `n/a` | `0.0119` | `12.4574` | `+1.8874` | `22502.96s` |
| `32`-chunk run | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib32.yaml` | `32` | `2.9984` | `2.9865` | `3.0373` | `n/a` | `n/a` | `0.0119` | `11.7806` | `+1.2106` | `23211.72s` |

## Finished `32`-Chunk Run

- run: `llama32_1b_full_3p0bit_reftrue_rescaler_calib32`
- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib32.yaml`
- report bundle:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.md`
- artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_calib32/`
- log:
  - `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_calib32_20260315_205647.log`

Core metrics:

- baseline WikiText-2 PPL: `9.7041`
- quantized WikiText-2 PPL: `11.7806`
- effective bits: `2.9984`
- entropy bits: `2.9865`
- Huffman bits: `3.0373`
- Huffman shortest symbol length: `n/a`
- Huffman longest symbol length: `n/a`
- side-information overhead: `0.0119`
- total runtime: `23211.72s`
- quantization runtime: `23119.82s`
- peak GPU memory: `18.65 GiB`
- quantization anomalies: none

## Distortion Change Across Calibration Sizes

| Kind | `8` chunks | `16` chunks | `32` chunks |
| --- | ---: | ---: | ---: |
| `o_proj` mean rel MSE | `0.3170` | `0.2382` | `0.2025` |
| `down_proj` mean rel MSE | `0.3023` | `0.1709` | `0.1292` |
| `q_proj` mean rel MSE | `0.0696` | `0.0603` | `0.0770` |
| `k_proj` mean rel MSE | `0.0983` | `0.0854` | `0.1051` |
| `v_proj` mean rel MSE | `0.0876` | `0.0756` | `0.0908` |

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
- `32` chunks:
  - `model.layers.13.self_attn.o_proj` `0.2863`
  - `model.layers.0.self_attn.o_proj` `0.2650`
  - `model.layers.11.self_attn.o_proj` `0.2556`
  - `model.layers.4.self_attn.o_proj` `0.2540`
  - `model.layers.12.self_attn.o_proj` `0.2503`

## Interpretation

1. `32` chunks materially improves over `16` chunks.
   - PPL: `12.4574 -> 11.7806`
   - paper gap: `+1.8874 -> +1.2106`
   - absolute gain vs `16` chunks: `0.6768`

2. Calibration is still the strongest mainline quality lever observed so far.
   - `8 -> 16` improved by `3.2456` PPL
   - `16 -> 32` improved by another `0.6768` PPL

3. The dominant remaining distortion is still concentrated in `o_proj`, with `down_proj` next.
   - all top-5 worst layers at `32` chunks are still `o_proj`
   - `down_proj` remains the second-largest mean-error family even after improving to `0.1292`

4. The `32`-chunk run is now the best completed `Llama-3.2-1B` point in the repo.
   - effective bits: `2.9984`
   - quantized WikiText-2 PPL: `11.7806`

## Config / Runtime Notes

Before launch, the stale `32`-chunk config copy was corrected to match the validated mainline path:

- `use_adaptive_mixing: false`
- `epsilon_qr: 0.0`
- `epsilon_aw: 0.0`

This was a correctness alignment, not a new algorithmic change.

No new runtime-only optimization was added in this round beyond the existing repo-local WikiText-2 token-block cache.

## Conclusion

The completed `32`-chunk point further closes the paper gap while preserving the same stable algorithmic path:

- `8` chunks: `15.7029`
- `16` chunks: `12.4574`
- `32` chunks: `11.7806`

Calibration still appears to be the main remaining limiter on the best validated path, although the marginal gains are now smaller than the first `8 -> 16` jump.
