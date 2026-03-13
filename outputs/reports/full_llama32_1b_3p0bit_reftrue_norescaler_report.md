# Full Llama-3.2-1B 3.0-Bit Run

## Status

This is the first completed full-model `Llama-3.2-1B` WaterSIC run in the repo with:

- `reference_stats: true`
- residual correction enabled with the fixed formula
- staged same-layer stat refresh enabled
- diagonal rescalers disabled
- real WikiText-2 evaluation at context length `2048`

It is the first paper-comparable full-model point produced by the current codebase.

## Run Inputs

- Model config: `configs/models/llama32_1b.yaml`
- Quant config: `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml`
- Eval config: `configs/eval/wikitext2.yaml`
- Log: `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_norescaler_20260313_143248.log`
- Auto-generated report bundle:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_norescaler.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_norescaler.md`
- Quantized artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_norescaler/`

## Core Result

| Metric | Value |
| --- | ---: |
| Target global bitwidth | `3.0000` |
| Achieved effective global bitwidth | `2.9984` |
| Raw average bitwidth | `8.4774` |
| Entropy average bitwidth | `2.9865` |
| Huffman average bitwidth | `3.0368` |
| Side-information overhead | `0.0119` |
| Baseline WikiText-2 PPL | `9.7041` |
| Quantized WikiText-2 PPL | `16.8684` |
| Runtime | `19408.47s` (`5.39h`) |
| Peak GPU memory | `18.65 GiB` |

Additional run facts:

- `reference_stats` requested: `true`
- reference stats effective count: `109 / 112` quantized matrices
- rescalers enabled: `false`
- quantization anomalies: none

## Paper Comparison

Reference numbers from the local WaterSIC paper for `Llama-3.2-1B`:

- paper BF16 baseline PPL: `9.76`
- paper WaterSIC `3.00`-bit PPL: `10.57`

Comparison:

| Quantity | Paper | Ours | Absolute Diff |
| --- | ---: | ---: | ---: |
| Baseline PPL | `9.76` | `9.7041` | `-0.0559` |
| Quantized PPL at ~`3.0` bits | `10.57` | `16.8684` | `+6.2984` |
| Effective bits | `3.00` | `2.9984` | `-0.0016` |

Interpretation:

- The bitrate target is matched closely enough to count as a real paper-comparable point.
- The baseline is aligned with the paper.
- The quantized run is numerically sane but still far from the paper’s quality level.
- This is therefore a successful end-to-end run, but not yet a successful reproduction of the paper’s final accuracy.

## What Stayed Sane

- All `16` transformer layers completed.
- No NaN/Inf was reported.
- No Cholesky failure was reported.
- No residual-correction blow-up reappeared.
- The worst full-run distortion is concentrated in specific projection families rather than showing a global collapse.

Most distorted layers by relative weight MSE:

- `model.layers.11.self_attn.o_proj`: `0.4860`
- `model.layers.13.self_attn.o_proj`: `0.4685`
- `model.layers.12.self_attn.o_proj`: `0.4215`
- `model.layers.0.self_attn.o_proj`: `0.4164`
- `model.layers.4.self_attn.o_proj`: `0.4098`

Per-kind mean relative weight MSE:

- `o_proj`: `0.3215`
- `down_proj`: `0.3034`
- `k_proj`: `0.0979`
- `v_proj`: `0.0876`
- `q_proj`: `0.0696`
- `gate_proj`: `0.0502`
- `up_proj`: `0.0500`

## Diagnosis

The remaining gap to the paper is no longer explained by the earlier correctness bugs. The full run is stable, so the current problem is quality, not basic numerical failure.

Most likely causes of the gap:

1. Diagonal rescalers are still disabled in the best completed run.
2. The general sequential pipeline still uses fixed `epsilon_qr` / `epsilon_aw` values from config instead of per-attention-block search.
3. The full run used only `8` calibration chunks as a runtime shortcut.
4. The residual-path projections, especially `o_proj` and `down_proj`, still dominate distortion even though they no longer explode.

## Honest Bottom Line

- `reference_stats: true` was used.
- Residual correction was enabled with the fixed formula.
- Rescalers were explicitly disabled.
- The run is the first real full-model benchmark point in this repo.
- It is paper-comparable in setup and bitrate.
- It is not yet paper-matching in quality.

## Next Step

Use this run as the new stable baseline and recover the remaining quality gap in this order:

1. validate diagonal rescalers on the now-stable full-model path
2. move adaptive mixing search into the general sequential pipeline
3. increase calibration size
4. rerun the full-model `Llama-3.2-1B` benchmark and compare against the paper again
