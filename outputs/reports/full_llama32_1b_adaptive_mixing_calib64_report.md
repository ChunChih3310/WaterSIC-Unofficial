# Full-Model Llama-3.2-1B Repaired Adaptive Mixing 64-Chunk Report

Date: 2026-03-17

## Run

- run: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64`
- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired_calib64.yaml`
- machine-readable report:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64.json`
- markdown run report:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64.md`
- artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64/`

## Result

- achieved effective bits: `2.9984`
- entropy bits: `2.9865`
- Huffman bits: `3.0376`
- Huffman shortest symbol length: `1`
- Huffman longest symbol length: `25`
- side-information overhead: `0.0119`
- baseline WikiText-2 PPL: `9.7041`
- quantized WikiText-2 PPL: `11.1874`
- quantization runtime: `39193.57s`
- total runtime: `39290.59s`
- peak GPU memory: `32.69 GiB`
- quantization anomalies: none

## Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

Selected completed comparison points:

| Run | Effective Bits | Quantized PPL | Delta vs Paper `10.57` | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `rescaler-only 32` | `2.9984` | `11.7806` | `+1.2106` | `23211.72s` |
| `repaired mixing 8` | `2.9984` | `16.2796` | `+5.7096` | `30248.17s` |
| `repaired mixing 64` | `2.9984` | `11.1874` | `+0.6174` | `39290.59s` |

Direct comparisons:

- `repaired mixing 64` vs `repaired mixing 8`:
  - PPL improvement: `-5.0922`
  - paper-gap reduction: `-5.0922`
- `repaired mixing 64` vs `rescaler-only 32`:
  - PPL improvement: `-0.5932`
  - paper-gap reduction: `-0.5932`

## Error Profile

Mean relative weight MSE by kind:

- `o_proj`: `0.1652`
- `down_proj`: `0.1095`
- `q_proj`: `0.0458`
- `k_proj`: `0.0688`
- `v_proj`: `0.0610`

Worst layers by relative weight MSE:

1. `model.layers.1.mlp.down_proj`: `0.2587`
2. `model.layers.13.self_attn.o_proj`: `0.2491`
3. `model.layers.11.self_attn.o_proj`: `0.2460`
4. `model.layers.0.self_attn.o_proj`: `0.2349`
5. `model.layers.10.mlp.down_proj`: `0.2300`

## Interpretation

This is the first completed run in the repo where repaired adaptive mixing becomes the best full-model `Llama-3.2-1B` path instead of lagging behind the rescaler-only reference.

What changed relative to the earlier adaptive-mixing runs:

- larger calibration made the repaired path materially more effective
- the repaired path now beats the best completed rescaler-only point by `0.5932` PPL
- the remaining paper gap shrank to `+0.6174`

What still remains:

- the paper is not yet matched
- the dominant remaining distortion is still concentrated in residual-path projections, now split across both `o_proj` and `down_proj`
- Qwen3-8B remains intentionally deferred until `Llama-3.2-1B` is pushed closer to the paper
