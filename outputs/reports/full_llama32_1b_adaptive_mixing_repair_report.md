# Full-Model Llama-3.2-1B Adaptive Mixing Repair Report

Date: 2026-03-15

## Scope

This round focused only on repairing adaptive mixing for full-model `Llama-3.2-1B` at about `3.0` bits.

Reference points entering the round:

- best completed full-model point: `llama32_1b_full_3p0bit_reftrue_rescaler`
  - effective bits: `2.9984`
  - WikiText-2 PPL: `15.7029`
- old adaptive-mixing full-model point: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing`
  - effective bits: `2.9984`
  - WikiText-2 PPL: `16.6096`

## Paper Audit Outcome

The paper audit identified the main mismatch:

- the old implementation re-ran binary search over `c` inside every `epsilon_qr` and `epsilon_aw` candidate evaluation
- the paper-backed sequence is:
  - initial rate calibration at `(epsilon_qr, epsilon_aw) = (0, 0)`
  - coordinate search over `epsilon_qr`
  - coordinate search over `epsilon_aw`
  - final recalibration once the pair is selected

Saved audit notes:

- `outputs/reports/adaptive_mixing_paper_audit.md`
- `outputs/reports/adaptive_mixing_mismatch_diagnosis.md`
- `outputs/reports/adaptive_mixing_runtime_optimization.md`

## Accepted Repair

The adopted repair is:

1. run the normal initial QKV calibration at `(epsilon_qr, epsilon_aw) = (0, 0)`
2. record the step-1 calibrated per-matrix Q/K/V scales
3. reuse those scales during the `epsilon_qr` and `epsilon_aw` searches
4. keep the final recalibration pass after selecting `(epsilon_qr*, epsilon_aw*)`

This removes repeated binary searches from the coordinate search while preserving the final stable quantization path.

## Prefix Validation

Validation run:

- config: `configs/quant/watersic_llama32_1b_prefix2_reftrue_rescaler_mixing_repaircheck.yaml`
- reports:
  - `archive/reports/milestones/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.json`
  - `archive/reports/milestones/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.md`

Result:

- achieved effective bits: `2.9862`
- entropy bits: `2.9743`
- Huffman bits: `3.0262`
- baseline smoke-8 PPL: `8.9880`
- quantized smoke-8 PPL: `9.5600`
- total runtime: `3441.95s`
- peak GPU memory: `18.65 GiB`

Measured adaptive-mixing runtime reduction on the repaired path:

- layer 0 search time:
  - old full-model path: about `4002s`
  - repaired path: about `394.5s`
  - improvement: about `10.1x`
- layer 1 search time:
  - old full-model path: about `2526s`
  - repaired path: about `444.8s`
  - improvement: about `5.7x`

## Finished Full-Model Run

Run:

- run name: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired`
- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml`
- log: `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_20260315_015946.log`
- reports:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired.md`
- artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired/`

The run finished successfully.

- achieved effective bits: `2.9984`
- entropy bits: `2.9865`
- Huffman bits: `3.0368`
- side-information overhead: `0.0119`
- baseline WikiText-2 PPL: `9.7041`
- quantized WikiText-2 PPL: `16.2796`
- quantization runtime: `30142.43s`
- total runtime: `30248.17s`
- peak GPU memory: `18.65 GiB`
- quantization anomalies: none

## Comparison Against Prior Full-Model Points

| Run | Effective Bits | Quantized PPL | Delta vs Paper `10.57` | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `rescaler-only` | `2.9984` | `15.7029` | `+5.1329` | `18525.23s` |
| `old rescaler+mixing` | `2.9984` | `16.6096` | `+6.0396` | `74349.91s` |
| `repaired rescaler+mixing` | `2.9984` | `16.2796` | `+5.7096` | `30248.17s` |

Direct comparisons:

- repaired vs old adaptive mixing:
  - PPL improvement: `-0.3300`
  - runtime reduction: `-44101.74s`
  - speedup: about `2.46x`
- repaired vs rescaler-only:
  - PPL regression: `+0.5767`
  - runtime increase: `+11722.94s`

Per-kind mean relative weight MSE:

| Kind | Rescaler-Only | Old Mixing | Repaired Mixing |
| --- | ---: | ---: | ---: |
| `o_proj` | `0.3170` | `0.3175` | `0.3176` |
| `down_proj` | `0.3023` | `0.3024` | `0.3051` |
| `q_proj` | `0.0696` | `0.0611` | `0.0604` |
| `k_proj` | `0.0983` | `0.0898` | `0.0886` |
| `v_proj` | `0.0876` | `0.0806` | `0.0797` |

Worst repaired-run layers by relative weight MSE:

1. `model.layers.11.self_attn.o_proj`: `0.4796`
2. `model.layers.13.self_attn.o_proj`: `0.4597`
3. `model.layers.12.self_attn.o_proj`: `0.4185`
4. `model.layers.4.self_attn.o_proj`: `0.3997`
5. `model.layers.0.self_attn.o_proj`: `0.3934`

## Interpretation

The repaired adaptive-mixing path helped relative to the old adaptive-mixing run, but it did not fix the regression against the best completed reference.

What improved:

- QKV reconstruction improved again relative to both the rescaler-only and old-mixing runs.
- runtime dropped substantially versus the old adaptive-mixing path.
- the run remained numerically stable end to end.

What did not improve enough:

- the repaired path still did not beat the rescaler-only run on final WikiText-2 PPL
- the dominant late-layer error modes remain `self_attn.o_proj`, with `down_proj` still the next largest residual-path kind

Bottom line:

- repaired adaptive mixing is beneficial relative to the old adaptive-mixing implementation
- repaired adaptive mixing is still not beneficial enough to replace the rescaler-only point as the repo’s best completed `Llama-3.2-1B` result
- the current best completed point remains:
  - `llama32_1b_full_3p0bit_reftrue_rescaler`
  - `15.7029` PPL at `2.9984` effective bits

## Current Recommendation

The single best next step is to increase calibration beyond `8` chunks and rerun the current best validated path, which is still the rescaler-only configuration.
