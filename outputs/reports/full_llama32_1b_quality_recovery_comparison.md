# Full-Model Llama-3.2-1B Quality Recovery Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

Huffman shortest/longest symbol lengths are now part of the run-report schema. These historical completed runs predate integer-code serialization for exact backfill, so their code-length ranges are shown as `n/a`.

## Overview

| Run | Config | Effective Bits | Entropy Bits | Huffman Bits | Huff Min | Huff Max | Side-Info Bits | Quantized PPL | Delta vs Paper | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler 8` | `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `n/a` | `n/a` | `0.0119` | `16.8684` | `+6.2984` | `19408.47s` |
| `B. rescaler-only 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `n/a` | `n/a` | `0.0119` | `15.7029` | `+5.1329` | `18525.23s` |
| `C. old rescaler+mixing 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml` | `2.9984` | `2.9865` | `3.0368` | `n/a` | `n/a` | `0.0119` | `16.6096` | `+6.0396` | `74349.91s` |
| `D. repaired rescaler+mixing 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml` | `2.9984` | `2.9865` | `3.0368` | `n/a` | `n/a` | `0.0119` | `16.2796` | `+5.7096` | `30248.17s` |
| `E. rescaler-only 16` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml` | `2.9984` | `2.9865` | `3.0371` | `n/a` | `n/a` | `0.0119` | `12.4574` | `+1.8874` | `22502.96s` |
| `F. rescaler-only 32` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib32.yaml` | `2.9984` | `2.9865` | `3.0373` | `n/a` | `n/a` | `0.0119` | `11.7806` | `+1.2106` | `23211.72s` |

## Top Worst Layers

| Run | Top Worst Layers |
| --- | --- |
| `A. no-rescaler 8` | `l11 o_proj 0.4860`; `l13 o_proj 0.4685`; `l12 o_proj 0.4215`; `l0 o_proj 0.4164`; `l4 o_proj 0.4098` |
| `B. rescaler-only 8` | `l11 o_proj 0.4763`; `l13 o_proj 0.4596`; `l12 o_proj 0.4178`; `l4 o_proj 0.4032`; `l0 o_proj 0.3936` |
| `C. old rescaler+mixing 8` | `l11 o_proj 0.4777`; `l13 o_proj 0.4647`; `l12 o_proj 0.4202`; `l4 o_proj 0.4021`; `l0 o_proj 0.3926` |
| `D. repaired rescaler+mixing 8` | `l11 o_proj 0.4796`; `l13 o_proj 0.4597`; `l12 o_proj 0.4185`; `l4 o_proj 0.3997`; `l0 o_proj 0.3934` |
| `E. rescaler-only 16` | `l13 o_proj 0.3427`; `l11 o_proj 0.3223`; `l0 o_proj 0.3108`; `l12 o_proj 0.3035`; `l4 o_proj 0.2935` |
| `F. rescaler-only 32` | `l13 o_proj 0.2863`; `l0 o_proj 0.2650`; `l11 o_proj 0.2556`; `l4 o_proj 0.2540`; `l12 o_proj 0.2503` |

## Mean Relative Weight MSE By Kind

| Run | `o_proj` | `down_proj` | `q_proj` | `k_proj` | `v_proj` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler 8` | `0.3215` | `0.3034` | `0.0696` | `0.0979` | `0.0876` |
| `B. rescaler-only 8` | `0.3170` | `0.3023` | `0.0696` | `0.0983` | `0.0876` |
| `C. old rescaler+mixing 8` | `0.3175` | `0.3024` | `0.0611` | `0.0898` | `0.0806` |
| `D. repaired rescaler+mixing 8` | `0.3176` | `0.3051` | `0.0604` | `0.0886` | `0.0797` |
| `E. rescaler-only 16` | `0.2382` | `0.1709` | `0.0603` | `0.0854` | `0.0756` |
| `F. rescaler-only 32` | `0.2025` | `0.1292` | `0.0770` | `0.1051` | `0.0908` |

## Interpretation

1. `F` is now the best completed `Llama-3.2-1B` point in the repo.
   - It improves PPL by `0.6768` over `E`.
   - It improves PPL by `3.9223` over `B`.
   - It reduces the paper gap from `+1.8874` to `+1.2106`.

2. Calibration continues to help on the validated rescaler-only path.
   - `8 -> 16`: `15.7029 -> 12.4574`
   - `16 -> 32`: `12.4574 -> 11.7806`

3. The dominant remaining error is still concentrated in residual-path projections.
   - all top-5 worst layers at `F` are still `o_proj`
   - `down_proj` remains the second-largest mean-error family

4. The `32`-chunk run improves the error families that matter most for end-task quality, even though the Q/K/V means do not continue to monotonically improve.
   - `E -> F`:
     - `o_proj`: `0.2382 -> 0.2025`
     - `down_proj`: `0.1709 -> 0.1292`
     - `q_proj`: `0.0603 -> 0.0770`
     - `k_proj`: `0.0854 -> 0.1051`
     - `v_proj`: `0.0756 -> 0.0908`

5. Adaptive mixing remains off the mainline quality-recovery path.
   - both completed adaptive-mixing full-model runs remain worse than `E` and `F`
   - the best paper-gap reduction has come from calibration on the stable rescaler-only path

## Current Best Point

- Run: `llama32_1b_full_3p0bit_reftrue_rescaler_calib32`
- Effective bits: `2.9984`
- Quantized PPL: `11.7806`
- Status: best completed `Llama-3.2-1B` point in this repo as of this update

## Next Step

Increase calibration further on the same validated rescaler-only path before returning to adaptive-mixing work or starting Qwen3-8B.
