# Full-Model Llama-3.2-1B Quality Recovery Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

## Overview

| Run | Config | Effective Bits | Entropy Bits | Huffman Bits | Side-Info Bits | Quantized PPL | Delta vs Paper | Top Worst Layers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `A. no-rescaler` | `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.8684` | `+6.2984` | `l11 o_proj 0.4860`; `l13 o_proj 0.4685`; `l12 o_proj 0.4215`; `l0 o_proj 0.4164`; `l4 o_proj 0.4098` |
| `B. rescaler-only` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `15.7029` | `+5.1329` | `l11 o_proj 0.4763`; `l13 o_proj 0.4596`; `l12 o_proj 0.4178`; `l4 o_proj 0.4032`; `l0 o_proj 0.3936` |
| `C. old rescaler+mixing` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.6096` | `+6.0396` | `l11 o_proj 0.4777`; `l13 o_proj 0.4647`; `l12 o_proj 0.4202`; `l4 o_proj 0.4021`; `l0 o_proj 0.3926` |
| `D. repaired rescaler+mixing` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.2796` | `+5.7096` | `l11 o_proj 0.4796`; `l13 o_proj 0.4597`; `l12 o_proj 0.4185`; `l4 o_proj 0.3997`; `l0 o_proj 0.3934` |

## Mean Relative Weight MSE By Kind

| Run | `o_proj` | `down_proj` | `q_proj` | `k_proj` | `v_proj` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler` | `0.3215` | `0.3034` | `0.0696` | `0.0979` | `0.0876` |
| `B. rescaler-only` | `0.3170` | `0.3023` | `0.0696` | `0.0983` | `0.0876` |
| `C. old rescaler+mixing` | `0.3175` | `0.3024` | `0.0611` | `0.0898` | `0.0806` |
| `D. repaired rescaler+mixing` | `0.3176` | `0.3051` | `0.0604` | `0.0886` | `0.0797` |

## Interpretation

1. `B` remains the best completed point.
   - It improves PPL by `1.1654` over `A`.
   - It reduces the paper gap from `+6.2984` to `+5.1329`.

2. `C` validated the first adaptive-mixing path operationally but regressed final quality.
   - It is `0.9067` PPL worse than `B`.

3. `D` repairs part of the adaptive-mixing regression but still does not beat `B`.
   - It improves over `C` by `0.3300` PPL.
   - It is still `0.5767` PPL worse than `B`.
   - It remains `0.5887` PPL better than `A`.

4. The repaired adaptive-mixing path improves QKV reconstruction but still does not improve the dominant residual-path error modes enough.
   - `D` improves `q_proj`, `k_proj`, and `v_proj` mean relative MSE relative to both `B` and `C`.
   - `D` leaves `o_proj` flat and worsens `down_proj` relative to `B`.

5. Current best paper-comparable point remains `B`.
   - `15.7029` PPL at `2.9984` effective bits
   - still meaningfully behind the paper’s `10.57`

## Current Best Point

- Run: `llama32_1b_full_3p0bit_reftrue_rescaler`
- Effective bits: `2.9984`
- Quantized PPL: `15.7029`
- Status: best completed `Llama-3.2-1B` point in this repo as of this update
