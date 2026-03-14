# Full-Model Llama-3.2-1B Quality Recovery Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

## Overview

| Run | Config | Effective Bits | Entropy Bits | Huffman Bits | Side-Info Bits | Quantized PPL | Delta vs Paper | Top Worst Layers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `A. no-rescaler` | `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.8684` | `+6.2984` | `l11 o_proj 0.4860`; `l13 o_proj 0.4685`; `l12 o_proj 0.4215`; `l0 o_proj 0.4164`; `l4 o_proj 0.4098` |
| `B. rescaler-only` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `15.7029` | `+5.1329` | `l11 o_proj 0.4763`; `l13 o_proj 0.4596`; `l12 o_proj 0.4178`; `l4 o_proj 0.4032`; `l0 o_proj 0.3936` |
| `C. rescaler + mixing search` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.6096` | `+6.0396` | `l11 o_proj 0.4777`; `l13 o_proj 0.4647`; `l12 o_proj 0.4202`; `l4 o_proj 0.4021`; `l0 o_proj 0.3926` |

## Mean Relative Weight MSE By Kind

| Run | `o_proj` | `down_proj` | `q_proj` | `k_proj` | `v_proj` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler` | `0.3215` | `0.3034` | `0.0696` | `0.0979` | `0.0876` |
| `B. rescaler-only` | `0.3170` | `0.3023` | `0.0696` | `0.0983` | `0.0876` |
| `C. rescaler + mixing search` | `0.3175` | `0.3024` | `0.0611` | `0.0898` | `0.0806` |

## Interpretation

1. `B` is the best completed result.
   - It improves PPL by `1.1654` over `A`.
   - It reduces the paper gap from `+6.2984` to `+5.1329`.

2. `C` validates the adaptive-mixing path operationally but does not improve the best point.
   - It is `0.9067` PPL worse than `B`.
   - It is still `0.2587` PPL better than `A`.

3. The adaptive-mixing run improves QKV reconstruction but does not improve the dominant residual-path error modes enough.
   - `C` improves `q_proj`, `k_proj`, and `v_proj` mean relative MSE relative to `B`.
   - `C` slightly worsens `o_proj` and `down_proj` mean relative MSE relative to `B`.

4. Current best paper-comparable point remains `B`.
   - `15.7029` PPL at `2.9984` effective bits
   - still meaningfully behind the paper’s `10.57`

## Current Best Point

- Run: `llama32_1b_full_3p0bit_reftrue_rescaler`
- Effective bits: `2.9984`
- Quantized PPL: `15.7029`
- Status: best completed `Llama-3.2-1B` point in this repo as of this update
