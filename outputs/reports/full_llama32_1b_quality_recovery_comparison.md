# Full-Model Llama-3.2-1B Quality Recovery Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

## Overview

| Run | Config | Effective Bits | Entropy Bits | Huffman Bits | Side-Info Bits | Quantized PPL | Delta vs Paper | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler 8` | `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.8684` | `+6.2984` | `19408.47s` |
| `B. rescaler-only 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `15.7029` | `+5.1329` | `18525.23s` |
| `C. old rescaler+mixing 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.6096` | `+6.0396` | `74349.91s` |
| `D. repaired rescaler+mixing 8` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml` | `2.9984` | `2.9865` | `3.0368` | `0.0119` | `16.2796` | `+5.7096` | `30248.17s` |
| `E. rescaler-only 16` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml` | `2.9984` | `2.9865` | `3.0371` | `0.0119` | `12.4574` | `+1.8874` | `22502.96s` |

## Top Worst Layers

| Run | Top Worst Layers |
| --- | --- |
| `A. no-rescaler 8` | `l11 o_proj 0.4860`; `l13 o_proj 0.4685`; `l12 o_proj 0.4215`; `l0 o_proj 0.4164`; `l4 o_proj 0.4098` |
| `B. rescaler-only 8` | `l11 o_proj 0.4763`; `l13 o_proj 0.4596`; `l12 o_proj 0.4178`; `l4 o_proj 0.4032`; `l0 o_proj 0.3936` |
| `C. old rescaler+mixing 8` | `l11 o_proj 0.4777`; `l13 o_proj 0.4647`; `l12 o_proj 0.4202`; `l4 o_proj 0.4021`; `l0 o_proj 0.3926` |
| `D. repaired rescaler+mixing 8` | `l11 o_proj 0.4796`; `l13 o_proj 0.4597`; `l12 o_proj 0.4185`; `l4 o_proj 0.3997`; `l0 o_proj 0.3934` |
| `E. rescaler-only 16` | `l13 o_proj 0.3427`; `l11 o_proj 0.3223`; `l0 o_proj 0.3108`; `l12 o_proj 0.3035`; `l4 o_proj 0.2935` |

## Mean Relative Weight MSE By Kind

| Run | `o_proj` | `down_proj` | `q_proj` | `k_proj` | `v_proj` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler 8` | `0.3215` | `0.3034` | `0.0696` | `0.0979` | `0.0876` |
| `B. rescaler-only 8` | `0.3170` | `0.3023` | `0.0696` | `0.0983` | `0.0876` |
| `C. old rescaler+mixing 8` | `0.3175` | `0.3024` | `0.0611` | `0.0898` | `0.0806` |
| `D. repaired rescaler+mixing 8` | `0.3176` | `0.3051` | `0.0604` | `0.0886` | `0.0797` |
| `E. rescaler-only 16` | `0.2382` | `0.1709` | `0.0603` | `0.0854` | `0.0756` |

## Interpretation

1. `E` is now the best completed `Llama-3.2-1B` point in the repo.
   - It improves PPL by `3.2456` over `B`.
   - It reduces the paper gap from `+5.1329` to `+1.8874`.

2. Increasing calibration from `8` to `16` chunks produced a much larger gain than the adaptive-mixing variants produced.
   - `B -> E`: `15.7029 -> 12.4574`
   - `D -> E`: `16.2796 -> 12.4574`

3. The remaining distortion is still concentrated in `o_proj`, with `down_proj` next, but both are much lower in `E` than in every `8`-chunk run.
   - `o_proj`: `0.3170 -> 0.2382`
   - `down_proj`: `0.3023 -> 0.1709`

4. Adaptive mixing is still not the best full-model path.
   - both completed adaptive-mixing runs remain worse than the rescaler-only `16`-chunk run
   - repaired adaptive mixing remains useful as a stable paper-audited implementation, not yet as the best quality path

5. Calibration now looks like the dominant near-term limiter on the validated rescaler-only path.
   - the `16`-chunk result is close enough to the paper to justify the next controlled sweep point at `32` chunks

## Current Best Point

- Run: `llama32_1b_full_3p0bit_reftrue_rescaler_calib16`
- Effective bits: `2.9984`
- Quantized PPL: `12.4574`
- Status: best completed `Llama-3.2-1B` point in this repo as of this update

## Next Step

Run the `32`-chunk rescaler-only full-model configuration on the same validated path before returning to more adaptive-mixing work or starting Qwen3-8B.
