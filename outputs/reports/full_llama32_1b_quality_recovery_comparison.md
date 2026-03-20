# Full-Model Llama-3.2-1B Quality Recovery Comparison

Paper reference at `3.00` bits:

- baseline BF16 PPL: `9.76`
- WaterSIC PPL: `10.57`

Validation rerun for the current best paper-scale artifact:

- artifact: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- validation-split PPL: `10.9310`
- gap vs paper: `+0.3610`
- note: the `H` row below is the original test-split run report for the same artifact (`10.6031`)

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
| `G. repaired rescaler+mixing 64` | `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired_calib64.yaml` | `2.9984` | `2.9865` | `3.0376` | `1` | `25` | `0.0119` | `11.1874` | `+0.6174` | `39290.59s` |
| `H. repaired rescaler+mixing paperscale` | `configs/paper_comparable/quant/watersic_llama32_1b_paperscale.yaml` | `2.9984` | `2.9864` | `3.0379` | `1` | `25` | `0.0119` | `10.6031` | `+0.0331` | `188979.22s` |

## Top Worst Layers

| Run | Top Worst Layers |
| --- | --- |
| `A. no-rescaler 8` | `l11 o_proj 0.4860`; `l13 o_proj 0.4685`; `l12 o_proj 0.4215`; `l0 o_proj 0.4164`; `l4 o_proj 0.4098` |
| `B. rescaler-only 8` | `l11 o_proj 0.4763`; `l13 o_proj 0.4596`; `l12 o_proj 0.4178`; `l4 o_proj 0.4032`; `l0 o_proj 0.3936` |
| `C. old rescaler+mixing 8` | `l11 o_proj 0.4777`; `l13 o_proj 0.4647`; `l12 o_proj 0.4202`; `l4 o_proj 0.4021`; `l0 o_proj 0.3926` |
| `D. repaired rescaler+mixing 8` | `l11 o_proj 0.4796`; `l13 o_proj 0.4597`; `l12 o_proj 0.4185`; `l4 o_proj 0.3997`; `l0 o_proj 0.3934` |
| `E. rescaler-only 16` | `l13 o_proj 0.3427`; `l11 o_proj 0.3223`; `l0 o_proj 0.3108`; `l12 o_proj 0.3035`; `l4 o_proj 0.2935` |
| `F. rescaler-only 32` | `l13 o_proj 0.2863`; `l0 o_proj 0.2650`; `l11 o_proj 0.2556`; `l4 o_proj 0.2540`; `l12 o_proj 0.2503` |
| `G. repaired rescaler+mixing 64` | `l1 down_proj 0.2587`; `l13 o_proj 0.2491`; `l11 o_proj 0.2460`; `l0 o_proj 0.2349`; `l10 down_proj 0.2300` |
| `H. repaired rescaler+mixing paperscale` | `l0 o_proj 0.1924`; `l2 o_proj 0.1473`; `l1 down_proj 0.1415`; `l4 o_proj 0.1342`; `l5 o_proj 0.1224` |

## Mean Relative Weight MSE By Kind

| Run | `o_proj` | `down_proj` | `q_proj` | `k_proj` | `v_proj` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A. no-rescaler 8` | `0.3215` | `0.3034` | `0.0696` | `0.0979` | `0.0876` |
| `B. rescaler-only 8` | `0.3170` | `0.3023` | `0.0696` | `0.0983` | `0.0876` |
| `C. old rescaler+mixing 8` | `0.3175` | `0.3024` | `0.0611` | `0.0898` | `0.0806` |
| `D. repaired rescaler+mixing 8` | `0.3176` | `0.3051` | `0.0604` | `0.0886` | `0.0797` |
| `E. rescaler-only 16` | `0.2382` | `0.1709` | `0.0603` | `0.0854` | `0.0756` |
| `F. rescaler-only 32` | `0.2025` | `0.1292` | `0.0770` | `0.1051` | `0.0908` |
| `G. repaired rescaler+mixing 64` | `0.1652` | `0.1095` | `0.0458` | `0.0688` | `0.0610` |
| `H. repaired rescaler+mixing paperscale` | `0.1099` | `0.0706` | `0.0457` | `0.0674` | `0.0590` |

## Interpretation

1. `H` is now the best completed `Llama-3.2-1B` quantized artifact in the repo.
   - It improves PPL by `0.5843` over `G`.
   - It improves PPL by `1.1775` over `F`.
   - It improves PPL by `5.0998` over `B`.
   - It reduces the paper gap from `+0.6174` to `+0.0331`.
   - On a validation rerun of the same artifact, the paper-comparable gap is `+0.3610`.

2. Calibration continues to help on the validated rescaler-only path.
   - `8 -> 16`: `15.7029 -> 12.4574`
   - `16 -> 32`: `12.4574 -> 11.7806`

3. Larger calibration makes repaired adaptive mixing finally competitive and then nearly paper-matching.
   - `D -> G`: `16.2796 -> 11.1874`
   - `G -> H`: `11.1874 -> 10.6031`
   - the repaired adaptive-mixing path now beats the best completed rescaler-only point `F`
   - the best completed path is no longer rescaler-only

4. The dominant remaining error is still concentrated in residual-path projections, especially `o_proj`, with `down_proj` still next.
   - `H` still has `o_proj` and `down_proj` as the largest-error kinds
   - the worst single layer is now `model.layers.0.self_attn.o_proj`
   - top-5 worst layers at `H` are mostly `o_proj`, with `model.layers.1.mlp.down_proj` still present

5. `H` improves every reported module family versus `F`.
   - `o_proj`: `0.2025 -> 0.1099`
   - `down_proj`: `0.1292 -> 0.0706`
   - `q_proj`: `0.0770 -> 0.0457`
   - `k_proj`: `0.1051 -> 0.0674`
   - `v_proj`: `0.0908 -> 0.0590`

6. Adaptive mixing is no longer merely a secondary path.
   - at `8` chunks it remained worse than the best rescaler-only runs
   - at `64` chunks on the repaired path it became the best completed full-model result
   - at paper-scale calibration it nearly matches the paper result
   - this strongly suggests adaptive mixing was heavily calibration-limited in prior completed runs

## Current Best Point

- Run: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- Effective bits: `2.9984`
- Quantized PPL: `10.6031`
- Status: best completed `Llama-3.2-1B` point in this repo as of this update

## Next Step

The next step is to explain why the validation rerun (`10.9310`) is noticeably higher than the original test-split benchmark (`10.6031`) before broadening to Qwen3-8B.
