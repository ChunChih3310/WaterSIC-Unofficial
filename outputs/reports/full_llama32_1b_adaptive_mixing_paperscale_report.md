# Full-Model Llama-3.2-1B Paper-Scale Repaired Adaptive-Mixing Report

## Run

- run: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- model config: `configs/paper_comparable/models/llama32_1b.yaml`
- quant config: `configs/paper_comparable/quant/watersic_llama32_1b_paperscale.yaml`
- eval config: `configs/eval/wikitext2.yaml`

## Result

- effective bits: `2.9984`
- entropy bits: `2.9864`
- Huffman bits: `3.0379`
- Huffman shortest/longest symbol lengths: `1` / `25`
- side-information overhead: `0.0119`
- baseline WikiText-2 PPL: `9.7041`
- quantized WikiText-2 PPL: `10.6031`
- runtime: `188979.22s` total, `188876.89s` quantization
- peak GPU memory: `32.69 GiB`
- quantization anomalies: none

## Validation Benchmark

- eval config: `configs/eval/wikitext2_validation.yaml`
- validation-split PPL: `10.9310`
- previous test-split PPL from the same artifact: `10.6031`
- paper reference at `3.00` bits: `10.57`
- validation gap vs paper: `+0.3610`

This removes the previous split-mismatch caveat for the final `Llama-3.2-1B` paper comparison. The remaining difference is now a direct validation-vs-validation numerical gap, not an evaluation-split mismatch.

## Comparison

- previous best completed point:
  - `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64`
  - PPL: `11.1874`
- paper reference at `3.00` bits:
  - PPL: `10.57`

Absolute differences:

- vs repaired adaptive-mixing `64` chunks: `-0.5843`
- vs paper on the original test-split run: `+0.0331`
- vs paper on the validation rerun: `+0.3610`

## Error Concentration

Top worst layers by relative weight MSE:

1. `model.layers.0.self_attn.o_proj`: `0.1924`
2. `model.layers.2.self_attn.o_proj`: `0.1473`
3. `model.layers.1.mlp.down_proj`: `0.1415`
4. `model.layers.4.self_attn.o_proj`: `0.1342`
5. `model.layers.5.self_attn.o_proj`: `0.1224`

Per-kind mean relative weight MSE:

- `o_proj`: `0.1099`
- `down_proj`: `0.0706`
- `q_proj`: `0.0457`
- `k_proj`: `0.0674`
- `v_proj`: `0.0590`

The remaining distortion is still concentrated in residual-path projections, especially `o_proj`, with `down_proj` still next.

## Interpretation

This paper-scale run remains the current best completed `Llama-3.2-1B` quantized artifact in the repo. The original test-split benchmark was `10.6031`, but the validation rerun on the same artifact is `10.9310`. That means the main split-mismatch caveat is now removed, but the paper-comparable validation number is not as close as the earlier test-split number. The remaining gap to the paper on the matched validation protocol is `+0.3610`, which is still modest but no longer small enough to dismiss as pure noise.
