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

## Comparison

- previous best completed point:
  - `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64`
  - PPL: `11.1874`
- paper reference at `3.00` bits:
  - PPL: `10.57`

Absolute differences:

- vs repaired adaptive-mixing `64` chunks: `-0.5843`
- vs paper: `+0.0331`

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

This paper-scale run is the current best completed `Llama-3.2-1B` result in the repo. It nearly matches the paper’s reported `10.57` PPL at `3.00` bits, with only a `+0.0331` absolute gap. At this point the remaining discrepancy is small enough that ordinary run variance, model/tokenizer revision differences, or small implementation details are more plausible explanations than any large missing algorithmic component.
