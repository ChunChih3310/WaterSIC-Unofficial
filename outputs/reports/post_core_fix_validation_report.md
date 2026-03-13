# Post-Core-Fix Validation Report

## Scope

This round did not attempt Qwen3-8B and did not attempt a full-model paper-comparison run.
The goal was narrower:

1. verify that the fixed shared WaterSIC math stays sane beyond the tiny layer-0 debug slice,
2. validate `reference_stats: true` at larger scope,
3. determine whether the next blocker is in sequential interaction rather than the shared core math.

## Fixed Before This Round

Already fixed before the runs below:

- ZSIC recursive update now subtracts `alpha_i z_i L_i,:` instead of `c z_i L_i,:`.
- transformed-target construction now uses `target_cross @ (L^T)^(-1)`.
- paired-model calibration stats now run under `torch.no_grad()`.

## Run Summary

### A. Larger Layer-0 Validation

- Config: `configs/debug/llama32_1b_layer0_attention_refsafe_large.yaml`
- Model: `meta-llama/Llama-3.2-1B`
- Calibration: `8` WikiText-2 train chunks, sequence length `2048`
- Probe eval: `8` WikiText-2 test chunks, sequence length `2048`
- `reference_stats`: active through the paired-model path
- diagonal rescalers: disabled

Results:

- baseline small-eval PPL: `8.9880`
- quantized small-eval PPL: `9.0138`
- entropy bitwidth: `2.8087`
- Huffman bitwidth: `2.8561`
- optimized `epsilon_qr`: `0.0031056200151418556`
- optimized `epsilon_aw`: `0.7983738762488434`

Interpretation:

- the fixed path stayed numerically sane at larger layer-0 scope,
- `reference_stats` were actually effective for `o_proj`,
- no NaN/Inf, failed Cholesky, or recursive-update failures were observed.

### B. Multi-Layer Smoke With Staged Same-Layer Refresh

Code change added before the runs below:

- the sequential quantizer now refreshes stats within each layer in the order
  - `q_proj`, `k_proj`, `v_proj`
  - `o_proj`
  - `gate_proj`, `up_proj`
  - `down_proj`

Run:

- Config: `configs/quant/watersic_llama32_1b_multilayer_smoke_ref_stagefix.yaml`
- Eval config: `configs/eval/wikitext2_smoke8.yaml`
- Quantized prefix: first `11` modules
  - full layer `0`
  - layer `1` attention block
- `reference_stats`: `true`
- diagonal rescalers: disabled
- residual correction: enabled

Results:

- achieved effective bitwidth: `2.9912`
- entropy bitwidth: `2.9779`
- Huffman bitwidth: `3.0340`
- small-eval PPL: `700443.0656`
- quantization runtime: `978.59 s`
- total runtime: `1031.27 s`
- peak memory: `18.65 GiB`

First unstable point:

- module: `model.layers.1.self_attn.o_proj`
- reference stats effective: `true`
- relative weight MSE: `4.0915395089e9`
- `target_y_max_abs`: `85.9470`
- `alpha` range: `[10.6854, 435.0766]`
- `gamma` range: `[0.8600, 1.0000]`
- compensation applied: `true`

Interpretation:

- `layer1 q/k/v` stayed sane.
- the first second-order failure is not shared core math.
- the failure is concentrated in the sequential `o_proj` path.

### C. Controlled Residual-Correction Ablation

Run:

- Config: `configs/quant/watersic_llama32_1b_multilayer_smoke_ref_stagefix_noresid.yaml`
- Same setup as run B, except `use_residual_correction: false`

Results:

- achieved effective bitwidth: `2.9917`
- entropy bitwidth: `2.9784`
- Huffman bitwidth: `3.0341`
- small-eval PPL: `9.6219`
- quantization runtime: `909.16 s`
- total runtime: `961.09 s`
- peak memory: `18.65 GiB`

Critical comparison at `model.layers.1.self_attn.o_proj`:

| Setting | Rel Weight MSE | target_y_max_abs | Compensation |
| --- | ---: | ---: | --- |
| staged refresh + residual correction | `4.0915395089e9` | `85.9470` | `true` |
| staged refresh + no residual correction | `1.5426e-1` | `0.1904` | `false` |

Interpretation:

- the staged same-layer refresh was necessary but not sufficient,
- the current residual-compensation path is the remaining blocker,
- with residual correction disabled, the multi-layer smoke is sane again.

## What Stayed Sane

- larger layer-0 `reference_stats` path,
- layer-1 `q_proj`, `k_proj`, `v_proj` with `reference_stats: true`,
- the staged same-layer refresh logic itself,
- Cholesky, `alpha`, `gamma`, and recursive-update diagnostics outside the residual-correction failure case.

## What Failed

- the first true remaining correctness bug is on the residual-correction path for sequential `o_proj`,
- specifically at `model.layers.1.self_attn.o_proj`,
- after the model has already quantized layer `0` and layer `1` QKV.

## Paper Comparison

Paper reference from Table 1 in `paper/2603.04956v1.pdf`:

- model: `Llama-3.2-1B`
- evaluation: WikiText-2, context length `2048`
- BF16 baseline PPL: `9.76`
- WaterSIC at `3.00` bits: `10.57`

Closest runs from this repo in this round:

| Run | Our Rate | Our PPL | Paper PPL | Abs. Diff | Comparable? | Diagnosis |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| staged refresh + residual correction | `2.9912` | `700443.0656` | `10.57` | `+700432.4956` | No | catastrophic `layer1 o_proj` failure from residual correction |
| staged refresh + no residual correction | `2.9917` | `9.6219` | `10.57` | `-0.9481` | No | only first `11` modules were quantized and eval used `8` test chunks, so this is a smoke sanity result, not a reproduction |

Important note:

- the successful no-residual smoke must not be claimed as a paper-faithful reproduction,
- but it is strong evidence that the post-core-fix path is broadly sane and that the next blocker is narrow and specific.

## Current Bottom Line

1. The shared WaterSIC math now scales beyond the tiny layer-0 debug slice.
2. `reference_stats: true` is active and helpful beyond layer 0.
3. The next true blocker is the residual-compensation implementation for sequential `o_proj` / `down_proj`.
4. A full-model ~`3.0`-bit `Llama-3.2-1B` run should wait until that residual path is fixed.
