# `layer1 self_attn.o_proj` Residual Debug Report

## Conclusion

The residual-correction blocker was caused by the wrong residual formula, not by wrong timing or wrong sign.

- Correct paper-faithful form at the target cross term:
  - `W Sigma_X,Xhat + Sigma_Delta,Xhat`
- Old buggy form:
  - residual addend solved against `Sigma_Xhat^{-1}`
- Effect on the exact previous blocker (`model.layers.1.self_attn.o_proj`):
  - corrected path `target_y_max_abs = 0.190439`
  - legacy path `target_y_max_abs = 85.947011`

The corrected residual path is numerically sane on the narrow layer1 audit and on the same `11`-module smoke that previously failed catastrophically.

## What Was Already Fixed Before This Round

- ZSIC recursive update now uses `alpha_i` instead of the global `c`
- transformed target `Y` now uses the correct triangular solve
- paired calibration stats are collected under `torch.no_grad()`
- staged same-layer stat refresh is active in sequential quantization

## Narrow Audit Setup

- Model: `meta-llama/Llama-3.2-1B`
- Target: `model.layers.1.self_attn.o_proj`
- `reference_stats: true`
- rescalers: off
- calibration: `6` WikiText-2 train chunks, length `2048`
- probe eval: `8` WikiText-2 test chunks, length `2048`
- target timing: same-layer, post-QKV, pre-o_proj

## Timing And Sign Audit

- `Delta` definition used:
  - `Delta = R - R_hat = reference_layer_input - quantized_layer_input`
- official vs manual `Sigma_Delta,Xhat` mismatch: `0`
- official vs manual `Sigma_Xhat` mismatch: `0`
- wrong-sign mismatch: `3.8401e-3`
- q/k/v were already quantized before the audit target:
  - `q_proj` rel weight MSE: `5.5656e-2`
  - `k_proj` rel weight MSE: `7.7440e-2`
  - `v_proj` rel weight MSE: `6.1513e-2`

Diagnosis:

- paired-state timing is correct
- sign is correct
- no stale pre-QKV stats were reused for the `o_proj` target

## Residual Magnitude Audit

- `||W Sigma_X,Xhat||_F = 5.3348487864e-1`
- `||Sigma_Delta,Xhat||_F = 1.9200293112e-3`
- residual/base ratio at scale `1.0`: `3.5990e-3`
- `cond(Sigma_Xhat) = 6.5425e5`
- `cond(H after damping) = 5.4329e5`
- dead-feature pruning: `2048 -> 2048` kept, `0` pruned
- Cholesky diagonal range:
  - min: `2.1751e-3`
  - max: `8.8565e-2`

This shows the corrected residual term is tiny relative to the base target. The catastrophic behavior came from amplifying that tiny term with the wrong formula.

## Residual-Strength Sweep

| Residual scale | small-eval PPL | rel weight MSE | `target_y_max_abs` | entropy bits | Huffman bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `9.6219` | `1.5426e-1` | `0.190427` | `3.3992` | `3.4454` |
| `0.25` | `9.6313` | `1.5752e-1` | `0.190430` | `3.3993` | `3.4464` |
| `0.50` | `9.6356` | `1.6780e-1` | `0.190433` | `3.4000` | `3.4467` |
| `0.75` | `9.6459` | `1.8609e-1` | `0.190436` | `3.4022` | `3.4499` |
| `1.00` | `9.6433` | `2.1246e-1` | `0.190439` | `3.4029` | `3.4490` |

Key observation:

- the corrected path changes smoothly with residual scale
- there is no sign-flip or runaway behavior
- scale `1.0` remains sane

## Legacy-Formula Audit

Using the old solved residual addend on the exact same statistics gives:

- legacy residual-term norm: `5.9003`
- legacy residual/base ratio: `11.0599`
- legacy `target_y_fro_norm = 2515.3740`
- legacy `target_y_max_abs = 85.9470`

So the old residual path failed because the wrong formula converted a tiny residual covariance into a dominant target term.

## Multi-Layer Smoke Re-Run

After fixing the formula, I reran the exact same residual-enabled smoke that previously failed:

- quant config: `configs/quant/watersic_llama32_1b_multilayer_smoke_ref_stagefix_residfixed.yaml`
- eval config: `configs/eval/wikitext2_smoke8.yaml`
- scope: first `11` linear modules
- rescalers: off

Result:

- achieved effective bits: `2.9919`
- entropy bits: `2.9786`
- Huffman bits: `3.0343`
- baseline small-eval PPL: `8.9880`
- quantized small-eval PPL: `9.6433`
- runtime: `993.99s`
- peak memory: `18.65 GiB`

At the old failure point:

| Run | `layer1 self_attn.o_proj` rel MSE | `target_y_max_abs` | small-eval PPL |
| --- | ---: | ---: | ---: |
| old residual-enabled smoke | `4.0915e9` | `85.9470` | `700443.07` |
| no-residual ablation | `1.5426e-1` | `0.190427` | `9.6219` |
| fixed residual-enabled smoke | `2.1246e-1` | `0.190439` | `9.6433` |

## Final Diagnosis

The answer to the residual blocker question is:

- timing: correct
- sign: correct
- formula: wrong before, fixed now
- scale blow-up: yes, but as a consequence of the wrong formula

The true residual term is tiny. The old inverse-based path amplified it into the blow-up.

## Next Step

Run the fuller `Llama-3.2-1B` ~`3.0`-bit experiment with:

- `reference_stats: true`
- fixed residual correction
- rescalers still off

and only then compare the first full-model post-fix result against the paper.
