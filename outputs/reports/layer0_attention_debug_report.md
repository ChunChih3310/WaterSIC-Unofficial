# Layer-0 Attention Debug Report

## Scope

This report covers the strict staged correctness-debug campaign on `Llama-3.2-1B`, restricted to layer `0` attention:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

Calibration and eval used the dedicated small debug config:

- config: `configs/debug/llama32_1b_layer0_attention.yaml`
- calibration: `4` WikiText-2 train chunks at length `2048`
- small eval: `4` WikiText-2 test chunks at length `2048`

Artifacts for each stage are saved under:

- `outputs/reports/llama32_1b_layer0_attention_debug/`

## Root Cause Fixes Applied Before The Final Ladder

1. **ZSIC recursive update bug**
   - buggy behavior: subtracting `c * z_i * L_i,:`
   - correct behavior: subtract `alpha_i * z_i * L_i,:`

2. **Transformed-target `Y` solve bug**
   - buggy behavior: solving for `target_cross @ L^(-1)`
   - correct behavior: solving for `target_cross @ (L^T)^(-1)`
   - plain-WaterSIC consequence: `Y` now correctly reduces to `W L`

3. **Calibration-stat memory bug**
   - paired-model attention-stat collection was missing `torch.no_grad()`
   - this caused unnecessary memory growth / OOM

These fixes are the reason the final staged ladder below is materially different from the earlier catastrophic runs.

## Stage Results

Baseline small-eval PPL: `10.8101`

| Stage | Small-Eval PPL | Aggregate Rel MSE | q Rel MSE | k Rel MSE | v Rel MSE | o Rel MSE | Entropy bw | Huffman bw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. HPTQ-equivalent | 10.8238 | 0.001172 | 0.000967 | 0.001859 | 0.035443 | 0.020700 | 2.9154 | 2.9647 |
| B. PlainWaterSIC | 10.8601 | 0.002865 | 0.002365 | 0.004561 | 0.076288 | 0.040874 | 2.8878 | 2.9343 |
| C. + LMMSE | 10.8666 | 0.001990 | 0.001715 | 0.002904 | 0.051497 | 0.026458 | 2.8542 | 2.9002 |
| D. + activation drift correction | 10.8691 | 0.001990 | 0.001715 | 0.002904 | 0.051497 | 0.028639 | 2.8684 | 2.9170 |
| E. + residual correction | 10.8691 | 0.001990 | 0.001715 | 0.002904 | 0.051497 | 0.028639 | 2.8684 | 2.9170 |
| F. + attention weighting | 10.8712 | 0.002263 | 0.001951 | 0.003307 | 0.056346 | 0.031131 | 2.8574 | 2.9055 |
| G. + adaptive mixing | 10.8744 | 0.002046 | 0.001797 | 0.002861 | 0.052254 | 0.028477 | 2.8633 | 2.9143 |
| H. + diagonal rescalers | 10.8925 | 0.001928 | 0.001609 | 0.003005 | 0.052574 | 0.028381 | 2.8972 | 2.9480 |

Adaptive-mixing search results:

- stage `G`: `epsilon_qr = 0.034442`, `epsilon_aw = 0.381966`
- stage `H`: `epsilon_qr = 0.034442`, `epsilon_aw = 0.326238`

## Answers To The Debug Questions

### Which stage first becomes unstable?

After the two core math fixes above, **no stage in `A` through `H` became unstable on this small layer-0 attention debug run**.

Before those fixes, the ladder was already catastrophically bad at stage `A`, which showed that the first real bug was in the shared core math rather than in later WaterSIC extras.

### Is HPTQ-equivalent sane?

Yes.

- stage `A` PPL: `10.8238`
- baseline PPL: `10.8101`
- aggregate relative MSE: `0.001172`

### Is PlainWaterSIC sane?

Yes.

- stage `B` PPL: `10.8601`
- aggregate relative MSE: `0.002865`

PlainWaterSIC is slightly worse than stage `A` on this small debug slice, but it is numerically well-behaved.

### Is the instability caused by shared core math or by QKV-specific WaterSIC extras?

The earlier instability was caused by **shared core math**:

1. wrong ZSIC recursive update scale
2. wrong transformed-target triangular solve

Once those were fixed, the QKV-specific extras (`attention weighting`, `adaptive mixing`) no longer caused immediate instability on the small layer-0 ladder.

### What did the per-stage audit show?

- `reference_stats` are effectively neutral for `q_proj`, `k_proj`, and `v_proj` in layer `0`, because there are no earlier quantized layers feeding them.
- `reference_stats` matter for `o_proj`, where the `q/k/v` decisions change the downstream attention output.
- residual correction is a no-op in this exact layer-0-only debug setup, which is why stages `D` and `E` are numerically identical.
- recursive-update audit errors are near machine precision, so the corrected ZSIC recursion is now internally self-consistent.

## Current Conclusion

The strict staged debug campaign achieved its goal:

- the first true correctness bugs were identified
- they were fixed
- the previously catastrophic layer-0 attention path is now numerically sane through stages `A` to `H`

What is **not** concluded here:

- this is not yet a full-model reproduction
- this is not yet a final paper-comparison run
- diagonal rescalers should still remain disabled by default until broader validation is completed
