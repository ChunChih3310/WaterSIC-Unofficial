# Final Worst-Layer Diagnosis

## Scope

This diagnosis compares:

- best rescaler-only calibration-sweep point:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.json`
- final best paper-scale repaired-mixing point:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.json`
- matched validation benchmark context:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_validation_benchmark.json`

Important note:

- the validation benchmark reuses the same saved paper-scale quantized artifact
- therefore it does not change the layerwise distortion profile
- it only changes the evaluation split from test to validation

## 1. Worst Layers by Relative Weight MSE

### Rescaler-only 32-chunk run

Top 5:

1. `model.layers.13.self_attn.o_proj` `0.2863`
2. `model.layers.0.self_attn.o_proj` `0.2650`
3. `model.layers.11.self_attn.o_proj` `0.2556`
4. `model.layers.4.self_attn.o_proj` `0.2540`
5. `model.layers.12.self_attn.o_proj` `0.2503`

Family summary:

- `o_proj` mean relative weight MSE: `0.2025`
- `down_proj` mean relative weight MSE: `0.1292`
- next highest family:
  - `k_proj` `0.1051`

Conclusion:

- the remaining distortion was overwhelmingly concentrated in residual-path `self_attn.o_proj`

### Paper-scale repaired-mixing run

Top 5:

1. `model.layers.0.self_attn.o_proj` `0.1924`
2. `model.layers.2.self_attn.o_proj` `0.1473`
3. `model.layers.1.mlp.down_proj` `0.1415`
4. `model.layers.4.self_attn.o_proj` `0.1342`
5. `model.layers.5.self_attn.o_proj` `0.1224`

Family summary:

- `o_proj` mean relative weight MSE: `0.1099`
- `down_proj` mean relative weight MSE: `0.0706`
- next highest family:
  - `k_proj` `0.0674`

Conclusion:

- the residual-path family still dominates
- paper-scale calibration and repaired adaptive mixing reduced the errors sharply, but did not change the overall family ranking

## 2. Worst Layers by Available Activation/Input Proxy

The available per-layer proxy in the saved reports is `weighted_error`.

### Rescaler-only 32-chunk run

Top entries:

1. `model.layers.6.self_attn.k_proj` `3.23e-06`
2. `model.layers.5.self_attn.k_proj` `3.05e-06`
3. `model.layers.7.self_attn.k_proj` `3.00e-06`
4. `model.layers.8.self_attn.k_proj` `2.97e-06`
5. `model.layers.9.self_attn.k_proj` `2.83e-06`

### Paper-scale repaired-mixing run

Top entries:

1. `model.layers.11.self_attn.k_proj` `4.39e-06`
2. `model.layers.6.self_attn.k_proj` `4.28e-06`
3. `model.layers.10.self_attn.k_proj` `4.22e-06`
4. `model.layers.5.self_attn.k_proj` `4.20e-06`
5. `model.layers.2.self_attn.k_proj` `3.74e-06`

Interpretation:

- by relative weight MSE, the hardest family is still residual-path `o_proj` and then `down_proj`
- by the saved `weighted_error` proxy, the largest remaining importance-weighted activation/input errors are concentrated in `k_proj`
- the two diagnostics are not contradictory; they measure different failure modes

## 3. Are the Same Layers Consistently Worst Across Runs?

### Family-level consistency

Yes.

- `self_attn.o_proj` remains the hardest family across the best `32`-chunk rescaler-only run and the final paper-scale repaired-mixing run
- `mlp.down_proj` remains the second-hardest family by mean relative weight MSE

### Exact layer-index consistency

Only partially.

- `32`-chunk rescaler-only worst layers were dominated by later `o_proj` blocks:
  - layers `13`, `11`, `12`
- paper-scale repaired mixing shifted the worst relative-weight-MSE layers toward earlier residual-path blocks:
  - `layer 0 o_proj`
  - `layer 2 o_proj`
  - `layer 1 down_proj`
  - `layer 4 o_proj`
  - `layer 5 o_proj`

Interpretation:

- the exact bottleneck layers moved as calibration improved
- the bottleneck family did not

## 4. What Family Still Looks Hardest?

The residual-path family is still the main source of the remaining paper gap:

- `self_attn.o_proj` is still the largest mean-error family
- `mlp.down_proj` is still second
- the top-5 relative-weight-MSE list in the final best run contains four `o_proj` layers and one `down_proj`

This is stronger evidence than the `weighted_error` proxy, because the largest absolute remaining family-level distortion still sits on the residual-contributing projections.

## 5. Most Plausible Reason These Layers Remain Hard

The most plausible explanation is structural rather than a missing feature:

- `o_proj` and `down_proj` directly feed the residual stream
- small errors there are repeatedly re-injected into later computation
- these layers also rely on the most complicated correction path:
  - activation drift correction
  - residual compensation
  - same-layer sequential refresh

The repo already repaired the residual-compensation correctness bug, so the remaining difficulty now looks like residual-path sensitivity, not a simple formula/sign bug.

The `k_proj` dominance under `weighted_error` suggests that attention importance remains sharp in certain attention blocks, but the final model-quality bottleneck still appears to come from the residual-path families.

## 6. What Should Be Targeted First If More Gap Reduction Is Needed?

Priority order:

1. `self_attn.o_proj`
   - especially layers `0`, `2`, `4`, `5`
2. `mlp.down_proj`
   - especially `model.layers.1.mlp.down_proj`
3. secondary watchlist:
   - `k_proj` blocks with the highest `weighted_error`, especially layers `11`, `10`, `6`, `5`, `2`

Why this order:

- the residual-path family still dominates the final relative-weight-MSE ranking
- these layers remain the most plausible source of the remaining validation gap
- the validation benchmark uses the same artifact, so the validation-gap diagnosis points back to these same paper-scale layer distortions rather than a new benchmark-only issue

## Bottom Line

1. The exact worst layers in the final best path are:
   - `model.layers.0.self_attn.o_proj`
   - `model.layers.2.self_attn.o_proj`
   - `model.layers.1.mlp.down_proj`
   - `model.layers.4.self_attn.o_proj`
   - `model.layers.5.self_attn.o_proj`
2. They still belong mainly to the residual-path families:
   - `self_attn.o_proj`
   - `mlp.down_proj`
3. The exact worst layer indices changed from the `32`-chunk rescaler-only run, but the hardest family did not.
4. The most plausible remaining limiter is residual-path sensitivity, not a missing major WaterSIC component.
5. If more gap reduction is pursued, the first target should still be residual-path projections before anything else.
