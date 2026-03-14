# Adaptive Mixing Paper Audit

Date: 2026-03-15

Model focus for this round: `Llama-3.2-1B`

Paper source:
- `paper/2603.04956v1.pdf`
- adaptive-mixing definitions: Appendix C, eqs. `(58)`, `(59)`, objective `(60)` on pages `16-17`
- per-layer search procedure: Section D, pages `18-19`

## Paper-Backed Adaptive-Mixing Purpose

The paper introduces adaptive mixing because drift-corrected attention calibration can become overly fixated on distorted quantized activations `X_hat` when prior quantization error is large. The stated goal is to stabilize the QKV quantizer by interpolating back toward the original statistics when the drift-corrected Hessian becomes less reliable.

For attention layers, the paper says the relevant quantity to improve is the relative MSE at the `wo` input, not the individual Q/K/V projection outputs. The reason given is that the softmax nonlinearity amplifies QKV errors, so `wo`-input distortion is the better proxy for final attention loss.

## Exact Paper Procedure

### Scope

Adaptive mixing applies only to the joint quantization of `wq`, `wk`, and `wv`.

The paper explicitly says it is not applied to the other matrices; for the rest, both `epsilon` values are set to zero.

### Statistics Being Mixed

The paper defines drift mixing as:

- `Sigma_hatX^(mix) = (1 - epsilon_qr) Sigma_hatX + epsilon_qr Sigma_X`
- `Sigma_X,hatX^(mix) = (1 - epsilon_qr) Sigma_X,hatX + epsilon_qr Sigma_X`

Then attention weighting is applied on top of the drift-mixed statistics:

- `Sigma_•^(final) = (1 - epsilon_aw) Sigma_•^(w) + epsilon_aw Sigma_•^(mix)`

for `Sigma_• in {Sigma_X, Sigma_hatX, Sigma_X,hatX}`.

This means:

1. drift mixing is applied first
2. attention weighting is applied second
3. damping is applied after the final blended statistics are formed

### Search Sequence

The paper’s per-layer procedure on pages `18-19` is:

1. Run binary search over `c` with `epsilon_qr = 0` and `epsilon_aw = 0`.
2. Hold `epsilon_aw = 0`, search `epsilon_qr in [0, 1]` by golden-section search.
3. Fix `epsilon_qr = epsilon_qr*`, search `epsilon_aw in [0, 1]` by golden-section search.
4. With `(epsilon_qr*, epsilon_aw*)` fixed, rerun binary search over `c` for the final quantization.

The paper is explicit that steps `2` and `3` reuse the scale `c` from step `1`, and only step `4` reruns rate calibration.

### Local Objective

The search objective is the relative MSE at the `wo` input:

- numerator: `||Attn(X; wq, wk, wv) - Attn(X_hat; wq_hat, wk_hat, wv_hat)||_F^2`
- denominator: `||Attn(X; wq, wk, wv)||_F^2`

The paper states:

1. `Attn(X; ...)` uses the original layer input `X` and original QKV weights
2. `Attn(X_hat; ...)` uses the already-quantized upstream activations `X_hat` and the candidate quantized QKV weights
3. the comparison point is the attention-block output before `wo`
4. each candidate evaluation re-quantizes the QKV triplet jointly and runs a forward pass through the attention block on the calibration set

### Interaction With Other Components

- activation drift correction: adaptive mixing directly modifies the drift-corrected covariances
- attention weighting: applied on top of the drift-mixed statistics
- residual correction: not part of the QKV adaptive-mixing objective itself; residual correction is for `wo` / `w2`
- damping: applied after adaptive mixing on the final blended matrices
- rate search: initial `c` search at `(0, 0)`, fixed-`c` candidate evaluations during coordinate search, then final `c` recalibration

## Current Implementation Audit

Code inspected:
- `src/watersic/quant/attention_mixing.py`
- `src/watersic/quant/watersic_layer.py`
- `src/watersic/quant/watersic_model.py`

### Matches The Paper

1. Scope is restricted to `q_proj`, `k_proj`, `v_proj`.
2. The search order is `epsilon_qr` first, then `epsilon_aw`.
3. The local search objective is measured at the `o_proj` input, which is the `wo` input.
4. The objective compares cached reference `wo` inputs from the full-precision model against candidate `wo` inputs from the partially quantized model.
5. The implementation applies damping after the selected blended statistics are formed.

### Current Differences From The Paper

1. The current search re-runs rate search inside every candidate evaluation.
   - Current code path: `attention_mixing.py -> objective_for() -> _quantize_qkv_stage_candidate() -> quantize_linear_layer() -> binary_search_c()`
   - Paper procedure: reuse the step-1 `c` during the `epsilon_qr` and `epsilon_aw` searches, then rerun rate calibration only once at the end.

2. The current candidate path recalibrates each QKV matrix independently instead of reusing the step-1 calibrated scales during the coordinate search.
   - Current code quantizes `q_proj`, `k_proj`, and `v_proj` sequentially with fresh per-matrix rate recalibration inside every candidate evaluation.
   - The paper describes an initial rate-calibration step for the QKV triplet, then fixed-scale requantization of that triplet during the coordinate search.

3. The current implementation uses a model forward with an early-stop pre-hook to capture the `wo` input.
   - This is mathematically consistent with the objective.
   - It is slower than a direct attention-block-only evaluation, but it is not an algorithmic mismatch by itself.

## Audit Conclusion

The most important current mismatch is not numerical stability; it is the adaptive-mixing search procedure itself.

The paper says:

1. calibrate the QKV stage once at `(epsilon_qr, epsilon_aw) = (0, 0)`
2. reuse the step-1 calibrated Q/K/V scales during both coordinate searches
3. requantize QKV jointly during the candidate evaluations
4. rerun rate calibration only once at the end

The current code instead performs a fresh per-matrix binary search inside every candidate evaluation. That differs from the paper and also explains most of the runtime explosion in the completed adaptive-mixing run.

The paper wording does not clearly require a single shared scalar `c` across all three QKV matrices. During this round, that stricter shared-`c` interpretation was tested and rejected because it caused severe rate imbalance within the QKV triplet. The adopted repair is therefore the conservative paper-backed version:

- reuse the step-1 calibrated per-matrix Q/K/V scales during the coordinate search
- keep the final recalibration pass after the optimal `(epsilon_qr*, epsilon_aw*)` is selected

This paper audit therefore identifies the first repair target for this round:

- switch the adaptive-mixing search to a paper-aligned fixed-scale QKV candidate path
- keep the final post-search recalibration step
