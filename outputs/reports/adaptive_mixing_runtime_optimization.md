# Adaptive Mixing Runtime Optimization

Date: 2026-03-15

## Baseline Runtime Problem

The completed full-model adaptive-mixing run was stable but too expensive:

- rescaler-only full-model run: `18525.23s`
- old full adaptive-mixing run: `74349.91s`
- slowdown vs rescaler-only: about `4.01x`

The dominant runtime cost was the adaptive-mixing search itself.

## Old Runtime Bottleneck

The old implementation performed a fresh binary search over `c` inside every `epsilon_qr` and `epsilon_aw` candidate evaluation.

That means each candidate evaluation did:

1. `q_proj` binary search over `c`
2. `k_proj` binary search over `c`
3. `v_proj` binary search over `c`
4. a forward pass to the `wo` input

With:

- `30` binary-search iterations
- `15` golden-section iterations for `epsilon_qr`
- `15` golden-section iterations for `epsilon_aw`

the search loop was dominated by repeated rate calibration rather than by the actual `wo`-input objective.

## Paper-Backed Optimization Adopted

The paper procedure on pages `18-19` says:

1. do an initial rate calibration at `epsilon_qr = 0`, `epsilon_aw = 0`
2. reuse the step-1 calibrated scales during the coordinate search
3. rerun rate calibration only once at the end

The repaired implementation now follows that procedure in the adaptive-mixing search:

- initial candidate:
  - run the normal per-matrix rate calibration once
  - record the selected `c` for each of `q_proj`, `k_proj`, `v_proj`
- coordinate search:
  - reuse those step-1 per-matrix scales
  - do not rerun binary search inside the candidate loop
- final quantization:
  - keep the existing final per-matrix recalibration pass

This changes the implementation cost, not the intended paper objective.

## Rejected Interpretation

A stricter shared-stage-`c` interpretation was tested and rejected.

That attempt forced one common `c` across the whole QKV triplet. In the first repair-check it produced pathological layer-0 rate imbalance:

- `q_proj`: `3.0246`
- `k_proj`: `3.4078`
- `v_proj`: `1.3199`
- `v_proj` rel weight MSE: about `0.93`

That interpretation was not kept.

The accepted repair is therefore:

- fixed step-1 per-matrix scales during the coordinate search
- final recalibration preserved

## Measured Runtime Improvement

Validation run:

- config: `configs/quant/watersic_llama32_1b_prefix2_reftrue_rescaler_mixing_repaircheck.yaml`
- run name: `llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2`
- calibration: `8` chunks
- eval: WikiText-2 smoke-8

Result:

- total runtime: `3441.95s`
- quantized small-eval PPL: `9.5600`
- effective bits: `2.9862`

Per-layer adaptive-mixing timing from the saved audit:

### Layer 0

Old full-model run:

- initial objective logged at `03:34:38`
- `epsilon_qr` selected at `04:12:58`
- `epsilon_aw` selected at `04:41:20`
- old search time from initial log to final `epsilon_aw`: about `4002s`

Repaired run:

- initial candidate total: `75.52s`
- `epsilon_qr` search: `162.01s`
- `epsilon_aw` search: `156.99s`
- repaired search total: about `394.51s`

Improvement:

- about `10.1x` faster on the layer-0 adaptive-mixing search

### Layer 1

Old full-model run:

- initial objective logged at `05:00:58`
- `epsilon_qr` selected at `05:21:44`
- `epsilon_aw` selected at `05:43:04`
- old search time from initial log to final `epsilon_aw`: about `2526s`

Repaired run:

- initial candidate total: `73.78s`
- `epsilon_qr` search: `186.19s`
- `epsilon_aw` search: `184.83s`
- repaired search total: about `444.81s`

Improvement:

- about `5.7x` faster on the layer-1 adaptive-mixing search

## Correctness Risk Assessment

### Safe optimization

Reusing the step-1 calibrated scales during the coordinate search is paper-backed and therefore low-risk.

Why it is safe:

1. it matches the per-layer procedure described in the paper
2. it preserves the same local search objective at the `wo` input
3. it preserves the final recalibration step after `(epsilon_qr*, epsilon_aw*)` is selected
4. it does not reduce the paper defaults of `15` golden-section iterations or `30` binary-search iterations

### Not adopted

The shared-stage-`c` interpretation was not safe enough because the paper does not state that the whole QKV triplet must share a single scalar `c`, and the first validation attempt showed strong rate collapse.

## Current Status

The accepted repaired path is:

- numerically sane on the first two layers
- materially faster
- still paper-grounded

The next step is the full-model repaired benchmark run:

- `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml`

That run is intended to answer whether the repaired search path closes the quality gap relative to:

- rescaler-only full-model baseline: `15.7029`
- old adaptive-mixing full-model run: `16.6096`
