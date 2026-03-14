# Full-Model Llama-3.2-1B Adaptive Mixing Repair Report

Date: 2026-03-15

## Scope

This round focused only on repairing adaptive mixing for `Llama-3.2-1B` at about `3.0` bits.

Reference point before this round:

- best completed full-model point: `llama32_1b_full_3p0bit_reftrue_rescaler`
- effective bits: `2.9984`
- WikiText-2 PPL: `15.7029`

Problem entering this round:

- the completed adaptive-mixing full-model run was stable but worse:
  - effective bits: `2.9984`
  - WikiText-2 PPL: `16.6096`

## Paper Audit Outcome

The paper audit identified the main mismatch:

- the old implementation re-ran binary search over `c` inside every `epsilon_qr` and `epsilon_aw` candidate evaluation
- the paper says to calibrate first, reuse the step-1 scales during the coordinate search, and recalibrate only once at the end

Saved audit notes:

- `outputs/reports/adaptive_mixing_paper_audit.md`
- `outputs/reports/adaptive_mixing_mismatch_diagnosis.md`
- `outputs/reports/adaptive_mixing_runtime_optimization.md`

## Rejected Repair

A stricter shared-stage-`c` interpretation was tested first and rejected.

Observed failure on the interrupted repair-check:

- layer: `model.layers.0`
- `q_proj` achieved rate: `3.0246`
- `k_proj` achieved rate: `3.4078`
- `v_proj` achieved rate: `1.3199`
- `v_proj` relative weight MSE: about `0.93`

Conclusion:

- forcing a single shared scalar across the entire QKV triplet was not credible enough to keep

## Accepted Repair

The adopted repair is:

1. run the normal initial QKV calibration at `(epsilon_qr, epsilon_aw) = (0, 0)`
2. record the step-1 calibrated Q/K/V scales
3. reuse those scales during the `epsilon_qr` and `epsilon_aw` searches
4. keep the final recalibration pass after selecting `(epsilon_qr*, epsilon_aw*)`

This removes repeated binary searches from the search loop while preserving the stable final quantization path.

## Prefix Validation

Validation run:

- config: `configs/quant/watersic_llama32_1b_prefix2_reftrue_rescaler_mixing_repaircheck.yaml`
- report bundle:
  - `outputs/reports/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.json`
  - `outputs/reports/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.md`

Result:

- calibration: `8` chunks
- eval: WikiText-2 smoke-8
- achieved effective bits: `2.9862`
- entropy bits: `2.9743`
- Huffman bits: `3.0262`
- baseline small-eval PPL: `8.9880`
- quantized small-eval PPL: `9.5600`
- total runtime: `3441.95s`
- peak GPU memory: `18.65 GiB`

Adaptive-mixing timing improvement on the repaired path:

- layer 0 search time:
  - old full-model run: about `4002s`
  - repaired validation: about `394.5s`
  - improvement: about `10.1x`
- layer 1 search time:
  - old full-model run: about `2526s`
  - repaired validation: about `444.8s`
  - improvement: about `5.7x`

Numerical behavior:

- layer 0 remained sane after the repair
- layer 1 remained sane after the repair
- no new NaN/Inf or residual-path instability appeared

## Full-Model Run Status

The repaired full-model run has been launched and is still in progress as of this report update.

Run:

- config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml`
- log: `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_20260315_015946.log`
- GPU pin: `CUDA_VISIBLE_DEVICES=4`

Latest confirmed progress from the live run:

- layer 0 completed
- layer 1 completed
- layer 2 completed
- layer 3 completed
- layer 4 completed
- layer 5 adaptive-mixing search reached:
  - `epsilon_qr = 0.516348`
  - `epsilon_aw = 0.837561`
  - timestamp: `2026-03-15 04:54:34`

No numerical instability has been observed in the repaired full-model path so far.

## Current Honest Status

What is now true:

- the adaptive-mixing search mismatch has been audited
- the expensive repeated-rate-search path has been removed from the coordinate search
- the repaired path is validated on a nontrivial 2-layer run
- the full repaired benchmark run is actively in progress

What is not yet true:

- there is not yet a new completed full-model repaired PPL number in the repo
- the best completed full-model point is still the rescaler-only run at `15.7029`

The final comparison against:

- rescaler-only baseline
- old adaptive-mixing run
- paper `10.57` point

must be appended after the in-progress full-model run finishes.
