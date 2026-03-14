# Adaptive Mixing Mismatch Diagnosis

Date: 2026-03-15

Compared runs:
- `full_reftrue_rescaler`
- `full_reftrue_rescaler_mixing`

Reference point:
- best completed full-model point is still `full_reftrue_rescaler`
- effective bits: `2.9984`
- WikiText-2 PPL: `15.7029`

Regressing point:
- `full_reftrue_rescaler_mixing`
- effective bits: `2.9984`
- WikiText-2 PPL: `16.6096`

## Observed Mismatch

The completed adaptive-mixing run improved the local QKV search objective in every attention layer, but the full-model PPL got worse.

Global evidence:

- full-model PPL: `15.7029 -> 16.6096` (`+0.9067`)
- mean relative weight MSE:
  - `q_proj`: `0.0696 -> 0.0611`
  - `k_proj`: `0.0983 -> 0.0898`
  - `v_proj`: `0.0876 -> 0.0806`
  - `o_proj`: `0.3170 -> 0.3175`
  - `down_proj`: `0.3023 -> 0.3024`

This means the search improved QKV reconstruction but did not improve the downstream error modes that still dominate the final benchmark.

## Representative Attention Blocks

The table below compares the rescaler-only baseline against the completed adaptive-mixing run.

| Layer | `epsilon_qr*` | `epsilon_aw*` | local `wo`-input rel MSE | `q_proj` rel MSE | `k_proj` rel MSE | `v_proj` rel MSE | `o_proj` delta | `down_proj` delta |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| `0` | `0.0003` | `0.7082` | `0.007913 -> 0.006750` | `0.1541 -> 0.1562` | `0.2057 -> 0.2082` | `0.2044 -> 0.2047` | `-0.0009` | `-0.0011` |
| `1` | `0.5228` | `0.5248` | `0.020070 -> 0.017892` | `0.0690 -> 0.0646` | `0.1097 -> 0.1060` | `0.0983 -> 0.0957` | `-0.0008` | `-0.0093` |
| `8` | `0.4164` | `0.7638` | `0.021616 -> 0.018887` | `0.0748 -> 0.0595` | `0.0863 -> 0.0723` | `0.0856 -> 0.0732` | `+0.0001` | `+0.0010` |
| `12` | `0.3827` | `0.6819` | `0.017053 -> 0.015065` | `0.0570 -> 0.0523` | `0.0859 -> 0.0819` | `0.0727 -> 0.0695` | `+0.0023` | `+0.0011` |
| `13` | `0.5298` | `0.7252` | `0.018850 -> 0.016596` | `0.0509 -> 0.0466` | `0.0818 -> 0.0773` | `0.0637 -> 0.0607` | `+0.0051` | `+0.0007` |
| `15` | `0.3790` | `0.6056` | `0.008610 -> 0.007413` | `0.0403 -> 0.0386` | `0.0591 -> 0.0577` | `0.0506 -> 0.0493` | `-0.0006` | `-0.0000` |

Interpretation:

1. The local objective improved in all representative blocks.
2. Q/K/V reconstruction improved in almost all representative blocks.
3. The downstream residual-path modules did not improve consistently.
4. Late-layer `o_proj` often got slightly worse even when Q/K/V improved.

## Epsilon Pattern

Selected coefficients from the completed run show a consistent pattern:

- `epsilon_qr*` is moderate in most layers: about `0.33` to `0.53`
- `epsilon_aw*` is high in most layers: about `0.52` to `0.86`

This means the current search often prefers substantial interpolation away from full attention weighting.

That pattern is not itself a proof of a bug, but in the current implementation it is suspicious because the paper-aligned search should evaluate candidate `epsilon` values at a fixed stage scale `c`, while the current implementation re-runs rate search inside every candidate evaluation.

## Most Likely Cause

The strongest diagnosis from the paper audit plus the completed run data is:

`adaptive mixing is currently optimizing a moving target`

Specifically:

1. during the `epsilon_qr` and `epsilon_aw` searches, the code re-runs binary search over `c` inside every candidate evaluation
2. therefore the local search is not ranking candidate `(epsilon_qr, epsilon_aw)` pairs at a fixed rate-calibrated quantizer family
3. the selected `epsilon` values can look good under the candidate-specific re-searched quantizer, but fail to transfer once the stage is quantized again in the real sequential pipeline

This directly matches the paper mismatch identified in `adaptive_mixing_paper_audit.md`.

## What This Diagnosis Does Not Claim

1. This is not a new numerical-stability bug. The run was stable.
2. This is not yet evidence that calibration size is the dominant blocker.
3. This is not evidence that adaptive mixing itself is wrong in the paper; the current problem is that the implementation does not yet follow the paper’s search procedure closely enough.

## Conclusion

The most likely reason adaptive mixing currently hurts full-model PPL is that the implementation does not evaluate candidate `epsilon` values under the paper’s fixed-`c`, joint-QKV search procedure.

The first repair for this round should therefore be:

1. initial QKV rate calibration at `(0, 0)`
2. fixed-scale candidate evaluations during the coordinate search, reusing the step-1 calibrated Q/K/V scales
3. final recalibration after selecting `(epsilon_qr*, epsilon_aw*)`

This repair is both:

- paper-backed
- the main available runtime reduction, because it removes repeated binary searches from every golden-section candidate evaluation
