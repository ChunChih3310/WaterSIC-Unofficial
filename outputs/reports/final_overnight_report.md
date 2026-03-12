# Final Overnight Report

## Scope

This overnight session produced a real, runnable WaterSIC codebase and four real `Llama-3.2-1B` smoke quantization runs. The completed runs are not full-paper reproductions yet: they quantize layer 0 only, save reconstructed-weight artifacts, benchmark them on WikiText-2, and expose a concrete numerical failure mode in the current Q/K path.

## What Was Implemented

- repo-local conda environment and cache isolation
- repo-local path guard
- automatic idle-GPU selection
- deterministic WikiText-2 calibration/eval chunking
- transformed-space ZSIC with unequal spacing and LMMSE correction
- entropy / Huffman / side-information bitrate accounting
- dead-feature erasure
- transformed activation-drift and residual-stream objectives
- transformed-objective diagonal row/column rescalers
- sequential model quantization in model order
- saved artifacts, reports, logs, configs, and unit tests

## Completed Runs

All completed runs used:

- model: `meta-llama/Llama-3.2-1B`
- eval: WikiText-2 test, context length `2048`
- calibration: WikiText-2 train, `2` sequences of length `2048`
- scope: layer `0` only, `7` linear modules

| Run | Git | Attention Weighting | Achieved Bits | Baseline PPL | Quantized PPL |
| --- | --- | --- | ---: | ---: | ---: |
| `llama32_1b_smoke_3p0bit` | `unknown` | original first attempt | 3.0104 | 9.7041 | 15835.2186 |
| `llama32_1b_smoke_3p0bit_attnfix` | `ec44fef` | corrected eq. (19) implementation | 3.0101 | 9.7041 | 24887.4715 |
| `llama32_1b_smoke_3p0bit_covfix` | `00d7223` | corrected dead-feature source covariance | 3.0101 | 9.7041 | 24887.4715 |
| `llama32_1b_smoke_3p0bit_noaw` | `00d7223` | disabled for QKV diagnosis | 3.0118 | 9.7041 | 18227.7697 |

## Comparison Against The Paper

### Llama-3.2-1B baseline

- paper unquantized PPL: `9.76`
- our baseline PPL: `9.7041`
- absolute difference: `-0.0559`
- diagnosis: the evaluation setup is close enough to the paper baseline to trust the WikiText-2 runner

### Llama-3.2-1B around 3 bits

- paper WaterSIC: `3.00` bits, `10.57` PPL
- our best completed smoke run: `3.0104` bits, `15835.2186` PPL
- absolute difference: `+15824.6486`
- diagnosis: not a valid reproduction yet; this run quantizes only layer 0 and already reveals a severe instability in `q_proj` and `k_proj`

### Qwen3-8B

- paper WaterSIC at `3.125` bits: `10.03` PPL
- our result: not run yet
- diagnosis: config support exists, but runtime was better spent first exposing and localizing the Llama instability

## Failure Analysis

The dominant failure mode is concentrated in Q/K rather than the whole layer:

- first smoke run:
  - `q_proj` weighted error: `1.260e+07`
  - `k_proj` weighted error: `1.098e+07`
  - `v_proj` weighted error: `2.213e+03`
- disabling QKV attention weighting improved those transformed errors materially:
  - `q_proj`: `4.778e+06`
  - `k_proj`: `4.244e+06`
  - `v_proj`: `1.012e+03`
- despite that improvement, the final PPL remained catastrophic

Most likely explanations, ordered by current evidence:

1. The Q/K transformed target or rescaling path is still not faithful enough in the current implementation.
2. Attention-weighted calibration is still too unstable in smoke mode and needs the full adaptive mixing search from the paper.
3. Reference-model paired statistics are still disabled in the completed runs, which weakens drift correction.

## What Is Already Enabled In The Runs

- integer-grid quantization with entropy/Huffman reporting
- unequal per-column spacing
- ZSIC from right to left
- LMMSE shrinkage
- dead-feature erasure
- residual correction for `o_proj` and `down_proj`
- sequential model-order quantization

## What Is Still Missing Or Only Partially Validated

- adaptive mixing search for `epsilon_qr` and `epsilon_aw`
- stable, paper-faithful QKV attention-weighted calibration in real runs
- reference-model paired statistics in the completed runs
- full-model Llama-3.2-1B quantization
- any Qwen3-8B experiment

## Concrete Artifacts To Inspect

- reports:
  - `outputs/reports/llama32_1b_smoke_3p0bit*.{json,md}`
  - `outputs/reports/final_overnight_report.md`
  - `outputs/reports/aggregate_report.md`
- logs:
  - `outputs/logs/run_llama32_1b_smoke_3p0bit*.log`
- quantized metadata:
  - `outputs/quantized/llama32_1b_smoke_3p0bit*/metadata.json`
  - `outputs/quantized/llama32_1b_smoke_3p0bit*/layer_results.json`

## Next Steps

1. Implement and enable the paper’s adaptive mixing search around the `wo`-input distortion objective.
2. Run the smoke path again with reference-model paired statistics enabled.
3. Only after Q/K stabilize, scale from layer-0 smoke to multi-layer and then full-model `Llama-3.2-1B`.
