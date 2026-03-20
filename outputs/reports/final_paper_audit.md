# Final Paper Audit

## Scope

This audit covers the completed `Llama-3.2-1B ~3.0-bit` paper-scale run and its matched WikiText-2 validation rerun:

- quantization report:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.md`
- validation benchmark:
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_validation_benchmark.json`
  - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_validation_benchmark.md`
- comparison context:
  - `outputs/reports/full_llama32_1b_adaptive_mixing_paperscale_report.md`
  - `outputs/reports/full_llama32_1b_quality_recovery_comparison.md`
- repo status:
  - `docs/reproduction_log.md`
  - `docs/known_issues.md`
- paper:
  - `paper/2603.04956v1.pdf`

## Paper-Match Checklist

| Category | Paper | Repo final run | Match status | Likely significance |
| --- | --- | --- | --- | --- |
| Model | `Llama-3.2-1B` | `meta-llama/Llama-3.2-1B` | close match | minor |
| Model/tokenizer revision | paper text does not pin a public HF revision in the extracted passages | `model_revision: main`, `tokenizer_revision: main` | not exact | minor reproducibility caveat |
| Calibration dataset | WikiText-2 train split | WikiText-2 train split | matches | negligible |
| Calibration chunking | non-overlapping `2048`-token sequences | non-overlapping `2048`-token sequences, `1188` chunks | matches in method | minor exact-count caveat |
| Evaluation split | WikiText-2 validation perplexity | matched rerun now uses WikiText-2 validation | matches | important caveat removed |
| Evaluation method | perplexity at context length `2048` | perplexity at `2048`, same runner family as prior results | matches closely | minor |
| Target bitrate | `3.00` bits | target `3.0`, achieved `2.9984` effective bits | matches | negligible |
| Activation drift correction | enabled | enabled | matches | negligible |
| Residual compensation | enabled for residual-path projections | enabled; corrected to paper equation (18) form | matches | negligible |
| Attention-weighted calibration | QKV only | QKV only | matches | negligible |
| Adaptive mixing | QKV only, with staged search | repaired per-block search enabled | matches closely | minor implementation-detail caveat only if paper used a different exact HF revision |
| Diagonal rescalers | enabled | enabled (`max_rescaler_iters: 4`) | matches | negligible |
| Damping | `1e-4` | `1e-4` | matches | negligible |
| Binary search iterations | `30` | `30` | matches | negligible |
| Golden-section iterations | `15` | `15` | matches | negligible |
| Rate-search row sampling | `10%` | `0.1` | matches | negligible |
| Dead-feature erasure | median-based threshold with `tau = 1e-3` | `dead_feature_tau: 1e-3` | matches closely | negligible |
| Remaining fallback behavior | no paper fallback path described | `reference_stats_effective_count = 109 / 112`; first-layer QKV is expected non-effective edge case | acceptable | negligible |
| Runtime shortcuts | paper describes the algorithm, not engineering cache details | repo-local token-block cache, `batch_size: 2` collection, no algorithm shortcut | not literal, but semantics-preserving | minor |
| Reporting style | paper emphasizes entropy-derived effective rate and perplexity | repo reports entropy/effective bits, plus Huffman and side info | matches and is more verbose | negligible |

## What Now Matches Closely

- The model family and target operating point now line up with the paper point:
  - `Llama-3.2-1B`
  - target `3.00` bits
  - achieved `2.9984` effective bits
- The final run uses the paper-scale calibration setup:
  - WikiText-2 train split
  - non-overlapping `2048`-token chunks
  - `1188` chunks under the current tokenizer pipeline
- The full WaterSIC path required for the paper result is present in the final run:
  - activation drift correction
  - residual compensation
  - attention-weighted QKV calibration
  - repaired adaptive mixing
  - diagonal rescalers
  - dead-feature erasure
  - paper-default damping and search settings
- The final benchmark caveat about test versus validation is now removed because the saved paper-scale artifact was rerun on WikiText-2 validation.

## What Still Does Not Fully Match

### 1. Exact model/tokenizer revision is still not pinned to a paper-specific public snapshot

- Paper:
  - the extracted passages do not give a public HF revision hash
- Repo:
  - uses `model_revision: main` and `tokenizer_revision: main`
- Assessment:
  - this is a real reproducibility caveat, but not evidence of a missing algorithmic component
- Likely impact:
  - minor

### 2. Exact tokenizer/BOS/chunk-count realization is not proven identical

- Paper:
  - states full-train-split calibration at `2048` context
  - does not publish the exact chunk count in the extracted text
- Repo:
  - current tokenizer path produces `1188` non-overlapping chunks
- Assessment:
  - methodology matches, exact token-stream realization is not proven identical
- Likely impact:
  - minor

### 3. The remaining validation gap is now a real numeric gap, not a split mismatch

- Paper validation PPL:
  - `10.57`
- Repo validation PPL:
  - `10.9310`
- Assessment:
  - this is no longer explainable by the test/validation split mismatch
  - it is a real paper-comparable difference
- Likely impact:
  - methodologically meaningful, even though it is much smaller than the earlier gaps

## Runtime Shortcuts and Whether They Matter

The final run still uses a few engineering optimizations that are not described in the paper, but they do not change the intended algorithm:

- repo-local WikiText-2 token-block cache
  - effect: removes repeated tokenization/chunk-build cost
  - semantic impact: none
- `batch_size: 2` for calibration collection on the A6000
  - effect: implementation-only throughput improvement
  - semantic impact: none, because chunking/order/objective are unchanged
- repo-local Hugging Face/model caches
  - effect: avoids repeated downloads
  - semantic impact: none

These are implementation shortcuts, not algorithm substitutions.

## Final Judgment

### Classification

This is best described as a **near-reproduction**, not an exact paper match.

### Why not “exact paper match”?

- The validation result is still above the paper by `+0.3610`.
- The repo does not pin a paper-specific public model/tokenizer revision.
- The exact tokenizer/BOS/chunk-count realization is not proven identical to the paper authors' environment.

### Why not merely “partial reproduction”?

- There is no obvious missing core WaterSIC component left in the `Llama-3.2-1B` best path.
- The paper-scale algorithmic path is present and end-to-end validated.
- The remaining difference is small relative to the earlier gaps and is no longer dominated by an obvious missing feature or evaluation-split mismatch.

## Gap Summary

### Test split

- repo result:
  - `10.6031`
- paper reference:
  - `10.57`
- gap:
  - `+0.0331`

Interpretation:

- This looked nearly exact numerically, but it used WikiText-2 test rather than validation.
- That made it encouraging, but not fully paper-comparable.

### Validation split

- repo result:
  - `10.9310`
- paper reference:
  - `10.57`
- gap:
  - `+0.3610`

Interpretation:

- This is the strict paper-comparable number.
- The remaining gap is now more likely due to a combination of:
  - model/tokenizer revision drift
  - exact tokenization/chunking realization differences
  - remaining residual-path distortion in the hardest projections
- It is less likely to be caused by a missing major algorithmic block, because the final run already matches the paper path on the main WaterSIC components and hyperparameters.

## Bottom Line

- The repo now closely matches the paper for `Llama-3.2-1B ~3.0-bit` WaterSIC.
- The main methodological caveat about validation versus test has been removed.
- The remaining gaps are:
  - one methodologically meaningful numeric gap on validation: `+0.3610`
  - a few minor reproducibility details around revision pinning and exact chunk realization
- The most honest description is:
  - **near-reproduction with a small but real validation gap**
