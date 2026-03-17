# Full-Model Llama-3.2-1B Paper-Scale Launch Preparation

Date: 2026-03-17

## Target Run

- run: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- model: `meta-llama/Llama-3.2-1B`
- target rate: `3.0` bits
- calibration budget: `1188` non-overlapping WikiText-2 train chunks at sequence length `2048`
- benchmark: WikiText-2 test perplexity at sequence length `2048`

## Resume / Reuse Audit

True resume is **not** currently supported for this run.

What can be safely reused:

- repo-local token-block cache:
  - `outputs/stats/wikitext2_cache/train_seq2048_895aed66e8af833c.pt`
  - exact cached shape: `(1188, 2048)`
- repo-local test token-block cache:
  - `outputs/stats/wikitext2_cache/test_seq2048_44625d9fc9c502ec.pt`
- repo-local HF/model cache under `outputs/hf_cache/`

What cannot be safely reused:

- no per-layer checkpoint restart path
- no saved intermediate calibration statistics for this paper-scale run
- no reusable partial quantization artifact for this paper-scale run
- no safe layerwise resume path from previous smaller-budget runs

Current save behavior remains end-of-run:

- quantized artifact and `layer_results.json` are written only after sequential quantization completes

## Config Verification

Paper-scale config:

- `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired_paperscale.yaml`

Verified settings:

- `reference_stats: true`
- `reference_device: cuda`
- fixed residual correction enabled
- staged same-layer stat refresh enabled in the validated sequential pipeline
- diagonal rescalers enabled: `max_rescaler_iters: 4`
- attention weighting enabled
- repaired adaptive mixing enabled:
  - `use_adaptive_mixing: true`
  - `optimize_adaptive_mixing: true`
  - `epsilon_qr: 0.0`
  - `epsilon_aw: 0.0`
- no stale shared-`c` QKV interpretation
- sequence length `2048`
- calibration chunks `1188`
- safe validated batch size `2`
- idle-only GPU policy:
  - `min_free_memory_gib: 24.0`
  - `max_used_memory_gib: 2.0`
  - `allow_busy_fallback: false`

## Current GPU Snapshot

At launch preparation time, `CUDA_VISIBLE_DEVICES` was unset and every visible A6000 was busy:

| Rank | GPU | Processes | Used MiB | Free MiB | Util |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `3` | `2` | `6293` | `42378` | `100%` |
| `2` | `5` | `2` | `7313` | `41358` | `96%` |
| `3` | `6` | `2` | `18385` | `30286` | `32%` |
| `4` | `7` | `2` | `26347` | `22324` | `100%` |
| `5` | `4` | `2` | `27239` | `21432` | `100%` |
| `6` | `2` | `2` | `31337` | `17334` | `100%` |
| `7` | `1` | `2` | `38319` | `10352` | `60%` |
| `8` | `0` | `4` | `39188` | `9483` | `43%` |

Idle-class decision:

- no GPU met the idle thresholds
- the selector correctly refused to auto-pick a busy GPU

To avoid a bad launch, the run was queued behind a wait-for-idle wrapper:

- launcher script:
  - `scripts/launch_when_idle.py`
- wrapper log:
  - `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_155306.log`
- shell capture:
  - `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.out`

## Runtime Estimate For This Exact Run

Assumptions:

1. The run starts on the first truly idle A6000 satisfying the current thresholds.
2. `batch_size=2` is used for calibration/reference-stat collection.
3. The repo-local token cache is warm, so no retokenization cost is paid.
4. Search budgets remain unchanged:
   - binary search iterations: `30`
   - golden-section iterations: `15`
5. No resume is available; all quantization work is recomputed from scratch.

Estimated time breakdown:

- quantization/statistics/search:
  - about `63.65h`
- evaluation:
  - about `0.02h` (`~59s` for baseline + quantized WikiText-2 evaluation)
- total wall time:
  - about `63.67h`

Practical interpretation:

- this is about a `2.65` day one-off run on one idle A6000
- the token cache helps startup cost slightly, but it does **not** materially shorten the core quantization/search wall time

## Launch Status

The actual paper-scale quantization run has **not** started yet, because no qualifying idle GPU existed at launch time.

What is running now:

- a background launcher that re-checks the idle-GPU policy every `60s`
- once a truly idle GPU appears, it will start the real experiment with the paper-scale config above
