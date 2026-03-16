# WaterSIC Reproduction

This repository is a serious start on a runnable reproduction of the WaterSIC paper, `"WaterSIC: information-theoretically (near) optimal linear layer quantization"` (`paper/2603.04956v1.pdf`).

The current codebase implements:

- repo-local conda environment setup
- repo-local path safety guard
- conservative idle-GPU selection when `CUDA_VISIBLE_DEVICES` is unset
- deterministic WikiText-2 calibration/evaluation chunking
- core WaterSIC math primitives:
  - transformed-space ZSIC
  - unequal per-column spacing `alpha_i = c / L_ii`
  - LMMSE shrinkage
  - dead-feature erasure
  - layer-level rate search by binary search over `c`
  - entropy and canonical Huffman bitrate reporting
  - transformed objective for activation-drift and residual correction
  - diagonal row/column rescaler optimization in the transformed objective
- sequential model-level quantization over transformer layers
- saved quantized artifacts, JSON reports, and Markdown summaries
- WikiText-2 perplexity benchmarking
- unit tests for the main numerical helpers

The first implementation focus is `Llama-3.2-1B`, followed by `Qwen3-8B`.

## What Is Implemented vs Missing

Implemented now:

- plain WaterSIC foundation from the paper
- transformed objective with `Y = ...` and Cholesky factorization
- layerwise sequential quantization
- quantized-model activation statistics
- residual correction term for `o_proj` and `down_proj`
- bitrate accounting for raw, entropy, Huffman, and side-information overhead
- reproducible configs and runnable scripts

Partially implemented / still being tightened:

- adaptive mixing search for `epsilon_qr` and `epsilon_aw`
- exact paper-style calibration pairing between full-precision and partially-quantized models in long runs
- broader overnight reproductions beyond the first Llama smoke/full runs
- final paper-vs-ours comparison once overnight experiments complete

See [docs/known_issues.md](/nfs_tmp/Compression_team/src/WaterSIC/docs/known_issues.md) for the honest status.

## Environment Setup

Create the repo-local conda environment:

```bash
./scripts/setup_env.sh
conda activate /nfs_tmp/Compression_team/src/WaterSIC/.conda/watersic
```

The script keeps the environment and Hugging Face caches inside this repository.

## Device Selection

Runtime device selection follows a conservative policy:

- if `CUDA_VISIBLE_DEVICES` is already set, the code respects it exactly
- otherwise, the selector ranks GPUs by:
  - zero active compute processes first
  - then lower used memory / higher free memory
  - then lower instantaneous GPU utilization
- a GPU is only classified as truly idle if it has:
  - `0` visible compute processes
  - used memory below `device.max_used_memory_gib`
  - free memory above `device.min_free_memory_gib`
- if no GPU meets those thresholds, the default behavior is to fail clearly and stop
- only an explicit `device.allow_busy_fallback: true` override allows choosing the least-bad busy GPU
- auto-selection happens before any CUDA initialization so the later torch device mapping is safe
- after selection, torch uses logical `cuda:0`, which maps to the chosen physical GPU via `CUDA_VISIBLE_DEVICES`

The default thresholds live in [watersic_default.yaml](/nfs_tmp/Compression_team/src/WaterSIC/configs/quant/watersic_default.yaml):

```yaml
device:
  override:
  min_free_memory_gib: 24.0
  max_used_memory_gib: 2.0
  allow_busy_fallback: false
```

## Main Commands

Download a model snapshot:

```bash
python scripts/download_model.py \
  --model-config configs/models/llama32_1b.yaml
```

Export deterministic calibration tokens:

```bash
python scripts/collect_calibration.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml
```

Run quantization plus WikiText-2 evaluation:

```bash
python scripts/run_experiment.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml \
  --eval-config configs/eval/wikitext2.yaml
```

Benchmark a saved artifact:

```bash
python scripts/benchmark_model.py \
  --model-path outputs/quantized/llama32_1b_smoke_3p0bit \
  --eval-config configs/eval/wikitext2.yaml
```

Aggregate report JSON files:

```bash
python scripts/make_report.py
```

## Outputs

- `outputs/logs/`: run logs
- `outputs/stats/`: exported calibration tokens and auxiliary stats
- `outputs/quantized/`: saved quantized artifacts
- `outputs/eval/`: evaluation outputs when emitted separately
- `outputs/reports/`: JSON and Markdown run reports

## Repository Layout

- `configs/`: environment, model, quantization, and evaluation configs
- `src/watersic/`: implementation
- `scripts/`: runnable entrypoints
- `tests/`: numerical and guardrail tests
- `docs/`: implementation notes, pipeline docs, and reproduction log

## Troubleshooting

- If GPU selection is wrong or you want to pin a device manually, set `CUDA_VISIBLE_DEVICES` before launching a script.
- If all GPUs are partially occupied, the default selector now fails instead of silently taking a busy GPU.
- If you intentionally want the least-bad busy GPU, set `device.allow_busy_fallback: true` in the quant config and inspect the logged ranking carefully.
- If a model is gated on Hugging Face, ensure the token in `.env` is valid and has access.
- If `output_attentions=True` causes a model-specific issue, reduce to the smoke config first and inspect the saved log in `outputs/logs/`.
- If a quantization run fails on a covariance or rescaler step, inspect [docs/known_issues.md](/nfs_tmp/Compression_team/src/WaterSIC/docs/known_issues.md) and the run log before changing defaults.
