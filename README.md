# WaterSIC Unofficial Reproduction

This repository contains an unofficial implementation and reproduction harness for
WaterSIC, the linear-layer quantization method from:

> Egor Lifar, Semyon Savkin, Or Ordentlich, and Yury Polyanskiy.
> "WaterSIC: information-theoretically (near) optimal linear layer quantization."
> arXiv:2603.04956v1, 2026.

WaterSIC controls compressed rate by quantizing each linear-layer column with an
unequal spacing derived from the input covariance. The key implementation path in
this repository follows the transformed objective, ZSIC recursion, LMMSE
correction, entropy/Huffman accounting, and sequential model-level quantization
needed for Llama and Qwen experiments.

This is not an official repository for the paper.

## Implementation Status

Implemented:

- repo-local environment and cache setup
- path guard for repository-local writes
- idle-GPU selection that respects an existing `CUDA_VISIBLE_DEVICES`
- deterministic WikiText-2 calibration/evaluation block construction
- transformed-space ZSIC with WaterSIC column spacing
- LMMSE shrinkage and dead-feature handling
- activation-drift, residual, attention-weighted, and diagonal-rescaler terms
- staged sequential transformer-layer quantization
- partial checkpoint/resume support at stage and statistics boundaries
- JSON and Markdown report generation
- WikiText-2 perplexity evaluation
- unit tests for core numerical helpers and guardrails

Known gaps:

- the implementation is a near-reproduction, not a proven exact match to every
  paper setting
- public model revisions are not pinned to a paper-specific snapshot
- Qwen3-8B paper-scale reproduction is configured but not yet validated here
- adaptive-mixing and rate-search internals are not checkpointed mid-search
- generated model weights, logs, and reports are intentionally not tracked

See [docs/known_issues.md](docs/known_issues.md) and
[docs/results.md](docs/results.md) for the current release status.

## Repository Layout

- `configs/`: environment, model, quantization, and evaluation configs
- `configs/paper_comparable/`: paper-scale launch configs
- `scripts/`: supported command-line entrypoints
- `src/watersic/`: implementation package
- `tests/`: unit and smoke-level tests
- `docs/`: public reference documentation
- `CITATION.cff`: citation metadata for this implementation and the WaterSIC paper

Generated outputs are written under `outputs/` and ignored by git.

## Installation

Create the repo-local conda environment:

```bash
./scripts/setup_env.sh
conda activate "$PWD/.conda/watersic"
python -m pip install --no-deps -e .
```

The setup script keeps the conda environment, pip cache, and Hugging Face cache
inside this repository. It does not require editing global conda environments.

## Model Access

Some model configs use gated Hugging Face models. Keep credentials local:

```bash
cp .env.example .env
# edit .env locally and set HF_TOKEN
```

Do not commit `.env`. The runtime loads it locally and configures Hugging Face
caches under `outputs/hf_cache/`.

## Quickstart Smoke Run

The smoke config quantizes a small Llama-3.2-1B slice and evaluates a short
WikiText-2 test subset. It is intended for checking imports, model access, GPU
selection, artifact writing, and report generation.

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

Run quantization plus evaluation:

```bash
python scripts/run_experiment.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml \
  --eval-config configs/eval/wikitext2_smoke8.yaml
```

Benchmark a saved artifact:

```bash
python scripts/benchmark_model.py \
  --model-path outputs/quantized/llama32_1b_smoke_3p0bit \
  --eval-config configs/eval/wikitext2_smoke8.yaml
```

Aggregate reports:

```bash
python scripts/make_report.py \
  --reports-dir outputs/reports \
  --output-path outputs/reports/aggregate_report.md
```

## Paper-Scale Configs

Paper-comparable configs live in `configs/paper_comparable/`. For example:

```bash
python scripts/run_experiment.py \
  --model-config configs/paper_comparable/models/llama32_1b.yaml \
  --quant-config configs/paper_comparable/quant/watersic_llama32_1b_paperscale.yaml \
  --eval-config configs/eval/wikitext2_validation.yaml
```

Paper-scale runs require substantial GPU memory and runtime. The Llama-3.2-1B
paper-scale config uses 1188 calibration sequences of length 2048 and enables
reference statistics, adaptive mixing, residual correction, attention weighting,
and diagonal rescalers.

## Outputs

Typical generated paths:

- `outputs/original/`: downloaded model snapshots
- `outputs/stats/`: exported calibration token blocks
- `outputs/quantized/`: saved quantized artifacts
- `outputs/reports/`: JSON and Markdown reports
- `outputs/logs/`: command logs
- `outputs/checkpoints/`: resumable run checkpoints

These paths are ignored by git because they may contain model weights, large
metadata, local paths, or private run details.

## Configuration

Model configs specify model/tokenizer ids and architecture adapters. Quant configs
specify calibration size, target bitwidth, WaterSIC layer options, device policy,
and optional checkpointing. Eval configs specify WikiText-2 split, sequence
length, batch size, and optional subset size.

See [docs/configuration.md](docs/configuration.md) for field details.

## Documentation

- [Algorithm](docs/algorithm.md)
- [Calibration](docs/calibration.md)
- [Quantization](docs/quantization.md)
- [Evaluation](docs/evaluation.md)
- [Configuration](docs/configuration.md)
- [Artifacts](docs/artifacts.md)
- [Checkpointing](docs/checkpointing.md)
- [Results](docs/results.md)
- [Known issues](docs/known_issues.md)
- [Troubleshooting](docs/troubleshooting.md)

## Tests

Run the test suite from the repo-local environment:

```bash
.conda/watersic/bin/python -m pytest
```

The ambient system Python on this machine is not the release target; use the
repo-local Python 3.11 environment created by `scripts/setup_env.sh`.

## License

This repository's original code and original repository materials are released
under the MIT License; see [LICENSE](LICENSE). This is an unofficial
reproduction of the WaterSIC paper and is not affiliated with or endorsed by the
paper authors. The WaterSIC paper and any third-party materials should be cited
and licensed separately.

## Citation

Use the paper citation above for the WaterSIC method. You may cite this
unofficial software repository separately using [CITATION.cff](CITATION.cff).
