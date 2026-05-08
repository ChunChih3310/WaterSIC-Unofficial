# Troubleshooting

## Hugging Face Access

If model download fails for a gated model, create a local `.env` from
`.env.example` and set `HF_TOKEN`. Do not print or commit the token.

## GPU Selection

If no idle GPU is found, the runtime fails clearly by default. Options:

- set `CUDA_VISIBLE_DEVICES` before launching
- set `device.override: cpu` for CPU-only smoke checks
- set `device.allow_busy_fallback: true` only when using a busy GPU is intended

## Import Errors

Use the repo-local environment:

```bash
./scripts/setup_env.sh
conda activate "$PWD/.conda/watersic"
python -m pip install --no-deps -e .
```

The scripts also insert `src/` into `sys.path`, but editable install is the
recommended public workflow.

## Slow Runs

Paper-scale configs are expensive. Start with:

```bash
python scripts/run_experiment.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml \
  --eval-config configs/eval/wikitext2_smoke8.yaml
```

Then move to paper-comparable configs only after the smoke path works.

## Reports Are Missing

Reports are generated under `outputs/reports/`, which is ignored by git. If the
directory is empty, run a smoke experiment first.
