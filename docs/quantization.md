# Quantization

The public quantization entrypoint is:

```bash
python scripts/run_experiment.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml \
  --eval-config configs/eval/wikitext2_smoke8.yaml
```

`scripts/quantize_model.py` accepts the same arguments and runs the same
quantize-plus-evaluate pipeline. The evaluation config is required because the
run report records final perplexity.

## Per-Layer Flow

1. Collect second-moment statistics for the current stage.
2. Detect and remove dead input dimensions.
3. Form the transformed target with enabled drift/residual terms.
4. Cholesky-factorize the damped covariance.
5. Search for the WaterSIC spacing parameter `c`.
6. Quantize with transformed-space ZSIC.
7. Apply LMMSE correction and optional diagonal rescalers.
8. Reinsert dead features as zero columns.
9. Record entropy, Huffman, raw-rate, and side-information metrics.

## Model-Level Flow

The model is quantized in transformer order. Within each layer, the durable stage
order is:

1. `q_proj`, `k_proj`, `v_proj`
2. `o_proj`
3. `gate_proj`, `up_proj`
4. `down_proj`

This stage structure matters because several modules share statistics collected
from the same pre-stage model state.

## Outputs

Completed runs save:

- reconstructed model artifact under `outputs/quantized/<run_name>/`
- artifact metadata at `metadata.json`
- per-run JSON and Markdown reports under `outputs/reports/`
- command logs under `outputs/logs/`

All generated artifacts are ignored by git.
