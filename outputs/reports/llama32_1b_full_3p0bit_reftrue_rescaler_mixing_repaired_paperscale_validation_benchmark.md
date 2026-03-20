# Llama-3.2-1B Paper-Scale Validation Benchmark

- Artifact: `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- Eval config: `configs/eval/wikitext2_validation.yaml`
- Split: `validation`
- Sequence length: `2048`
- Batch size: `1`
- Log: `outputs/logs/benchmark_model_20260320_202732.log`

## Result

- Validation-split PPL: `10.9310`
- Previous test-split PPL from the same completed artifact: `10.6031`
- Paper reference at `3.00` bits: `10.57`
- Validation gap vs paper: `+0.3610`

This benchmark reused the completed paper-scale quantized artifact and reran only evaluation. It did not rerun quantization.
