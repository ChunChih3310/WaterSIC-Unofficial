# Calibration

Calibration data is built from WikiText-2 by concatenating text, tokenizing with
the configured model tokenizer, and splitting the stream into non-overlapping
blocks.

The default public fields are:

- dataset: `wikitext`
- subset: `wikitext-2-raw-v1`
- calibration split: `train`
- sequence length: `2048`
- number of sequences: controlled by the quant config
- batch size: controlled by the quant config

Export calibration tokens with:

```bash
python scripts/collect_calibration.py \
  --model-config configs/models/llama32_1b.yaml \
  --quant-config configs/quant/watersic_llama32_1b_smoke.yaml
```

The command writes a tensor payload under `outputs/stats/`. The payload contains
token ids, split name, sequence length, and sequence count. It is generated data
and is not tracked by git.

During quantization, later-layer statistics are gathered from the partially
quantized model after earlier layers have been replaced. When `reference_stats`
is enabled, the pipeline also loads an unquantized reference model to form paired
original/current statistics.
