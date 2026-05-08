# Artifacts

Generated artifacts are not part of the tracked public repository.

## Output Layout

Typical generated paths:

- `outputs/original/`: downloaded original model snapshots
- `outputs/stats/`: calibration token exports and auxiliary statistics
- `outputs/quantized/`: quantized model artifacts
- `outputs/reports/`: JSON and Markdown run reports
- `outputs/logs/`: command logs
- `outputs/checkpoints/`: checkpoint/resume state
- `outputs/hf_cache/`: Hugging Face cache

These directories are ignored because they can contain model weights, private
local paths, large tensors, logs, and credentials-derived access details.

## Report Fields

Run reports are intended to include:

- timestamp and git commit
- model/tokenizer ids and revisions
- quant and eval config paths
- dataset split and sequence length
- calibration sequence count
- target and achieved global bitwidth
- raw, entropy, Huffman, and side-information rates
- final perplexity when evaluation is run
- per-layer target and achieved rates
- device and runtime metadata in `extras`

Huffman bitrate and entropy bitrate are reported separately. They should not be
treated as interchangeable metrics.

## Aggregation

Aggregate generated report JSON files with:

```bash
python scripts/make_report.py \
  --reports-dir outputs/reports \
  --output-path outputs/reports/aggregate_report.md
```
