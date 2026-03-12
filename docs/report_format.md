# Report Format

## JSON

Each run JSON report is intended to contain:

- timestamp
- git commit hash
- environment name
- device
- model/tokenizer identifiers and revisions
- config paths
- calibration size
- target and achieved global bitwidth
- raw / entropy / Huffman / side-information rates
- perplexity
- per-layer details

## Markdown

The Markdown run report summarizes:

- run metadata
- global bitrate metrics
- final perplexity
- per-layer target/achieved rates
- weighted reconstruction error

## Aggregate Report

`scripts/make_report.py` scans `outputs/reports/*.json` and emits a compact Markdown summary table.
