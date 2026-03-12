# Reproduction Log

## 2026-03-13

- Initialized a repo-local git repository inside `WaterSIC`.
- Created the repo-local conda spec at `configs/env/watersic_conda.yml`.
- Built the repo-local conda environment under `.conda/watersic`.
- Implemented:
  - path guard
  - repo-local cache/env utilities
  - GPU selection
  - Huffman bitrate accounting
  - second-moment accumulation
  - dead-feature erasure
  - transformed-space ZSIC with LMMSE correction
  - binary search over `c`
  - transformed-objective row/column rescalers
  - sequential model quantizer
  - WikiText-2 loader and perplexity evaluator
  - runnable scripts/configs
- Added unit tests for the numerical helpers and path guard.
- Extracted key formulas and reference tables from the local paper PDF.
- Started the first real `Llama-3.2-1B` smoke run using:
  - `configs/models/llama32_1b.yaml`
  - `configs/quant/watersic_llama32_1b_smoke.yaml`
  - `configs/eval/wikitext2.yaml`

This file will be updated again after the smoke run and any broader overnight runs finish.
