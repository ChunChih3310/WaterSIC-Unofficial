# Implementation Plan

## Goal

Reproduce WaterSIC inside this repository with a runnable pipeline for:

- calibration data collection
- WaterSIC quantization
- saved quantized artifacts
- WikiText-2 benchmarking
- bitrate reporting
- paper-vs-ours reporting

## Current Phase Status

Completed:

- repo-local git initialization
- repo-local conda environment spec and setup script
- path safety guard
- cache/environment utilities
- GPU auto-selection utility
- core WaterSIC math modules
- layer quantizer and sequential model quantizer skeleton
- evaluation pipeline
- scripts/configs scaffolding
- unit tests for the core helpers

In progress:

- first real Llama-3.2-1B smoke run
- tightening of model-level calibration/runtime behavior from real failures
- overnight experiment reports

Next:

1. Finish the first real Llama-3.2-1B quantize+benchmark run.
2. Expand from smoke to broader Llama-3.2-1B coverage.
3. Start Qwen3-8B once the Llama path is stable.
4. Write the final paper comparison report from completed runs.
