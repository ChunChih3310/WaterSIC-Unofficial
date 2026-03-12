# Benchmarking

## Required Benchmark

- Dataset: WikiText-2 test split
- Context length: 2048
- Metric: perplexity

## Current Implementation

- `src/watersic/eval/perplexity.py` evaluates causal-LM perplexity on non-overlapping token blocks.
- `scripts/benchmark_model.py` benchmarks a saved artifact or repo-local model snapshot.
- `scripts/run_experiment.py` runs the baseline and quantized evaluations under the same evaluation config.

## Notes

- The current evaluator uses non-overlapping blocks and the model’s built-in label loss.
- Batch size is config-controlled.
- Runs should be compared only when the model config, tokenizer config, and eval config match.
