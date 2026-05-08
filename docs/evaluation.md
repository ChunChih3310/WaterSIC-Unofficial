# Evaluation

The supported evaluation path is WikiText-2 perplexity for causal language
models.

Configs live under `configs/eval/`:

- `wikitext2_smoke8.yaml`: short smoke evaluation
- `wikitext2.yaml`: full test split evaluation
- `wikitext2_validation.yaml`: validation split evaluation

Benchmark a saved artifact with:

```bash
python scripts/benchmark_model.py \
  --model-path outputs/quantized/llama32_1b_smoke_3p0bit \
  --eval-config configs/eval/wikitext2_smoke8.yaml
```

The evaluator uses non-overlapping token blocks and the model's built-in
causal-LM label loss. Compare runs only when model, tokenizer, dataset split,
sequence length, and evaluation block count match.
