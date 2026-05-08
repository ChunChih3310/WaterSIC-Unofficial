# Results

Generated artifacts and full reports are not tracked in this public tree. The
numbers below summarize the best local run recorded before artifact cleanup.

## Llama-3.2-1B

Configuration:

- model: `meta-llama/Llama-3.2-1B`
- target rate: `3.0` bits
- calibration split: WikiText-2 train
- calibration sequences: `1188`
- sequence length: `2048`
- evaluation split: WikiText-2 validation
- enabled terms: reference stats, activation drift, residual correction,
  attention weighting, adaptive mixing, diagonal rescalers, dead-feature erasure

Recorded local result:

- achieved effective bits: `2.9984`
- WikiText-2 test perplexity: `10.6031`
- WikiText-2 validation perplexity: `10.9310`
- paper reference at 3.00 bits: `10.57`

Interpretation:

- This is a near-reproduction for the Llama-3.2-1B setting.
- It is not claimed to be an exact paper match.
- The remaining validation gap is about `+0.3610` perplexity.
- Exact model/tokenizer revision equivalence with the paper is not proven.

## Qwen3-8B

Paper-scale Qwen3-8B configs are present, but a validated paper-scale result is
not included in this repository. Treat those configs as launch points rather
than completed evidence.
