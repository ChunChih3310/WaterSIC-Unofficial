# Full-Model Llama-3.2-1B Paper-Scale Runtime Estimate

Date: 2026-03-16

## Scope

This estimate covers the current validated `Llama-3.2-1B` `~3.0`-bit path on the local A6000 setup, using the paper-scale calibration definition:

- dataset: full WikiText-2 train split
- tokenization: current repo tokenizer path for `meta-llama/Llama-3.2-1B`
- chunking: concatenate into one token stream, prepend one BOS token, split into non-overlapping `2048`-token chunks, discard remainder
- measured current chunk count: `1188`
- paper reference: `≈1189`

No new long run was launched for this estimate. It is extrapolated from completed runs plus log-level timing breakdowns.

## Main Assumptions

1. Calibration and evaluation still run at `batch_size=1`.
   - This is what the completed `8` / `16` / `32` chunk runs used.
   - No safely validated larger batch size exists yet for the full reference-stats path on this A6000 setup.
2. The repo-local WikiText-2 block cache is warm.
   - This removes repeated tokenization cost.
   - It does not change tokenization or chunking semantics.
3. For the rescaler-only mainline path, the non-collection quantization/search cost is treated as approximately calibration-size-independent, because the completed `16` and `32` chunk runs stayed in the same band.
4. For adaptive mixing, candidate forward-pass cost is treated as linear in the number of calibration chunks, while the extra candidate-quantization/search overhead is treated as approximately fixed at the current search budget.

## Timing Evidence From Completed Runs

| Run | Calibration Chunks | Quant Window | Statistics Collection | Non-Collection Quant/Search | Eval Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rescaler-only 8` | `8` | `18444s` | `766s` | `17678s` | `59s` |
| `rescaler-only 16` | `16` | `22431s` | `1676s` | `20755s` | `61s` |
| `rescaler-only 32` | `32` | `23146s` | `3402s` | `19744s` | `59s` |
| `repaired adaptive mixing 8` | `8` | `30169s` | `772s` | `29397s` | `58s` |

Adaptive-mixing search audit from the completed repaired `8`-chunk run:

- attention blocks searched: `16`
- objective evaluations: `576`
- objective forward time on `8` chunks: `561.47s`
- objective quantization time: `8209.01s`

## A. Paper-Scale Estimate: Rescaler-Only Mainline Path

Validated path:

- `reference_stats: true`
- fixed residual correction
- staged same-layer stat refresh
- diagonal rescalers enabled
- adaptive mixing disabled

Observed scaling terms:

- collection slope from completed `16` and `32` chunk runs:
  - `1676 / 16 = 104.75s/chunk`
  - `3402 / 32 = 106.31s/chunk`
  - central estimate: `105.53s/chunk`
- non-collection quant/search band:
  - `19744s` to `20755s`
  - central estimate: `20249.5s`
- evaluation:
  - baseline eval: about `31s`
  - quantized eval: about `28s`

Estimated paper-scale runtime at `1188` chunks:

- calibration/statistics collection:
  - `1188 * 104.75s = 124443s = 34.57h` using the `16`-chunk slope
  - `1188 * 106.31s = 126299s = 35.08h` using the `32`-chunk slope
  - central estimate: `125371s = 34.83h`
- non-collection quant/search: about `20250s = 5.62h`
- baseline evaluation: about `31s`
- quantized evaluation: about `28s`

Estimated total:

- lower-end estimate: `40.35h`
- upper-end estimate: `40.58h`
- practical central estimate: `40.47h`

Practical answer:

- a paper-scale rescaler-only mainline run is roughly a `40.5` hour job on the current A6000 setup
- this is expensive, but still operationally feasible as a single long run

## B. Paper-Scale Estimate: Adaptive-Mixing Full-Model Path

Path definition:

- same validated sequential path as above
- rescalers enabled
- adaptive mixing enabled with the repaired search path
- current search budget preserved

Observed repaired-path overhead relative to the `8`-chunk rescaler-only reference:

- extra non-collection quant/search overhead at `8` chunks:
  - `29397s - 17678s = 11719s = 3.26h`
- of that extra overhead, the adaptive-mixing audit attributes:
  - `561.47s` to candidate forward passes on the calibration set
  - the remaining `11157.53s = 3.10h` to candidate quantization/search/Python overhead

Scaling the candidate forward cost from `8` chunks to `1188` chunks:

- multiplier: `1188 / 8 = 148.5`
- projected objective-forward time:
  - `561.47s * 148.5 = 83378s = 23.16h`

Estimated paper-scale runtime:

- start from the rescaler-only central estimate: `40.47h`
- add fixed adaptive-mixing overhead: `3.10h`
- add projected paper-scale candidate-forward time: `23.16h`

Practical central estimate:

- `40.47h + 3.10h + 23.16h = 66.73h`

Practical answer:

- a paper-scale repaired adaptive-mixing run is roughly a `66.7` hour job on the current A6000 setup
- that is about `2.78` days of wall time at `batch_size=1`

## Feasibility Judgment

1. Rescaler-only paper-scale calibration looks expensive but feasible.
   - Expected wall time: about `40.5h`
   - This is long-run territory, but still reasonable for a deliberate mainline benchmark.

2. Paper-scale adaptive mixing is currently too expensive for routine iteration.
   - Expected wall time: about `66.7h`
   - This is not impossible, but it is effectively impractical for interactive debugging on a single A6000 with the current implementation.

3. The main reason adaptive mixing blows up at paper-scale is not the base WaterSIC quantization itself.
   - The dominant extra term is repeated forward evaluation of QKV candidates over the calibration set.
   - That term scales directly with the number of calibration chunks.

## Safe Runtime Notes

- The existing repo-local WikiText-2 token-block cache remains a safe runtime optimization:
  - it reuses the exact tokenized `2048`-token blocks
  - it does not change the data, tokenizer, or chunking semantics
- No additional runtime shortcut was introduced for this estimate.
- No new long experiment was launched.

## Bottom Line

- Paper-scale rescaler-only mainline path: about `40.5h`
- Paper-scale repaired adaptive-mixing path: about `66.7h`
- Current judgment:
  - rescaler-only is feasible for a deliberate benchmark run
  - adaptive mixing at paper-scale is currently too expensive for normal iteration without further runtime work
