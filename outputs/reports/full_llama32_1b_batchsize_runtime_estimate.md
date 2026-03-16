# Full-Model Llama-3.2-1B Batch-Size Runtime Estimate

Date: 2026-03-16

## Scope

This report estimates how much the current paper-scale `Llama-3.2-1B` runtime on the local A6000 could improve from safely increasing batch size, without changing experiment semantics.

Paper-scale calibration here means:

- full WikiText-2 train split
- current tokenizer/chunking path
- non-overlapping `2048`-token blocks
- current measured chunk count: `1188`

Validated mainline path:

- `reference_stats: true`
- fixed residual correction
- staged same-layer stat refresh
- diagonal rescalers enabled
- adaptive mixing disabled

Completed batch-1 paper-scale estimates from the previous report:

- rescaler-only mainline: about `40.5h`
- repaired adaptive mixing: about `66.7h`

## Why Batching Is Semantically Safe Here

The existing implementation already supports batched execution without changing the algorithm:

1. Calibration blocks and their order do not change.
   - `TokenBlockDataset.batches(batch_size)` only groups adjacent precomputed `2048`-token blocks.
   - No tokenization, reordering, or chunk-boundary changes are introduced.

2. Second-moment accumulation is batch-equivalent.
   - `SecondMomentAccumulator.update()` reshapes `[B, T, D]` to `[B*T, D]` and normalizes by the total row count.
   - Grouping the same chunks into larger batches therefore preserves the same token set and weighting.

3. Adaptive-mixing objective evaluation is also batching-equivalent.
   - `_collect_module_inputs()` and `_module_input_relative_mse()` iterate over the same calibration batches, only with more than one chunk per forward.
   - The paper requires forward passes on the calibration set; batching changes implementation cost, not the scored objective.

The remaining caveat is ordinary floating-point associativity in the model forward pass. The statistics accumulation itself remains stable because accumulation is done in `float64`.

## Probe Method

No new long quantization run was launched. A short real probe was run on the A6000 using the actual code paths:

- collection probe:
  - function: `collect_layer_statistics()`
  - stage: layer `0` QKV (`q_proj`, `k_proj`, `v_proj`)
  - models resident on GPU: quantized-path model + reference model
  - attention outputs enabled, which makes this the heaviest collection stage family
  - calibration slice: first `8` train chunks

- adaptive objective probe:
  - functions: `_collect_module_inputs()` and `_module_input_relative_mse()`
  - target module: `model.layers.15.self_attn.o_proj`
  - calibration slice: same `8` train chunks
  - this measures the repeated candidate forward path used during adaptive-mixing search

- eval probe:
  - same model
  - first `8` WikiText-2 test chunks
  - used only to estimate the already-small evaluation component

Raw probe output is saved to:

- `outputs/reports/full_llama32_1b_batchsize_runtime_estimate_probe.json`

## Probe Results

| Batch | Collection Fits | Collection Peak GB | Collection Sec/Chunk | Adaptive Objective Fits | Adaptive Peak GB | Adaptive Objective Sec/Chunk | Eval Fits | Eval Sec/Chunk |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| `1` | yes | `18.65` | `2.3563` | yes | `5.92` | `0.2142` | yes | `0.1910` |
| `2` | yes | `32.69` | `2.1589` | yes | `7.23` | `0.2130` | yes | `0.1900` |
| `4` | no | `43.73` | `OOM` | yes | `9.85` | `0.2048` | yes | `0.1851` |
| `8` | no | `41.08` | `OOM` | yes | `15.08` | `0.2045` | yes | `0.1857` |

Key observations:

1. The real limiter is calibration/reference-stat collection, not adaptive candidate forward evaluation.
   - `batch_size=2` fits.
   - `batch_size=4` and `8` both OOM on the real collection path.

2. The adaptive objective forward path is much lighter and fits up to `8`.
   - But throughput barely improves:
     - `bs=1`: `0.2142s/chunk`
     - `bs=8`: `0.2045s/chunk`
   - That is only about a `4.8%` speedup.

3. Evaluation is negligible either way.
   - `bs=1` to `bs=8` only changes eval throughput by a few percent.
   - This does not materially change the full-run wall clock.

## Speedup Factors Used For The Estimate

Relative to `batch_size=1` on the probe:

- collection speedup:
  - `bs=2`: `1.091x`
- adaptive objective speedup:
  - `bs=2`: `1.006x`
  - `bs=4`: `1.046x`
  - `bs=8`: `1.048x`
- eval speedup:
  - `bs=2`: `1.005x`
  - `bs=4`: `1.032x`
  - `bs=8`: `1.029x`

## Updated Paper-Scale Runtime Estimates

These estimates reuse the previously documented batch-1 runtime decomposition and only rescale the components that are actually batch-sensitive:

- mainline collection component
- adaptive objective-forward component
- evaluation component

The fixed quantization/search component is left unchanged.

### Single Global Batch Size

| Global Batch Size | Rescaler-Only Mainline | Repaired Adaptive Mixing | Feasible As Full Run? |
| --- | ---: | ---: | --- |
| `1` | `40.45h` | `66.71h` | yes |
| `2` | `37.53h` | `63.67h` | yes |
| `4` | `n/a` | `n/a` | no, collection OOM |
| `8` | `n/a` | `n/a` | no, collection OOM |

Interpretation:

- Moving from `1 -> 2` saves about:
  - `2.92h` on the rescaler-only mainline path
  - `3.05h` on the repaired adaptive-mixing path
- The savings are real but modest because most of the fixed quantization/search cost does not depend on batch size.

### If Collection And Adaptive Objective Used Different Batch Sizes

This is not the current config path, but it is useful as an upper bound because the objective-forward path fits larger batches than collection.

Assuming:

- collection/reference stats stay at `batch_size=2`
- adaptive candidate objective uses a larger batch size

Then the repaired adaptive-mixing paper-scale estimate would be:

| Collection Batch | Adaptive Objective Batch | Estimated Adaptive Runtime |
| --- | --- | ---: |
| `2` | `2` | `63.67h` |
| `2` | `4` | `62.78h` |
| `2` | `8` | `62.74h` |

So even with a separate larger batch size for the candidate forward path, the total gain is less than `1h`.

## Conclusions

1. Recommended safe batch size for the current A6000 mainline path: `2`.
   - It fits on the real collection path with both models resident.
   - `4` and `8` do not.

2. Recommended batch size for adaptive-mixing evaluation: also `2` in practice.
   - `4` and `8` fit the objective-forward path, but throughput almost flattens.
   - The best-case total adaptive runtime gain from splitting collection/objective batches is too small to justify extra implementation complexity right now.

3. Updated paper-scale runtime with the recommended batch size:
   - rescaler-only mainline at `bs=2`: about `37.5h`
   - repaired adaptive mixing at `bs=2`: about `63.7h`

4. Practical judgment:
   - `bs=2` is the only safe, evidence-backed batch-size increase worth using on this A6000 for the current full reference-stats path
   - adaptive mixing remains too expensive for routine paper-scale iteration even after the safe batch-size increase

## Bottom Line

- Recommended safe batch size on the current A6000: `2`
- Paper-scale rescaler-only runtime at that batch size: about `37.5h`
- Paper-scale repaired adaptive-mixing runtime at that batch size: about `63.7h`
- Adaptive mixing is still too expensive for normal iterative use
