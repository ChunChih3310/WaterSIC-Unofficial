# Known Issues

## Critical Blockers

1. The best completed full-model `Llama-3.2-1B` run at about `3.0` bits is now the `32`-chunk rescaler-only point, but it is still above the paper’s `10.57` PPL result. Current best completed result: `11.7806` PPL at `2.9984` effective bits, an absolute gap of `+1.2106`.
2. Repaired adaptive mixing improved the old adaptive-mixing full-model point from `16.6096` to `16.2796` and cut runtime substantially, but it still does not beat the rescaler-only references. The repaired path is therefore beneficial relative to the old mixing implementation, but still harmful relative to the current best completed point.
3. Calibration clearly helped when moving from `8` to `16` and again from `16` to `32` chunks. The strongest remaining uncertainty on the best validated path is how much more of the residual `+1.2106` paper gap is still calibration-limited versus how much is now concentrated in the remaining late-layer residual-path outliers.
4. Qwen3-8B remains intentionally deferred until the `Llama-3.2-1B` quality gap is reduced further on the now-validated mainline path.
5. The loader inefficiency is partially addressed: tokenized WikiText-2 blocks are now cached in-repo. The first uncached build still emits the long-sequence tokenizer warning once, but later runs reuse the cached blocks.
6. Historical completed run artifacts do not serialize the integer Huffman symbols, so exact shortest/longest Huffman code lengths cannot be backfilled for those runs. The updated report fields are therefore shown as `unavailable` on older completed bundles unless a run is repeated.
7. Paper-scale calibration on the current A6000 setup is expensive even on the stable mainline path:
   - batch-1 rescaler-only mainline estimate: about `40.5h`
   - batch-2 rescaler-only mainline estimate: about `37.5h`
   - batch-1 repaired adaptive-mixing estimate: about `66.7h`
   - batch-2 repaired adaptive-mixing estimate: about `63.7h`
   - the adaptive-mixing paper-scale path is therefore currently too expensive for normal iterative debugging
8. Short A6000 probing now shows that the real reference-stat collection path, not the adaptive objective forward path, is the batch-size limiter:
   - `batch_size=2` fits on collection
   - `batch_size=4` and `8` OOM on collection
   - adaptive objective forward fits through `batch_size=8`, but throughput gains are minor

## Implementation Gaps

1. The repo now has a full-model `Llama-3.2-1B` result, but not yet a paper-matching one. The remaining work is quality recovery, not basic end-to-end execution.
2. Diagonal rescalers are now validated on the full-model path and improve PPL materially.
3. The original upgraded general-pipeline adaptive-mixing search is validated end-to-end, but its old local objective/search coupling did not improve full-model quality. The repaired search path now reuses the step-1 Q/K/V scales during the coordinate search and is validated on both a `2`-layer prefix and a full-model run.
4. The current best validated path has now been completed at `8`, `16`, and `32` calibration chunks.
5. Adaptive mixing remains implemented, paper-audited, and stable, but it is not yet the best quality path at full-model scope.
6. Qwen3-8B was intentionally not run in this round.
7. New runs will report exact Huffman shortest/longest symbol lengths, but older reports can only mark those fields as unavailable unless the quantization run is repeated.
8. If larger batch sizes are used later, `batch_size=2` is the current evidence-backed ceiling for the full reference-stat mainline path on this A6000 without changing experiment structure.
9. GPU auto-selection is now more conservative than the original implementation, but it still depends on `nvidia-smi` process visibility. On MIG-enabled or driver-restricted systems where per-process visibility is incomplete, the selector falls back to memory-first ranking and logs a warning instead of pretending the GPU is fully idle.

## Already Validated

1. The ZSIC recursive-update bug is fixed. The update now uses `alpha_i` instead of the global `c`.
2. The transformed-target `Y` construction bug is fixed. The triangular solve now implements `target_cross @ (L^T)^(-1)` instead of `target_cross @ L^(-1)`.
3. Calibration statistics are now collected under `torch.no_grad()`, removing the paired-model attention OOM path.
4. The staged layer-0 attention ladder `A` through `H` completed successfully on `Llama-3.2-1B`, with saved configs, logs, and results under `outputs/reports/llama32_1b_layer0_attention_debug/`.
5. A larger layer-0 validation with `reference_stats` active and rescalers still off remained sane on `8` calibration chunks and `8` eval chunks:
   - baseline small-eval PPL: `8.9880`
   - quantized small-eval PPL: `9.0138`
   - optimized `epsilon_qr = 0.00310562`
   - optimized `epsilon_aw = 0.79837388`
6. The sequential quantizer now refreshes statistics within each layer in block order:
   - `q_proj`, `k_proj`, `v_proj`
   - `o_proj`
   - `gate_proj`, `up_proj`
   - `down_proj`
7. The residual-correction formula bug is fixed. The shared code now matches the paper’s equation (18):
   - correct addend: `Sigma_Delta,Xhat`
   - removed incorrect legacy solve against `Sigma_Xhat^{-1}`
8. The narrow `layer1 self_attn.o_proj` audit is fully sane with `reference_stats: true` and rescalers off:
   - stage timing verified: same-layer, post-QKV, pre-o_proj
   - manual `Sigma_Delta,Xhat` vs official collector mismatch: `0`
   - wrong-sign mismatch: `3.8401e-3`
   - corrected-path `target_y_max_abs = 0.190439`
   - legacy-formula `target_y_max_abs = 85.947011`
9. The fixed residual path is stable on the first `11` modules of `Llama-3.2-1B` with residual correction enabled:
   - achieved effective bitwidth: `2.9919`
   - entropy bitwidth: `2.9786`
   - Huffman bitwidth: `3.0343`
   - small-eval PPL: `9.6433`
10. The first full-model `Llama-3.2-1B` run at about `3.0` bits completed successfully with real WikiText-2 evaluation:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_norescaler.yaml`
   - `reference_stats: true`
   - residual correction enabled
   - rescalers disabled
   - achieved effective bitwidth: `2.9984`
   - entropy bitwidth: `2.9865`
   - Huffman bitwidth: `3.0368`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `16.8684`
   - runtime: `19408.47s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
11. The full-model run stayed numerically sane through all `16` layers:
   - no NaN/Inf
   - no failed Cholesky
   - no catastrophic residual-path blow-up
   - worst remaining distortion is concentrated in `o_proj` / `down_proj`, not spread uniformly across all module kinds
12. Diagonal rescalers are now validated on the full-model path:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler.yaml`
   - achieved effective bitwidth: `2.9984`
   - entropy bitwidth: `2.9865`
   - Huffman bitwidth: `3.0368`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `15.7029`
   - absolute improvement vs no-rescaler baseline: `-1.1654`
   - runtime: `18525.23s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
13. The full-model reference-stats coverage audit is now explicit and unchanged across the stable runs:
   - effective count: `109 / 112`
   - non-effective matrices:
     - `model.layers.0.self_attn.q_proj`
     - `model.layers.0.self_attn.k_proj`
     - `model.layers.0.self_attn.v_proj`
   - diagnosis:
     - expected, not a bug
     - no prior quantized layer exists before the first QKV stage, so the paired reference path has zero drift there
14. The upgraded full-model adaptive-mixing run completed successfully:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml`
   - achieved effective bitwidth: `2.9984`
   - entropy bitwidth: `2.9865`
   - Huffman bitwidth: `3.0368`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `16.6096`
   - runtime: `74349.91s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
15. The adaptive-mixing search is therefore validated as:
   - implemented for all attention blocks
   - numerically stable
   - able to improve the local `wo`-input objective
   - not yet beneficial for final full-model PPL
16. The best completed `Llama-3.2-1B` point was the `8`-chunk rescaler-only run before the calibration sweep completed:
   - `15.7029` PPL at `2.9984` effective bits
   - better than the no-rescaler baseline by `1.1654`
   - better than the adaptive-mixing upgraded run by `0.9067`
17. A paper-backed adaptive-mixing repair is now implemented:
   - the old search re-ran binary search over `c` inside every `epsilon` candidate evaluation
   - the repaired search reuses the step-1 calibrated Q/K/V scales during the `epsilon_qr` and `epsilon_aw` searches
   - the final recalibration pass remains enabled
18. The repaired path passed the first larger-scope validation:
   - config: `configs/quant/watersic_llama32_1b_prefix2_reftrue_rescaler_mixing_repaircheck.yaml`
   - achieved effective bits: `2.9862`
   - entropy bits: `2.9743`
   - Huffman bits: `3.0262`
   - baseline small-eval PPL: `8.9880`
   - quantized small-eval PPL: `9.5600`
   - total runtime: `3441.95s`
19. The repaired search is materially faster on the validated prefix:
   - layer 0 adaptive-mixing search: about `10.1x` faster than the old full-model path
   - layer 1 adaptive-mixing search: about `5.7x` faster than the old full-model path
20. The repaired full-model adaptive-mixing run completed successfully:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing_repaired.yaml`
   - reports:
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired.json`
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired.md`
   - achieved effective bits: `2.9984`
   - entropy bits: `2.9865`
   - Huffman bits: `3.0368`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `16.2796`
   - runtime: `30248.17s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
21. The repaired adaptive-mixing path is now validated as:
   - paper-audited
   - numerically stable end to end
   - materially faster than the old adaptive-mixing path
   - partially quality-recovering relative to the old adaptive-mixing run
   - still not good enough to replace the rescaler-only run as the best completed point
22. A safe runtime optimization now supports the calibration sweep:
   - repo-local cached WikiText-2 token blocks
   - same split, tokenizer, concatenation, and non-overlapping `2048`-token chunking semantics
   - reduced repeated tokenization cost without changing experiment meaning
23. The `16`-chunk rescaler-only full-model run completed successfully:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib16.yaml`
   - reports:
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.json`
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.md`
   - achieved effective bits: `2.9984`
   - entropy bits: `2.9865`
   - Huffman bits: `3.0371`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `12.4574`
   - quantization runtime: `22405.30s`
   - total runtime: `22502.96s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
24. Increasing calibration from `8` to `16` chunks materially improved the dominant residual-path errors:
   - `o_proj` mean relative weight MSE: `0.3170 -> 0.2382`
   - `down_proj` mean relative weight MSE: `0.3023 -> 0.1709`
   - paper gap: `+5.1329 -> +1.8874`
25. The current best completed `Llama-3.2-1B` point is now the `32`-chunk rescaler-only run:
   - `11.7806` PPL at `2.9984` effective bits
   - better than the `16`-chunk rescaler-only point by `0.6768`
   - better than the `8`-chunk rescaler-only reference by `3.9223`
   - better than both adaptive-mixing full-model variants
26. The `32`-chunk rescaler-only full-model run completed successfully:
   - config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_calib32.yaml`
   - reports:
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.json`
     - `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.md`
   - achieved effective bits: `2.9984`
   - entropy bits: `2.9865`
   - Huffman bits: `3.0373`
   - side-information overhead: `0.0119`
   - baseline PPL: `9.7041`
   - quantized PPL: `11.7806`
   - quantization runtime: `23119.82s`
   - total runtime: `23211.72s`
   - peak memory: `18.65 GiB`
   - quantization anomalies: none
27. Increasing calibration from `16` to `32` chunks materially improved the dominant residual-path errors:
   - `o_proj` mean relative weight MSE: `0.2382 -> 0.2025`
   - `down_proj` mean relative weight MSE: `0.1709 -> 0.1292`
   - paper gap: `+1.8874 -> +1.2106`
28. Paper-scale calibration size is now measured and documented for the current tokenizer path:
   - exact chunk count: `1188`
   - chunking rule: non-overlapping `2048`-token sequences over the full WikiText-2 train split
29. Huffman shortest/longest symbol-length reporting is now implemented in the active pipeline:
   - exact for new runs
   - unavailable for older historical runs that did not serialize integer Huffman symbols
30. A short real batch-size probe has now been completed on the A6000:
   - collection path:
     - `bs=1` peak `18.65 GiB`
     - `bs=2` peak `32.69 GiB`
     - `bs=4` OOM at peak attempt `43.73 GiB`
     - `bs=8` OOM at peak attempt `41.08 GiB`
   - adaptive candidate objective path:
     - fits through `bs=8`
     - throughput improvement from `bs=1` to `bs=8` is only about `4.8%`
31. The GPU selector now uses a conservative ranking rule when `CUDA_VISIBLE_DEVICES` is unset:
   - first: prefer GPUs with `0` visible compute processes
   - second: prefer lower used memory / higher free memory
   - third: use utilization only as a tie-breaker
   - if no GPU meets the idle thresholds, the selector warns and chooses the least-bad GPU by the same rule
32. The GPU assignment path now sets `CUDA_VISIBLE_DEVICES` before any CUDA initialization:
   - this avoids the old logical/physical mismatch where the selector could choose physical GPU `N` but torch still initialize on physical GPU `0`
   - runs now log both logical torch device and physical GPU mapping explicitly

## Not Yet Valid To Claim

1. It is not yet valid to claim a faithful reproduction of the paper’s final `Llama-3.2-1B` `3.00`-bit result, because the best completed run is still `+1.2106` PPL worse than the paper.
2. It is not yet valid to claim that the current repo reproduces the paper’s final adaptive-mixing behavior, because the repaired full-model adaptive-mixing point is still worse than the best rescaler-only reference and still `+5.7096` above the paper’s `10.57` point.
3. It is not yet valid to claim that the current repo reproduces the paper’s final accuracy-quality frontier; the repo now reproduces the end-to-end pipeline, not yet the final paper number.
4. It is not yet valid to claim that Qwen3-8B reproduction is done; it was intentionally not run in this round.
5. It is not yet valid to claim that calibration is fully saturated on the stable path, because only `8`, `16`, and `32` chunks have been completed so far.
6. It is not yet valid to claim that paper-scale adaptive mixing is a practical next benchmark path on the current implementation, because the current estimate is about `66.7h` on one A6000 and no quality win over the rescaler-only reference has been demonstrated.
7. It is not yet valid to claim that simply increasing batch size will make paper-scale adaptive mixing practical, because the safe mainline collection ceiling is only `bs=2` and the lighter adaptive objective path gains little from larger batches.

## Immediate Next Step

1. Keep the `32`-chunk rescaler-only run as the current best completed reference point.
2. Do not launch a paper-scale adaptive-mixing run on the current implementation before more runtime work; it is currently too expensive relative to its demonstrated quality.
3. When long mainline runs resume, use `batch_size=2` on the validated rescaler-only path rather than `1`; this is the only safe batch-size increase currently supported by the A6000 probe.
4. Qwen3-8B remains intentionally deferred until the `Llama-3.2-1B` quality gap is reduced further.
