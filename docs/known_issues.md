# Known Issues

## Critical Blockers

1. The best completed full-model `Llama-3.2-1B` run at about `3.0` bits is now the rescaler-enabled validation point, but it is still far from the paper’s `10.57` PPL point. Current best completed result: `15.7029` PPL at `2.9984` effective bits, an absolute gap of `+5.1329`.
2. Full-model per-attention-block adaptive mixing search has been implemented in the general sequential pipeline, but its upgraded full-model run is not finished yet. Until that run lands, the best completed point still uses fixed `epsilon_qr` / `epsilon_aw` values.
3. The full-model runs still use only `8` calibration chunks. This remains a likely quality limiter even after the major correctness bugs were fixed.
4. The current WikiText-2 loader tokenizes the full concatenated split before chunking. It works, but it is inefficient and emits a long-sequence tokenizer warning.

## Implementation Gaps

1. The repo now has a full-model `Llama-3.2-1B` result, but not yet a paper-matching one. The remaining work is quality recovery, not basic end-to-end execution.
2. Diagonal rescalers are now validated on the full-model path and improve PPL, but they have not yet been combined with the new full-model adaptive-mixing search result.
3. The upgraded general-pipeline adaptive-mixing search is in place, but the full-model benchmark for that upgraded path is still running.
4. The full-model runs use only `8` calibration sequences as a runtime shortcut; this is smaller than a fully paper-faithful calibration budget.
5. Qwen3-8B was intentionally not run in this round.

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

## Not Yet Valid To Claim

1. It is not yet valid to claim a faithful reproduction of the paper’s final `Llama-3.2-1B` `3.00`-bit result, because the best completed run is still `+5.1329` PPL worse than the paper.
2. It is not yet valid to claim that the current best completed run is fully paper-faithful, because the new general-pipeline per-block adaptive-mixing search has not finished a full-model benchmark yet.
3. It is not yet valid to claim that the current repo reproduces the paper’s final accuracy-quality frontier; the repo now reproduces the end-to-end pipeline, not yet the final paper number.
4. It is not yet valid to claim that Qwen3-8B reproduction is done; it was intentionally not run in this round.

## Immediate Next Step

1. Keep the successful no-rescaler full-model run as the A reference point and the rescaler validation run as the new best completed B point.
2. Finish the upgraded full-model run with:
   - diagonal rescalers enabled
   - per-attention-block adaptive mixing search enabled
3. Compare A/B/C directly on PPL, bitrate, top worst layers, and per-kind distortion.
4. Increase calibration size beyond `8` chunks only after the upgraded quality-recovery path is benchmarked cleanly.
