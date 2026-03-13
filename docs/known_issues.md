# Known Issues

## Critical Blockers

1. The first full-model `Llama-3.2-1B` run at about `3.0` bits is now complete and numerically sane, but it is still far from the paper’s `10.57` PPL point. Our first paper-comparable result is `16.8684` PPL at `2.9984` effective bits, an absolute gap of `+6.2984`.
2. Diagonal rescalers remain disabled by default and unvalidated on the successful full-model path. This is now the largest known missing paper-faithful component in the best completed run.
3. The general sequential pipeline still uses fixed `epsilon_qr` / `epsilon_aw` values from config instead of per-attention-block golden-section search during the full run.
4. The current WikiText-2 loader tokenizes the full concatenated split before chunking. It works, but it is inefficient and emits a long-sequence tokenizer warning.

## Implementation Gaps

1. The repo now has a full-model `Llama-3.2-1B` result, but not yet a paper-matching one. The remaining work is quality recovery, not basic end-to-end execution.
2. Diagonal rescaler optimization is implemented but still disabled for the successful full-model run, so the main reported point is not yet using the paper’s full final stage.
3. Adaptive mixing is optimized in the dedicated attention debugger, but the general sequential pipeline still consumes fixed `epsilon_qr` / `epsilon_aw` values from config.
4. The full-model run used only `8` calibration sequences as a runtime shortcut; this is smaller than a fully paper-faithful calibration budget.
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

## Not Yet Valid To Claim

1. It is not yet valid to claim a faithful reproduction of the paper’s final `Llama-3.2-1B` `3.00`-bit result, because our first full-model point is still `+6.2984` PPL worse than the paper.
2. It is not yet valid to claim that the current best run is fully paper-faithful, because diagonal rescalers were disabled and full per-block adaptive mixing search was not active in the general sequential pipeline.
3. It is not yet valid to claim that the current repo reproduces the paper’s final accuracy-quality frontier; the repo now reproduces the end-to-end pipeline, not yet the final paper number.
4. It is not yet valid to claim that Qwen3-8B reproduction is done; it was intentionally not run in this round.

## Immediate Next Step

1. Keep the successful no-rescaler full-model run as the new reference point.
2. Validate diagonal rescalers on top of this now-stable full-model path, not on the earlier broken path.
3. Promote adaptive mixing from fixed config values to proper per-attention-block search inside the general sequential pipeline.
4. Increase calibration size beyond `8` chunks once the quality-recovery path is stable, then rerun the full-model benchmark and compare again to the paper.
