# Known Issues

## Critical Blockers

1. A full-model paper-faithful `Llama-3.2-1B` run at about `3.0` bits is still not completed after the residual-correction fix, so the repo does not yet have a paper-comparable end-to-end post-fix result.
2. Residual correction is now fixed and validated for the exact previous blocker (`model.layers.1.self_attn.o_proj`) and for the corresponding `11`-module smoke, but broader validation is still needed for later residual targets, especially deeper `down_proj` / `o_proj` layers in a longer sequential run.
3. Diagonal rescalers remain disabled by default. They are still not validated beyond the dedicated layer-0 debug ladder.
4. The current WikiText-2 loader tokenizes the full concatenated split before chunking. It works, but it is inefficient and emits a long-sequence tokenizer warning.

## Implementation Gaps

1. A full-model `Llama-3.2-1B` run at about `3.0` bits with `reference_stats: true`, fixed residual correction, and rescalers still off has not yet been executed.
2. Adaptive mixing is optimized in the dedicated attention debugger, but the general sequential pipeline still consumes fixed `epsilon_qr` / `epsilon_aw` values from config.
3. The residual-correction math is now fixed in shared code, but only `layer1 self_attn.o_proj` received the full line-by-line manual audit in this round.
4. Qwen3-8B was intentionally not run in this round.

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
8. The narrow `layer1 self_attn.o_proj` audit is now fully sane with `reference_stats: true` and rescalers off:
   - stage timing verified: same-layer, post-QKV, pre-o_proj
   - manual `Sigma_Delta,Xhat` vs official collector mismatch: `0`
   - wrong-sign mismatch: `3.8401e-3`
   - `||Sigma_Delta,Xhat||_F = 1.9200e-3`
   - `||W Sigma_X,Xhat||_F = 5.3348e-1`
   - residual/base ratio at scale `1.0`: `3.5990e-3`
   - corrected-path `target_y_max_abs = 0.190439`
   - legacy-formula `target_y_max_abs = 85.947011`
9. With that fix, the first `11` modules of `Llama-3.2-1B` are stable again with residual correction enabled:
   - config: `configs/quant/watersic_llama32_1b_multilayer_smoke_ref_stagefix_residfixed.yaml`
   - achieved effective bitwidth: `2.9919`
   - entropy bitwidth: `2.9786`
   - Huffman bitwidth: `3.0343`
   - small-eval PPL: `9.6433`
   - `model.layers.1.self_attn.o_proj` relative weight MSE: `2.1246e-1`
   - `model.layers.1.self_attn.o_proj` `target_y_max_abs = 0.190439`

## Not Yet Valid To Claim

1. It is not yet valid to claim a full paper-faithful WaterSIC reproduction.
2. It is not yet valid to claim that the current repo reproduces the paper’s final `Llama-3.2-1B` `3.00`-bit result (`10.57` PPL in Table 1), because no full-model post-fix run has completed.
3. It is not yet valid to claim that every residual target is fully validated; this round proved the fix for the previous `layer1 self_attn.o_proj` blocker and the surrounding smoke path, not for every later layer.
4. It is not yet valid to claim that diagonal rescalers are safe as the default path for broader runs.
5. It is not yet valid to compare the successful `11`-module smoke directly to the paper as if it were a full reproduction; it uses truncated evaluation and leaves most of the model unquantized.

## Immediate Next Step

1. Keep rescalers disabled by default.
2. Start the fuller `Llama-3.2-1B` run with:
   - `reference_stats: true`
   - fixed residual correction
   - rescalers still off
   - paper-faithful ~`3.0`-bit settings
3. If that fuller run exposes a new instability, localize the first failing layer with the same staged per-layer diagnostics now added for the smoke path.
