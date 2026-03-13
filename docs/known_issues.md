# Known Issues

## Critical Blockers

1. Full-model paper-faithful reproduction is still blocked on the residual-correction path. In the first nontrivial multi-layer smoke with `reference_stats: true` and staged same-layer stat refresh, the first catastrophic module is `model.layers.1.self_attn.o_proj`.
2. With residual correction enabled, `model.layers.1.self_attn.o_proj` reaches relative weight MSE `4.0915395089e9`, `target_y_max_abs = 85.9470`, `alpha` range `[10.6854, 435.0766]`, and the smoke perplexity jumps to `700443.07`.
3. Disabling residual correction while keeping the same staged-refresh path makes the same smoke sane again: `model.layers.1.self_attn.o_proj` relative weight MSE drops to `1.5426e-1` and small-eval PPL to `9.6219`.
4. Diagonal rescalers remain disabled by default. They are still not validated beyond the dedicated layer-0 debug ladder.
5. The current WikiText-2 loader tokenizes the full concatenated split before chunking. It works, but it is inefficient and emits a long-sequence tokenizer warning.

## Implementation Gaps

1. A full-model `Llama-3.2-1B` run at ~`3.0` bits has still not been completed after the latest fixes, because the residual-correction blocker should be fixed before spending more runtime on a paper-comparison run.
2. The current successful multi-layer smoke uses `use_residual_correction: false`, so it is not yet a fully paper-faithful WaterSIC run.
3. Adaptive mixing is optimized in the dedicated attention debugger, but the general sequential pipeline still consumes fixed `epsilon_qr` / `epsilon_aw` values from config.
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
7. With that staged refresh and `reference_stats: true`, the first `11` modules of `Llama-3.2-1B` can be quantized stably at `2.9917` effective bits if residual correction is disabled:
   - baseline small-eval PPL: `8.9880`
   - quantized small-eval PPL: `9.6219`
   - `model.layers.1.self_attn.o_proj` relative weight MSE: `1.5426e-1`

## Not Yet Valid To Claim

1. It is not yet valid to claim a full paper-faithful WaterSIC reproduction.
2. It is not yet valid to claim that the current residual-correction implementation is correct for sequential multi-layer quantization of `o_proj` / `down_proj`.
3. It is not yet valid to claim that the current repo reproduces the paper’s final `Llama-3.2-1B` `3.00`-bit result (`10.57` PPL in Table 1), because no full-model post-fix run has completed.
4. It is not yet valid to claim that diagonal rescalers are safe as the default path for broader runs.
5. It is not yet valid to compare the successful `11`-module smoke directly to the paper as if it were a full reproduction; it uses truncated evaluation and leaves most of the model unquantized.

## Immediate Next Step

1. Keep rescalers disabled by default.
2. Debug the residual-compensation term for `o_proj` / `down_proj` under sequential quantization by comparing:
   - `target_cross` with and without compensation
   - `target_y` magnitude before and after the compensation addend
   - the exact `Sigma_Delta,Xhat` construction against the paper’s intended residual stream definition
3. After fixing that residual path, rerun:
   - `configs/quant/watersic_llama32_1b_multilayer_smoke_ref_stagefix.yaml`
   - then `configs/quant/watersic_llama32_1b_3p0bit_refsafe.yaml`
