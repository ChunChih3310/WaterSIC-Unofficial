# Known Issues

## Critical Blockers

1. Full-model paper-faithful reproduction is still blocked. This round only revalidated the `Llama-3.2-1B` layer-0 attention block.
2. The fixed core math has not yet been rerun on a larger calibration/eval slice or on multi-layer sequential quantization with `reference_stats: true`.
3. Diagonal rescalers remain disabled by default outside the dedicated debug stage. Stage `H` is sane on the small layer-0 ladder, but that is not enough evidence to enable rescalers globally.
4. The current WikiText-2 loader tokenizes the full concatenated split before chunking. It works, but it is inefficient and produces a long-sequence tokenizer warning.

## Implementation Gaps

1. Full-model sequential quantization with the fixed core math has not yet been rerun.
2. Qwen3-8B was intentionally not run in this debugging round.
3. The layer-0 debug ladder uses a small slice (`4` train chunks for calibration, `4` test chunks for eval). It is a correctness run, not a final benchmark.
4. `reference_stats` are meaningfully exercised for `o_proj` in this layer-0 run, but later-layer and full-model reference-stat refresh remains unvalidated.

## Already Validated

1. The ZSIC recursive-update bug is fixed. The update now uses `alpha_i` instead of the global `c`.
2. The transformed-target `Y` construction bug is fixed. The triangular solve now implements `target_cross @ (L^T)^(-1)` instead of `target_cross @ L^(-1)`.
3. Calibration statistics are now collected under `torch.no_grad()`, removing the paired-model attention OOM path.
4. The staged layer-0 attention ladder `A` through `H` completed successfully on `Llama-3.2-1B`, with saved configs, logs, and results under `outputs/reports/llama32_1b_layer0_attention_debug/`.
5. On this small layer-0 run, `HPTQ-equivalent`, `PlainWaterSIC`, `+LMMSE`, `+activation drift`, `+residual correction`, `+attention weighting`, `+adaptive mixing`, and `+diagonal rescalers` all remained numerically sane.

## Not Yet Valid To Claim

1. It is not yet valid to claim a full paper-faithful WaterSIC reproduction.
2. It is not yet valid to claim that `reference_stats: true` has been validated for full sequential quantization beyond this layer-0 attention block.
3. It is not yet valid to claim that diagonal rescalers are safe as the default path for broader runs.
4. It is not yet valid to claim that the current repo reproduces the paper’s final WikiText-2 results after the recent math fixes, because the fixed pipeline has not yet been rerun end-to-end.

## Immediate Next Step

1. Keep rescalers disabled by default.
2. Rerun the fixed pipeline on a slightly larger layer-0 or layer-0-plus-MLP debug slice with `reference_stats: true`.
3. Once that remains sane, rerun the first multi-module / multi-layer smoke quantization for `Llama-3.2-1B` with the corrected core math and compare against the earlier catastrophic runs.
