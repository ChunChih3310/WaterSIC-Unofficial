# Quantization Pipeline

## Per-Layer Flow

1. Build calibration moments for the target linear module.
2. Detect dead features using the median-variance rule with `tau = 1e-3`.
3. Remove dead input dimensions from the solve.
4. Form the transformed target using:
   - activation-drift correction
   - residual correction for `o_proj` and `down_proj`
5. Cholesky-factorize the damped `Sigma_Xhat`.
6. Run binary search over the WaterSIC parameter `c`.
7. For each candidate `c`, run transformed-space ZSIC and estimate entropy-based rate.
8. Re-run the chosen `c` on the full matrix.
9. Optimize diagonal row/column rescalers.
10. Reinsert dead features as zero columns.

## Model-Level Flow

1. Discover all target transformer linears in model order.
2. Maintain a global remaining bitrate budget over the remaining weights.
3. Quantize each target module in sequence.
4. Replace the module weight in the live model.
5. Save the resulting reconstructed-weight artifact plus JSON metadata.

## Saved Metadata

Each run saves:

- per-layer target/achieved rates
- raw/entropy/Huffman/side-information bitwidths
- selected `c`
- per-column spacings
- LMMSE factors
- dead-feature counts
- report JSON and Markdown
