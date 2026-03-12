# Known Issues

## Implementation Gaps

1. Adaptive mixing search for `epsilon_qr` and `epsilon_aw` is not fully optimized yet.
2. The current default configs keep `reference_stats: false` for runtime safety during the first overnight runs.
3. The paper-comparison report will only be complete once the first real quantized runs finish and the actual numbers are written out.

## Runtime Risks

1. Full attention-output collection can be slow on long-context calibration.
2. Saving full reconstructed-weight artifacts is disk-heavy by design.
3. A second full-precision reference model for paired stats increases memory and runtime substantially, especially for `Qwen3-8B`.

## Honesty Note

This repository already contains real WaterSIC math and runnable infrastructure, but it is not yet valid to claim a complete paper reproduction until:

- the first Llama-3.2-1B run has finished cleanly,
- the resulting artifact has been benchmarked,
- and the paper-vs-ours report has been written from real outputs.
