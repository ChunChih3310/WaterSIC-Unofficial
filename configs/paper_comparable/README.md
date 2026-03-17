Paper-comparable configs live here.

Purpose:
- keep the paper-scale calibration configs separate from smoke, sweep, and debugging configs
- provide one obvious place to launch the closest repo-supported paper-faithful runs

Paper-scale calibration definition used here:
- dataset: WikiText-2 train split
- chunking: concatenate and split into non-overlapping 2048-token sequences
- calibration sequences: 1188

Pipeline defaults used here:
- `reference_stats: true`
- fixed residual correction
- staged same-layer stat refresh
- diagonal rescalers enabled
- repaired adaptive mixing enabled
- conservative idle-only GPU auto-selection

Bitrate notes:
- `Llama-3.2-1B` uses `3.0` bits to match Figure 1 in the paper.
- `Qwen3-8B` uses `3.125` bits to match Figure 2 in the paper.
- `Llama-3.1-8B`, `Llama-3-8B`, and `Llama-2-7B` use paper-scale calibration with a `3.0`-bit default in this repo folder. These are suitable paper-scale launch points, but you should still match the exact appendix/figure rate manually if you are targeting a specific published curve point.
