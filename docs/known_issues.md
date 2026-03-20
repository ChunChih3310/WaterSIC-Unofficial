# Known Issues

Detailed historical status is archived at:

- `archive/docs/known_issues_detailed.md`

## Current Best Result

- model:
  - `Llama-3.2-1B`
- best completed artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- effective bits:
  - `2.9984`
- original WikiText-2 test PPL:
  - `10.6031`
- matched WikiText-2 validation PPL:
  - `10.9310`
- paper reference at `3.00` bits:
  - `10.57`

## Active Issues

1. The main remaining `Llama-3.2-1B` gap is now a real validation-to-validation gap:
   - repo: `10.9310`
   - paper: `10.57`
   - delta: `+0.3610`

2. The repo is best described as a near-reproduction, not an exact paper match.
   Remaining caveats are:
   - Hugging Face `main` model/tokenizer revisions are not pinned to a paper-specific public snapshot
   - exact tokenizer/BOS/chunk-count realization is not proven identical

3. The remaining distortion is still concentrated mainly in residual-path projections:
   - `self_attn.o_proj`
   - `mlp.down_proj`

4. `Qwen3-8B` is still intentionally deferred until the `Llama-3.2-1B` validation-gap story is fully wrapped up.

5. Paper-scale runs remain expensive on the current A6000 setup:
   - rescaler-only estimate: about `37.5h` at `batch_size=2`
   - repaired adaptive-mixing estimate: about `63.7h` at `batch_size=2`

6. Historical runs that predate integer-symbol serialization cannot backfill exact Huffman shortest/longest code lengths.

## Already Settled

- Core WaterSIC math bugs in ZSIC, transformed-target construction, and residual compensation are fixed.
- Diagonal rescalers are validated and beneficial.
- Repaired adaptive mixing is stable and beneficial at larger calibration budget.
- The final `Llama-3.2-1B` paper-scale path includes:
  - activation drift correction
  - residual compensation
  - attention-weighted calibration
  - adaptive mixing
  - diagonal rescalers
  - dead-feature erasure
  - paper-faithful damping and search defaults

## Cleanup Status

- Debug and superseded milestone material has been archived under `archive/`.
- Low-risk redundant logs and probe outputs have been removed.
- Manual-review items were intentionally left in place.
