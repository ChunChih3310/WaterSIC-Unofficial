# Reproduction Log

Detailed chronological history is archived at:

- `archive/docs/reproduction_log_full.md`

## Main Milestones

### 2026-03-20

- Completed the paper-scale repaired adaptive-mixing `Llama-3.2-1B` run:
  - run: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
  - effective bits: `2.9984`
  - WikiText-2 test PPL: `10.6031`
- Reran the final benchmark on WikiText-2 validation:
  - validation PPL: `10.9310`
  - paper reference: `10.57`
  - validation gap: `+0.3610`
- Produced:
  - `outputs/reports/final_paper_audit.md`
  - `outputs/reports/final_worst_layer_diagnosis.md`
  - `outputs/reports/debug_artifact_cleanup_inventory.md`
- Executed the repository cleanup:
  - low-risk redundant files deleted
  - debug and superseded milestone material archived under `archive/`
  - mainline result surface preserved

### 2026-03-17

- Completed the repaired adaptive-mixing `64`-chunk full-model `Llama-3.2-1B` run:
  - effective bits: `2.9984`
  - WikiText-2 PPL: `11.1874`
- Hardened GPU auto-selection and CUDA device assignment to require truly idle GPUs by default.

### 2026-03-15

- Completed the `16`-chunk and `32`-chunk rescaler-only calibration-sweep milestones:
  - `16` chunks: `12.4574` PPL
  - `32` chunks: `11.7806` PPL
- Completed the repaired adaptive-mixing full-model validation run:
  - `16.2796` PPL
- Completed the repaired adaptive-mixing prefix validation:
  - proved the repaired search path was materially faster and numerically sane

### 2026-03-13

- Reproduced the core WaterSIC pipeline end to end on `Llama-3.2-1B`.
- Fixed the core ZSIC and transformed-target math bugs.
- Fixed the residual-compensation formula for sequential `o_proj`.
- Established the first stable full-model runs:
  - no-rescaler baseline
  - rescaler-enabled baseline
- Completed the narrow debug ladders and smoke runs that isolated the major early blockers.

## Current Status

- Best completed `Llama-3.2-1B` artifact:
  - `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale`
- Best paper-comparable number:
  - WikiText-2 validation PPL `10.9310`
- Repo state:
  - near-reproduction for `Llama-3.2-1B`
  - not yet an exact paper match
- Next major unresolved item:
  - explain the remaining `+0.3610` validation gap before broadening to `Qwen3-8B`
