# Full-Model Llama-3.2-1B Calibration Sweep Report

Date: 2026-03-15

## Status

This sweep is in progress.

- Completed anchor reference:
  - `llama32_1b_full_3p0bit_reftrue_rescaler`
  - effective bits: `2.9984`
  - quantized WikiText-2 PPL: `15.7029`
- New completed sweep points:
  - none yet
- Active run:
  - `llama32_1b_full_3p0bit_reftrue_rescaler_calib16`
  - log: `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_calib16_20260315_133857.log`
- Deferred until `16`-chunk completion:
  - final `32`-chunk run

## Safe Runtime Change

This round added a repo-local WikiText-2 token-block cache:

- code: `src/watersic/data/wikitext2.py`
- test: `tests/test_wikitext2.py`
- commit: `6173f21`

What changed:

- the loader now caches the exact tokenized `input_ids` blocks for each `(split, sequence_length, tokenizer identity)` under:
  - `outputs/stats/wikitext2_cache/`

Why it is safe:

- the dataset split is unchanged
- the full text concatenation is unchanged
- the tokenizer call is unchanged
- chunking into non-overlapping length-`2048` blocks is unchanged
- later runs reuse the same saved block tensor instead of retokenizing

Observed effect:

- first uncached launch still tokenized the split and emitted the long-sequence tokenizer warning once
- later cached launches reused the saved block tensors and skipped that retokenization path
- for the current runs, the cached `16`-chunk launch reached baseline evaluation roughly `10s` earlier than the uncached anchor rerun

## Anchor Handling

The repository already had the completed `8`-chunk anchor point from the validated rescaler-only run, so that point remains the sweep anchor.

A redundant `8`-chunk anchor rerun was started with the cached loader and matched the expected early layer-0 and layer-1 rates and errors, but it was then stopped to free resources for the `16`-chunk run.

## Current `16`-Chunk Evidence

The `16`-chunk run is numerically stable through layer 1 so far. No NaN/Inf, Cholesky failure, or residual-path blow-up has appeared.

Early relative weight MSE comparison against the completed `8`-chunk anchor:

| Layer / Kind | `8` chunks | `16` chunks | Delta |
| --- | ---: | ---: | ---: |
| `l0 q_proj` | `0.1541` | `0.0965` | `-0.0576` |
| `l0 k_proj` | `0.2057` | `0.1285` | `-0.0772` |
| `l0 v_proj` | `0.2044` | `0.1278` | `-0.0766` |
| `l0 o_proj` | `0.3936` | `0.3108` | `-0.0828` |
| `l0 gate_proj` | `0.0598` | `0.0476` | `-0.0122` |
| `l0 up_proj` | `0.0586` | `0.0467` | `-0.0119` |
| `l0 down_proj` | `0.3635` | `0.1634` | `-0.2001` |
| `l1 q_proj` | `0.0690` | `0.0619` | `-0.0071` |
| `l1 k_proj` | `0.1097` | `0.0980` | `-0.0118` |
| `l1 v_proj` | `0.0983` | `0.0877` | `-0.0105` |
| `l1 o_proj` | `0.2754` | `0.2080` | `-0.0674` |

Interpretation of the current partial evidence:

- more calibration is already helping the dominant residual-path kinds locally
- the strongest early gain is on `down_proj`
- `o_proj` is also improving materially in the first two layers
- this increases confidence that calibration budget is a real limiter on the stable rescaler-only path

## What Is Still Missing

The sweep does not yet have a new completed full-model PPL point beyond the existing `8`-chunk reference.

Because the `16`-chunk run is still active:

- there is no completed `16`-chunk effective-bits / entropy / Huffman / final PPL report yet
- there is no completed `32`-chunk point yet
- it is not yet valid to claim how much of the paper gap is calibration-limited at the full-model benchmark level

## Current Best Completed Point

- run: `llama32_1b_full_3p0bit_reftrue_rescaler`
- effective bits: `2.9984`
- quantized WikiText-2 PPL: `15.7029`

## Immediate Next Action

Let the active `16`-chunk run finish before making any more algorithm-path changes.
