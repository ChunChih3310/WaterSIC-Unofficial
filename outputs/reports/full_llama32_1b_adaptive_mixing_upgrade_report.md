# Full-Model Llama-3.2-1B Adaptive Mixing Upgrade Report

## Scope

- Run name: `llama32_1b_full_3p0bit_reftrue_rescaler_mixing`
- Config: `configs/quant/watersic_llama32_1b_full_reftrue_rescaler_mixing.yaml`
- Model: `Llama-3.2-1B`
- Calibration: WikiText-2 train, `8` chunks, sequence length `2048`
- Evaluation: full WikiText-2 test perplexity
- Shared path:
  - `reference_stats: true`
  - fixed residual correction enabled
  - staged same-layer stat refresh enabled
  - diagonal rescalers enabled
- Upgrade under test:
  - full per-attention-block adaptive mixing search enabled
  - `golden_section_iterations: 15`

## Completion Status

The run finished successfully.

- Achieved effective bits: `2.9984`
- Entropy bits: `2.9865`
- Huffman bits: `3.0368`
- Side-information overhead: `0.0119`
- Baseline WikiText-2 PPL: `9.7041`
- Quantized WikiText-2 PPL: `16.6096`
- Quantization runtime: `74219.60s`
- Total runtime: `74349.91s`
- Peak GPU memory: `18.65 GiB`
- Quantization anomalies: none
- Artifact: `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing/`

## Comparison Against The Rescaler-Only Reference

Reference point:

- run: `llama32_1b_full_3p0bit_reftrue_rescaler`
- quantized PPL: `15.7029`
- effective bits: `2.9984`

Upgrade result:

- quantized PPL: `16.6096`
- delta vs rescaler-only: `+0.9067`

Bitrate changed negligibly:

| Metric | Rescaler Only | Rescaler + Mixing | Delta |
| --- | ---: | ---: | ---: |
| Effective bits | `2.9984` | `2.9984` | `+0.0000` |
| Entropy bits | `2.9865` | `2.9865` | `+0.0000` |
| Huffman bits | `3.0368` | `3.0368` | `+0.0000` |
| Side-info bits | `0.0119` | `0.0119` | `+0.0000` |
| Quantized PPL | `15.7029` | `16.6096` | `+0.9067` |

## What Improved Locally

The adaptive-mixing search was fully active and improved the local `wo`-input objective inside each attention block.

Examples from the live log:

- layer 12:
  - initial: `1.705318e-02`
  - after `epsilon_qr* = 0.382699`: `1.658778e-02`
  - after `epsilon_aw* = 0.681893`: `1.506454e-02`
- layer 13:
  - initial: `1.885044e-02`
  - after `epsilon_qr* = 0.529783`: `1.840021e-02`
  - after `epsilon_aw* = 0.725198`: `1.659636e-02`
- layer 14:
  - initial: `1.547815e-02`
  - after `epsilon_qr* = 0.395122`: `1.522349e-02`
  - after `epsilon_aw* = 0.621140`: `1.410376e-02`
- layer 15:
  - initial: `8.610459e-03`
  - after `epsilon_qr* = 0.379033`: `8.350664e-03`
  - after `epsilon_aw* = 0.605612`: `7.412780e-03`

All `48` QKV matrices were quantized with optimized adaptive-mixing parameters.

Per-kind mean relative weight MSE also improved for QKV:

| Kind | Rescaler Only | Rescaler + Mixing | Delta |
| --- | ---: | ---: | ---: |
| `q_proj` | `0.0696` | `0.0611` | `-0.0084` |
| `k_proj` | `0.0983` | `0.0898` | `-0.0085` |
| `v_proj` | `0.0876` | `0.0806` | `-0.0069` |

## What Did Not Improve

The improved local QKV objective did not translate into a better full-model result.

Per-kind mean relative weight MSE for the main residual-path error concentrations was flat to slightly worse:

| Kind | Rescaler Only | Rescaler + Mixing | Delta |
| --- | ---: | ---: | ---: |
| `o_proj` | `0.3170` | `0.3175` | `+0.0005` |
| `down_proj` | `0.3023` | `0.3024` | `+0.0002` |

Representative late-layer comparisons:

| Layer | Kind | Rescaler Only | Rescaler + Mixing | Delta |
| --- | --- | ---: | ---: | ---: |
| `model.layers.12.self_attn.o_proj` | `o_proj` | `0.4178` | `0.4202` | `+0.0023` |
| `model.layers.13.self_attn.o_proj` | `o_proj` | `0.4596` | `0.4647` | `+0.0051` |
| `model.layers.14.self_attn.o_proj` | `o_proj` | `0.2670` | `0.2688` | `+0.0018` |
| `model.layers.11.mlp.down_proj` | `down_proj` | `0.3146` | `0.3168` | `+0.0021` |

Worst-layer list remained dominated by `o_proj`:

- `model.layers.11.self_attn.o_proj`: `0.4777`
- `model.layers.13.self_attn.o_proj`: `0.4647`
- `model.layers.12.self_attn.o_proj`: `0.4202`
- `model.layers.4.self_attn.o_proj`: `0.4021`
- `model.layers.0.self_attn.o_proj`: `0.3926`

## Diagnosis

This run validates that the general-pipeline adaptive-mixing search:

- is implemented end-to-end
- is numerically stable
- can optimize the intended local `wo`-input objective

But it does **not** yet improve the global model-quality objective.

Current evidence points to a mismatch between:

- the local search target used for adaptive mixing
- and the final full-model WikiText-2 PPL objective

Because the upgraded run is worse than the rescaler-only reference despite better local QKV objectives, calibration size is not yet the clearest immediate limiter. The more immediate issue is validating or revising the adaptive-mixing objective/search coupling.

## Bottom Line

- Adaptive mixing is now fully exercised on the real full-model pipeline.
- It is stable.
- It does **not** beat the rescaler-only reference.
- The best completed `Llama-3.2-1B` result therefore remains:
  - `llama32_1b_full_3p0bit_reftrue_rescaler`
  - `15.7029` PPL at `2.9984` effective bits
