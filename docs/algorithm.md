# Algorithm

This repository implements WaterSIC as an unequal-rate linear-layer quantizer.
The implementation is organized around the paper's transformed objective rather
than fixed-range integer quantization.

## Plain WaterSIC Path

For a linear layer with weight matrix `W` and input covariance `Sigma_X`, the
plain objective uses:

- Cholesky factorization `Sigma_X = L L^T`
- transformed target `Y = W L`
- per-column spacing `alpha_i = c / L_ii`
- ZSIC recursion from the last column to the first
- LMMSE shrinkage factors after integer rounding
- entropy and canonical Huffman estimates for the integer symbols

The core ZSIC implementation lives in `src/watersic/quant/zsic.py`.
Rate search over `c` lives in `src/watersic/quant/rate_search.py`.

## Full Model Terms

The model-level path adds the engineering terms used by this reproduction:

- activation-drift correction using current-model activations
- residual correction for `o_proj` and `down_proj`
- attention-weighted second moments for Q/K/V projections
- adaptive mixing for attention statistics
- diagonal row/column rescalers in the transformed objective
- dead-feature erasure and reinsertion

These terms are configured in `layer:` sections of quantization YAML files.

## Scientific Status

The code is intended to preserve the WaterSIC algorithmic structure. It does not
claim to be an official implementation, and it does not claim exact paper
equivalence unless the relevant config, model revision, dataset split, tokenizer,
and metric are explicitly stated.
