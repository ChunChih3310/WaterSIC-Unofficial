# Calibration Pipeline

## Dataset

- Dataset: WikiText-2
- Calibration split: `train`
- Tokenization: tokenizer from the model config
- Sequence construction: concatenate text, tokenize deterministically, split into non-overlapping length-2048 blocks

## Statistics Collected

For each target linear module:

- `Sigma_X`
- `Sigma_Xhat`
- `Sigma_X,Xhat`
- `Sigma_Delta,Xhat` for residual targets

For QKV modules:

- weighted versions of the same second moments using attention-derived token weights

## Sequential Dependency

The intended pipeline is sequential:

- earlier layers are quantized first
- later-layer statistics are then gathered from the partially quantized model

The current implementation already quantizes layers sequentially and gathers current-model activations after earlier replacements. Reference-model paired stats are supported by loading an unquantized copy on a separate device, but this path is still being exercised in real runs.
