# Configuration

Configs are YAML files grouped by purpose.

## Model Configs

Model configs live under `configs/models/` and
`configs/paper_comparable/models/`. Important fields:

- `model_id`: Hugging Face model id or a repo-local model path
- `model_revision`: model revision to request
- `tokenizer_id`: tokenizer id or path
- `tokenizer_revision`: tokenizer revision
- `architecture`: adapter family, currently `llama` or `qwen`
- `dtype`: model dtype, typically `bfloat16`
- `attn_implementation`: transformer attention implementation
- `trust_remote_code`: passed to Transformers

Public configs should avoid machine-specific absolute paths.

## Quantization Configs

Quant configs live under `configs/quant/` and
`configs/paper_comparable/quant/`. Important sections:

- `run_name`: output directory and report stem
- `target_global_bitwidth`: target average compressed rate
- `reference_stats`: whether to load an unquantized reference model
- `reference_device`: device for the reference model when enabled
- `max_layers` / `max_modules`: optional smoke limits
- `device`: idle-GPU selection policy and optional override
- `calibration`: split, sequence length, sequence count, and batch size
- `layer`: WaterSIC numerical options
- `checkpoint`: optional resume behavior

## Device Policy

If `CUDA_VISIBLE_DEVICES` is already set, the runtime respects it. Otherwise, the
selector ranks visible GPUs by active process count, used/free memory, and
utilization. If no GPU satisfies the configured idle thresholds, the default is a
clear failure. Set `device.allow_busy_fallback: true` only when that behavior is
intentional.

After selection, torch normally sees logical `cuda:0`; that logical device maps
to the selected physical GPU through `CUDA_VISIBLE_DEVICES`.

## Paper-Comparable Configs

`configs/paper_comparable/` provides one clear launch surface for the closest
repo-supported paper-scale settings. These configs are expensive and should not
be used as quick smoke tests.
