# Known Issues

## Release Blockers

- No known release blocker remains in the curated public `main` tree.
- The public `main` branch is intended to be published as a single clean
  public-release commit. Local backup branches may retain pre-release history and
  must not be pushed.

## Implementation Limitations

- The repository is a near-reproduction, not an official or proven exact
  paper-faithful implementation.
- Public model/tokenizer revisions use `main` unless otherwise configured; the
  exact paper snapshot is not pinned.
- Qwen3-8B paper-scale configs are present but not validated here.
- Adaptive-mixing search and binary-search internals cannot resume mid-search.
- Historical runs that predate integer-symbol serialization cannot backfill exact
  Huffman shortest/longest code lengths.

## Operational Limitations

- Paper-scale runs require substantial GPU memory and runtime.
- Generated reports, logs, checkpoints, model snapshots, and quantized artifacts
  are ignored by git and must be regenerated locally.
- `scripts/quantize_model.py` and `scripts/run_experiment.py` both run
  quantization followed by evaluation; there is not currently a no-evaluation
  artifact-only CLI path.
