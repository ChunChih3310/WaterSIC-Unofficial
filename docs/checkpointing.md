# Checkpointing

Checkpointing is intended for long paper-scale quantization runs. It is optional
and configured under the `checkpoint:` section of quantization configs.

Example:

```yaml
checkpoint:
  enabled: true
  dir: outputs/checkpoints
  resume: auto
  strict_git_commit: true
  keep_after_completion: true
  save_collection_every_batches: 32
```

Implemented:

- run manifest validation
- committed stage checkpoints
- replay of committed stage weights into a fresh model
- saved finalized stage statistics
- periodic accumulator checkpoints during statistics collection

Not implemented:

- resume from the middle of adaptive-mixing search
- resume from the middle of binary-search iterations
- automatic cleanup of completed checkpoint directories

The durable unit is a layer-local stage, not an individual module. This preserves
the sequential statistics semantics for QKV and gate/up groups that share a
pre-stage collection pass.
