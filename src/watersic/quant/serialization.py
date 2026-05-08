from __future__ import annotations

from pathlib import Path
from typing import Any

from watersic.utils.io import save_json
from watersic.utils.path_guard import ensure_parent_dir, repo_path


def save_quantized_artifact(model, tokenizer, run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    target_dir = ensure_parent_dir(Path(run_dir) / "metadata.json").parent
    model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(target_dir)
    save_json(target_dir / "metadata.json", metadata)
    return target_dir
