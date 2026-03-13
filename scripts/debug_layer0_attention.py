#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.debug.layer0_attention import run_layer0_attention_debug
from watersic.utils.runtime import load_config, prepare_runtime, resolve_torch_device, select_runtime_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged layer-0 attention WaterSIC debug ladder")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--debug-config", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    debug_config = load_config(args.debug_config)
    seed = int(debug_config.get("quant", {}).get("seed", 0))

    _, logger = prepare_runtime(log_name=f"debug_{debug_config['run_name']}", debug=args.debug, seed=seed)
    selection = select_runtime_device(debug_config.get("device"), logger)
    device = resolve_torch_device(selection)

    reference_device = device
    if debug_config.get("reference_device") == "cpu":
        from torch import device as torch_device

        reference_device = torch_device("cpu")

    result = run_layer0_attention_debug(
        model_config,
        debug_config,
        device=device,
        logger=logger,
        reference_device=reference_device,
    )
    logger.info("Layer-0 attention debug complete: %s", result["summary_markdown_path"])


if __name__ == "__main__":
    main()
