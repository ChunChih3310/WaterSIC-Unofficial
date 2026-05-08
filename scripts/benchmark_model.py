#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.pipeline import benchmark_saved_model
from watersic.utils.runtime import load_config, prepare_runtime, resolve_torch_device, select_runtime_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a saved quantized artifact or repo-local model snapshot")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--device-config", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_config = load_config(args.eval_config)
    device_config = load_config(args.device_config) if args.device_config else {}
    _, logger = prepare_runtime(log_name="benchmark_model", debug=args.debug, seed=0)
    selection = select_runtime_device(device_config.get("device"), logger, seed=0)
    result = benchmark_saved_model(args.model_path, eval_config, device=resolve_torch_device(selection), logger=logger)
    logger.info("Benchmark result: %s", result)


if __name__ == "__main__":
    main()
