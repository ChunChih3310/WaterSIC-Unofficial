#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.pipeline import run_full_experiment
from watersic.utils.runtime import load_config, prepare_runtime, resolve_torch_device, select_runtime_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full WaterSIC quantize+benchmark experiment")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    quant_config = load_config(args.quant_config)
    eval_config = load_config(args.eval_config)
    quant_config["_config_path"] = args.quant_config
    eval_config["_config_path"] = args.eval_config

    seed = int(quant_config.get("layer", {}).get("seed", 0))
    _, logger = prepare_runtime(log_name=f"run_{quant_config['run_name']}", debug=args.debug, seed=seed)
    selection = select_runtime_device(quant_config.get("device"), logger, seed=seed)
    device = resolve_torch_device(selection)

    result = run_full_experiment(
        model_config,
        quant_config,
        eval_config,
        device=device,
        logger=logger,
    )
    logger.info("Experiment complete: %s", result)


if __name__ == "__main__":
    main()
