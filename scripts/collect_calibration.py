#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.pipeline import export_calibration
from watersic.utils.runtime import load_config, prepare_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic calibration token blocks")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    quant_config = load_config(args.quant_config)
    _, logger = prepare_runtime(log_name="collect_calibration", debug=args.debug, seed=0)
    output_path = args.output_path or ROOT / "outputs" / "stats" / f"{quant_config['run_name']}_calibration.pt"
    path = export_calibration(model_config, quant_config, output_path=output_path)
    logger.info("Calibration tokens saved to %s", path)


if __name__ == "__main__":
    main()
