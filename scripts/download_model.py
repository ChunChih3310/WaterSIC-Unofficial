#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.pipeline import download_model_snapshot
from watersic.utils.runtime import load_config, prepare_runtime, sanitize_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and snapshot a model into the repo")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    _, logger = prepare_runtime(log_name="download_model", debug=args.debug, seed=0)
    output_dir = args.output_dir or ROOT / "outputs" / "original" / sanitize_name(model_config["model_id"])
    download_model_snapshot(model_config, output_dir=output_dir, logger=logger)


if __name__ == "__main__":
    main()
