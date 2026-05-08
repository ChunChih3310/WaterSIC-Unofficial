#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate JSON reports into a single markdown table")
    parser.add_argument("--reports-dir", default=str(ROOT / "outputs" / "reports"))
    parser.add_argument("--output-path", default=str(ROOT / "outputs" / "reports" / "aggregate_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    rows = []
    for json_path in sorted(reports_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(
            (
                payload["model_id"],
                payload["target_global_bitwidth"],
                payload["achieved_global_bitwidth"],
                payload.get("perplexity"),
                json_path.name,
            )
        )

    lines = [
        "# Aggregate WaterSIC Runs",
        "",
        "| Model | Target Bits | Achieved Bits | Perplexity | JSON |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for model_id, target_bits, achieved_bits, perplexity, json_name in rows:
        lines.append(f"| {model_id} | {target_bits:.4f} | {achieved_bits:.4f} | {perplexity} | {json_name} |")
    Path(args.output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
