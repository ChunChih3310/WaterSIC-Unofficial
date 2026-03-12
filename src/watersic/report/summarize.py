from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from watersic.utils.io import load_json, save_json


def aggregate_reports(report_paths: list[Path]) -> list[dict]:
    aggregated = []
    for path in report_paths:
        aggregated.append(load_json(path))
    return aggregated
