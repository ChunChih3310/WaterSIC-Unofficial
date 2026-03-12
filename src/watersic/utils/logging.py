from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .path_guard import ensure_parent_dir, repo_path


def configure_logging(name: str = "watersic", *, debug: bool = False, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = repo_path("outputs", "logs", f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(ensure_parent_dir(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
