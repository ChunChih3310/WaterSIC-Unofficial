#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watersic.utils.device import pick_idle_gpu
from watersic.utils.io import load_yaml
from watersic.utils.logging import configure_logging
from watersic.utils.path_guard import repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a truly idle GPU, then launch a WaterSIC experiment.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve_run_log(run_name: str) -> Path | None:
    candidates = sorted(repo_path("outputs", "logs").glob(f"run_{run_name}_*.log"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    quant_config = load_yaml(args.quant_config)
    run_name = str(quant_config["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log = repo_path("outputs", "logs", f"launcher_{run_name}_{timestamp}.log")
    logger = configure_logging(f"launcher_{run_name}", debug=args.debug, log_file=launcher_log)
    logger.info("Launcher log: %s", launcher_log)
    logger.info("Preparing to wait for a truly idle GPU before launching %s", run_name)

    device_config = quant_config.get("device")
    selection = None
    while selection is None:
        try:
            selection = pick_idle_gpu(device_config=device_config, logger=logger)
        except RuntimeError as exc:
            logger.warning("No idle GPU available yet: %s", exc)
            logger.info("Sleeping for %d seconds before retrying idle-GPU selection", args.poll_seconds)
            time.sleep(args.poll_seconds)

    logger.info("Idle GPU selected for launch: %s", selection)

    env = os.environ.copy()
    if selection.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = selection.cuda_visible_devices
    env["PYTHONPATH"] = str(SRC) if not env.get("PYTHONPATH") else f"{SRC}:{env['PYTHONPATH']}"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--model-config",
        args.model_config,
        "--quant-config",
        args.quant_config,
        "--eval-config",
        args.eval_config,
    ]
    if args.debug:
        cmd.append("--debug")

    logger.info("Launching command: %s", cmd)
    with launcher_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    logger.info("Launched child PID=%s", process.pid)

    run_log = None
    for _ in range(10):
        time.sleep(1)
        run_log = _resolve_run_log(run_name)
        if run_log is not None:
            break
    if run_log is not None:
        logger.info("Resolved experiment log path: %s", run_log)
    else:
        logger.warning("Experiment log path not resolved yet; check outputs/logs for run_%s_*.log", run_name)

    exit_code = process.wait()
    if exit_code != 0:
        logger.error("Child process exited with code %d", exit_code)
        raise SystemExit(exit_code)
    logger.info("Child process completed successfully")


if __name__ == "__main__":
    main()
