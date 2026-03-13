from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watersic.data.calibration import CalibrationConfig, load_calibration_dataset
from watersic.data.wikitext2 import load_wikitext2_blocks
from watersic.eval.runner import run_wikitext2_perplexity
from watersic.models.registry import load_model_and_tokenizer
from watersic.quant.watersic_layer import LayerQuantizationConfig
from watersic.quant.watersic_model import SequentialQuantizationConfig, quantize_model_sequential
from watersic.report.render_markdown import render_run_report_markdown
from watersic.utils.io import save_json
from watersic.utils.runtime import git_commit_hash, sanitize_name, utc_timestamp, write_report_bundle


def build_layer_config(quant_config: dict) -> LayerQuantizationConfig:
    layer_cfg = quant_config.get("layer", {})
    return LayerQuantizationConfig(
        target_rate=quant_config["target_global_bitwidth"],
        damping=float(layer_cfg.get("damping", 1e-4)),
        binary_search_iterations=int(layer_cfg.get("binary_search_iterations", 30)),
        row_sample_fraction=float(layer_cfg.get("row_sample_fraction", 0.1)),
        golden_section_iterations=int(layer_cfg.get("golden_section_iterations", 15)),
        dead_feature_tau=float(layer_cfg.get("dead_feature_tau", 1e-3)),
        epsilon_qr=float(layer_cfg.get("epsilon_qr", 0.0)),
        epsilon_aw=float(layer_cfg.get("epsilon_aw", 0.0)),
        max_rescaler_iters=int(layer_cfg.get("max_rescaler_iters", 0)),
        rescaler_ridge=float(layer_cfg.get("rescaler_ridge", 1e-8)),
        seed=int(layer_cfg.get("seed", 0)),
        use_lmmse=bool(layer_cfg.get("use_lmmse", True)),
        use_activation_drift=bool(layer_cfg.get("use_activation_drift", True)),
        use_residual_correction=bool(layer_cfg.get("use_residual_correction", True)),
        residual_scale=float(layer_cfg.get("residual_scale", 1.0)),
        use_attention_weighting=bool(layer_cfg.get("use_attention_weighting", True)),
        use_adaptive_mixing=bool(layer_cfg.get("use_adaptive_mixing", True)),
        spacing_strategy=str(layer_cfg.get("spacing_strategy", "watersic")),
    )


def build_calibration_config(quant_config: dict) -> CalibrationConfig:
    calib_cfg = quant_config.get("calibration", {})
    return CalibrationConfig(
        split=calib_cfg.get("split", "train"),
        sequence_length=int(calib_cfg.get("sequence_length", 2048)),
        num_sequences=int(calib_cfg.get("num_sequences", 32)),
        batch_size=int(calib_cfg.get("batch_size", 1)),
    )


def build_sequential_config(quant_config: dict) -> SequentialQuantizationConfig:
    return SequentialQuantizationConfig(
        run_name=quant_config["run_name"],
        target_global_bitwidth=float(quant_config["target_global_bitwidth"]),
        calibration_batch_size=int(quant_config.get("calibration", {}).get("batch_size", 1)),
        reference_stats=bool(quant_config.get("reference_stats", False)),
        max_layers=quant_config.get("max_layers"),
        max_modules=quant_config.get("max_modules"),
        layer_config=build_layer_config(quant_config),
    )


def build_eval_dataset(tokenizer, eval_config: dict):
    return load_wikitext2_blocks(
        tokenizer,
        split=eval_config.get("split", "test"),
        sequence_length=int(eval_config.get("sequence_length", 2048)),
        limit_sequences=eval_config.get("num_sequences"),
    )


def run_full_experiment(
    model_config: dict,
    quant_config: dict,
    eval_config: dict,
    *,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.to(device)
    model.eval()

    calibration_config = build_calibration_config(quant_config)
    calibration_dataset = load_calibration_dataset(tokenizer, calibration_config)
    eval_dataset = build_eval_dataset(tokenizer, eval_config)

    logger.info("Running BF16/FP baseline evaluation")
    baseline = run_wikitext2_perplexity(
        model,
        eval_dataset,
        device=str(device),
        batch_size=int(eval_config.get("batch_size", 1)),
    )

    reference_model = None
    reference_device = None
    if quant_config.get("reference_stats", False):
        logger.info("Loading reference model for original-vs-quantized statistics")
        reference_model, _ = load_model_and_tokenizer(model_config)
        reference_device = torch.device(quant_config.get("reference_device", str(device)))
        reference_model.to(reference_device)
        reference_model.eval()

    sequential_config = build_sequential_config(quant_config)
    report_metadata = {
        "timestamp": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "environment_name": "watersic",
        "model_id": model_config["model_id"],
        "model_revision": model_config.get("model_revision"),
        "tokenizer_id": model_config.get("tokenizer_id", model_config["model_id"]),
        "tokenizer_revision": model_config.get("tokenizer_revision", model_config.get("model_revision")),
        "quant_config_path": quant_config.get("_config_path", ""),
        "eval_config_path": eval_config.get("_config_path", ""),
        "sequence_length": calibration_config.sequence_length,
        "calibration_sequences": calibration_config.num_sequences,
        "notes": [],
    }
    run_report, artifact_dir = quantize_model_sequential(
        model,
        tokenizer,
        calibration_dataset,
        config=sequential_config,
        device=device,
        logger=logger,
        report_metadata=report_metadata,
        reference_model=reference_model,
        reference_device=reference_device,
    )

    logger.info("Running quantized evaluation")
    quantized_eval = run_wikitext2_perplexity(
        model,
        eval_dataset,
        device=str(device),
        batch_size=int(eval_config.get("batch_size", 1)),
    )
    run_report.perplexity = quantized_eval.perplexity
    run_report.extras["baseline_perplexity"] = baseline.perplexity
    run_report.extras["artifact_dir"] = str(artifact_dir)
    run_report.extras["reference_device"] = str(reference_device) if reference_device is not None else None
    run_report.extras["rescalers_enabled"] = bool(sequential_config.layer_config.max_rescaler_iters > 0)
    run_report.extras["runtime_seconds_total"] = time.perf_counter() - run_start
    if device.type == "cuda":
        run_report.extras["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    final_report = run_report.to_dict()
    save_json(Path(artifact_dir) / "metadata.json", final_report)
    markdown = render_run_report_markdown(run_report)
    report_json_path, report_md_path = write_report_bundle(quant_config["run_name"], final_report, markdown)
    return {
        "artifact_dir": artifact_dir,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "baseline_perplexity": baseline.perplexity,
        "quantized_perplexity": quantized_eval.perplexity,
    }


def benchmark_saved_model(model_path: str | Path, eval_config: dict, *, device: torch.device, logger) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    eval_dataset = build_eval_dataset(tokenizer, eval_config)
    result = run_wikitext2_perplexity(
        model,
        eval_dataset,
        device=str(device),
        batch_size=int(eval_config.get("batch_size", 1)),
    )
    logger.info("Perplexity for %s: %.4f", model_path, result.perplexity)
    return {"perplexity": result.perplexity, "dataset_split": result.dataset_split, "sequence_length": result.sequence_length}


def export_calibration(model_config: dict, quant_config: dict, *, output_path: str | Path) -> Path:
    _, tokenizer = load_model_and_tokenizer(model_config, device_map="cpu")
    calibration_config = build_calibration_config(quant_config)
    calibration_dataset = load_calibration_dataset(tokenizer, calibration_config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": calibration_dataset.input_ids,
            "split": calibration_dataset.split,
            "sequence_length": calibration_dataset.sequence_length,
            "num_sequences": len(calibration_dataset),
        },
        output_path,
    )
    return output_path


def download_model_snapshot(model_config: dict, *, output_dir: str | Path, logger) -> Path:
    model, tokenizer = load_model_and_tokenizer(model_config, device_map="cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved model snapshot to %s", output_dir)
    return output_dir
