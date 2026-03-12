from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from watersic.models.hooks import ModuleInputCollector
from watersic.models.registry import LinearModuleSpec, group_specs_by_layer, iter_linear_module_specs
from watersic.quant.serialization import save_quantized_artifact
from watersic.report.render_markdown import render_run_report_markdown
from watersic.report.schema import LayerReport, RunReport
from watersic.stats.attention_weighting import is_attention_weighted_target, token_importance_from_attention, weighted_second_moment
from watersic.stats.covariance import SecondMomentAccumulator
from watersic.utils.io import save_json
from watersic.utils.path_guard import ensure_parent_dir, repo_path

from .watersic_layer import LayerQuantizationConfig, LayerQuantizationResult, LayerStatistics, quantize_linear_layer


@dataclass(frozen=True)
class SequentialQuantizationConfig:
    run_name: str
    target_global_bitwidth: float
    calibration_batch_size: int = 1
    reference_stats: bool = False
    max_layers: int | None = None
    max_modules: int | None = None
    layer_config: LayerQuantizationConfig = LayerQuantizationConfig(target_rate=4.0)


def _layer_stats_template(spec: LinearModuleSpec, hidden_dim: int, input_dim: int) -> dict[str, SecondMomentAccumulator]:
    template: dict[str, SecondMomentAccumulator] = {
        "sigma_x": SecondMomentAccumulator(input_dim),
        "sigma_x_hat": SecondMomentAccumulator(input_dim),
        "sigma_x_x_hat": SecondMomentAccumulator(input_dim, input_dim),
    }
    if is_attention_weighted_target(spec.kind):
        template["sigma_x_weighted"] = SecondMomentAccumulator(input_dim)
        template["sigma_x_hat_weighted"] = SecondMomentAccumulator(input_dim)
        template["sigma_x_x_hat_weighted"] = SecondMomentAccumulator(input_dim, input_dim)
    if spec.kind in {"o_proj", "down_proj"}:
        template["sigma_delta_x_hat"] = SecondMomentAccumulator(hidden_dim, input_dim)
    return template


def collect_layer_statistics(
    model,
    calibration_dataset,
    layer_specs: list[LinearModuleSpec],
    *,
    device: torch.device,
    logger,
    calibration_batch_size: int,
    reference_model=None,
    reference_device: torch.device | None = None,
) -> dict[str, LayerStatistics]:
    module_paths = [spec.full_path for spec in layer_specs]
    layer_path = layer_specs[0].layer_path
    hidden_dim = model.get_submodule(layer_path).input_layernorm.weight.numel()
    accumulators: dict[str, dict[str, SecondMomentAccumulator]] = {}
    for spec in layer_specs:
        module = model.get_submodule(spec.full_path)
        accumulators[spec.full_path] = _layer_stats_template(spec, hidden_dim=hidden_dim, input_dim=module.in_features)

    collect_attentions = any(is_attention_weighted_target(spec.kind) for spec in layer_specs)
    for batch_index, batch in enumerate(calibration_dataset.batches(calibration_batch_size)):
        batch = batch.to(device)
        with ModuleInputCollector(model, module_paths=module_paths, layer_paths=[layer_path]) as q_store:
            outputs_q = model(input_ids=batch, use_cache=False, output_attentions=collect_attentions)
        q_layer_input = q_store.layer_inputs[layer_path]

        ref_store = None
        ref_layer_input = None
        ref_attn = None
        if reference_model is not None:
            with ModuleInputCollector(reference_model, module_paths=module_paths, layer_paths=[layer_path]) as ref_store:
                outputs_ref = reference_model(input_ids=batch.to(reference_device), use_cache=False, output_attentions=collect_attentions)
            ref_layer_input = ref_store.layer_inputs[layer_path]
            ref_attn = outputs_ref.attentions[layer_specs[0].layer_index].detach().cpu() if collect_attentions else None
        q_attn = outputs_q.attentions[layer_specs[0].layer_index].detach().cpu() if collect_attentions else None
        token_weights = token_importance_from_attention(ref_attn if ref_attn is not None else q_attn) if collect_attentions else None

        for spec in layer_specs:
            q_input = q_store.inputs[spec.full_path]
            ref_input = q_input if ref_store is None else ref_store.inputs[spec.full_path]
            acc = accumulators[spec.full_path]
            acc["sigma_x"].update(ref_input)
            acc["sigma_x_hat"].update(q_input)
            acc["sigma_x_x_hat"].update(ref_input, q_input)
            if "sigma_x_weighted" in acc and token_weights is not None:
                acc["sigma_x_weighted"].update(ref_input, weights=token_weights)
                acc["sigma_x_hat_weighted"].update(q_input, weights=token_weights)
                acc["sigma_x_x_hat_weighted"].update(ref_input, q_input, weights=token_weights)
            if "sigma_delta_x_hat" in acc and ref_layer_input is not None:
                delta = ref_layer_input - q_layer_input
                acc["sigma_delta_x_hat"].update(delta, q_input)

        del outputs_q
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Collected calibration batch %d for layer %s", batch_index, layer_path)

    finalized: dict[str, LayerStatistics] = {}
    for spec in layer_specs:
        acc = accumulators[spec.full_path]
        sigma_delta = acc["sigma_delta_x_hat"].finalize() if "sigma_delta_x_hat" in acc else None
        sigma_x_hat = acc["sigma_x_hat"].finalize()
        finalized[spec.full_path] = LayerStatistics(
            sigma_x=acc["sigma_x"].finalize(),
            sigma_x_hat=sigma_x_hat,
            sigma_x_x_hat=acc["sigma_x_x_hat"].finalize(),
            sigma_x_weighted=acc["sigma_x_weighted"].finalize() if "sigma_x_weighted" in acc else None,
            sigma_x_hat_weighted=acc["sigma_x_hat_weighted"].finalize() if "sigma_x_hat_weighted" in acc else None,
            sigma_x_x_hat_weighted=acc["sigma_x_x_hat_weighted"].finalize() if "sigma_x_x_hat_weighted" in acc else None,
            sigma_delta_x_hat=sigma_delta,
            variances=torch.diag(sigma_x_hat),
        )
    return finalized


def quantize_model_sequential(
    model,
    tokenizer,
    calibration_dataset,
    *,
    config: SequentialQuantizationConfig,
    device: torch.device,
    logger,
    report_metadata: dict[str, Any],
    reference_model=None,
    reference_device: torch.device | None = None,
) -> tuple[RunReport, Path]:
    specs = iter_linear_module_specs(model)
    grouped = group_specs_by_layer(specs)
    layer_indices = sorted(grouped)
    if config.max_layers is not None:
        layer_indices = layer_indices[: config.max_layers]

    selected_specs = [spec for layer_idx in layer_indices for spec in grouped[layer_idx]]
    if config.max_modules is not None:
        selected_specs = selected_specs[: config.max_modules]
        filtered_grouped: dict[int, list[LinearModuleSpec]] = defaultdict(list)
        for spec in selected_specs:
            filtered_grouped[spec.layer_index].append(spec)
        grouped = filtered_grouped
        layer_indices = sorted(grouped)

    total_weights = sum(model.get_submodule(spec.full_path).weight.numel() for spec in selected_specs)
    remaining_budget = config.target_global_bitwidth * total_weights
    remaining_weights = total_weights
    layer_reports: list[LayerReport] = []
    layer_results: dict[str, dict[str, Any]] = {}

    for layer_index in layer_indices:
        layer_specs = grouped[layer_index]
        logger.info("Collecting statistics for layer %d", layer_index)
        stats_map = collect_layer_statistics(
            model,
            calibration_dataset,
            layer_specs,
            device=device,
            logger=logger,
            calibration_batch_size=config.calibration_batch_size,
            reference_model=reference_model,
            reference_device=reference_device,
        )
        for spec in layer_specs:
            module = model.get_submodule(spec.full_path)
            num_weights = module.weight.numel()
            target_rate = remaining_budget / max(remaining_weights, 1)
            layer_config = LayerQuantizationConfig(
                target_rate=target_rate,
                damping=config.layer_config.damping,
                binary_search_iterations=config.layer_config.binary_search_iterations,
                row_sample_fraction=config.layer_config.row_sample_fraction,
                golden_section_iterations=config.layer_config.golden_section_iterations,
                dead_feature_tau=config.layer_config.dead_feature_tau,
                epsilon_qr=config.layer_config.epsilon_qr,
                epsilon_aw=config.layer_config.epsilon_aw,
                max_rescaler_iters=config.layer_config.max_rescaler_iters,
                rescaler_ridge=config.layer_config.rescaler_ridge,
                seed=config.layer_config.seed,
            )
            result = quantize_linear_layer(
                module.weight.detach().cpu(),
                stats_map[spec.full_path],
                layer_config,
                kind=spec.kind,
            )
            module.weight.data.copy_(result.quantized_weight.to(module.weight.device, dtype=module.weight.dtype))
            remaining_budget -= result.bitrate.final_effective_average_bitwidth * num_weights
            remaining_weights -= num_weights
            layer_reports.append(
                LayerReport(
                    name=spec.full_path,
                    kind=spec.kind,
                    target_bitwidth=target_rate,
                    achieved_bitwidth=result.bitrate.final_effective_average_bitwidth,
                    raw_bitwidth=result.bitrate.raw_average_bitwidth,
                    entropy_bitwidth=result.bitrate.entropy_average_bitwidth,
                    huffman_bitwidth=result.bitrate.huffman_average_bitwidth,
                    side_information_bitwidth=result.bitrate.side_information_average_bitwidth,
                    weighted_error=result.search.quantization.weighted_error,
                    applied_damping=result.applied_damping,
                )
            )
            layer_results[spec.full_path] = {
                "kind": spec.kind,
                "target_rate": target_rate,
                "achieved_rate": result.bitrate.final_effective_average_bitwidth,
                "raw_rate": result.bitrate.raw_average_bitwidth,
                "entropy_rate": result.bitrate.entropy_average_bitwidth,
                "huffman_rate": result.bitrate.huffman_average_bitwidth,
                "side_information_rate": result.bitrate.side_information_average_bitwidth,
                "selected_c": result.search.selected_c,
                "spacings": [float(x) for x in result.spacings.tolist()],
                "lmmse_gammas": [float(x) for x in result.lmmse_gammas.tolist()],
                "row_scale_shape": list(result.row_scale.shape),
                "column_scale_shape": list(result.column_scale.shape),
                "num_dead_features": int((~result.dead_features.keep_mask).sum().item()),
                "weighted_error": result.search.quantization.weighted_error,
            }
            logger.info(
                "Quantized %s with target %.4f and achieved %.4f",
                spec.full_path,
                target_rate,
                result.bitrate.final_effective_average_bitwidth,
            )

    achieved_global = sum(layer.achieved_bitwidth * model.get_submodule(layer.name).weight.numel() for layer in layer_reports) / max(total_weights, 1)
    raw_global = sum(layer.raw_bitwidth * model.get_submodule(layer.name).weight.numel() for layer in layer_reports) / max(total_weights, 1)
    entropy_global = sum(layer.entropy_bitwidth * model.get_submodule(layer.name).weight.numel() for layer in layer_reports) / max(total_weights, 1)
    huffman_global = sum(layer.huffman_bitwidth * model.get_submodule(layer.name).weight.numel() for layer in layer_reports) / max(total_weights, 1)
    side_global = sum(layer.side_information_bitwidth * model.get_submodule(layer.name).weight.numel() for layer in layer_reports) / max(total_weights, 1)

    run_report = RunReport(
        timestamp=report_metadata["timestamp"],
        git_commit=report_metadata["git_commit"],
        environment_name=report_metadata["environment_name"],
        device=str(device),
        model_id=report_metadata["model_id"],
        model_revision=report_metadata.get("model_revision"),
        tokenizer_id=report_metadata["tokenizer_id"],
        tokenizer_revision=report_metadata.get("tokenizer_revision"),
        quant_config_path=report_metadata["quant_config_path"],
        eval_config_path=report_metadata.get("eval_config_path"),
        sequence_length=report_metadata["sequence_length"],
        calibration_sequences=report_metadata["calibration_sequences"],
        target_global_bitwidth=config.target_global_bitwidth,
        achieved_global_bitwidth=achieved_global,
        raw_average_bitwidth=raw_global,
        entropy_average_bitwidth=entropy_global,
        huffman_average_bitwidth=huffman_global,
        side_information_overhead=side_global,
        layers=layer_reports,
        notes=report_metadata.get("notes", []),
        extras={"layer_results": layer_results},
    )

    run_dir = repo_path("outputs", "quantized", config.run_name)
    artifact_dir = save_quantized_artifact(model, tokenizer, run_dir, metadata=run_report.to_dict())
    save_json(artifact_dir / "layer_results.json", layer_results)
    markdown_path = ensure_parent_dir(artifact_dir / "report.md")
    markdown_path.write_text(render_run_report_markdown(run_report), encoding="utf-8")
    return run_report, artifact_dir
