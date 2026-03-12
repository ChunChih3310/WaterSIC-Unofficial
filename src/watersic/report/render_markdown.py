from __future__ import annotations

from watersic.report.schema import RunReport


def render_run_report_markdown(report: RunReport) -> str:
    lines = [
        f"# WaterSIC Run Report",
        "",
        f"- Timestamp: `{report.timestamp}`",
        f"- Git commit: `{report.git_commit}`",
        f"- Environment: `{report.environment_name}`",
        f"- Device: `{report.device}`",
        f"- Model: `{report.model_id}`",
        f"- Model revision: `{report.model_revision}`",
        f"- Tokenizer: `{report.tokenizer_id}`",
        f"- Sequence length: `{report.sequence_length}`",
        f"- Calibration sequences: `{report.calibration_sequences}`",
        f"- Target global bitwidth: `{report.target_global_bitwidth:.4f}`",
        f"- Achieved global bitwidth: `{report.achieved_global_bitwidth:.4f}`",
        f"- Raw average bitwidth: `{report.raw_average_bitwidth:.4f}`",
        f"- Entropy average bitwidth: `{report.entropy_average_bitwidth:.4f}`",
        f"- Huffman average bitwidth: `{report.huffman_average_bitwidth:.4f}`",
        f"- Side-information overhead: `{report.side_information_overhead:.4f}`",
        f"- Perplexity: `{report.perplexity}`",
        "",
        "## Layer Summary",
        "",
        "| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in report.layers:
        lines.append(
            f"| {layer.name} | {layer.kind} | {layer.target_bitwidth:.4f} | {layer.achieved_bitwidth:.4f} "
            f"| {layer.entropy_bitwidth:.4f} | {layer.huffman_bitwidth:.4f} | {layer.side_information_bitwidth:.4f} "
            f"| {layer.weighted_error:.6e} |"
        )
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"
