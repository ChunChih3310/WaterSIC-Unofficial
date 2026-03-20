# Debug Artifact Cleanup Inventory

## Scope and Rule

This is an inventory and recommendation report only.

- No files were deleted.
- No files were moved.
- No artifacts were rewritten.

The goal is to identify conservative cleanup candidates while protecting:

- the final paper-scale result
- the best validated calibration-sweep results
- reproducibility of the mainline path
- core code, tests, and documentation

## 1. Likely Safe To Delete Later

| Path | Category | Why it looks removable | Risk | Tied to | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `outputs/logs/run_watersic_default_20260317_171436.log` | redundant log | short failed launcher/default-path attempt, superseded by later model-specific runs | low | broken path | delete later |
| `outputs/logs/run_watersic_default_20260317_171527.log` | redundant log | same as above | low | broken path | delete later |
| `outputs/logs/run_watersic_default_20260317_172628.log` | redundant log | same as above | low | broken path | delete later |
| `outputs/logs/run_watersic_default_20260317_173040.log` | redundant log | same as above | low | broken path | delete later |
| `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64.out` | redundant launcher log | tiny launcher stdout stub, superseded by the successful main run log | low | redundant log | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_014607.log` | failed relaunch log | short failed relaunch before correct final calib64 run | low | broken path | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_015216.log` | failed relaunch log | short failed relaunch before correct final calib64 run | low | broken path | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_015325.log` | failed relaunch log | short failed relaunch before correct final calib64 run | low | broken path | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_015728.log` | failed relaunch log | tiny failed relaunch log | low | broken path | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_023347.log` | failed relaunch log | short failed relaunch log | low | broken path | delete later |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_calib64_20260317_024023.log` | failed relaunch log | short failed relaunch log | low | broken path | delete later |
| `outputs/reports/full_llama32_1b_batchsize_runtime_estimate_probe.json` | temporary probe output | raw support file for the runtime-estimate note; final markdown report already summarizes the conclusion | low | temporary debug | delete later |

## 2. Archive Rather Than Delete

These items are likely not needed for the final mainline paper result, but they still document important debugging or milestone history.

| Path | Category | Why archive instead of delete | Risk | Tied to | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `outputs/reports/llama32_1b_layer0_attention_debug/` | debug report bundle | records the staged `A` through `H` core-math recovery ladder | medium | temporary debug | archive |
| `outputs/reports/layer0_attention_debug_report.md` | debug summary | human-readable summary of the same recovery ladder | medium | temporary debug | archive |
| `outputs/reports/llama32_1b_layer0_attention_refsafe_large/` | debug report bundle | documents the first larger `reference_stats=true` sanity check | medium | temporary debug | archive |
| `outputs/reports/post_core_fix_validation_report.md` | debug summary | summary of the larger layer-0 safe-path validation | medium | temporary debug | archive |
| `outputs/reports/llama32_1b_layer1_o_proj_residual_debug/` | debug report bundle | exact residual-compensation fault-localization evidence | medium | temporary debug | archive |
| `outputs/reports/o_proj_residual_debug_report.md` | debug summary | narrow residual-path correction audit | medium | temporary debug | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix.json` | superseded intermediate report | pre-fix smoke run on the broken residual path | medium | broken path | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix.md` | superseded intermediate report | same as above | medium | broken path | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix_noresid.json` | diagnostic ablation | proved residual correction was the blocker | medium | temporary debug | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix_noresid.md` | diagnostic ablation | same as above | medium | temporary debug | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix_residfixed.json` | milestone report | first smoke after the residual-formula fix | medium | superseded path | archive |
| `outputs/reports/llama32_1b_multilayer_smoke_3p0bit_ref_stagefix_residfixed.md` | milestone report | same as above | medium | superseded path | archive |
| `outputs/reports/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.json` | milestone report | first repaired-mixing prefix validation | medium | superseded path | archive |
| `outputs/reports/llama32_1b_prefix2_3p0bit_reftrue_rescaler_mixing_repaircheck_v2.md` | milestone report | same as above | medium | superseded path | archive |
| `outputs/reports/full_llama32_1b_paperscale_launch_prep.md` | pre-launch note | useful for forensics, but superseded by the completed paper-scale run | medium | redundant report | archive |
| `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_155232.log` | launcher log | idle-wait launcher history before the final successful run | medium | redundant log | archive |
| `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_155306.log` | launcher log | same as above | medium | redundant log | archive |
| `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.out` | launcher stdout | same as above | medium | redundant log | archive |
| `outputs/logs/launcher_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_pin6.out` | launcher stdout | explicit pin-6 launcher capture, useful for device-placement history but not needed for final result | medium | redundant log | archive |
| `outputs/logs/run_llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_162038.log` | interrupted long-run log | contains interrupted paper-scale progress history; useful for forensics, not needed for final benchmark | medium | redundant log | archive |

## 3. Review Manually Before Deleting

These are plausible cleanup candidates, but they are tied to reproducibility history, future model work, or multiple reports that may still be useful.

| Path | Category | Why it needs review | Risk | Tied to | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `outputs/reports/aggregate_report.md` | redundant report | unclear whether it is still referenced by README/docs | review manually | redundant report | review manually |
| `outputs/reports/final_overnight_report.md` | redundant report | title suggests an intermediate summary; check whether anything links to it | review manually | superseded path | review manually |
| `outputs/reports/adaptive_mixing_mismatch_diagnosis.md` | technical report | still relevant if adaptive mixing is revisited on other models | review manually | superseded path | review manually |
| `outputs/reports/adaptive_mixing_paper_audit.md` | technical report | still useful as implementation provenance | review manually | superseded path | review manually |
| `outputs/reports/adaptive_mixing_runtime_optimization.md` | technical report | still relevant if paper-scale runtime needs to be revisited | review manually | superseded path | review manually |
| `outputs/reports/full_llama32_1b_batchsize_runtime_estimate.md` | estimate report | still useful for future long runs, but superseded for the completed `Llama-3.2-1B` paper result | review manually | redundant report | review manually |
| `outputs/reports/full_llama32_1b_paperscale_runtime_estimate.md` | estimate report | still relevant for future model launches | review manually | redundant report | review manually |
| `outputs/reports/full_llama32_1b_adaptive_mixing_upgrade_report.md` | milestone report | documents the first failed adaptive-mixing full-model attempt | review manually | superseded path | review manually |
| `outputs/reports/full_llama32_1b_adaptive_mixing_repair_report.md` | milestone report | documents the repaired search logic and prefix/full results | review manually | superseded path | review manually |
| `outputs/reports/full_llama32_1b_rescaler_validation_report.md` | milestone report | important for quality-recovery history | review manually | superseded path | review manually |
| `outputs/reports/full_llama32_1b_calibration_sweep_report.md` | milestone report | tied to the best validated calibration sweep and still useful | review manually | superseded path | review manually |
| `outputs/logs/run_llama31_8b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_173706.log` | other-model log | future `Llama-3.1-8B` work may still use it | review manually | stale config / other model | review manually |
| `outputs/logs/run_llama31_8b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_174448.log` | other-model log | same as above | review manually | stale config / other model | review manually |
| `outputs/logs/run_llama31_8b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_20260317_175228.log` | other-model log | same as above | review manually | stale config / other model | review manually |
| `outputs/logs/run_qwen3_8b_full_3p125bit_20260318_024606.log` | other-model log | `Qwen3-8B` remains intentionally deferred, so do not discard evidence casually | review manually | stale config / other model | review manually |
| `outputs/logs/run_qwen3_8b_full_3p125bit_reftrue_rescaler_mixing_repaired_paperscale_20260318_035338.log` | other-model log | same as above | review manually | stale config / other model | review manually |
| `outputs/logs/run_qwen3_8b_full_3p125bit_reftrue_rescaler_mixing_repaired_paperscale_20260318_042003.log` | other-model log | same as above | review manually | stale config / other model | review manually |
| `configs/debug/llama32_1b_layer0_attention.yaml` | stale debug config | likely no longer needed for mainline work, but useful if the core-math ladder must be rerun | review manually | temporary debug | review manually |
| `configs/debug/llama32_1b_layer0_attention_refsafe_large.yaml` | stale debug config | same as above | review manually | temporary debug | review manually |
| `configs/debug/llama32_1b_layer1_o_proj_residual.yaml` | stale debug config | same as above | review manually | temporary debug | review manually |
| `scripts/debug_layer0_attention.py` | one-off debug script | probably archive-worthy, but still useful if the core attention ladder must be reproduced | review manually | temporary debug | review manually |
| `scripts/debug_o_proj_residual.py` | one-off debug script | same as above | review manually | temporary debug | review manually |
| `outputs/quantized/llama32_1b_full_3p0bit_reftrue_norescaler` | superseded artifact | useful provenance for the first stable full-model point | review manually | superseded path | review manually |
| `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler` | superseded artifact | useful provenance for the first beneficial rescaler point | review manually | superseded path | review manually |
| `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing` | superseded artifact | records the old harmful adaptive-mixing full-model result | review manually | superseded path | review manually |
| `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired` | superseded artifact | records the repaired-but-still-worse mixing result | review manually | superseded path | review manually |

## 4. Definitely Keep

These should not be treated as cleanup candidates.

| Path | Why keep |
| --- | --- |
| `outputs/quantized/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale` | final best `Llama-3.2-1B` paper-scale artifact |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.json` | machine-readable final paper-scale result |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale.md` | human-readable final paper-scale result |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_validation_benchmark.json` | strict validation benchmark for paper comparison |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_mixing_repaired_paperscale_validation_benchmark.md` | human-readable validation benchmark |
| `outputs/reports/full_llama32_1b_adaptive_mixing_paperscale_report.md` | final paperscale comparison report |
| `outputs/reports/full_llama32_1b_quality_recovery_comparison.md` | main comparison report across all key paths |
| `outputs/reports/final_paper_audit.md` | final paper-audit record |
| `outputs/reports/final_worst_layer_diagnosis.md` | final distortion diagnosis |
| `outputs/reports/debug_artifact_cleanup_inventory.md` | cleanup inventory itself |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.json` | key calibration-sweep provenance |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib16.md` | key calibration-sweep provenance |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.json` | key calibration-sweep provenance |
| `outputs/reports/llama32_1b_full_3p0bit_reftrue_rescaler_calib32.md` | key calibration-sweep provenance |
| `outputs/reports/full_llama32_1b_calibration_sweep_report.md` | calibration-sweep narrative and comparisons |
| `outputs/reports/full_llama32_1b_adaptive_mixing_calib64_report.md` | key step in reaching the final best path |
| `docs/reproduction_log.md` | primary chronological provenance |
| `docs/known_issues.md` | current status and caveats |
| `configs/paper_comparable/models/llama32_1b.yaml` | paper-comparable model config |
| `configs/paper_comparable/quant/watersic_llama32_1b_paperscale.yaml` | paper-comparable quant config |
| `configs/eval/wikitext2_validation.yaml` | strict validation benchmark config |
| `tests/` | core correctness and regression coverage |
| `src/` | implementation itself |

## High-Level Recommendation

If cleanup happens later, use this order:

1. delete the clearly redundant short failed logs first
2. archive the debug ladders, interrupted long-run logs, and superseded smoke/milestone bundles
3. review other-model logs and historical milestone artifacts manually before deleting anything
4. keep the final paper-scale result, validation benchmark, calibration-sweep milestones, code, tests, and docs

## Current Status

- Cleanup inventory complete: yes
- Any deletion performed: no
- Any file moved or rewritten for cleanup: no
