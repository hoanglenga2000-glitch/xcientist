#!/usr/bin/env python3
"""Build the 92-second evidence-mapped Chinese narration after recording release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from verify_pre_recording_gate import (
    CONTRACT_PATH,
    PROJECT_ROOT,
    VIDEO_ROOT,
    PreRecordingGateError,
    read_json,
    require,
    same_run,
    validate_gate,
)

OUTPUT_PATH = VIDEO_ROOT / "preproduction" / "narration-92s.json"
PROFILE_LABELS = {
    "raw_multiview_v1": "原始多视图",
    "raw_multiview": "原始多视图",
    "border_multiview_v1": "黑边裁剪",
    "color_multiview_v1": "颜色恒常",
    "hair_multiview_v1": "毛发抑制",
    "robust_multiview_v1": "稳健组合预处理",
    "robust_combined_multiview_v1": "稳健组合预处理",
    "robust_combined_pipeline": "稳健组合预处理",
    "raw": "原始多视图",
    "border_removal": "黑边裁剪",
    "color_constancy": "颜色恒常",
    "hair_suppression": "毛发抑制",
    "robust_combined": "稳健组合预处理",
}
REVIEW_PASSED_STATUSES = frozenset({"passed", "review_passed"})


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreRecordingGateError(f"missing narration metric: {label}")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise PreRecordingGateError(f"narration metric is outside 0..1: {label}")
    return result


def optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite(value, label)


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreRecordingGateError(f"missing narration count: {label}")
    result = int(value)
    if not math.isfinite(float(value)) or float(value) != result or result < 0:
        raise PreRecordingGateError(f"invalid narration count: {label}")
    return result


def validate_segments(run_dir: Path, segments: list[dict[str, Any]]) -> None:
    expected_start = 0.0
    for segment in segments:
        start = float(segment.get("start", -1))
        end = float(segment.get("end", -1))
        require(abs(start - expected_start) < 1e-9, f"narration timeline gap before {segment.get('id')}")
        require(end > start, f"narration segment has invalid duration: {segment.get('id')}")
        require(bool(str(segment.get("text") or "").strip()), f"narration segment is empty: {segment.get('id')}")
        evidence = segment.get("evidence")
        require(isinstance(evidence, list) and bool(evidence), f"narration evidence is empty: {segment.get('id')}")
        for relative in evidence:
            path = run_dir / str(relative)
            require(
                path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                f"missing narration evidence: {relative}",
            )
        expected_start = end
    require(abs(expected_start - 92.0) < 1e-9, "narration timeline is not exactly 92 seconds")


def build_manifest(project_root: Path, contract_path: Path) -> dict[str, Any]:
    gate = validate_gate(project_root, contract_path)
    run_id = str(gate["run_id"])
    run_dir = project_root.resolve() / "workspace" / "evomind_runs" / run_id
    dataset = read_json(run_dir / "dataset_profile.json", "dataset_profile.json")
    comparison = read_json(run_dir / "experiment_comparison.json", "experiment_comparison.json")
    metrics = read_json(run_dir / "metrics.json", "metrics.json")
    runtime = read_json(run_dir / "hpc_runtime.json", "hpc_runtime.json")
    review = read_json(run_dir / "review.json", "review.json")
    claim = read_json(run_dir / "claim_audit.json", "claim_audit.json")
    grader = read_json(run_dir / "private_grader.json", "private_grader.json")
    for label, payload in (
        ("dataset_profile.json", dataset),
        ("experiment_comparison.json", comparison),
        ("metrics.json", metrics),
        ("hpc_runtime.json", runtime),
        ("review.json", review),
        ("claim_audit.json", claim),
        ("private_grader.json", grader),
    ):
        same_run(payload, run_id, label)
    counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else dataset
    confidence = metrics.get("patient_grouped_bootstrap_roc_auc_95ci")
    confidence = confidence if isinstance(confidence, dict) else {}
    thresholds = metrics.get("historical_thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    progress = runtime.get("training_progress") if isinstance(runtime.get("training_progress"), dict) else {}
    selected = str(comparison.get("selected_profile") or "")
    require(selected in PROFILE_LABELS, "narration preprocessing profile is unknown")
    selected_label = PROFILE_LABELS[selected]
    values = {
        "files": integer(counts.get("files"), "files"),
        "train_images": integer(counts.get("train_images"), "train_images"),
        "test_images": integer(counts.get("test_images"), "test_images"),
        "positive_rows": integer(counts.get("positive_rows"), "positive_rows"),
        "patients": integer(counts.get("patients"), "patients"),
        "positive_rate": finite(dataset.get("positive_rate"), "positive_rate"),
        "roc_auc": finite(metrics.get("roc_auc"), "roc_auc"),
        "pr_auc": finite(metrics.get("pr_auc"), "pr_auc"),
        "ci_lower": finite(confidence.get("lower"), "ci_lower"),
        "ci_upper": finite(confidence.get("upper"), "ci_upper"),
        "grader_score": optional_finite(
            grader.get("mle_private_grader_score", grader.get("score")),
            "grader_score",
        ),
        "grader_status": str(grader.get("status") or ""),
        "bronze_threshold": finite(thresholds.get("bronze"), "bronze_threshold"),
        "silver_threshold": finite(thresholds.get("silver"), "silver_threshold"),
        "gold_threshold": finite(thresholds.get("gold"), "gold_threshold"),
        "formal_seeds": integer(progress.get("completed_formal_seeds"), "formal_seeds"),
        "outer_folds": integer(progress.get("completed_outer_folds"), "outer_folds"),
        "selected_profile": selected_label,
    }
    if review.get("status") not in REVIEW_PASSED_STATUSES or claim.get("status") != "passed":
        raise PreRecordingGateError("narration cannot describe an unreviewed Run")
    require(progress.get("status") == "completed", "narration runtime is not completed")
    require(values["train_images"] > 0, "narration training image count is empty")
    require(0 < values["positive_rows"] <= values["train_images"], "narration positive count is invalid")
    require(
        abs(values["positive_rate"] - values["positive_rows"] / values["train_images"]) < 1e-12,
        "narration positive rate does not match dataset counts",
    )
    require(values["ci_lower"] <= values["roc_auc"] <= values["ci_upper"], "narration ROC-AUC is outside its confidence interval")
    require(values["formal_seeds"] > 0 and values["outer_folds"] > 0, "narration training progress is empty")
    require(
        values["bronze_threshold"] <= values["silver_threshold"] <= values["gold_threshold"],
        "narration historical medal thresholds are not ordered",
    )
    if values["grader_status"] == "failed_closed":
        require(values["grader_score"] is None, "failed-closed grader unexpectedly contains a score")
        medal_reference = "终局私有评分失败关闭，未产出可比较分数"
    elif values["grader_score"] is None:
        raise PreRecordingGateError("terminal private grader score is missing")
    elif values["grader_score"] >= values["gold_threshold"]:
        medal_reference = "数值上达到历史金牌线"
    elif values["grader_score"] >= values["silver_threshold"]:
        medal_reference = "数值上达到历史银牌线"
    elif values["grader_score"] >= values["bronze_threshold"]:
        medal_reference = "数值上达到历史铜牌线"
    else:
        medal_reference = "仍低于历史铜牌线"
    values["historical_medal_reference"] = medal_reference

    segments = [
        {
            "id": "S01",
            "start": 0.0,
            "end": 8.0,
            "text": "第一次使用 EvoMind，我只要在智能助手里说清医疗影像研究问题、计算位置和需要交付的文件。",
            "evidence": ["request.json", "run.json"],
        },
        {
            "id": "S02",
            "start": 8.0,
            "end": 15.0,
            "text": "系统会把这句话转成数据、A800、分组验证、独立复核和四项交付的完整研究合同。",
            "evidence": ["workflow_contract.json", "task_graph.json"],
        },
        {
            "id": "S03",
            "start": 15.0,
            "end": 27.0,
            "text": f"这次共有 {values['files']:,} 个文件，恶性样本只占 {values['positive_rate']:.2%}。系统还把 {values['patients']:,} 名患者和重复内容组完全隔离，避免同一患者跨折泄漏。",
            "evidence": ["dataset_profile.json", "data_audit.json"],
        },
        {
            "id": "S04",
            "start": 27.0,
            "end": 39.0,
            "text": f"第一轮先建立原始多视图基线；第二轮让五种预处理在相同分组和三种种子上竞争，最终保留{values['selected_profile']}。没有稳定增益的改动会被主动拒绝。",
            "evidence": ["preprocessing_ablation.json", "experiment_comparison.json"],
        },
        {
            "id": "S05",
            "start": 39.0,
            "end": 53.0,
            "text": f"第三轮把全图、病灶视图和患者元数据融合，再完成 {values['formal_seeds']} 个正式种子、{values['outer_folds']} 个外层折的真实训练。这里回放的是同一 Run 的证据账本。",
            "evidence": ["training_history.json", "hpc_telemetry.jsonl", "hpc_runtime.json"],
        },
        {
            "id": "S06",
            "start": 53.0,
            "end": 65.0,
            "text": (
                f"第四轮聚合三种子并冻结候选。患者分组 ROC-AUC 为 {values['roc_auc']:.5f}，"
                f"百分之九十五置信区间为 {values['ci_lower']:.5f} 到 {values['ci_upper']:.5f}。"
                if values["grader_status"] == "failed_closed"
                else f"第四轮聚合三种子并冻结候选。患者分组 ROC-AUC 为 {values['roc_auc']:.5f}，终局离线评分为 {values['grader_score']:.5f}，{values['historical_medal_reference']}；这只是历史数值线对照。"
            ),
            "evidence": ["metrics.json", "fold_metrics.csv", "private_grader.json"],
        },
        {
            "id": "S07",
            "start": 65.0,
            "end": 76.0,
            "text": (
                "Independent Review 和 Claim Audit 都已通过。终局私有评分只执行一次并失败关闭，"
                "没有产出私有分数，也没有反馈调参或提交公开榜单。"
                if values["grader_status"] == "failed_closed"
                else "Independent Review 和 Claim Audit 都已通过。终局评分只执行一次，没有反馈给后续调参；本次也没有提交公开榜单。"
            ),
            "evidence": ["review.json", "candidate_freeze.json", "private_grader_ledger.json", "claim_audit.json"],
        },
        {
            "id": "S08",
            "start": 76.0,
            "end": 86.0,
            "text": "专业报告会同时展示 ROC、PR、概率校准、折间稳定性和元数据融合贡献。它是医学影像研究基准，不代表临床诊断能力。",
            "evidence": ["research_report.html", "evomind-siim-isic-report.pdf"],
        },
        {
            "id": "S09",
            "start": 86.0,
            "end": 92.0,
            "text": "最后可以直接下载 PDF 报告、结果 CSV、代码 ZIP 和证据 ZIP。一句话完成研究、复核和可追溯交付。",
            "evidence": ["deliverables.json", "artifact_manifest.json"],
        },
    ]
    validate_segments(run_dir, segments)
    return {
        "schema": "evomind.siim.video_narration_manifest.v1",
        "run_id": run_id,
        "status": "ready",
        "duration_seconds": 92.0,
        "language": "zh-CN",
        "voice": "single_female",
        "voice_count": 1,
        "background_music": False,
        "source_values": values,
        "segments": segments,
        "claim_boundary": "医学影像研究基准；独立离线评测；未提交公开榜单",
        "same_run_evidence": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.project_root, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "ready", "run_id": manifest["run_id"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
