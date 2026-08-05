#!/usr/bin/env python3
"""Build the reviewed SIIM-ISIC user delivery without inventing evidence.

The builder is deliberately downstream of training, candidate freeze, the one
terminal private-grader execution, Independent Review, and Claim Audit.  It
will not create a report from a partial run.  Every displayed number is either
recomputed from the frozen OOF predictions or read from a hash-bound reviewed
artifact belonging to the same EvoMind run.

Outputs are deterministic for identical inputs:

* ``research_report.html``
* ``evomind-siim-isic-report.pdf``
* ``evomind-siim-isic-results.csv``
* ``evomind-siim-isic-code.zip``
* ``evomind-siim-isic-evidence.zip``
* ``artifact_manifest.json``
* rendered PDF QA pages and ``qa/pdf-qa.json``

The script never submits to Kaggle and never reads private labels.  The only
terminal score it accepts is the already-recorded, post-freeze grader result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TASK_ID = "siim-isic-melanoma-classification"
EXPECTED = {
    "files": 33_129,
    "train_images": 28_984,
    "test_images": 4_142,
    "positive_rows": 513,
    "patients": 2_056,
}
PROFILE_ORDER = (
    "raw_multiview",
    "border_removal",
    "color_constancy",
    "hair_suppression",
    "robust_combined_pipeline",
)
PROFILE_LABELS = {
    "raw_multiview": "原始多视图",
    "border_removal": "边框去除",
    "color_constancy": "颜色恒常",
    "hair_suppression": "毛发抑制",
    "robust_combined_pipeline": "稳健组合流程",
}
DOWNLOAD_NAMES = (
    "evomind-siim-isic-report.pdf",
    "evomind-siim-isic-results.csv",
    "evomind-siim-isic-code.zip",
    "evomind-siim-isic-evidence.zip",
)
PDF_PAGE_COUNT = 9
REQUIRED_EVIDENCE = (
    "request.json",
    "run.json",
    "task_graph.json",
    "events.jsonl",
    "dataset_profile.json",
    "hpc_runtime.json",
    "data_audit.json",
    "research_design.json",
    "preprocessing_ablation.json",
    "experiment_comparison.json",
    "metrics.json",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "sample_submission.csv",
    "submission.csv",
    "training_history.json",
    "training_result.json",
    "hpc_telemetry.jsonl",
    "review.json",
    "candidate_freeze.json",
    "private_grader.json",
    "private_grader_ledger.json",
    "claim_audit.json",
)
DEFAULT_CODE_FILES = (
    "src/research_os/agent/siim_hpc_workflow.py",
    "scripts/run_siim_job89508_campaign.py",
    "scripts/manage_siim_job89508_campaign.py",
    "scripts/run_siim_preprocessing_ablation.py",
    "scripts/run_mlebench_lite_full.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/aggregate_siim_multiseed_candidate.py",
    "scripts/build_siim_delivery.py",
)
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".ps1",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "credential_literal",
        re.compile(
            rb"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)"
            rb"\s*[:=]\s*[\"'][^\"'\r\n]{6,}[\"']"
        ),
    ),
    ("credential_uri", re.compile(rb"(?i)[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]{4,}@")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9_]{30,}")),
)


class DeliveryBuildError(RuntimeError):
    """A fail-closed delivery gate rejected the available evidence."""


@dataclass(frozen=True)
class CsvTable:
    path: Path
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ValidatedRun:
    run_dir: Path
    run_id: str
    stable_time: str
    evidence_paths: Mapping[str, Path]
    payloads: Mapping[str, dict[str, Any]]
    metrics: Mapping[str, float]
    fold_metrics: tuple[dict[str, float], ...]
    oof_rows: tuple[dict[str, str], ...]
    submission_rows: tuple[dict[str, str], ...]
    roc_points: tuple[tuple[float, float], ...]
    pr_points: tuple[tuple[float, float], ...]
    calibration_points: tuple[tuple[float, float], ...]
    component_scores: tuple[tuple[str, float], ...]
    profile_scores: tuple[tuple[str, float], ...]
    selected_profile: str
    source_hashes: Mapping[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeliveryBuildError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, _canonical_json(payload))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeliveryBuildError(f"缺少真实证据：{label}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeliveryBuildError(f"证据不是有效 JSON：{label}") from exc
    _require(isinstance(payload, dict), f"证据必须为 JSON 对象：{label}")
    return payload


def _read_csv(path: Path, label: str) -> CsvTable:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except FileNotFoundError as exc:
        raise DeliveryBuildError(f"缺少真实证据：{label}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DeliveryBuildError(f"证据不是有效 CSV：{label}") from exc
    _require(bool(fields), f"CSV 缺少表头：{label}")
    return CsvTable(path=path, fields=fields, rows=rows)


def _finite(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DeliveryBuildError(f"{label} 不是数值") from exc
    _require(math.isfinite(number), f"{label} 不是有限数")
    if minimum is not None:
        _require(number >= minimum, f"{label} 小于 {minimum}")
    if maximum is not None:
        _require(number <= maximum, f"{label} 大于 {maximum}")
    return number


def _status_passed(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").strip().lower() in {
        "passed",
        "completed",
        "verified",
        "review_passed",
        "ready",
        "terminal_execution_recorded",
        "frozen_before_private_grader",
    }


def _locate_evidence(run_dir: Path, name: str) -> Path:
    if name == "claim_audit.json":
        immutable_source = run_dir / "claim_audit_source.json"
        if immutable_source.is_file():
            return immutable_source
    direct = run_dir / name
    if direct.is_file():
        return direct
    if name == "claim_audit.json":
        ingress = run_dir / "ingress" / name
        if ingress.is_file():
            return ingress
    raise DeliveryBuildError(f"缺少真实证据：{name}")


def _assert_same_run(payload: Mapping[str, Any], run_id: str, label: str, *, required: bool = True) -> None:
    value = payload.get("run_id")
    if value is None and not required:
        return
    _require(str(value or "") == run_id, f"{label} 不属于同一 Run")


def _safe_relative(run_dir: Path, record_path: Any) -> Path:
    relative = Path(str(record_path or ""))
    _require(bool(str(relative)) and not relative.is_absolute() and ".." not in relative.parts, "冻结路径越界")
    resolved = (run_dir / relative).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise DeliveryBuildError("冻结路径越界") from exc
    return resolved


def _normalize_profile(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _metric_from_record(record: Mapping[str, Any]) -> float:
    for key in ("mean_roc_auc", "roc_auc_mean", "mean_auc", "roc_auc", "score"):
        if record.get(key) is not None:
            return _finite(record[key], f"预处理方案 {record.get('profile') or record.get('name')} ROC-AUC", minimum=0, maximum=1)
    raise DeliveryBuildError("预处理消融缺少 ROC-AUC")


def _binary_curve(rows: Sequence[dict[str, str]], probability_key: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, float, float]:
    pairs: list[tuple[float, int]] = []
    for index, row in enumerate(rows):
        target = int(_finite(row.get("target"), f"OOF target row {index}", minimum=0, maximum=1))
        _require(target in {0, 1}, "OOF target 必须为 0/1")
        probability = _finite(row.get(probability_key), f"OOF probability row {index}", minimum=0, maximum=1)
        pairs.append((probability, target))
    positives = sum(target for _, target in pairs)
    negatives = len(pairs) - positives
    _require(positives > 0 and negatives > 0, "OOF 必须同时包含阳性与阴性")
    pairs.sort(key=lambda item: item[0], reverse=True)
    roc = [(0.0, 0.0)]
    pr = [(0.0, 1.0)]
    tp = fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        end = index
        group_tp = group_fp = 0
        while end < len(pairs) and pairs[end][0] == score:
            if pairs[end][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            end += 1
        tp += group_tp
        fp += group_fp
        tpr = tp / positives
        fpr = fp / negatives
        precision = tp / (tp + fp)
        recall = tpr
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        roc.append((fpr, tpr))
        pr.append((recall, precision))
        index = end
    auc = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(roc, roc[1:]))
    brier = sum((probability - target) ** 2 for probability, target in pairs) / len(pairs)
    return roc, pr, auc, average_precision, brier


def _downsample(points: Sequence[tuple[float, float]], limit: int = 240) -> tuple[tuple[float, float], ...]:
    if len(points) <= limit:
        return tuple(points)
    indices = {round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)}
    return tuple(points[index] for index in sorted(indices))


def _prediction_key(fields: Sequence[str]) -> str:
    for key in ("probability", "blended_probability", "prediction", "oof_probability", "target_probability"):
        if key in fields:
            return key
    raise DeliveryBuildError("OOF 缺少最终概率列")


def _component_scores(oof: CsvTable, history: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    candidates = (
        ("全图影像", "pure_image_probability"),
        ("病灶聚焦", "lesion_focus_probability"),
        ("影像+元数据融合", "image_metadata_fusion_probability"),
        ("CatBoost 元数据", "metadata_catboost_probability"),
        ("最终融合", "blended_probability"),
    )
    found: list[tuple[str, float]] = []
    for label, key in candidates:
        if key in oof.fields:
            _, _, auc, _, _ = _binary_curve(oof.rows, key)
            found.append((label, auc))
    if len(found) >= 3 and any("元数据" in label for label, _ in found):
        return tuple(found)

    histories: list[Mapping[str, Any]] = [history]
    source_runs = history.get("source_runs") if isinstance(history.get("source_runs"), list) else []
    if source_runs:
        seeds = tuple(int(item.get("seed") or -1) for item in source_runs if isinstance(item, dict))
        _require(seeds == (43, 44, 45), "三种子训练历史顺序不是 43/44/45")
        for item in source_runs:
            nested = item.get("history") if isinstance(item, dict) else None
            _require(isinstance(nested, dict), "正式种子训练历史缺失")
            histories.append(nested)
    nested_components: list[dict[str, float]] = []
    for item in histories[1:]:
        final_blend = item.get("final_blend") if isinstance(item.get("final_blend"), dict) else {}
        raw = final_blend.get("component_oof_auc")
        _require(isinstance(raw, dict), "正式种子训练历史缺少组件 OOF ROC-AUC")
        nested_components.append(
            {str(name): _finite(value, f"模型组件 {name} ROC-AUC", minimum=0, maximum=1) for name, value in raw.items()}
        )
    if nested_components:
        keys = set(nested_components[0])
        _require(all(set(item) == keys for item in nested_components), "三种子模型组件集合不一致")
        labels = {
            "pure_image": "全图影像",
            "lesion_focus": "病灶聚焦",
            "image_metadata_fusion": "影像+元数据融合",
            "metadata_catboost": "CatBoost 元数据",
        }
        averaged = [
            (labels.get(name, name), sum(item[name] for item in nested_components) / len(nested_components))
            for name in sorted(keys, key=lambda value: (list(labels).index(value) if value in labels else 99, value))
        ]
        final_key = _prediction_key(oof.fields)
        _, _, final_auc, _, _ = _binary_curve(oof.rows, final_key)
        averaged.append(("最终三种子融合", final_auc))
        _require(len(averaged) >= 4 and any("元数据" in label for label, _ in averaged), "元数据融合证据不完整")
        return tuple(averaged)

    containers: list[Any] = [
        history.get("model_component_roc_auc"),
        history.get("component_roc_auc"),
        history.get("component_scores"),
    ]
    final_blend = history.get("final_blend") if isinstance(history.get("final_blend"), dict) else {}
    containers.extend((final_blend.get("component_oof_auc"), final_blend.get("component_fit_auc")))
    for container in containers:
        if not isinstance(container, dict):
            continue
        mapped = []
        for raw_name, raw_score in container.items():
            key = str(raw_name).lower()
            if "lesion" in key:
                label = "病灶聚焦"
            elif "metadata" in key and "fusion" in key:
                label = "影像+元数据融合"
            elif "metadata" in key:
                label = "CatBoost 元数据"
            elif "image" in key or "full" in key:
                label = "全图影像"
            elif "blend" in key or "ensemble" in key:
                label = "最终融合"
            else:
                label = str(raw_name)
            mapped.append((label, _finite(raw_score, f"模型组件 {raw_name} ROC-AUC", minimum=0, maximum=1)))
        if len(mapped) >= 3 and any("元数据" in label for label, _ in mapped):
            return tuple(mapped)
    raise DeliveryBuildError("缺少可追溯的元数据融合对比，报告拒绝生成")


def _stable_time(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = (
        payloads["private_grader_ledger.json"].get("recorded_at"),
        payloads["private_grader.json"].get("completed_at"),
        payloads["candidate_freeze.json"].get("frozen_at"),
        payloads["run.json"].get("created_at"),
    )
    for value in candidates:
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed.isoformat(timespec="seconds")
    raise DeliveryBuildError("Run 缺少稳定、可解析的证据时间")


def _validate_freeze(run_dir: Path, freeze: Mapping[str, Any], run_id: str) -> None:
    _assert_same_run(freeze, run_id, "candidate_freeze.json")
    _require(freeze.get("status") == "frozen_before_private_grader", "候选未在 grader 前冻结")
    _require(freeze.get("tuning_closed") is True, "候选冻结后仍允许调参")
    _require(int(freeze.get("private_grader_execution_count_before_freeze") or 0) == 0, "冻结前已执行 grader")
    records = freeze.get("artifacts")
    _require(isinstance(records, list) and bool(records), "候选冻结清单为空")
    for record in records:
        _require(isinstance(record, dict), "候选冻结记录无效")
        path = _safe_relative(run_dir, record.get("path"))
        _require(path.is_file(), f"冻结产物缺失：{record.get('path')}")
        _require(sha256_file(path) == str(record.get("sha256") or "").lower(), f"冻结产物发生漂移：{record.get('path')}")


def _validate_reviews(payloads: Mapping[str, dict[str, Any]], run_dir: Path, run_id: str) -> None:
    review = payloads["review.json"]
    _assert_same_run(review, run_id, "review.json")
    _require(_status_passed(review), "Independent Review 未通过")
    checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    for key in (
        "patient_group_overlap_zero",
        "content_group_overlap_zero",
        "oof_coverage_exactly_once",
        "submission_schema_and_order",
        "private_labels_unavailable_during_training",
        "private_grader_not_executed",
        "official_submission_not_executed",
    ):
        _require(checks.get(key) is True, f"Independent Review 检查未通过：{key}")
    for name, expected_hash in (review.get("artifact_hashes") or {}).items():
        path = _safe_relative(run_dir, name)
        _require(path.is_file(), f"Review 绑定产物缺失：{name}")
        _require(sha256_file(path) == str(expected_hash).lower(), f"Review 哈希不一致：{name}")

    claim = payloads["claim_audit.json"]
    _assert_same_run(claim, run_id, "claim_audit.json")
    _require(_status_passed(claim), "Claim Audit 未通过")
    claim_checks = claim.get("checks") if isinstance(claim.get("checks"), dict) else {}
    for key in (
        "no_public_leaderboard_claim",
        "no_official_medal_claim",
        "no_clinical_diagnosis_claim",
        "private_grader_not_used_for_tuning",
        "candidate_hashes_unchanged",
        "official_submission_not_executed",
    ):
        _require(claim_checks.get(key) is True, f"Claim Audit 检查未通过：{key}")


def _validate_grader(payloads: Mapping[str, dict[str, Any]], paths: Mapping[str, Path], run_id: str) -> float | None:
    grader = payloads["private_grader.json"]
    ledger = payloads["private_grader_ledger.json"]
    _assert_same_run(grader, run_id, "private_grader.json")
    _assert_same_run(ledger, run_id, "private_grader_ledger.json")
    grader_status = str(grader.get("status") or "").strip().lower()
    _require(_status_passed(grader) or grader_status == "failed_closed", "终局私有 grader 未记录")
    _require(int(grader.get("execution_index") or grader.get("execution_count") or 0) == 1, "grader 必须且只能执行一次")
    _require(int(ledger.get("execution_count") or 0) == 1, "grader ledger 执行次数不是 1")
    freeze_hash = sha256_file(paths["candidate_freeze.json"])
    _require(str(grader.get("candidate_freeze_sha256") or "").lower() == freeze_hash, "grader 未绑定当前候选冻结")
    _require(str(ledger.get("candidate_freeze_sha256") or "").lower() == freeze_hash, "grader ledger 未绑定当前候选冻结")
    _require(str(ledger.get("result_sha256") or "").lower() == sha256_file(paths["private_grader.json"]), "grader 结果哈希不一致")
    _require(grader.get("executed_after_freeze") is True, "grader 未在候选冻结后执行")
    _require(grader.get("feedback_used_for_tuning") is False, "grader 结果被用于后续调参")
    _require(ledger.get("feedback_used_for_tuning") is False, "grader ledger 显示结果被用于调参")
    _require(grader.get("official_submission_executed") is False, "grader 路径执行了公开提交")
    if grader_status == "failed_closed":
        _require(grader.get("mle_private_grader_score", grader.get("score")) in (None, ""), "失败关闭的 grader 不能携带分数")
        _require(ledger.get("score") in (None, ""), "失败关闭的 grader ledger 不能携带分数")
        _require(str(ledger.get("outcome") or "").lower() == "failed_closed", "grader ledger 未记录失败关闭结果")
        _require(str(grader.get("error") or grader.get("failure_reason") or "").strip(), "失败关闭的 grader 缺少错误证据")
        return None
    score = _finite(grader.get("mle_private_grader_score", grader.get("score")), "终局离线 grader 分数", minimum=0, maximum=1)
    _require(abs(score - _finite(ledger.get("score"), "grader ledger 分数", minimum=0, maximum=1)) <= 1e-12, "grader 与 ledger 分数不一致")
    return score


def validate_run(run_dir: Path) -> ValidatedRun:
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), f"Run 目录不存在：{run_dir}")
    paths = {name: _locate_evidence(run_dir, name) for name in REQUIRED_EVIDENCE}
    json_names = [name for name in REQUIRED_EVIDENCE if name.endswith(".json")]
    payloads = {name: _read_json(paths[name], name) for name in json_names}
    run = payloads["run.json"]
    run_id = str(run.get("run_id") or "")
    _require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,160}", run_id)), "Run ID 无效")
    request = payloads["request.json"]
    _require(str(request.get("dataset") or "") == TASK_ID, "请求不是 SIIM-ISIC 任务")
    task_type = str(request.get("task_type") or "")
    _require(task_type == "image_classification", "请求不是图像分类任务")
    submission_policy = request.get("submission_policy") if isinstance(request.get("submission_policy"), dict) else {}
    _require(submission_policy.get("official_submission") == "forbidden", "公开提交策略未关闭")

    for name, payload in payloads.items():
        if name in {"request.json", "task_graph.json", "research_design.json", "training_history.json"}:
            _assert_same_run(payload, run_id, name, required=False)
        else:
            _assert_same_run(payload, run_id, name)

    dataset = payloads["dataset_profile.json"]
    _require(_status_passed(dataset), "数据清单未通过")
    counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else dataset
    for key, expected in EXPECTED.items():
        _require(int(counts.get(key) or -1) == expected, f"数据规模不一致：{key}")
    _require(dataset.get("complete") is True, "数据集未标记为完整")

    audit = payloads["data_audit.json"]
    _require(_status_passed(audit), "数据审计未通过")
    _require(int(audit.get("patient_group_overlap") or 0) == 0, "患者分组存在重叠")
    _require(int(audit.get("content_group_overlap") or 0) == 0, "重复内容分组存在重叠")
    _require(int(audit.get("private_label_access_count") or 0) == 0, "训练前访问了私有标签")
    _require(audit.get("target_in_test_features") is False, "目标泄漏到测试特征")

    runtime = payloads["hpc_runtime.json"]
    gpu = runtime.get("gpu") if isinstance(runtime.get("gpu"), dict) else {}
    _require("A800" in str(gpu.get("name") or runtime.get("gpu_name") or ""), "运行证据不是 A800")
    _require(str(runtime.get("training_status") or "").lower() == "completed", "A800 训练尚未完成")
    _require(runtime.get("other_processes_modified") is False, "运行修改了其他进程")
    _require(int(runtime.get("signals_sent") or 0) == 0, "运行向其他进程发送过信号")

    training_result = payloads["training_result.json"]
    _require(_status_passed(training_result), "正式训练结果未完成")
    _require(
        tuple(int(value) for value in training_result.get("formal_seeds") or ())
        == (43, 44, 45),
        "正式训练种子不是 43/44/45",
    )
    _require(training_result.get("official_submission_executed") is False, "正式训练执行了公开提交")
    _require(int(training_result.get("private_grader_execution_count") or 0) == 0, "正式训练阶段执行了 grader")
    _require(int(training_result.get("private_label_access_count") or 0) == 0, "正式训练阶段访问了私有标签")

    ablation = payloads["preprocessing_ablation.json"]
    comparison = payloads["experiment_comparison.json"]
    _require(_status_passed(ablation) and _status_passed(comparison), "预处理消融未完成")
    _require(tuple(int(value) for value in ablation.get("seeds") or ()) == (40, 41, 42), "消融种子不是 40/41/42")
    records = ablation.get("profiles") if isinstance(ablation.get("profiles"), list) else ablation.get("candidates")
    _require(isinstance(records, list), "预处理消融缺少五个候选")
    profile_map: dict[str, float] = {}
    for record in records:
        _require(isinstance(record, dict), "预处理候选记录无效")
        name = _normalize_profile(record.get("profile") or record.get("name"))
        _require(name in PROFILE_ORDER and name not in profile_map, f"预处理候选无效或重复：{name}")
        profile_map[name] = _metric_from_record(record)
    _require(set(profile_map) == set(PROFILE_ORDER), "预处理消融未覆盖五种方案")
    selected_profile = _normalize_profile(ablation.get("selected_profile") or comparison.get("selected_profile"))
    _require(selected_profile in PROFILE_ORDER, "最终预处理方案无效")
    _require(_normalize_profile(comparison.get("selected_profile")) == selected_profile, "消融与比较文件的最终方案不一致")

    metrics_payload = payloads["metrics.json"]
    for key in ("roc_auc", "pr_auc", "brier"):
        _finite(metrics_payload.get(key), key, minimum=0, maximum=1)
    _require(int(metrics_payload.get("private_grader_execution_count") or 0) == 0, "训练指标混入终局 grader")
    _require(metrics_payload.get("kaggle_submission_executed") is False, "指标文件显示执行了 Kaggle 提交")
    _require(metrics_payload.get("clinical_diagnosis_claimed") is False, "指标文件包含临床诊断声明")

    oof = _read_csv(paths["oof_predictions.csv"], "oof_predictions.csv")
    _require(len(oof.rows) == EXPECTED["train_images"], "OOF 行数不等于 28,984")
    for required in ("image_name", "patient_id", "leakage_group", "target", "fold"):
        _require(required in oof.fields, f"OOF 缺少列：{required}")
    probability_key = _prediction_key(oof.fields)
    identifiers = [row["image_name"] for row in oof.rows]
    _require(len(set(identifiers)) == len(identifiers), "OOF image_name 重复")
    _require(sum(int(float(row["target"])) for row in oof.rows) == EXPECTED["positive_rows"], "OOF 阳性数不等于 513")
    _require(len({row["patient_id"] for row in oof.rows if row["patient_id"]}) == EXPECTED["patients"], "OOF 患者数不等于 2,056")
    folds = {int(float(row["fold"])) for row in oof.rows}
    _require(folds == set(range(5)), "OOF 未覆盖 5 个外层折")
    patient_folds: defaultdict[str, set[int]] = defaultdict(set)
    content_folds: defaultdict[str, set[int]] = defaultdict(set)
    for row in oof.rows:
        fold = int(float(row["fold"]))
        patient_folds[row["patient_id"]].add(fold)
        content_folds[row["leakage_group"]].add(fold)
    _require(all(len(values) == 1 for values in patient_folds.values()), "患者跨外层折重叠")
    _require(all(len(values) == 1 for values in content_folds.values()), "重复内容组跨外层折重叠")
    roc, pr, auc, average_precision, brier = _binary_curve(oof.rows, probability_key)
    _require(abs(auc - float(metrics_payload["roc_auc"])) <= 1e-8, "OOF 重算 ROC-AUC 与 metrics.json 不一致")
    _require(abs(average_precision - float(metrics_payload["pr_auc"])) <= 1e-8, "OOF 重算 PR-AUC 与 metrics.json 不一致")
    _require(abs(brier - float(metrics_payload["brier"])) <= 1e-8, "OOF 重算 Brier 与 metrics.json 不一致")

    fold_table = _read_csv(paths["fold_metrics.csv"], "fold_metrics.csv")
    _require("fold" in fold_table.fields and "roc_auc" in fold_table.fields, "fold_metrics.csv 结构不完整")
    _require(len(fold_table.rows) == 5, "fold_metrics.csv 必须正好 5 行")
    fold_records: list[dict[str, float]] = []
    for row in fold_table.rows:
        fold = int(float(row["fold"]))
        selected = [record for record in oof.rows if int(float(record["fold"])) == fold]
        _, _, fold_auc, fold_pr, fold_brier = _binary_curve(selected, probability_key)
        _require(abs(fold_auc - _finite(row["roc_auc"], f"fold {fold} ROC-AUC", minimum=0, maximum=1)) <= 1e-8, f"fold {fold} ROC-AUC 不一致")
        if row.get("pr_auc") not in (None, ""):
            _require(abs(fold_pr - _finite(row["pr_auc"], f"fold {fold} PR-AUC", minimum=0, maximum=1)) <= 1e-8, f"fold {fold} PR-AUC 不一致")
        brier_key = "brier" if row.get("brier") not in (None, "") else "brier_score"
        if row.get(brier_key) not in (None, ""):
            _require(abs(fold_brier - _finite(row[brier_key], f"fold {fold} Brier", minimum=0, maximum=1)) <= 1e-8, f"fold {fold} Brier 不一致")
        fold_records.append({"fold": float(fold), "roc_auc": fold_auc, "pr_auc": fold_pr, "brier": fold_brier})
    fold_records.sort(key=lambda item: item["fold"])

    submission = _read_csv(paths["submission.csv"], "submission.csv")
    sample = _read_csv(paths["sample_submission.csv"], "sample_submission.csv")
    _require(submission.fields == ("image_name", "target"), "submission.csv 表头必须为 image_name,target")
    _require(sample.fields == submission.fields, "sample_submission.csv 表头不一致")
    _require(len(submission.rows) == EXPECTED["test_images"] == len(sample.rows), "测试结果不是 4,142 行")
    _require([row["image_name"] for row in submission.rows] == [row["image_name"] for row in sample.rows], "测试结果顺序与样例提交不一致")
    _require(len({row["image_name"] for row in submission.rows}) == len(submission.rows), "测试结果 image_name 重复")
    for index, row in enumerate(submission.rows):
        _finite(row["target"], f"submission target row {index}", minimum=0, maximum=1)

    curve = metrics_payload.get("calibration_curve") if isinstance(metrics_payload.get("calibration_curve"), dict) else {}
    predicted = curve.get("mean_predicted_probability")
    observed = curve.get("fraction_positive")
    _require(isinstance(predicted, list) and isinstance(observed, list) and len(predicted) == len(observed) >= 5, "校准曲线证据不完整")
    calibration = tuple(
        (
            _finite(x, "校准预测概率", minimum=0, maximum=1),
            _finite(y, "校准观察比例", minimum=0, maximum=1),
        )
        for x, y in zip(predicted, observed, strict=True)
    )

    _validate_freeze(run_dir, payloads["candidate_freeze.json"], run_id)
    grader_score = _validate_grader(payloads, paths, run_id)
    _validate_reviews(payloads, run_dir, run_id)
    history = payloads["training_history.json"]
    component_scores = _component_scores(oof, history)

    thresholds = metrics_payload.get("historical_thresholds") if isinstance(metrics_payload.get("historical_thresholds"), dict) else run.get("historical_thresholds")
    _require(isinstance(thresholds, dict), "缺少历史门槛证据")
    for key in ("historical_private_score", "bronze", "silver", "gold"):
        _finite(thresholds.get(key), f"历史门槛 {key}", minimum=0, maximum=1)
    ci = metrics_payload.get("patient_grouped_bootstrap_roc_auc_95ci")
    _require(isinstance(ci, dict), "缺少患者分组 bootstrap 95% CI")
    lower = _finite(ci.get("lower"), "ROC-AUC 95% CI 下界", minimum=0, maximum=1)
    upper = _finite(ci.get("upper"), "ROC-AUC 95% CI 上界", minimum=0, maximum=1)
    _require(lower <= float(metrics_payload["roc_auc"]) <= upper, "ROC-AUC 不在报告的 95% CI 内")
    threshold_metrics = metrics_payload.get("fixed_oof_threshold_metrics")
    _require(isinstance(threshold_metrics, dict), "缺少固定 OOF 阈值指标")
    for key in ("threshold", "sensitivity", "specificity", "precision", "negative_predictive_value"):
        _finite(threshold_metrics.get(key), f"固定阈值指标 {key}", minimum=0, maximum=1)

    metrics = {
        "roc_auc": auc,
        "pr_auc": average_precision,
        "brier": brier,
        "grader_score": grader_score if grader_score is not None else auc,
        "grader_score_available": grader_score is not None,
        "ci_lower": lower,
        "ci_upper": upper,
        **{f"threshold_{key}": float(threshold_metrics[key]) for key in ("threshold", "sensitivity", "specificity", "precision", "negative_predictive_value")},
        **{f"historical_{key}": float(thresholds[key]) for key in ("historical_private_score", "bronze", "silver", "gold")},
    }
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    return ValidatedRun(
        run_dir=run_dir,
        run_id=run_id,
        stable_time=_stable_time(payloads),
        evidence_paths=paths,
        payloads=payloads,
        metrics=metrics,
        fold_metrics=tuple(fold_records),
        oof_rows=oof.rows,
        submission_rows=submission.rows,
        roc_points=_downsample(roc),
        pr_points=_downsample(pr),
        calibration_points=calibration,
        component_scores=component_scores,
        profile_scores=tuple((name, profile_map[name]) for name in PROFILE_ORDER),
        selected_profile=selected_profile,
        source_hashes=source_hashes,
    )


def _grader_score_available(data: ValidatedRun) -> bool:
    return bool(data.metrics.get("grader_score_available"))


def _terminal_metric_label(data: ValidatedRun, *, compact: bool = False) -> str:
    if _grader_score_available(data):
        return "终局离线 grader" if compact else "冻结后一次终局离线 grader"
    return "终局 grader 失败关闭" if compact else "冻结候选 OOF ROC-AUC；grader 失败关闭"


def _terminal_metric_value(data: ValidatedRun) -> str:
    if _grader_score_available(data):
        return f"{float(data.metrics['grader_score']):.5f}"
    return "失败关闭"


def _terminal_numeric_label(data: ValidatedRun) -> str:
    return "本次终局离线 grader" if _grader_score_available(data) else "本次分组 OOF"


def _terminal_numeric_title(data: ValidatedRun) -> str:
    return "终局离线 grader 与历史数值线" if _grader_score_available(data) else "分组 OOF 与历史数值线"


def _terminal_narrative(data: ValidatedRun) -> str:
    if _grader_score_available(data):
        return "终局分数只在候选哈希冻结且 Independent Review 通过后执行一次，未反馈给后续调参。"
    grader = data.payloads["private_grader.json"]
    reason = str(grader.get("error") or grader.get("failure_reason") or "官方 grader 未返回分数").strip()
    return (
        "终局 grader 已在候选哈希冻结且 Independent Review 通过后恰好记录一次，"
        f"但本次官方 grader 失败关闭（{reason}），未产生可用私有分数，也未反馈给后续调参。"
    )


def _terminal_evolution_evidence(data: ValidatedRun) -> str:
    if _grader_score_available(data):
        return "终局 grader 结果未反馈给调参。"
    return "终局 grader 恰好一次并失败关闭；无分数、无调参反馈。"


def _evolution_rounds(data: ValidatedRun) -> tuple[dict[str, Any], ...]:
    """Build the four evidence-bound EvoMind research rounds.

    The stages describe governed research decisions rather than claiming that
    every attempted change improved the score.  In particular, R2 explicitly
    records when the ablation gate rejected a non-robust change.
    """

    profile_scores = dict(data.profile_scores)
    raw_score = float(profile_scores["raw_multiview"])
    selected_score = float(profile_scores[data.selected_profile])
    comparison = data.payloads["experiment_comparison.json"]
    decision = comparison.get("decision") if isinstance(comparison.get("decision"), dict) else {}
    adopted = bool(decision.get("adopted")) and data.selected_profile != "raw_multiview"
    r2_status = "门禁通过" if adopted else "拒绝无效改动"
    r2_evidence = (
        f"种子 40/41/42；最终保留 {PROFILE_LABELS[data.selected_profile]}；"
        f"mean_gain={float(decision.get('mean_gain') or 0):+.5f}，"
        f"worst_fold_delta={float(decision.get('worst_fold_delta') or 0):+.5f}，"
        f"seed_passes={int(decision.get('seed_passes') or 0)}/3。"
    )
    component_candidates = [
        (label, float(value))
        for label, value in data.component_scores
        if label != "最终三种子融合"
    ]
    best_component = max(component_candidates, key=lambda item: item[1])
    formal_seeds = tuple(
        int(value)
        for value in data.payloads["training_result.json"].get("formal_seeds") or ()
    )
    freeze_status = str(data.payloads["candidate_freeze.json"].get("status") or "")
    return (
        {
            "code": "R1",
            "title": "原始多视图基线",
            "status": "建立可复核基线",
            "metric": f"{raw_score:.5f}",
            "metric_label": "消融三种子均值 ROC-AUC",
            "evidence": "固定患者/内容分组，以原始全图与病灶视图建立同一验证基线。",
        },
        {
            "code": "R2",
            "title": "五种预处理消融与稳定性门禁",
            "status": r2_status,
            "metric": f"{selected_score:.5f}",
            "metric_label": "最终保留方案均值 ROC-AUC",
            "evidence": r2_evidence,
        },
        {
            "code": "R3",
            "title": "全图、病灶与患者元数据融合",
            "status": "完成多通道比较",
            "metric": f"{data.metrics['roc_auc']:.5f}",
            "metric_label": "最终分组 OOF ROC-AUC",
            "evidence": (
                f"全图、病灶、影像+元数据与 CatBoost 分支均保留；"
                f"最佳单组件为 {best_component[0]} {best_component[1]:.5f}。"
            ),
        },
        {
            "code": "R4",
            "title": "三种子聚合并冻结候选",
            "status": "候选冻结后终局评测",
            "metric": _terminal_metric_value(data),
            "metric_label": _terminal_metric_label(data),
            "evidence": (
                f"正式种子 {'/'.join(str(seed) for seed in formal_seeds)} 聚合；"
                f"冻结状态 {freeze_status}；{_terminal_evolution_evidence(data)}"
            ),
        },
    )


def _svg_line_chart(title: str, points: Sequence[tuple[float, float]], *, reference: bool = False) -> str:
    width, height = 620, 320
    left, top, chart_w, chart_h = 62, 45, 520, 220
    path = " ".join(
        f"{'M' if index == 0 else 'L'} {left + x * chart_w:.2f} {top + (1 - y) * chart_h:.2f}"
        for index, (x, y) in enumerate(points)
    )
    diagonal = f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top}" stroke="#94a3b8" stroke-dasharray="7 6"/>' if reference else ""
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<text x="{left}" y="24" fill="#0f172a" font-size="18" font-weight="700">{html.escape(title)}</text>'
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" rx="10" fill="#f8fafc" stroke="#dbe4ef"/>'
        f'{diagonal}<path d="{path}" fill="none" stroke="#136fd2" stroke-width="4" stroke-linejoin="round"/>'
        f'<text x="{left}" y="{top + chart_h + 30}" fill="#64748b" font-size="13">0</text>'
        f'<text x="{left + chart_w - 5}" y="{top + chart_h + 30}" fill="#64748b" font-size="13">1</text>'
        f'<text x="{left - 24}" y="{top + chart_h + 4}" fill="#64748b" font-size="13">0</text>'
        f'<text x="{left - 24}" y="{top + 7}" fill="#64748b" font-size="13">1</text>'
        "</svg>"
    )


def _svg_bar_chart(title: str, values: Sequence[tuple[str, float]], selected: str | None = None) -> str:
    width, height = 700, 65 + 45 * len(values)
    minimum = max(0.0, min(value for _, value in values) - 0.03)
    span = max(max(value for _, value in values) - minimum, 0.01)
    bars = []
    for index, (label, value) in enumerate(values):
        y = 55 + index * 45
        bar_width = 390 * (value - minimum) / span
        color = "#16a07a" if label == selected else "#4f7cff"
        bars.append(
            f'<text x="12" y="{y + 18}" fill="#334155" font-size="14">{html.escape(label)}</text>'
            f'<rect x="190" y="{y}" width="{max(3, bar_width):.2f}" height="25" rx="6" fill="{color}"/>'
            f'<text x="{200 + max(3, bar_width):.2f}" y="{y + 18}" fill="#0f172a" font-size="14" font-weight="700">{value:.5f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<text x="12" y="25" fill="#0f172a" font-size="18" font-weight="700">{html.escape(title)}</text>'
        + "".join(bars)
        + "</svg>"
    )


def _report_html(data: ValidatedRun) -> bytes:
    metrics = data.metrics
    dataset = data.payloads["dataset_profile.json"]
    counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else dataset
    audit = data.payloads["data_audit.json"]
    runtime = data.payloads["hpc_runtime.json"]
    gpu = runtime.get("gpu") if isinstance(runtime.get("gpu"), dict) else {}
    claim = data.payloads["claim_audit.json"]
    review = data.payloads["review.json"]
    profile_values = [(PROFILE_LABELS[name], value) for name, value in data.profile_scores]
    selected_label = PROFILE_LABELS[data.selected_profile]
    component_values = list(data.component_scores)
    fold_values = [(f"Fold {int(item['fold']) + 1}", item["roc_auc"]) for item in data.fold_metrics]
    calibration_points = ((0.0, 0.0), *data.calibration_points, (1.0, 1.0))
    terminal_label = _terminal_numeric_label(data)
    terminal_title = _terminal_numeric_title(data)
    threshold_rows = [
        ("历史私有评分（既有记录）", metrics["historical_historical_private_score"]),
        ("历史铜牌数值线", metrics["historical_bronze"]),
        ("历史银牌数值线", metrics["historical_silver"]),
        ("历史金牌数值线", metrics["historical_gold"]),
        (terminal_label, metrics["grader_score"]),
    ]
    check_items = [
        "患者与重复内容组跨折零重叠",
        "每个训练样本恰好一次 OOF 覆盖",
        "候选冻结后终局 grader 仅记录一次",
        "grader 结果未反馈给调参",
        "未执行 Kaggle 公开提交",
        "不声明正式名次、奖牌或临床诊断能力",
    ]
    evolution_html = "".join(
        (
            '<article class="evolution-card">'
            f'<div class="round">{html.escape(str(item["code"]))}</div>'
            f'<h3>{html.escape(str(item["title"]))}</h3>'
            f'<div class="status">{html.escape(str(item["status"]))}</div>'
            f'<div class="metric">{html.escape(str(item["metric"]))}</div>'
            f'<div class="label">{html.escape(str(item["metric_label"]))}</div>'
            f'<p>{html.escape(str(item["evidence"]))}</p>'
            '</article>'
        )
        for item in _evolution_rounds(data)
    )
    css = """
    :root{--ink:#0f172a;--muted:#58677c;--line:#dbe4ef;--blue:#136fd2;--cyan:#10a7c4;--green:#16a07a;--paper:#f4f7fb}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;line-height:1.55}
    main{max-width:1180px;margin:0 auto;background:white;box-shadow:0 18px 55px rgba(15,23,42,.12)}
    header{padding:62px 68px 54px;background:linear-gradient(135deg,#071329,#103a68 58%,#0a7890);color:white;position:relative;overflow:hidden}
    header:after{content:"";position:absolute;width:420px;height:420px;border:1px solid rgba(255,255,255,.18);border-radius:50%;right:-145px;top:-210px}
    .eyebrow{letter-spacing:.14em;text-transform:uppercase;color:#7fe8ff;font-size:13px;font-weight:700}.run{font-family:Consolas,monospace;color:#b7d7ff;font-size:13px}
    h1{font-size:40px;line-height:1.2;margin:16px 0 18px}.lead{font-size:19px;max-width:840px;color:#e4edf8}
    section{padding:44px 68px;border-bottom:1px solid var(--line)}h2{font-size:27px;margin:0 0 8px}.section-lead{color:var(--muted);margin:0 0 28px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{padding:20px;border:1px solid var(--line);border-radius:14px;background:#fff}.metric{font-size:30px;font-weight:800;color:#103a68}.label{font-size:13px;color:var(--muted)}
    .evolution-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.evolution-card{padding:19px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(180deg,#f8fbff,#fff)}.evolution-card .round{font-family:Consolas,monospace;color:var(--blue);font-weight:800;letter-spacing:.08em}.evolution-card h3{font-size:18px;line-height:1.35;margin:8px 0}.evolution-card .status{display:inline-block;padding:4px 9px;border-radius:999px;background:#eaf7f3;color:#087253;font-size:12px;font-weight:700}.evolution-card .metric{margin-top:16px;font-size:26px}.evolution-card p{font-size:13px;color:var(--muted);margin:14px 0 0}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{border:1px solid var(--line);border-radius:16px;padding:18px;background:#fff;overflow:hidden}.panel svg{width:100%;height:auto}
    table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}th{background:#f7f9fc;color:#41516a}
    .pass{color:#0b7b58;font-weight:700}.boundary{padding:22px;border-left:5px solid var(--cyan);background:#eefaff;border-radius:0 12px 12px 0}
    ul.checks{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;padding:0;list-style:none}.checks li:before{content:"✓";color:var(--green);font-weight:900;margin-right:9px}
    footer{padding:30px 68px;background:#071329;color:#cbd5e1;font-size:13px}.small{font-size:12px;color:var(--muted)}
    @media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}section,header{padding-left:24px;padding-right:24px}}
    @media print{body{background:white}main{box-shadow:none}section{break-inside:avoid}.panel{break-inside:avoid}}
    """
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvoMind SIIM-ISIC 医学影像研究报告</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">EvoMind · Evidence-bound research</div><h1>SIIM-ISIC 黑色素瘤医学影像研究报告</h1>
<p class="lead">从一句自然语言需求到 A800 训练、患者分组验证、独立复核、终局离线评测与可追溯交付。所有结果绑定同一真实 Run。</p>
<p class="run">RUN {html.escape(data.run_id)} · EVIDENCE {html.escape(data.stable_time)}</p></header>
<section><h2>结论总览</h2><p class="section-lead">主指标采用患者及重复内容分组的独立 OOF ROC-AUC；{html.escape(_terminal_narrative(data))} 不代表 Kaggle 正式排名。</p>
<div class="grid"><div class="card"><div class="metric">{metrics['roc_auc']:.5f}</div><div class="label">分组 OOF ROC-AUC</div></div>
<div class="card"><div class="metric">{metrics['pr_auc']:.5f}</div><div class="label">OOF PR-AUC</div></div>
<div class="card"><div class="metric">{html.escape(_terminal_metric_value(data))}</div><div class="label">{html.escape(_terminal_metric_label(data, compact=True))}</div></div>
<div class="card"><div class="metric">[{metrics['ci_lower']:.4f}, {metrics['ci_upper']:.4f}]</div><div class="label">患者组 bootstrap 95% CI</div></div></div></section>
<section><h2>数据与泄漏控制</h2><p class="section-lead">类别极不平衡，验证单位不是单张图片，而是患者与重复内容的连接组。</p>
<div class="grid"><div class="card"><div class="metric">{int(counts['files']):,}</div><div class="label">数据文件</div></div>
<div class="card"><div class="metric">{int(counts['train_images']):,}</div><div class="label">训练图像</div></div>
<div class="card"><div class="metric">{int(counts['patients']):,}</div><div class="label">患者</div></div>
<div class="card"><div class="metric">{int(counts['positive_rows']) / int(counts['train_images']) * 100:.2f}%</div><div class="label">恶性样本占比</div></div></div>
<ul class="checks"><li>患者跨折重叠：{int(audit.get('patient_group_overlap') or 0)}</li><li>重复内容组跨折重叠：{int(audit.get('content_group_overlap') or 0)}</li><li>私有标签训练期访问：{int(audit.get('private_label_access_count') or 0)}</li><li>OOF 覆盖：28,984 / 28,984</li></ul></section>
<section><h2>EvoMind 四轮进化轨迹</h2><p class="section-lead">每一轮都绑定同一 Run 和冻结证据。轨迹展示研究决策，不强行制造单调提升；退化或不稳定的改动会被门禁拒绝。</p>
<div class="evolution-grid">{evolution_html}</div></section>
<section><h2>五种预处理方案消融</h2><p class="section-lead">相同分组、相同外部种子 40/41/42；只有满足均值、最差折与种子通过数门槛才采用变化。</p>
<div class="panel">{_svg_bar_chart('预处理方案三种子均值 ROC-AUC', profile_values, selected_label)}</div>
<p>最终保留：<strong>{html.escape(selected_label)}</strong>。未通过门槛的变化被系统记录而不是强行采用。</p></section>
<section><h2>判别能力与类别不平衡</h2><div class="two"><div class="panel">{_svg_line_chart('ROC 曲线', data.roc_points, reference=True)}</div>
<div class="panel">{_svg_line_chart('Precision-Recall 曲线', data.pr_points)}</div></div></section>
<section><h2>校准、阈值与折稳定性</h2><div class="two"><div class="panel">{_svg_line_chart('OOF 校准曲线', calibration_points, reference=True)}</div>
<div class="panel">{_svg_bar_chart('五个患者/内容分组外层折 ROC-AUC', fold_values)}</div></div>
<table><thead><tr><th>固定 OOF 阈值</th><th>敏感度</th><th>特异度</th><th>精确率</th><th>阴性预测值</th><th>Brier</th></tr></thead>
<tbody><tr><td>{metrics['threshold_threshold']:.4f}</td><td>{metrics['threshold_sensitivity']:.2%}</td><td>{metrics['threshold_specificity']:.2%}</td><td>{metrics['threshold_precision']:.2%}</td><td>{metrics['threshold_negative_predictive_value']:.2%}</td><td>{metrics['brier']:.5f}</td></tr></tbody></table></section>
<section><h2>多通道与元数据融合</h2><p class="section-lead">全图、病灶聚焦、患者元数据与 CatBoost 分支在冻结验证协议下比较并融合。</p>
<div class="panel">{_svg_bar_chart('组件 OOF ROC-AUC', component_values)}</div></section>
<section><h2>终局离线评测与历史数值门槛</h2><p class="section-lead">下图只做历史数值对照；比赛已截止，本 Run 未进行公开提交，也不据此声明正式名次或奖牌。</p>
<div class="panel">{_svg_bar_chart(terminal_title, threshold_rows, terminal_label)}</div>
<p>A800：<strong>{html.escape(str(gpu.get('name') or runtime.get('gpu_name')))}</strong>；其他进程被修改：<strong>{str(runtime.get('other_processes_modified')).lower()}</strong>；发送信号：<strong>{int(runtime.get('signals_sent') or 0)}</strong>。</p></section>
<section><h2>Independent Review 与 Claim Audit</h2><ul class="checks">{''.join(f'<li>{html.escape(item)}</li>' for item in check_items)}</ul>
<p>Independent Review：<span class="pass">{html.escape(str(review.get('status')).upper())}</span>；Claim Audit：<span class="pass">{html.escape(str(claim.get('status')).upper())}</span>。</p></section>
<section><h2>交付与复现</h2><div class="two"><div><table><thead><tr><th>用户交付</th><th>用途</th></tr></thead><tbody>
<tr><td>evomind-siim-isic-report.pdf</td><td>专业研究报告</td></tr><tr><td>evomind-siim-isic-results.csv</td><td>4,142 行测试预测</td></tr>
<tr><td>evomind-siim-isic-code.zip</td><td>可复核代码</td></tr><tr><td>evomind-siim-isic-evidence.zip</td><td>指标、审计、账本与哈希证据</td></tr></tbody></table></div>
<div class="boundary"><strong>适用边界</strong><p>本报告是医学影像研究基准结果，不是临床诊断工具或医疗建议。终局 grader 记录是独立离线评测流程证据，不是 Kaggle 官方提交、排名或奖牌。</p></div></div></section>
<footer>Run {html.escape(data.run_id)} · 同一 Run 证据闭环 · 生成基准时间 {html.escape(data.stable_time)}</footer>
</main></body></html>"""
    return html_text.encode("utf-8")


def _load_fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise DeliveryBuildError("生成 PDF 需要 PyMuPDF（fitz）") from exc
    return fitz


@lru_cache(maxsize=1)
def _cjk_font_path() -> Path:
    configured = os.environ.get("EVOMIND_CJK_FONT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise DeliveryBuildError("未找到可嵌入的中文字体；可通过 EVOMIND_CJK_FONT 指定 TTF/TTC")


def _ensure_pdf_font(page: Any) -> None:
    if not any(len(font) > 4 and font[4] == "evocjk" for font in page.get_fonts()):
        page.insert_font(fontname="evocjk", fontfile=str(_cjk_font_path()))


def _pdf_text(page: Any, rect: Any, text: str, *, size: float = 10, color: tuple[float, float, float] = (0.08, 0.12, 0.2), align: int = 0) -> None:
    _ensure_pdf_font(page)
    result = page.insert_textbox(rect, str(text), fontname="evocjk", fontsize=size, color=color, align=align, lineheight=1.22)
    _require(result >= -1.5, f"PDF 文本溢出：{str(text)[:40]}")


def _pdf_metric_card(page: Any, rect: Any, value: str, label: str) -> None:
    page.draw_rect(rect, color=(0.84, 0.89, 0.95), fill=(0.97, 0.98, 1.0), width=0.8)
    preferred_size = 13 if len(value) > 20 else 15 if len(value) > 12 else 18
    value_size = min(preferred_size, max(7.5, (rect.width - 24) * 0.82 / max(1, len(value))))
    _pdf_text(
        page,
        (rect.x0 + 12, rect.y0 + 8, rect.x1 - 12, rect.y0 + 49),
        value,
        size=value_size,
        color=(0.05, 0.25, 0.44),
    )
    _pdf_text(page, (rect.x0 + 12, rect.y0 + 49, rect.x1 - 12, rect.y1 - 5), label, size=7.8, color=(0.35, 0.42, 0.52))


def _pdf_line_chart(page: Any, rect: Any, title: str, points: Sequence[tuple[float, float]], *, reference: bool = False) -> None:
    fitz = _load_fitz()
    page.draw_rect(rect, color=(0.84, 0.89, 0.95), fill=(0.985, 0.99, 1.0), width=0.8)
    _pdf_text(page, (rect.x0 + 12, rect.y0 + 9, rect.x1 - 12, rect.y0 + 31), title, size=10.5, color=(0.06, 0.16, 0.29))
    chart = fitz.Rect(rect.x0 + 40, rect.y0 + 39, rect.x1 - 18, rect.y1 - 31)
    page.draw_rect(chart, color=(0.86, 0.89, 0.93), width=0.6)
    if reference:
        page.draw_line(
            (chart.x0, chart.y1),
            (chart.x1, chart.y0),
            color=(0.62, 0.67, 0.74),
            width=0.7,
            dashes="[4 3] 0",
        )
    mapped = [(chart.x0 + x * chart.width, chart.y1 - y * chart.height) for x, y in points]
    for first, second in zip(mapped, mapped[1:]):
        page.draw_line(first, second, color=(0.07, 0.43, 0.82), width=1.6)
    _pdf_text(page, (chart.x0 - 4, chart.y1 + 5, chart.x0 + 22, chart.y1 + 22), "0", size=7, color=(0.4, 0.46, 0.55))
    _pdf_text(page, (chart.x1 - 14, chart.y1 + 5, chart.x1 + 4, chart.y1 + 22), "1", size=7, color=(0.4, 0.46, 0.55))


def _pdf_bar_chart(page: Any, rect: Any, title: str, values: Sequence[tuple[str, float]], *, selected: str | None = None) -> None:
    fitz = _load_fitz()
    page.draw_rect(rect, color=(0.84, 0.89, 0.95), fill=(0.985, 0.99, 1.0), width=0.8)
    _pdf_text(page, (rect.x0 + 12, rect.y0 + 9, rect.x1 - 12, rect.y0 + 31), title, size=10.5, color=(0.06, 0.16, 0.29))
    minimum = max(0.0, min(value for _, value in values) - 0.03)
    span = max(max(value for _, value in values) - minimum, 0.01)
    available = rect.height - 48
    row_h = available / len(values)
    for index, (label, value) in enumerate(values):
        y0 = rect.y0 + 37 + index * row_h
        _pdf_text(page, (rect.x0 + 12, y0, rect.x0 + 125, y0 + row_h - 2), label, size=7.4, color=(0.2, 0.27, 0.36))
        width = max(3, (rect.width - 205) * (value - minimum) / span)
        color = (0.09, 0.63, 0.48) if label == selected else (0.31, 0.49, 1.0)
        page.draw_rect(fitz.Rect(rect.x0 + 128, y0 + 2, rect.x0 + 128 + width, y0 + min(15, row_h - 5)), color=color, fill=color)
        _pdf_text(page, (rect.x0 + 134 + width, y0 - 1, rect.x1 - 8, y0 + row_h - 1), f"{value:.5f}", size=7.3, color=(0.06, 0.16, 0.29))


def _pdf_header(page: Any, title: str, subtitle: str, run_id: str, page_number: int) -> None:
    fitz = _load_fitz()
    page.draw_rect(fitz.Rect(0, 0, page.rect.width, 82), color=(0.03, 0.11, 0.22), fill=(0.03, 0.11, 0.22))
    _pdf_text(page, (36, 19, page.rect.width - 36, 50), title, size=19, color=(1, 1, 1))
    _pdf_text(page, (36, 52, page.rect.width - 36, 73), subtitle, size=8, color=(0.72, 0.84, 0.95))
    page.draw_line((36, page.rect.height - 31), (page.rect.width - 36, page.rect.height - 31), color=(0.84, 0.88, 0.92), width=0.6)
    _pdf_text(page, (36, page.rect.height - 25, page.rect.width - 80, page.rect.height - 8), f"RUN {run_id}", size=6.8, color=(0.38, 0.45, 0.54))
    _pdf_text(page, (page.rect.width - 70, page.rect.height - 25, page.rect.width - 36, page.rect.height - 8), f"{page_number:02d}", size=7, color=(0.38, 0.45, 0.54), align=2)


def _report_pdf(data: ValidatedRun, path: Path) -> None:
    fitz = _load_fitz()
    doc = fitz.open()
    width, height = fitz.paper_size("a4")
    metrics = data.metrics
    dataset = data.payloads["dataset_profile.json"]
    counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else dataset
    audit = data.payloads["data_audit.json"]
    runtime = data.payloads["hpc_runtime.json"]
    gpu = runtime.get("gpu") if isinstance(runtime.get("gpu"), dict) else {}
    review = data.payloads["review.json"]
    claim = data.payloads["claim_audit.json"]
    terminal_label = _terminal_numeric_label(data)
    terminal_title = _terminal_numeric_title(data)
    terminal_metric_label = _terminal_metric_label(data, compact=True)
    terminal_metric_value = _terminal_metric_value(data)
    terminal_narrative = _terminal_narrative(data)

    page = doc.new_page(width=width, height=height)
    page.draw_rect(page.rect, color=(0.025, 0.075, 0.16), fill=(0.025, 0.075, 0.16))
    page.draw_circle((width - 40, 60), 180, color=(0.08, 0.55, 0.68), width=1.1)
    page.draw_circle((width - 40, 60), 130, color=(0.18, 0.42, 0.82), width=0.8)
    _pdf_text(page, (48, 105, width - 48, 137), "EVOMIND · EVIDENCE-BOUND RESEARCH", size=10, color=(0.39, 0.91, 1.0))
    _pdf_text(page, (48, 165, width - 48, 262), "SIIM-ISIC 黑色素瘤\n医学影像研究报告", size=27, color=(1, 1, 1))
    _pdf_text(page, (48, 290, width - 70, 372), "从一句自然语言需求到 A800 训练、患者分组验证、独立复核、终局离线评测与可追溯交付。", size=13, color=(0.83, 0.9, 0.98))
    _pdf_text(page, (48, 420, width - 48, 470), f"RUN\n{data.run_id}", size=9.5, color=(0.56, 0.73, 0.93))
    _pdf_text(page, (48, height - 102, width - 48, height - 55), "医学影像研究基准 · 非临床诊断 · 未进行 Kaggle 公开提交", size=9, color=(0.52, 0.74, 0.82))

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "结论总览", "同一 Run 的分组 OOF、终局 grader 记录与审核结果", data.run_id, 2)
    cards = [
        (f"{metrics['roc_auc']:.5f}", "分组 OOF ROC-AUC"),
        (f"{metrics['pr_auc']:.5f}", "OOF PR-AUC"),
        (terminal_metric_value, terminal_metric_label),
        (f"[{metrics['ci_lower']:.4f}, {metrics['ci_upper']:.4f}]", "患者组 bootstrap 95% CI"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 36 + (index % 2) * 264
        y = 110 + (index // 2) * 90
        _pdf_metric_card(page, fitz.Rect(x, y, x + 248, y + 72), value, label)
    _pdf_text(page, (40, 320, width - 40, 366), "结论解释", size=15, color=(0.05, 0.25, 0.44))
    _pdf_text(page, (40, 368, width - 40, 462), f"主指标来自患者与重复内容连接组隔离的完整 OOF。{terminal_narrative}", size=11)
    boundary = fitz.Rect(40, 500, width - 40, 636)
    page.draw_rect(boundary, color=(0.12, 0.62, 0.72), fill=(0.93, 0.98, 1.0), width=1.2)
    _pdf_text(page, (56, 518, width - 56, 552), "声明边界", size=14, color=(0.04, 0.38, 0.48))
    _pdf_text(page, (56, 558, width - 56, 622), "本报告是医学影像研究基准结果，不是临床诊断工具。终局 grader 记录属于独立离线评测流程证据，不是 Kaggle 官方提交、排名或奖牌。", size=10)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "数据与验证设计", "33,129 个文件 · 1.77% 恶性样本 · 患者/内容组隔离", data.run_id, 3)
    cards = [
        (f"{int(counts['files']):,}", "数据文件"),
        (f"{int(counts['train_images']):,}", "训练图像"),
        (f"{int(counts['test_images']):,}", "测试图像"),
        (f"{int(counts['patients']):,}", "患者"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 36 + index * 132
        _pdf_metric_card(page, fitz.Rect(x, 106, x + 120, 178), value, label)
    _pdf_text(page, (40, 215, width - 40, 252), "类别分布", size=14, color=(0.05, 0.25, 0.44))
    bar = fitz.Rect(40, 260, width - 40, 300)
    positive_fraction = EXPECTED["positive_rows"] / EXPECTED["train_images"]
    page.draw_rect(bar, color=(0.84, 0.89, 0.95), fill=(0.91, 0.94, 0.98))
    page.draw_rect(fitz.Rect(bar.x0, bar.y0, bar.x0 + max(8, bar.width * positive_fraction), bar.y1), color=(0.9, 0.25, 0.35), fill=(0.9, 0.25, 0.35))
    _pdf_text(page, (40, 307, width - 40, 335), f"恶性 513 / 28,984（{positive_fraction * 100:.2f}%）· 采用类别加权损失，不复制少数类图像", size=9)
    checks = [
        ("患者跨折重叠", int(audit.get("patient_group_overlap") or 0)),
        ("重复内容组跨折重叠", int(audit.get("content_group_overlap") or 0)),
        ("训练期私有标签访问", int(audit.get("private_label_access_count") or 0)),
        ("OOF 覆盖缺失", 0),
    ]
    for index, (label, value) in enumerate(checks):
        y = 386 + index * 56
        page.draw_rect(fitz.Rect(40, y, width - 40, y + 42), color=(0.82, 0.91, 0.88), fill=(0.95, 0.99, 0.97), width=0.8)
        _pdf_text(page, (55, y + 10, width - 130, y + 34), label, size=9)
        _pdf_text(page, (width - 120, y + 8, width - 55, y + 34), str(value), size=12, color=(0.04, 0.46, 0.32), align=2)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "EvoMind 四轮进化", "同一 Run 下建立基线、拒绝无效改动、融合与冻结交付", data.run_id, 4)
    for index, item in enumerate(_evolution_rounds(data)):
        y = 104 + index * 137
        box = fitz.Rect(36, y, width - 36, y + 120)
        page.draw_rect(box, color=(0.78, 0.86, 0.94), fill=(0.97, 0.985, 1.0), width=0.8)
        _pdf_text(page, (52, y + 12, 90, y + 36), str(item["code"]), size=12, color=(0.06, 0.36, 0.66))
        _pdf_text(page, (94, y + 10, width - 190, y + 37), str(item["title"]), size=11.5, color=(0.05, 0.22, 0.39))
        _pdf_text(page, (width - 180, y + 10, width - 52, y + 36), str(item["status"]), size=7.7, color=(0.04, 0.46, 0.32), align=2)
        _pdf_text(page, (52, y + 46, 170, y + 82), str(item["metric"]), size=18, color=(0.05, 0.26, 0.46))
        _pdf_text(page, (178, y + 50, width - 52, y + 76), str(item["metric_label"]), size=7.8, color=(0.35, 0.42, 0.52))
        _pdf_text(page, (52, y + 84, width - 52, y + 111), str(item["evidence"]), size=7.5, color=(0.23, 0.31, 0.41))
    _pdf_text(page, (40, 664, width - 40, 728), "说明：四轮轨迹代表 EvoMind 如何比较、拒绝、融合和冻结研究方案，不要求每次尝试都提升。最终 grader 只在候选冻结后记录一次。", size=9.2)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "预处理消融", "固定分组与种子 40/41/42，失败变化明确拒绝", data.run_id, 5)
    profile_values = [(PROFILE_LABELS[name], value) for name, value in data.profile_scores]
    _pdf_bar_chart(page, fitz.Rect(36, 108, width - 36, 430), "五种预处理方案三种子均值 ROC-AUC", profile_values, selected=PROFILE_LABELS[data.selected_profile])
    _pdf_text(page, (40, 468, width - 40, 505), f"最终保留：{PROFILE_LABELS[data.selected_profile]}", size=14, color=(0.04, 0.46, 0.32))
    decision = data.payloads["experiment_comparison.json"].get("decision") or {}
    _pdf_text(page, (40, 515, width - 40, 612), f"采用门槛：三种子均值增益至少 0.0005；最差折回退不超过 0.002；至少 2/3 种子通过。\n真实决策：mean_gain={float(decision.get('mean_gain') or 0):.5f}，worst_fold_delta={float(decision.get('worst_fold_delta') or 0):.5f}，seed_passes={int(decision.get('seed_passes') or 0)}。", size=9.5)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "ROC / PR 与校准", "类别不平衡场景下同时报告判别、检出质量与概率校准", data.run_id, 6)
    _pdf_line_chart(page, fitz.Rect(36, 105, width - 36, 327), "ROC 曲线", data.roc_points, reference=True)
    _pdf_line_chart(page, fitz.Rect(36, 352, width - 36, 574), "Precision-Recall 曲线", data.pr_points)
    cards = [
        (f"{metrics['roc_auc']:.5f}", "ROC-AUC"),
        (f"{metrics['pr_auc']:.5f}", "PR-AUC"),
        (f"{metrics['brier']:.5f}", "Brier"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 36 + index * 176
        _pdf_metric_card(page, fitz.Rect(x, 607, x + 160, 678), value, label)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "折稳定性、校准与元数据融合", "五折稳定性和多通道贡献均来自冻结 OOF", data.run_id, 7)
    fold_values = [(f"Fold {int(item['fold']) + 1}", item["roc_auc"]) for item in data.fold_metrics]
    _pdf_bar_chart(page, fitz.Rect(36, 105, width - 36, 304), "五个外层折 ROC-AUC", fold_values)
    calibration = ((0.0, 0.0), *data.calibration_points, (1.0, 1.0))
    _pdf_line_chart(page, fitz.Rect(36, 330, width - 36, 522), "OOF 校准曲线", calibration, reference=True)
    _pdf_bar_chart(page, fitz.Rect(36, 548, width - 36, 745), "多通道与元数据组件 OOF ROC-AUC", data.component_scores)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "固定阈值与历史数值对照", "阈值只由 OOF 冻结；历史门槛不等于本 Run 的正式名次", data.run_id, 8)
    threshold_values = [
        ("历史既有评分", metrics["historical_historical_private_score"]),
        ("历史铜牌数值线", metrics["historical_bronze"]),
        ("历史银牌数值线", metrics["historical_silver"]),
        ("历史金牌数值线", metrics["historical_gold"]),
        (terminal_label, metrics["grader_score"]),
    ]
    _pdf_bar_chart(page, fitz.Rect(36, 105, width - 36, 385), terminal_title, threshold_values, selected=terminal_label)
    threshold_cards = [
        (f"{metrics['threshold_sensitivity']:.2%}", "敏感度"),
        (f"{metrics['threshold_specificity']:.2%}", "特异度"),
        (f"{metrics['threshold_precision']:.2%}", "精确率"),
        (f"{metrics['threshold_negative_predictive_value']:.2%}", "阴性预测值"),
    ]
    for index, (value, label) in enumerate(threshold_cards):
        x = 36 + index * 132
        _pdf_metric_card(page, fitz.Rect(x, 420, x + 120, 492), value, label)
    _pdf_text(page, (40, 535, width - 40, 620), f"固定 OOF 阈值：{metrics['threshold_threshold']:.5f}。该阈值未在终局验证或私有 grader 上反复调整。历史数值线仅供回顾，不构成官方名次、奖牌或临床性能声明。", size=10)

    page = doc.new_page(width=width, height=height)
    _pdf_header(page, "审核、资源与可追溯交付", "Independent Review + Claim Audit + 4 个哈希绑定下载", data.run_id, 9)
    cards = [
        (str(review.get("status")).upper(), "Independent Review"),
        (str(claim.get("status")).upper(), "Claim Audit"),
        (str(gpu.get("name") or runtime.get("gpu_name") or "A800"), "训练 GPU"),
        ("0", "向其他进程发送信号"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 36 + (index % 2) * 264
        y = 108 + (index // 2) * 90
        _pdf_metric_card(page, fitz.Rect(x, y, x + 248, y + 72), value, label)
    _pdf_text(page, (40, 315, width - 40, 350), "用户可直接下载", size=14, color=(0.05, 0.25, 0.44))
    for index, name in enumerate(DOWNLOAD_NAMES):
        y = 363 + index * 55
        page.draw_rect(fitz.Rect(40, y, width - 40, y + 40), color=(0.83, 0.88, 0.94), fill=(0.97, 0.98, 1.0), width=0.7)
        _pdf_text(page, (55, y + 10, width - 55, y + 32), name, size=8.5, color=(0.06, 0.28, 0.5))
    _pdf_text(page, (40, 610, width - 40, 694), "复现原则：使用相同患者/重复内容分组、正式种子、冻结预处理、训练配置与候选哈希。报告和证据包不包含连接密码、API Token 或私有标签。", size=9.5)

    doc.subset_fonts(verbose=False)
    stable = datetime.fromisoformat(data.stable_time)
    pdf_date = stable.strftime("D:%Y%m%d%H%M%S")
    doc.set_metadata(
        {
            "title": "EvoMind SIIM-ISIC Medical Imaging Research Report",
            "author": "EvoMind",
            "subject": f"Evidence-bound medical imaging research benchmark; Run {data.run_id}",
            "keywords": "SIIM-ISIC, melanoma, grouped OOF, research benchmark",
            "creator": "EvoMind deterministic delivery builder",
            "producer": "PyMuPDF",
            "creationDate": pdf_date,
            "modDate": pdf_date,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, clean=True, deflate=True, deflate_fonts=True, no_new_id=True, use_objstms=0)
    doc.close()


def _scan_secret_bytes(data: bytes, label: str) -> None:
    for kind, pattern in SECRET_PATTERNS:
        _require(pattern.search(data) is None, f"{label} 命中秘密扫描规则：{kind}")


def _scan_secret_file(path: Path, label: str) -> None:
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"events.jsonl", "hpc_telemetry.jsonl"}:
        _scan_secret_bytes(path.read_bytes(), label)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name.replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def _write_reproducible_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name in sorted(members):
            _require(not Path(name).is_absolute() and ".." not in Path(name).parts, "ZIP 成员路径越界")
            archive.writestr(_zip_info(name), members[name])


def _code_members(data: ValidatedRun, repo_root: Path, code_files: Sequence[Path] | None) -> dict[str, bytes]:
    selected = list(code_files) if code_files is not None else [repo_root / relative for relative in DEFAULT_CODE_FILES]
    members: dict[str, bytes] = {}
    records = []
    for path in selected:
        resolved = Path(path).resolve()
        _require(resolved.is_file(), f"代码包源文件不存在：{resolved}")
        try:
            relative = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise DeliveryBuildError(f"代码包源文件不在仓库内：{resolved}") from exc
        content = resolved.read_bytes()
        _scan_secret_bytes(content, relative)
        archive_name = f"code/{relative}"
        members[archive_name] = content
        records.append({"path": archive_name, "bytes": len(content), "sha256": _sha256_bytes(content)})
    readme = (
        "EvoMind SIIM-ISIC reproducible code package\n"
        f"Run ID: {data.run_id}\n"
        "Boundary: medical imaging research benchmark; not clinical diagnosis.\n"
        "Official Kaggle submission: forbidden and not executed.\n"
        "All code files are copied byte-for-byte and listed in code_manifest.json.\n"
    ).encode("utf-8")
    members["README.txt"] = readme
    members["code_manifest.json"] = _canonical_json(
        {
            "schema": "evomind.siim.code_manifest.v1",
            "run_id": data.run_id,
            "files": sorted(records, key=lambda item: item["path"]),
            "secret_scan": "passed",
            "official_submission": "forbidden",
        }
    )
    return members


def _evidence_members(data: ValidatedRun) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    records = []
    for name in REQUIRED_EVIDENCE:
        source = data.evidence_paths[name]
        content = source.read_bytes()
        _scan_secret_bytes(content, name)
        archive_name = f"evidence/{name}"
        members[archive_name] = content
        records.append({"path": archive_name, "bytes": len(content), "sha256": _sha256_bytes(content)})
    source_manifest = {
        "schema": "evomind.siim.evidence_source_manifest.v1",
        "run_id": data.run_id,
        "status": "verified",
        "files": sorted(records, key=lambda item: item["path"]),
        "same_run_verified": True,
        "candidate_freeze_sha256": data.source_hashes["candidate_freeze.json"],
        "private_grader_execution_count": 1,
        "official_submission": "forbidden",
        "clinical_use": "not_claimed",
        "secret_scan": "passed",
    }
    members["evidence/source_manifest.json"] = _canonical_json(source_manifest)
    return members


def _validate_zip(path: Path, *, required_basenames: set[str], require_python: bool = False) -> dict[str, Any]:
    _require(zipfile.is_zipfile(path), f"ZIP 无效：{path.name}")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        _require(names == sorted(names), f"ZIP 成员未按确定性顺序排列：{path.name}")
        basenames = {Path(name).name for name in names}
        _require(required_basenames <= basenames, f"ZIP 缺少必需文件：{path.name}")
        if require_python:
            _require(any(name.endswith(".py") for name in names), "代码 ZIP 不含 Python 源码")
        for info in archive.infolist():
            _require(info.date_time == (1980, 1, 1, 0, 0, 0), f"ZIP 时间戳不确定：{info.filename}")
            if not info.is_dir():
                _scan_secret_bytes(archive.read(info), f"{path.name}:{info.filename}")
        return {"members": len(names), "uncompressed_bytes": sum(info.file_size for info in archive.infolist())}


def _render_pdf_qa(pdf_path: Path, qa_dir: Path, run_id: str) -> dict[str, Any]:
    fitz = _load_fitz()
    doc = fitz.open(pdf_path)
    _require(doc.page_count == PDF_PAGE_COUNT, f"PDF 页数不是预期的 {PDF_PAGE_COUNT} 页")
    rendered = []
    for index, page in enumerate(doc):
        text_content = page.get_text("text")
        _require(len(text_content.strip()) >= 40, f"PDF 第 {index + 1} 页文本不足")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False, colorspace=fitz.csRGB)
        samples = bytes(pixmap.samples)
        _require(len(set(samples[:: max(1, len(samples) // 20_000)])) > 8, f"PDF 第 {index + 1} 页可能为空白")
        output = qa_dir / f"report-page-{index + 1:02d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(output)
        rendered.append(
            {
                "page": index + 1,
                "path": f"qa/{output.name}",
                "width": pixmap.width,
                "height": pixmap.height,
                "sha256": sha256_file(output),
            }
        )
    metadata = doc.metadata
    doc.close()
    _require(run_id in str(metadata.get("subject") or ""), "PDF 元数据未绑定 Run ID")
    qa = {
        "schema": "evomind.siim.pdf_qa.v1",
        "run_id": run_id,
        "status": "passed",
        "page_count": PDF_PAGE_COUNT,
        "rendered_pages": rendered,
        "pdf_sha256": sha256_file(pdf_path),
        "checks": {
            "all_pages_rendered": True,
            "no_blank_pages": True,
            "minimum_text_per_page": True,
            "run_id_metadata_bound": True,
        },
    }
    _write_json(qa_dir / "pdf-qa.json", qa)
    return qa


def _artifact_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _directory_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build_delivery(
    run_dir: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    code_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    data = validate_run(run_dir)
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    _require(not output_dir.exists() or output_dir.is_dir(), f"输出路径已存在且不是目录：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    idempotent_reuse = False
    try:
        html_path = temporary / "research_report.html"
        report_pdf = temporary / DOWNLOAD_NAMES[0]
        results_csv = temporary / DOWNLOAD_NAMES[1]
        code_zip = temporary / DOWNLOAD_NAMES[2]
        evidence_zip = temporary / DOWNLOAD_NAMES[3]
        _write_bytes(html_path, _report_html(data))
        _report_pdf(data, report_pdf)
        shutil.copyfile(data.evidence_paths["submission.csv"], results_csv)
        _write_reproducible_zip(code_zip, _code_members(data, repo_root, code_files))
        _write_reproducible_zip(evidence_zip, _evidence_members(data))
        code_qa = _validate_zip(code_zip, required_basenames={"README.txt", "code_manifest.json"}, require_python=True)
        evidence_qa = _validate_zip(
            evidence_zip,
            required_basenames={"metrics.json", "review.json", "claim_audit.json", "private_grader.json", "source_manifest.json"},
        )
        pdf_qa = _render_pdf_qa(report_pdf, temporary / "qa", data.run_id)
        result_table = _read_csv(results_csv, results_csv.name)
        _require(result_table.fields == ("image_name", "target") and len(result_table.rows) == EXPECTED["test_images"], "结果 CSV 验收失败")

        deliverables = [
            _artifact_record(report_pdf, temporary, "professional_pdf_report"),
            _artifact_record(results_csv, temporary, "test_predictions_csv"),
            _artifact_record(code_zip, temporary, "reproducible_code_zip"),
            _artifact_record(evidence_zip, temporary, "evidence_zip"),
        ]
        manifest = {
            "schema": "evomind.siim.delivery_bundle_manifest.v1",
            "run_id": data.run_id,
            "task_id": TASK_ID,
            "status": "verified",
            "evidence_time": data.stable_time,
            "report_html": _artifact_record(html_path, temporary, "professional_html_report"),
            "deliverables": deliverables,
            "source_evidence": [
                {
                    "path": name,
                    "bytes": data.evidence_paths[name].stat().st_size,
                    "sha256": data.source_hashes[name],
                }
                for name in sorted(data.evidence_paths)
            ],
            "candidate_freeze_sha256": data.source_hashes["candidate_freeze.json"],
            "private_grader_execution_count": 1,
            "same_run_verified": True,
            "all_sha256_bound": True,
            "official_submission": "forbidden",
            "clinical_use": "not_claimed",
            "secret_scan": {
                "status": "passed",
                "code_members": code_qa["members"],
                "evidence_members": evidence_qa["members"],
            },
            "reproducibility": {
                "zip_member_order": "lexicographic",
                "zip_timestamp": "1980-01-01T00:00:00",
                "zip_compression": "stored",
                "pdf_pages": pdf_qa["page_count"],
                "dynamic_wall_clock_values": False,
            },
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        qa_manifest = {
            "schema": "evomind.siim.delivery_qa.v1",
            "run_id": data.run_id,
            "status": "passed",
            "download_file_count": 4,
            "download_names": list(DOWNLOAD_NAMES),
            "pdf": pdf_qa,
            "code_zip": code_qa,
            "evidence_zip": evidence_qa,
            "results_rows": len(result_table.rows),
            "same_run_verified": True,
            "official_submission_executed": False,
            "clinical_diagnosis_claimed": False,
        }
        _write_json(temporary / "qa" / "delivery-qa.json", qa_manifest)
        if output_dir.exists():
            existing_inventory = _directory_inventory(output_dir)
            rebuilt_inventory = _directory_inventory(temporary)
            _require(bool(existing_inventory), f"既有交付目录为空，拒绝覆盖：{output_dir}")
            _require(
                existing_inventory == rebuilt_inventory,
                f"既有交付目录与当前同 Run 确定性构建不一致，拒绝覆盖：{output_dir}",
            )
            shutil.rmtree(temporary)
            idempotent_reuse = True
        else:
            os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_manifest = _read_json(output_dir / "artifact_manifest.json", "artifact_manifest.json")
    return {
        "status": "verified",
        "run_id": data.run_id,
        "idempotent_reuse": idempotent_reuse,
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "artifact_manifest.json"),
        "report_html": str(output_dir / "research_report.html"),
        "deliverables": [str(output_dir / name) for name in DOWNLOAD_NAMES],
        "deliverable_sha256": {item["path"]: item["sha256"] for item in final_manifest["deliverables"]},
        "pdf_qa": str(output_dir / "qa" / "pdf-qa.json"),
        "delivery_qa": str(output_dir / "qa" / "delivery-qa.json"),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed SIIM EvoMind run directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for deterministic delivery outputs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--code-file",
        action="append",
        type=Path,
        dest="code_files",
        help="Repository-contained code file to include; repeat to override the default allowlist",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_delivery(
            args.run_dir,
            args.output_dir,
            repo_root=args.repo_root,
            code_files=args.code_files,
        )
    except DeliveryBuildError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
