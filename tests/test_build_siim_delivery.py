from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import build_siim_delivery as delivery


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration(rows: list[dict[str, str]], key: str) -> dict[str, list[float]]:
    ordered = sorted(rows, key=lambda item: float(item[key]))
    bins = []
    for index in range(10):
        start = len(ordered) * index // 10
        end = len(ordered) * (index + 1) // 10
        bins.append(ordered[start:end])
    return {
        "mean_predicted_probability": [sum(float(row[key]) for row in group) / len(group) for group in bins],
        "fraction_positive": [sum(int(row["target"]) for row in group) / len(group) for group in bins],
    }


def _complete_run(tmp_path: Path, *, run_id: str = "evomind_siim_delivery_fixture") -> tuple[Path, Path]:
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    _json(
        run_dir / "request.json",
        {
            "task_type": "image_classification",
            "dataset": "siim-isic-melanoma-classification",
            "submission_policy": {"official_submission": "forbidden"},
        },
    )
    _json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "status": "needs_continuation",
            "created_at": "2026-07-29T22:36:40+08:00",
            "historical_thresholds": {
                "historical_private_score": 0.92165,
                "bronze": 0.937,
                "silver": 0.9401,
                "gold": 0.9455,
            },
        },
    )
    _json(run_dir / "task_graph.json", {"schema": "fixture.task_graph.v1", "nodes": list(range(9))})
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps({"run_id": run_id, "seq": index, "event": "completed"}) + "\n" for index in range(9)),
        encoding="utf-8",
    )
    _json(
        run_dir / "dataset_profile.json",
        {
            "run_id": run_id,
            "status": "passed",
            "dataset": "siim-isic-melanoma-classification",
            "counts": {
                "files": 33129,
                "train_images": 28984,
                "test_images": 4142,
                "positive_rows": 513,
                "patients": 2056,
            },
            "complete": True,
        },
    )
    _json(
        run_dir / "hpc_runtime.json",
        {
            "run_id": run_id,
            "status": "passed",
            "job_id": "89508",
            "gpu": {"name": "NVIDIA A800-SXM4-80GB", "memory_total_mb": 81920},
            "training_status": "completed",
            "other_processes_modified": False,
            "signals_sent": 0,
        },
    )
    _json(
        run_dir / "data_audit.json",
        {
            "run_id": run_id,
            "status": "passed",
            "train_rows": 28984,
            "test_rows": 4142,
            "positive_rows": 513,
            "patients": 2056,
            "patient_group_overlap": 0,
            "content_group_overlap": 0,
            "private_label_access_count": 0,
            "target_in_test_features": False,
            "split_policy": "patient_and_content_grouped",
        },
    )
    _json(
        run_dir / "research_design.json",
        {
            "run_id": run_id,
            "status": "frozen_before_experiment",
            "primary_metric": "roc_auc",
            "validation": {"outer_folds": 5, "inner_folds": 3, "grouping": ["patient_id", "duplicate_content_group"]},
            "official_submission": "forbidden",
        },
    )
    profile_scores = {
        "raw_multiview": 0.8870,
        "border_removal": 0.8872,
        "color_constancy": 0.8868,
        "hair_suppression": 0.8871,
        "robust_combined_pipeline": 0.8882,
    }
    _json(
        run_dir / "preprocessing_ablation.json",
        {
            "run_id": run_id,
            "status": "passed",
            "seeds": [40, 41, 42],
            "profiles": [{"profile": name, "mean_roc_auc": score} for name, score in profile_scores.items()],
            "selected_profile": "robust_combined_pipeline",
            "patient_group_overlap": 0,
            "content_group_overlap": 0,
            "decision": {"mean_gain": 0.0012, "worst_fold_delta": -0.0004, "seed_passes": 3},
        },
    )
    _json(
        run_dir / "experiment_comparison.json",
        {
            "run_id": run_id,
            "status": "passed",
            "selected_profile": "robust_combined_pipeline",
            "decision": {"mean_gain": 0.0012, "worst_fold_delta": -0.0004, "seed_passes": 3, "adopted": True},
        },
    )

    positive_indices = {round(index * 28983 / 512) for index in range(513)}
    oof_rows: list[dict[str, str]] = []
    for index in range(28984):
        patient = index % 2056
        target = int(index in positive_indices)
        noise = ((index * 37) % 1000) / 1000
        image = min(0.995, 0.04 + 0.78 * noise + 0.08 * target)
        lesion = min(0.995, 0.03 + 0.76 * (((index * 53) % 1000) / 1000) + 0.10 * target)
        fusion = min(0.995, 0.5 * image + 0.35 * lesion + 0.15 * (0.08 + 0.7 * target))
        metadata = min(0.995, 0.07 + 0.58 * (((index * 29) % 1000) / 1000) + 0.11 * target)
        final = min(0.995, 0.35 * image + 0.25 * lesion + 0.28 * fusion + 0.12 * metadata)
        oof_rows.append(
            {
                "image_name": f"train_{index:05d}",
                "patient_id": f"patient_{patient:04d}",
                "leakage_group": f"patient_{patient:04d}",
                "target": str(target),
                "fold": str(patient % 5),
                "pure_image_probability": f"{image:.12f}",
                "lesion_focus_probability": f"{lesion:.12f}",
                "image_metadata_fusion_probability": f"{fusion:.12f}",
                "metadata_catboost_probability": f"{metadata:.12f}",
                "blended_probability": f"{final:.12f}",
                "probability": f"{final:.12f}",
            }
        )
    oof_fields = list(oof_rows[0])
    _csv(run_dir / "oof_predictions.csv", oof_fields, oof_rows)
    roc, pr, auc, average_precision, brier = delivery._binary_curve(oof_rows, "probability")
    assert roc and pr
    fold_rows = []
    for fold in range(5):
        selected = [row for row in oof_rows if int(row["fold"]) == fold]
        _, _, fold_auc, fold_pr, fold_brier = delivery._binary_curve(selected, "probability")
        fold_rows.append(
            {
                "fold": fold,
                "rows": len(selected),
                "roc_auc": f"{fold_auc:.15f}",
                "pr_auc": f"{fold_pr:.15f}",
                "brier": f"{fold_brier:.15f}",
            }
        )
    _csv(run_dir / "fold_metrics.csv", ["fold", "rows", "roc_auc", "pr_auc", "brier"], fold_rows)
    _json(
        run_dir / "metrics.json",
        {
            "schema": "evomind.siim.multiseed_metrics.v1",
            "run_id": run_id,
            "metric_scope": "independent_offline_patient_content_grouped_oof",
            "roc_auc": auc,
            "pr_auc": average_precision,
            "brier": brier,
            "calibration_curve": _calibration(oof_rows, "probability"),
            "patient_grouped_bootstrap_roc_auc_95ci": {
                "method": "leakage_group_cluster_bootstrap",
                "lower": max(0.0, auc - 0.04),
                "upper": min(1.0, auc + 0.04),
            },
            "fixed_oof_threshold_metrics": {
                "threshold": 0.43,
                "sensitivity": 0.81,
                "specificity": 0.83,
                "precision": 0.08,
                "negative_predictive_value": 0.996,
            },
            "historical_thresholds": {
                "historical_private_score": 0.92165,
                "bronze": 0.937,
                "silver": 0.9401,
                "gold": 0.9455,
            },
            "private_grader_execution_count": 0,
            "kaggle_submission_executed": False,
            "clinical_diagnosis_claimed": False,
        },
    )
    test_rows = [
        {"image_name": f"test_{index:05d}", "target": f"{0.05 + ((index * 17) % 900) / 1000:.12f}"}
        for index in range(4142)
    ]
    _csv(run_dir / "sample_submission.csv", ["image_name", "target"], [{"image_name": row["image_name"], "target": 0} for row in test_rows])
    _csv(run_dir / "submission.csv", ["image_name", "target"], test_rows)
    _json(
        run_dir / "training_history.json",
        {
            "run_id": run_id,
            "outer_folds": 5,
            "inner_folds": 3,
            "final_blend": {"weights": {"pure_image": 0.3, "lesion_focus": 0.2, "image_metadata_fusion": 0.35, "metadata_catboost": 0.15}},
        },
    )
    _json(
        run_dir / "training_result.json",
        {
            "run_id": run_id,
            "status": "completed",
            "formal_seeds": [43, 44, 45],
            "official_submission_executed": False,
            "private_grader_execution_count": 0,
            "private_label_access_count": 0,
            "selected_batch_size": 96,
        },
    )
    (run_dir / "hpc_telemetry.jsonl").write_text(
        "".join(json.dumps({"run_id": run_id, "seq": index, "gpu": "A800", "memory_mib": 50000}) + "\n" for index in range(12)),
        encoding="utf-8",
    )

    reviewed_names = [
        "dataset_profile.json",
        "data_audit.json",
        "research_design.json",
        "preprocessing_ablation.json",
        "metrics.json",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "submission.csv",
        "training_history.json",
        "hpc_telemetry.jsonl",
    ]
    _json(
        run_dir / "review.json",
        {
            "run_id": run_id,
            "status": "passed",
            "checks": {
                "patient_group_overlap_zero": True,
                "content_group_overlap_zero": True,
                "oof_coverage_exactly_once": True,
                "submission_schema_and_order": True,
                "private_labels_unavailable_during_training": True,
                "private_grader_not_executed": True,
                "official_submission_not_executed": True,
            },
            "artifact_hashes": {name: _sha(run_dir / name) for name in reviewed_names},
        },
    )
    freeze_records = [
        {"path": name, "sha256": _sha(run_dir / name), "bytes": (run_dir / name).stat().st_size}
        for name in reviewed_names
    ]
    freeze = _json(
        run_dir / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "frozen_at": "2026-07-30T10:00:00+08:00",
            "configuration_sha256": "a" * 64,
            "artifacts": freeze_records,
            "private_grader_execution_count_before_freeze": 0,
            "tuning_closed": True,
            "official_submission": "forbidden",
        },
    )
    grader = _json(
        run_dir / "private_grader.json",
        {
            "run_id": run_id,
            "status": "passed",
            "execution_id": "terminal-once-fixture",
            "execution_index": 1,
            "candidate_freeze_sha256": _sha(freeze),
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "mle_private_grader_score": 0.92345,
        },
    )
    _json(
        run_dir / "private_grader_ledger.json",
        {
            "run_id": run_id,
            "status": "terminal_execution_recorded",
            "execution_count": 1,
            "candidate_freeze_sha256": _sha(freeze),
            "result_sha256": _sha(grader),
            "score": 0.92345,
            "feedback_used_for_tuning": False,
            "recorded_at": "2026-07-30T10:18:00+08:00",
        },
    )
    _json(
        run_dir / "claim_audit.json",
        {
            "run_id": run_id,
            "status": "passed",
            "checks": {
                "no_public_leaderboard_claim": True,
                "no_official_medal_claim": True,
                "no_clinical_diagnosis_claim": True,
                "private_grader_not_used_for_tuning": True,
                "candidate_hashes_unchanged": True,
                "official_submission_not_executed": True,
            },
        },
    )
    code_file = tmp_path / "code" / "train.py"
    code_file.parent.mkdir(parents=True)
    code_file.write_text("print('deterministic SIIM fixture')\n", encoding="utf-8")
    return run_dir, code_file


def test_missing_evidence_fails_closed_without_outputs(tmp_path):
    run_dir = tmp_path / "partial"
    run_dir.mkdir()
    _json(run_dir / "run.json", {"run_id": "partial", "created_at": "2026-07-29T10:00:00+08:00"})
    output = tmp_path / "delivery"

    with pytest.raises(delivery.DeliveryBuildError, match="缺少真实证据"):
        delivery.build_delivery(run_dir, output, repo_root=tmp_path, code_files=[])

    assert not output.exists()


def test_complete_delivery_is_hash_bound_rendered_and_reproducible(tmp_path):
    run_dir, code_file = _complete_run(tmp_path)
    first = tmp_path / "delivery-first"
    second = tmp_path / "delivery-second"

    result = delivery.build_delivery(run_dir, first, repo_root=tmp_path, code_files=[code_file])
    reused = delivery.build_delivery(run_dir, first, repo_root=tmp_path, code_files=[code_file])
    delivery.build_delivery(run_dir, second, repo_root=tmp_path, code_files=[code_file])

    assert result["status"] == "verified"
    assert result["idempotent_reuse"] is False
    assert reused["idempotent_reuse"] is True
    assert result["run_id"] == "evomind_siim_delivery_fixture"
    for name in delivery.DOWNLOAD_NAMES:
        assert (first / name).is_file()
        assert _sha(first / name) == _sha(second / name)
    assert (first / "research_report.html").read_bytes() == (second / "research_report.html").read_bytes()
    assert (first / "artifact_manifest.json").read_bytes() == (second / "artifact_manifest.json").read_bytes()
    assert (first / "evomind-siim-isic-results.csv").read_bytes() == (run_dir / "submission.csv").read_bytes()

    manifest = json.loads((first / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["same_run_verified"] is True
    assert manifest["private_grader_execution_count"] == 1
    assert manifest["official_submission"] == "forbidden"
    assert manifest["clinical_use"] == "not_claimed"
    assert len(manifest["deliverables"]) == 4
    assert all(_sha(first / item["path"]) == item["sha256"] for item in manifest["deliverables"])

    with zipfile.ZipFile(first / "evomind-siim-isic-code.zip") as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert "code/code/train.py" in archive.namelist()
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    with zipfile.ZipFile(first / "evomind-siim-isic-evidence.zip") as archive:
        basenames = {Path(name).name for name in archive.namelist()}
        assert {"metrics.json", "review.json", "claim_audit.json", "private_grader.json", "source_manifest.json"} <= basenames
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())

    pdf_qa = json.loads((first / "qa" / "pdf-qa.json").read_text(encoding="utf-8"))
    assert pdf_qa["status"] == "passed"
    assert pdf_qa["page_count"] == delivery.PDF_PAGE_COUNT == 9
    assert len(pdf_qa["rendered_pages"]) == delivery.PDF_PAGE_COUNT
    assert all((first / item["path"]).is_file() for item in pdf_qa["rendered_pages"])
    report = (first / "research_report.html").read_text(encoding="utf-8")
    assert "ROC 曲线" in report
    assert "Precision-Recall 曲线" in report
    assert "OOF 校准曲线" in report
    assert "五个患者/内容分组外层折" in report
    assert "多通道与元数据融合" in report
    assert "EvoMind 四轮进化轨迹" in report
    assert "R1" in report and "R2" in report and "R3" in report and "R4" in report
    assert "三种子聚合并冻结候选" in report
    assert "不是临床诊断" in report


def test_evolution_rounds_are_evidence_bound_and_preserve_rejection_semantics(tmp_path):
    run_dir, _ = _complete_run(tmp_path)
    validated = delivery.validate_run(run_dir)

    rounds = delivery._evolution_rounds(validated)

    assert tuple(item["code"] for item in rounds) == ("R1", "R2", "R3", "R4")
    assert rounds[0]["metric"] == f"{dict(validated.profile_scores)['raw_multiview']:.5f}"
    assert rounds[1]["status"] == "门禁通过"
    assert "40/41/42" in rounds[1]["evidence"]
    assert rounds[2]["metric"] == f"{validated.metrics['roc_auc']:.5f}"
    assert "43/44/45" in rounds[3]["evidence"]
    assert rounds[3]["metric"] == f"{validated.metrics['grader_score']:.5f}"

    comparison = dict(validated.payloads["experiment_comparison.json"])
    comparison["decision"] = dict(comparison["decision"])
    comparison["selected_profile"] = "raw_multiview"
    comparison["decision"]["adopted"] = False
    ablation = dict(validated.payloads["preprocessing_ablation.json"])
    ablation["selected_profile"] = "raw_multiview"
    payloads = dict(validated.payloads)
    payloads["experiment_comparison.json"] = comparison
    payloads["preprocessing_ablation.json"] = ablation
    rejected = delivery._evolution_rounds(
        replace(validated, payloads=payloads, selected_profile="raw_multiview")
    )
    assert rejected[1]["status"] == "拒绝无效改动"
    assert rejected[1]["metric"] == rejected[0]["metric"]


def test_existing_delivery_drift_is_rejected_without_overwrite(tmp_path):
    run_dir, code_file = _complete_run(tmp_path)
    output = tmp_path / "delivery"
    delivery.build_delivery(run_dir, output, repo_root=tmp_path, code_files=[code_file])
    report = output / "research_report.html"
    report.write_text("drift\n", encoding="utf-8")
    before = report.read_bytes()

    with pytest.raises(delivery.DeliveryBuildError, match="确定性构建不一致"):
        delivery.build_delivery(run_dir, output, repo_root=tmp_path, code_files=[code_file])

    assert report.read_bytes() == before
    assert not list(tmp_path.glob(".delivery.building-*"))


def test_delivery_prefers_immutable_claim_source_after_workflow_normalization(tmp_path):
    run_dir, _code_file = _complete_run(tmp_path)
    direct = run_dir / "claim_audit.json"
    source = run_dir / "claim_audit_source.json"
    source.write_bytes(direct.read_bytes())
    normalized = json.loads(direct.read_text(encoding="utf-8"))
    normalized["schema"] = "evomind.siim.claim_audit_ingress.v1"
    normalized["source_claim_audit_sha256"] = _sha(source)
    _json(direct, normalized)

    validated = delivery.validate_run(run_dir)

    assert validated.evidence_paths["claim_audit.json"] == source.resolve()
    assert validated.payloads["claim_audit.json"] == json.loads(source.read_text(encoding="utf-8"))
    assert validated.payloads["claim_audit.json"].get("schema") != "evomind.siim.claim_audit_ingress.v1"


def test_private_grader_freeze_drift_blocks_delivery(tmp_path):
    run_dir, code_file = _complete_run(tmp_path)
    grader = json.loads((run_dir / "private_grader.json").read_text(encoding="utf-8"))
    grader["candidate_freeze_sha256"] = "0" * 64
    _json(run_dir / "private_grader.json", grader)

    with pytest.raises(delivery.DeliveryBuildError, match="grader 未绑定当前候选冻结"):
        delivery.build_delivery(run_dir, tmp_path / "blocked", repo_root=tmp_path, code_files=[code_file])

    assert not (tmp_path / "blocked").exists()


def test_secret_literal_in_code_package_is_rejected_and_temp_is_removed(tmp_path):
    run_dir, code_file = _complete_run(tmp_path)
    code_file.write_text('password = "not-for-a-delivery"\n', encoding="utf-8")
    output = tmp_path / "secret-blocked"

    with pytest.raises(delivery.DeliveryBuildError, match="秘密扫描规则"):
        delivery.build_delivery(run_dir, output, repo_root=tmp_path, code_files=[code_file])

    assert not output.exists()
    assert not list(tmp_path.glob(".secret-blocked.building-*"))


def test_metadata_fusion_evidence_is_mandatory(tmp_path):
    run_dir, _ = _complete_run(tmp_path)
    table = delivery._read_csv(run_dir / "oof_predictions.csv", "oof_predictions.csv")
    reduced = [
        {
            "image_name": row["image_name"],
            "patient_id": row["patient_id"],
            "leakage_group": row["leakage_group"],
            "target": row["target"],
            "fold": row["fold"],
            "probability": row["probability"],
        }
        for row in table.rows
    ]
    _csv(
        run_dir / "oof_predictions.csv",
        ["image_name", "patient_id", "leakage_group", "target", "fold", "probability"],
        reduced,
    )
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    history.pop("final_blend", None)
    _json(run_dir / "training_history.json", history)
    review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
    review["artifact_hashes"]["oof_predictions.csv"] = _sha(run_dir / "oof_predictions.csv")
    review["artifact_hashes"]["training_history.json"] = _sha(run_dir / "training_history.json")
    _json(run_dir / "review.json", review)
    freeze = json.loads((run_dir / "candidate_freeze.json").read_text(encoding="utf-8"))
    for record in freeze["artifacts"]:
        if record["path"] in {"oof_predictions.csv", "training_history.json"}:
            record["sha256"] = _sha(run_dir / record["path"])
            record["bytes"] = (run_dir / record["path"]).stat().st_size
    _json(run_dir / "candidate_freeze.json", freeze)
    grader = json.loads((run_dir / "private_grader.json").read_text(encoding="utf-8"))
    grader["candidate_freeze_sha256"] = _sha(run_dir / "candidate_freeze.json")
    _json(run_dir / "private_grader.json", grader)
    ledger = json.loads((run_dir / "private_grader_ledger.json").read_text(encoding="utf-8"))
    ledger["candidate_freeze_sha256"] = _sha(run_dir / "candidate_freeze.json")
    ledger["result_sha256"] = _sha(run_dir / "private_grader.json")
    _json(run_dir / "private_grader_ledger.json", ledger)

    with pytest.raises(delivery.DeliveryBuildError, match="元数据融合对比"):
        delivery.validate_run(run_dir)


def test_workflow_ingress_nested_seed_histories_supply_component_comparison(tmp_path):
    rows = (
        {"target": "0", "probability": "0.1"},
        {"target": "1", "probability": "0.8"},
        {"target": "0", "probability": "0.4"},
        {"target": "1", "probability": "0.7"},
    )
    oof = delivery.CsvTable(path=tmp_path / "oof.csv", fields=("target", "probability"), rows=rows)
    component_maps = (
        {"pure_image": 0.81, "lesion_focus": 0.79, "image_metadata_fusion": 0.84, "metadata_catboost": 0.76},
        {"pure_image": 0.82, "lesion_focus": 0.80, "image_metadata_fusion": 0.85, "metadata_catboost": 0.77},
        {"pure_image": 0.83, "lesion_focus": 0.81, "image_metadata_fusion": 0.86, "metadata_catboost": 0.78},
    )
    history = {
        "source_runs": [
            {"seed": seed, "history": {"final_blend": {"component_oof_auc": components}}}
            for seed, components in zip((43, 44, 45), component_maps, strict=True)
        ]
    }

    scores = delivery._component_scores(oof, history)

    assert scores[:4] == (
        ("全图影像", pytest.approx(0.82)),
        ("病灶聚焦", pytest.approx(0.80)),
        ("影像+元数据融合", pytest.approx(0.85)),
        ("CatBoost 元数据", pytest.approx(0.77)),
    )
    assert scores[-1] == ("最终三种子融合", pytest.approx(1.0))
