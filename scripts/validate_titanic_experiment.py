from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Titanic local Kaggle experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--config", default="configs/titanic.yaml")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        fail(f"missing or empty {label}: {path}")


def normalize_recorded_path(path_text: str) -> Path:
    return Path(path_text.replace("\\", "/"))


def resolve_evidence_path(path_text: str, experiment_dir: Path | None = None) -> Path:
    path = normalize_recorded_path(path_text)
    if path.is_absolute() and path.exists():
        return path
    evidence_root_text = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    if evidence_root_text and not path.is_absolute():
        candidate = Path(evidence_root_text).resolve() / path
        if candidate.exists():
            return candidate
    if experiment_dir is not None:
        candidate = experiment_dir / path.name
        if candidate.exists():
            return candidate
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thresholds = config["thresholds"]

    log_path = experiment_dir / "experiment_log.json"
    quality_path = experiment_dir / "data_quality.json"
    results_path = experiment_dir / "model_results.json"
    report_path = experiment_dir / "titanic_local_report.md"
    report_docx_path = experiment_dir / "titanic_local_report.docx"
    scaffold_path = experiment_dir / "task_scaffold.json"
    scaffold_md_path = experiment_dir / "task_scaffold.md"
    stage_audit_path = experiment_dir / "workflow_stage_audit.json"
    stage_audit_md_path = experiment_dir / "workflow_stage_audit.md"

    for path, label in [
        (log_path, "experiment log"),
        (quality_path, "data quality report"),
        (results_path, "model results"),
        (report_path, "markdown report"),
        (report_docx_path, "docx report"),
        (scaffold_path, "task scaffold json"),
        (scaffold_md_path, "task scaffold markdown"),
        (stage_audit_path, "workflow stage audit json"),
        (stage_audit_md_path, "workflow stage audit markdown"),
    ]:
        require_file(path, label)

    log: dict[str, Any] = json.loads(log_path.read_text(encoding="utf-8"))
    stage_audit: dict[str, Any] = json.loads(stage_audit_path.read_text(encoding="utf-8"))
    best_model = log["evaluation"]["best_model"]
    best_metrics = log["evaluation"]["model_results"][best_model]
    submission_check = log["submission_check"]
    quality = log["data_quality"]

    if best_metrics["cv_accuracy_mean"] < thresholds["min_validation_accuracy"]:
        fail(
            f"cv_accuracy_mean {best_metrics['cv_accuracy_mean']} < "
            f"{thresholds['min_validation_accuracy']}"
        )
    if thresholds.get("require_submission_schema_valid", True) and not submission_check["valid"]:
        fail("submission schema check did not pass")
    if thresholds.get("require_no_missing_predictions", True) and submission_check["missing_predictions"] != 0:
        fail(f"missing predictions: {submission_check['missing_predictions']}")
    if thresholds.get("require_train_test_features_match", True) and not quality["train_test_feature_columns_match"]:
        fail("train/test feature columns do not match")
    if not stage_audit.get("all_stages_passed"):
        fail(f"workflow stage audit failed: {stage_audit.get('failed_stages')}")
    required_stages = set(config["workflow"])
    audited_stages = {stage["stage"] for stage in stage_audit.get("stages", [])}
    missing_stages = sorted(required_stages - audited_stages)
    if missing_stages:
        fail(f"workflow stage audit missing stages: {missing_stages}")

    submission_path = resolve_evidence_path(submission_check["path"], experiment_dir)
    require_file(submission_path, "submission")

    sample_path = resolve_evidence_path(config["data"]["sample_submission"])
    require_file(sample_path, "sample submission")
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    if submission.columns.tolist() != sample.columns.tolist():
        fail(f"submission columns {submission.columns.tolist()} != sample columns {sample.columns.tolist()}")
    if len(submission) != len(sample):
        fail(f"submission rows {len(submission)} != sample rows {len(sample)}")
    if submission.iloc[:, 1].isna().any():
        fail("submission has missing prediction values")

    summary = {
        "status": "passed",
        "experiment_dir": str(experiment_dir),
        "best_model": best_model,
        "cv_accuracy_mean": best_metrics["cv_accuracy_mean"],
        "holdout_accuracy": best_metrics["holdout_accuracy"],
        "submission_rows": len(submission),
        "submission_columns": submission.columns.tolist(),
        "scaffold_present": True,
        "stage_audit_present": True,
        "reports_present": True,
    }
    if os.environ.get("RESEARCH_AGENT_READ_ONLY_ACCEPTANCE") != "1":
        gate_path = experiment_dir / "validation_gate.json"
        gate_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
