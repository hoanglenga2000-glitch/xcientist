from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a generic staged tabular experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        fail(f"missing or empty {label}: {path}")


def resolve_path(path_text: str, experiment_dir: Path | None = None) -> Path:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute() and path.exists():
        return path
    evidence_root_text = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    if evidence_root_text and not path.is_absolute():
        candidate = Path(evidence_root_text).resolve() / path
        if candidate.exists():
            return candidate
    normalized = path_text.replace("\\", "/")
    marker = "/experiments/"
    if marker in normalized:
        candidate = Path.cwd() / "experiments" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    if experiment_dir is not None:
        candidate = experiment_dir / path.name
        if candidate.exists():
            return candidate
    if path.is_absolute():
        return path
    return Path.cwd() / path


def validate_metric(config: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["thresholds"]
    evaluation = log["evaluation"]
    best_model = evaluation["best_model"]
    best_metrics = evaluation["model_results"][best_model]
    metric = config["task"]["metric"]

    if config["task"]["type"] == "regression":
        metric_name = str(metric).lower()
        cv_key = f"cv_{metric_name}_mean"
        holdout_key = f"holdout_{metric_name}"
        cv = best_metrics[cv_key]
        holdout = best_metrics[holdout_key]
        cv_threshold = thresholds.get(f"max_cv_{metric_name}")
        holdout_threshold = thresholds.get(f"max_holdout_{metric_name}")
        if cv_threshold is not None and cv > cv_threshold:
            fail(f"{cv_key} {cv} > {cv_threshold}")
        if holdout_threshold is not None and holdout > holdout_threshold:
            fail(f"{holdout_key} {holdout} > {holdout_threshold}")
        return {"best_model": best_model, cv_key: cv, holdout_key: holdout, "metric_gate_mode": "threshold" if cv_threshold is not None or holdout_threshold is not None else "advisory"}

    cv = best_metrics["cv_accuracy_mean"]
    holdout = best_metrics["holdout_accuracy"]
    threshold = thresholds.get("min_validation_accuracy")
    if threshold is not None and cv < threshold:
        fail(f"cv_accuracy_mean {cv} < {thresholds['min_validation_accuracy']}")
    return {"best_model": best_model, "cv_accuracy_mean": cv, "holdout_accuracy": holdout, "metric_gate_mode": "threshold" if threshold is not None else "advisory"}


def validate_submission(config: dict[str, Any], log: dict[str, Any], experiment_dir: Path) -> dict[str, Any]:
    thresholds = config["thresholds"]
    submission_check = log["submission_check"]
    quality = log["data_quality"]

    if thresholds.get("require_train_test_features_match", True) and not quality["train_test_feature_columns_match"]:
        fail("train/test feature columns do not match")
    if thresholds.get("require_submission_schema_valid", True) and not submission_check["valid"]:
        fail("submission schema check did not pass")
    if thresholds.get("require_no_missing_predictions", True) and submission_check["missing_predictions"] != 0:
        fail(f"missing predictions: {submission_check['missing_predictions']}")
    if thresholds.get("require_positive_predictions", False) and not submission_check.get("positive_predictions", False):
        fail("submission has non-positive predictions")

    submission_path = resolve_path(submission_check["path"], experiment_dir)
    require_file(submission_path, "submission")
    sample_path = resolve_path(config["data"]["sample_submission"])
    require_file(sample_path, "sample submission")

    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    expected_columns = thresholds.get("expected_submission_columns", sample.columns.tolist())
    expected_rows = thresholds.get("expected_submission_rows", len(sample))

    if submission.columns.tolist() != expected_columns:
        fail(f"submission columns {submission.columns.tolist()} != expected columns {expected_columns}")
    if submission.columns.tolist() != sample.columns.tolist():
        fail(f"submission columns {submission.columns.tolist()} != sample columns {sample.columns.tolist()}")
    if len(submission) != expected_rows:
        fail(f"submission rows {len(submission)} != expected rows {expected_rows}")
    if len(submission) != len(sample):
        fail(f"submission rows {len(submission)} != sample rows {len(sample)}")
    if submission.iloc[:, 1].isna().any():
        fail("submission has missing prediction values")
    if thresholds.get("require_positive_predictions", False) and not (submission.iloc[:, 1] > 0).all():
        fail("submission prediction values must be positive")

    return {
        "submission_rows": len(submission),
        "submission_columns": submission.columns.tolist(),
        "submission_path": str(submission_path),
    }


def validate_stage_audit(config: dict[str, Any], experiment_dir: Path) -> dict[str, Any]:
    stage_audit_path = experiment_dir / "workflow_stage_audit.json"
    require_file(stage_audit_path, "workflow stage audit json")
    stage_audit = json.loads(stage_audit_path.read_text(encoding="utf-8"))

    if not stage_audit.get("all_stages_passed"):
        fail(f"workflow stage audit failed: {stage_audit.get('failed_stages')}")
    required_stages = set(config["workflow"])
    audited_stages = {stage["stage"] for stage in stage_audit.get("stages", [])}
    missing_stages = sorted(required_stages - audited_stages)
    if missing_stages:
        fail(f"workflow stage audit missing stages: {missing_stages}")
    if stage_audit.get("server_private_templates_modified"):
        fail("server private templates should not be modified")

    return {
        "workflow_version": stage_audit.get("workflow_version"),
        "stage_audit_present": True,
        "all_stages_passed": True,
    }


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    required_files = [
        (experiment_dir / "experiment_log.json", "experiment log"),
        (experiment_dir / "data_quality.json", "data quality report"),
        (experiment_dir / "model_results.json", "model results"),
        (experiment_dir / "submission.csv", "submission"),
        (experiment_dir / "task_scaffold.json", "task scaffold json"),
        (experiment_dir / "task_scaffold.md", "task scaffold markdown"),
        (experiment_dir / "post_scaffold_improvement.json", "post scaffold json"),
        (experiment_dir / "post_scaffold_improvement.md", "post scaffold markdown"),
        (experiment_dir / "workflow_stage_audit.json", "workflow stage audit json"),
        (experiment_dir / "workflow_stage_audit.md", "workflow stage audit markdown"),
        (experiment_dir / "local_report.md", "markdown report"),
        (experiment_dir / "local_report.docx", "docx report"),
    ]
    for path, label in required_files:
        require_file(path, label)

    log = json.loads((experiment_dir / "experiment_log.json").read_text(encoding="utf-8"))
    metric_summary = validate_metric(config, log)
    submission_summary = validate_submission(config, log, experiment_dir)
    audit_summary = validate_stage_audit(config, experiment_dir)

    summary = {
        "status": "passed",
        "task": config["task"]["name"],
        "experiment_dir": str(experiment_dir),
        **metric_summary,
        **submission_summary,
        "scaffold_present": True,
        "post_scaffold_present": True,
        "reports_present": True,
        **audit_summary,
    }
    if os.environ.get("RESEARCH_AGENT_READ_ONLY_ACCEPTANCE") != "1":
        gate_path = experiment_dir / "validation_gate.json"
        gate_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
