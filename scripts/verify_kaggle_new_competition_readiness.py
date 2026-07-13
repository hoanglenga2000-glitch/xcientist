from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "docs" / "kaggle_new_competition_readiness.json"


def fail(message: str) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message}, ensure_ascii=False, indent=2))


def latest_experiment(task_id: str) -> Path:
    root = ROOT / "experiments" / task_id
    if not root.exists():
        fail(f"missing experiment root: {root}")
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        fail(f"no experiment runs found for {task_id}")
    return runs[-1]


def validate_hpc_gpu_report(report: dict) -> dict | None:
    hpc_run = report.get("hpc_gpu_run") or {}
    submission_validation = report.get("hpc_gpu_submission_validation") or {}
    local_dir = hpc_run.get("local_artifact_dir")
    if not hpc_run and not submission_validation:
        return None
    if hpc_run.get("status") != "passed":
        fail(f"HPC GPU run did not pass: {hpc_run.get('status')}")
    metrics = hpc_run.get("metrics") or {}
    if metrics.get("device") != "cuda" or metrics.get("cuda_available") is not True:
        fail(f"HPC GPU run did not use CUDA: {metrics}")
    if submission_validation.get("status") != "passed":
        fail(f"HPC GPU submission validation did not pass: {submission_validation}")
    if not submission_validation.get("rows_match") or not submission_validation.get("columns_match"):
        fail(f"HPC GPU submission schema mismatch: {submission_validation}")
    if int(submission_validation.get("missing_predictions") or 0) != 0:
        fail(f"HPC GPU submission has missing predictions: {submission_validation}")
    if int(submission_validation.get("invalid_prediction_count") or 0) != 0:
        fail(f"HPC GPU submission has invalid predictions: {submission_validation}")
    if local_dir and not (ROOT / str(local_dir) / "submission.csv").is_file():
        fail(f"HPC GPU submission artifact is missing: {local_dir}/submission.csv")
    return {
        "validation_status": "passed",
        "runner": metrics.get("runner"),
        "device": metrics.get("device"),
        "accuracy": (metrics.get("best") or {}).get("accuracy"),
        "submission_path": submission_validation.get("submission_path"),
        "local_artifact_dir": local_dir,
    }


def main() -> None:
    if not READINESS.is_file():
        fail("missing docs/kaggle_new_competition_readiness.json")
    report = json.loads(READINESS.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        fail(f"readiness status is not passed: {report.get('status')}")
    task_id = report.get("task_id")
    config_path = ROOT / str(report.get("config_path", ""))
    if not task_id or not config_path.is_file():
        fail("readiness report does not point to a generated config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required_config_keys = ["task", "data", "workflow", "thresholds", "scaffold", "feature_engineering"]
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        fail(f"generated config is incomplete: {missing_keys}")
    gpu_validation = validate_hpc_gpu_report(report)
    latest_run_text = None
    validation_status = None
    if gpu_validation is None:
        run_dir = latest_experiment(str(task_id))
        validation_path = run_dir / "validation_gate.json"
        submission_path = run_dir / "submission.csv"
        if not validation_path.is_file():
            fail(f"missing validation gate: {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "passed":
            fail(f"validation gate did not pass: {validation}")
        if not submission_path.is_file() or submission_path.stat().st_size == 0:
            fail(f"missing submission: {submission_path}")
        latest_run_text = str(run_dir.relative_to(ROOT)).replace("\\", "/")
        validation_status = validation.get("status")
    else:
        latest_run_text = str(gpu_validation.get("local_artifact_dir") or "")
        validation_status = gpu_validation.get("validation_status")
    print(json.dumps({
        "status": "passed",
        "task_id": task_id,
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "latest_experiment": latest_run_text,
        "metric": report.get("metric"),
        "official_download_status": (report.get("official_download") or {}).get("status"),
        "local_baseline_ready": report.get("local_baseline_ready"),
        "gpu_baseline_ready": report.get("gpu_baseline_ready"),
        "validation_status": validation_status,
        "hpc_gpu_validation": gpu_validation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
