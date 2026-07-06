from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ["house_prices", "titanic", "telco_churn"]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT)).replace("\\", "/")


def latest_experiment(task_id: str) -> Path | None:
    root = ROOT / "experiments" / task_id
    if not root.exists():
        return None
    runs = sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "model_results.json").exists()),
        key=lambda path: (path / "model_results.json").stat().st_mtime,
    )
    return runs[-1] if runs else None


def validation_from_stage_audit(run_dir: Path, model_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    stage_audit = read_json(run_dir / "workflow_stage_audit.json")
    if not stage_audit:
        return None
    stages = stage_audit.get("stages")
    if not isinstance(stages, list) or not all(isinstance(stage, dict) for stage in stages):
        return None
    if not stage_audit.get("all_stages_passed") and any(stage.get("status") != "passed" for stage in stages):
        return None

    model_stage = next((stage for stage in stages if stage.get("stage") == "model_validation"), None)
    submission_stage = next((stage for stage in stages if stage.get("stage") == "submission_generation"), None)
    model_checks = model_stage.get("checks") if isinstance(model_stage, dict) and isinstance(model_stage.get("checks"), dict) else {}
    submission_checks = submission_stage.get("checks") if isinstance(submission_stage, dict) and isinstance(submission_stage.get("checks"), dict) else {}
    best_model = str((model_payload or {}).get("best_model") or model_checks.get("best_model") or "")
    best_metrics = {}
    if isinstance(model_checks.get("best_metrics"), dict):
        best_metrics.update(model_checks["best_metrics"])
    if model_payload and isinstance(model_payload.get("model_results"), dict) and best_model in model_payload["model_results"]:
        best_metrics.update(model_payload["model_results"][best_model])
    if "submission_rows" in submission_checks:
        best_metrics["submission_rows"] = submission_checks["submission_rows"]
    elif submission_checks.get("rows_match") is True:
        submission_path = run_dir / "submission.csv"
        if submission_path.exists():
            with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
                best_metrics["submission_rows"] = max(0, sum(1 for _ in csv.reader(handle)) - 1)

    return {
        "status": "passed",
        "experiment_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "best_model": best_model or None,
        "metrics": best_metrics,
        "source": "workflow_stage_audit",
        "all_stages_passed": bool(stage_audit.get("all_stages_passed")),
    }


def infer_cv_metric_key(model_results: dict[str, dict[str, Any]], metric: str) -> str | None:
    preferred = {
        "rmsle": "cv_rmsle_mean",
        "accuracy": "cv_accuracy_mean",
        "auc": "cv_auc_mean",
        "f1": "cv_f1_mean",
        "mae": "cv_mae_mean",
        "rmse": "cv_rmse_mean",
    }.get(metric.lower())
    if preferred and all(preferred in metrics for metrics in model_results.values()):
        return preferred
    keys = set.intersection(*(set(metrics.keys()) for metrics in model_results.values())) if model_results else set()
    candidates = sorted(key for key in keys if key.startswith("cv_") and key.endswith("_mean"))
    return candidates[0] if candidates else None


def check_thresholds(task_id: str, config: dict[str, Any], validation: dict[str, Any] | None) -> list[dict[str, Any]]:
    thresholds = config.get("thresholds") or {}
    validation = validation or {}
    metrics = {}
    if isinstance(validation.get("metrics"), dict):
        metrics.update(validation["metrics"])
    metrics.update({key: value for key, value in validation.items() if isinstance(value, (int, float))})

    checks: list[dict[str, Any]] = []
    if "max_cv_rmsle" in thresholds:
        value = metrics.get("cv_rmsle_mean")
        checks.append({"metric": "cv_rmsle_mean", "operator": "<=", "threshold": thresholds["max_cv_rmsle"], "value": value, "passed": isinstance(value, (int, float)) and value <= thresholds["max_cv_rmsle"]})
    if "max_holdout_rmsle" in thresholds:
        value = metrics.get("holdout_rmsle")
        checks.append({"metric": "holdout_rmsle", "operator": "<=", "threshold": thresholds["max_holdout_rmsle"], "value": value, "passed": isinstance(value, (int, float)) and value <= thresholds["max_holdout_rmsle"]})
    if "min_validation_accuracy" in thresholds:
        value = metrics.get("cv_accuracy_mean")
        checks.append({"metric": "cv_accuracy_mean", "operator": ">=", "threshold": thresholds["min_validation_accuracy"], "value": value, "passed": isinstance(value, (int, float)) and value >= thresholds["min_validation_accuracy"]})
    if "expected_submission_rows" in thresholds:
        value = metrics.get("submission_rows")
        checks.append({"metric": "submission_rows", "operator": "==", "threshold": thresholds["expected_submission_rows"], "value": value, "passed": value == thresholds["expected_submission_rows"]})

    if task_id == "titanic" and not checks:
        value = metrics.get("cv_accuracy_mean")
        checks.append({"metric": "cv_accuracy_mean", "operator": ">=", "threshold": 0.78, "value": value, "passed": isinstance(value, (int, float)) and value >= 0.78})
    return checks


def inspect_task(task_id: str) -> dict[str, Any]:
    config_path = ROOT / "configs" / f"{task_id}.yaml"
    config = read_yaml(config_path)
    run_dir = latest_experiment(task_id)
    model_payload = read_json(run_dir / "model_results.json") if run_dir else None
    validation = read_json(run_dir / "validation_gate.json") if run_dir else None
    if run_dir and not validation:
        validation = validation_from_stage_audit(run_dir, model_payload)

    expected_models = list(((config.get("scaffold") or {}).get("first_stage_models") or []))
    model_results = (model_payload or {}).get("model_results") if model_payload else None
    model_results = model_results if isinstance(model_results, dict) else {}
    actual_models = sorted(model_results.keys())
    missing_models = [model for model in expected_models if model not in model_results]
    extra_models = [model for model in actual_models if model not in expected_models]
    metric = str((model_payload or {}).get("metric") or ((config.get("task") or {}).get("metric") or ""))
    selection_direction = str((model_payload or {}).get("selection_direction") or ("minimize" if metric in {"rmsle", "rmse", "mae", "logloss"} else "maximize"))
    metric_key = infer_cv_metric_key(model_results, metric)
    best_model = (model_payload or {}).get("best_model")

    ranked_models: list[dict[str, Any]] = []
    computed_best = None
    best_model_selection_valid = False
    if metric_key:
        scored = [
            (model, metrics.get(metric_key))
            for model, metrics in model_results.items()
            if isinstance(metrics, dict) and isinstance(metrics.get(metric_key), (int, float))
        ]
        reverse = selection_direction == "maximize"
        ranked = sorted(scored, key=lambda item: item[1], reverse=reverse)
        ranked_models = [{"model": model, metric_key: value} for model, value in ranked]
        computed_best = ranked[0][0] if ranked else None
        best_model_selection_valid = bool(computed_best and computed_best == best_model)

    threshold_checks = check_thresholds(task_id, config, validation)
    validation_gate_passed = bool(validation and validation.get("status") == "passed")
    candidates_complete = bool(expected_models) and not missing_models and len(actual_models) >= len(expected_models)
    metrics_passed = all(check["passed"] for check in threshold_checks)
    ready = bool(run_dir and validation_gate_passed and candidates_complete and best_model_selection_valid and metrics_passed)

    return {
        "task_id": task_id,
        "config_path": rel(config_path),
        "latest_experiment": rel(run_dir),
        "metric": metric,
        "selection_direction": selection_direction,
        "selection_metric_key": metric_key,
        "expected_candidate_models": expected_models,
        "actual_candidate_models": actual_models,
        "candidate_model_count": len(actual_models),
        "configured_candidate_count": len(expected_models),
        "missing_candidate_models": missing_models,
        "extra_candidate_models": extra_models,
        "best_model_recorded": best_model,
        "best_model_computed": computed_best,
        "best_model_selection_valid": best_model_selection_valid,
        "ranked_models": ranked_models,
        "threshold_checks": threshold_checks,
        "validation_gate_passed": validation_gate_passed,
        "ready_for_optimized_local_training": ready,
    }


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# \u8bad\u7ec3\u4f18\u5316\u4e0e\u4efb\u52a1\u5b8c\u6210\u7387\u5c31\u7eea\u5ba1\u8ba1",
        "",
        f"- \u751f\u6210\u65f6\u95f4\uff1a{report['generated_at']}",
        f"- \u603b\u4f53\u72b6\u6001\uff1a{report['overall_status']}",
        f"- \u672c\u5730\u4efb\u52a1\u5b8c\u6210\u7387\uff1a{report['completion_rate_percent']}% ({report['ready_task_count']}/{report['required_task_count']})",
        "",
        "## \u7ed3\u8bba",
        "",
        report["conclusion"],
        "",
        "## \u4efb\u52a1\u660e\u7ec6",
        "",
    ]
    # Labels are pre-computed here (not inline in the f-string expression part)
    # because Python < 3.12 forbids backslashes inside f-string `{}` expressions,
    # and these \u escapes would otherwise raise SyntaxError.
    pass_label = "\u901a\u8fc7"
    fail_label = "\u672a\u901a\u8fc7"
    yes_label = "\u662f"
    no_label = "\u5426"
    for task in report["tasks"]:
        best_model_label = pass_label if task["best_model_selection_valid"] else fail_label
        gate_label = pass_label if task["validation_gate_passed"] else fail_label
        ready_label = yes_label if task["ready_for_optimized_local_training"] else no_label
        lines.extend([
            f"### {task['task_id']}",
            f"- \u6700\u65b0\u5b9e\u9a8c\uff1a{task['latest_experiment']}",
            f"- \u6a21\u578b\u9009\u62e9\u65b9\u5411\uff1a{task['selection_direction']} / {task['selection_metric_key']}",
            f"- \u914d\u7f6e\u5019\u9009\u6a21\u578b\u6570\uff1a{task['configured_candidate_count']}",
            f"- \u5b9e\u9645\u5019\u9009\u6a21\u578b\u6570\uff1a{task['candidate_model_count']}",
            f"- \u8bb0\u5f55\u6700\u4f73\u6a21\u578b\uff1a{task['best_model_recorded']}",
            f"- \u91cd\u65b0\u8ba1\u7b97\u6700\u4f73\u6a21\u578b\uff1a{task['best_model_computed']}",
            f"- \u6700\u4f73\u6a21\u578b\u9009\u62e9\u6821\u9a8c\uff1a{best_model_label}",
            f"- Validation Gate\uff1a{gate_label}",
            f"- \u4f18\u5316\u8bad\u7ec3\u5c31\u7eea\uff1a{ready_label}",
        ])
        if task["missing_candidate_models"]:
            lines.append(f"- \u7f3a\u5c11\u5019\u9009\u6a21\u578b\uff1a{', '.join(task['missing_candidate_models'])}")
        for check in task["threshold_checks"]:
            check_label = pass_label if check["passed"] else fail_label
            lines.append(
                f"- \u6307\u6807 {check['metric']} {check['operator']} {check['threshold']}\uff1a"
                f"\u5f53\u524d {check['value']}\uff0c{check_label}"
            )
        if task["ranked_models"]:
            top = task["ranked_models"][:4]
            lines.append("- \u5019\u9009\u6a21\u578b\u6392\u5e8f\uff1a" + "; ".join(f"{item['model']}={item[task['selection_metric_key']]}" for item in top))
        lines.append("")

    lines.extend([
        "## \u4e0a\u7ebf\u542b\u4e49",
        "",
        "1. \u5f53\u524d 3 \u4e2a\u672c\u5730 Kaggle \u98ce\u683c\u4efb\u52a1\u5747\u6709\u53ef\u590d\u6d4b\u8bad\u7ec3\u4ea7\u7269\u3001Gate \u548c\u6a21\u578b\u5019\u9009\u5bf9\u6bd4\u3002",
        "2. \u8be5\u7ed3\u8bba\u53ea\u8bf4\u660e\u672c\u5730\u4f18\u5316\u8bad\u7ec3 readiness \u901a\u8fc7\uff0c\u4e0d\u4ee3\u8868\u5b98\u65b9 Kaggle \u6392\u540d\u3001\u5956\u724c\u6216 MLE-Bench 75 \u4efb\u52a1\u8fbe\u6807\u3002",
        "3. \u540e\u7eed\u5927\u89c4\u6a21\u8bad\u7ec3\u4ecd\u9700\u901a\u8fc7 HPC/GPU \u8d44\u6e90\u95e8\u7981\u3001\u7f13\u5b58\u95e8\u7981\u3001claim audit \u548c\u4eba\u5de5\u63d0\u4ea4 Gate\u3002",
    ])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local training optimization readiness across required tabular tasks.")
    parser.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under docs/.")
    args = parser.parse_args()

    tasks = [inspect_task(task_id) for task_id in TASK_IDS]
    ready_count = sum(1 for task in tasks if task["ready_for_optimized_local_training"])
    total = len(tasks)
    completion_rate = round(ready_count / total * 100, 2) if total else 0.0
    all_ready = ready_count == total
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "passed" if all_ready else "failed",
        "required_task_count": total,
        "ready_task_count": ready_count,
        "completion_rate_percent": completion_rate,
        "tasks": tasks,
        "conclusion": "\u672c\u5730\u8bad\u7ec3\u4efb\u52a1\u5b8c\u6210\u7387\u4e3a 100%\uff0c\u4e14\u6bcf\u4e2a\u4efb\u52a1\u7684\u6700\u4f73\u6a21\u578b\u5747\u7531\u5019\u9009\u6a21\u578b\u6307\u6807\u91cd\u65b0\u8ba1\u7b97\u9a8c\u8bc1\u901a\u8fc7\u3002" if all_ready else "\u4ecd\u6709\u672c\u5730\u8bad\u7ec3\u4efb\u52a1\u672a\u5b8c\u6210\u5019\u9009\u6a21\u578b\u3001\u6307\u6807\u9608\u503c\u6216\u6700\u4f73\u6a21\u578b\u9009\u62e9\u6821\u9a8c\u3002",
    }

    if args.write_report:
        json_path = ROOT / "docs" / "training_optimization_readiness.json"
        md_path = ROOT / "docs" / "training_optimization_readiness.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {"json": rel(json_path), "markdown": rel(md_path)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
