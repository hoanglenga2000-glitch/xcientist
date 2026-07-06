from __future__ import annotations

import argparse
from datetime import datetime
import json
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "最终上线交付状态-20260612.md"
SCREENS = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx"
JSON_SNAPSHOT = ROOT / "docs" / "final_delivery_status_20260612.json"


REQUIRED_DOC_TERMS = [
    "本地科研工作站可完整使用",
    "GPU 容器已验证",
    "http://127.0.0.1:8088",
    "house_prices",
    "titanic",
    "telco_churn",
    "专注文档",
    "专注代码",
    "report_outline_toggle",
    "report_figure_tray_toggle",
    "code_workspace_rail_toggle",
    "GPU_SSH_HOST",
    "GPU_SSH_USER",
    "GPU_SSH_PASSWORD",
    "GPU_REMOTE_WORKSPACE",
    "ANTHROPIC_API_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "Not Configured",
]


VERIFIED_TASKS = {
    "house_prices": {
        "metric": "cv_rmsle_mean",
        "max_value": 0.18,
    },
    "titanic": {
        "metric": "cv_accuracy_mean",
        "min_value": 0.78,
    },
    "telco_churn": {
        "metric": "cv_accuracy_mean",
        "min_value": 0.78,
    },
}


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {
                "status": "failed",
                "message": message,
                "evidence": evidence or {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def ready_or_verified(state: str) -> bool:
    normalized = state.lower()
    return "ready" in normalized or "verified" in normalized


def metric_value(run: dict[str, Any], metric: str) -> float | None:
    metrics = run.get("best_metrics") or {}
    value = metrics.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    metric_aliases = {
        "cv_rmsle_mean": {"rmsle", "rmse_log"},
        "cv_accuracy_mean": {"accuracy", "acc"},
    }
    metric_name = str(metrics.get("metric") or "").lower()
    best_score = metrics.get("best_score")
    if metric_name in metric_aliases.get(metric, set()) and isinstance(best_score, (int, float)):
        return float(best_score)
    return None


def read_artifact_json(run: dict[str, Any], name: str) -> dict[str, Any]:
    output_dir = run.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return {}
    path = ROOT / output_dir / name
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric_value_from_artifacts(run: dict[str, Any], metric: str) -> float | None:
    metrics = read_artifact_json(run, "metrics.json")
    validation = read_artifact_json(run, "validation_gate.json")
    model_results = read_artifact_json(run, "model_results.json")

    for payload in [metrics, validation, model_results]:
        value = payload.get(metric)
        if isinstance(value, (int, float)):
            return float(value)

    if metric == "cv_accuracy_mean":
        ensemble = metrics.get("ensemble")
        if isinstance(ensemble, dict):
            best_validation_score = ensemble.get("best_validation_score")
            if isinstance(best_validation_score, (int, float)):
                return float(best_validation_score)
            best_method = ensemble.get("best_method")
            best_payload = ensemble.get(str(best_method)) if best_method else None
            if isinstance(best_payload, dict) and isinstance(best_payload.get("accuracy"), (int, float)):
                return float(best_payload["accuracy"])
        results = model_results.get("model_results")
        best_model = model_results.get("best_model")
        best_payload = results.get(str(best_model)) if isinstance(results, dict) and best_model else None
        if isinstance(best_payload, dict) and isinstance(best_payload.get("cv_accuracy_mean"), (int, float)):
            return float(best_payload["cv_accuracy_mean"])

    if metric == "cv_rmsle_mean":
        for key in ["best_score", "best_validation_score", "cv_rmsle_mean"]:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def select_passed_run_with_metric(runs: list[dict[str, Any]], task_id: str, metric: str) -> tuple[dict[str, Any] | None, float | None]:
    for run in runs:
        if run.get("task_id") != task_id:
            continue
        if run.get("validation_gate", {}).get("status") != "passed":
            continue
        value = metric_value(run, metric)
        if value is None:
            value = metric_value_from_artifacts(run, metric)
        if value is not None:
            return run, value
    return None, None


def verify_json_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    if not JSON_SNAPSHOT.exists():
        fail("final delivery JSON snapshot is missing", {"path": str(JSON_SNAPSHOT.relative_to(ROOT))})
    snapshot = json.loads(JSON_SNAPSHOT.read_text(encoding="utf-8"))
    required_top_level = [
        "status",
        "document",
        "dashboard_url",
        "task_results",
        "connectors",
        "audited_actions",
        "ready_mode",
    ]
    missing_keys = [key for key in required_top_level if key not in snapshot]
    if missing_keys:
        fail("final delivery JSON snapshot is missing required keys", {"missing_keys": missing_keys})

    if snapshot.get("status") != "passed":
        fail("final delivery JSON snapshot is not passed", {"status": snapshot.get("status")})

    if set((snapshot.get("task_results") or {}).keys()) != set(result["task_results"].keys()):
        fail(
            "final delivery JSON snapshot task set does not match live result",
            {
                "snapshot_tasks": sorted((snapshot.get("task_results") or {}).keys()),
                "live_tasks": sorted(result["task_results"].keys()),
            },
        )

    for task_id, live_task in result["task_results"].items():
        snapshot_task = (snapshot.get("task_results") or {}).get(task_id) or {}
        if snapshot_task.get("run_id") != live_task.get("run_id") or snapshot_task.get("gate") != live_task.get("gate"):
            fail(
                "final delivery JSON snapshot task result is stale",
                {"task_id": task_id, "snapshot": snapshot_task, "live": live_task},
            )

    snapshot_connectors = snapshot.get("connectors") or {}
    for key in ["code_agent", "gpu", "kaggle"]:
        if key not in snapshot_connectors:
            fail("final delivery JSON snapshot connector is missing", {"connector": key})
        if bool(snapshot_connectors[key].get("configured")) != bool(result["connectors"][key].get("configured")):
            fail(
                "final delivery JSON snapshot connector configured state is stale",
                {
                    "connector": key,
                    "snapshot": snapshot_connectors[key],
                    "live": result["connectors"][key],
                },
            )

    if sorted(snapshot.get("audited_actions") or []) != sorted(result["audited_actions"]):
        fail(
            "final delivery JSON snapshot audited actions are stale",
            {"snapshot": snapshot.get("audited_actions"), "live": result["audited_actions"]},
        )

    if snapshot.get("ready_mode") != result["ready_mode"]:
        fail(
            "final delivery JSON snapshot ready mode is stale",
            {"snapshot": snapshot.get("ready_mode"), "live": result["ready_mode"]},
        )

    return {
        "path": str(JSON_SNAPSHOT.relative_to(ROOT)),
        "generated_at": snapshot.get("generated_at"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the final delivery status document against live workstation state.")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--write-json", action="store_true", help="Write a machine-readable final delivery status snapshot.")
    parser.add_argument("--require-json", action="store_true", help="Require and validate the machine-readable final delivery status snapshot.")
    args = parser.parse_args()

    if not DOC.exists():
        fail("final delivery document is missing", {"path": str(DOC.relative_to(ROOT))})

    doc_text = DOC.read_text(encoding="utf-8")
    missing_terms = [term for term in REQUIRED_DOC_TERMS if term not in doc_text]
    if missing_terms:
        fail("final delivery document is missing required terms", {"missing_terms": missing_terms})

    base = args.url.rstrip("/")
    summary = get_json(f"{base}/api/workstation-summary")
    tasks = {task.get("id"): task for task in summary.get("tasks", [])}
    runs = summary.get("runs", [])
    connector_status = summary.get("connector_status") or {}
    summary_final_delivery = summary.get("final_delivery_status") or {}

    missing_tasks = [task_id for task_id in VERIFIED_TASKS if task_id not in tasks]
    if missing_tasks:
        fail("verified tasks are missing from dashboard summary", {"missing_tasks": missing_tasks})

    task_results: dict[str, Any] = {}
    for task_id, rule in VERIFIED_TASKS.items():
        task_runs = [run for run in runs if run.get("task_id") == task_id and run.get("validation_gate", {}).get("status") == "passed"]
        if not task_runs:
            fail("task has no passed validation run", {"task_id": task_id})
        latest, value = select_passed_run_with_metric(runs, task_id, rule["metric"])
        if latest is None or value is None:
            fail("task metric is missing", {"task_id": task_id, "metric": rule["metric"], "candidate_runs": [run.get("id") for run in task_runs[:5]]})
        if "max_value" in rule and value > float(rule["max_value"]):
            fail("task metric exceeds threshold", {"task_id": task_id, "metric": rule["metric"], "value": value, "threshold": rule["max_value"]})
        if "min_value" in rule and value < float(rule["min_value"]):
            fail("task metric is below threshold", {"task_id": task_id, "metric": rule["metric"], "value": value, "threshold": rule["min_value"]})
        task_results[task_id] = {
            "run_id": latest.get("id"),
            "metric": rule["metric"],
            "value": value,
            "gate": latest.get("validation_gate", {}).get("status"),
        }

    code_agent = connector_status.get("code_agent") or {}
    gpu = connector_status.get("gpu") or {}
    if not code_agent:
        fail("backend code_agent connector status is missing", {"connector_status": connector_status})
    if code_agent.get("configured") and "ready" not in str(code_agent.get("state", "")).lower():
        fail("backend code_agent is configured but not reporting a ready state", {"code_agent": code_agent})
    if not code_agent.get("configured") and "not configured" not in str(code_agent.get("state", "")).lower():
        fail("backend code_agent is unavailable but not clearly marked Not Configured", {"code_agent": code_agent})
    if gpu.get("configured") and not ready_or_verified(str(gpu.get("state", ""))):
        fail("backend GPU is configured but not reporting a ready state", {"gpu": gpu})
    if not code_agent.get("configured") and "| Claude Code / Claude Agent SDK | ready |" in doc_text:
        fail("final delivery document says Claude is ready while backend reports Not Configured", {"code_agent": code_agent})
    if not gpu.get("configured") and "| GPU SSH Gateway | ready |" in doc_text:
        fail("final delivery document says GPU is ready while backend reports Not Configured", {"gpu": gpu})
    if not (connector_status.get("kaggle") or {}).get("configured") and "| Kaggle 官方提交 | ready |" in doc_text:
        fail("final delivery document says Kaggle is ready while backend reports Not Configured", {"kaggle": connector_status.get("kaggle")})

    source = SCREENS.read_text(encoding="utf-8")
    required_actions = ["report_outline_toggle", "report_figure_tray_toggle", "code_workspace_rail_toggle"]
    missing_source_actions = [action for action in required_actions if action not in source]
    if missing_source_actions:
        fail("audited UI actions are missing from workstation source", {"missing_source_actions": missing_source_actions})
    if 'data-testid="final-delivery-status-card"' not in source:
        fail("overview board is missing the final delivery status card")
    if not summary_final_delivery:
        fail("workstation summary is missing final_delivery_status")

    kaggle = connector_status.get("kaggle") or {}
    result = {
        "status": "passed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "document": str(DOC.relative_to(ROOT)),
        "dashboard_url": base,
        "task_results": task_results,
        "connectors": {
            "code_agent": {
                "configured": bool(code_agent.get("configured")),
                "state": code_agent.get("state"),
            },
            "gpu": {
                "configured": bool(gpu.get("configured")),
                "state": gpu.get("state"),
                "required_env": [
                    "GPU_SSH_HOST",
                    "GPU_SSH_USER",
                    "GPU_SSH_KEY_PATH or GPU_SSH_PASSWORD",
                    "GPU_REMOTE_WORKSPACE",
                ],
            },
            "kaggle": {
                "configured": bool(kaggle.get("configured")),
                "state": kaggle.get("state"),
                "optional": True,
            },
        },
        "audited_actions": required_actions,
        "acceptance_command": "python scripts/run_full_acceptance.py --dashboard-url http://127.0.0.1:8088 --container-name research-agent-workstation",
        "resource_smoke_command": "python scripts/run_real_resource_smoke.py --dashboard-url http://127.0.0.1:8088 --container-name research-agent-workstation --require-configured --skip-full-acceptance",
        "ready_mode": "fully_ready" if code_agent.get("configured") and gpu.get("configured") and kaggle.get("configured") else "ready_for_external_resources",
    }

    if args.write_json:
        JSON_SNAPSHOT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["json_snapshot"] = str(JSON_SNAPSHOT.relative_to(ROOT))
    elif args.require_json:
        result["json_snapshot"] = verify_json_snapshot(result)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
