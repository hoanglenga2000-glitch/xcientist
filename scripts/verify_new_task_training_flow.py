from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def request_json(url: str, payload: dict | None = None, timeout: int = 240) -> dict:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {error.code}: {body}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify create-task to local training result flow through the dashboard API.")
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="Running workstation dashboard URL.")
    parser.add_argument("--timeout", type=int, default=300, help="Training request timeout in seconds.")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    create_payload = request_json(
        f"{base}/api/workstation-actions",
        {
            "action": "create_task",
            "task_id": "house_prices",
            "metadata": {
                "title": "Acceptance Runnable Research Task",
                "source": "verify_new_task_training_flow",
            },
        },
    )
    task_id = create_payload.get("task_id")
    config_path = create_payload.get("config_path")
    require(create_payload.get("ok") is True, "create_task did not return ok=true")
    require(create_payload.get("runnable") is True, "create_task did not mark the task runnable")
    require(isinstance(task_id, str) and task_id.startswith("task_"), "create_task did not return a generated task id")
    require(isinstance(config_path, str) and config_path.startswith("configs/generated/"), "generated config path is not under configs/generated")
    require((ROOT / config_path).is_file(), f"generated config file does not exist: {config_path}")

    summary_after_create = request_json(f"{base}/api/workstation-summary")
    tasks = summary_after_create.get("tasks") or []
    require(any(task.get("id") == task_id and task.get("status") == "ready_to_train" for task in tasks), "created task is not visible as ready_to_train")

    run_payload = request_json(f"{base}/api/tasks/{task_id}/run-local-experiment", {}, timeout=args.timeout)
    experiment_dir = run_payload.get("experiment_dir")
    normalized_experiment_dir = experiment_dir.replace("\\", "/") if isinstance(experiment_dir, str) else ""
    validation = run_payload.get("validation") or {}
    require(run_payload.get("ok") is True, "run-local-experiment did not return ok=true")
    require(validation.get("status") == "passed", f"validation did not pass: {validation}")
    require(normalized_experiment_dir.startswith(f"experiments/{task_id}/"), "experiment_dir is not under the generated task")
    require((ROOT / experiment_dir / "validation_gate.json").is_file(), "validation_gate.json was not written")
    require((ROOT / experiment_dir / "submission.csv").is_file(), "submission.csv was not written")

    summary_after_run = request_json(f"{base}/api/workstation-summary")
    runs = summary_after_run.get("runs") or []
    require(any(run.get("task_id") == task_id and run.get("status") == "passed" for run in runs), "passed run is not visible in dashboard summary")

    print(json.dumps({
        "status": "passed",
        "task_id": task_id,
        "config_path": config_path,
        "experiment_dir": experiment_dir,
        "validation_status": validation.get("status"),
        "primary_metric": validation.get("cv_rmsle_mean") or validation.get("cv_accuracy_mean"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
