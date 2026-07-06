from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TASKS = ["house_prices", "titanic", "telco_churn"]


def run_command(command: list[str], cwd: Path | None = None, timeout: int = 300) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(os.environ.get("TEMP", "/tmp")) / "research_agent_pycache"))
    completed = subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True, env=env, timeout=timeout)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def require_success(result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "failed_command": result["command"],
                    "stdout": result["stdout"][-4000:],
                    "stderr": result["stderr"][-4000:],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def latest_experiment(task_id: str) -> Path:
    task_root = ROOT / "experiments" / task_id
    runs = sorted(path for path in task_root.iterdir() if path.is_dir())
    if not runs:
        raise SystemExit(f"SMOKE_FAILED: no experiment run found for {task_id}")
    return runs[-1]


def verify_api_summary(base_url: str) -> dict[str, Any]:
    summary = fetch_json(f"{base_url.rstrip('/')}/api/workstation-summary")
    task_ids = [task.get("id") for task in summary.get("tasks", [])]
    missing = sorted(set(REQUIRED_TASKS) - set(task_ids))
    if missing:
        raise SystemExit(f"SMOKE_FAILED: dashboard API missing tasks: {missing}")

    runtime_by_task = summary.get("runtime_by_task", {})
    runtime_missing = sorted(set(REQUIRED_TASKS) - set(runtime_by_task))
    if runtime_missing:
        raise SystemExit(f"SMOKE_FAILED: runtime_by_task missing tasks: {runtime_missing}")

    task_states = {}
    for task_id in REQUIRED_TASKS:
        runtime = runtime_by_task[task_id]
        if not runtime.get("latest_experiment_dir"):
            raise SystemExit(f"SMOKE_FAILED: {task_id} has no latest_experiment_dir")
        if len(runtime.get("agent_trace", [])) < 5:
            raise SystemExit(f"SMOKE_FAILED: {task_id} has insufficient agent_trace")
        task_states[task_id] = runtime.get("task_state", {}).get("state")

    return {
        "task_ids": task_ids,
        "task_states": task_states,
        "runtime_task_count": len(runtime_by_task),
    }


def verify_latest_validation_gates() -> list[dict[str, Any]]:
    gates = []
    for task_id in REQUIRED_TASKS:
        run_dir = latest_experiment(task_id)
        gate_path = run_dir / "validation_gate.json"
        if not gate_path.exists() or gate_path.stat().st_size == 0:
            raise SystemExit(f"SMOKE_FAILED: missing validation gate for {task_id}: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "passed":
            raise SystemExit(f"SMOKE_FAILED: validation gate not passed for {task_id}: {gate.get('status')}")
        gates.append(
            {
                "task_id": task_id,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "best_model": gate.get("best_model"),
                "metrics": {
                    key: value
                    for key, value in gate.items()
                    if key in {"cv_accuracy_mean", "holdout_accuracy", "cv_rmsle_mean", "holdout_rmsle"}
                },
            }
        )
    return gates


def check_container(container_name: str) -> dict[str, Any]:
    result = run_command(["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}} {{.Status}} {{.Ports}}"], timeout=60)
    require_success(result)
    if container_name not in result["stdout"] or "healthy" not in result["stdout"]:
        raise SystemExit(f"SMOKE_FAILED: container is not healthy: {result['stdout']}")
    return {"container": container_name, "status": result["stdout"]}


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 科研 Agent 工作站完整冒烟测试记录",
        "",
        f"- 时间：{summary['timestamp']}",
        f"- URL：{summary['dashboard_url']}",
        f"- 状态：{summary['status']}",
        "",
        "## 容器",
        "",
        f"- {summary['container']['status']}",
        "",
        "## API 与 Runtime",
        "",
        f"- API 任务：{', '.join(summary['api']['task_ids'])}",
        f"- Runtime 任务数：{summary['api']['runtime_task_count']}",
        "",
        "## Validation Gates",
        "",
    ]
    for gate in summary["validation_gates"]:
        metric_text = ", ".join(f"{key}={value}" for key, value in gate["metrics"].items())
        lines.append(f"- {gate['task_id']}：{gate['run_dir']}，{gate['best_model']}，{metric_text}")

    lines.extend(["", "## 已执行命令", ""])
    lines.extend(f"- `{item['command']}` -> `{item['returncode']}`" for item in summary["commands"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a production smoke test for the Research Agent Workstation.")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8088")
    parser.add_argument("--container-name", default="research-agent-workstation")
    parser.add_argument("--write-report", default=None)
    args = parser.parse_args()

    base_url = args.dashboard_url.rstrip("/")
    commands = [
        [sys.executable, "-m", "compileall", "src", "scripts"],
        [sys.executable, "scripts/verify_dashboard.py", "--url", base_url],
        [sys.executable, "scripts/verify_runtime_completeness.py"],
        [sys.executable, "scripts/verify_workstation_action_contract.py", "--url", base_url, "--container-name", args.container_name],
        [sys.executable, "scripts/run_full_acceptance.py", "--dashboard-url", base_url, "--container-name", args.container_name],
    ]

    command_results = []
    for command in commands:
        result = run_command(command)
        require_success(result)
        command_results.append(result)

    index_html = fetch_text(base_url)
    if "Research Agent Workstation" not in index_html and "Research Agent Lab" not in index_html:
        raise SystemExit("SMOKE_FAILED: dashboard HTML did not contain expected app title")

    summary = {
        "status": "passed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dashboard_url": base_url,
        "container": check_container(args.container_name),
        "api": verify_api_summary(base_url),
        "validation_gates": verify_latest_validation_gates(),
        "commands": command_results,
    }

    if args.write_report:
        report_path = ROOT / args.write_report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report_path, summary)
        summary["report"] = str(report_path.relative_to(ROOT))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
