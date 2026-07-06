from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


RUNTIME_TASKS = ["titanic", "house_prices", "telco_churn"]


def load_summary(url: str | None) -> dict:
    if url:
        base = url.rstrip("/")
        for endpoint in ["api/workstation-summary", "api/summary"]:
            try:
                with urllib.request.urlopen(f"{base}/{endpoint}", timeout=10) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError:
                continue
        fail(f"could not load dashboard summary from {base}")

    sys.path.insert(0, str(Path.cwd()))
    from src.research_agent_workstation.dashboard import build_summary

    return build_summary()


def fail(message: str) -> None:
    raise SystemExit(f"DASHBOARD_VALIDATION_FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the dashboard summary data.")
    parser.add_argument("--url", default=None, help="Optional running dashboard URL, for example http://127.0.0.1:8088")
    args = parser.parse_args()

    summary = load_summary(args.url)
    if "runtime_by_task" in summary:
        runtime_by_task = summary.get("runtime_by_task", {})
        missing_tasks = sorted(set(RUNTIME_TASKS) - set(runtime_by_task))
        if missing_tasks:
            fail(f"missing runtime tasks: {missing_tasks}")
        task_results = {}
        for task_name in RUNTIME_TASKS:
            runtime = runtime_by_task[task_name]
            if not runtime.get("latest_experiment_dir"):
                fail(f"{task_name} has no latest experiment dir")
            if len(runtime.get("agent_trace", [])) < 5:
                fail(f"{task_name} has too few agent traces")
            if len(runtime.get("event_log", [])) < 5:
                fail(f"{task_name} has too few runtime events")
            if not runtime.get("artifact_manifest", {}).get("artifacts"):
                fail(f"{task_name} has no artifact manifest entries")
            if not runtime.get("evidence_graph", {}).get("claims"):
                fail(f"{task_name} has no claim-evidence graph")
            gates = runtime.get("gate_engine", {}).get("gates", [])
            if len(gates) < 3:
                fail(f"{task_name} has incomplete gate engine state")
            task_results[task_name] = {
                "run": runtime.get("latest_experiment_dir"),
                "state": runtime.get("task_state", {}).get("state"),
                "trace_count": len(runtime.get("agent_trace", [])),
                "gate_count": len(gates),
            }
        print(json.dumps({"status": "passed", "workstation_runtime": task_results}, ensure_ascii=False, indent=2))
        return

    tasks = {task["name"]: task for task in summary.get("tasks", [])}
    for task_name in RUNTIME_TASKS:
        task = tasks.get(task_name)
        if not task:
            fail(f"missing task in dashboard summary: {task_name}")
        if task.get("status") != "passed":
            fail(f"{task_name} gate is not passed: {task.get('status')}")
        if not task.get("all_stages_passed"):
            fail(f"{task_name} stages are not all passed")
        if not task.get("artifacts"):
            fail(f"{task_name} has no artifacts")

    plan = summary.get("plan_completion", [])
    failed_plan = [item["title"] for item in plan if item.get("status") != "passed"]
    if failed_plan:
        fail(f"plan completion has non-passed items: {failed_plan}")

    agents = summary.get("agent_templates", {})
    required_agents = {"orchestrator_planner", "analyst", "developer", "evidence_summarizer", "reviewer_gate"}
    missing_agents = sorted(required_agents - set(agents))
    if missing_agents:
        fail(f"missing agent template mappings: {missing_agents}")

    sources = summary.get("research_sources", {}).get("sources", [])
    source_ids = {source.get("id") for source in sources}
    missing_sources = sorted({"autokaggle", "agent_k", "autoresearch_ai", "kaggle_cli"} - source_ids)
    if missing_sources:
        fail(f"missing research sources: {missing_sources}")

    integrity = summary.get("research_integrity", {})
    if integrity.get("status") != "passed":
        fail(f"research integrity gate is not passed: {integrity.get('status')}")

    roadmap = summary.get("long_term_roadmap", {}).get("items", [])
    if len(roadmap) < 6:
        fail("long-term roadmap is incomplete")

    env = summary.get("environment", {})
    if env.get("server_private_agents_modified"):
        fail("server private agents must remain unmodified")
    if env.get("gpu_server_connected"):
        fail("gpu server should not be connected in this local dashboard")

    print(
        json.dumps(
            {
                "status": "passed",
                "tasks": sorted(tasks),
                "plan_items": len(plan),
                "agent_templates": sorted(agents),
                "research_sources": sorted(source_ids),
                "integrity_status": integrity.get("status"),
                "roadmap_items": len(roadmap),
                "local_only": env.get("local_only"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
