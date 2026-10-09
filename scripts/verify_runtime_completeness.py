from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RUNTIME_FILES = [
    "task_state_machine.json",
    "agent_trace.jsonl",
    "event_log.jsonl",
    "artifact_manifest.json",
    "evidence_index.json",
    "experiment_graph.json",
    "gate_engine.json",
    "gate_audit_log.jsonl",
    "runtime_snapshot.json",
    "reflection.json",
    "reflection.md",
    "memory_records.json",
    "orchestrator_run.json",
    "validation_gate.json",
]


def fail(message: str) -> None:
    raise SystemExit(f"RUNTIME_COMPLETENESS_FAILED: {message}")


def configured_experiment_root() -> Path:
    evidence_root = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    default = Path(evidence_root) / "experiments" if evidence_root else ROOT / "experiments"
    configured = Path(os.environ.get("RESEARCH_EXPERIMENT_ROOT", default))
    return (configured if configured.is_absolute() else ROOT / configured).resolve()


def display_run_path(run_dir: Path) -> str:
    resolved = run_dir.resolve()
    root = ROOT.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    experiment_root = configured_experiment_root()
    if resolved.is_relative_to(experiment_root):
        return f"runtime/experiments/{resolved.relative_to(experiment_root).as_posix()}"
    return "runtime/experiments/external-run"


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        fail(f"missing or empty runtime file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        fail(f"missing or empty runtime file: {path.name}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_run(task_id: str) -> Path:
    task_root = configured_experiment_root() / task_id
    try:
        runs = sorted(path for path in task_root.iterdir() if path.is_dir())
    except OSError as error:
        raise SystemExit(f"RUNTIME_COMPLETENESS_FAILED: experiment root unavailable for task {task_id!r}") from error
    if not runs:
        fail(f"no experiment runs found for {task_id}")
    for run_dir in reversed(runs):
        if all((run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in REQUIRED_RUNTIME_FILES):
            return run_dir
    fail(f"no complete runtime run found for {task_id}")


def verify_task(task_id: str) -> dict[str, Any]:
    run_dir = latest_run(task_id)
    for name in REQUIRED_RUNTIME_FILES:
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            fail(f"{task_id} missing runtime artifact: {name}")

    trace = read_jsonl(run_dir / "agent_trace.jsonl")
    events = read_jsonl(run_dir / "event_log.jsonl")
    manifest = read_json(run_dir / "artifact_manifest.json")
    evidence = read_json(run_dir / "evidence_index.json")
    graph = read_json(run_dir / "experiment_graph.json")
    gates = read_json(run_dir / "gate_engine.json")
    state = read_json(run_dir / "task_state_machine.json")
    reflection = read_json(run_dir / "reflection.json")
    memory = read_json(run_dir / "memory_records.json")
    validation = read_json(run_dir / "validation_gate.json")

    if len(trace) < 5:
        fail(f"{task_id} has too few agent traces: {len(trace)}")
    if len(events) < 5:
        fail(f"{task_id} has too few runtime events: {len(events)}")
    if not manifest.get("artifacts"):
        fail(f"{task_id} artifact manifest has no artifacts")
    if not evidence.get("evidence"):
        fail(f"{task_id} evidence graph has no evidence items")
    claims = evidence.get("claims", [])
    if not claims:
        fail(f"{task_id} evidence graph has no claims")
    missing_evidence = [claim.get("claim_id") for claim in claims if not claim.get("evidence_ids")]
    if missing_evidence:
        fail(f"{task_id} has claims without evidence: {missing_evidence}")
    if not graph.get("nodes"):
        fail(f"{task_id} experiment graph has no nodes")
    gate_rows = gates.get("gates", [])
    gate_types = {gate.get("gate_type") for gate in gate_rows}
    for expected in {"PLAN_APPROVAL", "SUBMISSION_APPROVAL", "FINAL_CLAIM_APPROVAL"}:
        if expected not in gate_types:
            fail(f"{task_id} missing gate type: {expected}")
    if validation.get("status") != "passed":
        fail(f"{task_id} validation gate is not passed: {validation.get('status')}")
    if not state.get("history"):
        fail(f"{task_id} state machine has no transition history")
    if not reflection.get("next_experiment_suggestion"):
        fail(f"{task_id} reflection has no next experiment suggestion")
    if not memory:
        fail(f"{task_id} memory store is empty")

    return {
        "task_id": task_id,
        "run_dir": display_run_path(run_dir),
        "trace_count": len(trace),
        "event_count": len(events),
        "artifact_count": len(manifest.get("artifacts", [])),
        "evidence_count": len(evidence.get("evidence", [])),
        "claim_count": len(claims),
        "gate_count": len(gate_rows),
        "state": state.get("state"),
        "validation": validation.get("status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Research Agent runtime completeness for latest task runs.")
    parser.add_argument("--tasks", nargs="+", default=["titanic", "house_prices", "telco_churn"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {"status": "passed", "tasks": [verify_task(task_id) for task_id in args.tasks]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
