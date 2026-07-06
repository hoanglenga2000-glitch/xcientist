from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
SUMMARY = ROOT / "workspace" / "kaggle4_self_evolution_rounds_20260624.json"
OUT_JSON = ROOT / "workspace" / "kaggle4_self_evolution_verification_20260624.json"
OUT_MD = ROOT / "reports" / "KAGGLE4_SELF_EVOLUTION_VERIFICATION_20260624.md"
REQUIRED_TASKS = {
    "spaceship_titanic",
    "bike_sharing_demand",
    "porto_seguro_safe_driver_prediction",
    "tabular_playground_series_aug_2022",
}
REQUIRED_ARTIFACT_FILES = [
    "metrics.json",
    "submission.csv",
    "oof_predictions.csv",
    "artifact_manifest.json",
    "agent_trace.json",
    "agent_trace.jsonl",
    "gate_audit_log.jsonl",
    "research_os_search_graph.json",
    "score_promotion_gate.json",
    "orchestrator_run.json",
    "report.md",
]

checks: list[dict] = []

def add(name: str, passed: bool, detail: str, evidence: str | None = None):
    checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail, "evidence": evidence})

if not SUMMARY.exists():
    add("summary_exists", False, "kaggle4 summary missing", str(SUMMARY))
    payload = {"tasks": []}
else:
    add("summary_exists", True, "kaggle4 summary exists", SUMMARY.relative_to(ROOT).as_posix())
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

observed_tasks = {t.get("task_id") for t in payload.get("tasks", [])}
add("four_tasks_present", observed_tasks == REQUIRED_TASKS, f"observed={sorted(observed_tasks)} expected={sorted(REQUIRED_TASKS)}", SUMMARY.relative_to(ROOT).as_posix())
add("no_official_submit", payload.get("official_kaggle_submit") is False, f"official_kaggle_submit={payload.get('official_kaggle_submit')}", SUMMARY.relative_to(ROOT).as_posix())
add("workstation_executor", "AgentOrchestrator" in str(payload.get("codex_role", "")) or "AgentOrchestrator" in json.dumps(payload, ensure_ascii=False), str(payload.get("codex_role")), SUMMARY.relative_to(ROOT).as_posix())

verified_tasks = []
for task in payload.get("tasks", []):
    task_id = task.get("task_id")
    runs = task.get("runs", [])
    add(f"{task_id}_has_multiple_runs", len(runs) >= 2, f"run_count={len(runs)}", f"experiments/{task_id}")
    improvement = task.get("improvement_within_current_metric")
    add(f"{task_id}_improved_current_metric", isinstance(improvement, (int, float)) and improvement > 0, f"improvement={improvement}", SUMMARY.relative_to(ROOT).as_posix())
    best_run_id = task.get("best_run_id")
    best_path = ROOT / "experiments" / str(task_id) / str(best_run_id)
    add(f"{task_id}_best_run_dir_exists", best_path.is_dir(), f"best_run={best_run_id}", best_path.relative_to(ROOT).as_posix() if best_path.exists() else str(best_path))
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (best_path / name).exists()]
    add(f"{task_id}_best_run_required_artifacts", not missing, f"missing={missing}", best_path.relative_to(ROOT).as_posix() if best_path.exists() else str(best_path))
    gate_path = best_path / "score_promotion_gate.json"
    gate_passed = False
    gate_detail = "missing"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        decision = gate.get("decision", {})
        gate_passed = decision.get("decision") == "promote" and decision.get("promoted") is True
        gate_detail = f"decision={decision.get('decision')} promoted={decision.get('promoted')} parent={decision.get('parent_score')} delta={decision.get('promotion_delta')}"
    add(f"{task_id}_score_promotion_gate_promoted", gate_passed, gate_detail, gate_path.relative_to(ROOT).as_posix() if gate_path.exists() else str(gate_path))
    latest_run = runs[-1] if runs else {}
    latest_run_id = latest_run.get("run_id")
    if latest_run_id and latest_run_id != best_run_id:
        latest_gate_path = ROOT / "experiments" / str(task_id) / str(latest_run_id) / "score_promotion_gate.json"
        latest_hold_passed = False
        latest_hold_detail = "missing"
        if latest_gate_path.exists():
            latest_gate = json.loads(latest_gate_path.read_text(encoding="utf-8"))
            latest_decision = latest_gate.get("decision", {})
            latest_hold_passed = latest_decision.get("decision") == "hold" and latest_decision.get("promoted") is False
            latest_hold_detail = (
                f"latest={latest_run_id} decision={latest_decision.get('decision')} "
                f"promoted={latest_decision.get('promoted')} "
                f"candidate={latest_decision.get('candidate_score')} "
                f"parent={latest_decision.get('parent_score')} "
                f"delta={latest_decision.get('promotion_delta')}"
            )
        add(
            f"{task_id}_latest_negative_ablation_held",
            latest_hold_passed,
            latest_hold_detail,
            latest_gate_path.relative_to(ROOT).as_posix() if latest_gate_path.exists() else str(latest_gate_path),
        )
    trace_path = best_path / "agent_trace.json"
    agent_count = 0
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if isinstance(trace, list):
            agent_count = len(trace)
        elif isinstance(trace, dict):
            agent_count = len(trace.get("traces", trace.get("trace", trace.get("agents", []))))
    add(f"{task_id}_agent_trace_nonempty", agent_count > 0, f"agent_count={agent_count}", trace_path.relative_to(ROOT).as_posix() if trace_path.exists() else str(trace_path))
    verified_tasks.append({
        "task_id": task_id,
        "metric": task.get("latest_metric"),
        "best_score": task.get("best_score"),
        "improvement": improvement,
        "best_run_id": best_run_id,
        "best_run_path": best_path.relative_to(ROOT).as_posix() if best_path.exists() else str(best_path),
    })

all_passed = all(c["status"] == "passed" for c in checks)
report = {
    "schema": "academic_research_os.kaggle4_self_evolution_verification.v1",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "status": "passed" if all_passed else "failed",
    "checks_passed": sum(c["status"] == "passed" for c in checks),
    "checks_total": len(checks),
    "verified_tasks": verified_tasks,
    "checks": checks,
    "claim_boundary": "All scores are local CV/proxy evidence produced by workstation runs. No official Kaggle rank, medal, or submission is claimed.",
}
OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# Kaggle 4 Self-Evolution Verification",
    "",
    f"- Status: `{report['status']}`",
    f"- Checks: `{report['checks_passed']}/{report['checks_total']}`",
    "- Official Kaggle submit: `False`",
    "- Claim boundary: local CV/proxy only; no leaderboard rank/medal claim.",
    "",
    "## Verified Tasks",
    "",
    "| task | metric | best score | improvement | best run |",
    "|---|---|---:|---:|---|",
]
for item in verified_tasks:
    lines.append(f"| `{item['task_id']}` | `{item['metric']}` | {item['best_score']} | {item['improvement']} | `{item['best_run_id']}` |")
lines += ["", "## Checks", ""]
for check in checks:
    lines.append(f"- `{check['status']}` {check['name']}: {check['detail']} ({check.get('evidence')})")
OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"status": report["status"], "checks": f"{report['checks_passed']}/{report['checks_total']}", "json": OUT_JSON.relative_to(ROOT).as_posix(), "md": OUT_MD.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))
raise SystemExit(0 if all_passed else 1)

