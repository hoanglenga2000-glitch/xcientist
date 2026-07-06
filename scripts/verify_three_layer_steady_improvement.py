from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "workspace" / "paper_evidence_bundle_20260623.json"
PROTOCOL_PATH = ROOT / "workspace" / "steady_improvement_protocol_20260623.json"
ROUND4_PLAN_PATH = ROOT / "workspace" / "round4_search_plan_20260623.json"
ROUND3_SUMMARY_PATH = ROOT / "workspace" / "three_layer_evolution_round3_20260623.json"
ROUND4_SUMMARY_PATH = ROOT / "workspace" / "three_layer_evolution_round4_20260623.json"
ROUND3_MEMORY_PATH = ROOT / "workspace" / "retrospective_memory_round3_20260623.json"
ROUND4_MEMORY_PATH = ROOT / "workspace" / "retrospective_memory_round4_20260623.json"
VERIFICATION_PATH = ROOT / "workspace" / "three_layer_steady_improvement_verification_20260623.json"
REPORT_PATH = ROOT / "reports" / "THREE_LAYER_STEADY_IMPROVEMENT_VERIFICATION_20260623.md"

REQUIRED_LAYER_KEY_PARTS = ["multi_agent_research_os", "mlevolve_style_search_controller", "xcientist_research_harness"]
REQUIRED_ARTIFACTS = [
    "agent_trace.json",
    "artifact_manifest.json",
    "metrics.json",
    "oof_predictions.csv",
    "submission.csv",
    "search_controller_decision.json",
    "validation_contract.json",
    "claim_audit.json",
    "report.md",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_optional(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def exists_nonempty(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    return {"path": relative_path.replace("\\", "/"), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0, "passed": path.exists() and path.stat().st_size > 0}


def metric_not_worse(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate <= reference + eps if direction == "minimize" else candidate >= reference - eps


def metric_strictly_better(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate < reference - eps if direction == "minimize" else candidate > reference + eps


def artifact_json(relative_path: str) -> dict[str, Any] | None:
    path = ROOT / relative_path
    return read_json(path) if path.exists() else None


def verify_branch_artifacts(row: dict[str, Any], round_label: str) -> dict[str, Any]:
    exp_dir = Path(row["output_dir"])
    checks = [exists_nonempty(str(exp_dir / name)) for name in REQUIRED_ARTIFACTS]
    validation_payload = artifact_json(row["validation_contract"])
    claim_payload = artifact_json(row["claim_audit"])
    artifact_manifest = artifact_json(row.get("artifact_manifest") or str(exp_dir / "artifact_manifest.json"))
    search_decision = artifact_json(row.get("search_controller_decision") or str(exp_dir / "search_controller_decision.json"))
    contract_valid = bool(validation_payload) and all(key in validation_payload for key in ["schema", "task_id", "branch_id", "hypothesis", "metric", "acceptance_criteria", "risk_checklist", "required_artifacts"])
    claim_valid = bool(claim_payload) and all(key in claim_payload for key in ["schema", "task_id", "branch_id", "supporting_metrics", "audit_result", "allowed_conclusion"])
    search_valid = bool(search_decision) and bool(search_decision.get("branch_id")) and bool(search_decision.get("decision"))
    manifest_valid = bool(artifact_manifest) and bool(artifact_manifest.get("artifacts"))
    return {
        "round": round_label,
        "task_id": row["task_id"],
        "experiment_dir": str(exp_dir).replace("\\", "/"),
        "file_checks": checks,
        "all_required_files_present": all(item["passed"] for item in checks),
        "validation_contract_valid": contract_valid,
        "claim_audit_valid": claim_valid,
        "search_controller_decision_valid": search_valid,
        "artifact_manifest_valid": manifest_valid,
        "passed": all(item["passed"] for item in checks) and contract_valid and claim_valid and search_valid and manifest_valid,
    }


def verify_round3_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for row in rows:
        direction = row["direction"]
        r1 = float(row["round1_baseline"])
        r2 = float(row["round2_best_so_far"])
        r3 = float(row["round3_score"])
        final_best = float(row["final_best_so_far"])
        decision = row["round3_decision"]
        checks.append({
            "round": "round3",
            "task_id": row["task_id"],
            "decision": decision,
            "direction": direction,
            "round1": r1,
            "round2": r2,
            "candidate": r3,
            "final_best": final_best,
            "parent_not_worse_than_previous": metric_not_worse(direction, r2, r1),
            "final_not_worse_than_parent": metric_not_worse(direction, final_best, r2),
            "final_better_than_round1": metric_strictly_better(direction, final_best, r1),
            "promotion_consistent": (decision == "promote_round3" and metric_strictly_better(direction, r3, r2)) or (decision != "promote_round3" and metric_not_worse(direction, final_best, r2)),
        })
    return checks


def verify_round4_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for row in rows:
        direction = row["direction"]
        parent = float(row["round3_best_so_far"])
        score = float(row["round4_score"])
        final_best = float(row["final_best_so_far"])
        decision = row["round4_decision"]
        checks.append({
            "round": "round4",
            "task_id": row["task_id"],
            "decision": decision,
            "direction": direction,
            "parent": parent,
            "candidate": score,
            "final_best": final_best,
            "final_not_worse_than_parent": metric_not_worse(direction, final_best, parent),
            "candidate_better_than_parent": metric_strictly_better(direction, score, parent),
            "promotion_consistent": (decision == "promote_round4" and metric_strictly_better(direction, score, parent)) or (decision != "promote_round4" and metric_not_worse(direction, final_best, parent)),
        })
    return checks


def verify() -> dict[str, Any]:
    bundle = read_json_optional(BUNDLE_PATH) or {}
    protocol = read_json(PROTOCOL_PATH)
    plan = read_json(ROUND4_PLAN_PATH)
    round3 = read_json(ROUND3_SUMMARY_PATH)
    round4 = read_json_optional(ROUND4_SUMMARY_PATH)
    memory3 = read_json(ROUND3_MEMORY_PATH)
    memory4 = read_json_optional(ROUND4_MEMORY_PATH)
    r3_rows = round3.get("trajectory", [])
    r4_rows = round4.get("trajectory", []) if round4 else []
    trajectory_checks = verify_round3_trajectory(r3_rows) + verify_round4_trajectory(r4_rows)
    artifact_checks = [verify_branch_artifacts(row, "round3") for row in r3_rows] + [verify_branch_artifacts(row, "round4") for row in r4_rows]
    task_ids = {row["task_id"] for row in r3_rows}
    memory3_tasks = {record.get("task_id") for record in memory3.get("records", [])}
    memory4_tasks = {record.get("task_id") for record in (memory4 or {}).get("records", [])}
    plan_tasks = {branch.get("task_id") for branch in plan.get("branches", [])}
    round4_promoted = sum(1 for row in r4_rows if row.get("round4_decision") == "promote_round4")
    checks = [
        {"id": "three_layer_summary_present", "description": "Round3/Round4 summaries expose all three architecture layers.", "passed": all(any(part in key for key in round3.get("three_layer_evidence", {})) for part in REQUIRED_LAYER_KEY_PARTS) and (not round4 or all(any(part in key for key in round4.get("three_layer_evidence", {})) for part in REQUIRED_LAYER_KEY_PARTS)), "evidence": [rel(ROUND3_SUMMARY_PATH), rel(ROUND4_SUMMARY_PATH)] if round4 else [rel(ROUND3_SUMMARY_PATH)]},
        {"id": "best_so_far_monotonic", "description": "Best-so-far never regresses across all checked rounds and tasks.", "passed": all(item["final_not_worse_than_parent"] for item in trajectory_checks), "evidence": trajectory_checks},
        {"id": "promotion_gate_consistent", "description": "Promote/preserve decisions are consistent with metric direction and parent best.", "passed": all(item["promotion_consistent"] for item in trajectory_checks), "evidence": trajectory_checks},
        {"id": "branch_artifacts_complete", "description": "Every branch has Research OS artifacts, MLEvolve decision, XCIENTIST contract and audit.", "passed": all(item["passed"] for item in artifact_checks), "evidence": artifact_checks},
        {"id": "retrospective_memory_complete", "description": "Every task writes memory records for completed rounds.", "passed": task_ids.issubset(memory3_tasks) and (not r4_rows or task_ids.issubset(memory4_tasks)), "evidence": [rel(ROUND3_MEMORY_PATH), rel(ROUND4_MEMORY_PATH)] if memory4 else [rel(ROUND3_MEMORY_PATH)]},
        {"id": "round4_plan_ready", "description": "Round4 plan has one Search Controller branch per task with mode, hypothesis and rollback condition.", "passed": task_ids.issubset(plan_tasks) and all(branch.get("code_generation_mode") and branch.get("hypothesis") and branch.get("rollback_condition") for branch in plan.get("branches", [])), "evidence": rel(ROUND4_PLAN_PATH)},
        {"id": "round4_execution_evidence", "description": "Round4 has executed through workstation-controlled local proxy branch runner.", "passed": bool(r4_rows) and bool(round4.get("aggregate", {}).get("best_so_far_never_regressed")) and round4_promoted >= 1, "evidence": rel(ROUND4_SUMMARY_PATH) if round4 else None},
        {"id": "claim_boundary_enforced", "description": "Bundle blocks official Kaggle/GPU/MLE-Bench overclaims while allowing only local-proxy claims.", "passed": bool(bundle.get("claim_boundary", {}).get("allowed")) and bool(bundle.get("claim_boundary", {}).get("not_allowed")) and any("Kaggle" in item for item in bundle.get("claim_boundary", {}).get("not_allowed", [])), "evidence": rel(BUNDLE_PATH)},
        {"id": "paper_figures_present", "description": "Paper figure manifest is attached and all generated figure files exist.", "passed": all(exists_nonempty(path)["passed"] for fig in (bundle.get("figure_manifest_payload", {}).get("figures") or []) for path in (fig.get("paths") or {}).values()), "evidence": bundle.get("figure_manifest")},
        {"id": "protocol_certificate_present", "description": "Steady improvement protocol contains monotonicity certificate.", "passed": bool(protocol.get("monotonicity_certificate", {}).get("all_tasks_best_so_far_never_regressed")), "evidence": rel(PROTOCOL_PATH)},
    ]
    result = {
        "schema": "academic_research_os.three_layer_steady_improvement_verification.v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "claim_verified": "Three-layer architecture is verified by local proxy evidence: best-so-far protection, failure-to-memory conversion, memory-guided Round4 search, and claim-boundary audit.",
        "scope_boundary": ["Local three-task proxy evidence only.", "No official Kaggle leaderboard claim.", "No GPU/HPC execution claim for these rounds.", "No MLE-Bench 75 medal-rate or MLEvolve parity claim."],
        "checks": checks,
        "artifact_checks": artifact_checks,
        "trajectory_checks": trajectory_checks,
        "monotonicity_certificate": {"all_tasks_best_so_far_never_regressed": all(item["final_not_worse_than_parent"] for item in trajectory_checks), "tasks_total": len(task_ids), "round4_promoted_tasks": round4_promoted, "round4_preserved_parent_tasks": len(r4_rows) - round4_promoted if r4_rows else 0},
        "round4_plan": rel(ROUND4_PLAN_PATH),
        "round4_summary": rel(ROUND4_SUMMARY_PATH) if round4 else None,
        "paper_evidence_bundle": rel(BUNDLE_PATH),
    }
    return result


def write_markdown(result: dict[str, Any]) -> None:
    lines = ["# Three-layer steady-improvement verification", "", f"- Generated at: {result['generated_at']}", f"- Status: `{result['status']}`", f"- Claim: {result['claim_verified']}", "", "## Scope boundary"]
    lines += [f"- {item}" for item in result["scope_boundary"]]
    lines += ["", "## Gate checks", "", "| Gate | Status | Description |", "|---|---|---|"]
    for check in result["checks"]:
        lines.append(f"| {check['id']} | {'PASSED' if check['passed'] else 'FAILED'} | {check['description']} |")
    lines += ["", "## Trajectory checks", "", "| Round | Task | Decision | Final not worse | Promotion consistent |", "|---|---|---|---|---|"]
    for item in result["trajectory_checks"]:
        lines.append(f"| {item['round']} | {item['task_id']} | {item['decision']} | {item['final_not_worse_than_parent']} | {item['promotion_consistent']} |")
    lines += ["", "## Thesis-safe statement", "", "> The system does not guarantee every candidate branch improves. It guarantees that weaker branches cannot overwrite best-so-far, and that success/neutral/failure outcomes become memory for the next Search Controller decision under claim-audited reporting."]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify three-layer architecture and steady-improvement evidence gates.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.write_report:
        VERIFICATION_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(result)
        result["verification_path"] = rel(VERIFICATION_PATH)
        result["report_path"] = rel(REPORT_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
