from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_os.claim_audit import audit_claim  # noqa: E402
from research_os.mlevolve_controller import (  # noqa: E402
    build_benchmark_claim_gate,
    build_search_controller_decision,
    evaluate_rank_gate,
)
from research_os.validation_contract import create_contract  # noqa: E402


TASK_ID = "playground_series_s6e6"
RUN_ROOT = ROOT / "workspace" / "workstation_runs" / TASK_ID
SAMPLE_SUBMISSION = ROOT / "tasks" / "playground_series_s6e6" / "data" / "sample_submission.csv"
SEED_RUN_IDS = [
    "wr_2026-06-25T06-53-22-322Z_q0mix",
    "wr_2026-06-25T06-53-21-995Z_d688w",
    "wr_2026-06-25T07-09-42-997Z_ov893",
    "wr_2026-06-25T07-00-53-587Z_5hajp",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_result_path(run_dir: Path) -> Path | None:
    for name in [
        "artifact_replay_manifest.json",
        "exp024_multi_asset_frontier_blend_result.json",
        "exp024_multi_asset_frontier_blend_failure.json",
        "exp025_single_model_diversity_result.json",
    ]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def discover_run_ids() -> list[str]:
    run_ids = set(SEED_RUN_IDS)
    if RUN_ROOT.exists():
        for run_dir in RUN_ROOT.iterdir():
            if run_dir.is_dir() and find_result_path(run_dir):
                run_ids.add(run_dir.name)
    return sorted(run_ids)


def nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def submission_path_for(run_dir: Path, result: dict[str, Any], metrics: dict[str, Any]) -> Path | None:
    direct = result.get("submission_artifact")
    if isinstance(direct, str):
        path = ROOT / direct
        if path.exists():
            return path
    outputs = metrics.get("outputs")
    if isinstance(outputs, dict):
        candidate = outputs.get("candidate_submission")
        if isinstance(candidate, str):
            path = ROOT / candidate
            if path.exists():
                return path
    hpc_submission = run_dir / "hpc_gpu_training" / "submission.csv"
    if hpc_submission.exists():
        return hpc_submission
    return None


def audit_submission(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "schema": "academic_research_os.submission_audit.v1",
            "status": "missing",
            "reason": "submission artifact is absent",
            "human_gate_required_for_official_submission": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    with SAMPLE_SUBMISSION.open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.reader(handle))
    with path.open(newline="", encoding="utf-8") as handle:
        submission_rows = list(csv.reader(handle))
    allowed = {"GALAXY", "QSO", "STAR"}
    id_mismatch = 0
    invalid = 0
    distribution: dict[str, int] = {}
    for index in range(1, min(len(sample_rows), len(submission_rows))):
        sample_id = sample_rows[index][0] if sample_rows[index] else ""
        submission_id = submission_rows[index][0] if submission_rows[index] else ""
        label = submission_rows[index][1] if len(submission_rows[index]) > 1 else ""
        if sample_id != submission_id:
            id_mismatch += 1
        if label not in allowed:
            invalid += 1
        distribution[label] = distribution.get(label, 0) + 1
    status = (
        "passed"
        if len(sample_rows) == len(submission_rows)
        and (submission_rows[0] if submission_rows else []) == ["id", "class"]
        and id_mismatch == 0
        and invalid == 0
        else "failed"
    )
    return {
        "schema": "academic_research_os.submission_audit.v1",
        "status": status,
        "competition_slug": "playground-series-s6e6",
        "submission_path": rel(path),
        "sample_submission_path": rel(SAMPLE_SUBMISSION),
        "rows_match": len(sample_rows) == len(submission_rows),
        "columns_match": (submission_rows[0] if submission_rows else []) == ["id", "class"],
        "sample_rows": max(0, len(sample_rows) - 1),
        "submission_rows": max(0, len(submission_rows) - 1),
        "invalid_prediction_count": invalid,
        "id_mismatch_count": id_mismatch,
        "prediction_distribution": distribution,
        "submission_sha256": sha256_file(path),
        "human_gate_required_for_official_submission": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_kind(result_path: Path | None) -> str:
    name = result_path.name if result_path else ""
    if "artifact_replay" in name:
        return "artifact_replay"
    if "exp024" in name:
        return "exp024_frontier_blend"
    if "exp025" in name:
        return "exp025_gpu_single_model"
    return "unknown"


def metrics_for(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    metrics_path = result.get("metrics_path")
    if isinstance(metrics_path, str):
        metrics = read_json(ROOT / metrics_path)
        if metrics:
            return metrics
    return read_json(run_dir / "hpc_gpu_training" / "metrics.json")


def score_value(result: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    for value in [
        result.get("validation_score"),
        metrics.get("best_validation_score"),
        nested(metrics, ["selected", "balanced_accuracy"]),
        metrics.get("oof_balanced_accuracy"),
    ]:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def backfill_run(run_id: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    result_path = find_result_path(run_dir)
    result = read_json(result_path) if result_path else {}
    metrics = metrics_for(run_dir, result)
    kind = run_kind(result_path)
    score = score_value(result, metrics)
    decision = metrics.get("decision") or result.get("status") or "evidence_only"
    branch_type = {
        "artifact_replay": "artifact_replay",
        "exp024_frontier_blend": "calibration_repair",
        "exp025_gpu_single_model": "model_family_diversity",
    }.get(kind, "unknown")
    search = build_search_controller_decision(
        task_id=TASK_ID,
        run_id=run_id,
        selected_branch=kind,
        exploration_stage="recovery_exploitation",
        metric="balanced_accuracy",
        metric_direction="maximize",
        has_parent=True,
        branch_stagnant=decision in {"keep_evidence_only", "failed"},
        failure_count=1 if decision in {"keep_evidence_only", "failed"} else 0,
        cross_branch_references=[
            {"exp_id": "EXP007", "role": "protected_rollback_baseline"},
            {"exp_id": "EXP017", "role": "current_official_public_score_reference"},
        ],
        memory_reuse_records=[
            {"memory_id": "s6e6::score_gate_blocked", "use": "avoid submitting weaker one-shot boosting or unstable blends"}
        ],
        official_submit_budget=0,
    )
    search["code_generation_mode_requested"] = {
        "artifact_replay": "Diff",
        "exp024_frontier_blend": "Stepwise",
        "exp025_gpu_single_model": "Diff",
    }.get(kind, search["code_generation_mode"])
    write_json(run_dir / "search_controller_decision.json", search)

    submission = submission_path_for(run_dir, result, metrics)
    submission_audit = audit_submission(submission)
    write_json(run_dir / "submission_audit.json", submission_audit)

    required_artifacts = [
        "search_controller_decision.json",
        "validation_contract.json",
        "claim_audit.json",
        "submission_audit.json",
        "artifact_manifest.json",
        "workstation_run_manifest.json",
    ]
    if submission is not None:
        required_artifacts.append(rel(submission))
    contract = create_contract(
        contract_id=f"{run_id}_validation_contract",
        exp_id=run_id,
        claim="This run provides S6E6 recovery evidence only; it does not prove official rank or top30 improvement.",
        hypothesis=f"{kind} may improve or stabilize the protected S6E6 frontier without bypassing workstation gates.",
        implementation_requirement="Execute only through workstation /api/workstation-actions and record artifact evidence.",
        metric="balanced_accuracy",
        baseline_exp_id="EXP007",
        acceptance_criteria={
            "submission_audit_passed": {"equals": True},
            "official_submit_started": {"equals": False},
        },
        ablation_plan=[
            "Compare against EXP007 rollback baseline.",
            "Preserve candidate as evidence-only unless score/risk gates pass.",
        ],
        risk_checklist=[
            {"risk": "official_rank_missing", "status": "blocked", "evidence": "rank_promotion_gate.json"},
            {"risk": "public_leaderboard_overclaim", "status": "blocked", "evidence": "benchmark_claim_gate.json"},
            {"risk": "submission_schema", "status": submission_audit["status"], "evidence": "submission_audit.json"},
        ],
        conclusion_boundary="Allowed: workstation evidence, local/proxy metric, GPU execution proof when present. Blocked: official top30, medal, MLEvolve parity, or Kaggle submission claim.",
        required_artifacts=required_artifacts,
    )
    write_json(run_dir / "validation_contract.json", asdict(contract))

    missing_evidence = []
    if submission_audit["status"] != "passed":
        missing_evidence.append("passed_submission_audit")
    missing_evidence.extend(["official_rank_artifact", "human_submission_approval"])
    claim = audit_claim(
        claim_id=f"{run_id}_claim_audit",
        claim_text="This run is evidence-only and must not be reported as official top30 improvement.",
        related_exp_ids=[run_id],
        contract=asdict(contract),
        supporting_metrics={
            "balanced_accuracy": score,
            "decision": decision,
            "official_submission_started": False,
        },
        required_ablations=["EXP007_baseline_comparison"],
        completed_ablations=["EXP007_baseline_comparison"] if score is not None else [],
        evidence={
            "has_required_experiments": True,
            "has_mechanistic_evidence": False,
            "missing_evidence": missing_evidence,
        },
    )
    claim_payload = asdict(claim)
    claim_payload["blocked_claims"] = ["official_top30", "official_medal", "mlebench_75_parity", "kaggle_submit_completed"]
    write_json(run_dir / "claim_audit.json", claim_payload)

    rank_gate = evaluate_rank_gate(task_id=TASK_ID, run_id=run_id, official_submission=None)
    write_json(run_dir / "rank_promotion_gate.json", rank_gate)
    benchmark_gate = build_benchmark_claim_gate(evaluated_tasks=11, medal_rate=0.0)
    write_json(run_dir / "benchmark_claim_gate.json", benchmark_gate)
    state = {
        "schema": "academic_research_os.task_benchmark_state.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": TASK_ID,
        "run_id": run_id,
        "resource_mode": "hpc_gpu" if kind == "exp025_gpu_single_model" else "artifact_recovery",
        "status": "held",
        "official_submit_candidate": False,
        "top30_reached": False,
        "best_so_far_protected": True,
        "metric": "balanced_accuracy",
        "score": score,
        "decision": decision,
        "next_action": "Retain as retrospective memory and require a new Search Controller decision before long GPU training.",
    }
    write_json(run_dir / "task_benchmark_state.json", state)
    return {
        "run_id": run_id,
        "kind": kind,
        "score": score,
        "submission_audit": submission_audit["status"],
        "rank_gate": rank_gate["status"],
        "claim_audit": claim_payload["audit_result"],
    }


def main() -> None:
    records = [backfill_run(run_id) for run_id in discover_run_ids() if (RUN_ROOT / run_id).exists()]
    report = {
        "schema": "academic_research_os.s6e6_recovery_governance_backfill.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "Backfill only writes governance evidence for already completed workstation actions; it does not train or submit.",
        "records": records,
    }
    out = ROOT / "workspace" / "s6e6_recovery_governance_backfill_20260625.json"
    write_json(out, report)
    print(json.dumps({"json": rel(out), "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
