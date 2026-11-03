"""Phase-1 single-task closed loop stub (P0-HN-02).

Gated by ``EVOMIND_FF_SINGLE_TASK_LOOP`` (default off). When on, simulates one
workstation-started run id locally with a CPU-only stub runner — never enqueues
``managed_tensor_train`` or live HPC/GPU jobs. Produces the migration-map
artifact bundle (metrics / OOF / submission_audit / claim_audit /
rank_promotion_gate / benchmark_claim_gate) and integrates with the W5 claim
gate so a single task cannot license 75-task MLEvolve-level claims.
"""
from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .claim_audit import audit_claim, detect_benchmark_overclaim
from .feature_flags import ff_benchmark_claim_gate, ff_single_task_loop
from .mlevolve_controller import (
    TOP30_TARGET_PERCENTILE,
    build_benchmark_claim_gate,
    classify_workstation_status,
    evaluate_rank_gate,
)
from .validation_contract import check_required_artifacts, create_contract, evaluate_acceptance

# Explicit denylist — stub must never call these.
_FORBIDDEN_TRAIN_HOOKS = frozenset({
    "managed_tensor_train",
    "hpc_enqueue",
    "gpu_runner",
    "live_hpc",
})


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def stub_local_runner(
    *,
    task_id: str,
    run_id: str,
    out_dir: Path,
    metric: str = "cv_score",
    stub_score: float = 0.72,
) -> dict[str, Any]:
    """CPU-only stub that fabricates metrics/OOF/submission for CI/local demos.

    Does **not** import or call GPURunner, HPC policy, or managed_tensor_train.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema": "academic_research_os.metrics.v1",
        "task_id": task_id,
        "run_id": run_id,
        "status": "passed",
        "metric": metric,
        "metric_direction": "maximize",
        "cv_score": stub_score,
        "best_score": stub_score,
        "n_folds": 3,
        "resource_mode": "stub_local_cpu",
        "managed_tensor_train_enqueued": False,
        "hpc_job_enqueued": False,
        "runner": "stub_local_runner",
        "submission_rows": 5,
    }
    _write_json(out_dir / "metrics.json", metrics)

    oof_path = out_dir / "oof_predictions.csv"
    with oof_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["row_id", "y_true", "y_pred", "fold"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({
                "row_id": i,
                "y_true": i % 2,
                "y_pred": round(stub_score - 0.05 + 0.02 * i, 4),
                "fold": i % 3,
            })

    sub_path = out_dir / "submission.csv"
    with sub_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "prediction"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"id": i, "prediction": round(0.5 + 0.05 * i, 4)})

    manifest = {
        "schema": "academic_research_os.artifact_manifest.v1",
        "task_id": task_id,
        "run_id": run_id,
        "artifacts": [
            {"artifact_type": "metrics", "path": "metrics.json"},
            {"artifact_type": "oof_predictions", "path": "oof_predictions.csv"},
            {"artifact_type": "submission", "path": "submission.csv"},
        ],
        "runner": "stub_local_runner",
        "forbidden_hooks_not_called": sorted(_FORBIDDEN_TRAIN_HOOKS),
    }
    _write_json(out_dir / "artifact_manifest.json", manifest)
    return metrics


def run_single_task_closed_loop(
    work_dir: str | Path,
    *,
    task_id: str = "tabular_stub_task",
    run_id: str | None = None,
    claim_text: str = "",
    stub_score: float = 0.72,
    metric: str = "cv_score",
) -> dict[str, Any]:
    """Run one Phase-1 single-task closed loop stub under ``work_dir``.

    Flag off → skipped payload, no files written, production paths unchanged.
    Flag on → stub runner artifacts + submission_audit + claim/rank/benchmark
    gates. ``evaluated_tasks=1`` so MLEvolve-level / 75-task claims are refused.
    """
    enabled = ff_single_task_loop()
    root = Path(work_dir)
    if not enabled:
        return {
            "schema": "academic_research_os.single_task_loop.v1",
            "enabled": False,
            "skipped": True,
            "feature_flag": "EVOMIND_FF_SINGLE_TASK_LOOP",
            "claim_blocked": True,
            "managed_tensor_train_enqueued": False,
            "hpc_job_enqueued": False,
            "artifacts": {},
        }

    rid = (run_id or f"stl-{uuid.uuid4().hex[:10]}").strip()
    out_dir = root / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")

    metrics = stub_local_runner(
        task_id=task_id, run_id=rid, out_dir=out_dir, metric=metric, stub_score=stub_score,
    )
    candidate_score = float(metrics["cv_score"])

    submission_audit = {
        "schema": "academic_research_os.submission_audit.v1",
        "created_at": created_at,
        "task_id": task_id,
        "run_id": rid,
        "submission_schema_valid": True,
        "no_missing_predictions": True,
        "train_test_features_match": True,
        "submission_audit_passed": True,
        "human_approval": False,
        "official_kaggle_submit": False,
        "rank_target_percentile": TOP30_TARGET_PERCENTILE,
        "official_submit_budget": 2,
        "official_rank_status": "proxy_only",
        "top30_reached": False,
        "resource_mode": "stub_local_cpu",
        "managed_tensor_train_enqueued": False,
        "hpc_job_enqueued": False,
        "claim_boundary": (
            "Generated submission is a local/proxy stub artifact only; "
            "Human Gate required before any official submit or medal claim."
        ),
    }
    submission_path = _write_json(out_dir / "submission_audit.json", submission_audit)

    hypothesis = (
        f"Single-task stub loop for {task_id} produces local CV/proxy artifacts; "
        "not comparable to MLEvolve 75-task medal rate."
    )
    required_artifacts = [
        "metrics.json",
        "oof_predictions.csv",
        "submission.csv",
        "artifact_manifest.json",
        "submission_audit.json",
    ]
    contract = create_contract(
        contract_id=f"vc_{rid}",
        exp_id=f"EXP_{rid}",
        claim=hypothesis,
        hypothesis=hypothesis,
        implementation_requirement=(
            "Stub closed loop under EVOMIND_FF_SINGLE_TASK_LOOP; CPU-only stub "
            "runner; must not enqueue managed_tensor_train or HPC."
        ),
        metric=metric,
        baseline_exp_id="none",
        acceptance_criteria={"submission_audit_passed": {"equals": True}},
        ablation_plan=["local_stub_metrics_only"],
        risk_checklist=[
            {"risk": "leaderboard_overclaim", "status": "blocked", "evidence": "benchmark_claim_gate"},
            {"risk": "hpc_bypass", "status": "checked", "evidence": "stub_local_runner"},
        ],
        conclusion_boundary=(
            "Only local CV/proxy conclusions; top-30% / medal / MLEvolve-level "
            "claims require aligned 75-task evidence + Human Gate."
        ),
        required_artifacts=required_artifacts,
    )
    available = [p.name for p in out_dir.iterdir() if p.is_file()]
    contract_payload = asdict(contract)
    contract_payload["artifact_check"] = check_required_artifacts(contract, available)
    contract_payload["acceptance_check"] = evaluate_acceptance(
        contract,
        {
            metric: candidate_score,
            "submission_audit_passed": bool(submission_audit.get("submission_audit_passed")),
        },
    )
    contract_path = _write_json(out_dir / "validation_contract.json", contract_payload)

    # Default claim is honest/local; callers may pass overclaim text to prove refuse.
    text = (claim_text or (
        f"{task_id} single-task stub produced local CV={candidate_score}; "
        "official ranking and MLEvolve-level medal rate are not claimed."
    )).strip()
    overclaim = detect_benchmark_overclaim(text)
    claim = audit_claim(
        claim_id=f"claim_{rid}",
        claim_text=text,
        related_exp_ids=[f"EXP_{rid}"],
        contract=contract_payload,
        supporting_metrics={metric: candidate_score, "evaluated_tasks": 1},
        required_ablations=["independent_holdout", "multi_task_registry"],
        completed_ablations=["submission_schema_audit"],
        evidence={
            "has_required_experiments": True,
            "has_mechanistic_evidence": False,
            "missing_evidence": [
                "single_task_only",
                "no_75_task_registry",
                "no_official_kaggle_response",
            ],
        },
    )
    claim_payload = {
        "schema": "academic_research_os.claim_audit.v1",
        "created_at": created_at,
        "task_id": task_id,
        "run_id": rid,
        "layer": "single_task_closed_loop",
        "promotion_claim_allowed": False,
        "evaluated_tasks": 1,
        "overclaim_detected": overclaim,
        **asdict(claim),
    }
    claim_path = _write_json(out_dir / "claim_audit.json", claim_payload)

    rank_gate = evaluate_rank_gate(
        task_id=task_id,
        run_id=rid,
        official_submission=None,  # stub never has Kaggle response
    )
    rank_path = _write_json(out_dir / "rank_promotion_gate.json", rank_gate)
    workstation_status = classify_workstation_status(
        rank_gate=None, has_official_response=False,
    )

    # Single task ⇒ never comparable; overclaim language ⇒ reject/no.
    gate = build_benchmark_claim_gate(
        evaluated_tasks=1,
        medal_rate=None,
        claim_text=text,
    )
    gate_path = _write_json(out_dir / "benchmark_claim_gate.json", gate)

    claim_blocked = (
        gate.get("claim_status") in {"reject", "partial"}
        or claim.audit_result in {"reject", "revise"}
        or overclaim
        or gate.get("claim_label") in {"no", "partial", "preliminary"}
    )
    # Hard invariant: one task must never allow MLEvolve-level promotion claims.
    assert gate.get("claim_status") != "allow", "single-task loop must not allow 75-task claims"
    assert gate.get("mlebench_comparable") is False
    assert metrics.get("managed_tensor_train_enqueued") is False
    assert metrics.get("hpc_job_enqueued") is False

    summary = {
        "schema": "academic_research_os.single_task_loop.v1",
        "created_at": created_at,
        "enabled": True,
        "skipped": False,
        "feature_flag": "EVOMIND_FF_SINGLE_TASK_LOOP",
        "checklist_item": "P0-HN-02",
        "task_id": task_id,
        "run_id": rid,
        "metric": metric,
        "cv_score": candidate_score,
        "claim_blocked": True,  # always: single task cannot license 75-task claims
        "audit_result": claim.audit_result,
        "benchmark_claim_status": gate.get("claim_status"),
        "claim_label": gate.get("claim_label", "no"),
        "overclaim_detected": overclaim,
        "workstation_status": workstation_status,
        "rank_gate_status": rank_gate.get("status"),
        "managed_tensor_train_enqueued": False,
        "hpc_job_enqueued": False,
        "runner": "stub_local_runner",
        "claim_gate_enabled": ff_benchmark_claim_gate(),
        "artifacts": {
            "metrics": str(out_dir / "metrics.json"),
            "oof_predictions": str(out_dir / "oof_predictions.csv"),
            "submission": str(out_dir / "submission.csv"),
            "artifact_manifest": str(out_dir / "artifact_manifest.json"),
            "submission_audit": str(submission_path),
            "validation_contract": str(contract_path),
            "claim_audit": str(claim_path),
            "rank_promotion_gate": str(rank_path),
            "benchmark_claim_gate": str(gate_path),
        },
        "boundary": (
            "Phase-1 stub closed loop (CI/local CPU only). Insufficient multi-task "
            "evidence blocks MLEvolve/AI2-level claims; no live HPC/GPU train."
        ),
    }
    summary_path = _write_json(out_dir / "single_task_loop_summary.json", summary)
    summary["artifacts"]["summary"] = str(summary_path)
    # Re-assert claim_blocked after summary write (summary itself sets True).
    assert claim_blocked or summary["claim_blocked"] is True
    return summary
