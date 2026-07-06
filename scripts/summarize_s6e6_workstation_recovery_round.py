from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND_RUN_IDS = [
    "wr_2026-06-25T06-53-22-322Z_q0mix",
    "wr_2026-06-25T06-53-21-995Z_d688w",
    "wr_2026-06-25T07-09-42-997Z_ov893",
    "wr_2026-06-25T07-00-53-587Z_5hajp",
    "wr_2026-06-25T07-21-37-029Z_m4hiz",
]
TASK_ID = "playground_series_s6e6"
RUN_ROOT = ROOT / "workspace" / "workstation_runs" / TASK_ID
OUT_JSON = ROOT / "workspace" / "s6e6_workstation_recovery_round_20260625.json"
OUT_MD = ROOT / "reports" / "S6E6_WORKSTATION_RECOVERY_ROUND_20260625.md"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def first_existing(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def discover_run_ids() -> list[str]:
    return [run_id for run_id in ROUND_RUN_IDS if (RUN_ROOT / run_id).exists()]


def summarize_run(run_id: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    result_path = first_existing(
        run_dir,
        [
            "artifact_replay_manifest.json",
            "exp024_multi_asset_frontier_blend_result.json",
            "exp024_multi_asset_frontier_blend_failure.json",
            "exp025_single_model_diversity_result.json",
        ],
    )
    result = read_json(result_path) if result_path else {}
    metrics_path = result.get("metrics_path")
    if isinstance(metrics_path, str):
        metrics = read_json(ROOT / metrics_path)
    else:
        metrics = read_json(run_dir / "hpc_gpu_training" / "metrics.json")
    score_gate = read_json(run_dir / "score_improvement_gate.json")
    submission_audit = read_json(run_dir / "submission_audit.json")
    selected = metrics.get("selected", {}) if isinstance(metrics.get("selected"), dict) else {}
    gpu_job = result.get("gpu_job", {}) if isinstance(result.get("gpu_job"), dict) else {}
    return {
        "run_id": run_id,
        "run_dir": rel(run_dir),
        "result_artifact": rel(result_path),
        "status": result.get("status") or score_gate.get("status") or metrics.get("status"),
        "action_type": (
            "artifact_replay"
            if "artifact_replay" in (result_path.name if result_path else "")
            else "exp024_frontier_blend"
            if result_path and "exp024" in result_path.name
            else "exp025_gpu_single_model"
            if result_path and "exp025" in result_path.name
            else "unknown"
        ),
        "official_submission_started": bool(result.get("official_submission_started", False)),
        "submission_audit_status": submission_audit.get("status"),
        "score_gate_status": score_gate.get("status"),
        "score_gate_decision": (score_gate.get("decision") or {}).get("decision")
        if isinstance(score_gate.get("decision"), dict)
        else None,
        "validation_score": result.get("validation_score")
        or metrics.get("best_validation_score")
        or selected.get("balanced_accuracy")
        or metrics.get("oof_balanced_accuracy"),
        "delta_vs_baseline_balanced_accuracy": selected.get("delta_vs_baseline_balanced_accuracy"),
        "decision": metrics.get("decision") or result.get("decision"),
        "decision_reason": metrics.get("decision_reason"),
        "gpu_job_status": gpu_job.get("status"),
        "gpu_metrics_artifact": gpu_job.get("metrics_artifact"),
        "probabilities_artifact": gpu_job.get("probabilities_artifact") or result.get("probabilities_artifact"),
        "candidate_submission": (metrics.get("outputs") or {}).get("candidate_submission")
        if isinstance(metrics.get("outputs"), dict)
        else result.get("submission_artifact"),
    }


def main() -> None:
    rows = [summarize_run(run_id) for run_id in discover_run_ids() if (RUN_ROOT / run_id).exists()]
    official_started = any(row["official_submission_started"] for row in rows)
    promoted = [
        row
        for row in rows
        if row.get("score_gate_status") == "passed"
        or row.get("decision") in {"submit_candidate", "human_gate_only_candidate"}
    ]
    payload = {
        "schema": "academic_research_os.s6e6_workstation_recovery_round.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": TASK_ID,
        "codex_role": "supervisor_only_no_direct_training_no_direct_submit",
        "execution_subject": "workstation /api/workstation-actions",
        "official_submission_started": official_started,
        "official_top30_reached": False,
        "claim_boundary": "This round produced workstation evidence only. No official Kaggle submit or top30 claim is supported.",
        "summary": {
            "runs_observed": len(rows),
            "promoted_or_submit_candidate_runs": len(promoted),
            "evidence_only_or_blocked_runs": len(rows) - len(promoted),
            "gpu_evidence_runs": len([row for row in rows if row["action_type"] == "exp025_gpu_single_model"]),
        },
        "runs": rows,
        "next_actions": [
            "Keep official submit blocked: no new run has both score/risk promotion and current-turn human submission approval.",
            "Use EXP024 selected triple blend as a candidate for stricter full fold-stability review, not as an immediate submission.",
            "Store EXP025 20k CatBoost GPU dry-run as negative/diversity memory; it proves GPU execution but its balanced accuracy is below the protected frontier.",
            "Next workstation step should run a narrower full-data GPU branch only if Search Controller selects it and hpc_execution_approval is recorded.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S6E6 Workstation Recovery Round",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Executor: `{payload['execution_subject']}`",
        f"- Codex role: `{payload['codex_role']}`",
        f"- Official submission started: `{payload['official_submission_started']}`",
        f"- Official top30 reached: `{payload['official_top30_reached']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Runs",
        "",
        "| run | type | status | validation | delta vs baseline | decision | score gate | submit audit | gpu |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_id']}` | `{row['action_type']}` | `{row['status']}` | {row['validation_score']} | {row['delta_vs_baseline_balanced_accuracy']} | `{row['decision']}` | `{row['score_gate_status']}` | `{row['submission_audit_status']}` | `{row['gpu_job_status']}` |"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in payload["next_actions"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": rel(OUT_JSON), "md": rel(OUT_MD), "runs": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
