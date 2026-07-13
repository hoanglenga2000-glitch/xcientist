from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def post_json(base_url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/workstation-actions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def require(condition: bool, message: str, evidence: Any = None) -> None:
    if not condition:
        print(json.dumps({"status": "failed", "message": message, "evidence": evidence}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify S6E6 score-aware strategy and submission gates through workstation APIs.")
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--run-weak-gate", action="store_true", help="Create a short workstation run that should block before HPC execution.")
    args = parser.parse_args()

    recommendations = post_json(args.url, {
        "action": "recommend_strategies",
        "task_id": "playground_series_s6e6",
        "metadata": {"top_k": 3},
    })
    recs = recommendations.get("recommendations") or []
    require(bool(recs), "No strategy recommendations returned.", recommendations)
    top = recs[0]
    require(top["strategy"]["gpu_template"] == "playground_s6e6_boosting_ensemble", "Top S6E6 strategy is not the true boosting template.", top)
    require(top["score_gate"]["official_submit_policy"] == "candidate", "Top strategy is not marked as score-improvement candidate.", top)

    default_gate = post_json(args.url, {
        "action": "evaluate_s6e6_strategy_execution_gate",
        "task_id": "playground_series_s6e6",
        "metadata": {},
    })
    default_gate_payload = default_gate.get("gate") or {}
    require(default_gate_payload.get("selected_template") == "playground_s6e6_boosting_ensemble", "Default execution gate does not select the boosting ensemble.", default_gate)
    require(default_gate_payload.get("allowed_to_execute") is True, "Default boosting execution gate is not allowed.", default_gate)

    weak_gate = post_json(args.url, {
        "action": "evaluate_s6e6_strategy_execution_gate",
        "task_id": "playground_series_s6e6",
        "metadata": {
            "gpu_template": "playground_s6e6_pytorch_mlp",
        },
    })
    weak_gate_payload = weak_gate.get("gate") or {}
    require(weak_gate_payload.get("selected_template") == "playground_s6e6_pytorch_mlp", "Weak execution probe did not evaluate the requested MLP template.", weak_gate)
    require(weak_gate_payload.get("allowed_to_execute") is False, "Weak MLP execution probe was not blocked.", weak_gate)

    weak_submit = post_json(args.url, {
        "action": "retry_s6e6_kaggle_submission",
        "task_id": "playground_series_s6e6",
        "metadata": {
            "run_id": "wr_2026-06-15T10-12-59-426Z_eawg4",
            "submit_message": "verify weak candidate blocked",
            "approval_reason": "strategy gate verifier",
        },
    })
    require(weak_submit.get("status") == "blocked_score_gate", "Known weak S6E6 run was not blocked by score gate.", weak_submit)

    mlp_score_gate = post_json(args.url, {
        "action": "evaluate_s6e6_score_improvement_gate",
        "task_id": "playground_series_s6e6",
        "metadata": {
            "run_id": "wr_2026-06-15T07-58-58-337Z_j9fxi",
        },
    })
    require(mlp_score_gate.get("status") == "blocked", "Known MLP low-score candidate passed the score improvement gate.", mlp_score_gate)

    sklearn_score_gate = post_json(args.url, {
        "action": "evaluate_s6e6_score_improvement_gate",
        "task_id": "playground_series_s6e6",
        "metadata": {
            "run_id": "wr_2026-06-15T10-12-59-426Z_eawg4",
        },
    })
    require(sklearn_score_gate.get("status") == "blocked", "Sklearn-only fallback candidate passed the score improvement gate.", sklearn_score_gate)
    sklearn_gate_payload = sklearn_score_gate.get("gate") or {}
    sklearn_blocked_reasons = sklearn_gate_payload.get("blocked_reasons") or []
    require(
        any("log_loss" in str(reason) for reason in sklearn_blocked_reasons),
        "Sklearn fallback public-score failure is not blocked by the risk-adjusted frontier/log_loss guard.",
        sklearn_score_gate,
    )

    recovery_plan = post_json(args.url, {
        "action": "generate_s6e6_score_recovery_plan",
        "task_id": "playground_series_s6e6",
        "metadata": {},
    })
    frontier = recovery_plan.get("frontier") or {}
    failed_path = frontier.get("known_failed_workstation_path") or {}
    require(recovery_plan.get("ok") is True and recovery_plan.get("artifact_path"), "S6E6 score recovery plan was not generated.", recovery_plan)
    require(failed_path.get("public_score") == 0.95474, "S6E6 recovery plan does not record the known lower public score.", recovery_plan)
    require(
        (recovery_plan.get("recommended_next_strategy") or {}).get("gpu_template") == "playground_s6e6_boosting_ensemble",
        "S6E6 recovery plan did not keep the boosting ensemble as the next workstation strategy.",
        recovery_plan,
    )

    dependency_gate = post_json(args.url, {
        "action": "verify_s6e6_boosting_environment",
        "task_id": "playground_series_s6e6",
        "metadata": {},
    })
    dependency_status = dependency_gate.get("status")
    dependency_artifact = dependency_gate.get("artifact_path")
    require(
        dependency_status in {"passed", "blocked_dependency", "blocked_resource_gateway"},
        "S6E6 boosting dependency gate did not return a controlled status.",
        dependency_gate,
    )
    require(isinstance(dependency_artifact, str) and (ROOT / dependency_artifact).is_file(), "S6E6 boosting dependency gate artifact is missing.", dependency_gate)
    dependency_payload = json.loads((ROOT / dependency_artifact).read_text(encoding="utf-8-sig"))
    require(dependency_payload.get("training_started") is False, "S6E6 boosting dependency gate must not start training.", dependency_payload)
    if dependency_status == "blocked_dependency":
        require(bool(dependency_payload.get("missing_packages")), "Blocked dependency gate did not record missing packages.", dependency_payload)
    if dependency_status == "blocked_resource_gateway":
        require(bool(dependency_payload.get("blocker")), "Blocked resource gateway did not record a blocker.", dependency_payload)

    weak_execution = None
    if args.run_weak_gate:
        weak_execution = post_json(args.url, {
            "action": "run_s6e6_workstation_closed_loop",
            "task_id": "playground_series_s6e6",
            "metadata": {
                "gpu_template": "playground_s6e6_pytorch_mlp",
                "allow_official_submit_after_gate": False,
            },
        }, timeout=240)
        require(weak_execution.get("status") == "blocked_strategy_gate", "Weak MLP execution was not blocked before HPC training.", weak_execution)

    print(json.dumps({
        "status": "passed",
        "dashboard_url": args.url,
        "top_strategy": {
            "strategy_id": top["strategy"]["strategy_id"],
            "gpu_template": top["strategy"]["gpu_template"],
            "official_submit_policy": top["score_gate"]["official_submit_policy"],
            "expected_public_score": top["score_gate"]["expected_public_score"],
            "historical_best_public_score": top["score_gate"]["historical_best_public_score"],
        },
        "default_execution_gate": {
            "selected_template": default_gate_payload.get("selected_template"),
            "allowed_to_execute": default_gate_payload.get("allowed_to_execute"),
        },
        "weak_execution_probe": {
            "selected_template": weak_gate_payload.get("selected_template"),
            "allowed_to_execute": weak_gate_payload.get("allowed_to_execute"),
            "blocked_reasons": weak_gate_payload.get("blocked_reasons"),
        },
        "weak_submit_status": weak_submit.get("status"),
        "mlp_score_gate_status": mlp_score_gate.get("status"),
        "sklearn_score_gate_status": sklearn_score_gate.get("status"),
        "sklearn_score_gate_blocked_reasons": sklearn_blocked_reasons,
        "recovery_plan_artifact": recovery_plan.get("artifact_path"),
        "boosting_dependency_gate_status": dependency_status,
        "boosting_dependency_gate_artifact": dependency_artifact,
        "weak_execution_status": weak_execution.get("status") if weak_execution else "not_run",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
