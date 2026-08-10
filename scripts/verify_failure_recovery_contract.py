from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_os.agent.multi_agent import (
    AgentRoleSpec,
    AgentTask,
    MultiAgentStore,
    MultiAgentSupervisor,
    create_run,
)

REQUIRED_FAILURE_FILES = {
    "error.json",
    "traceback.txt",
    "environment.json",
    "node_status.json",
    "recovery_plan.json",
}


def verify(output_root: Path) -> dict[str, object]:
    run_id = f"recovery_crash_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root.resolve() / run_id
    run = create_run(
        objective="Controlled training crash for the production recovery contract.",
        tasks=[AgentTask("training_crash", "Execute controlled training fault injection", "TrainingAgent", max_retries=0)],
        roles=[AgentRoleSpec("TrainingAgent", capabilities=("training",), output_contract=("failure_evidence",))],
        run_id=run_id,
        max_concurrency=1,
    )

    def crash_training(_task, _handoff, _run):
        raise RuntimeError("intentional_training_crash_for_recovery_verification")

    result = MultiAgentSupervisor(
        run,
        MultiAgentStore(run_dir),
        {"TrainingAgent": crash_training},
    ).run_until_blocked()
    failure_dir = run_dir / "failure" / "training_crash"
    present = {path.name for path in failure_dir.iterdir()} if failure_dir.is_dir() else set()
    actions = []
    action_log = run_dir / "action_log.jsonl"
    if action_log.is_file():
        actions = [
            json.loads(line).get("action")
            for line in action_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    passed = (
        result.status == "needs_continuation"
        and present == REQUIRED_FAILURE_FILES
        and actions[-4:] == ["detect", "analyze", "repair", "retry"]
    )
    report = {
        "schema": "evomind.failure_recovery_verification.v1",
        "status": "passed" if passed else "failed",
        "run_id": run_id,
        "run_status": result.status,
        "failure_dir": str(failure_dir),
        "failure_files": sorted(present),
        "recovery_actions": actions[-4:],
        "recovery_agent": "RecoveryAgent",
        "fault_injection": "intentional_training_crash_for_recovery_verification",
        "mock_used": False,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    report_path = run_dir / "verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "verification_path": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workspace/verification/production_recovery/failure_recovery"),
    )
    args = parser.parse_args()
    report = verify(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
