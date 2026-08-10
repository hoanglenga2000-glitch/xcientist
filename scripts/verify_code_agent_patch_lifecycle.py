from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.adapters.storage_adapter import LocalStorageAdapter  # noqa: E402
from research_agent_workstation.server.services.code_agent_context_service import CodeAgentContextService  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"PATCH_LIFECYCLE_FAILED: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Code Agent patch import/review/apply/rollback/compare lifecycle.")
    parser.add_argument("--task-id", default="house_prices")
    return parser.parse_args()


def configured_experiment_root() -> Path:
    evidence_root = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    default = Path(evidence_root) / "experiments" if evidence_root else ROOT / "experiments"
    configured = Path(os.environ.get("RESEARCH_EXPERIMENT_ROOT", default))
    return (configured if configured.is_absolute() else ROOT / configured).resolve()


def latest_comparable_runs(task_id: str) -> tuple[Path | None, Path]:
    task_root = configured_experiment_root() / task_id
    try:
        runs = sorted(
            path
            for path in task_root.iterdir()
            if path.is_dir() and (path / "model_results.json").is_file()
        )
    except OSError as error:
        fail(f"experiment root unavailable for task {task_id!r}")
        raise AssertionError("unreachable") from error
    if not runs:
        fail(f"no comparable experiment runs found for task {task_id!r}")
    return (runs[-2] if len(runs) >= 2 else None, runs[-1])


def main() -> None:
    args = parse_args()
    service = CodeAgentContextService(LocalStorageAdapter(ROOT), ROOT)
    patch_path = service.import_patch(
        args.task_id,
        "diff --git a/workspace_note.md b/workspace_note.md\n"
        "--- a/workspace_note.md\n"
        "+++ b/workspace_note.md\n"
        "@@\n"
        "+Patch lifecycle acceptance record.\n",
        "acceptance",
    )
    patch_id = patch_path.stem
    review = service.review_patch(args.task_id, patch_id)
    if review.get("review_status") not in {"passed", "warning"}:
        fail(f"unexpected review status: {review}")
    applied = service.apply_patch(args.task_id, patch_id)
    if applied.get("apply_status") != "recorded":
        fail(f"patch was not recorded as applied: {applied}")
    rollback_path = Path(applied.get("rollback_path", ""))
    if not rollback_path.exists():
        fail(f"rollback manifest missing: {rollback_path}")
    rolled_back = service.rollback_patch(args.task_id, patch_id)
    if rolled_back.get("apply_status") != "rolled_back":
        fail(f"patch was not rolled back: {rolled_back}")
    before_run, after_run = latest_comparable_runs(args.task_id)
    comparison = service.compare_runs_before_after_patch(args.task_id, before_run, after_run)
    if not comparison.get("after_run"):
        fail(f"comparison did not find an after run: {comparison}")
    print(
        json.dumps(
            {
                "status": "passed",
                "task_id": args.task_id,
                "patch_id": patch_id,
                "review_status": review.get("review_status"),
                "apply_status": rolled_back.get("apply_status"),
                "comparison_after_run": comparison.get("after_run"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
