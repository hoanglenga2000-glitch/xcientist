from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.adapters.storage_adapter import LocalStorageAdapter
from research_agent_workstation.server.services.code_agent_context_service import CodeAgentContextService


def fail(message: str) -> None:
    raise SystemExit(f"PATCH_LIFECYCLE_FAILED: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Code Agent patch import/review/apply/rollback/compare lifecycle.")
    parser.add_argument("--task-id", default="house_prices")
    return parser.parse_args()


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
    comparison = service.compare_runs_before_after_patch(args.task_id)
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
