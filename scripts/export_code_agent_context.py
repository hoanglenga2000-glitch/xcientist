from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.adapters.storage_adapter import LocalStorageAdapter
from research_agent_workstation.server.services.code_agent_context_service import CodeAgentContextService
from research_agent_workstation.server.services.task_service import TaskService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export task context for Codex/Claude Code.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--latest-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = TaskService(ROOT).import_from_config(ROOT / args.config)
    service = CodeAgentContextService(LocalStorageAdapter(ROOT), ROOT)
    context_dir = service.export_context(
        task=task,
        latest_output_dir=Path(args.latest_output_dir) if args.latest_output_dir else None,
        experiment_plan="Use local template baseline first; request external code optimization only through patch import.",
    )
    print(context_dir)


if __name__ == "__main__":
    main()

