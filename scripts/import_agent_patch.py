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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import an external Code Agent patch into the workspace patch queue.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--patch-file", required=True)
    parser.add_argument("--source-agent", default="external")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_text = Path(args.patch_file).read_text(encoding="utf-8")
    path = CodeAgentContextService(LocalStorageAdapter(ROOT), ROOT).import_patch(args.task_id, patch_text, args.source_agent)
    print(path)


if __name__ == "__main__":
    main()
