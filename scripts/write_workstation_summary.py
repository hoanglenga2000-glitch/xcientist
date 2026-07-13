from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.services import AgentOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a frontend-readable summary from latest existing experiments.")
    parser.add_argument("--tasks", nargs="+", default=["house_prices", "titanic"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = AgentOrchestrator(ROOT).summarize_latest_existing(args.tasks)
    print(json.dumps({"runs": len(payload["runs"]), "summary": str(ROOT / "workspace" / "workstation_summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

