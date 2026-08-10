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
    parser = argparse.ArgumentParser(description="Run the Research Agent Workstation local MVP orchestrator.")
    parser.add_argument("--config", required=True, help="Task config YAML, e.g. configs/house_prices.yaml")
    parser.add_argument("--output-base", default="experiments")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({
        "status": "blocked_local_training_disabled",
        "training_started": False,
        "hpc_queue_command": [sys.executable, "scripts/run_workstation_ensemble.py", "--config", args.config],
    }, ensure_ascii=False))
    raise SystemExit(2)


if __name__ == "__main__":
    main()

