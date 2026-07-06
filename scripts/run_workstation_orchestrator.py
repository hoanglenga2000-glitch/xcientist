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
    orchestrator = AgentOrchestrator(ROOT)
    summary = orchestrator.run_local_tabular_closed_loop(
        config_path=ROOT / args.config,
        output_base=ROOT / args.output_base,
        random_state=args.random_state,
    )
    print(json.dumps(summary["run"], ensure_ascii=False, indent=2))
    print(f"Summary written to: {ROOT / 'workspace' / 'workstation_summary.json'}")


if __name__ == "__main__":
    main()

