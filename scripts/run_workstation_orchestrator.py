"""Retired local-training launcher kept as an explicit fail-closed shim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.services import AgentOrchestrator  # noqa: E402
from research_os.hpc_policy import HPCPolicyError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retired local runner. This command reports Blocked and points to the gated HPC manifest queue.",
    )
    parser.add_argument("--config", required=True, help="Task config YAML, e.g. configs/house_prices.yaml")
    parser.add_argument("--output-base", default="experiments")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orchestrator = AgentOrchestrator(ROOT)
    try:
        orchestrator.run_local_tabular_closed_loop(
            config_path=ROOT / args.config,
            output_base=ROOT / args.output_base,
            random_state=args.random_state,
        )
    except HPCPolicyError as exc:
        blocked = {
            "status": "blocked_local_training_disabled",
            "reason": str(exc),
            "training_started": False,
            "hpc_queue_command": [
                sys.executable,
                "scripts/run_workstation_ensemble.py",
                "--config",
                args.config,
                "--output-base",
                args.output_base,
            ],
        }
        print(json.dumps(blocked, ensure_ascii=False, indent=2))
        return 2
    raise RuntimeError("Local training policy returned without blocking")


if __name__ == "__main__":
    raise SystemExit(main())
