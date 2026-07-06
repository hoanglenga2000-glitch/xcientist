from __future__ import annotations

from pathlib import Path
from typing import Any

from research_agent_workstation.tabular_pipeline import load_yaml, run


def run_tabular_baseline(config_path: Path, output_base: Path, random_state: int = 42) -> dict[str, Any]:
    return run(load_yaml(config_path), output_base, random_state)

