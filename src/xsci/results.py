"""Read-only access to experiment artifacts under experiments/evolution/.

Pure and offline: parses each run's summary.json into a stable dataclass. This is
the shared foundation for `xsci report` and the phase-5 dashboard/watch. It never
writes, never runs the engine, and tolerates partial/corrupt runs (an in-progress
run may have no summary yet).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import find_project_dir

# run dirs are "<task>_<compute>_<YYYYMMDD>_<HHMMSS>"; recency is the timestamp,
# not the whole name (else "_local_" would sort after "_gpu_" regardless of time)
_TS_RE = re.compile(r"(\d{8}_\d{6})$")


def _recency_key(r: "ExperimentResult") -> tuple[str, str]:
    m = _TS_RE.search(r.run_id)
    return (m.group(1) if m else "", r.run_id)


@dataclass
class Iteration:
    exp_id: str
    mode: str
    success: bool
    cv_score: Optional[float]
    promoted: bool
    provider: str = ""
    model: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Iteration":
        return cls(
            exp_id=str(d.get("exp_id", "")), mode=str(d.get("mode", "")),
            success=bool(d.get("success", False)), cv_score=d.get("cv_score"),
            promoted=bool(d.get("promoted", False)),
            provider=str(d.get("provider", "")), model=str(d.get("model", "")),
            note=str(d.get("note", "")),
        )


@dataclass
class ExperimentResult:
    run_dir: Path
    task: str
    best_exp_id: str
    best_cv_score: Optional[float]
    metric: str
    metric_direction: str
    n_iterations: int
    n_promotions: int
    iterations: list[Iteration] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def success_rate(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(1 for it in self.iterations if it.success) / len(self.iterations)

    @classmethod
    def from_summary(cls, run_dir: Path, data: dict[str, Any]) -> "ExperimentResult":
        return cls(
            run_dir=run_dir, task=str(data.get("task", run_dir.name)),
            best_exp_id=str(data.get("best_exp_id", "")),
            best_cv_score=data.get("best_cv_score"),
            metric=str(data.get("metric", "")),
            metric_direction=str(data.get("metric_direction", "maximize")),
            n_iterations=int(data.get("n_iterations", 0)),
            n_promotions=int(data.get("n_promotions", 0)),
            iterations=[Iteration.from_dict(x) for x in data.get("iterations", [])],
        )


def evolution_dir(project_root: Optional[Path] = None) -> Path:
    root = project_root or find_project_dir() or Path.cwd()
    return root / "experiments" / "evolution"


def load_result(run_dir: Path) -> Optional[ExperimentResult]:
    """Load one run; returns None if it has no (or unreadable) summary.json."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return ExperimentResult.from_summary(run_dir, data)


def list_results(project_root: Optional[Path] = None) -> list[ExperimentResult]:
    """All completed runs, newest first (run dirs are timestamp-suffixed)."""
    base = evolution_dir(project_root)
    if not base.is_dir():
        return []
    results = []
    for child in base.iterdir():
        if child.is_dir():
            r = load_result(child)
            if r is not None:
                results.append(r)
    return sorted(results, key=_recency_key, reverse=True)


def find_result(run_id_or_task: str, project_root: Optional[Path] = None) -> Optional[ExperimentResult]:
    """Exact run-dir match first, else the newest run for a task name."""
    base = evolution_dir(project_root)
    exact = base / run_id_or_task
    if exact.is_dir():
        return load_result(exact)
    for r in list_results(project_root):
        if r.task == run_id_or_task:
            return r
    return None
