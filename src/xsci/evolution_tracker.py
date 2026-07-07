r"""Self-Evolution Tracker — measure and visualize the agent's improvement.

Tracks EvoMind's own skill acquisition over time:
  - Repair success rate (how often auto-repair works)
  - Innovation hit rate (how often novel combinations succeed)
  - Training efficiency (time-to-baseline, average CV improvement per run)
  - Cross-task transfer (how much faster task N is than task 1)
  - Skill milestones (first baseline, first CV improvement, first innovation)

This is the agent's "resume" — a quantifiable record of getting smarter.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvolutionSnapshot:
    """One point-in-time measurement of agent capability."""
    ts: str = ""
    total_runs: int = 0
    total_promotions: int = 0
    best_cv_ever: Optional[float] = None
    repair_attempts: int = 0
    repair_successes: int = 0
    innovations_tried: int = 0
    innovation_successes: int = 0
    tasks_completed: int = 0
    cross_task_transfers: int = 0
    skill_level: str = "novice"   # novice → apprentice → competent → expert → master


_SKILL_THRESHOLDS = {
    "novice":      (0,     "First steps — basic training runs working"),
    "apprentice":  (5,     "Consistent baselines, first repairs succeeding"),
    "competent":   (15,    "Multiple tasks, cross-task transfer active"),
    "expert":      (30,    "Innovations succeeding, deep domain knowledge"),
    "master":      (60,    "Self-improving, novel methods discovered"),
}


class EvolutionTracker:
    """Tracks and reports EvoMind's self-evolution progress."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self._workspace = Path(workspace_root) if workspace_root else Path.cwd()
        self._path = self._workspace / ".xsci" / "evolution_tracker.json"
        self._history: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._history = data.get("history", []) or []
        except (json.JSONDecodeError, OSError):
            self._history = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "history": self._history[-200:],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ── Recording events ──────────────────────────────────────────────

    def record_run(self, *, success: bool, cv_score: Optional[float] = None,
                   promotions: int = 0, task: str = "") -> None:
        snap = self._latest()
        snap["total_runs"] = snap.get("total_runs", 0) + 1
        if promotions:
            snap["total_promotions"] = snap.get("total_promotions", 0) + promotions
        if cv_score is not None:
            prev_best = snap.get("best_cv_ever")
            if prev_best is None or cv_score > prev_best:
                snap["best_cv_ever"] = cv_score
        snap["ts"] = datetime.now().isoformat(timespec="seconds")
        snap["skill_level"] = self._compute_skill(snap)
        self._history.append(dict(snap))

    def record_repair(self, success: bool) -> None:
        snap = self._latest()
        snap["repair_attempts"] = snap.get("repair_attempts", 0) + 1
        if success:
            snap["repair_successes"] = snap.get("repair_successes", 0) + 1
        snap["ts"] = datetime.now().isoformat(timespec="seconds")
        snap["skill_level"] = self._compute_skill(snap)
        self._history.append(dict(snap))
        self._save()

    def record_innovation(self, success: bool, strategy: str = "") -> None:
        snap = self._latest()
        snap["innovations_tried"] = snap.get("innovations_tried", 0) + 1
        if success:
            snap["innovation_successes"] = snap.get("innovation_successes", 0) + 1
        snap["ts"] = datetime.now().isoformat(timespec="seconds")
        snap["skill_level"] = self._compute_skill(snap)
        self._history.append(dict(snap))
        self._save()

    def record_task_completed(self, task: str) -> None:
        snap = self._latest()
        snap["tasks_completed"] = snap.get("tasks_completed", 0) + 1
        snap["ts"] = datetime.now().isoformat(timespec="seconds")
        snap["skill_level"] = self._compute_skill(snap)
        self._history.append(dict(snap))
        self._save()

    def record_cross_task_transfer(self, from_task: str, to_task: str) -> None:
        snap = self._latest()
        snap["cross_task_transfers"] = snap.get("cross_task_transfers", 0) + 1
        snap["ts"] = datetime.now().isoformat(timespec="seconds")
        self._history.append(dict(snap))
        self._save()

    # ── Current state ─────────────────────────────────────────────────

    def _latest(self) -> dict:
        if self._history:
            return dict(self._history[-1])
        return {}

    def current_snapshot(self) -> EvolutionSnapshot:
        snap = self._latest()
        return EvolutionSnapshot(
            ts=snap.get("ts", ""),
            total_runs=snap.get("total_runs", 0),
            total_promotions=snap.get("total_promotions", 0),
            best_cv_ever=snap.get("best_cv_ever"),
            repair_attempts=snap.get("repair_attempts", 0),
            repair_successes=snap.get("repair_successes", 0),
            innovations_tried=snap.get("innovations_tried", 0),
            innovation_successes=snap.get("innovation_successes", 0),
            tasks_completed=snap.get("tasks_completed", 0),
            cross_task_transfers=snap.get("cross_task_transfers", 0),
            skill_level=snap.get("skill_level", "novice"),
        )

    def _compute_skill(self, snap: dict) -> str:
        """Determine skill level from cumulative achievements."""
        score = 0
        score += snap.get("total_promotions", 0) * 2
        score += snap.get("repair_successes", 0) * 3
        score += snap.get("innovation_successes", 0) * 5
        score += snap.get("tasks_completed", 0) * 3
        score += snap.get("cross_task_transfers", 0) * 2

        for level, (threshold, _) in reversed(_SKILL_THRESHOLDS.items()):
            if score >= threshold:
                return level
        return "novice"

    # ── Reporting ─────────────────────────────────────────────────────

    def report(self) -> str:
        """A human-readable evolution report."""
        snap = self.current_snapshot()
        level_desc = _SKILL_THRESHOLDS.get(snap.skill_level, ("", ""))[1]

        lines = ["🧬 EvoMind Self-Evolution Report", "=" * 45]
        lines.append(f"Skill Level: {snap.skill_level.upper()}")
        lines.append(f"  {level_desc}")
        lines.append("")

        lines.append("📊 Performance Metrics:")
        lines.append(f"  Total runs:            {snap.total_runs}")
        lines.append(f"  Promotions (improved):  {snap.total_promotions}")
        lines.append(f"  Best CV ever:           {snap.best_cv_ever or 'N/A'}")
        lines.append(f"  Tasks completed:        {snap.tasks_completed}")
        lines.append("")

        lines.append("🔧 Self-Repair:")
        repair_rate = (snap.repair_successes / max(snap.repair_attempts, 1) * 100)
        lines.append(f"  Repair attempts:  {snap.repair_attempts}")
        lines.append(f"  Repair successes: {snap.repair_successes}")
        lines.append(f"  Success rate:     {repair_rate:.1f}%")
        lines.append("")

        lines.append("💡 Innovation:")
        inno_rate = (snap.innovation_successes / max(snap.innovations_tried, 1) * 100)
        lines.append(f"  Innovations tried:  {snap.innovations_tried}")
        lines.append(f"  Innovation hits:    {snap.innovation_successes}")
        lines.append(f"  Hit rate:           {inno_rate:.1f}%")
        lines.append(f"  Cross-task transfer: {snap.cross_task_transfers}")
        lines.append("")

        # Growth projection
        if snap.skill_level in ("novice", "apprentice"):
            lines.append("📈 Next milestone: Reach 'competent' by completing 2+ tasks "
                         "with successful repairs.")
        elif snap.skill_level == "competent":
            lines.append("📈 Next milestone: Reach 'expert' by generating and "
                         "validating novel innovations.")
        elif snap.skill_level == "expert":
            lines.append("📈 Next milestone: Reach 'master' by consistently "
                         "producing innovations that outperform baselines across tasks.")
        else:
            lines.append("🏆 You've reached MASTER level. EvoMind is now a self-improving scientist.")

        return "\n".join(lines)

    def growth_curve(self) -> list[dict]:
        """Return the full history for charting."""
        return list(self._history)
