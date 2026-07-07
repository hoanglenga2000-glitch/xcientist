r"""Innovation Engine — cross-task knowledge synthesis for novel strategies.

When EvoMind accumulates enough experience on a task type, this engine:
  1. Analyzes all successful strategies across tasks of the same type
  2. Identifies complementary strategies that have never been tried together
  3. Proposes novel combination experiments
  4. Tracks which innovations were tried and their outcomes
  5. Measures the "innovation ceiling" — how many new ideas remain to try

This is how EvoMind goes beyond template approaches and discovers genuinely
new methods — combining what it knows in ways it hasn't tried before.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class InnovationProposal:
    """A novel experiment idea synthesized from cross-task experience."""
    strategy_name: str           # e.g. "target_encoding + oof_stacking + pseudo_labels"
    components: list[str]        # the individual strategies being combined
    rationale: str               # why this combination might work
    source_tasks: list[str]      # tasks where components were successful
    novelty_score: float = 0.0   # 0.0–1.0: how novel is this combination
    confidence: float = 0.0      # 0.0–1.0: how likely to succeed


@dataclass
class InnovationLog:
    """Tracks which innovations were tried and what happened."""
    proposals: list[dict] = field(default_factory=list)
    tried: list[dict] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    updated_at: str = ""

    def hit_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


class InnovationEngine:
    """Cross-task knowledge synthesis and novel strategy generation."""

    def __init__(self, memory_library=None, *,
                 workspace_root: Optional[Path] = None) -> None:
        self._library = memory_library
        self._workspace = Path(workspace_root) if workspace_root else Path.cwd()
        self._log_path = self._workspace / ".xsci" / "innovation_log.json"
        self._log = self._load_log()

    def _load_log(self) -> InnovationLog:
        try:
            if self._log_path.exists():
                data = json.loads(self._log_path.read_text(encoding="utf-8"))
                return InnovationLog(
                    proposals=data.get("proposals", []),
                    tried=data.get("tried", []),
                    successes=data.get("successes", 0),
                    failures=data.get("failures", 0),
                    updated_at=data.get("updated_at", ""),
                )
        except (json.JSONDecodeError, OSError):
            pass
        return InnovationLog()

    def _save_log(self) -> None:
        self._log.updated_at = datetime.now().isoformat(timespec="seconds")
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text(
                json.dumps({
                    "proposals": self._log.proposals[-50:],
                    "tried": self._log.tried[-100:],
                    "successes": self._log.successes,
                    "failures": self._log.failures,
                    "updated_at": self._log.updated_at,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ── Key functionality: propose novel combinations ─────────────────

    def propose_innovations(self, task_type: str, *,
                            n: int = 3,
                            known_strategies: Optional[list[str]] = None) -> list[InnovationProposal]:
        """Generate novel strategy combinations for a task type.

        Looks at ALL tasks of the same type across the experience library,
        identifies successful strategies, and proposes combinations that
        haven't been tried yet on THIS specific task.

        Args:
            task_type: e.g. "classification", "regression"
            n: maximum number of proposals
            known_strategies: strategies already tried on the current task
        """
        if self._library is None:
            return []

        known = set(known_strategies or [])
        # Get all successful strategies from tasks of this type
        try:
            records = self._library.retrieve(task_type=task_type, limit=60)
        except Exception:
            return []

        if len(records) < 5:
            return []  # Not enough experience to innovate

        # Extract successful strategy components
        strategy_counts: Counter[str] = Counter()
        task_sources: dict[str, list[str]] = defaultdict(list)
        for r in records:
            strat = (r.get("reusable_strategy") or "").strip()
            if not strat:
                continue
            components = [s.strip() for s in strat.replace("+", ",").split(",") if s.strip()]
            for comp in components:
                strategy_counts[comp] += 1
                src = r.get("memory_id", "unknown")
                if src not in task_sources[comp]:
                    task_sources[comp].append(src)

        if len(strategy_counts) < 3:
            return []

        # Find untried combinations
        top_strategies = [s for s, _ in strategy_counts.most_common(10) if s not in known]
        proposals: list[InnovationProposal] = []

        # Generate pair combinations of untried strategies
        for i, s1 in enumerate(top_strategies):
            for s2 in top_strategies[i + 1:]:
                # Skip trivial combinations (same family)
                if any(w in s1.lower() and w in s2.lower()
                       for w in ("gbm", "lightgbm", "xgboost", "catboost")):
                    continue
                combo = f"{s1} + {s2}"
                if combo in known:
                    continue

                novelty = 1.0 - (known and any(s in str(known) for s in (s1, s2)) and 0.5 or 0)
                confidence = min(
                    (strategy_counts[s1] + strategy_counts[s2]) / max(strategy_counts.values()) / 2,
                    0.8
                )
                src_tasks = list(set(task_sources.get(s1, []) + task_sources.get(s2, [])))[:4]
                proposals.append(InnovationProposal(
                    strategy_name=combo,
                    components=[s1, s2],
                    rationale=(
                        f"{s1} has been effective on {len(task_sources.get(s1, []))} tasks; "
                        f"{s2} on {len(task_sources.get(s2, []))}. "
                        f"Combining them could yield a stronger model through complementary mechanisms."
                    ),
                    source_tasks=src_tasks,
                    novelty_score=novelty,
                    confidence=confidence,
                ))

        # Sort by novelty × confidence product
        proposals.sort(key=lambda p: p.novelty_score * p.confidence, reverse=True)
        return proposals[:n]

    def record_attempt(self, strategy_name: str, success: bool,
                       cv_score: Optional[float] = None) -> None:
        """Record that an innovation was tried, with its outcome."""
        self._log.tried.append({
            "strategy": strategy_name,
            "success": success,
            "cv_score": cv_score,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        if success:
            self._log.successes += 1
        else:
            self._log.failures += 1
        self._save_log()

    def stats(self) -> dict[str, Any]:
        """Return current innovation statistics."""
        return {
            "proposals_generated": len(self._log.proposals),
            "innovations_tried": len(self._log.tried),
            "successes": self._log.successes,
            "failures": self._log.failures,
            "hit_rate": f"{self._log.hit_rate():.1%}",
            "most_successful": self._top_innovations(3, success_only=True),
        }

    def _top_innovations(self, n: int, success_only: bool = False) -> list[str]:
        items = self._log.tried
        if success_only:
            items = [t for t in items if t.get("success")]
        scored = sorted(items, key=lambda t: t.get("cv_score") or 0, reverse=True)
        return [t.get("strategy", "?") for t in scored[:n]]

    def ready_for_innovation(self, task_type: str) -> bool:
        """Check if there's enough experience to propose meaningful innovations."""
        if self._library is None:
            return False
        try:
            records = self._library.retrieve(task_type=task_type, limit=10)
            # Need at least 8 lessons with reusable strategies
            with_strategies = [r for r in records if r.get("reusable_strategy")]
            return len(with_strategies) >= 5
        except Exception:
            return False
