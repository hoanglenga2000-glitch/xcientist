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
import math
import uuid
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
        self._recount_outcomes()

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

    @staticmethod
    def _positive_delta(value: Any) -> bool:
        try:
            delta = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(delta) and delta > 0

    @classmethod
    def _is_validated_success(cls, record: dict[str, Any]) -> bool:
        """Return whether a record proves a successful innovation attempt.

        Planning artifacts, gate outcomes, and historical records that lack an
        explicit run/promotion verdict remain useful context, but they are not
        evidence that a strategy was tried successfully.
        """
        profile = record.get("dataset_profile")
        profile = profile if isinstance(profile, dict) else {}
        evidence_level = str(
            record.get("evidence_level") or profile.get("evidence_level") or ""
        ).strip().lower()
        run_success = record.get("run_success", profile.get("run_success"))
        promoted = record.get("promoted", profile.get("promoted"))
        no_training_started = record.get(
            "no_training_started", profile.get("no_training_started")
        )
        return (
            evidence_level == "validated"
            and run_success is True
            and promoted is True
            and no_training_started is not True
            and cls._positive_delta(record.get("metric_delta"))
        )

    @classmethod
    def _is_executed_attempt(cls, record: dict[str, Any]) -> bool:
        """Return whether a log item represents a real executed experiment."""
        profile = record.get("dataset_profile")
        profile = profile if isinstance(profile, dict) else {}
        run_success = record.get("run_success", profile.get("run_success"))
        promoted = record.get("promoted", profile.get("promoted"))
        no_training_started = record.get(
            "no_training_started", profile.get("no_training_started")
        )
        return no_training_started is not True and (
            isinstance(run_success, bool) or isinstance(promoted, bool)
        )

    @classmethod
    def _validated_strategy_records(
        cls, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [record for record in records if cls._is_validated_success(record)]

    def _executed_attempts(self) -> list[dict[str, Any]]:
        return [record for record in self._log.tried if self._is_executed_attempt(record)]

    def _recount_outcomes(self) -> None:
        executed = self._executed_attempts()
        self._log.successes = sum(1 for item in executed if self._is_validated_success(item))
        self._log.failures = sum(1 for item in executed if not self._is_validated_success(item))

    def _successful_strategy_evidence(
        self, task_type: str, *, library_limit: int
    ) -> list[dict[str, Any]]:
        """Return proof-backed strategies from memory and recorded attempts."""
        records: list[dict[str, Any]] = []
        if self._library is not None:
            try:
                memory_records = self._library.retrieve(
                    task_type=task_type, limit=library_limit
                )
                records.extend(self._validated_strategy_records(memory_records))
            except Exception:
                pass

        for attempt in self._log.tried:
            if not self._is_validated_success(attempt):
                continue
            if str(attempt.get("task_type") or "") != task_type:
                continue
            records.append({
                **attempt,
                "memory_id": attempt.get("attempt_id") or "innovation_attempt",
                "reusable_strategy": attempt.get("strategy") or "",
            })

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            record_id = str(record.get("memory_id") or record.get("attempt_id") or "")
            if record_id and record_id in seen:
                continue
            if record_id:
                seen.add(record_id)
            deduped.append(record)
        return deduped

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
        known = set(known_strategies or [])
        successful_records = self._successful_strategy_evidence(
            task_type, library_limit=60
        )
        if len(successful_records) < 5:
            return []  # Not enough experience to innovate

        # Extract successful strategy components
        strategy_counts: Counter[str] = Counter()
        task_sources: dict[str, list[str]] = defaultdict(list)
        for r in successful_records:
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

    def record_attempt(
        self,
        strategy_name: str,
        success: bool,
        cv_score: Optional[float] = None,
        *,
        metric_delta: Optional[float] = None,
        run_success: Optional[bool] = None,
        promoted: Optional[bool] = None,
        evidence_level: str = "",
        no_training_started: bool = False,
        task_id: str = "",
        task_type: str = "",
        source_memory_ids: Optional[list[str]] = None,
        attempt_id: str = "",
    ) -> dict[str, Any]:
        """Record a real or gated innovation outcome for later decisions.

        ``success`` is retained for API compatibility, but only an explicitly
        validated, promoted run with positive metric delta counts as a successful
        innovation. Gate-only and blueprint records may be persisted as evidence;
        they never increase the tried/success counters.
        """
        requested_success = bool(success)
        if not evidence_level:
            evidence_level = "provisional"

        record: dict[str, Any] = {
            "attempt_id": attempt_id or f"innovation_attempt_{uuid.uuid4().hex[:16]}",
            "task_id": task_id,
            "task_type": task_type,
            "strategy": strategy_name,
            "requested_success": requested_success,
            "success": False,
            "cv_score": cv_score,
            "metric_delta": metric_delta,
            "run_success": run_success,
            "promoted": promoted,
            "evidence_level": evidence_level.strip().lower(),
            "no_training_started": bool(no_training_started),
            "source_memory_ids": list(source_memory_ids or [])[:20],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        validated_success = self._is_validated_success(record)
        executed_attempt = self._is_executed_attempt(record)
        record["success"] = validated_success
        record["attempt_status"] = (
            "validated_success"
            if validated_success
            else "executed_held_or_failed"
            if executed_attempt
            else "gated_or_planning_evidence"
        )

        replaced = False
        for index, item in enumerate(self._log.tried):
            if isinstance(item, dict) and item.get("attempt_id") == record["attempt_id"]:
                self._log.tried[index] = record
                replaced = True
                break
        if not replaced:
            self._log.tried.append(record)

        self._recount_outcomes()
        self._save_log()
        return record

    def stats(self) -> dict[str, Any]:
        """Return current innovation statistics."""
        executed = self._executed_attempts()
        validated = [item for item in executed if self._is_validated_success(item)]
        negative_evidence = [
            item for item in self._log.tried
            if not self._is_validated_success(item)
        ]
        return {
            "proposals_generated": len(self._log.proposals),
            "innovations_tried": len(validated),
            "executed_attempts": len(executed),
            "successes": self._log.successes,
            "failures": self._log.failures,
            "hit_rate": f"{self._log.hit_rate():.1%}",
            "most_successful": self._top_innovations(3, success_only=True),
            "evidence_records": len(self._log.tried),
            "negative_evidence": len(negative_evidence),
        }

    def _top_innovations(self, n: int, success_only: bool = False) -> list[str]:
        items = self._executed_attempts()
        if success_only:
            items = [t for t in items if self._is_validated_success(t)]
        scored = sorted(items, key=lambda t: t.get("metric_delta") or 0, reverse=True)
        return [t.get("strategy", "?") for t in scored[:n]]

    def ready_for_innovation(self, task_type: str) -> bool:
        """Check if there's enough experience to propose meaningful innovations."""
        records = self._successful_strategy_evidence(task_type, library_limit=10)
        # Need multiple independently validated outcomes, not merely plans or
        # gate artifacts that happened to contain reusable_strategy text.
        with_strategies = [record for record in records if record.get("reusable_strategy")]
        return len(with_strategies) >= 5
