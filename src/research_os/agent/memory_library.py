"""Layered experience library over the flat RetrospectiveMemoryStore.

The plan's core thesis is *experience-driven* research: lessons from past
experiments (and past tasks of the same type) must actually reach the next
hypothesis. The flat store already persists records across runs; this library
adds the two access patterns an agent needs:

  * ``index_digest`` — a COMPACT, always-injected summary (grouped by task_type,
    top reusable strategies, failure patterns with counts). Cheap enough to put in
    every session's opening prompt so the agent starts grounded in what it already
    knows — without dumping the whole store into context.
  * ``retrieve`` — DETAILED records on demand (optionally filtered by failure
    pattern), for when the agent wants the specifics behind a digest line.

This is the "index + on-demand" layering the reference agents use for memory:
the small digest is free and constant; the full records are pulled only when the
agent asks. Writes go straight through to the underlying store (single source of
truth, still one JSON file, still UTF-8 — the Windows gbk trap is avoided because
the store already opens with encoding='utf-8').
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from typing import Optional

from ..retrospective_memory import MemoryRecord, RetrospectiveMemoryStore


class MemoryLibrary:
    """Read/write experience with a compact index and on-demand detail."""

    def __init__(self, store: RetrospectiveMemoryStore) -> None:
        self.store = store

    # ── writes go straight through (one source of truth) ──────────────────────
    def add(self, record: MemoryRecord) -> None:
        self.store.add_memory(record)

    def _all(self) -> list[MemoryRecord]:
        return self.store._load()

    @staticmethod
    def _is_validated_promotion(record: MemoryRecord) -> bool:
        profile = record.dataset_profile if isinstance(record.dataset_profile, dict) else {}
        explicit = str(profile.get("evidence_level") or "").strip().lower()
        try:
            delta = float(record.metric_delta) if record.metric_delta is not None else 0.0
            positive_delta = math.isfinite(delta) and delta > 0
        except (TypeError, ValueError):
            positive_delta = False
        return (
            explicit == "validated"
            and profile.get("run_success") is True
            and profile.get("promoted") is True
            and profile.get("no_training_started") is not True
            and positive_delta
        )

    @classmethod
    def _evidence_level(cls, record: MemoryRecord) -> str:
        profile = record.dataset_profile if isinstance(record.dataset_profile, dict) else {}
        explicit = str(profile.get("evidence_level") or "").strip().lower()
        outcome = str(profile.get("outcome_status") or "").strip().lower()
        if outcome in {"held", "failed", "held_or_failed", "rejected"}:
            return "failure"
        if explicit == "failure":
            return "failure"
        if cls._is_validated_promotion(record):
            return "validated"
        if explicit == "observed":
            return "observed"
        if explicit in {"validated", "provisional"}:
            return "provisional"
        method = (record.method or "").lower()
        if method in {"isolated_engineering_validation", "isolated_engineering_trial"}:
            return "observed"
        if record.failure_pattern and not record.what_worked:
            return "failure"
        return "provisional"

    # ── layer 1: the always-injected compact index ────────────────────────────
    def index_digest(self, task_type: Optional[str] = None, *, max_lines: int = 12) -> str:
        """A small text digest for the session-opening prompt.

        Prioritizes the current task_type but also surfaces cross-task signal, so
        a brand-new task still benefits from lessons learned elsewhere. Empty
        library returns a clear placeholder (not a misleading blank)."""
        records = self._all()
        if not records:
            return "(experience library is empty — this is the first run on this project)"

        by_type: Counter[str] = Counter(r.task_type for r in records)
        scoped = [r for r in records if task_type is None or r.task_type == task_type]
        # Only evidence-backed outcomes are called proven. Plans, read-only loop
        # artifacts, and untested blueprints remain provisional hypotheses.
        strategies = Counter(
            r.reusable_strategy for r in scoped
            if r.reusable_strategy and self._evidence_level(r) == "validated"
        )
        observed = Counter(
            r.reusable_strategy for r in scoped
            if r.reusable_strategy and self._evidence_level(r) == "observed"
        )
        provisional = Counter(
            r.reusable_strategy for r in scoped
            if r.reusable_strategy and self._evidence_level(r) == "provisional"
        )
        # Failure patterns, most-common first (the "avoid this" signal).
        failures = Counter(
            r.failure_pattern for r in records
            if r.failure_pattern and (task_type is None or r.task_type == task_type)
        )
        lines = [f"experience library: {len(records)} lessons across "
                 f"{len(by_type)} task types ({dict(by_type.most_common(6))})"]
        if task_type:
            same = sum(1 for r in records if r.task_type == task_type)
            lines.append(f"for THIS task_type={task_type}: {same} lessons")
        if strategies:
            lines.append("evidence-backed strategies (validated promotions): "
                         + ", ".join(f"{s}×{n}" for s, n in strategies.most_common(max_lines)))
        if observed:
            lines.append("observed strategies (not promotion-proven): "
                         + ", ".join(f"{s}×{n}" for s, n in observed.most_common(max_lines)))
        if provisional:
            lines.append("provisional hypotheses (must validate before reuse): "
                         + ", ".join(f"{s}×{n}" for s, n in provisional.most_common(max_lines)))
        if failures:
            lines.append("recurring failure patterns to avoid: "
                         + ", ".join(f"{f}×{n}" for f, n in failures.most_common(max_lines)))
        lines.append("(call read_memory for the detailed lessons behind these.)")
        return "\n".join(lines)

    # ── layer 2: detailed records on demand ───────────────────────────────────
    def retrieve(self, task_type: Optional[str] = None, *,
                 failure_pattern: Optional[str] = None, limit: int = 12) -> list[dict]:
        """Detailed lesson dicts, newest last, optionally filtered.

        ``task_type=None`` returns cross-task lessons (useful when the agent wants
        transferable strategies). ``failure_pattern`` filters to a specific class
        (e.g. 'timeout') so the agent can study exactly what went wrong before."""
        records = self._all()
        if task_type is not None:
            records = [r for r in records if r.task_type == task_type]
        if failure_pattern:
            fp = failure_pattern.strip().lower()
            records = [r for r in records if (r.failure_pattern or "").lower() == fp]
        return [asdict(r) for r in records[-limit:]]

    # ── layer 3: SEMANTIC vector search (scales to 10,000+ records) ──────────
    def semantic_search(self, query: str, *, k: int = 8,
                        task_type: Optional[str] = None) -> list[dict]:
        """Semantic (TF-IDF + cosine) search over all memory records.

        Unlike ``retrieve`` (exact filter), this finds the MOST RELEVANT
        records ranked by text similarity. Query can be natural language:

          "timeout errors when training large models on text data"
          "successful feature engineering with high-cardinality categories"
          "GPU import failures with CatBoost"

        ``task_type`` optionally narrows results but still ranks by semantic
        similarity within that type. Returns record dicts with ``_score``
        (0-1 semantic similarity).

        This is the layer that enables the "AI Scientist" to actually LEARN:
        with 5000 records, exact "regression" filter returns 3000 results;
        semantic search returns the 8 that are TRULY relevant to the query.
        """
        from ..memory_vector_index import MemoryVectorIndex

        index = MemoryVectorIndex(self.store)
        index.build()
        return index.query(query, k=k, task_type=task_type)
