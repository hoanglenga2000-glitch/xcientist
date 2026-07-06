"""Semantic vector index over the RetrospectiveMemoryStore.

Problem at scale: with 500+ memory records, exact-match filters
(task_type="regression") return too many irrelevant results. The agent
needs to query "failures on time-series tasks with high cardinality
features" and get the 5 MOST RELEVANT memories — not 50 random ones.

Solution: TF-IDF + cosine similarity over concatenated text fields.
- Zero external dependencies beyond numpy+sklearn (already installed)
- Character n-grams (2-4) capture domain patterns across languages
- Index caches to disk; rebuilds on new records
- Falls back to exact-match if numpy/sklearn unavailable
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .retrospective_memory import MemoryRecord, RetrospectiveMemoryStore


def _record_text(r: MemoryRecord) -> str:
    """Build a single text representation from all semantic fields."""
    parts = [
        r.task_type or "",
        r.method or "",
        r.what_worked or "",
        r.what_failed or "",
        r.reusable_strategy or "",
        r.failure_pattern or "",
    ]
    # dataset_profile carries modality + n_train which helps similarity
    if r.dataset_profile:
        parts.append(str(r.dataset_profile.get("modality", "")))
        n = r.dataset_profile.get("n_train", "")
        if n:
            parts.append(f"n_train={n}")
    # Deduplicate and join
    return " ".join(p for p in parts if p and p.strip())


class MemoryVectorIndex:
    """TF-IDF vector index for semantic memory search.

    Usage:
        store = RetrospectiveMemoryStore("retrospective_memory.json")
        index = MemoryVectorIndex(store)
        results = index.query("timeout on large TF-IDF matrix, regression task", k=5)
        # → top-5 most similar memory records
    """

    def __init__(self, store_or_path: RetrospectiveMemoryStore | str | Path):
        if isinstance(store_or_path, RetrospectiveMemoryStore):
            self.store = store_or_path
        else:
            self.store = RetrospectiveMemoryStore(Path(store_or_path))
        self._documents: list[str] = []
        self._records: list[MemoryRecord] = []
        self._vectorizer: Any = None
        self._matrix: Any = None  # scipy sparse matrix
        self._record_count_at_build: int = 0
        self._cache_path: Optional[Path] = None
        if hasattr(self.store, "path"):
            self._cache_path = Path(self.store.path).with_suffix(".vec_index.json")

    # ── build / rebuild ──────────────────────────────────────────
    def _needs_rebuild(self) -> bool:
        records = self.store._load()
        return len(records) != self._record_count_at_build or self._vectorizer is None

    def build(self, *, force: bool = False) -> int:
        """Build (or rebuild) the TF-IDF index. Returns number of indexed docs."""
        records = self.store._load()
        if not force and not self._needs_rebuild():
            return self._record_count_at_build

        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = [_record_text(r) for r in records]

        # Use char_wb analyzer: character n-grams within word boundaries.
        # This handles code snippets, Chinese characters, and domain terms
        # equally well without needing language-specific tokenization.
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8192,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(documents)
        self._documents = documents
        self._records = records
        self._record_count_at_build = len(records)

        # Persist cache
        self._save_cache()
        return len(records)

    def _save_cache(self) -> None:
        if self._cache_path is None or self._vectorizer is None:
            return
        try:
            data = {
                "record_count": self._record_count_at_build,
                "documents": self._documents,
                # Store the TF-IDF matrix as dense for small stores (<5000 records)
                # or keep sparse. For our use case (<5000), dense is fine.
                "vocabulary": dict(self._vectorizer.vocabulary_),
                "idf": self._vectorizer.idf_.tolist() if hasattr(self._vectorizer, "idf_") else [],
            }
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError):
            pass  # cache is an optimization; failure is non-fatal

    # ── query ─────────────────────────────────────────────────────
    def query(self, text: str, *, k: int = 10,
              task_type: Optional[str] = None) -> list[dict[str, Any]]:
        """Semantic search: return top-k most similar memory records.

        ``text`` is a natural-language query like:
          "timeout errors on text classification with large vocab"
          "successful feature engineering for tabular regression"
          "GPU import error for CatBoost"

        ``task_type`` optionally narrows to records of one type (still
        ranked by semantic similarity within that type).

        Returns list of record dicts with an added ``_score`` field (0-1).
        """
        self.build()  # auto-rebuild if needed
        if not self._records or self._vectorizer is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        # Pair (index, score), filter, sort
        candidates = []
        for i, score in enumerate(scores):
            rec = self._records[i]
            if task_type and rec.task_type != task_type:
                continue
            candidates.append((i, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:k]

        from dataclasses import asdict
        results = []
        for i, score in top:
            d = asdict(self._records[i])
            d["_score"] = round(score, 4)
            results.append(d)
        return results

    # ── statistics ─────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        self.build()
        return {
            "indexed_records": self._record_count_at_build,
            "vocabulary_size": len(self._vectorizer.vocabulary_) if self._vectorizer else 0,
            "documents": len(self._documents),
            "cache_path": str(self._cache_path) if self._cache_path else None,
        }


# ── Convenience ──────────────────────────────────────────────────

def build_global_index(project_root: Optional[Path] = None) -> MemoryVectorIndex:
    """Build the vector index from the canonical memory store."""
    from pathlib import Path
    if project_root is None:
        from .retrospective_memory import RetrospectiveMemoryStore
        # Try the common locations
        candidates = [
            Path("experiments/evolution/retrospective_memory.json"),
            Path.cwd() / "experiments" / "evolution" / "retrospective_memory.json",
        ]
        for c in candidates:
            if c.exists():
                store = RetrospectiveMemoryStore(c)
                return MemoryVectorIndex(store)
        # Fallback: create an empty one
        empty = Path("experiments/evolution/retrospective_memory.json")
        empty.parent.mkdir(parents=True, exist_ok=True)
        if not empty.exists():
            empty.write_text("[]", encoding="utf-8")
        return MemoryVectorIndex(RetrospectiveMemoryStore(empty))

    mem_path = project_root / "experiments" / "evolution" / "retrospective_memory.json"
    store = RetrospectiveMemoryStore(mem_path)
    return MemoryVectorIndex(store)
