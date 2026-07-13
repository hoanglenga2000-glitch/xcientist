"""
MLEvolve Retrospective Memory System
Based on: MLEvolve (arXiv:2606.06473v1) Section 3.3

Combines two complementary memory stores:
1. Cold-start Knowledge Base: Curated domain priors (model templates, architecture patterns,
   feature engineering recipes, known pitfalls)
2. Dynamic Global Experience: Automatically accumulated task-specific records during search
   (plans, code, metrics, analysis, errors, fixes)

Retrieval: BM25 (lexical) + embedding similarity -> RRF fusion -> Top-K records
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re
import math


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class KnowledgeEntry:
    """A single entry in the cold-start knowledge base."""
    entry_id: str
    category: str                        # model_prior, feature_recipe, architecture, pitfall, baseline
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)   # Which tasks this applies to
    quality_score: float = 0.5           # Human or auto-assessed quality
    source: str = "curated"             # curated, learned, imported
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceRecord:
    """A dynamic experience record from the ongoing search."""
    record_id: str
    task_id: str
    run_id: str
    record_type: str                     # plan, code, metric, error, fix, insight, ablation
    content: str
    score_delta: Optional[float] = None  # How much this change improved/worsened score
    parent_record_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Cold-Start Knowledge Base ───────────────────────────────────────────────

class KnowledgeBase:
    """Curated domain knowledge for cold-start initialization."""

    # Pre-built knowledge entries covering common Kaggle patterns
    DEFAULT_ENTRIES: list[dict] = [
        {
            "category": "model_prior",
            "title": "GBDT Ensemble Baseline",
            "content": "LightGBM + XGBoost + CatBoost 5-fold ensemble with Optuna tuning. "
                       "Works well for tabular classification/regression. Use early_stopping_rounds=50, "
                       "n_estimators=2000+ for CatBoost, learning_rate=0.01-0.05.",
            "tags": ["ensemble", "gbdt", "tabular", "baseline"],
            "tasks": ["*"]
        },
        {
            "category": "model_prior",
            "title": "HistGradientBoosting Fast Baseline",
            "content": "sklearn HistGradientBoostingClassifier/Regressor as fast first baseline. "
                       "No missing value imputation needed. Use max_iter=500, learning_rate=0.05, "
                       "max_depth=None for best results. 3-fold CV sufficient for initial screening.",
            "tags": ["sklearn", "fast", "baseline", "tabular"],
            "tasks": ["*"]
        },
        {
            "category": "feature_recipe",
            "title": "Categorical Encoding Strategy",
            "content": "For high-cardinality categorical features: target encoding with 5-fold "
                       "out-of-fold to prevent leakage. For low-cardinality (<10): one-hot. "
                       "For ordinal: label encoding. Always handle unseen categories in test set.",
            "tags": ["encoding", "categorical", "tabular", "preprocessing"],
            "tasks": ["*"]
        },
        {
            "category": "feature_recipe",
            "title": "Missing Value Strategy",
            "content": "Add missing indicator columns for features with >5% missing. "
                       "Impute numerical: median. Impute categorical: mode or 'missing' token. "
                       "Tree-based models handle missing natively but indicator columns often help.",
            "tags": ["missing", "imputation", "tabular", "preprocessing"],
            "tasks": ["*"]
        },
        {
            "category": "architecture",
            "title": "Two-Stage Classification Pipeline",
            "content": "Stage 1: Train base models (LGB, XGB, CatBoost). Stage 2: Train meta-model "
                       "(LogisticRegression or simple weighted blend) on OOF predictions. "
                       "Use 5-fold stratified CV. Calibrate with isotonic regression if needed.",
            "tags": ["ensemble", "stacking", "classification"],
            "tasks": ["*"]
        },
        {
            "category": "pitfall",
            "title": "CV-Public Gap Detection",
            "content": "If CV > public LB by >0.005 for accuracy/AUC tasks, suspect: "
                       "1) Target leakage in features, 2) Overfitting to CV split, "
                       "3) Test distribution shift. Mitigation: adversarial validation, "
                       "feature selection by importance stability across folds.",
            "tags": ["validation", "overfitting", "tabular"],
            "tasks": ["*"]
        },
        {
            "category": "pitfall",
            "title": "Submission Format Errors",
            "content": "Common failures: 1) Wrong column names (case-sensitive), "
                       "2) Wrong row count (must match test.csv exactly), "
                       "3) Float precision issues (use np.round with appropriate decimals), "
                       "4) Index column accidentally included as prediction.",
            "tags": ["submission", "validation", "tabular"],
            "tasks": ["*"]
        },
        {
            "category": "baseline",
            "title": "Spaceship Titanic Specific",
            "content": "Key features: Cabin deck/num/side split, group features by PassengerId prefix, "
                       "spending ratio features (RoomService/FoodCourt/etc.). Target: Transported (bool). "
                       "Metric: accuracy. Ensemble blend weights: HGB=0.5, CatBoost=0.3, LGB=0.2 works well.",
            "tags": ["spaceship_titanic", "classification", "feature_engineering"],
            "tasks": ["spaceship_titanic"]
        },
        {
            "category": "baseline",
            "title": "House Prices Specific",
            "content": "Target: log(SalePrice). Metric: RMSLE. Key features: total SF, "
                       "quality * area interactions, neighborhood target encoding. "
                       "Remove outliers (>400000 SalePrice). Use LGB+Lasso stack.",
            "tags": ["house_prices", "regression", "feature_engineering"],
            "tasks": ["house_prices", "house_prices_advanced_regression_techniques"]
        },
    ]

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("workspace/knowledge_base")
        self.entries: dict[str, KnowledgeEntry] = {}
        self._load_or_init()

    def _load_or_init(self):
        """Load existing KB or initialize with defaults."""
        kb_file = self.storage_path / "knowledge_base.json"
        if kb_file.exists():
            data = json.loads(kb_file.read_text())
            for item in data.get("entries", []):
                entry = KnowledgeEntry(**item)
                self.entries[entry.entry_id] = entry
        else:
            self._init_defaults()

    def _init_defaults(self):
        for item in self.DEFAULT_ENTRIES:
            entry_id = f"kb_{item['category']}_{hash(item['title']) & 0xFFFF:04x}"
            entry = KnowledgeEntry(
                entry_id=entry_id,
                category=item["category"],
                title=item["title"],
                content=item["content"],
                tags=item.get("tags", []),
                tasks=item.get("tasks", ["*"]),
                source="builtin"
            )
            self.entries[entry_id] = entry

    def query(self, task_id: str, query_text: str = "",
              categories: Optional[list[str]] = None,
              top_k: int = 5) -> list[KnowledgeEntry]:
        """Retrieve relevant knowledge entries for a task."""
        candidates = []
        for entry in self.entries.values():
            # Filter by task applicability
            if "*" not in entry.tasks and task_id not in entry.tasks:
                continue
            if categories and entry.category not in categories:
                continue

            # Score by BM25-like relevance
            score = self._bm25_score(entry, query_text)
            if score > 0:
                candidates.append((score, entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in candidates[:top_k]]

    def _bm25_score(self, entry: KnowledgeEntry, query: str, k1: float = 1.2, b: float = 0.75) -> float:
        """Simple BM25 scoring. For full BM25+FAISS, use embedding retrieval."""
        if not query:
            return entry.quality_score

        doc = f"{entry.title} {entry.content} {' '.join(entry.tags)}"
        query_terms = set(re.findall(r'\w+', query.lower()))
        doc_terms = re.findall(r'\w+', doc.lower())
        doc_len = len(doc_terms)
        avg_doc_len = 50  # approximate

        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            df = sum(1 for e in self.entries.values()
                    if term in f"{e.title} {e.content}".lower())
            idf = math.log((len(self.entries) - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            score += idf * tf_norm

        return score * entry.quality_score

    def add_entry(self, entry: KnowledgeEntry):
        self.entries[entry.entry_id] = entry
        self._save()

    def _save(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "academic_research_os.knowledge_base.v1",
            "entries": [
                {
                    "entry_id": e.entry_id, "category": e.category,
                    "title": e.title, "content": e.content, "tags": e.tags,
                    "tasks": e.tasks, "quality_score": e.quality_score,
                    "source": e.source, "metadata": e.metadata
                }
                for e in self.entries.values()
            ]
        }
        (self.storage_path / "knowledge_base.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False))


# ── Dynamic Global Experience ───────────────────────────────────────────────

class GlobalExperience:
    """
    Dynamically accumulated task-specific experience during search.

    Records: plans, code, metrics, errors, fixes, insights, ablations.
    Retrieval: Hybrid BM25 + embedding cosine -> RRF fusion -> Top-K.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("workspace/global_experience")
        self.records: dict[str, ExperienceRecord] = {}
        self._load_or_init()

    def _load_or_init(self):
        exp_file = self.storage_path / "global_experience.json"
        if exp_file.exists():
            data = json.loads(exp_file.read_text())
            for item in data.get("records", []):
                rec = ExperienceRecord(**item)
                self.records[rec.record_id] = rec

    def record(self, task_id: str, run_id: str, record_type: str,
               content: str, score_delta: Optional[float] = None,
               tags: Optional[list[str]] = None,
               metadata: Optional[dict] = None) -> ExperienceRecord:
        record_id = f"exp_{task_id}_{run_id}_{record_type}_{int(time.time()*1000)}"
        rec = ExperienceRecord(
            record_id=record_id,
            task_id=task_id,
            run_id=run_id,
            record_type=record_type,
            content=content,
            score_delta=score_delta,
            tags=tags or [],
            metadata=metadata or {}
        )
        self.records[record_id] = rec
        return rec

    def query(self, task_id: str, query_text: str = "",
              record_types: Optional[list[str]] = None,
              top_k: int = 8) -> list[ExperienceRecord]:
        """Hybrid retrieval: BM25 + time-decay ranking."""
        candidates = []
        for rec in self.records.values():
            if rec.task_id != task_id:
                continue
            if record_types and rec.record_type not in record_types:
                continue

            bm25 = self._bm25_score(rec, query_text)
            # Time decay: recent records get bonus
            hours_ago = (time.time() - rec.created_at) / 3600
            time_weight = math.exp(-hours_ago / 24)  # 24h half-life

            # Score delta bonus: successful changes weighted higher
            delta_bonus = 0.0
            if rec.score_delta is not None:
                delta_bonus = max(0, rec.score_delta) * 2  # reward improvements

            score = bm25 * time_weight + delta_bonus
            if score > 0:
                candidates.append((score, rec))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in candidates[:top_k]]

    def _bm25_score(self, rec: ExperienceRecord, query: str) -> float:
        if not query:
            return 0.3  # default relevance
        doc = f"{rec.record_type} {rec.content} {' '.join(rec.tags)}"
        query_terms = set(re.findall(r'\w+', query.lower()))
        doc_terms = re.findall(r'\w+', doc.lower())
        score = sum(1.0 for t in query_terms if t in doc_terms)
        return score / max(len(query_terms), 1)

    def get_successful_patterns(self, task_id: str, min_delta: float = 0.001) -> list[ExperienceRecord]:
        """Get records that led to score improvements."""
        return sorted(
            [r for r in self.records.values()
             if r.task_id == task_id and r.score_delta is not None and r.score_delta > min_delta],
            key=lambda r: r.score_delta or 0, reverse=True
        )

    def get_error_patterns(self, task_id: str) -> list[ExperienceRecord]:
        """Get error records to avoid repeating failures."""
        return [r for r in self.records.values()
                if r.task_id == task_id and r.record_type in ("error", "fix")]

    def save(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [
                {
                    "record_id": r.record_id, "task_id": r.task_id,
                    "run_id": r.run_id, "record_type": r.record_type,
                    "content": r.content, "score_delta": r.score_delta,
                    "tags": r.tags, "created_at": r.created_at,
                    "metadata": r.metadata
                }
                for r in self.records.values()
            ]
        }
        (self.storage_path / "global_experience.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ── Retrospective Memory (unified) ──────────────────────────────────────────

class RetrospectiveMemory:
    """
    Unified memory system combining cold-start Knowledge Base
    and dynamic Global Experience for task-specific retrieval.

    Usage in search loop:
        memory = RetrospectiveMemory(task_id)
        # Before generating plan:
        knowledge = memory.get_context("How to improve CV for tabular classification")
        # After experiment:
        memory.record_experience(run_id, "code", code_content, score_delta=0.002)
    """

    def __init__(self, task_id: str, workspace_root: Optional[Path] = None):
        root = workspace_root or Path(".")
        self.task_id = task_id
        self.kb = KnowledgeBase(root / "workspace" / "knowledge_base")
        self.experience = GlobalExperience(root / "workspace" / "global_experience")

    def get_context(self, query: str = "", top_k: int = 5,
                    include_experience: bool = True) -> dict:
        """Get combined knowledge + experience context for planning."""
        kb_entries = self.kb.query(self.task_id, query, top_k=top_k)
        kb_text = "\n\n".join(
            f"[{e.category}] {e.title}: {e.content[:300]}" for e in kb_entries
        )

        exp_text = ""
        if include_experience:
            exp_records = self.experience.query(self.task_id, query, top_k=top_k)
            exp_text = "\n\n".join(
                f"[{r.record_type}] (delta={r.score_delta}): {r.content[:300]}"
                for r in exp_records
            )

        return {
            "knowledge_base": kb_text,
            "experience": exp_text,
            "combined": f"KNOWLEDGE BASE:\n{kb_text}\n\nTASK EXPERIENCE:\n{exp_text}"
        }

    def record_experience(self, run_id: str, record_type: str,
                          content: str, score_delta: Optional[float] = None,
                          tags: Optional[list[str]] = None):
        return self.experience.record(
            self.task_id, run_id, record_type, content, score_delta, tags)

    def get_successful_patterns(self) -> list[ExperienceRecord]:
        return self.experience.get_successful_patterns(self.task_id)

    def get_error_patterns(self) -> list[ExperienceRecord]:
        return self.experience.get_error_patterns(self.task_id)

    def get_cross_task_patterns(self, source_task_id: str = None,
                                 min_delta: float = 0.001) -> list[ExperienceRecord]:
        """Get successful patterns from other tasks that may transfer here."""
        kb_tags = set()
        for entry in self.kb.query(self.task_id, "", top_k=20):
            kb_tags.update(entry.tags)

        candidates = []
        for rec in self.experience.records.values():
            if source_task_id and rec.task_id != source_task_id:
                continue
            if rec.task_id == self.task_id:
                continue
            if rec.score_delta is None or rec.score_delta <= min_delta:
                continue
            rec_tags = set(rec.tags)
            if not kb_tags or rec_tags & kb_tags:
                candidates.append(rec)
        return sorted(candidates, key=lambda r: r.score_delta or 0, reverse=True)
