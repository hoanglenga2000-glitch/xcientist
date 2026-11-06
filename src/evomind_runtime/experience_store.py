"""Phase 2 knowledge base: training experience in ``knowledge.sqlite3``.

Three scopes:

* ``private``  - one row per tenant training attempt, readable and deletable only by its owner
  (tenant + owner; the project is kept so same-project experience ranks first).
* ``anonymous_global`` - ``global_priors``, rebuilt nightly from private *successes* of owners who did
  not opt out. Only enum / numeric fields (task, data-profile buckets, trainer numbers, metric medians);
  never data, file names, prompts or text. A pattern needs at least ``GLOBAL_MIN_TENANTS`` tenants.
* ``platform`` - imported research knowledge and retrospectives, offered to operator sessions only.

Rows are written deterministically by the training tool (no LLM). Search uses SQLite FTS5 with the
trigram tokenizer (Chinese without a segmenter) and BM25; a LIKE scan is the fallback when the
tokenizer is missing.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from typing import Any, Iterable

from .models import new_id, utc_now
from .tenant_access import AccessError, Principal

DB_NAME = "knowledge.sqlite3"
EXPERIENCE_ID = re.compile(r"exp_[a-f0-9]{32}")
LIST_LIMIT = 200
GLOBAL_MIN_TENANTS = 3
RECALL_PRIVATE = 3
RECALL_GLOBAL = 3
RECALL_CHAR_BUDGET = 1500
PLATFORM_RECALL = 3
# Refusals that say nothing about the data or the configuration: no experience is written.
NOT_EXPERIENCE_CODES = frozenset({
    "exact_approval_required", "tenant_training_queue", "tenant_training_tenant_training_disabled",
    "tenant_training_config_invalid_repeated", "gpu_cancelled", "tenant_training_cancelled",
    "tenant_training_remote_path_rejected", "tool_breaker_open", "tool_breaker_repeat",
})
TRAINER_FIELDS = ("hidden_sizes", "epochs", "batch_size", "learning_rate", "model_family")
PUBLIC_FIELDS = ("id", "task", "outcome", "rows_bucket", "cols_bucket", "target_kind", "balance_bucket", "card",
                 "primary_metric", "primary_value", "elapsed_seconds", "failure_code", "fixed_failure_id",
                 "success_count", "use_count", "project_id", "title", "created_at")
FAILURE_TEXT = {
    "tenant_training_config_invalid": "训练参数不合法",
    "tenant_training_data_invalid": "数据格式不符合要求",
    "tenant_training_data_too_many_cells": "数据太大",
    "tenant_training_data_changed": "数据在确认后被改动",
    "tenant_training_environment_not_ready": "训练环境没有准备好",
    "tenant_training_identity_gate_not_ready": "算力卡没有通过检查",
    "tenant_training_failed": "训练没有完成",
    "tenant_training_unverified": "结果文件没有通过校验",
    "tenant_training_stopped": "达到最长运行时间被停止",
    "tenant_training_timeout": "训练超时",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL, tenant_id TEXT NOT NULL DEFAULT '', owner_id TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL, source_key TEXT UNIQUE,
    run_id TEXT, data_key TEXT, task TEXT, outcome TEXT,
    rows_bucket TEXT, cols_bucket TEXT, target_kind TEXT, balance_bucket TEXT,
    config_json TEXT, metrics_json TEXT, primary_metric TEXT, primary_value REAL,
    elapsed_seconds REAL, card TEXT, failure_code TEXT, fixed_failure_id TEXT, fixed_by_id TEXT,
    title TEXT NOT NULL, body TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0, use_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS experiences_owner ON experiences(scope, tenant_id, owner_id, deleted_at);
CREATE INDEX IF NOT EXISTS experiences_data ON experiences(tenant_id, owner_id, data_key, outcome);
CREATE TABLE IF NOT EXISTS global_priors (
    pattern_key TEXT PRIMARY KEY, task TEXT NOT NULL, rows_bucket TEXT, cols_bucket TEXT, target_kind TEXT,
    balance_bucket TEXT, config_json TEXT NOT NULL, tenants INTEGER NOT NULL, runs INTEGER NOT NULL,
    primary_metric TEXT, median_metric REAL, median_elapsed REAL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experience_settings (
    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, contribute_global INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL, PRIMARY KEY(tenant_id, owner_id)
);
CREATE TABLE IF NOT EXISTS knowledge_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


# ---------------------------------------------------------------- data profile


def bucket_rows(rows: int) -> str:
    return "lt1k" if rows < 1000 else "1k-10k" if rows < 10_000 else "10k-100k" if rows < 100_000 else "gt100k"


def bucket_cols(cols: int) -> str:
    return "lt10" if cols < 10 else "10-50" if cols < 50 else "50-200" if cols < 200 else "gt200"


def data_profile(task: str, rows: int, cols: int, labels: Iterable[Any] | None = None, classes: int = 0) -> dict[str, str]:
    """Bucketed shape of one dataset: the only description of data that ever leaves the private scope."""
    profile = {"rows_bucket": bucket_rows(int(rows)), "cols_bucket": bucket_cols(int(cols))}
    if task == "regression":
        return {**profile, "target_kind": "regression", "balance_bucket": "na"}
    profile["target_kind"] = "binary" if int(classes) == 2 else "multiclass" if int(classes) > 2 else "unknown"
    counts: dict[int, int] = {}
    for value in labels if labels is not None else ():
        counts[int(value)] = counts.get(int(value), 0) + 1
    if len(counts) < 2:
        profile["balance_bucket"] = "unknown"
    else:
        ratio = min(counts.values()) / max(counts.values())
        profile["balance_bucket"] = "balanced" if ratio >= 0.5 else "moderate" if ratio >= 0.2 else "imbalanced"
    return profile


def quick_csv_shape(path: Path, *, max_rows: int = 200_000) -> tuple[int, int] | None:
    """(data rows, feature columns) of a CSV without parsing numbers; None when unreadable."""
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = 0
            for row in reader:
                if row:
                    rows += 1
                if rows >= max_rows:
                    break
        return rows, max(0, len(header) - 1)
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
        return None


def primary_metric(task: str, metrics: dict[str, Any] | None) -> tuple[str, float | None]:
    """(name, value), higher is better: macro_f1 for classification; r2, else negative rmse, for regression."""
    metrics = metrics if isinstance(metrics, dict) else {}
    candidates = (("macro_f1", 1.0),) if task == "classification" else (("r2", 1.0), ("rmse", -1.0))
    for name, sign in candidates:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return (name if sign > 0 else "neg_" + name), round(sign * float(value), 6)
    return candidates[0][0], None


def guess_task_profile(path: Path, *, max_rows: int = 200_000) -> tuple[str, dict[str, str]] | None:
    """(task, profile) of a CSV whose target is the last column, before any configuration exists (recall only).

    Non-negative integer targets with at most 50 distinct values read as classification; anything else as regression.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            counts: dict[str, int] = {}
            rows, integral = 0, True
            for row in reader:
                if not row:
                    continue
                rows += 1
                cell = row[-1].strip()
                if integral:
                    try:
                        number = float(cell)
                        integral = number >= 0 and number == int(number) and len(counts) <= 50
                    except ValueError:
                        integral = False
                    if integral:
                        counts[str(int(number))] = counts.get(str(int(number)), 0) + 1
                if rows >= max_rows:
                    break
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
        return None
    if rows < 10 or len(header) < 2:
        return None
    cols = len(header) - 1
    if integral and 2 <= len(counts) <= 50:
        labels = [int(key) for key, count in counts.items() for _ in range(count)]
        return "classification", data_profile("classification", rows, cols, labels=labels, classes=len(counts))
    return "regression", data_profile("regression", rows, cols)


def trainer_numbers(config: dict[str, Any] | None) -> dict[str, Any]:
    """Only the numeric / enum trainer fields (never paths or column names)."""
    config = config if isinstance(config, dict) else {}
    kept: dict[str, Any] = {}
    hidden = config.get("hidden_sizes")
    if isinstance(hidden, (list, tuple)) and hidden and all(type(item) is int and 0 < item <= 65536 for item in hidden):
        kept["hidden_sizes"] = [int(item) for item in hidden][:8]
    for name in ("epochs", "batch_size"):
        if type(config.get(name)) is int and 0 < config[name] <= 1_000_000:
            kept[name] = int(config[name])
    rate = config.get("learning_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 < float(rate) < 10:
        kept["learning_rate"] = float(rate)
    if config.get("model_family") in {"auto", "mlp", "gbdt"}:
        kept["model_family"] = config["model_family"]
    if config.get("class_weight") in {"none", "balanced"}:
        kept["class_weight"] = config["class_weight"]
    for name in ("max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf"):
        if type(config.get(name)) is int and 0 < config[name] <= 100_000:
            kept[name] = int(config[name])
    return kept


def config_text(config: dict[str, Any]) -> str:
    parts = []
    if config.get("model_family"):
        parts.append({"mlp": "神经网络", "gbdt": "梯度提升树", "auto": "自动选择"}.get(config["model_family"], config["model_family"]))
    if config.get("hidden_sizes"):
        parts.append("隐藏层 " + "x".join(str(item) for item in config["hidden_sizes"]))
    for name, label in (("epochs", "轮数"), ("batch_size", "批大小"), ("learning_rate", "学习率"), ("max_iter", "迭代"), ("max_leaf_nodes", "叶子数")):
        if name in config:
            parts.append(f"{label} {config[name]:g}" if isinstance(config[name], float) else f"{label} {config[name]}")
    if config.get("class_weight") == "balanced":
        parts.append("类别加权")
    return "，".join(parts) or "默认设置"


_PROFILE_TEXT = {"lt1k": "千行以内", "1k-10k": "1千到1万行", "10k-100k": "1万到10万行", "gt100k": "10万行以上",
                 "lt10": "10列以内", "10-50": "10到50列", "50-200": "50到200列", "gt200": "200列以上",
                 "binary": "二分类", "multiclass": "多分类", "regression": "回归",
                 "balanced": "类别均衡", "moderate": "类别略不均衡", "imbalanced": "类别很不均衡"}


def profile_text(row: dict[str, Any]) -> str:
    keys = ("target_kind", "rows_bucket", "cols_bucket", "balance_bucket")
    return "，".join(_PROFILE_TEXT[row[key]] for key in keys if row.get(key) in _PROFILE_TEXT)


def metric_text(name: str | None, value: float | None) -> str:
    if value is None or not name:
        return ""
    label = {"macro_f1": "macro-F1", "r2": "R²", "neg_rmse": "RMSE"}.get(name, name)
    shown = -value if name.startswith("neg_") else value
    return f"{label} {shown:.3f}"


def fts_query(text: str) -> str:
    """FTS5 trigram MATCH expression: quoted 3-char windows of CJK runs and latin words of 3+ chars, OR-joined."""
    terms: list[str] = []
    lowered = str(text or "").casefold()
    for word in re.findall(r"[a-z0-9_]{3,}", lowered):
        terms.append(word)
    for run in re.findall(r"[\u4e00-\u9fff]{3,}", lowered):
        terms.extend(run[index:index + 3] for index in range(len(run) - 2))
    unique = list(dict.fromkeys(terms))[:24]
    return " OR ".join('"' + term.replace('"', '') + '"' for term in unique)


def bm25_rank(query: str, documents: list[str], *, k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Okapi BM25 over latin words + CJK bigrams (same tokens as user_memory.tokens); one score per document."""
    from .user_memory import tokens

    wanted = tokens(query)
    if not wanted or not documents:
        return [0.0 for _ in documents]
    bags = []
    for document in documents:
        lowered = str(document).casefold()
        counts: dict[str, int] = {}
        for term in wanted:
            hits = lowered.count(term)
            if hits:
                counts[term] = hits
        bags.append((counts, max(1, len(tokens(document)))))
    average = sum(length for _, length in bags) / len(bags)
    total = len(documents)
    frequency = {term: sum(1 for counts, _ in bags if term in counts) for term in wanted}
    scores = []
    for counts, length in bags:
        score = 0.0
        for term, hits in counts.items():
            idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * hits * (k1 + 1) / (hits + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def _canonical(config: dict[str, Any]) -> str:
    return json.dumps(trainer_numbers(config), sort_keys=True, separators=(",", ":"))


def _flatten_text(value: Any, limit: int = 1200) -> str:
    """Strings of a JSON document (depth first), joined and capped: the searchable body of a platform entry."""
    out: list[str] = []
    size = 0

    def walk(item: Any) -> None:
        nonlocal size
        if size >= limit:
            return
        if isinstance(item, str):
            text = " ".join(item.split())
            if text and not re.fullmatch(r"[a-f0-9]{32,}", text):
                out.append(text)
                size += len(text) + 1
        elif isinstance(item, dict):
            for key in sorted(item):
                walk(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                walk(element)

    walk(value)
    return " ".join(out)[:limit]


class ExperienceStore:
    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / DB_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fts = True
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            try:
                connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(title, body, tokenize='trigram')")
            except sqlite3.OperationalError:
                self.fts = False  # no trigram tokenizer in this SQLite build: LIKE fallback

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def owner(principal: Principal | None) -> tuple[str, str]:
        if principal is None or not principal.tenant_id or not principal.owner_id:
            raise AccessError(403, "experience_principal_required")
        return principal.tenant_id, principal.owner_id

    @staticmethod
    def public(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = {key: row[key] for key in PUBLIC_FIELDS}
        item["summary"] = row["body"]
        try:
            item["config"] = json.loads(row["config_json"] or "{}")
        except ValueError:
            item["config"] = {}
        return item

    def _insert(self, connection: sqlite3.Connection, values: dict[str, Any]) -> str | None:
        columns = list(values)
        cursor = connection.execute(
            f"INSERT OR IGNORE INTO experiences({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
            [values[name] for name in columns])
        if not cursor.rowcount:
            return None
        if self.fts:
            connection.execute("INSERT INTO experiences_fts(rowid,title,body) VALUES(?,?,?)",
                               (cursor.lastrowid, values["title"], values["body"]))
        return str(values["id"])

    def _reindex(self, connection: sqlite3.Connection, seq: int, title: str, body: str | None) -> None:
        if not self.fts:
            return
        connection.execute("DELETE FROM experiences_fts WHERE rowid=?", (seq,))
        if body is not None:
            connection.execute("INSERT INTO experiences_fts(rowid,title,body) VALUES(?,?,?)", (seq, title, body))

    # ------------------------------------------------------------ private: write

    def record_training(self, principal: Principal | None, *, run_id: str, data_key: str, task: str,
                        profile: dict[str, str] | None, config: dict[str, Any] | None, outcome: str,
                        metrics: dict[str, Any] | None = None, failure_code: str = "", elapsed_seconds: float | None = None,
                        card: str = "", project_id: str = "", used_ids: Iterable[str] = ()) -> str | None:
        """One private experience per (run, data, configuration, outcome); repeats are ignored. Returns the new id."""
        tenant_id, owner_id = self.owner(principal)
        if outcome not in {"success", "failed"} or task not in {"classification", "regression"}:
            raise ValueError("experience_invalid")
        if outcome == "failed" and failure_code in NOT_EXPERIENCE_CODES:
            return None
        profile = profile or {}
        numbers = trainer_numbers(config)
        name, value = primary_metric(task, metrics) if outcome == "success" else (primary_metric(task, {})[0], None)
        kept_metrics = {key: round(float(item), 6) for key, item in (metrics or {}).items()
                        if key in {"accuracy", "macro_f1", "roc_auc", "rmse", "mae", "r2"} and isinstance(item, (int, float))
                        and not isinstance(item, bool) and math.isfinite(float(item))}
        shape = profile_text(profile)
        card = re.sub(r"[^A-Za-z0-9 _.-]", "", str(card or ""))[:32]
        if outcome == "success":
            title = "成功：" + "，".join(part for part in (shape, metric_text(name, value)) if part)
        else:
            title = "失败：" + FAILURE_TEXT.get(failure_code, "训练没有完成") + (f"（{shape}）" if shape else "")
        body_parts = [shape, "设置：" + config_text(numbers)]
        if outcome == "success" and value is not None:
            body_parts.append("结果：" + metric_text(name, value))
        if outcome == "failed":
            body_parts.append("原因：" + FAILURE_TEXT.get(failure_code, "训练没有完成"))
        if card:
            body_parts.append("算力卡：" + card)
        if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds > 0:
            body_parts.append(f"用时约 {max(1, round(float(elapsed_seconds) / 60))} 分钟")
        experience_id = new_id("exp")
        values = {
            "id": experience_id, "scope": "private", "tenant_id": tenant_id, "owner_id": owner_id,
            "project_id": str(project_id or "")[:100], "kind": "training",
            "source_key": f"train:{tenant_id}:{owner_id}:{run_id}:{data_key}:{_canonical(numbers)}:{outcome}:{failure_code}",
            "run_id": str(run_id)[:100], "data_key": str(data_key)[:64], "task": task, "outcome": outcome,
            "rows_bucket": profile.get("rows_bucket"), "cols_bucket": profile.get("cols_bucket"),
            "target_kind": profile.get("target_kind"), "balance_bucket": profile.get("balance_bucket"),
            "config_json": _canonical(numbers), "metrics_json": json.dumps(kept_metrics, sort_keys=True),
            "primary_metric": name, "primary_value": value,
            "elapsed_seconds": round(float(elapsed_seconds), 1) if isinstance(elapsed_seconds, (int, float)) else None,
            "card": card, "failure_code": failure_code if outcome == "failed" else "",
            "title": title[:120], "body": "；".join(part for part in body_parts if part)[:800], "created_at": utc_now(),
        }
        with self.connect() as connection:
            created = self._insert(connection, values)
            if created and outcome == "success":
                self._link_fix(connection, tenant_id, owner_id, created, str(data_key), numbers)
                self._credit_used(connection, tenant_id, owner_id, list(used_ids), name, value)
        return created

    def _link_fix(self, connection: sqlite3.Connection, tenant_id: str, owner_id: str, success_id: str,
                  data_key: str, numbers: dict[str, Any]) -> None:
        """The newest unfixed failure on the same data gets "what fixed it" (this success's configuration)."""
        failure = connection.execute(
            "SELECT seq,id,title,body FROM experiences WHERE scope='private' AND tenant_id=? AND owner_id=? AND data_key=? "
            "AND outcome='failed' AND fixed_by_id IS NULL AND deleted_at IS NULL ORDER BY seq DESC LIMIT 1",
            (tenant_id, owner_id, data_key)).fetchone()
        if failure is None:
            return
        body = (failure["body"] + "；后来改成这样成功了：" + config_text(numbers))[:800]
        connection.execute("UPDATE experiences SET fixed_by_id=?, body=? WHERE seq=?", (success_id, body, failure["seq"]))
        connection.execute("UPDATE experiences SET fixed_failure_id=? WHERE id=?", (failure["id"], success_id))
        self._reindex(connection, int(failure["seq"]), failure["title"], body)

    @staticmethod
    def _credit_used(connection: sqlite3.Connection, tenant_id: str, owner_id: str, used_ids: list[str],
                     name: str, value: float | None) -> None:
        """A recalled success whose result this run matched or beat earns one success credit."""
        if value is None:
            return
        for experience_id in used_ids[:RECALL_PRIVATE]:
            if not isinstance(experience_id, str) or not EXPERIENCE_ID.fullmatch(experience_id):
                continue
            connection.execute(
                "UPDATE experiences SET success_count=success_count+1 WHERE id=? AND scope='private' AND tenant_id=? "
                "AND owner_id=? AND outcome='success' AND primary_metric=? AND primary_value IS NOT NULL "
                "AND primary_value <= ? + 1e-9 AND deleted_at IS NULL",
                (experience_id, tenant_id, owner_id, name, value))

    # ------------------------------------------------------------ private: owner views

    def list_mine(self, principal: Principal | None, *, limit: int = LIST_LIMIT) -> list[dict[str, Any]]:
        tenant_id, owner_id = self.owner(principal)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiences WHERE scope='private' AND tenant_id=? AND owner_id=? AND deleted_at IS NULL "
                "ORDER BY seq DESC LIMIT ?", (tenant_id, owner_id, max(1, min(int(limit), LIST_LIMIT)))).fetchall()
        return [self.public(row) for row in rows]

    def count_mine(self, principal: Principal | None) -> int:
        tenant_id, owner_id = self.owner(principal)
        with self.connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM experiences WHERE scope='private' AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                (tenant_id, owner_id)).fetchone()[0])

    def delete_mine(self, principal: Principal | None, experience_id: str) -> dict[str, Any]:
        """Hard to recover on purpose: the row leaves search, recall and the next nightly aggregate. Others get 404."""
        tenant_id, owner_id = self.owner(principal)
        if not isinstance(experience_id, str) or not EXPERIENCE_ID.fullmatch(experience_id):
            raise AccessError(404, "experience_not_found")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT seq,title FROM experiences WHERE id=? AND scope='private' AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                (experience_id, tenant_id, owner_id)).fetchone()
            if row is None:
                raise AccessError(404, "experience_not_found")
            connection.execute("UPDATE experiences SET deleted_at=?, body='', config_json='{}', metrics_json='{}' WHERE seq=?",
                               (utc_now(), row["seq"]))
            self._reindex(connection, int(row["seq"]), row["title"], None)
        return {"id": experience_id, "deleted": True}

    def settings(self, principal: Principal | None) -> dict[str, Any]:
        tenant_id, owner_id = self.owner(principal)
        with self.connect() as connection:
            row = connection.execute("SELECT contribute_global FROM experience_settings WHERE tenant_id=? AND owner_id=?",
                                     (tenant_id, owner_id)).fetchone()
        return {"contribute_global": True if row is None else bool(row["contribute_global"])}

    def patch_settings(self, principal: Principal | None, body: Any) -> dict[str, Any]:
        tenant_id, owner_id = self.owner(principal)
        if not isinstance(body, dict) or set(body) != {"contribute_global"} or type(body["contribute_global"]) is not bool:
            raise AccessError(400, "invalid_experience_settings")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO experience_settings(tenant_id,owner_id,contribute_global,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(tenant_id,owner_id) DO UPDATE SET contribute_global=excluded.contribute_global, updated_at=excluded.updated_at",
                (tenant_id, owner_id, int(body["contribute_global"]), utc_now()))
        return self.settings(principal)

    # ------------------------------------------------------------ search and recall

    def search(self, query: str, *, scope: str, principal: Principal | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """BM25-ranked rows of one scope. ``private`` is always filtered to the principal; ``platform`` has no owner."""
        if scope not in {"private", "platform"}:
            raise ValueError("experience_scope_invalid")
        where, params = "e.scope=? AND e.deleted_at IS NULL", [scope]
        if scope == "private":
            tenant_id, owner_id = self.owner(principal)
            where += " AND e.tenant_id=? AND e.owner_id=?"
            params += [tenant_id, owner_id]
        limit = max(1, min(int(limit), 50))
        match = fts_query(query)
        with self.connect() as connection:
            if self.fts and match:
                rows = connection.execute(
                    f"SELECT e.*, bm25(experiences_fts) AS rank FROM experiences_fts JOIN experiences e ON e.seq=experiences_fts.rowid "
                    f"WHERE experiences_fts MATCH ? AND {where} ORDER BY rank LIMIT ?", [match, *params, limit]).fetchall()
                return [dict(row) for row in rows]
            rows = connection.execute(f"SELECT e.* FROM experiences e WHERE {where} ORDER BY e.seq DESC LIMIT 500", params).fetchall()
        items = [dict(row) for row in rows]
        scores = bm25_rank(query, [item["title"] + " " + item["body"] for item in items])
        ranked = sorted(zip(scores, range(len(items))), key=lambda pair: (-pair[0], pair[1]))
        return [items[index] for score, index in ranked if score > 0][:limit]

    def recall(self, principal: Principal | None, *, task: str, profile: dict[str, str] | None,
               project_id: str = "", exclude_run_id: str = "") -> dict[str, Any]:
        """Up to 3 private + 3 global experiences for a planned training, as prompt lines within RECALL_CHAR_BUDGET.

        ``recommended`` is the trainer numbers of the best matching success (private first, then global).
        ``exclude_run_id`` drops what the current run itself wrote (a run never recalls its own result).
        """
        tenant_id, owner_id = self.owner(principal)
        profile = profile or {}
        keys = ("rows_bucket", "cols_bucket", "target_kind", "balance_bucket")

        def closeness(row: dict[str, Any]) -> int:
            return sum(1 for key in keys if profile.get(key) and row.get(key) == profile.get(key))

        with self.connect() as connection:
            own = [dict(row) for row in connection.execute(
                "SELECT * FROM experiences WHERE scope='private' AND tenant_id=? AND owner_id=? AND task=? AND deleted_at IS NULL "
                "AND COALESCE(run_id,'')<>? ORDER BY seq DESC LIMIT 300", (tenant_id, owner_id, task, str(exclude_run_id or ""))).fetchall()]
            priors = [dict(row) for row in connection.execute(
                "SELECT * FROM global_priors WHERE task=?", (task,)).fetchall()]
        own.sort(key=lambda row: (-(closeness(row) + (2 if project_id and row["project_id"] == project_id else 0)
                                   + (1 if row["outcome"] == "success" else 0) + 0.5 * min(int(row["success_count"]), 4)),
                                  -(row["primary_value"] if row["primary_value"] is not None else -1e9), -int(row["seq"])))
        own = [row for row in own if closeness(row) >= 2][:RECALL_PRIVATE]
        priors = [row for row in priors if closeness(row) >= 2 and (not profile.get("target_kind") or row["target_kind"] == profile["target_kind"])]
        priors.sort(key=lambda row: (-closeness(row), -(row["median_metric"] if row["median_metric"] is not None else -1e9), -int(row["tenants"])))
        priors = priors[:RECALL_GLOBAL]
        recommended: dict[str, Any] = {}
        for row in [*own, *priors]:
            if row.get("outcome", "success") == "success" and row.get("config_json"):
                try:
                    recommended = json.loads(row["config_json"])
                except ValueError:
                    recommended = {}
                if recommended:
                    break
        if not own and not priors:
            return {"lines": [], "private_ids": [], "prior_keys": [], "recommended": {}}
        header = ("Training experience (reference only; the person's request and the approved configuration still win. "
                  "Mention it to the person in plain Chinese only when it changes your suggestion):")
        tail = ("Suggested starting configuration (trainer fields only): " + json.dumps(recommended, sort_keys=True)) if recommended else ""
        size = len(header) + len(tail) + 2
        kept, private_ids, prior_keys = [header], [], []
        candidates = [(f"- 你以前的训练：{row['title']}；{row['body']}", "private", row["id"]) for row in own]
        candidates += [(f"- 平台匿名统计（{int(row['tenants'])} 个用户、{int(row['runs'])} 次成功）：{profile_text(row)}；"
                        f"设置：{config_text(json.loads(row['config_json']))}；中位结果：{metric_text(row['primary_metric'], row['median_metric'])}",
                        "global", row["pattern_key"]) for row in priors]
        for line, kind, key in candidates:
            if size + len(line) + 1 > RECALL_CHAR_BUDGET:
                continue
            kept.append(line)
            size += len(line) + 1
            (private_ids if kind == "private" else prior_keys).append(key)
        if tail:
            kept.append(tail)
        return {"lines": kept, "private_ids": private_ids, "prior_keys": prior_keys, "recommended": recommended}

    def mark_used(self, principal: Principal | None, experience_ids: Iterable[str]) -> None:
        tenant_id, owner_id = self.owner(principal)
        ids = [item for item in experience_ids if isinstance(item, str) and EXPERIENCE_ID.fullmatch(item)]
        if not ids:
            return
        with self.connect() as connection:
            connection.executemany("UPDATE experiences SET use_count=use_count+1 WHERE id=? AND tenant_id=? AND owner_id=?",
                                   [(item, tenant_id, owner_id) for item in ids])

    # ------------------------------------------------------------ anonymous global

    def aggregate_global(self) -> dict[str, int]:
        """Rebuild ``global_priors`` from live private successes of contributing owners (>= GLOBAL_MIN_TENANTS tenants)."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT e.tenant_id, e.task, e.rows_bucket, e.cols_bucket, e.target_kind, e.balance_bucket, e.config_json, "
                "e.primary_metric, e.primary_value, e.elapsed_seconds FROM experiences e "
                "LEFT JOIN experience_settings s ON s.tenant_id=e.tenant_id AND s.owner_id=e.owner_id "
                "WHERE e.scope='private' AND e.kind='training' AND e.outcome='success' AND e.deleted_at IS NULL "
                "AND COALESCE(s.contribute_global,1)=1").fetchall()
            groups: dict[tuple, list[sqlite3.Row]] = {}
            for row in rows:
                key = (row["task"], row["rows_bucket"], row["cols_bucket"], row["target_kind"], row["balance_bucket"],
                       _canonical(json.loads(row["config_json"] or "{}")), row["primary_metric"])
                groups.setdefault(key, []).append(row)
            connection.execute("DELETE FROM global_priors")
            kept = 0
            now = utc_now()
            for key, members in groups.items():
                tenants = len({member["tenant_id"] for member in members})
                if tenants < GLOBAL_MIN_TENANTS:
                    continue
                values = [member["primary_value"] for member in members if member["primary_value"] is not None]
                times = [member["elapsed_seconds"] for member in members if member["elapsed_seconds"] is not None]
                pattern = "pat_" + hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:24]
                connection.execute(
                    "INSERT INTO global_priors VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pattern, key[0], key[1], key[2], key[3], key[4], key[5], tenants, len(members), key[6],
                     round(statistics.median(values), 6) if values else None,
                     round(statistics.median(times), 1) if times else None, now))
                kept += 1
            connection.execute("INSERT INTO knowledge_meta(key,value) VALUES('global_aggregated_at',?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now,))
        return {"patterns": len(groups), "published": kept}

    # ------------------------------------------------------------ platform

    def import_platform(self, entries: Iterable[dict[str, Any]]) -> int:
        """Idempotent by ``source_key``. Entries: {source_key, kind, title, body}."""
        added = 0
        with self.connect() as connection:
            for entry in entries:
                body = str(entry.get("body") or "").strip()
                key = str(entry.get("source_key") or "")
                if not body or not key:
                    continue
                created = self._insert(connection, {
                    "id": new_id("exp"), "scope": "platform", "kind": str(entry.get("kind") or "research")[:40],
                    "source_key": key[:300], "title": str(entry.get("title") or "平台经验")[:120], "body": body[:1500],
                    "created_at": utc_now()})
                added += 1 if created else 0
        return added

    def platform_lines(self, query: str, *, limit: int = PLATFORM_RECALL) -> list[str]:
        rows = self.search(query, scope="platform", limit=limit)
        if not rows:
            return []
        lines = ["Platform knowledge (operator reference; verify before relying on it):"]
        size = len(lines[0])
        for row in rows:
            line = f"- {row['title']}: {row['body'][:420]}"
            if size + len(line) > RECALL_CHAR_BUDGET:
                break
            lines.append(line)
            size += len(line) + 1
        return lines if len(lines) > 1 else []

    def meta(self, key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM knowledge_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO knowledge_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                               (key, value))


def research_knowledge_entries(database: Path) -> list[dict[str, Any]]:
    """Non-rejected ``research_knowledge`` rows as platform entries (read-only open)."""
    if not Path(database).is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{Path(database).as_posix()}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        rows = connection.execute("SELECT id,category,status,payload FROM research_knowledge WHERE status!='rejected'").fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    entries = []
    for record_id, category, status, payload in rows:
        try:
            document = json.loads(payload)
        except ValueError:
            document = payload
        entries.append({"source_key": f"research_knowledge:{record_id}", "kind": "research",
                        "title": f"{category}（{status}）", "body": _flatten_text(document)})
    return entries


def retrospective_entries(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in paths:
        try:
            if Path(path).stat().st_size > 2_000_000:
                continue
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        digest = hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:24]
        title = str(document.get("title") or document.get("competition") or Path(path).parent.name) if isinstance(document, dict) else Path(path).parent.name
        entries.append({"source_key": f"retrospective:{digest}", "kind": "retrospective", "title": "复盘：" + title[:100],
                        "body": _flatten_text(document)})
    return entries


def find_retrospectives(roots: Iterable[Path], *, max_entries: int = 200_000) -> list[Path]:
    """``*retrospective_memory*.json`` under the roots, visiting at most ``max_entries`` directory entries."""
    import os

    found: list[Path] = []
    visited = 0
    for root in roots:
        if not Path(root).is_dir():
            continue
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in {".venv", "node_modules", ".git", "__pycache__"}]
            visited += len(directories) + len(files)
            for name in files:
                if name.endswith(".json") and "retrospective_memory" in name.replace("-", "_"):
                    found.append(Path(current) / name)
            if visited > max_entries:
                return found
    return found


def housekeeping(runtime_root: Path, search_roots: Iterable[Path] = ()) -> dict[str, Any]:
    """Nightly: platform import (research knowledge every night; retrospectives once) and the global aggregate."""
    store = ExperienceStore(runtime_root)
    result: dict[str, Any] = {"research": store.import_platform(research_knowledge_entries(Path(runtime_root) / "runtime.sqlite3"))}
    if store.meta("retrospectives_imported_at") is None:
        result["retrospectives"] = store.import_platform(retrospective_entries(find_retrospectives(search_roots)))
        store.set_meta("retrospectives_imported_at", utc_now())
    result.update(store.aggregate_global())
    return result
