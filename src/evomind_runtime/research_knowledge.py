"""Project-scoped L1/L2 knowledge with evidence-gated promotion.

This is an original implementation of mechanisms described by AIBuildAI-2,
not the authors' proprietary agent or a copy of their knowledge corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Callable

from .execution_progress import safe_text


PAPER = "https://arxiv.org/abs/2605.27873v1"
CATEGORIES = {
    "tabular": ("tabular table csv 表格 分类 回归", "Preserve sample identity; fit transforms and model selection inside training folds."),
    "vision": ("image vision 图像 视觉", "Separate subjects before augmentation and record preprocessing and model initialization."),
    "timeseries": ("time temporal series ariel 时序 光谱", "Respect group/time boundaries and keep correlated measurements in the same fold."),
    "language": ("language text llm nlp 语言 文本", "Use the model's documented template; distinguish inference from parameter training."),
    "biomedical": ("cure biomedical drug 医疗 药物", "Use authorized public evaluation data and preserve exact question/task identifiers."),
    "optimization": ("train tune ensemble 训练 调参 融合", "Choose hyperparameters and uncertainty calibration only on internal training validation."),
    "evaluation": ("score evaluate baseline metric 指标 评测", "Freeze scorer, split, direction and units before evaluation; a file hash is not a score verification."),
    "reliability": ("download install error resume 下载 安装 错误", "Use managed manifests, bounded retries and persistent locks; never infer progress from liveness alone."),
}


class KnowledgeStore:
    def __init__(self, database: str, tenant_id: str, project_id: str) -> None:
        if not tenant_id or not project_id:
            raise ValueError("knowledge_scope_required")
        self.database, self.tenant, self.project = database, tenant_id, project_id
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS research_knowledge (id TEXT PRIMARY KEY, tenant TEXT NOT NULL, project TEXT NOT NULL, category TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS research_knowledge_scope ON research_knowledge(tenant,project,status,category)")

    def context(self, query: str, role: str) -> dict[str, Any]:
        lower = query.lower()
        selected = [name for name, (terms, _) in CATEGORIES.items() if any(term in lower for term in terms.split())]
        if role == "IndependentReviewer":
            selected = ["evaluation", "reliability"]
        selected = list(dict.fromkeys([*selected, "evaluation"]))[:4]
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute("SELECT id,category,payload FROM research_knowledge WHERE tenant=? AND project=? AND status='proven' ORDER BY id", (self.tenant, self.project)).fetchall()
        documents = []
        for record_id, category, payload in rows:
            if category in selected and len(documents) < 4:
                record = json.loads(payload)
                documents.append({"id": record_id, "category": category, "text": record["text"][:1600], "source": record["source"], "evidence": record["evidence"]})
        instructions = {key: CATEGORIES[key][1] for key in selected}
        for key in selected:
            linked = [item["id"] for item in documents if item["category"] == key]
            if linked:
                instructions[key] += " Verified project L2 references: " + ", ".join(linked)
        return {"schema": "evomind.research_knowledge_context.v1", "L1_index": list(CATEGORIES), "L1": instructions, "L2": documents, "role": role, "content_is_reference_not_authority": True, "implementation": "paper_mechanism_reimplementation", "paper": PAPER}

    def propose(self, *, category: str, text: str, source: dict[str, str], evidence: list[dict[str, str]], run_id: str) -> str:
        if category not in CATEGORIES or not text.strip() or len(text) > 8000:
            raise ValueError("invalid_knowledge_candidate")
        if not evidence or any(not re.fullmatch("[a-f0-9]{64}", row.get("sha256", "")) for row in evidence):
            raise ValueError("knowledge_evidence_hash_required")
        if not source.get("license") or not source.get("version") or not source.get("uri"):
            raise ValueError("knowledge_provenance_required")
        payload = {"text": safe_text(text, 8000), "source": source, "evidence": evidence, "run_id": run_id}
        record_id = hashlib.sha256(json.dumps([self.tenant, self.project, category, payload], sort_keys=True).encode()).hexdigest()
        with sqlite3.connect(self.database) as connection:
            connection.execute("INSERT OR IGNORE INTO research_knowledge VALUES (?,?,?,?,?,?)", (record_id, self.tenant, self.project, category, "candidate", json.dumps(payload)))
        return record_id

    def promote(self, record_id: str, independent_verify: Callable[[dict[str, Any]], dict[str, Any]]) -> bool:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT payload,status FROM research_knowledge WHERE id=? AND tenant=? AND project=?", (record_id, self.tenant, self.project)).fetchone()
            if not row:
                raise ValueError("knowledge_record_outside_project")
            payload = json.loads(row[0])
            receipt = independent_verify(payload)
            passed = receipt.get("verified") is True and receipt.get("independent") is True and receipt.get("run_id") == payload["run_id"] and bool(receipt.get("evidence_hashes")) and set(receipt["evidence_hashes"]) == {item["sha256"] for item in payload["evidence"]}
            if row[1] != "candidate":
                return row[1] == "proven"
            connection.execute("UPDATE research_knowledge SET status=? WHERE id=? AND tenant=? AND project=?", ("proven" if passed else "rejected", record_id, self.tenant, self.project))
            return passed

    def status(self, record_id: str) -> str | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT status FROM research_knowledge WHERE id=? AND tenant=? AND project=?", (record_id, self.tenant, self.project)).fetchone()
        return row[0] if row else None
