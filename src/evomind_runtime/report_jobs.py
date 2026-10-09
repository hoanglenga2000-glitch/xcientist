"""Durable, owner-scoped report jobs; no provider calls or GPU work."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .report_document import build_document, canonical, digest, safe_text
from .report_render import render_document


class ReportJobs:
    def __init__(self, runtime):
        self.runtime = runtime
        self.database = runtime.runtime_root / "report-jobs.sqlite3"
        self.lock = threading.RLock()
        self.workers = {}
        self.stopping = False
        self.boot = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,source_json TEXT NOT NULL,status TEXT NOT NULL,payload_json TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.database), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _claim(self, row, fields):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT status,updated_at,payload_json FROM jobs WHERE id=?", (row["id"],)).fetchone()
            if current is None or current["status"] != row["status"] or current["updated_at"] != row["updated_at"]:
                return False
            payload = json.loads(current["payload_json"])
            attempts = list(payload.get("attempts", []))
            if len(attempts) >= 16:
                raise ValueError("report_retry_limit")
            attempts.append({"started_at": datetime.now(timezone.utc).isoformat(), "status": "running"})
            payload.update(fields)
            payload["attempts"] = attempts
            connection.execute("UPDATE jobs SET status='running',payload_json=?,updated_at=? WHERE id=?", (
                canonical(payload), datetime.now(timezone.utc).isoformat(), row["id"]))
            return True

    @staticmethod
    def renderer_identity():
        from . import report_document, report_figures, report_render
        return {Path(module.__file__).name: digest(Path(module.__file__)) for module in (report_document, report_figures, report_render)}

    def _row(self, identifier):
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None: raise KeyError(identifier)
        return dict(row)

    def _update(self, identifier, status, **fields):
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload_json FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None: raise KeyError(identifier)
            payload = json.loads(row[0])
            payload.update(fields)
            if status in {"ready", "partial", "failed"} and payload.get("attempts"):
                payload["attempts"][-1].update(status=status, completed_at=now, error_code=fields.get("error_code", ""))
            connection.execute("UPDATE jobs SET status=?,payload_json=?,updated_at=? WHERE id=?", (status, canonical(payload), now, identifier))

    def get(self, run_id, identifier, *, include_document=True):
        from .tenant_access import AccessStore, current_principal
        principal = current_principal.get()
        if principal is not None: AccessStore(self.runtime.runtime_root).require_run(self.runtime, run_id, principal)
        self.runtime.get_session(run_id)
        row = self._row(identifier)
        if row["run_id"] != run_id: raise KeyError(identifier)
        payload = json.loads(row["payload_json"])
        document = json.loads(row["source_json"])
        result = {"schema": "evomind.report_job.v1", "id": row["id"], "run_id": run_id,
            "status": row["status"], "stage": payload.get("stage", "queued"),
            "report_status": payload.get("report_status", document["report_status"]),
            "execution_status": document["execution_status"], "evidence_status": document["evidence_status"],
            "document_sha256": hashlib.sha256(row["source_json"].encode()).hexdigest(),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "artifacts": payload.get("artifacts", []), "error_code": payload.get("error_code", ""),
            "source_artifact_ids": [source["id"] for source in document.get("sources", [])],
            "source_count": len(document.get("sources", [])),
            "missing": [safe_text(item, 2000) for item in document.get("missing", [])],
            "manifest": payload.get("manifest"), "attempts": payload.get("attempts", [])}
        if include_document: result["document"] = document
        return result

    def list(self, run_id):
        from .tenant_access import AccessStore, current_principal
        principal = current_principal.get()
        if principal is not None: AccessStore(self.runtime.runtime_root).require_run(self.runtime, run_id, principal)
        self.runtime.get_session(run_id)
        with self.connect() as connection:
            identifiers = [row[0] for row in connection.execute("SELECT id FROM jobs WHERE run_id=? ORDER BY created_at DESC LIMIT 50", (run_id,))]
        return [self.get(run_id, identifier, include_document=False) for identifier in identifiers]

    def enqueue(self, run_id, args):
        from .tenant_access import AccessStore, current_principal
        principal = current_principal.get()
        if principal is not None: AccessStore(self.runtime.runtime_root).require_run(self.runtime, run_id, principal)
        document = build_document(self.runtime, run_id, args)
        document["renderer_identity"] = self.renderer_identity()
        source = canonical(document)
        identifier = "report_" + hashlib.sha256(source.encode()).hexdigest()[:32]
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,?,?)", (identifier, run_id, source, "queued", "{}", now, now))
        self.start(identifier)
        return self.get(run_id, identifier, include_document=False)

    def resume(self, run_id, identifier):
        self.get(run_id, identifier, include_document=False)
        self.start(identifier)
        return self.get(run_id, identifier, include_document=False)

    def start(self, identifier):
        with self.lock:
            if self.stopping or identifier in self.workers:
                return
            row = self._row(identifier)
            if row["status"] in {"ready", "partial"}:
                return
            payload = json.loads(row["payload_json"])
            if row["status"] == "running":
                import psutil
                pid = payload.get("worker_pid")
                if pid and psutil.pid_exists(pid):
                    try:
                        if abs(psutil.Process(pid).create_time() - float(payload.get("worker_started", 0))) < 1:
                            return  # A different service instance may still own it.
                    except (psutil.Error, ValueError):
                        return
            if self.runtime.user_pause_requested(row["run_id"]):
                self._update(identifier, "paused", stage="user_paused")
                return
            source = json.loads(row["source_json"])
            if source.get("renderer_identity") != self.renderer_identity():
                self._update(identifier, "failed", error_code="renderer_changed_requires_new_version")
                return
            import psutil
            if not self._claim(row, {"stage": "collecting_evidence", "worker_pid": os.getpid(),
                         "worker_started": psutil.Process().create_time(), "boot_id": self.boot, "error_code": ""}):
                return
            context = contextvars.copy_context()
            thread = threading.Thread(target=lambda: context.run(self._work, identifier), name="report-"+identifier[-12:], daemon=True)
            self.workers[identifier] = thread
            thread.start()

    def _work(self, identifier):
        try:
            row = self._row(identifier)
            document = json.loads(row["source_json"])
            session = self.runtime.get_session(row["run_id"])
            root = Path(session["workspace_root"]).resolve()
            payload = json.loads(row["payload_json"])
            output = root / "outputs" / "reports" / identifier
            output.resolve().relative_to(root)
            manifest_path = output / "report-manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest["files"]:
                    path = output / item["path"]
                    path.resolve().relative_to(output.resolve())
                    if not path.is_file() or digest(path) != item["sha256"]:
                        raise ValueError("report_output_integrity_failed")
            else:
                if output.exists():
                    # Preserve the failed attempt. A retry gets a new physical
                    # output directory, not an overwrite of partial evidence.
                    output = root / "outputs" / "reports" / (identifier + "-" + uuid.uuid4().hex[:8])
                self._update(identifier, "running", stage="rendering_formats")
                manifest = render_document(document, output)
            self._update(identifier, "running", stage="publishing", report_status=manifest["report_status"])
            from .assistant_runs import public_artifact
            artifacts = list(payload.get("artifacts", []))
            existing = {(item["name"], item["sha256"]) for item in artifacts}
            for path in sorted(output.rglob("*")):
                if not path.is_file(): continue
                if path.is_symlink(): raise ValueError("report_output_symlink")
                signature = (path.name, digest(path))
                if signature in existing: continue
                published = self.runtime.assistant.publish_path(row["run_id"], path, source_tool_call="report_generate")
                artifacts.append(public_artifact(published))
                existing.add(signature)
                self._update(identifier, "running", stage="publishing", artifacts=artifacts)
            state = "ready" if manifest["report_status"] == "ready" else "partial"
            self._update(identifier, state, stage="finished", artifacts=artifacts, manifest=manifest, report_status=manifest["report_status"])
            self.runtime.store.append_event(row["run_id"], "report.generated", {"report_id": identifier, "report_status": state,
                "document_sha256": manifest["document_sha256"], "artifact_ids": [item["id"] for item in artifacts]})
        except Exception as error:
            code = str(error) if isinstance(error, ValueError) and __import__("re").fullmatch(r"[a-z_]{1,100}", str(error)) else type(error).__name__
            self._update(identifier, "failed", stage="failed", error_code=code)
        finally:
            with self.lock:
                self.workers.pop(identifier, None)

    def shutdown(self, timeout=2.0):
        with self.lock:
            self.stopping = True
            workers = list(self.workers.values())
        deadline = time.monotonic() + timeout
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in workers)
