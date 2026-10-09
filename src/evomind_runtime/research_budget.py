"""Durable reservation/charging for the authorized 12 GPU-hour study."""
from __future__ import annotations

import sqlite3
import math
import os
import time
import uuid
from pathlib import Path


ARMS = {"direct_tool_loop", "aibuildai2_mechanisms", "evomind_enhanced"}
GPU_LIMIT_SECONDS = 12 * 3600
ENGINEERING_LIMIT_SECONDS = 8 * 3600
TRIAL_LIMIT_SECONDS = 20 * 60
TRIAL_COUNT = 27
_BOOT_ID = uuid.uuid4().hex


def execution_budget_summary(rows: list, study: str, limits: dict | None = None) -> dict:
    limits = limits or {'gpu_limit_seconds': GPU_LIMIT_SECONDS, 'engineering_limit_seconds': ENGINEERING_LIMIT_SECONDS, 'revision': 0}
    committed = sum(reserved if status == 'reserved' else charged for _, reserved, charged, status, _ in rows)
    engineering = sum(reserved if status == 'reserved' else charged for kind, reserved, charged, status, _ in rows if kind == 'engineering')
    reconciliation = any(status in {'uncertain', 'exceeded'} or (status == 'reserved' and boot != _BOOT_ID) for _, _, _, status, boot in rows)
    return {
        'schema': 'evomind.gpu_execution_budget_status.v1', 'scope': 'general_hpc_execution',
        'study': study, 'operations': len(rows),
        'charged_wall_seconds': sum(row[2] for row in rows),
        'committed_seconds': committed, 'engineering_committed_seconds': engineering,
        'pending_reserved_seconds': sum(row[1] for row in rows if row[3] == 'reserved'),
        'failed_operations': sum(row[3] in {'failed', 'uncertain', 'exceeded'} for row in rows),
        **limits,
        'gpu_remaining_seconds': max(0, limits['gpu_limit_seconds'] - committed),
        'engineering_remaining_seconds': max(0, limits['engineering_limit_seconds'] - engineering),
        'reconciliation_required': reconciliation,
        'rebind_will_restore_budget': False,
        'ledger_reset_allowed': False,
        'study_switch_to_bypass_allowed': False,
        'budget_amendment_requires_explicit_authorization': True,
        'siim_calibration_budget_is_separate': True,
    }


def _execution_limits(connection: sqlite3.Connection, study: str) -> dict:
    default = {'gpu_limit_seconds': GPU_LIMIT_SECONDS, 'engineering_limit_seconds': ENGINEERING_LIMIT_SECONDS, 'revision': 0}
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gpu_budget_amendments'").fetchone():
        return default
    row = connection.execute('SELECT gpu_limit,engineering_limit,revision FROM gpu_budget_amendments WHERE study=? ORDER BY revision DESC LIMIT 1', (study,)).fetchone()
    return dict(zip(default, row)) if row else default


def read_execution_budget(database: Path, study: str) -> dict:
    """Inspect a study without creating a database, reservation or remote call."""
    if not database.exists():
        return execution_budget_summary([], study)
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        rows = connection.execute('SELECT kind,reserved,charged,status,boot_id FROM gpu_operations WHERE study=?', (study,)).fetchall()
        return execution_budget_summary(rows, study, _execution_limits(connection, study))
    finally:
        connection.close()


class GpuExecutionBudget:
    """Conservative wall-time reservations shared by all controlled HPC calls."""
    def __init__(self, database: str, study: str):
        self.database, self.study = database, study
        connection = sqlite3.connect(database, timeout=5)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS gpu_operations (id TEXT PRIMARY KEY, study TEXT NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL, reserved REAL NOT NULL, charged REAL NOT NULL, status TEXT NOT NULL, boot_id TEXT NOT NULL, controller_pid INTEGER NOT NULL, created REAL NOT NULL)")
            connection.commit()
        finally:
            connection.close()

    def reserve(self, run_id: str, seconds: int, *, kind: str = "engineering") -> dict:
        if type(seconds) is not int or not 30 <= seconds <= TRIAL_LIMIT_SECONDS or kind not in {"engineering", "trial"}:
            raise ValueError("gpu_execution_budget_request_invalid")
        connection = sqlite3.connect(self.database, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT kind,reserved,charged,status,boot_id FROM gpu_operations WHERE study=?", (self.study,)).fetchall()
            if any(status in {"uncertain", "exceeded"} or (status == "reserved" and boot != _BOOT_ID) for _, _, _, status, boot in rows):
                raise RuntimeError("gpu_execution_reconciliation_required")
            committed = sum(reserved if status == "reserved" else charged for _, reserved, charged, status, _ in rows)
            engineering = sum(reserved if status == "reserved" else charged for category, reserved, charged, status, _ in rows if category == "engineering")
            limits = _execution_limits(connection, self.study)
            if committed + seconds > limits['gpu_limit_seconds'] or (kind == "engineering" and engineering + seconds > limits['engineering_limit_seconds']):
                raise RuntimeError("gpu_execution_budget_exhausted")
            operation = uuid.uuid4().hex
            connection.execute("INSERT INTO gpu_operations VALUES(?,?,?,?,?,0,'reserved',?,?,?)", (operation, self.study, run_id, kind, seconds, _BOOT_ID, os.getpid(), time.time()))
            connection.commit()
            return {"id": operation, "study": self.study, "reserved_seconds": seconds, "kind": kind, **limits}
        finally:
            connection.close()

    def settle(self, operation: str, seconds: float, *, uncertain: bool = False, success: bool = False) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("gpu_execution_charge_invalid")
        connection = sqlite3.connect(self.database, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT reserved,status FROM gpu_operations WHERE id=? AND study=?", (operation, self.study)).fetchone()
            if row is None or row[1] != "reserved":
                raise ValueError("gpu_operation_settlement_invalid")
            status = "uncertain" if uncertain else "exceeded" if seconds > row[0] else "passed" if success else "failed"
            charged = max(seconds, row[0]) if uncertain else seconds
            connection.execute("UPDATE gpu_operations SET charged=?,status=? WHERE id=?", (charged, status, operation))
            connection.commit()
        finally:
            connection.close()

    def summary(self) -> dict:
        return read_execution_budget(Path(self.database), self.study)


class ResearchBudget:
    def __init__(self, database: str, study: str) -> None:
        self.database, self.study = database, study
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS research_trials (study TEXT NOT NULL, trial TEXT NOT NULL, arm TEXT NOT NULL, protocol_sha TEXT NOT NULL, reserved INTEGER NOT NULL, charged REAL NOT NULL, status TEXT NOT NULL, PRIMARY KEY(study,trial))")

    def reserve(self, trial: str, arm: str, protocol_sha: str, seconds: int = TRIAL_LIMIT_SECONDS, *, independent_data: bool) -> bool:
        if arm not in ARMS or not independent_data or len(protocol_sha) != 64 or not 0 < seconds <= TRIAL_LIMIT_SECONDS:
            raise ValueError("study_protocol_or_budget_gate")
        with sqlite3.connect(self.database, timeout=15) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT arm,protocol_sha,reserved FROM research_trials WHERE study=? AND trial=?", (self.study, trial)).fetchone()
            if previous:
                if previous != (arm, protocol_sha, seconds):
                    raise ValueError("trial_idempotency_conflict")
                return False
            count, committed = connection.execute("SELECT COUNT(*),COALESCE(SUM(MAX(reserved,charged)),0) FROM research_trials WHERE study=?", (self.study,)).fetchone()
            if count >= TRIAL_COUNT or committed + seconds > GPU_LIMIT_SECONDS:
                raise ValueError("study_gpu_budget_exhausted")
            connection.execute("INSERT INTO research_trials VALUES (?,?,?,?,?,0,'reserved')", (self.study, trial, arm, protocol_sha, seconds))
        return True

    def settle(self, trial: str, actual_seconds: float, success: bool) -> None:
        if actual_seconds < 0:
            raise ValueError("negative_gpu_time")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT charged,status FROM research_trials WHERE study=? AND trial=?", (self.study, trial)).fetchone()
            if row is None:
                raise ValueError("unknown_trial")
            if row[1] != "reserved":
                if row != (actual_seconds, "passed" if success else "failed"):
                    raise ValueError("settled_trial_is_immutable")
                return
            connection.execute("UPDATE research_trials SET charged=?,status=? WHERE study=? AND trial=?", (actual_seconds, "passed" if success else "failed", self.study, trial))

    def summary(self) -> dict:
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute("SELECT status,COUNT(*),SUM(charged) FROM research_trials WHERE study=? GROUP BY status", (self.study,)).fetchall()
        counts = {status: count for status, count, _seconds in rows}
        return {"trials": sum(counts.values()), "passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "gpu_seconds": sum(seconds or 0 for _status, _count, seconds in rows), "gpu_limit_seconds": GPU_LIMIT_SECONDS, "failed_trials_in_denominator": True}
