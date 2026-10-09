"""Human-only, policy-owner budget amendments. No tool or remote execution."""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from .research_budget import GpuExecutionBudget, _execution_limits, execution_budget_summary, read_execution_budget
from .tenant_access import AccessError, Principal
from .training_control import controls, load_policy

MAX_LIMIT_SECONDS = 365 * 24 * 3600


def _policy(root: Path, principal: Principal | None) -> dict:
    if principal is None:
        raise AccessError(403, 'budget_named_principal_required')
    try:
        policy = load_policy(root)
        authorized = controls(policy, {'tenant_id': principal.tenant_id, 'owner_principal_id': principal.owner_id})
    except (ValueError, OSError):
        raise AccessError(403, 'budget_policy_owner_required') from None
    if not authorized or not policy['enabled']:
        raise AccessError(403, 'budget_policy_owner_required')
    return policy


def status(root: Path, principal: Principal | None) -> dict:
    policy = _policy(root, principal)
    database = root / 'gpu_budget.sqlite3'
    budget = read_execution_budget(database, policy['study_id'])
    history = []
    if database.exists():
        connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            connection.row_factory = sqlite3.Row
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gpu_budget_amendments'").fetchone():
                history = [dict(row) for row in connection.execute(
                    'SELECT revision,gpu_limit,engineering_limit,previous_gpu_limit,previous_engineering_limit,owner_id,reason,created FROM gpu_budget_amendments WHERE study=? AND revision<=? ORDER BY revision DESC LIMIT 20',
                    (policy['study_id'], budget['revision']))]
        finally:
            connection.close()
    return {'ok': True, 'can_manage': True, 'budget': budget, 'history': history,
            'max_limit_seconds': MAX_LIMIT_SECONDS, 'no_execution_started': True}


def amend(root: Path, principal: Principal | None, body: dict) -> dict:
    policy = _policy(root, principal)
    fields = {'gpu_limit_seconds', 'engineering_limit_seconds', 'expected_revision', 'request_id', 'reason', 'confirmed'}
    if not isinstance(body, dict) or set(body) != fields or body['confirmed'] is not True:
        raise AccessError(400, 'budget_confirmation_required')
    gpu, engineering, revision = (body[key] for key in ('gpu_limit_seconds', 'engineering_limit_seconds', 'expected_revision'))
    if (type(gpu) is not int or type(engineering) is not int or not 0 <= engineering <= gpu <= MAX_LIMIT_SECONDS
            or type(revision) is not int or revision < 0):
        raise AccessError(400, 'budget_limits_invalid')
    reason, request_id = body['reason'], body['request_id']
    if (not isinstance(reason, str) or not 3 <= len(reason.strip()) <= 500
            or not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}', request_id)):
        raise AccessError(400, 'budget_audit_fields_invalid')
    database = root / 'gpu_budget.sqlite3'
    GpuExecutionBudget(str(database), policy['study_id'])
    connection = sqlite3.connect(database, timeout=5)
    try:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE IF NOT EXISTS gpu_budget_amendments (study TEXT NOT NULL, revision INTEGER NOT NULL, request_id TEXT NOT NULL, gpu_limit INTEGER NOT NULL, engineering_limit INTEGER NOT NULL, previous_gpu_limit INTEGER NOT NULL, previous_engineering_limit INTEGER NOT NULL, tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, policy_sha256 TEXT NOT NULL, reason TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(study,revision), UNIQUE(study,request_id))')
        prior = connection.execute('SELECT gpu_limit,engineering_limit,tenant_id,owner_id,reason FROM gpu_budget_amendments WHERE study=? AND request_id=?', (policy['study_id'], request_id)).fetchone()
        replayed = prior is not None
        if prior is not None:
            if prior != (gpu, engineering, principal.tenant_id, principal.owner_id, reason.strip()):
                raise AccessError(409, 'budget_request_conflict')
        else:
            limits = _execution_limits(connection, policy['study_id'])
            if limits['revision'] != revision:
                raise AccessError(409, 'budget_revision_conflict')
            rows = connection.execute('SELECT kind,reserved,charged,status,boot_id FROM gpu_operations WHERE study=?', (policy['study_id'],)).fetchall()
            current = execution_budget_summary(rows, policy['study_id'], limits)
            if gpu < current['committed_seconds'] or engineering < current['engineering_committed_seconds']:
                raise AccessError(409, 'budget_below_committed')
            if gpu == limits['gpu_limit_seconds'] and engineering == limits['engineering_limit_seconds']:
                raise AccessError(409, 'budget_unchanged')
            connection.execute('INSERT INTO gpu_budget_amendments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (policy['study_id'], revision + 1, request_id, gpu, engineering, limits['gpu_limit_seconds'], limits['engineering_limit_seconds'], principal.tenant_id, principal.owner_id, policy['policy_sha256'], reason.strip(), time.time()))
        connection.commit()
    finally:
        connection.close()
    return {**status(root, principal), 'replayed': replayed}
