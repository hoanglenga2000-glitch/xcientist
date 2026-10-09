"""Owned task drafts and explicit links; never infer or migrate historical owners."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .models import new_id, utc_now
from .tenant_access import AccessError, Principal

TASK_ID = re.compile(r'utask_[a-f0-9]{32}')
KEY = re.compile(r'[A-Za-z0-9_-]{8,128}')


class UserTasks:
    def __init__(self, root: Path):
        self.path = Path(root) / 'user_tasks.sqlite3'
        with self.connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS user_tasks (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    title TEXT NOT NULL, draft TEXT NOT NULL, version INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    key_hash TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    UNIQUE(tenant,owner,key_hash)
                );
                CREATE TABLE IF NOT EXISTS user_task_runs (
                    run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    tenant TEXT NOT NULL, owner TEXT NOT NULL, linked_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS user_task_runs_task ON user_task_runs(task_id,tenant,owner);
                CREATE TABLE IF NOT EXISTS user_task_files (
                    task_id TEXT NOT NULL, file_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY(task_id,file_id)
                );
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def scope(principal: Principal | None):
        if principal is None:
            raise AccessError(403, 'task_principal_required')
        return principal.tenant_id, principal.owner_id

    @staticmethod
    def public(row):
        return {**{key: row[key] for key in ('id', 'title', 'draft', 'version', 'conversation_id', 'created_at', 'updated_at')}, 'status': 'draft'}

    @staticmethod
    def validate_text(title, draft):
        if (not isinstance(title, str) or not 1 <= len(title.strip()) <= 120
                or re.search(r'[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]', title)
                or not isinstance(draft, str) or len(draft) > 20000 or '\x00' in draft):
            raise AccessError(400, 'invalid_task_request')
        return title.strip(), draft

    def get(self, principal, task_id):
        scope = self.scope(principal)
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise AccessError(404, 'task_not_found')
        with self.connect() as connection:
            row = connection.execute('SELECT * FROM user_tasks WHERE id=? AND tenant=? AND owner=?', (task_id, *scope)).fetchone()
            if row is None:
                raise AccessError(404, 'task_not_found')
            return self.public(row)

    def list(self, principal, runtime=None, access=None):
        with self.connect() as connection:
            tasks = [self.public(row) for row in connection.execute(
                'SELECT * FROM user_tasks WHERE tenant=? AND owner=? ORDER BY updated_at DESC,id DESC LIMIT 1000', self.scope(principal))]
        if runtime is not None:
            for task in tasks:
                runs = self.runs(principal, task['id'], runtime, access)
                if runs:
                    task['status'] = runs[0]['status']
        return tasks

    def create(self, principal, body):
        scope = self.scope(principal)
        if set(body) - {'title', 'draft', 'idempotency_key'}:
            raise AccessError(400, 'invalid_task_request')
        title, draft = self.validate_text(body.get('title'), body.get('draft', ''))
        key = body.get('idempotency_key')
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise AccessError(400, 'invalid_idempotency_key')
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps([title, draft], ensure_ascii=False).encode()).hexdigest()
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT * FROM user_tasks WHERE tenant=? AND owner=? AND key_hash=?', (*scope, key_hash)).fetchone()
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise AccessError(409, 'task_idempotency_conflict')
                return {'task': self.public(previous), 'replayed': True}
            if connection.execute('SELECT count(*) FROM user_tasks WHERE tenant=? AND owner=?', scope).fetchone()[0] >= 1000:
                raise AccessError(409, 'task_limit_reached')
            task_id, conversation, now = new_id('utask'), new_id('conversation'), utc_now()
            connection.execute('INSERT INTO user_tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                               (task_id, *scope, title, draft, 1, conversation, now, now, key_hash, fingerprint))
        return {'task': self.get(principal, task_id), 'replayed': False}

    def runs(self, principal, task_id, runtime, access):
        with self.connect() as connection:
            ids = [row[0] for row in connection.execute(
                'SELECT run_id FROM user_task_runs WHERE task_id=? AND tenant=? AND owner=?', (task_id, *self.scope(principal)))]
        rows = []
        for run_id in ids:
            if access.owns_run(runtime, run_id, principal):
                run = runtime.store.get_assistant_run(run_id)
                rows.append({key: run.get(key) for key in ('id', 'conversation_id', 'status', 'created_at', 'updated_at', 'model', 'model_provider', 'error_class')})
        return sorted(rows, key=lambda row: (row['created_at'], row['id']), reverse=True)

    def detail(self, principal, task_id, runtime, access):
        task = self.get(principal, task_id)
        runs = self.runs(principal, task_id, runtime, access)
        if runs:
            task['status'] = runs[0]['status']
        return {'task': task, 'runs': runs, 'files': self.files(principal, task_id, runtime, access)}

    def files(self, principal, task_id, runtime, access):
        from .user_files import metadata
        self.get(principal, task_id)
        with self.connect() as connection:
            rows = connection.execute('SELECT file_id,sha256 FROM user_task_files WHERE task_id=? ORDER BY file_id', (task_id,)).fetchall()
        files = []
        for row in rows:
            item = metadata(runtime, access, principal, row['file_id'])
            if item['sha256'] != row['sha256']:
                raise AccessError(409, 'task_file_version_changed')
            files.append(item)
        return files

    def select_files(self, principal, task_id, body, runtime, access):
        from .user_files import metadata
        self.get(principal, task_id)
        ids = body.get('attachment_ids')
        if (set(body) != {'attachment_ids', 'version'} or type(body.get('version')) is not int
                or not isinstance(ids, list) or len(ids) > 50 or any(not isinstance(value, str) for value in ids)
                or len(ids) != len(set(ids))):
            raise AccessError(400, 'invalid_task_files')
        files = [metadata(runtime, access, principal, file_id) for file_id in ids]
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            change = connection.execute('UPDATE user_tasks SET version=version+1,updated_at=? WHERE id=? AND tenant=? AND owner=? AND version=?',
                (utc_now(), task_id, *self.scope(principal), body['version']))
            if change.rowcount != 1:
                raise AccessError(409, 'task_version_conflict')
            connection.execute('DELETE FROM user_task_files WHERE task_id=?', (task_id,))
            connection.executemany('INSERT INTO user_task_files VALUES(?,?,?)', [(task_id, item['id'], item['sha256']) for item in files])
        return self.detail(principal, task_id, runtime, access)

    def link_run(self, principal, task_id, run_id, runtime, access):
        self.get(principal, task_id)
        access.require_run(runtime, run_id, principal)
        scope = self.scope(principal)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('INSERT OR IGNORE INTO user_task_runs VALUES(?,?,?,?,?)', (run_id, task_id, *scope, utc_now()))
            linked = connection.execute('SELECT task_id,tenant,owner FROM user_task_runs WHERE run_id=?', (run_id,)).fetchone()
            if tuple(linked) != (task_id, *scope):
                raise AccessError(409, 'run_task_link_is_immutable')
        return self.detail(principal, task_id, runtime, access)

    def update_draft(self, principal, task_id, body):
        self.get(principal, task_id)
        if set(body) != {'title', 'draft', 'version'} or type(body.get('version')) is not int:
            raise AccessError(400, 'invalid_task_request')
        title, draft = self.validate_text(body['title'], body['draft'])
        with self.connect() as connection:
            updated = connection.execute(
                'UPDATE user_tasks SET title=?,draft=?,version=version+1,updated_at=? WHERE id=? AND tenant=? AND owner=? AND version=?',
                (title, draft, utc_now(), task_id, *self.scope(principal), body['version']))
            if updated.rowcount != 1:
                raise AccessError(409, 'task_version_conflict')
        return {'task': self.get(principal, task_id)}
