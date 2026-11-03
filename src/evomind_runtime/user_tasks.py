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
# Additive C1b columns; NULL keeps every pre-existing row valid (never dropped or rewritten).
META_COLUMNS = ('pinned_at', 'archived_at', 'deleted_at', 'project_id')
SEARCH_MAX = 200
TASK_LIMIT = 1000  # live (not soft-deleted) conversations per owner
ACTIVE_WORK = {'queued', 'planning', 'running', 'verifying', 'recovering', 'pausing', 'waiting_approval'}


def delivery_status(runs):
    """Work-item label. Newest-run success must not hide an undelivered goal.

    ``delivered`` only when a goal-bearing run actually published a file, or when
    the task never asked for a file and some run completed. A failed or empty
    goal stays ``undelivered`` even if a later plain question succeeded.
    With no file goal, failed or cancelled runs are ``undelivered`` as well.
    An in-progress run keeps its own execution status.
    """
    if not runs:
        return 'draft'
    active = [run for run in runs if str(run.get('status') or '') in ACTIVE_WORK]
    if active:
        return str(active[0].get('status') or 'running')
    goals = [run for run in runs if bool(run.get('require_artifact'))]
    if goals:
        if any(str(run.get('status') or '') == 'completed' and int(run.get('artifact_count') or 0) > 0
               and run.get('training_outcome') != 'failed' for run in goals):
            return 'delivered'
        return 'undelivered'
    if any(str(run.get('status') or '') == 'completed' and run.get('training_outcome') != 'failed' for run in runs):
        return 'delivered'
    return 'undelivered'


def _training_outcome(runtime, run):
    try:
        return runtime.assistant._training_outcome(str(run.get('session_id') or run.get('id')))
    except Exception:  # noqa: BLE001 - presentation hint only
        return ''


def _contains(text, needle):
    return needle in ' '.join(text.split()).casefold()


def search_snippet(text, needle, width=60):
    """One plain line around the first match; the client bolds the term."""
    line = ' '.join(re.sub(r'[#*_`>|\[\]]+', ' ', text).split())
    at = line.casefold().find(needle)
    start = max(0, at - width // 3) if at >= 0 else 0
    piece = line[start:start + width]
    return ('…' if start else '') + piece + ('…' if start + width < len(line) else '')


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
            present = {row[1] for row in connection.execute('PRAGMA table_info(user_tasks)')}
            for column in META_COLUMNS:
                if column not in present:
                    connection.execute(f'ALTER TABLE user_tasks ADD COLUMN {column} TEXT')

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
        return {**{key: row[key] for key in ('id', 'title', 'draft', 'version', 'conversation_id', 'created_at', 'updated_at',
                                             'pinned_at', 'archived_at', 'project_id')},
                'archived': row['archived_at'] is not None, 'status': 'draft'}

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
            row = connection.execute('SELECT * FROM user_tasks WHERE id=? AND tenant=? AND owner=? AND deleted_at IS NULL',
                                     (task_id, *scope)).fetchone()
            if row is None:
                raise AccessError(404, 'task_not_found')
            return self.public(row)

    def list(self, principal, runtime=None, access=None, archived=False, query=''):
        """Default: live conversations. ``archived``: only archived. ``query``: both, with a snippet. Deleted: never."""
        if not isinstance(query, str) or len(query) > SEARCH_MAX:
            raise AccessError(400, 'invalid_task_query')
        needle = ' '.join(query.split()).casefold()
        where = '' if needle else ' AND archived_at IS NOT NULL' if archived else ' AND archived_at IS NULL'
        with self.connect() as connection:
            tasks = [self.public(row) for row in connection.execute(
                'SELECT * FROM user_tasks WHERE tenant=? AND owner=? AND deleted_at IS NULL' + where +
                ' ORDER BY updated_at DESC,id DESC LIMIT 1000', self.scope(principal))]
        found = []
        for task in tasks:
            runs = self.runs(principal, task['id'], runtime, access) if runtime is not None else []
            if runs:
                task['status'] = delivery_status(runs)
                # C6e G14: the bell dedupes per run + terminal kind; ids only, never prompts or answers.
                task['latest_run'] = {key: runs[0].get(key) for key in ('id', 'status', 'updated_at', 'plan_pending')}
            if not needle:
                found.append(task)
                continue
            bodies = [task['draft'], *(str((runtime.store.get_assistant_run(run['id']) or {}).get('prompt') or '') for run in runs)]
            body = next((text for text in bodies if _contains(text, needle)), None)
            if body is not None or _contains(task['title'], needle):
                found.append({**task, 'snippet': search_snippet(body or task['draft'], needle)})
        return found

    def hidden_run_ids(self, principal):
        """Runs of soft-deleted conversations: filtered out of every run list at read time, never modified."""
        with self.connect() as connection:
            return {row[0] for row in connection.execute(
                'SELECT r.run_id FROM user_task_runs r JOIN user_tasks t ON t.id=r.task_id AND t.tenant=r.tenant AND t.owner=r.owner '
                'WHERE r.tenant=? AND r.owner=? AND t.deleted_at IS NOT NULL', self.scope(principal))}

    def patch(self, principal, task_id, body, projects):
        self.get(principal, task_id)
        fields = set(body) - {'version'}
        if (not fields or fields - {'title', 'pinned', 'archived', 'project_id'} or type(body.get('version')) is not int
                or any(type(body[key]) is not bool for key in fields & {'pinned', 'archived'})
                or ('project_id' in body and body['project_id'] is not None and not isinstance(body['project_id'], str))):
            raise AccessError(400, 'invalid_task_request')
        if body.get('project_id') is not None:
            projects.get_owned(principal, body['project_id'])
        now, sets, values = utc_now(), [], []
        if 'title' in body:
            sets += ['title=?', 'updated_at=?']
            values += [self.validate_text(body['title'], '')[0], now]
        for key, column in (('pinned', 'pinned_at'), ('archived', 'archived_at')):
            if key in body:
                # Keep the original time on a repeated true; recency (updated_at) is untouched by metadata.
                sets.append(f'{column}=COALESCE({column},?)' if body[key] else f'{column}=NULL')
                values += [now] if body[key] else []
        if 'project_id' in body:
            sets.append('project_id=?')
            values.append(body['project_id'])
        with self.connect() as connection:
            updated = connection.execute(
                f'UPDATE user_tasks SET {",".join(sets)},version=version+1 WHERE id=? AND tenant=? AND owner=? AND version=? AND deleted_at IS NULL',
                (*values, task_id, *self.scope(principal), body['version']))
            if updated.rowcount != 1:
                raise AccessError(409, 'task_version_conflict')
        return {'task': self.get(principal, task_id)}

    def project_members(self, principal, project_id):
        """Every conversation in the project, soft-deleted ones too, so a project restore misses none."""
        with self.connect() as connection:
            return [row[0] for row in connection.execute(
                'SELECT id FROM user_tasks WHERE tenant=? AND owner=? AND project_id=? ORDER BY id', (*self.scope(principal), project_id))]

    def project_counts(self, principal):
        with self.connect() as connection:
            return dict(connection.execute(
                'SELECT project_id,count(*) FROM user_tasks WHERE tenant=? AND owner=? AND deleted_at IS NULL AND project_id IS NOT NULL '
                'GROUP BY project_id', self.scope(principal)).fetchall())

    def set_project(self, principal, task_ids, expected, project_id):
        """Move only tasks still in ``expected`` (None = no project); returns how many moved. Recency is untouched."""
        with self.connect() as connection:
            return sum(connection.execute(
                'UPDATE user_tasks SET project_id=?,version=version+1 WHERE id=? AND tenant=? AND owner=? AND project_id IS ?',
                (project_id, task_id, *self.scope(principal), expected)).rowcount for task_id in task_ids)

    def _set_deleted(self, principal, task_id, deleted):
        """Soft delete / restore. Idempotent; touches only ``deleted_at``. Runs, files and artifacts stay."""
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise AccessError(404, 'task_not_found')
        with self.connect() as connection:
            changed = connection.execute(
                'UPDATE user_tasks SET deleted_at=' + ('COALESCE(deleted_at,?)' if deleted else 'NULL') + ' WHERE id=? AND tenant=? AND owner=?',
                ((utc_now(),) if deleted else ()) + (task_id, *self.scope(principal)))
            if changed.rowcount != 1:
                raise AccessError(404, 'task_not_found')

    def delete(self, principal, task_id):
        self._set_deleted(principal, task_id, True)
        return {'id': task_id, 'deleted': True}

    def restore(self, principal, task_id):
        self._set_deleted(principal, task_id, False)
        return {'task': self.get(principal, task_id)}

    def export_markdown(self, principal, task_id, runtime, access):
        """(filename, text). Reads only; the turns, attachment and artifact names are redacted in task_export."""
        from .task_export import export_filename, render_markdown, today_utc
        task = self.get(principal, task_id)
        turns, artifacts = [], []
        for run in reversed(self.runs(principal, task_id, runtime, access)):
            snapshot = runtime.assistant.snapshot(run['id'])
            messages = (snapshot.get('message_history') or {}).get('messages') or []
            turns += [(item['role'], str(item.get('content') or '')) for item in messages if item.get('role') in {'user', 'assistant'}]
            artifacts += [str(item.get('name') or '') for item in runtime.store.list_deliverables(run['id'])]
        if task['draft'].strip():
            turns.append(('user', task['draft']))
        attachments = [item['name'] for item in self.files(principal, task_id, runtime, access)]
        today = today_utc()
        return export_filename(task['title'], today), render_markdown(task['title'], turns, attachments, artifacts, today)

    def create(self, principal, body):
        scope = self.scope(principal)
        if set(body) - {'title', 'draft', 'idempotency_key', 'project_id', 'intent'}:
            raise AccessError(400, 'invalid_task_request')
        from .experience_entry import valid_intent
        valid_intent(body.get('intent'))  # t237 C3.6: accepted for the 经验库 draft; the run carries the intent
        title, draft = self.validate_text(body.get('title'), body.get('draft', ''))
        project_id = body.get('project_id')
        if project_id is not None and not isinstance(project_id, str):
            raise AccessError(400, 'invalid_task_request')
        key = body.get('idempotency_key')
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise AccessError(400, 'invalid_idempotency_key')
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        # Legacy fingerprint is unchanged without a project.
        fingerprint = hashlib.sha256(json.dumps([title, draft, *([project_id] if project_id else [])], ensure_ascii=False).encode()).hexdigest()
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT * FROM user_tasks WHERE tenant=? AND owner=? AND key_hash=?', (*scope, key_hash)).fetchone()
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise AccessError(409, 'task_idempotency_conflict')
                return {'task': self.public(previous), 'replayed': True}
            if project_id:
                from .user_projects import ProjectStore
                ProjectStore(self.path.parent).get_owned(principal, project_id)
            if connection.execute('SELECT count(*) FROM user_tasks WHERE tenant=? AND owner=? AND deleted_at IS NULL', scope).fetchone()[0] >= TASK_LIMIT:
                raise AccessError(409, 'task_limit_reached')
            task_id, conversation, now = new_id('utask'), new_id('conversation'), utc_now()
            connection.execute('INSERT INTO user_tasks(id,tenant,owner,title,draft,version,conversation_id,created_at,updated_at,key_hash,fingerprint,project_id) '
                               'VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                               (task_id, *scope, title, draft, 1, conversation, now, now, key_hash, fingerprint, project_id or None))
        return {'task': self.get(principal, task_id), 'replayed': False}

    def runs(self, principal, task_id, runtime, access):
        from .experience_entry import run_plan_pending
        with self.connect() as connection:
            ids = [row[0] for row in connection.execute(
                'SELECT run_id FROM user_task_runs WHERE task_id=? AND tenant=? AND owner=?', (task_id, *self.scope(principal)))]
        rows = []
        for run_id in ids:
            if access.owns_run(runtime, run_id, principal):
                run = runtime.store.get_assistant_run(run_id)
                if not run:  # row removed after the ownership check; skip, never 500
                    continue
                plan = run.get('plan') if isinstance(run.get('plan'), dict) else {}
                rows.append({key: run.get(key) for key in ('id', 'conversation_id', 'status', 'created_at', 'updated_at', 'model', 'model_provider', 'error_class')} | {
                    'require_artifact': bool(plan.get('require_artifact')),
                    'artifact_count': len(runtime.store.list_deliverables(run_id)),
                    # C4c: lets the title say 等你确认 on first paint instead of the task's 已交付.
                    'plan_pending': run_plan_pending(runtime, run),
                    # t238 C7c-FIX4: a reply that only explains a failed training is not a delivery.
                    'training_outcome': _training_outcome(runtime, run),
                })
        return sorted(rows, key=lambda row: (row['created_at'], row['id']), reverse=True)

    def detail(self, principal, task_id, runtime, access):
        task = self.get(principal, task_id)
        runs = self.runs(principal, task_id, runtime, access)
        if runs:
            task['status'] = delivery_status(runs)
        return {'task': task, 'runs': runs, 'files': self.files(principal, task_id, runtime, access),
                'generated_files': self.generated_files(principal, task_id, runtime, access)}

    def files(self, principal, task_id, runtime, access):
        from .user_files import metadata
        self.get(principal, task_id)
        with self.connect() as connection:
            rows = connection.execute('SELECT file_id,sha256 FROM user_task_files WHERE task_id=? ORDER BY file_id', (task_id,)).fetchall()
        files = []
        for row in rows:
            if runtime.store.get_attachment(row['file_id']) is None:
                continue  # t238 C7b H2: a deleted 资料 file never 404s the task; delete drops the row too
            item = metadata(runtime, access, principal, row['file_id'])
            if item['sha256'] != row['sha256']:
                raise AccessError(409, 'task_file_version_changed')
            files.append(item)
        return files

    def generated_files(self, principal, task_id, runtime, access):
        from .user_files import public_generated
        self.get(principal, task_id)
        files = []
        for run in self.runs(principal, task_id, runtime, access):
            for row in runtime.store.list_deliverables(run['id']):
                files.append(public_generated(row))
                if len(files) >= 500:
                    return files
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
