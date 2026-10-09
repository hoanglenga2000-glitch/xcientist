"""Per-principal model configuration; saves have no network or execution side effect."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from .model_profile_secrets import ProfileSecrets
from .models import new_id, utc_now
from .personal_model_http import safe_url
from .tenant_access import AccessError
from .user_tasks import UserTasks

PROFILE_ID = re.compile(r'model_[a-f0-9]{32}')
PROVIDERS = {'openai', 'anthropic', 'deepseek'}


class ModelProfiles:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / 'personal_model_profiles.sqlite3'
        self.secrets = ProfileSecrets(self.root)
        with self.connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                    version INTEGER NOT NULL, enabled INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT NOT NULL, version INTEGER NOT NULL, name TEXT NOT NULL, provider TEXT NOT NULL,
                    base_url TEXT NOT NULL, model TEXT NOT NULL, saved_at TEXT NOT NULL,
                    verified_at TEXT NOT NULL DEFAULT '', used_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(id,version)
                );
                CREATE TABLE IF NOT EXISTS defaults (tenant TEXT, owner TEXT, profile_id TEXT NOT NULL,
                    PRIMARY KEY(tenant,owner));
                CREATE TABLE IF NOT EXISTS tests (id TEXT, version INTEGER, key_hash TEXT, started REAL,
                    state TEXT NOT NULL, PRIMARY KEY(id,version,key_hash));
                CREATE TABLE IF NOT EXISTS run_bindings (run_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                    version INTEGER NOT NULL, tenant TEXT NOT NULL, owner TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def public(row):
        return {key: row[key] for key in ('id', 'version', 'name', 'provider', 'base_url', 'model', 'saved_at', 'verified_at', 'used_at')} | {
            'enabled': bool(row['enabled']), 'key_saved': True,
            'verification': 'connection_test_passed' if row['verified_at'] else 'not_tested',
        }

    def get(self, principal, profile_id, version=None):
        scope = UserTasks.scope(principal)
        if not isinstance(profile_id, str) or not PROFILE_ID.fullmatch(profile_id):
            raise AccessError(404, 'model_profile_not_found')
        with self.connect() as connection:
            row = connection.execute('''SELECT r.*,p.enabled FROM profiles p JOIN revisions r
                ON r.id=p.id AND r.version=COALESCE(?,p.version)
                WHERE p.id=? AND p.tenant=? AND p.owner=?''', (version, profile_id, *scope)).fetchone()
            if row is None:
                raise AccessError(404, 'model_profile_not_found')
            return self.public(row)

    def list(self, principal):
        scope = UserTasks.scope(principal)
        with self.connect() as connection:
            profiles = [self.public(row) for row in connection.execute('''SELECT r.*,p.enabled FROM profiles p JOIN revisions r
                ON r.id=p.id AND r.version=p.version WHERE p.tenant=? AND p.owner=? ORDER BY p.created_at DESC''', scope)]
            default = connection.execute('SELECT profile_id FROM defaults WHERE tenant=? AND owner=?', scope).fetchone()
        return {'profiles': profiles, 'default_profile_id': default[0] if default else '',
                'platform_option': 'existing_permissions_only'}

    def save(self, principal, body, profile_id=''):
        scope = UserTasks.scope(principal)
        if set(body) != ({'name', 'provider', 'base_url', 'model', 'api_key', 'version'} if profile_id else {'name', 'provider', 'base_url', 'model', 'api_key'}):
            raise AccessError(400, 'invalid_model_profile')
        if any(not isinstance(body.get(key), str) for key in ('name', 'provider', 'base_url', 'model', 'api_key')):
            raise AccessError(400, 'invalid_model_profile')
        if (not 1 <= len(body['name'].strip()) <= 80 or re.search(r'[\x00-\x1f\x7f]', body['name'])
                or body['provider'] not in PROVIDERS or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', body['model'])
                or len(body['api_key']) > 8192 or re.search(r'[\s\x00-\x1f\x7f]', body['api_key'])):
            raise AccessError(400, 'invalid_model_profile')
        try:
            safe_url(body['base_url'])
        except ValueError:
            raise AccessError(400, 'model_address_invalid') from None
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if profile_id:
                previous = self.get(principal, profile_id)
                if type(body.get('version')) is not int or body['version'] != previous['version']:
                    raise AccessError(409, 'model_profile_version_conflict')
                version = previous['version'] + 1
                key = body['api_key'] or self.secrets.load(*scope, profile_id, previous['version'])
            else:
                if not body['api_key']:
                    raise AccessError(400, 'model_key_required')
                if connection.execute('SELECT count(*) FROM profiles WHERE tenant=? AND owner=?', scope).fetchone()[0] >= 20:
                    raise AccessError(409, 'model_profile_limit_reached')
                profile_id, version, key = new_id('model'), 1, body['api_key']
                connection.execute('INSERT INTO profiles VALUES(?,?,?,?,?,?)', (profile_id, *scope, version, 1, utc_now()))
            try:
                self.secrets.save(*scope, profile_id, version, key)
            except (ValueError, OSError):
                raise AccessError(503, 'model_credential_storage_unavailable') from None
            connection.execute('INSERT INTO revisions(id,version,name,provider,base_url,model,saved_at) VALUES(?,?,?,?,?,?,?)',
                (profile_id, version, body['name'].strip(), body['provider'], body['base_url'].rstrip('/'), body['model'], utc_now()))
            connection.execute('UPDATE profiles SET version=? WHERE id=?', (version, profile_id))
        return {'profile': self.get(principal, profile_id)}

    def command(self, principal, profile_id, action, body):
        profile = self.get(principal, profile_id)
        if body.get('version') != profile['version'] or type(body.get('version')) is not int:
            raise AccessError(409, 'model_profile_version_conflict')
        if action == 'test':
            return self.test(principal, profile, body)
        if action not in {'default', 'disable'} or set(body) != {'version'}:
            raise AccessError(400, 'invalid_model_profile_action')
        scope = UserTasks.scope(principal)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            current = self.get(principal, profile_id)
            if current['version'] != profile['version']:
                raise AccessError(409, 'model_profile_version_conflict')
            if action == 'default':
                if not current['enabled']:
                    raise AccessError(409, 'model_profile_disabled')
                connection.execute('INSERT INTO defaults VALUES(?,?,?) ON CONFLICT(tenant,owner) DO UPDATE SET profile_id=excluded.profile_id', (*scope, profile_id))
            else:
                connection.execute('UPDATE profiles SET enabled=0 WHERE id=?', (profile_id,))
                connection.execute('DELETE FROM defaults WHERE tenant=? AND owner=? AND profile_id=?', (*scope, profile_id))
        return self.list(principal)

    def binding(self, principal, profile_id, version):
        profile = self.get(principal, profile_id)
        if type(version) is not int or version != profile['version']:
            raise AccessError(409, 'model_profile_version_conflict')
        if not profile['enabled']:
            raise AccessError(409, 'model_profile_disabled')
        return {'id': profile_id, 'version': version, 'provider': profile['provider'], 'model': profile['model']}

    def bind_run(self, principal, run_id, profile_id, version):
        scope = UserTasks.scope(principal)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self.binding(principal, profile_id, version)
            connection.execute('INSERT OR IGNORE INTO run_bindings VALUES(?,?,?,?,?)', (run_id, profile_id, version, *scope))
            row = connection.execute('SELECT profile_id,version,tenant,owner FROM run_bindings WHERE run_id=?', (run_id,)).fetchone()
            if tuple(row) != (profile_id, version, *scope):
                raise AccessError(409, 'model_run_binding_is_immutable')

    def run_binding(self, principal, run_id):
        with self.connect() as connection:
            row = connection.execute('SELECT profile_id,version FROM run_bindings WHERE run_id=? AND tenant=? AND owner=?', (run_id, *UserTasks.scope(principal))).fetchone()
        if row is None:
            raise AccessError(409, 'model_run_binding_missing')
        return {'id': row[0], 'version': row[1]}

    def test(self, principal, profile, body):
        if (set(body) != {'version', 'confirm_cost', 'idempotency_key'} or body.get('confirm_cost') is not True
                or not isinstance(body.get('idempotency_key'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', body['idempotency_key'])):
            raise AccessError(400, 'model_test_confirmation_required')
        if not profile['enabled']:
            raise AccessError(409, 'model_profile_disabled')
        digest = hashlib.sha256(body['idempotency_key'].encode()).hexdigest()
        identity = (profile['id'], profile['version'], digest)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT state FROM tests WHERE id=? AND version=? AND key_hash=?', identity).fetchone()
            if previous:
                return {'status': previous[0], 'replayed': True, 'profile': self.get(principal, profile['id'])}
            # A user cannot bypass the hourly cap by creating another profile/revision.
            count = connection.execute('''SELECT count(*) FROM tests t JOIN profiles p ON t.id=p.id
                WHERE p.tenant=? AND p.owner=? AND t.started>?''', (*UserTasks.scope(principal), time.time() - 3600)).fetchone()[0]
            if count >= 3:
                raise AccessError(429, 'model_test_hourly_limit')
            connection.execute('INSERT INTO tests VALUES(?,?,?,?,?)', (*identity, time.time(), 'pending'))
        from .personal_model_client import PersonalModelClient
        try:
            client = PersonalModelClient(self, principal, {'id': profile['id'], 'version': profile['version']})
            client.send([{'role': 'user', 'content': 'Reply OK only.'}], system='Connection test only.', tools=[], max_tokens=16)
            status = 'passed'
        except Exception as error:
            code = str(error)
            status = code if re.fullmatch(r'model_[a-z0-9_]{1,80}', code) else 'model_test_failed'
        with self.connect() as connection:
            connection.execute('UPDATE tests SET state=? WHERE id=? AND version=? AND key_hash=?', (status, *identity))
            if status == 'passed':
                connection.execute('UPDATE revisions SET verified_at=? WHERE id=? AND version=?', (utc_now(), profile['id'], profile['version']))
        return {'status': status, 'replayed': False, 'profile': self.get(principal, profile['id'])}
