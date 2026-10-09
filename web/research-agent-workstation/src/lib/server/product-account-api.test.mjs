import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const resolve = Module._resolveFilename;
Module._resolveFilename = function(name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../' + name.slice(2), import.meta.url));
  return resolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename,
}).outputText, filename);

const { createPasswordHash } = require('./public-auth.ts');
process.env.WORKSTATION_SESSION_SECRET = 'isolated-product-account-session-key-not-a-real-secret';
process.env.WORKSTATION_ADMIN_USERNAME = 'legacy-admin';
process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT = createPasswordHash('legacy-fixture-password');
const directory = mkdtempSync(join(tmpdir(), 'evomind-account-contract-'));
process.env.WORKSTATION_ACCOUNTS_DB = join(directory, 'accounts.sqlite3');
const db = new DatabaseSync(process.env.WORKSTATION_ACCOUNTS_DB);
db.exec(`CREATE TABLE accounts (username TEXT PRIMARY KEY COLLATE NOCASE, tenant_id TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL, role TEXT NOT NULL, enabled INTEGER NOT NULL, session_version INTEGER NOT NULL)`);
const { tenantIdForUsername } = await import('./tenant-identity.mjs');
for (const name of ['alice', 'bob']) db.prepare('INSERT INTO accounts VALUES (?, ?, ?, ?, 1, 1)')
  .run(name, tenantIdForUsername(name), createPasswordHash(`${name}-fixture-password`), 'user');
const { POST } = require('../../app/api/auth/login/route.ts');

async function login(username, password = `${username}-fixture-password`) {
  return POST(new Request('http://127.0.0.1/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }) }));
}

test('two independently configured users log in without borrowing the administrator identity', async () => {
  const alice = await login('alice');
  assert.equal(alice.status, 200);
  const a = await alice.json();
  const bob = await login('bob');
  assert.equal(bob.status, 200);
  const b = await bob.json();
  assert.equal(a.username, 'alice'); assert.equal(b.username, 'bob');
  assert.notEqual(a.tenant_id, b.tenant_id);
  assert.doesNotMatch(JSON.stringify([a, b]), /password|scrypt\$/);
  assert.equal((await login('bob', 'alice-fixture-password')).status, 401);
  assert.equal((await login('legacy-admin', 'legacy-fixture-password')).status, 401);
});

test('disabling or resetting an account immediately invalidates its existing authenticated requests', async () => {
  const { proxy } = require('../../proxy.ts');
  const { NextRequest } = require('next/server');
  const access = (cookie) => proxy(new NextRequest('http://127.0.0.1:8102/api/assistant/tasks', {
    headers: { host: '127.0.0.1:8102', cookie },
  }));
  const cookie = (await login('alice')).headers.get('set-cookie').split(';')[0];
  assert.equal((await access(cookie)).status, 200);
  db.prepare('UPDATE accounts SET enabled = 0, session_version = session_version + 1 WHERE username = ?').run('alice');
  assert.equal((await access(cookie)).status, 401);
  assert.equal((await login('alice')).status, 401);
  db.prepare('UPDATE accounts SET enabled = 1, session_version = session_version + 1 WHERE username = ?').run('alice');
  const next = (await login('alice')).headers.get('set-cookie').split(';')[0];
  assert.equal((await access(next)).status, 200);
  assert.equal((await access(cookie)).status, 401);
});

test('personal accounts cannot read legacy global task lists or use unscoped write and download endpoints', async () => {
  const { proxy } = require('../../proxy.ts');
  const { NextRequest } = require('next/server');
  const cookie = (await login('bob')).headers.get('set-cookie').split(';')[0];
  for (const path of ['/api/tasks', '/api/tasks/house_prices', '/api/summary', '/api/artifacts?path=other',
    '/api/literature?task_id=house_prices', '/api/runtime/sessions', '/api/multi-agent/runs']) {
    const result = await proxy(new NextRequest('http://127.0.0.1:8102' + path, { headers: { host: '127.0.0.1:8102', cookie } }));
    assert.equal(result.status, 403, path);
    assert.equal((await result.json()).code, 'personal_workspace_required');
  }
});
