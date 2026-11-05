// C6-acct contract. Driven by tests/test_c6acct_accounts.py, which provisions the DB with the real
// scripts/manage_product_accounts.py `bootstrap` and passes the fixtures through the environment.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync, writeFileSync, renameSync, copyFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import ts from 'typescript';

const E = process.env;
const required = ['C6ACCT_DB', 'C6ACCT_ADMIN_USERNAME', 'C6ACCT_ADMIN_PASSWORD', 'C6ACCT_ADMIN_SCRYPT', 'C6ACCT_USERB_USERNAME', 'C6ACCT_USERB_PASSWORD'];
const skip = required.every((name) => E[name]) ? false : 'driven by tests/test_c6acct_accounts.py';
const require = createRequire(import.meta.url);
const resolve = Module._resolveFilename;
Module._resolveFilename = function(name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../' + name.slice(2), import.meta.url));
  return resolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename,
}).outputText, filename);

let login, access, principal, tenantIdForUsername;
if (!skip) {
  process.env.WORKSTATION_SESSION_SECRET = 'isolated-c6acct-session-key-not-a-real-secret';
  process.env.WORKSTATION_ADMIN_USERNAME = E.C6ACCT_ADMIN_USERNAME;
  process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT = E.C6ACCT_ADMIN_SCRYPT;
  process.env.WORKSTATION_ACCOUNTS_DB = E.C6ACCT_DB;
  ({ tenantIdForUsername } = await import('./tenant-identity.mjs'));
  const { POST } = require('../../app/api/auth/login/route.ts');
  const { proxy } = require('../../proxy.ts');
  const { NextRequest } = require('next/server');
  ({ sessionPrincipal: principal } = require('./local-session.ts'));
  const { issueLoginCaptcha } = require('./login-captcha.ts');
  // Solves the single-use SVG challenge the same way scripts/captcha_login_probe.py does.
  const captcha = () => {
    const { captcha_id, image } = issueLoginCaptcha();
    const svg = decodeURIComponent(image.split(',', 2)[1]);
    return { captcha_id, captcha_answer: [...svg.matchAll(/<text\b[^>]*>([^<]+)<\/text>/g)].map((m) => m[1]).join('') };
  };
  login = (username, password) => POST(new Request('http://127.0.0.1/api/auth/login', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password, ...captcha() }) }));
  access = (path, cookie) => proxy(new NextRequest('http://127.0.0.1:8102' + path, { headers: { host: '127.0.0.1:8102', cookie } }));
}
const cookieOf = (response) => response.headers.get('set-cookie').split(';')[0];
const sessionValue = (cookie) => cookie.slice(cookie.indexOf('=') + 1);

test('c6acct_admin_old_password_logs_in_via_db_with_unchanged_tenant', { skip }, async () => {
  const response = await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD);
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.tenant_id, tenantIdForUsername(E.C6ACCT_ADMIN_USERNAME));
  const current = principal(sessionValue(cookieOf(response)));
  assert.equal(current.role, 'admin');
  assert.equal(current.sessionVersion, 1); // proves the DB, not the env fallback, answered
  assert.equal((await access('/api/assistant/tasks', cookieOf(response))).status, 200);
  assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD + 'x')).status, 401);
});

test('c6acct_userb_logs_in_not_admin_and_cannot_read_admin_global_data', { skip }, async () => {
  const admin = await (await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD)).json();
  const response = await login(E.C6ACCT_USERB_USERNAME, E.C6ACCT_USERB_PASSWORD);
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.notEqual(body.tenant_id, admin.tenant_id);
  assert.equal(principal(sessionValue(cookieOf(response))).role, 'user');
  for (const path of ['/api/tasks', '/api/tasks/house_prices', '/api/summary', '/api/artifacts?path=other', '/api/runtime/sessions']) {
    const denied = await access(path, cookieOf(response));
    assert.equal(denied.status, 403, path);
    assert.equal((await denied.json()).code, 'personal_workspace_required');
  }
  assert.equal((await login(E.C6ACCT_USERB_USERNAME, E.C6ACCT_ADMIN_PASSWORD)).status, 401);
  assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_USERB_PASSWORD)).status, 401);
});

test('c6acct_admin_env_fallback_when_db_unreadable_and_only_admin', { skip }, async () => {
  const good = cookieOf(await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD));
  const backup = E.C6ACCT_DB + '.bak';
  copyFileSync(E.C6ACCT_DB, backup);
  try {
    writeFileSync(E.C6ACCT_DB, 'this is not a sqlite database'.repeat(40)); // corrupt registry
    const admin = await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD);
    assert.equal(admin.status, 200);
    assert.equal((await admin.json()).tenant_id, tenantIdForUsername(E.C6ACCT_ADMIN_USERNAME)); // ownership unchanged
    const fallback = principal(sessionValue(cookieOf(admin)));
    assert.equal(fallback.role, 'admin');
    assert.equal(fallback.sessionVersion, 0);
    assert.equal((await access('/api/assistant/tasks', cookieOf(admin))).status, 200);
    assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, 'wrong-password')).status, 401);
    assert.equal((await login(E.C6ACCT_USERB_USERNAME, E.C6ACCT_USERB_PASSWORD)).status, 401); // no one else gets in
    assert.equal(principal(sessionValue(good)), null); // DB-issued session (av=1) is not honoured without the DB
  } finally { renameSync(backup, E.C6ACCT_DB); }
  const fallbackCookie = cookieOf(await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD));
  assert.equal(principal(sessionValue(fallbackCookie)).sessionVersion, 1); // DB answers again
});

test('c6acct_fallback_session_dies_once_registry_is_readable_again', { skip }, async () => {
  const backup = E.C6ACCT_DB + '.bak2';
  copyFileSync(E.C6ACCT_DB, backup);
  let cookie;
  try {
    writeFileSync(E.C6ACCT_DB, 'garbage'.repeat(100));
    cookie = cookieOf(await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD));
    assert.notEqual(principal(sessionValue(cookie)), null);
  } finally { renameSync(backup, E.C6ACCT_DB); }
  assert.equal(principal(sessionValue(cookie)), null); // av=0 never matches a provisioned row (session_version >= 1)
});

test('c6acct_missing_registry_file_also_falls_back_to_admin_only', { skip }, async () => {
  const hidden = E.C6ACCT_DB + '.hidden';
  renameSync(E.C6ACCT_DB, hidden);
  try {
    assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD)).status, 200);
    assert.equal((await login(E.C6ACCT_USERB_USERNAME, E.C6ACCT_USERB_PASSWORD)).status, 401);
  } finally { renameSync(hidden, E.C6ACCT_DB); }
});

test('c6acct_readable_registry_never_falls_back_for_disabled_or_unknown_accounts', { skip }, async () => {
  assert.equal((await login('nobody', E.C6ACCT_ADMIN_PASSWORD)).status, 401);
  const db = new DatabaseSync(E.C6ACCT_DB);
  try {
    db.prepare('UPDATE accounts SET enabled = 0 WHERE username = ?').run(E.C6ACCT_ADMIN_USERNAME);
    assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD)).status, 401); // readable DB wins: no env bypass
  } finally {
    db.prepare('UPDATE accounts SET enabled = 1 WHERE username = ?').run(E.C6ACCT_ADMIN_USERNAME);
    db.close();
  }
  assert.equal((await login(E.C6ACCT_ADMIN_USERNAME, E.C6ACCT_ADMIN_PASSWORD)).status, 200);
});
