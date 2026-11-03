// t238 C7b O2: /api/healthz stays public and unchanged; ?deep=1 is operator-only and never hangs.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const app = (p) => fileURLToPath(new URL('../../../' + p, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename }).outputText, filename);
};
const fake = (path, exports) => { const m = new Module(app(path)); m.filename = m.id; m.loaded = true; m.exports = exports; require.cache[m.id] = m; };
const SESSIONS = { operator: { authentication: 'local_bootstrap' }, tenant: { authentication: 'public_password' } };
fake('lib/server/local-session.ts', { LOCAL_AUTOMATION_VERIFIED_HEADER: 'x-evomind-local-automation-verified', SESSION_COOKIE: 'evomind_local_session',
  validSessionCookie: (v) => Boolean(SESSIONS[v]), sessionPrincipal: (v) => SESSIONS[v] ?? null,
  LOCAL_AUTOMATION_HEADER: 'x-evomind-local-automation', validLocalAutomationToken: () => false });
fake('lib/server/runtime-version.ts', { runtimeVersionIdentity: async () => ({ ready: true, status: 'ready', frontend_version: '1.0', commit_hash: 'c', source_revision: 'r', source_commit: 's', source_tree_sha256: 't', source_provenance: 'p', build_id: 'b', failures: [] }) });
let runtimeReply = async () => Response.json({ status: 'ok', failed: [], components: { sqlite: { ok: true } } });
const runtimeCalls = [];
fake('lib/server/assistant-runtime.ts', { assistantRuntimeFetch: async (path, init) => { runtimeCalls.push(path); return runtimeReply(init); } });
const route = require('./route.ts');
const req = (query, who) => new Request(`http://127.0.0.1:8088/api/healthz${query}`, { headers: who ? { cookie: `evomind_local_session=${who}` } : {} });
const PLAIN_KEYS = ['base_commit', 'build_id', 'commit_hash', 'failures', 'ok', 'service', 'source_commit', 'source_provenance', 'source_revision', 'source_tree_sha256', 'status', 'version'];

test('t238 C7b O2: plain /api/healthz is unchanged and never calls the deep runtime check', async () => {
  for (const response of [await route.GET(req('')), await route.GET(new Request('http://127.0.0.1:8088/api/healthz?')), await route.GET(req('?deep=0', 'tenant'))]) {
    assert.equal(response.status, 200);
    assert.deepEqual(Object.keys(await response.json()).sort(), PLAIN_KEYS);
  }
  assert.deepEqual(runtimeCalls, []);
});

test('t238 C7b O2: deep=1 rejects anonymous (401) and tenant (403) with no deep data', async () => {
  const anonymous = await route.GET(req('?deep=1'));
  assert.equal(anonymous.status, 401);
  assert.deepEqual(await anonymous.json(), { ok: false, code: 'session_required' });
  const tenant = await route.GET(req('?deep=1', 'tenant'));
  assert.equal(tenant.status, 403);
  assert.deepEqual(await tenant.json(), { ok: false, code: 'operator_required' });
  assert.deepEqual(runtimeCalls, []);
});

test('t238 C7b O2: deep=1 for the operator returns the runtime report; an unreachable runtime is degraded, not a hang', async () => {
  const ok = await route.GET(req('?deep=1', 'operator'));
  assert.equal(ok.status, 200);
  assert.deepEqual(await ok.json(), { ok: true, status: 'ok', failed: [], components: { sqlite: { ok: true } } });
  runtimeReply = (init) => new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(new Error('aborted'))));
  const started = Date.now();
  const keepAlive = setTimeout(() => {}, 5000); // AbortSignal.timeout's timer is unref'd
  const slow = await route.GET(req('?deep=1', 'operator'));
  clearTimeout(keepAlive);
  assert.ok(Date.now() - started < 1000);
  assert.equal(slow.status, 503);
  assert.deepEqual((await slow.json()).failed, ['runtime']);
});
