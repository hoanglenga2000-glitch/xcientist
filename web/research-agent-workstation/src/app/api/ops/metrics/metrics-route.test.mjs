// t238 C7e O3: /api/ops/metrics is operator-only Prometheus text; tenants and anonymous callers get no data.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const app = (p) => fileURLToPath(new URL('../../../../' + p, import.meta.url));
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
let runtimeReply = async () => new Response('# TYPE evomind_queue_depth gauge\nevomind_queue_depth 2\n', { status: 200 });
const runtimeCalls = [];
fake('lib/server/assistant-runtime.ts', { assistantRuntimeFetch: async (path, init) => { runtimeCalls.push(path); return runtimeReply(init); } });
const route = require('./route.ts');
const req = (who) => new Request('http://127.0.0.1:8088/api/ops/metrics', { headers: who ? { cookie: `evomind_local_session=${who}` } : {} });

test('t238 C7e O3: anonymous 401 and tenant 403 never reach the runtime', async () => {
  assert.equal((await route.GET(req())).status, 401);
  const tenant = await route.GET(req('tenant'));
  assert.equal(tenant.status, 403);
  assert.deepEqual(await tenant.json(), { ok: false, code: 'operator_required' });
  assert.deepEqual(runtimeCalls, []);
});

test('t238 C7e O3: operator gets Prometheus text; a failing runtime is 503 text, not a hang', async () => {
  const ok = await route.GET(req('operator'));
  assert.equal(ok.status, 200);
  assert.match(ok.headers.get('content-type'), /^text\/plain; version=0\.0\.4/);
  assert.match(await ok.text(), /^evomind_queue_depth 2$/m);
  assert.deepEqual(runtimeCalls, ['/v1/metrics']);
  runtimeReply = async () => new Response('x', { status: 500 });
  const bad = await route.GET(req('operator'));
  assert.equal(bad.status, 503);
});
