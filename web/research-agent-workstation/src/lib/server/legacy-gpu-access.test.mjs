// t238 C7b S6: legacy /api/gpu/jobs* records are not tenant-attributable → operator only.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const app = (p) => fileURLToPath(new URL('../../' + p, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename }).outputText, filename);
};
// Session resolution is faked per cookie value; the real access helper and route handlers run.
const SESSIONS = {
  operator: { username: 'local', tenantId: null, authentication: 'local_bootstrap' },
  tenantA: { username: 'alice', tenantId: 'tenant_' + 'a'.repeat(24), authentication: 'public_password' },
  tenantB: { username: 'bob', tenantId: 'tenant_' + 'b'.repeat(24), authentication: 'public_password' },
};
const session = new Module(app('lib/server/local-session.ts'));
session.filename = session.id; session.loaded = true;
session.exports = {
  LOCAL_AUTOMATION_VERIFIED_HEADER: 'x-evomind-local-automation-verified',
  SESSION_COOKIE: 'evomind_local_session',
  validSessionCookie: (value) => Boolean(value && SESSIONS[value]),
  sessionPrincipal: (value) => SESSIONS[value] ?? null,
};
require.cache[session.id] = session;
const gateway = new Module(app('lib/server/gpu-ssh-gateway.ts'));
gateway.filename = gateway.id; gateway.loaded = true;
const calls = [];
gateway.exports = {
  submitGpuJob: async (input) => { calls.push(['submit', input.taskId]); return { ok: true }; },
  readGpuJob: async (id) => { calls.push(['read', id]); return { job_id: id }; },
  cancelGpuJob: async (id) => { calls.push(['cancel', id]); return { ok: true }; },
};
require.cache[gateway.id] = gateway;

const workspace = mkdtempSync(path.join(tmpdir(), 'c7b-s6-'));
mkdirSync(path.join(workspace, 'workspace', 'gpu', 'jobs'), { recursive: true });
writeFileSync(path.join(workspace, 'workspace', 'gpu', 'jobs', 'job_tenant_b.json'), JSON.stringify({ job_id: 'job_tenant_b', status: 'running' }));
process.env.EVOMIND_WORKSPACE_ROOT = workspace;
process.env.WORKSTATION_ROOT = workspace;

const list = require('../../app/api/gpu/jobs/route.ts');
const one = require('../../app/api/gpu/jobs/[jobId]/route.ts');
const cancel = require('../../app/api/gpu/jobs/[jobId]/cancel/route.ts');
const { LEGACY_GPU_TENANT_MESSAGE } = require('./legacy-gpu-access.ts');

const req = (who, init = {}) => new Request('http://127.0.0.1:8088/api/gpu/jobs', {
  ...init, headers: { ...(who ? { cookie: `evomind_local_session=${who}` } : {}), ...init.headers } });
const params = { params: Promise.resolve({ jobId: 'job_tenant_b' }) };

test('t238 C7b S6: a tenant session cannot list, read, submit or cancel legacy GPU jobs (403, Chinese, no data)', async () => {
  for (const who of ['tenantA', 'tenantB']) {
    for (const response of [await list.GET(req(who)), await one.GET(req(who), params), await cancel.POST(req(who, { method: 'POST' }), params),
      await list.POST(req(who, { method: 'POST', body: '{}', headers: { 'content-type': 'application/json' } }))]) {
      assert.equal(response.status, 403);
      const body = await response.json();
      assert.deepEqual(Object.keys(body).sort(), ['code', 'error', 'message', 'ok']);
      assert.equal(body.code, 'legacy_gpu_operator_only');
      assert.equal(body.message, LEGACY_GPU_TENANT_MESSAGE);
      assert.doesNotMatch(JSON.stringify(body), /job_tenant_b|您/);
    }
  }
  assert.deepEqual(calls, []);
});

test('t238 C7b S6: anonymous and forged cookies are rejected with 401', async () => {
  for (const who of [null, 'forged']) {
    const response = await list.GET(req(who));
    assert.equal(response.status, 401);
    assert.equal((await response.json()).code, 'session_required');
  }
  const ambiguous = new Request('http://127.0.0.1:8088/api/gpu/jobs', { headers: { cookie: 'evomind_local_session=operator; evomind_local_session=tenantA' } });
  assert.equal((await list.GET(ambiguous)).status, 401);
  assert.deepEqual(calls, []);
});

test('t238 C7b S6: the operator (local bootstrap or verified automation) keeps full access', async () => {
  const response = await list.GET(req('operator'));
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.ok(Array.isArray(body.jobs));
  const automation = await one.GET(new Request('http://127.0.0.1/api/gpu/jobs/x', { headers: { 'x-evomind-local-automation-verified': '1' } }), params);
  assert.equal(automation.status, 200);
  assert.equal((await cancel.POST(req('operator', { method: 'POST' }), params)).status, 200);
  assert.deepEqual(calls, [['read', 'job_tenant_b'], ['cancel', 'job_tenant_b']]);
});
