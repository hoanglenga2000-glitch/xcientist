import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const app = (path) => fileURLToPath(new URL('../../../' + path, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename }).outputText, filename);
};
// The runtime proxy is replaced by a recorder: these tests check which runtime path each browser request may reach.
const forwarded = [];
const fake = new Module(app('lib/server/assistant-runtime.ts'));
fake.filename = fake.id; fake.loaded = true;
fake.exports = { proxyAssistantRuntime: async (request, pathname) => { forwarded.push([request.method, pathname]); return Response.json({ ok: true }); } };
require.cache[fake.id] = fake;


// t237 C3.7: memory / settings / experience / plan proxies.
const MID = 'memory_' + 'c'.repeat(32);
const RID = 'run_' + 'd'.repeat(32);
const memory = require('../../../app/api/assistant/user-memory/[[...path]]/route.ts');
const settings = require('../../../app/api/assistant/user-settings/route.ts');
const experience = require('../../../app/api/assistant/experience/summary/route.ts');
const plan = require('../../../app/api/assistant/runs/[runId]/plan/route.ts');

async function call(route, method, url, params = {}) {
  forwarded.length = 0;
  const response = await route[method](new Request('http://localhost' + url, { method, ...(method === 'GET' ? {} : { body: '{}' }) }), { params: Promise.resolve(params) });
  return [response.status, forwarded[0]?.[1] ?? null];
}

test('t237 C3.7: the memory proxy passes list, edit, delete, restore and clear-all; no create', async () => {
  const base = '/api/assistant/user-memory';
  assert.deepEqual(await call(memory, 'GET', base), [200, '/v1/user-memory']);
  assert.deepEqual(await call(memory, 'DELETE', base), [200, '/v1/user-memory']);
  assert.deepEqual(await call(memory, 'PATCH', `${base}/${MID}`, { path: [MID] }), [200, `/v1/user-memory/${MID}`]);
  assert.deepEqual(await call(memory, 'DELETE', `${base}/${MID}`, { path: [MID] }), [200, `/v1/user-memory/${MID}`]);
  assert.deepEqual(await call(memory, 'POST', `${base}/${MID}/restore`, { path: [MID, 'restore'] }), [200, `/v1/user-memory/${MID}/restore`]);
  assert.deepEqual(await call(memory, 'GET', `${base}?owner=bob`), [200, '/v1/user-memory'], 'queries never reach the runtime');
  for (const [method, path] of [
    ['POST', []], ['PATCH', []], ['GET', [MID]], ['POST', [MID]], ['PATCH', [MID, 'restore']], ['DELETE', [MID, 'restore']],
    ['PATCH', ['memory_bogus']], ['DELETE', ['../user-settings']], ['POST', [MID, 'restore', 'x']], ['POST', ['restore_all']],
  ]) assert.deepEqual(await call(memory, method, base, { path }), [404, null], `${method} ${JSON.stringify(path)}`);
});

test('t237 C3.7: settings, experience summary and plan decision proxies', async () => {
  assert.deepEqual(await call(settings, 'GET', '/api/assistant/user-settings'), [200, '/v1/user-settings']);
  assert.deepEqual(await call(settings, 'PATCH', '/api/assistant/user-settings'), [200, '/v1/user-settings']);
  assert.equal(settings.POST, undefined);
  assert.deepEqual(await call(experience, 'GET', '/api/assistant/experience/summary?x=1'), [200, '/v1/experience/summary']);
  assert.equal(experience.POST, undefined);
  assert.deepEqual(await call(plan, 'POST', `/api/assistant/runs/${RID}/plan`, { runId: RID }), [200, `/v1/runs/${RID}/plan`]);
  assert.deepEqual(await call(plan, 'POST', '/api/assistant/runs/x/plan', { runId: '../user-memory' }), [404, null]);
  assert.equal(plan.GET, undefined);
});

test('t237 C3.7: new routes are personal APIs behind the CSRF + origin gate; owner comes from the session', () => {
  const proxy = readFileSync(app('proxy.ts'), 'utf8');
  const personal = proxy.match(/const personal = \/(.+)\/\.test\(pathname\)/)[1];
  for (const name of ['user-memory', 'user-settings', 'experience']) assert.ok(personal.includes(name), name);
  assert.match(proxy, /validCsrfToken\(request\.headers\.get\(CSRF_HEADER\)/);
  for (const path of ['app/api/assistant/user-memory/[[...path]]/route.ts', 'app/api/assistant/user-settings/route.ts',
    'app/api/assistant/experience/summary/route.ts', 'app/api/assistant/runs/[runId]/plan/route.ts']) {
    const route = readFileSync(app(path), 'utf8');
    assert.match(route, /proxyAssistantRuntime\(request, /, path);
    assert.doesNotMatch(route, /x-evomind-(tenant|principal)/i, path);
  }
});
