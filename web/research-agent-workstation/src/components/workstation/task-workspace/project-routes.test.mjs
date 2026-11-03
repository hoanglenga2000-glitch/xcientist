import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
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

const PID = 'project_' + 'a'.repeat(32);
const FID = 'attachment_' + 'b'.repeat(32);
const project = require('../../../app/api/assistant/projects/[projectId]/[[...path]]/route.ts');
const files = require('../../../app/api/assistant/files/[[...path]]/route.ts');

async function call(route, method, url, params) {
  forwarded.length = 0;
  const response = await route[method](new Request('http://localhost' + url, { method, ...(method === 'GET' ? {} : { body: '{}' }) }), { params: Promise.resolve(params) });
  return [response.status, forwarded[0]?.[1] ?? null];
}

test('t237 C2b.4: the project proxy passes GET/PATCH/DELETE, restore, files and unbind to the runtime', async () => {
  for (const method of ['GET', 'POST', 'PATCH', 'DELETE']) assert.equal(typeof project[method], 'function');
  const base = `/api/assistant/projects/${PID}`;
  assert.deepEqual(await call(project, 'GET', base, { projectId: PID }), [200, `/v1/projects/${PID}`]);
  assert.deepEqual(await call(project, 'PATCH', base, { projectId: PID }), [200, `/v1/projects/${PID}`]);
  assert.deepEqual(await call(project, 'DELETE', base, { projectId: PID }), [200, `/v1/projects/${PID}`]);
  assert.deepEqual(await call(project, 'POST', base + '/restore', { projectId: PID, path: ['restore'] }), [200, `/v1/projects/${PID}/restore`]);
  assert.deepEqual(await call(project, 'GET', base + '/files', { projectId: PID, path: ['files'] }), [200, `/v1/projects/${PID}/files`]);
  assert.deepEqual(await call(project, 'POST', base + '/files', { projectId: PID, path: ['files'] }), [200, `/v1/projects/${PID}/files`]);
  assert.deepEqual(await call(project, 'DELETE', `${base}/files/${FID}`, { projectId: PID, path: ['files', FID] }), [200, `/v1/projects/${PID}/files/${FID}`]);
});

test('t237 C2b.4: anything else on the project proxy is a 404 that never reaches the runtime', async () => {
  const base = `/api/assistant/projects/${PID}`;
  for (const [method, params] of [
    ['GET', { projectId: 'project_bogus' }], ['GET', { projectId: 'project_t237c2bbogus0000' }], ['POST', { projectId: PID }],
    ['GET', { projectId: PID, path: ['restore'] }], ['PATCH', { projectId: PID, path: ['restore'] }], ['DELETE', { projectId: PID, path: ['restore'] }],
    ['PATCH', { projectId: PID, path: ['files'] }], ['DELETE', { projectId: PID, path: ['files'] }],
    ['GET', { projectId: PID, path: ['files', FID] }], ['POST', { projectId: PID, path: ['files', FID] }],
    ['DELETE', { projectId: PID, path: ['files', '../user-files'] }], ['DELETE', { projectId: PID, path: ['files', FID, 'x'] }],
    ['GET', { projectId: PID, path: ['runs'] }],
  ]) assert.deepEqual(await call(project, method, base, params), [404, null], `${method} ${JSON.stringify(params)}`);
  // Queries never reach the runtime on project routes.
  assert.deepEqual(await call(project, 'GET', base + '?x=1', { projectId: PID }), [200, `/v1/projects/${PID}`]);
  assert.equal(existsSync(app(`app/api/assistant/projects/[projectId]/route.ts`)), false, 'one dynamic project route, no sibling conflict');
});

test('t237 C2b.4: the 资料 list can be narrowed to a project; other queries are dropped', async () => {
  assert.deepEqual(await call(files, 'GET', `/api/assistant/files?project_id=${PID}`, {}), [200, `/v1/user-files?project_id=${PID}`]);
  assert.deepEqual(await call(files, 'GET', `/api/assistant/files?q=x`, {}), [200, '/v1/user-files']);
  assert.deepEqual(await call(files, 'GET', `/api/assistant/files?project_id=../x`, {}), [404, null]);
  assert.deepEqual(await call(files, 'GET', `/api/assistant/files/${FID}?project_id=${PID}`, { path: [FID] }), [200, `/v1/user-files/${FID}`]);
});

test('t237 C2b.4: project mutations keep the CSRF + origin gate and the owner pass-through', () => {
  const proxy = readFileSync(app('proxy.ts'), 'utf8');
  assert.match(proxy, /MUTATING_METHODS = new Set\(\["POST", "PUT", "PATCH", "DELETE"\]\)/);
  assert.match(proxy, /validCsrfToken\(request\.headers\.get\(CSRF_HEADER\)/);
  const route = readFileSync(app('app/api/assistant/projects/[projectId]/[[...path]]/route.ts'), 'utf8');
  assert.match(route, /proxyAssistantRuntime\(request, /);
  assert.doesNotMatch(route, /x-evomind-(tenant|principal)/i, 'the owner comes from the session, never from the browser');
});
