import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const require = createRequire(import.meta.url);
const app = (path) => fileURLToPath(new URL('../../../' + path, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const forwarded = [];
const fake = new Module(app('lib/server/assistant-runtime.ts'));
fake.filename = fake.id; fake.loaded = true;
fake.exports = { proxyAssistantRuntime: async (request, pathname) => { forwarded.push([request.method, pathname]); return Response.json({ ok: true }); } };
require.cache[fake.id] = fake;
const h = React.createElement;
const EID = 'exp_' + 'e'.repeat(32);

test('phase 2: the experience proxy passes list, settings and single delete only', async () => {
  const route = require('../../../app/api/assistant/experience/[[...path]]/route.ts');
  const call = async (method, path = []) => {
    forwarded.length = 0;
    const url = 'http://localhost/api/assistant/experience' + (path.length ? '/' + path.join('/') : '') + '?owner=bob';
    const response = await route[method](new Request(url, { method, ...(method === 'GET' ? {} : { body: '{}' }) }), { params: Promise.resolve({ path }) });
    return [response.status, forwarded[0]?.[1] ?? null];
  };
  assert.deepEqual(await call('GET'), [200, '/v1/experience']);
  assert.deepEqual(await call('GET', ['settings']), [200, '/v1/experience/settings']);
  assert.deepEqual(await call('PATCH', ['settings']), [200, '/v1/experience/settings']);
  assert.deepEqual(await call('DELETE', [EID]), [200, `/v1/experience/${EID}`]);
  for (const [method, path] of [['DELETE', []], ['DELETE', ['settings']], ['PATCH', []], ['PATCH', [EID]], ['GET', [EID]],
    ['DELETE', ['exp_bogus']], ['DELETE', ['..', 'user-memory']], ['GET', ['settings', 'x']]]) {
    assert.deepEqual(await call(method, path), [404, null], `${method} ${JSON.stringify(path)}`);
  }
  assert.equal(route.POST, undefined, 'no create: the platform records experience');
});

test('phase 2: 我的经验 view lists rows with a labelled delete and the anonymous-statistics switch', () => {
  const { ExperienceView } = require('./MemorySettings.tsx');
  const row = { id: EID, title: '成功：多分类，1千到1万行，macro-F1 0.412', summary: '设置：隐藏层 128x64', outcome: 'success', task: 'classification',
    created_at: '2026-11-06T02:00:00Z', success_count: 2, use_count: 3 };
  const base = { now: Date.parse('2026-11-06T12:00:00+08:00'), onContribute() {}, onDelete() {} };
  const html = renderToStaticMarkup(h(ExperienceView, { ...base, experiences: [row], contribute: true }));
  assert.match(html, /我的经验/);
  assert.match(html, /aria-label="删除经验：成功：多分类/);
  assert.match(html, /帮到后续训练 2 次/);
  assert.match(html, /role="switch" aria-checked="true" aria-labelledby="tw-experience-contribute-label"/);
  assert.doesNotMatch(html, /exp_e{32}<|macro_f1|hidden_sizes/, 'no ids or field names as text');
  const empty = renderToStaticMarkup(h(ExperienceView, { ...base, experiences: [], contribute: false }));
  assert.match(empty, /还没有训练经验/);
  assert.match(empty, /aria-checked="false"/);
  const loading = renderToStaticMarkup(h(ExperienceView, { ...base, experiences: [], contribute: true, loading: true }));
  assert.doesNotMatch(loading, /还没有训练经验/);
});
