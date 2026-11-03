// t238 C7b D6: per-file delete (confirm / referenced / active refusal), export, and the runtime routes.
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
// The runtime launcher and the session-scoping fetch are faked; the real assistant-runtime proxy and route handlers run.
const scoped = new Module(app('lib/server/scoped-runtime-fetch.ts'));
scoped.filename = scoped.id; scoped.loaded = true;
scoped.exports = { scopedRuntimeFetch: (input, init) => globalThis.fetch(input, init) };
require.cache[scoped.id] = scoped;
const launcher = new Module(app('lib/server/evomind-runtime.ts'));
launcher.filename = launcher.id; launcher.loaded = true;
launcher.exports = { ensureEvoMindRuntime: async () => ({ baseUrl: 'http://runtime.test', token: 'fixture-token' }) };
require.cache[launcher.id] = launcher;


const read = (name) => readFileSync(new URL('./' + name, import.meta.url), 'utf8');

test('t238 C7b D6: dialog state — not referenced → plain confirm with 删除', () => {
  const { deleteDialogState, FILE_DELETE_COPY } = require('./file-delete.ts');
  assert.deepEqual(deleteDialogState('数据.csv', []), { kind: 'plain', title: '删除这个文件？', name: '数据.csv', line: '删除后不能恢复。', confirm: '删除' }); // t238 C7c-7b
  assert.equal(FILE_DELETE_COPY.cancel, '取消');
  assert.equal(FILE_DELETE_COPY.export, '导出我的数据');
});

test('t238 C7b D6: dialog state — referenced by a finished task → warning naming it, then 仍然删除', () => {
  const { deleteDialogState } = require('./file-delete.ts');
  const state = deleteDialogState('train.csv', [{ task_id: 'utask_1', task_title: '房价预测', run_id: 'run_1', active: false }]);
  assert.deepEqual(state, { kind: 'referenced', title: '删除这个文件？', name: 'train.csv',
    line: '这个文件正在被任务「房价预测」使用。删除后，这个任务就读不到它了。删除后不能恢复。', confirm: '仍然删除' }); // t238 C7c-7e
});

test('t238 C7b D6: dialog state — active run → refusal naming the task and no delete button', () => {
  const { deleteDialogState } = require('./file-delete.ts');
  const state = deleteDialogState('train.csv', [
    { task_id: 'utask_2', task_title: '旧任务', run_id: 'run_2', active: false },
    { task_id: 'utask_1', task_title: '房价预测', run_id: 'run_1', active: true }]);
  assert.deepEqual(state, { kind: 'refused', title: '现在不能删除「train.csv」', name: 'train.csv', line: '任务「房价预测」正在运行，结束后才能删除这个文件。' }); // t238 C7c-A3 #3
  assert.equal('confirm' in state, false);
  const dialog = read('DeleteFileDialog.tsx');
  assert.match(dialog, /state\.kind !== 'refused' && <button/);
});

test('t238 C7b D6: the dialog is an accessible alertdialog with the danger token and 取消 focused first', () => {
  const dialog = read('DeleteFileDialog.tsx');
  assert.match(dialog, /role="alertdialog"/);
  assert.match(dialog, /aria-labelledby="tw-delete-file-title"/);
  assert.match(dialog, /aria-describedby=/);
  assert.match(dialog, /cancel\.current\?\.focus\(\)/);
  assert.match(dialog, /event\.key === 'Escape'/);
  assert.match(dialog, /tw-danger-solid tw-nowrap/);
  assert.doesNotMatch(dialog, /#[0-9a-fA-F]{3,6}\b/);
  const css = read('workspace.css');
  assert.match(css, /\.tw-danger-solid\{background:var\(--tw-danger-strong\)/);
  assert.match(css, /\.tw-nowrap\{white-space:nowrap;flex:none\}/);
  const files = read('TaskFiles.tsx');
  assert.match(files, /fileUsageUrl\(file\.id\)/);
  assert.match(files, /tw-text-danger tw-nowrap/);
  assert.match(files, /下载资料/);
  assert.match(files, /<ExportData \/>/); // t238 C7c-A3 #4: fetch-based export
  assert.match(read('ExportData.tsx'), /fetch\(TENANT_EXPORT_URL/);
});

test('t238 C7b D6: routes forward delete / usage / tenant data / export to the runtime unchanged', async (context) => {
  const files = require('../../../app/api/assistant/files/[[...path]]/route.ts');
  const uploads = require('../../../app/api/assistant/uploads/[uploadId]/route.ts');
  const data = require('../../../app/api/assistant/tenant/data/route.ts');
  const exported = require('../../../app/api/assistant/tenant/export/route.ts');
  const seen = [];
  const IN_USE = '任务「房价预测」正在运行，结束后才能删除这个文件。'; // t238 C7c-7e
  context.mock.method(globalThis, 'fetch', async (url, init) => {
    seen.push(`${init?.method ?? 'GET'} ${String(url)}`);
    if (String(url).endsWith('/export')) return new Response('PK', { headers: { 'content-type': 'application/zip', 'content-disposition': "attachment; filename=\"evomind-export-20261101.zip\"" } });
    if (String(url).includes('upload_' + 'b'.repeat(32))) return Response.json({ error: 'attachment_in_use', message: IN_USE }, { status: 409 });
    return Response.json({ ok: true });
  });
  const id = 'attachment_' + 'a'.repeat(32);
  const del = (body = '{}') => new Request('http://localhost/x', { method: 'DELETE', body, headers: { 'content-type': 'application/json' } });
  assert.equal((await files.DELETE(del(), { params: Promise.resolve({ path: [id] }) })).status, 200);
  assert.equal((await files.GET(new Request('http://localhost/x'), { params: Promise.resolve({ path: [id, 'usage'] }) })).status, 200);
  assert.equal((await files.DELETE(del(), { params: Promise.resolve({ path: ['../x'] }) })).status, 404);
  const refused = await uploads.DELETE(del(), { params: Promise.resolve({ uploadId: 'upload_' + 'b'.repeat(32) }) });
  assert.equal(refused.status, 409);
  assert.deepEqual(await refused.json(), { error: 'attachment_in_use', message: IN_USE });
  assert.equal((await uploads.DELETE(del(), { params: Promise.resolve({ uploadId: 'nope' }) })).status, 404);
  assert.equal((await data.DELETE(del('{"confirm":"DELETE"}'))).status, 200);
  const zip = await exported.GET(new Request('http://localhost/x'));
  assert.equal(zip.headers.get('content-type'), 'application/zip');
  assert.match(zip.headers.get('content-disposition'), /^attachment; filename="evomind-export-\d{8}\.zip"$/);
  assert.deepEqual(seen, [
    `DELETE http://runtime.test/v1/attachments/${id}`, `GET http://runtime.test/v1/user-files/${id}/usage`,
    `DELETE http://runtime.test/v1/uploads/upload_${'b'.repeat(32)}`, 'DELETE http://runtime.test/v1/tenant/data',
    'GET http://runtime.test/v1/tenant/export']);
});

test('t238 C7b D6: attachment_in_use and export_busy keep the server Chinese line without code', async () => {
  const { errorCopy } = require('./error-copy.ts');
  const { TaskRequestError } = require('./types.ts');
  const inUse = errorCopy(new TaskRequestError('任务「A」正在运行，结束后才能删除这个文件。', 409, 'attachment_in_use'));
  assert.equal(inUse.line, '任务「A」正在运行，结束后才能删除这个文件。'); // t238 C7c-7e
  assert.doesNotMatch(inUse.line, /attachment_in_use|409/);
  assert.equal(errorCopy(new TaskRequestError('busy', 429, 'export_busy')).line, '现在导出的人较多，请稍后再试。'); // t238 C7c-7e
});

test('t238 C7b D6: the proxy treats /api/assistant/tenant/* as a personal route', () => {
  const proxy = readFileSync(new URL('../../../proxy.ts', import.meta.url), 'utf8');
  assert.match(proxy, /user-settings\|experience\|tenant(\|gpu)?\)/);
});
