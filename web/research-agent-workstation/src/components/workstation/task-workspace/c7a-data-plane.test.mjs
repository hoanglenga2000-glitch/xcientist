// t238 C7a: upload quota / disk low-water copy, trainability notice and the training input-bound status.
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

// t238 C7b COPY: the Critiquito strings.
const QUOTA_LINE = '存储空间已用完（已用 5 GiB，上限 5 GiB）。删掉不用的文件后再上传。'; // t238 C7c-7a
const DISK_LINE = '暂时无法上传。请稍后再试，或联系管理员。';
const NOTICE = '文件超过训练上限（256\u00a0MiB）。可以直接对话和分析；要训练，先缩小或预处理。';

test('t238 C7a: upload routes pass the runtime 413 status, code and Chinese message through unchanged', async (context) => {
  const create = require('../../../app/api/assistant/uploads/route.ts');
  const complete = require('../../../app/api/assistant/uploads/[uploadId]/complete/route.ts');
  const seen = [];
  context.mock.method(globalThis, 'fetch', async (url) => {
    seen.push(String(url));
    return String(url).endsWith('/complete')
      ? Response.json({ attachment: { id: 'attachment_1' }, trainability: { trainable: false, reasons: ['input_bytes_exceeded'] } })
      : Response.json({ error: 'quota_exceeded', message: QUOTA_LINE }, { status: 413 });
  });
  const response = await create.POST(new Request('http://localhost/api/assistant/uploads', { method: 'POST', body: '{}' }));
  assert.equal(response.status, 413);
  assert.deepEqual(await response.json(), { error: 'quota_exceeded', message: QUOTA_LINE });
  const done = await complete.POST(new Request('http://localhost/x', { method: 'POST', body: '{}' }), { params: Promise.resolve({ uploadId: 'upload_1' }) });
  assert.equal(done.status, 200);
  assert.deepEqual((await done.json()).trainability.reasons, ['input_bytes_exceeded']);
  assert.deepEqual(seen, ['http://runtime.test/v1/uploads', 'http://runtime.test/v1/uploads/upload_1/complete']);
});

test('t238 C7a: taskRequest and errorCopy keep the server upload-limit line, with Chinese fallbacks', async (context) => {
  const { taskRequest, TaskRequestError } = require('./types.ts');
  const { errorCopy, UPLOAD_LIMIT_COPY, ERROR_COPY } = require('./error-copy.ts');
  const fetch = context.mock.method(globalThis, 'fetch', async () => Response.json({ error: 'quota_exceeded', message: QUOTA_LINE }, { status: 413 }));
  const quota = await taskRequest('/api/assistant/uploads').catch(error => error);
  assert.ok(quota instanceof TaskRequestError);
  assert.deepEqual([quota.status, quota.code, quota.message], [413, 'quota_exceeded', QUOTA_LINE]);
  assert.deepEqual(errorCopy(quota), { key: '413', line: QUOTA_LINE, action: '管理文件', details: ['HTTP 413 · quota_exceeded'] });
  fetch.mock.mockImplementation(async () => Response.json({ error: 'disk_low_water', message: 'disk low' }, { status: 413 }));
  const disk = await taskRequest('/api/assistant/uploads').catch(error => error);
  assert.equal(disk.message, DISK_LINE); // a non-Chinese server message never reaches the main line
  assert.equal(errorCopy(disk).line, DISK_LINE);
  assert.equal(errorCopy(disk).action, '稍后再试');
  assert.deepEqual(UPLOAD_LIMIT_COPY.quota_exceeded, { line: '存储空间已用完。删掉不用的文件后再上传。', action: '管理文件' });
  // A plain 413 (body too large) keeps the approved generic copy.
  assert.equal(errorCopy(new TaskRequestError('x', 413, 'body_too_large')).line, ERROR_COPY[413].line);
});

test('t238 C7a D2: the complete result carries trainability onto the uploaded file', async (context) => {
  const { uploadUserFile, untrainable, TRAINABILITY_NOTICE } = require('./upload-attachment.ts');
  assert.equal(TRAINABILITY_NOTICE, NOTICE);
  const calls = [];
  context.mock.method(globalThis, 'fetch', async (path) => {
    calls.push(String(path));
    if (String(path) === '/api/assistant/uploads') return Response.json({ id: 'upload_1', chunk_size: 4 });
    if (String(path).includes('/chunks/')) return Response.json({ replayed: false });
    return Response.json({ attachment: { id: 'attachment_1', name: 'big.npz', bytes: 5 },
      trainability: { trainable: false, reasons: ['input_bytes_exceeded'], limits: { max_files: 256, max_bytes: 268435456 }, bytes: 5, files: 1 } });
  });
  const file = await uploadUserFile(new File(['12345'], 'big.npz'), new AbortController().signal);
  assert.equal(file.id, 'attachment_1');
  assert.equal(untrainable(file), true);
  assert.equal(untrainable({}), false);
  assert.deepEqual(calls.slice(-1), ['/api/assistant/uploads/upload_1/complete']);
  for (const source of ['TaskFiles.tsx', 'ComposerAttach.tsx']) {
    assert.match(readFileSync(new URL('./' + source, import.meta.url), 'utf8'), /untrainable\([^)]*\)[^\n]*trainabilityNotice\(/, source);
  }
  assert.match(readFileSync(new URL('./ProjectPage.tsx', import.meta.url), 'utf8'), /message \?\? '上传失败'/);
});

test('t238 C7a U3: the training input-bound error reads as a Chinese status and line', () => {
  const { blockedProgressCopy, blockedAssistantLine } = require('./task-presentation.ts');
  for (const message of ['ValueError: input directory exceeds the bounded byte limit', 'input directory exceeds the bounded file count']) {
    const copy = blockedProgressCopy({ status: 'failed', error_message: message });
    assert.deepEqual(copy, { status: '超过训练上限', action: '检查并继续' });
    assert.equal(blockedAssistantLine(copy.status), '训练数据超过训练上限（256\u00a0MiB / 256\u00a0个文件）。先缩小或预处理，再继续。');
  }
  assert.equal(blockedProgressCopy({ status: 'failed', error_message: 'other' }).status, '这次没有完成');
});

test('t238 C7a-fix m4: upload_completing (409) reads as the Chinese merge-in-progress line', async (context) => {
  const { taskRequest } = require('./types.ts');
  const { errorCopy, UPLOAD_LIMIT_COPY } = require('./error-copy.ts');
  const LINE = '文件还在处理，稍后点「刷新」。';
  assert.deepEqual(UPLOAD_LIMIT_COPY.upload_completing, { line: LINE, action: '刷新' });
  const fetch = context.mock.method(globalThis, 'fetch', async () => Response.json({ error: 'upload_completing', message: LINE }, { status: 409 }));
  const busy = await taskRequest('/api/assistant/uploads/upload_1/complete').catch(error => error);
  assert.deepEqual([busy.status, busy.code, busy.message], [409, 'upload_completing', LINE]);
  assert.deepEqual(errorCopy(busy), { key: '409', line: LINE, action: '刷新', details: ['HTTP 409 · upload_completing'] });
  fetch.mock.mockImplementation(async () => Response.json({ error: 'upload_completing' }, { status: 409 }));
  const bare = await taskRequest('/api/assistant/uploads/upload_1/complete').catch(error => error);
  assert.equal(bare.message, LINE); // no server message: the fixed Chinese line, never the code
  assert.doesNotMatch(errorCopy(bare).line, /[A-Za-z]/);
});
