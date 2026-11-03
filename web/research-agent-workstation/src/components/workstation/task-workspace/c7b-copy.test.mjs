// t238 C7b COPY + S1/S2/S3: Critiquito strings, no 重试 for quota/disk, no wrapping buttons, visible 资料 errors.
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
Module._extensions['.ts'] = Module._extensions['.tsx'] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX }, fileName: filename }).outputText, filename);
};
const read = (name) => readFileSync(new URL('./' + name, import.meta.url), 'utf8');
const NEW_COPY = {
  quota: '存储空间已用完。删掉不用的文件后再上传。',
  disk: '暂时无法上传。请稍后再试，或联系管理员。',
  completing: '文件还在处理，稍后点「刷新」。',
  singleBytes: '文件超过训练上限（256\u00a0MiB）。可以直接对话和分析；要训练，先缩小或预处理。',
  singleFiles: '文件数超过训练上限（256\u00a0个）。可以直接对话和分析；要训练，先缩小或预处理。',
  aggregate: '训练数据超过训练上限（256\u00a0MiB / 256\u00a0个文件）。先缩小或预处理，再继续。',
  label: '超过训练上限',
};

test('t238 C7b COPY: the Critiquito strings and actions, 你 not 您, no 配额/服务器/校验合并', () => {
  const { UPLOAD_LIMIT_COPY, TRAINING_INPUT_LIMIT_LINE, TRAINING_INPUT_LIMIT_STATUS } = require('./error-copy.ts');
  const { TRAINABILITY_NOTICE, TRAINABILITY_FILES_NOTICE, trainabilityNotice } = require('./upload-attachment.ts');
  assert.deepEqual(UPLOAD_LIMIT_COPY.quota_exceeded, { line: NEW_COPY.quota, action: '管理文件' });
  assert.deepEqual(UPLOAD_LIMIT_COPY.disk_low_water, { line: NEW_COPY.disk, action: '稍后再试' });
  assert.deepEqual(UPLOAD_LIMIT_COPY.upload_completing, { line: NEW_COPY.completing, action: '刷新' });
  assert.equal(TRAINABILITY_NOTICE, NEW_COPY.singleBytes);
  assert.equal(TRAINABILITY_FILES_NOTICE, NEW_COPY.singleFiles);
  assert.equal(trainabilityNotice({ trainability: { reasons: ['input_bytes_exceeded'] } }), NEW_COPY.singleBytes);
  assert.equal(trainabilityNotice({ trainability: { reasons: ['input_files_exceeded'] } }), NEW_COPY.singleFiles);
  assert.doesNotMatch(TRAINABILITY_NOTICE, /个文件/);
  assert.equal(TRAINING_INPUT_LIMIT_LINE, NEW_COPY.aggregate);
  assert.equal(TRAINING_INPUT_LIMIT_STATUS, NEW_COPY.label);
  for (const line of [...Object.values(NEW_COPY), ...Object.values(UPLOAD_LIMIT_COPY).map(item => item.line)]) {
    assert.doesNotMatch(line, /您|配额|服务器|校验合并|[A-Za-z_]{4,}|\b4\d\d\b/, line);
    for (const sentence of line.split('。').filter(Boolean)) assert.ok(sentence.length <= 40, sentence); // ≤ 2 lines at 393
  }
  for (const name of ['error-copy.ts', 'upload-attachment.ts', 'file-delete.ts', 'TaskFiles.tsx', 'ProjectPage.tsx', 'DeleteFileDialog.tsx', 'upload-failure.ts']) {
    assert.doesNotMatch(read(name), /您/, name);
  }
});

test('t238 C7b S1: quota → 管理文件, disk → 稍后再试, never 重试 for either; 409 keeps 刷新', () => {
  const { uploadFailureAction } = require('./upload-failure.ts');
  const { errorCopy } = require('./error-copy.ts');
  const { TaskRequestError } = require('./types.ts');
  assert.equal(uploadFailureAction('quota_exceeded'), '管理文件');
  assert.equal(uploadFailureAction('disk_low_water'), '稍后再试');
  assert.equal(uploadFailureAction('upload_completing'), '刷新');
  assert.equal(uploadFailureAction(undefined), '重试');
  for (const code of ['quota_exceeded', 'disk_low_water']) assert.notEqual(errorCopy(new TaskRequestError('x', 413, code)).action, '重试');
  assert.equal(errorCopy(new TaskRequestError('x', 409, 'upload_completing')).action, '刷新');
});

test('t238 C7b S1/S2: the project row renders the right action, never 重试 on 413, and its button does not wrap', () => {
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const { ProjectFileRow } = require('./ProjectPage.tsx');
  const row = (code, message) => renderToStaticMarkup(React.createElement(ProjectFileRow, { upload: { name: 'a.csv', state: 'failed', message, code }, onRetry() {} }));
  const quota = row('quota_exceeded', '文件数已达上限（已用 500 个，上限 500 个）。删掉不用的文件后再上传。'); // t238 C7c-7a
  // t238 C7c-A3 #2: lands on the data page and focuses the file list
  assert.match(quota, /<a class="tw-button tw-nowrap" href="\/workspace\?section=data&amp;focus=files">管理文件<\/a>/);
  assert.doesNotMatch(quota, /重试|quota_exceeded|413/);
  const disk = row('disk_low_water', NEW_COPY.disk);
  assert.match(disk, /<button type="button" class="tw-nowrap">稍后再试<\/button>/);
  assert.doesNotMatch(disk, /重试/);
  const css = read('workspace.css');
  assert.match(css, /\.tw-nowrap\{white-space:nowrap;flex:none\}/);
  assert.match(css, /\.tw-row-error\{display:flex;flex-wrap:wrap;[^}]*font-size:13px/);
  assert.match(css, /\.tw-row-error-text\{flex:1 1 180px;min-width:0/);
});

test('t238 C7b S3: 资料 errors render under 上传资料 above the lists, clear the stale progress, and scroll into view', () => {
  const files = read('TaskFiles.tsx');
  const upload = files.indexOf('上传资料<input');
  const feedback = files.indexOf('className="tw-upload-feedback"');
  const list = files.indexOf('className="tw-file-list"');
  assert.ok(upload > 0 && feedback > upload && list > feedback, 'error block sits between 上传资料 and the file lists');
  assert.match(files, /setNotice\(''\); setError\(cause\);/);
  assert.match(files, /scrollIntoView\?\.\(\{ block: 'nearest' \}\)/);
  assert.match(files, /action === '管理文件' \? focusFileList\(\)/);
  assert.match(read('ErrorNotice.tsx'), /role="alert"/);
});

test('t238 C7b COPY #5: the training hint is on the file row, not above the 文献 heading', () => {
  const files = read('TaskFiles.tsx');
  assert.match(files, /<h3>\{file\.name\}<\/h3>.*\{hints\[file\.id\] && <p className="tw-file-hint">\{hints\[file\.id\]\}<\/p>\}/);
  assert.doesNotMatch(files, /`\$\{saved\}\$\{TRAINABILITY_NOTICE\}`/);
  assert.match(read('workspace.css'), /\.tw-file-hint\{margin:4px 0 0;color:var\(--tw-secondary\);font-size:13px/);
});

test('t238 C7b COPY #10: new answers say 来自模型输出; the legacy prefix is still hidden', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  assert.equal(visibleMessageText('**本轮实际模型**：`m`（来自模型输出）\n\n正文'), '正文');
  assert.equal(visibleMessageText('**本轮实际模型**：`m`（来自受管 model_observed 事件）\n\n正文'), '正文');
});
