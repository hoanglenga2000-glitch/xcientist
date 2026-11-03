// t238 C7c-7: Design softs from the C7b review (quota two cases + nowrap units, delete dialog, spacing, 393 row buttons, copy).
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
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX }, fileName: filename }).outputText, filename);
};
Module._extensions['.tsx'] = Module._extensions['.ts'];
const read = (name) => readFileSync(new URL('./' + name, import.meta.url), 'utf8');
const css = () => read('workspace.css');
const NBSP = '\u00a0';

test('C7c-7a: quota copy has two cases, each at its limit, and number + unit are joined by a no-break space', () => {
  const { errorCopy } = require('./error-copy.ts');
  const { TaskRequestError } = require('./types.ts');
  const capacity = `存储空间已用完（已用 5${NBSP}GiB，上限 5${NBSP}GiB）。删掉不用的文件后再上传。`;
  const files = `文件数已达上限（已用 500${NBSP}个，上限 500${NBSP}个）。删掉不用的文件后再上传。`;
  for (const line of [capacity, files]) {
    const copy = errorCopy(new TaskRequestError(line, 413, 'quota_exceeded'));
    assert.equal(copy.line, line);
    assert.equal(copy.action, '管理文件');
    assert.doesNotMatch(line, / GiB| 个/); // never "5 / GiB" at 393
  }
  const backend = readFileSync(new URL('../../../../../../src/evomind_runtime/assistant_runs.py', import.meta.url), 'utf8');
  assert.match(backend, /文件数已达上限（已用 \{used_files\}\{_NBSP\}个，上限 \{max_files\}\{_NBSP\}个）/);
  assert.match(backend, /存储空间已用完（已用 \{_format_bytes\(used_bytes\)\}，上限 \{_format_bytes\(max_bytes\)\}）/);
});

test('C7c-7b: delete dialog — bold title alone, normal-weight body with the name on its own line, centred, dark overlay', () => {
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const { DeleteFileDialog } = require('./DeleteFileDialog.tsx');
  const { deleteDialogState } = require('./file-delete.ts');
  const long = 'kxxz_' + 'a'.repeat(160) + '.txt';
  const html = renderToStaticMarkup(React.createElement(DeleteFileDialog, { state: deleteDialogState(long, []), busy: false, onCancel() {}, onConfirm() {} }));
  assert.match(html, /<h2 id="tw-delete-file-title">删除这个文件？<\/h2>/);
  assert.match(html, new RegExp(`<div id="tw-delete-file-body" class="tw-dialog-body"><p class="tw-dialog-name">${long.replace('.', '\\.')}</p><p id="tw-delete-file-line">删除后不能恢复。</p></div>`));
  assert.match(html, /aria-describedby="tw-delete-file-body"/);
  const style = css();
  assert.match(style, /\.tw-dialog \.tw-dialog-name\{[^}]*font-weight:400;overflow-wrap:anywhere/);
  assert.match(style, /\.tw-dialog \.tw-dialog-body p\{[^}]*font-weight:400/);
  // C7c-A3 #3: the overlay is a token — ≥ .5 in light, darker (.72) in dark
  assert.match(style, /\.tw-dialog-backdrop\{background:var\(--tw-overlay\)\}/);
  assert.match(style.split('\n')[0], /--tw-overlay:rgba\(0,0,0,\.5\)/);
  assert.match(style, /\n--tw-danger-fill:#B42318;--tw-danger-fill-hover:#C8352A;--tw-on-danger:#FFFFFF;--tw-overlay:rgba\(0,0,0,\.72\);\n\}/);
  assert.match(style, /\.tw-dialog-backdrop\{position:fixed;inset:0;display:flex;align-items:center;justify-content:center\}/);
  // the C7c rules come after the C7b ones, so they win in both themes (no dark override of the backdrop exists)
  assert.ok(style.lastIndexOf('background:var(--tw-overlay)') > style.indexOf('rgba(31,41,51,.24)'));
  assert.doesNotMatch(style, /\[data-theme="dark"\][^{]*tw-dialog-backdrop/);
});

test('C7c-7c: error blocks — 8px text→button, tighter button→技术详情; 12px under 还没有资料。', () => {
  const style = css();
  assert.match(style, /\.tw-error>p\{margin:0 0 8px\}/);
  assert.match(style, /\.tw-error>\.tw-details\{margin-top:4px\}/);
  assert.match(style, /\.tw-empty-files\{margin:0 0 12px\}/);
  assert.match(read('TaskFiles.tsx'), /<p className="tw-muted tw-empty-files">还没有资料。<\/p>/);
  assert.match(read('ErrorNotice.tsx'), /<p>\{copy\.line\}<\/p>\{action\}/);
});

test('C7c-7d: at 393 the 删除 and 下载资料 buttons in a file row both get min-height 44px', () => {
  const files = read('TaskFiles.tsx');
  assert.match(files, /label="下载资料" className="tw-button tw-nowrap tw-row-action"/);
  assert.match(files, /tw-text-danger tw-nowrap tw-row-action/);
  assert.match(css(), /@media\(max-width:760px\)\{\.task-workspace \.tw-file-actions \.tw-row-action\{min-height:44px;box-sizing:border-box\}\}/);
  assert.match(read('DownloadFile.tsx'), /className = 'tw-button'/);
});

test('C7c-7e: minor copy — export busy, one in-use wording in both places, in-use confirm line', () => {
  const { UPLOAD_LIMIT_COPY } = require('./error-copy.ts');
  const { FILE_DELETE_COPY } = require('./file-delete.ts');
  assert.equal(UPLOAD_LIMIT_COPY.export_busy.line, '现在导出的人较多，请稍后再试。');
  assert.equal(FILE_DELETE_COPY.activeRefusal('X'), '任务「X」正在运行，结束后才能删除这个文件。');
  assert.equal(FILE_DELETE_COPY.inUse('X'), '这个文件正在被任务「X」使用。删除后，这个任务就读不到它了。');
  const backend = readFileSync(new URL('../../../../../../src/evomind_runtime/user_data.py', import.meta.url), 'utf8');
  assert.match(backend, /ATTACHMENT_IN_USE_MESSAGE = "任务「\{task\}」正在运行，结束后才能删除这个文件。"/);
  assert.match(backend, /EXPORT_BUSY_MESSAGE = "现在导出的人较多，请稍后再试。"/);
  for (const line of [UPLOAD_LIMIT_COPY.export_busy.line, UPLOAD_LIMIT_COPY.attachment_in_use.line, FILE_DELETE_COPY.inUse('X')]) {
    assert.doesNotMatch(line, /您|请求|该任务/);
  }
});
