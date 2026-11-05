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

// t238 C7c ADDENDUM 3/4 (Critiquito authoritative order): composer softs.

test('A3 #2: 「管理文件」 only in the composer quota-full error, new tab (draft and attachments stay)', () => {
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const { AttachmentChips, MANAGE_FILES } = require('./ComposerAttach.tsx');
  const chips = (errorCode, error = '存储空间已用完（已用 5 GiB，上限 5 GiB）。删掉不用的文件后再上传。', status = '') =>
    renderToStaticMarkup(React.createElement(AttachmentChips, { files: [], error, status, errorCode, onRemove() {} }));
  const quota = chips('quota_exceeded');
  assert.match(quota, /<span class="tw-chip tw-chip-error" role="alert">存储空间已用完[^<]*<a class="tw-chip-action tw-nowrap" href="\/workspace\?section=data&amp;focus=files" target="_blank" rel="noopener">管理文件<\/a><\/span>/);
  assert.equal(MANAGE_FILES.href, '/workspace?section=data&focus=files');
  for (const code of ['disk_low_water', 'upload_completing', '']) assert.doesNotMatch(chips(code, '暂时无法上传。'), /管理文件/);
  const hint = renderToStaticMarkup(React.createElement(AttachmentChips, { files: [], status: '文件超过训练上限（256 MiB）。', onRemove() {} }));
  assert.doesNotMatch(hint, /管理文件/);
  const css = read('workspace.css');
  assert.match(css, /@media\(max-width:760px\)\{\.task-workspace \.tw-chip-error \.tw-chip-action\{flex-basis:100%;[^}]*min-height:44px\}\}/);
  assert.match(css, /\.task-workspace \.tw-chip-error \.tw-chip-action\{[^}]*min-height:36px/);
});

test('A3 #2: landing on the data page scrolls to and focuses the file list (reduced-motion safe), project link too', () => {
  const files = read('TaskFiles.tsx');
  assert.match(files, /new URLSearchParams\(window\.location\.search\)\.get\('focus'\) !== MANAGE_FILES_FOCUS/);
  assert.match(files, /prefers-reduced-motion: reduce/);
  assert.match(files, /behavior: reduce \? 'auto' : 'smooth'/);
  assert.match(files, /fileList\.current\?\.focus\(\{ preventScroll: true \}\)/);
  assert.match(files, /<div ref=\{fileList\} tabIndex=\{-1\} className="tw-file-list" aria-label=/);
  assert.match(read('ProjectPage.tsx'), /<a className="tw-button tw-nowrap" href=\{MANAGE_FILES\.href\}>/);
  assert.match(read('TaskWorkspace.tsx'), /const library = section === 'data' \|\| section === 'library';/);
});

test('A3 #6: single-file training hint does not repeat the file name the chip already shows', () => {
  const attach = read('ComposerAttach.tsx');
  assert.match(attach, /advisory = files\.length > 1 \? `\$\{uploaded\.name\}：\$\{trainabilityNotice\(uploaded\)\}` : trainabilityNotice\(uploaded\);/);
});

test('A3 #3: running-task refusal — title 「现在不能删除「xxx」」, no 不能恢复, single 知道了, darker dark overlay', () => {
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const { DeleteFileDialog } = require('./DeleteFileDialog.tsx');
  const { deleteDialogState } = require('./file-delete.ts');
  const state = deleteDialogState('报告.txt', [{ task_id: 'u', task_title: '房价预测', run_id: 'r', active: true }]);
  const html = renderToStaticMarkup(React.createElement(DeleteFileDialog, { state, busy: false, onCancel() {}, onConfirm() {} }));
  assert.match(html, /<div class="tw-dialog is-refusal" role="alertdialog"/);
  assert.match(html, /<h2 id="tw-delete-file-title">现在不能删除<span class="tw-dialog-title-name">「报告\.txt」<\/span><\/h2>/);
  assert.equal(state.title, '现在不能删除「报告.txt」'); // exact title text (accessible name)
  assert.match(read('workspace.css'), /\.tw-dialog\.is-refusal \.tw-dialog-title-name\{display:block\}/);
  assert.match(html, /<p id="tw-delete-file-line">任务「房价预测」正在运行，结束后才能删除这个文件。<\/p>/);
  assert.doesNotMatch(html, /不能恢复|删除这个文件？/);
  assert.equal((html.match(/<button/g) ?? []).length, 1);
  assert.match(html, />知道了<\/button>/);
  const css = read('workspace.css');
  assert.match(css, /--tw-overlay:rgba\(0,0,0,\.72\);/);
  assert.match(css, /\.tw-dialog-backdrop\{background:var\(--tw-overlay\)\}/);
  assert.match(css, /\.tw-dialog\.is-refusal h2\{text-align:center;overflow-wrap:anywhere/);
});

test('A3 #3: dark danger token — dark red fill, light text, hover and focus visible', () => {
  const css = read('workspace.css');
  assert.match(css, /\n--tw-danger-fill:#B42318;--tw-danger-fill-hover:#C8352A;--tw-on-danger:#FFFFFF;/); // dark token block
  assert.match(css.split('\n')[0], /--tw-danger-fill:#B42318;--tw-danger-fill-hover:#912018;--tw-on-danger:#FFFFFF;/);
  assert.match(css, /\.task-workspace \.tw-danger-solid,\.task-workspace \.tw-dialog-actions \.tw-button\.tw-danger-solid\{background:var\(--tw-danger-fill\);border-color:var\(--tw-danger-fill\);color:var\(--tw-on-danger\)\}/);
  assert.match(css, /\.tw-danger-solid:not\(:disabled\):hover[^{]*\{background:var\(--tw-danger-fill-hover\)/);
  assert.match(css, /\.task-workspace \.tw-danger-solid:focus-visible\{outline:2px solid var\(--tw-on-danger\)/);
});

test('A3 #4: export is fetch → blob → save with 导出中… and in-app busy / error lines', () => {
  const { EXPORT_COPY, exportErrorLine, exportFileName } = require('./file-delete.ts');
  assert.equal(EXPORT_COPY.pending, '导出中…');
  assert.equal(exportErrorLine(429, '现在导出的人较多，请稍后再试。'), '现在导出的人较多，请稍后再试。');
  assert.equal(exportErrorLine(429, 'busy'), '现在导出的人较多，请稍后再试。');
  assert.equal(exportErrorLine(401), '登录已过期，请重新登录后再导出。');
  assert.equal(exportErrorLine(403), '你没有权限导出这些资料。');
  assert.equal(exportErrorLine(503), '导出没有完成，请稍后再试。');
  assert.equal(exportFileName('attachment; filename="evomind-export-20261101.zip"'), 'evomind-export-20261101.zip');
  assert.equal(exportFileName("attachment; filename*=UTF-8''%E6%88%91%E7%9A%84.zip"), '我的.zip');
  assert.equal(exportFileName('attachment; filename="../../x.zip"'), 'x.zip');
  assert.equal(exportFileName(null, new Date('2026-11-01T00:00:00Z')), 'evomind-export-20261101.zip');
  const component = read('ExportData.tsx');
  assert.match(component, /fetch\(TENANT_EXPORT_URL/);
  assert.match(component, /response\.blob\(\)/);
  assert.match(component, /exportFileName\(response\.headers\.get\('content-disposition'\)\)/);
  assert.match(component, /disabled=\{busy\} aria-busy=\{busy\}/);
  assert.match(component, /\{busy \? EXPORT_COPY\.pending : FILE_DELETE_COPY\.export\}/);
  assert.match(component, /<p role="alert" className="tw-error tw-export-error">/);
  assert.doesNotMatch(read('TaskFiles.tsx'), /<a [^>]*TENANT_EXPORT_URL[^>]*download/);
});

test('A3 #5: one height per breakpoint for row actions and dialog buttons (36 @1440, 44 @393)', () => {
  const css = read('workspace.css');
  assert.match(css, /\.task-workspace \.tw-row-action,\.task-workspace \.tw-dialog-actions \.tw-button,\.task-workspace \.tw-dialog-actions>button\{box-sizing:border-box;height:36px;min-height:36px/);
  assert.match(css, /@media\(max-width:760px\)\{\.task-workspace \.tw-row-action,\.task-workspace \.tw-dialog-actions \.tw-button,\.task-workspace \.tw-dialog-actions>button\{height:44px;min-height:44px\}\}/);
  const files = read('TaskFiles.tsx');
  assert.match(files, /label="下载资料" className="tw-button tw-nowrap tw-row-action"/);
  assert.match(files, /tw-text-danger tw-nowrap tw-row-action/);
  assert.match(read('ExportData.tsx'), /className="tw-button tw-nowrap tw-row-action"/);
});

test('A3 #6: 「超过训练上限」 everywhere, no split 「256 MiB」/「5 GiB」, hint contrast ≥ 4.5, spacing', () => {
  const { TRAINING_INPUT_LIMIT_LINE, TRAINING_INPUT_LIMIT_STATUS } = require('./error-copy.ts');
  const { TRAINABILITY_NOTICE, TRAINABILITY_FILES_NOTICE } = require('./upload-attachment.ts');
  assert.equal(TRAINING_INPUT_LIMIT_STATUS, '超过训练上限');
  for (const line of [TRAINING_INPUT_LIMIT_LINE, TRAINABILITY_NOTICE, TRAINABILITY_FILES_NOTICE]) {
    assert.match(line, /超过训练上限/);
    assert.doesNotMatch(line, /256 MiB|256 个|训练输入超出上限/);
  }
  const css = read('workspace.css');
  assert.match(css, /\.task-workspace \.tw-chip-status\{color:var\(--tw-secondary\)\}/);
  assert.match(css, /\.tw-file-hint,\.tw-chip-error,\.tw-chip-status,\.tw-error>p\{text-wrap:pretty\}/);
  assert.match(css, /\.tw-upload-feedback\{margin-bottom:12px\}/);
  // contrast of the light hint colour on the chip wash, computed (WCAG relative luminance)
  const lum = hex => { const c = [0, 2, 4].map(i => parseInt(hex.slice(i + 1, i + 3), 16) / 255).map(x => x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4); return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const secondary = css.match(/--tw-secondary:(#[0-9A-Fa-f]{6})/)[1];
  const wash = css.match(/--tw-wash:(#[0-9A-Fa-f]{6})/)[1];
  assert.ok(ratio(secondary, wash) >= 4.5, `${secondary} on ${wash}`);
  const backend = readFileSync(new URL('../../../../../../src/evomind_runtime/assistant_runs.py', import.meta.url), 'utf8');
  assert.match(backend, /文件超过训练上限（256\\u00a0MiB）/);
});

// t238 C7c FIX1 H5 + Minors 1, 8.
test('FIX1 a delete with a copy still being cleaned shows the server line, never 「已删除」', async () => {
  const { deleteResultLine, FILE_DELETE_COPY } = await import('./file-delete.ts');
  const line = '文件已删除，但训练时留下的副本还没清理完，系统会继续重试。';
  assert.equal(deleteResultLine('a.csv', { status: 'partial', message: line }), line);
  assert.equal(deleteResultLine('a.csv', { status: 'deleted' }), FILE_DELETE_COPY.deleted('a.csv'));
  assert.equal(deleteResultLine('a.csv', null), FILE_DELETE_COPY.deleted('a.csv'));
  assert.match(read('TaskFiles.tsx'), /setNotice\(deleteResultLine\(file\.name, result\)\)/);
});

test('FIX1 every declaration in the dark token block is terminated (parsed, not pattern-matched)', () => {
  const css = read('workspace.css');
  const blocks = [...css.matchAll(/([^{}]*)\{([^{}]*--tw-[^{}]*)\}/g)];
  assert.ok(blocks.length >= 2);
  for (const [, selector, body] of blocks) {
    for (const declaration of body.split(';')) {
      const text = declaration.replace(/\/\*.*?\*\//gs, '').trim();
      if (!text) continue;
      const custom = text.match(/--[\w-]+\s*:/g) ?? [];
      assert.ok(custom.length <= 1, `${selector.trim()}: unterminated declaration near ${text.slice(0, 60)}`);
    }
  }
  const dark = blocks.find(([, , body]) => /--tw-scrollbar:#3A3F44/.test(body));
  const tokens = Object.fromEntries(dark[2].split(';').map(item => item.split(':').map(part => part.trim())).filter(pair => pair.length === 2));
  assert.equal(tokens['--tw-scrollbar'], '#3A3F44');
  assert.equal(tokens['--tw-danger-fill'], '#B42318');
});

test('FIX1 the file list shows a focus ring when 管理文件 lands on it', () => {
  const css = read('workspace.css');
  assert.doesNotMatch(css, /\.tw-file-list:focus\{outline:none\}/);
  assert.match(css, /\.tw-file-list:focus\{outline:2px solid var\(--tw-focus-ring\)/);
});
