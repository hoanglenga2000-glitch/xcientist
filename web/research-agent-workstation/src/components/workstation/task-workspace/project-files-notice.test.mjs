import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const require = createRequire(import.meta.url);
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../../' + name.slice(2), import.meta.url));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const h = React.createElement;
const PID = 'project_' + 'e'.repeat(32);
const files = [{ name: '数据说明.pdf', error_class: 'file_unavailable' }, { name: 'labels.csv', error_class: 'file_unavailable' }];
const run = extra => ({ id: 'run_' + 'd'.repeat(32), status: 'completed', prompt: '根据项目文件回答', answer: '这是回答。', ...extra });

test('t237 C3.10: project-file notice kind — none, partial, total', () => {
  const { projectFilesNotice } = require('./ProjectFilesNotice.tsx');
  assert.equal(projectFilesNotice(run({})), null);
  assert.equal(projectFilesNotice(run({ project_files_unreadable: [] })), null);
  assert.deepEqual(projectFilesNotice(run({ project_files_unreadable: files })), { kind: 'partial', files });
  assert.deepEqual(projectFilesNotice(run({ project_files_unreadable: files, project_files_total: 2 })), { kind: 'total', files });
  assert.deepEqual(projectFilesNotice(run({ project_files_unreadable: files, project_files_total: 5 })), { kind: 'partial', files });
  assert.deepEqual(projectFilesNotice(run({ project_files_failed: true })), { kind: 'total', files: [] });
});

test('t237 C3.10: partial is a muted line + 查看项目文件; total is .tw-error + 重试; both keep 技术详情 collapsed', () => {
  const { ProjectFilesNotice } = require('./ProjectFilesNotice.tsx');
  const partial = renderToStaticMarkup(h(ProjectFilesNotice, { notice: { kind: 'partial', files }, projectId: PID }));
  assert.equal(partial, `<div class="tw-project-files-notice"><p class="tw-muted">有 2 个项目文件没能读取，这次回答没有用到。</p><a href="/workspace?project=${PID}">查看项目文件</a>`
    + '<details class="tw-details"><summary>技术详情</summary><p>数据说明.pdf · file_unavailable</p><p>labels.csv · file_unavailable</p></details></div>');
  const total = renderToStaticMarkup(h(ProjectFilesNotice, { notice: { kind: 'total', files: files.slice(0, 1) }, projectId: PID }));
  assert.equal(total, '<div role="alert" class="tw-error tw-project-files-notice"><p>项目文件暂时读不了，这次回答没有用到。</p><button type="button">重试</button>'
    + '<details class="tw-details"><summary>技术详情</summary><p>数据说明.pdf · file_unavailable</p></details></div>');
  assert.doesNotMatch(partial + total, /<details[^>]*open/);
  // Without a project id the link has nowhere to go and is left out.
  assert.doesNotMatch(renderToStaticMarkup(h(ProjectFilesNotice, { notice: { kind: 'partial', files } })), /查看项目文件/);
});

test('t237 C3.10: the notice sits under the assistant answer; no project-file chip in the composer', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run({ project_files_unreadable: files }), projectId: PID }));
  assert.match(html, /这是回答。[^]*?<\/article><div class="tw-project-files-notice"><p class="tw-muted">有 2 个项目文件没能读取/);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run({}) })), /tw-project-files-notice/);
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, /'project_files_unreadable'/, 'the live event is merged into the run');
  assert.match(session, /projectId=\{task\.project_id/);
  for (const file of ['ComposerAttach.tsx', 'ComposerPicker.tsx']) assert.doesNotMatch(readFileSync(new URL(`./${file}`, import.meta.url), 'utf8'), /项目文件/, file);
});
