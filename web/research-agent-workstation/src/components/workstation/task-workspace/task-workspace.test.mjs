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

test('login cannot send a native GET containing credentials before hydration', () => {
  const { default: LoginPage } = require('../../../app/login/page.tsx');
  const html = renderToStaticMarkup(React.createElement(LoginPage));
  assert.match(html, /<form[^>]*method="post"/);
  assert.match(html, /<button[^>]*type="submit"[^>]*disabled/);
});

test('my tasks explains how to begin and links to the exact saved task', () => {
  const { TaskList } = require('./TaskList.tsx');
  const empty = renderToStaticMarkup(React.createElement(TaskList, { tasks: [], query: '' }));
  assert.match(empty, /还没有任务/);
  assert.match(empty, /新任务/);
  const populated = renderToStaticMarkup(React.createElement(TaskList, { query: '', tasks: [{
    id: 'utask_123', title: '我的阅读任务', status: 'draft', updated_at: '2026-10-08T00:00:00Z',
  }] }));
  assert.match(populated, /我的阅读任务/);
  assert.match(populated, /尚未开始/);
  assert.match(populated, /\/workspace\?task=utask_123/);
  assert.doesNotMatch(populated, /sample|Stage 1|Gate 1|Planner|质量已通过/);
});

test('task tabs show real conversation, empty progress and exact run artifacts separately from quality', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const empty = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'progress', run: null }));
  assert.match(empty, /尚未开始/);
  assert.doesNotMatch(empty, /Planner|Stage|Gate|运行中/);
  const run = { id: 'run_owned_fixture', status: 'completed', prompt: '我的问题', answer: '这是当前运行的回复',
    artifacts: [{ id: 'artifact_owned_fixture', run_id: 'run_owned_fixture', name: '本次结果.md', bytes: 12, sha256: 'a'.repeat(64), verified_at: '', preview_kind: 'text' }] };
  const conversation = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run }));
  assert.match(conversation, /这是当前运行的回复/);
  const result = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'results', run }));
  assert.match(result, /本次结果.md/);
  assert.match(result, /\/api\/assistant\/artifacts\/artifact_owned_fixture/);
  assert.match(result, /不代表科研结论/);
  assert.doesNotMatch(result, /质量已通过|官方结果已确认/);
});

test('unavailable model explains the next user action and retains technical evidence separately', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_fixture', user_task_id: 'utask_fixture', status: 'blocked', error_class: 'model_unavailable',
    answer: 'Runtime session is durable. Model execution is unavailable: LLMError/RuntimeError. No tool was executed by the failed model request.',
  } }));
  assert.match(html, /模型暂时不可用/);
  assert.match(html, /检查模型配置/);
  assert.match(html, /<details/);
});

test('model settings clearly separates save, connection test and actual use', () => {
  const { ModelSettings } = require('./ModelSettings.tsx');
  const html = renderToStaticMarkup(React.createElement(ModelSettings, { returnTask: 'utask_fixture', isolated: true }));
  assert.match(html, /API 密钥/); assert.match(html, /加密保存/);
  assert.match(html, /保存不会发出请求/); assert.match(html, /隔离验收环境禁止外部模型请求/);
  assert.match(html, /返回原任务/); assert.doesNotMatch(html, /localStorage|value="sk-/);
});

test('product navigation uses one workspace and personal settings never send users to legacy global pages', () => {
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskWorkspace));
  for (const label of ['新任务', '我的任务', '资料与数据', '文献检索', '成果', '设置']) assert.ok(html.includes(label));
  assert.doesNotMatch(html, /href="\/\?page=/);
  const { ModelSettings } = require('./ModelSettings.tsx');
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(ModelSettings, { returnTask: '', isolated: true })), /打开原设置|href="\/\?page=/);
});

test('settings offers three simple groups and resources never invent a personal budget', () => {
  const { WorkspaceSettings } = require('./WorkspaceSettings.tsx');
  const { ThemeProvider } = require('../theme/ThemeProvider.tsx');
  const html = renderToStaticMarkup(React.createElement(ThemeProvider, null,
    React.createElement(WorkspaceSettings, { group: 'resources', returnTask: 'utask_fixture', isolated: true })));
  for (const label of ['模型', '计算资源与额度', '账户与外观', '额度由管理员管理', '返回原任务']) assert.ok(html.includes(label), label);
  assert.doesNotMatch(html, /保存预算|剩余额度|治理校验|打开原设置/);
});

test('legacy links retain identities without guessing a task and map settings to one shell', () => {
  const { workspaceLocation } = require('./workspace-location.ts');
  assert.equal(workspaceLocation(new URLSearchParams('page=gpu')), '/workspace?settings=resources');
  const url = workspaceLocation(new URLSearchParams('page=workflow&task=old-task&run=run_original'));
  assert.match(url, /legacy_task=old-task/); assert.match(url, /legacy_run=run_original/);
  assert.doesNotMatch(url, /[?&]task=/);
});

test('a pending confirmation explains target, refusal, and never starts pre-approved', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: {
    id: 'run_fixture', status: 'waiting_approval', approvals: [{ id: 'approval_fixture', status: 'pending',
      tool_name: 'file_delete', reversible: false, risk_level: 'high', normalized_arguments: { path: 'outputs/old.txt' }, impact_scope: {} }],
  } }));
  assert.match(html, /待你确认/); assert.match(html, /old.txt/); assert.match(html, /拒绝后不会执行/);
  assert.match(html, /确认执行/); assert.doesNotMatch(html, /checked=""/);
});

test('confirmation presents a task-relative file while preserving the exact target in details', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: {
    id: 'run_fixture', status: 'paused', approvals: [{ id: 'approval_fixture', status: 'pending',
      tool_name: 'file_delete', reversible: false, risk_level: 'high',
      normalized_arguments: { path: 'D:\\private-runtime\\run_fixture\\work\\old.txt' },
      impact_scope: { workspace_root: 'D:\\private-runtime\\run_fixture' } }],
  } }));
  assert.match(html, /目标：本次运行的 work\/old\.txt/);
  assert.doesNotMatch(html.split('<details')[0], /private-runtime/);
  assert.match(html.split('<details')[1], /private-runtime/);
});

test('a saved task with no run has a clear start action, not a continuation label', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_fixture', title: '阅读资料', draft: '请开始', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  assert.match(html, /开始任务/);
});

test('an accepted resume remains in progress and does not offer another resume button', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const { taskStatus } = require('./types.ts');
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { id: 'run_fixture', status: 'recovering' } }));
  assert.match(html, /暂停任务/);
  assert.doesNotMatch(html, /检查并继续/);
  assert.equal(taskStatus('recovering'), '正在恢复');
});
