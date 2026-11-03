import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// t237 C6e tip1+tip2: s_f_brand_aria_once + s_d_live_file_body (marker on .tw-md only, not article/chips).
const require = createRequire(import.meta.url);
const app = path => fileURLToPath(new URL('../../../' + path, import.meta.url));
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
const _source = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const h = (type, props, ...children) => React.createElement(type, props, ...children);

const BRAND_ARIA = /科研先知|EvoMind|Evomind|evomind/i;
const PID = 'project_' + 'e'.repeat(32);
const NOW = Date.parse('2026-10-31T12:00:00Z');
const at = hours => new Date(NOW - hours * 3600e3).toISOString();
const task = (id, title, extra = {}) => ({
  id: 'utask_' + id.padEnd(32, '0'), title, draft: '', version: 1, conversation_id: 'c', status: 'draft',
  created_at: at(1), updated_at: at(1), pinned_at: null, archived_at: null, project_id: null, ...extra,
});
const EVO_NAME = 'EvoMind 内测验收 20260905';
const sidebarProps = extra => ({
  tasks: [task('p0', '项目对话0', { project_id: PID, updated_at: at(1) }), task('t1', '今天的对话', { updated_at: new Date().toISOString() })],
  setTasks() {}, projects: [{ id: PID, name: EVO_NAME }], setProjects() {},
  archivedCount: 0, setArchivedCount() {}, selection: '', query: '', onLeaveOpen() {}, onNavigate() {}, ...extra,
});

/** Mirror VERIFY JS_ARIA_DUP brand_total over aria-label attributes only. */
function brandAriaTotal(html) {
  const labels = [...html.matchAll(/\saria-label="([^"]*)"/g)].map(match => match[1]);
  const byLabel = {};
  for (const label of labels) byLabel[label] = (byLabel[label] || 0) + 1;
  return Object.entries(byLabel).filter(([label]) => BRAND_ARIA.test(label)).reduce((sum, [, n]) => sum + n, 0);
}

test('C6e s_f_brand_aria_once: folder chevron/more use aria-labelledby; no EvoMind in aria-label', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const html = renderToStaticMarkup(h(SidebarHistory, sidebarProps()));
  assert.equal(brandAriaTotal(html), 0, html.match(/\saria-label="[^"]*EvoMind[^"]*"/g)?.join(' | ') || 'ok');
  assert.match(html, new RegExp(`id="tw-folder-title-${PID}"[^>]*class="tw-row-title">${EVO_NAME.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}<`));
  assert.match(html, new RegExp(`aria-labelledby="tw-folder-expand-${PID} tw-folder-title-${PID}"`));
  assert.match(html, new RegExp(`aria-labelledby="tw-folder-more-${PID} tw-folder-title-${PID}"`));
  assert.match(html, new RegExp(`id="tw-folder-expand-${PID}" class="tw-sr-only">展开<`));
  assert.match(html, new RegExp(`id="tw-folder-more-${PID}" class="tw-sr-only">项目操作<`));
  assert.doesNotMatch(html, /\saria-label="展开[^"]*EvoMind/);
  assert.doesNotMatch(html, /\saria-label="项目操作：[^"]*EvoMind/);
});

test('C6e s_f_brand_aria_once: expanded folder group + project menu omit brand aria-label', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const html = renderToStaticMarkup(h(SidebarHistory, sidebarProps({ selection: 'utask_p0'.padEnd(38, '0') })));
  assert.equal(brandAriaTotal(html), 0);
  assert.match(html, new RegExp(`role="group" aria-labelledby="tw-folder-title-${PID} tw-folder-convo-${PID}"`));
  assert.match(html, new RegExp(`id="tw-folder-convo-${PID}" class="tw-sr-only">的对话<`));
  const { ProjectMenu } = require('./ProjectMenu.tsx');
  const menu = renderToStaticMarkup(h(ProjectMenu, {
    id: 'tw-project-menu', project: { id: PID, name: EVO_NAME },
    anchor: { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }, returnFocus: null, onClose() {}, onCommand() {},
  }));
  assert.equal(brandAriaTotal(menu), 0);
  assert.match(menu, /role="menu" aria-label="项目操作"/);
  assert.doesNotMatch(menu, /EvoMind/);
});

test('C6e tip2 s_d_live_file_body: marker on .tw-md only; chip 下载 stays outside answer node', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const run = {
    id: 'run_c6e', status: 'completed', prompt: 'file ask', answer: '拆分方案要点如下。',
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'assistant_c6e', answer_is_current: true,
      messages: [{ id: 'assistant_c6e', role: 'assistant', content: '拆分方案要点如下。' }] },
    delivery_mode: 'inline+file', artifacts: [{ id: 'artifact_1', name: 'kxxz_c6_split_plan.md', bytes: 120, content_type: 'text/markdown' }],
    activity: [], started_at: '2026-10-31T12:00:00Z', finished_at: '2026-10-31T12:00:05Z',
  };
  const html = renderToStaticMarkup(h(TaskRunView, { run, memoryEnabled: false }));
  // Product-correct: markers live on the markdown node, not the article that also wraps .tw-attachments.
  assert.match(html, /<div class="tw-md tw-msg-answer" data-testid="tw-answer">/);
  assert.doesNotMatch(html, /<article class="[^"]*tw-msg-answer/);
  assert.doesNotMatch(html, /<article[^>]*data-testid="tw-answer"/);
  // Chip 下载 remains (deliverable control) but is a sibling under .tw-attachments, not inside the marked node.
  assert.match(html, /data-testid="tw-attachments"/);
  assert.match(html, /tw-filechip/);
  const markedOpen = html.indexOf('<div class="tw-md tw-msg-answer" data-testid="tw-answer">');
  const attAt = html.indexOf('<div class="tw-attachments"');
  assert.ok(markedOpen >= 0 && attAt > markedOpen, 'attachments follow marked md');
  const marked = html.slice(markedOpen, attAt);
  assert.match(marked, /^<div class="tw-md tw-msg-answer" data-testid="tw-answer">[\s\S]*<\/div>$/);
  assert.doesNotMatch(marked, /下载|download=1|tw-attachments|tw-filechip/);
  assert.match(html.slice(attAt), /下载|\?download=1|tw-filechip-action/);
});

test('C6e tip2 s_d_live_file_body: file_only summary .tw-md carries marker; chips still outside', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const run = {
    id: 'run_c6e_file', status: 'completed', prompt: 'file ask', answer: '',
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'assistant_file', answer_is_current: true,
      messages: [{ id: 'assistant_file', role: 'assistant', content: '' }] },
    delivery_mode: 'file_only', artifacts: [{ id: 'artifact_2', name: 'kxxz_c6_split_plan.md', bytes: 80, content_type: 'text/markdown' }],
    activity: [], started_at: '2026-10-31T12:00:00Z', finished_at: '2026-10-31T12:00:05Z',
  };
  const html = renderToStaticMarkup(h(TaskRunView, { run, memoryEnabled: false }));
  assert.match(html, /<div class="tw-md tw-msg-answer" data-testid="tw-answer"><p>已生成 <code>kxxz_c6_split_plan\.md<\/code>。<\/p><\/div>/);
  assert.doesNotMatch(html, /<article class="[^"]*tw-msg-answer/);
  const markedOpen = html.indexOf('<div class="tw-md tw-msg-answer" data-testid="tw-answer">');
  const attAt = html.indexOf('<div class="tw-attachments"');
  assert.ok(markedOpen >= 0 && attAt > markedOpen);
  assert.doesNotMatch(html.slice(markedOpen, attAt), /下载|download=1|tw-filechip/);
});

