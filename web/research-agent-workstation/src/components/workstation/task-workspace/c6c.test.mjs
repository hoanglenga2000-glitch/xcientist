import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

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
const source = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const css = source('./workspace.css');
const rule = selector => { const start = css.indexOf(selector + '{'); return start < 0 ? '' : css.slice(start, css.indexOf('}', start) + 1); };
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const presentation = () => require('./task-presentation.ts');
const { taskStatus } = require('./types.ts');
const ENGLISH_UI = /Temporary Chat|incognito|\bPII\b|GDPR|memory_mode|>on<|>off</i;
/** What a user reads: text nodes plus aria-labels / titles. */
const visible = html => [...html.replace(/<[^>]+>/g, ' ').split(/\s+/), ...[...html.matchAll(/(?:aria-label|title|placeholder)="([^"]*)"/g)].map(match => match[1])].join(' ');
const NOW = Date.parse('2026-10-28T12:00:00+08:00');
const mem = (id, text, extra = {}) => ({ id: 'memory_' + id.repeat(32).slice(0, 32), text, source_task_id: null, source_task_title: null, project_id: null,
  created_at: '2026-10-26T12:00:00Z', updated_at: '2026-10-26T12:00:00Z', kind: 'other', scope: 'user', pinned: false, last_used_at: null, use_count: 0, ...extra });
const history = (answer = '好的') => ({ schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [
  { id: 'u1', role: 'user', content: '帮我分析' }, { id: 'a1', role: 'assistant', content: answer }] });

// ---- soft s1: no 说明条 under Stop / 暂停 ----
test('t237 C6c H_c6c_s1: no help text under the running Stop/暂停 row', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  for (const status of ['running', 'queued', 'waiting_approval', 'pausing']) {
    const html = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: { id: 'run_live', status } }));
    assert.match(html, /data-testid="tw-stop"[^>]*>停止</, status);
    assert.doesNotMatch(html, /暂停会|停止会|已发出的操作|保留已写出的内容/, status);
    const after = html.slice(html.indexOf('tw-run-controls'));
    assert.doesNotMatch(after, /<p class="tw-muted">/, status);
  }
  assert.doesNotMatch(source('./TaskControls.tsx'), /暂停会阻止新操作/);
});

// ---- soft s3: 393 first fold ----
test('t237 C6c H_c6c_s3: the running Stop row sticks to the top of the chat scroll; Stop ≥44px @393', () => {
  assert.match(rule('.task-workspace .tw-run-controls'), /position:sticky;top:0/);
  assert.match(rule('.task-workspace .tw-run-controls'), /z-index:\d/);
  assert.match(rule('.task-workspace .tw-run-controls'), /background:var\(--tw-bg\)/);
  assert.match(rule('.task-workspace .tw-run-ops'), /display:contents/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-stop\{min-height:44px/);
  // Controls render before the run view inside the scroll container (first fold, above the answer).
  const session = source('./TaskSession.tsx');
  assert.ok(session.indexOf('<TaskControls') < session.indexOf('<TaskRunView'), 'controls come first in the chat scroll');
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: { id: 'run_live', status: 'running' } }));
  assert.match(html, /^<section class="tw-run-ops" aria-label="运行操作"><div class="tw-actions tw-run-controls"><button type="button" class="tw-primary tw-stop"/, 'Stop is the first control');
});

// ---- soft s2: W3 one status word ----
test('t237 C6c H_c6c_s2: runStatusWord — 正在处理 / 已停止 / 已完成 / 等你确认; never 尚未开始 or 已交付 for a run', () => {
  const { runStatusWord, statusPillClass } = presentation();
  const word = run => runStatusWord(run, taskStatus);
  assert.equal(word({ status: 'running' }), '正在处理');
  assert.equal(word({ status: 'cancelled' }), '已取消');
  assert.equal(word({ status: 'completed' }), '已完成');
  assert.equal(word({ status: 'delivered' }), '已完成');
  assert.equal(word({ status: 'waiting_approval' }), '等你确认');
  assert.equal(word({ status: 'completed', approvals: [{ status: 'pending' }] }), '等你确认');
  assert.equal(word({ status: 'completed', plan_proposed: { steps: [] } }), '等你确认');
  assert.equal(word({ status: 'blocked', error_class: 'planning_idle' }), '方案停住了');
  assert.equal(statusPillClass('已完成'), 'tw-status-pill tw-status-delivered');
  assert.equal(statusPillClass('已取消'), 'tw-status-pill tw-status-idle');
  for (const status of ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing', 'completed', 'cancelled', 'blocked', 'failed'])
    assert.doesNotMatch(word({ status }), /尚未开始|已交付/, status);
});

test('t237 C6c H_c6c_s2 (C6-answer): the progress card is gone; the process row never contradicts the header word', () => {
  const { chatEyebrow, runStatusWord } = presentation();
  const { TaskRunView } = require('./TaskRunView.tsx');
  const row = html => html.match(/<span class="tw-thinking-text">([^<]*)</)?.[1] ?? '';
  // Header word → what the process row may say for the same run.
  const agrees = { '正在处理': /^(思考中…|正在)/, '正在检查结果': /^(思考中…|正在)/, '已完成': /^(工作了|思考了|已完成)/, '已取消': /^已取消/, '等你确认': /^等你确认$/, '暂时无法继续': /^这一步没有完成/, '这一步停住了': /^这一步没有完成/ };
  for (const status of ['running', 'verifying', 'completed', 'cancelled', 'waiting_approval', 'blocked']) {
    const run = { id: 'run_w3', status, answer: '好的', created_at: '2026-10-29T00:00:00Z', completed_at: '2026-10-29T00:00:09Z', message_history: history() };
    for (const task of [{ status: 'draft' }, { status: 'delivered' }]) {
      const header = chatEyebrow({ run, runs: [run], runId: run.id, task, taskStatus });
      const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run, title: '分析' }));
      assert.equal(header, runStatusWord(run, taskStatus));
      assert.doesNotMatch(html, /tw-progress-head|tw-progress-card/, status);
      assert.match(row(html), agrees[header] ?? /^$/, `${status} / task ${task.status}: ${header} vs ${row(html)}`);
      assert.doesNotMatch(row(html), /尚未开始|已交付/);
    }
  }
});

// ---- soft s4: stuck / fail dedupe ----
test('t237 C6c H_c6c_s4: consecutive identical failed rows merge into one (×N)', () => {
  const { timelineRows } = presentation();
  const fail = { kind: 'tool', label: '写入文件', status: 'failed', error: '这一步没有完成' };
  const rows = timelineRows([fail, fail, fail], true);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].label, '写入文件 ×3');
  assert.equal(timelineRows([fail, { ...fail, label: '读资料' }, fail], true).length, 3, 'non-adjacent rows stay apart');
});

test('t237 C6c H_c6c_s4: one stuck surface — banner wins over the blocked line; model fail card stands alone', () => {
  const { stuckSurface } = presentation();
  const stale = { freshness: 'stale', heartbeat_age_seconds: 70 };
  assert.equal(stuckSurface({ status: 'running', progress: stale, answer: 'recoverable' }), 'banner');
  assert.equal(stuckSurface({ status: 'blocked', error_class: 'planning_idle' }), 'card');
  assert.equal(stuckSurface({ status: 'blocked', error_class: 'model_unavailable', progress: stale }), 'model');
  assert.equal(stuckSurface({ status: 'running', progress: { freshness: 'fresh' } }), null);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const both = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', onStop() {}, run: {
    id: 'run_s4', status: 'running', progress: stale, answer: 'recoverable step', created_at: '2026-10-29T00:00:00Z', message_history: history('recoverable step'),
  } }));
  assert.equal((both.match(/data-testid="tw-stuck-banner"/g) || []).length, 1);
  assert.doesNotMatch(both, /停住了，我先检查/, 'no second stuck line when the banner shows');
  assert.doesNotMatch(both, /is-stuck/, 'progress card does not repeat the stall');
  const blocked = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { id: 'run_b', status: 'blocked', error_class: 'planning_idle', message_history: history() } }));
  assert.equal((blocked.match(/方案停住了，我先检查/g) || []).length, 1);
  const model = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { id: 'run_m', status: 'blocked', error_class: 'model_unavailable', progress: stale, message_history: history() } }));
  assert.equal((model.match(/data-testid="tw-model-fail"/g) || []).length, 1);
  assert.doesNotMatch(model, /tw-stuck-banner|我先检查/);
});

// ---- CP6 memory page ----
test('t237 C6c CP6: kind labels, tabs and filter (preference|profile → 个人偏好; other only under 全部)', () => {
  const m = require('./memory.ts');
  assert.deepEqual(m.MEMORY_KIND_LABEL, { profile: '关于我', preference: '偏好', project: '项目', fact: '事实', other: '其他' });
  assert.deepEqual(m.MEMORY_TABS.map(([, label]) => label), ['全部', '个人偏好', '项目', '事实', '置顶']);
  const rows = [mem('1', '喜欢简洁', { kind: 'preference' }), mem('2', '博士三年级', { kind: 'profile' }), mem('3', '肺结节项目', { kind: 'project', pinned: true }),
    mem('4', '用 PyTorch', { kind: 'fact' }), mem('5', '杂项', { kind: 'other' })];
  const ids = (tab, q = '') => m.filterMemories(rows, tab, q).map(row => row.text);
  assert.deepEqual(ids('all'), ['喜欢简洁', '博士三年级', '肺结节项目', '用 PyTorch', '杂项']);
  assert.deepEqual(ids('preference'), ['喜欢简洁', '博士三年级']);
  assert.deepEqual(ids('project'), ['肺结节项目']);
  assert.deepEqual(ids('fact'), ['用 PyTorch']);
  assert.deepEqual(ids('pinned'), ['肺结节项目']);
  assert.deepEqual(ids('all', 'pytorch'), ['用 PyTorch']);
  assert.deepEqual(ids('preference', '肺'), []);
  // Old C3 rows without kind read as 其他.
  assert.deepEqual(m.filterMemories([{ ...rows[0], kind: undefined }], 'all', '').length, 1);
  assert.equal(m.memoryKindLabel(undefined), '其他');
});

test('t237 C6c CP6: page — title 记忆 only, tabs with accent underline, search, rows with pill + pin, no ids', () => {
  const { MemoryView } = require('./MemorySettings.tsx');
  const rows = [mem('1', '喜欢简洁回答', { kind: 'preference', pinned: true }), mem('2', '用 PyTorch', { kind: 'fact' })];
  const html = renderToStaticMarkup(h(MemoryView, { enabled: true, paused: false, memories: rows, now: NOW, tab: 'all', query: '',
    onToggle() {}, onPause() {}, onClearAll() {}, onEdit() {}, onDelete() {}, onPin() {}, onTab() {}, onQuery() {} }));
  assert.match(html, /<h2 class="tw-memory-title">记忆<\/h2>/);
  assert.doesNotMatch(html, /记住的内容|tw-memory-subtitle/);
  const tabs = [...html.matchAll(/role="tab" aria-selected="(true|false)"[^>]*>([^<]+)</g)].map(match => [match[2], match[1]]);
  assert.deepEqual(tabs, [['全部', 'true'], ['个人偏好', 'false'], ['项目', 'false'], ['事实', 'false'], ['置顶', 'false']]);
  assert.match(html, /<input type="search" class="tw-memory-search" aria-label="搜索记忆"/);
  assert.match(html, /class="tw-memory-row is-pinned"/);
  assert.match(html, /<span class="tw-memory-kind">偏好<\/span>/);
  assert.match(html, /<span class="tw-memory-kind">事实<\/span>/);
  assert.match(html, /class="tw-memory-pin" aria-pressed="true" aria-label="取消置顶：喜欢简洁回答"/);
  assert.match(html, /class="tw-memory-pin" aria-pressed="false" aria-label="置顶：用 PyTorch"/);
  assert.doesNotMatch(html.replace(/data-memory-id="[^"]*"/g, ''), /memory_[a-z0-9]{8}|tenant|owner|\/home\/|runtime/);
  assert.doesNotMatch(visible(html), ENGLISH_UI);
  assert.match(rule('.task-workspace .tw-memory-tab[aria-selected="true"]'), /border-bottom:2px solid var\(--tw-accent\)/);
  assert.doesNotMatch(rule('.task-workspace .tw-memory-tab[aria-selected="true"]'), /background:var\(--tw-accent\)/);
  assert.match(rule('.task-workspace .tw-memory-text'), /-webkit-line-clamp:2/);
  assert.match(rule('.task-workspace .tw-memory-kind'), /font-size:12px/);
  assert.match(rule('.task-workspace .tw-memory-row.is-pinned'), /background:var\(--tw-accent-wash\)/);
  assert.match(rule('.task-workspace .tw-memory-row'), /min-height:48px/);
});

test('t237 C6c CP6: search miss / empty / 暂停 vs 重置 danger zone', () => {
  const { MemoryView } = require('./MemorySettings.tsx');
  const base = { enabled: true, paused: false, now: NOW, tab: 'all', query: '', onToggle() {}, onPause() {}, onClearAll() {}, onEdit() {}, onDelete() {}, onPin() {}, onTab() {}, onQuery() {} };
  const miss = renderToStaticMarkup(h(MemoryView, { ...base, memories: [mem('1', '喜欢简洁')], query: '不存在' }));
  assert.match(miss, /<p class="tw-memory-miss">没有匹配的记忆<\/p>/);
  const empty = renderToStaticMarkup(h(MemoryView, { ...base, memories: [] }));
  assert.match(empty, /<div class="tw-empty"><p>还没有记忆<\/p><a class="tw-button" href="\/workspace">返回对话<\/a><\/div>/);
  const full = renderToStaticMarkup(h(MemoryView, { ...base, memories: [mem('1', '喜欢简洁')] }));
  assert.match(full, /<span id="tw-memory-pause-label">暂停记忆<\/span><button type="button" role="switch" aria-checked="false" aria-labelledby="tw-memory-pause-label" class="tw-switch">/);
  assert.match(full, /<div class="tw-memory-danger"><button type="button" class="tw-danger-button">重置记忆<\/button><\/div>/);
  assert.ok(full.indexOf('暂停记忆') < full.indexOf('tw-memory-list') && full.indexOf('tw-memory-list') < full.indexOf('重置记忆'), 'reset sits apart, below the list');
  assert.match(renderToStaticMarkup(h(MemoryView, { ...base, paused: true, memories: [mem('1', 'x')] })), /aria-labelledby="tw-memory-pause-label" class="tw-switch"/);
  assert.match(rule('.task-workspace .tw-danger-button'), /color:var\(--tw-danger-strong\)/);
});

test('t237 C6c CP6: editor rekinds; clear-all dialog copy + danger 清空, 取消 focused first', () => {
  const { MemoryEditor, ClearAllDialog } = require('./MemorySettings.tsx');
  const editor = renderToStaticMarkup(h(MemoryEditor, { value: '喜欢简洁', kind: 'other', truncated: false, onChange() {}, onKind() {}, onSave() {}, onCancel() {} }));
  assert.match(editor, /<select aria-label="分类" class="tw-memory-kind-select">/);
  for (const label of ['关于我', '偏好', '项目', '事实', '其他']) assert.match(editor, new RegExp(`>${label}</option>`));
  const dialog = renderToStaticMarkup(h(ClearAllDialog, { count: 3, onCancel() {}, onConfirm() {} }));
  assert.match(dialog, /<h2 id="tw-clear-memory-title">清空全部记忆？<\/h2><p id="tw-clear-memory-line">清空后不能恢复。<\/p>/);
  assert.match(dialog, /<button type="button" class="tw-button">取消<\/button><button type="button" class="tw-button tw-danger-solid">清空<\/button>/);
  const view = source('./MemorySettings.tsx');
  assert.match(view, /message: MEMORY_COPY\.deleted, action: \{ label: MEMORY_COPY\.undo/);
  assert.match(view, /patchMemory\(/);
});

test('t237 C6c CP6: memory API helpers — patch subset, paused, project memory-settings via proxies', () => {
  const m = source('./memory.ts');
  assert.match(m, /export const patchMemory = /);
  assert.match(m, /export const setMemoryPaused = /);
  assert.match(m, /\/api\/assistant\/projects\/\$\{encodeURIComponent\(projectId\)\}\/memory-settings/);
  const route = source('../../../app/api/assistant/projects/[projectId]/[[...path]]/route.ts');
  assert.match(route, /GET: \[[^\]]*'memory-settings'/);
  assert.match(route, /PATCH: \[[^\]]*'memory-settings'/);
});

test('t237 C6c H6-7: project page toggle reads 仅用本项目记忆 only', () => {
  const { ProjectMemoryToggleView } = require('./MemorySettings.tsx');
  const html = renderToStaticMarkup(h(ProjectMemoryToggleView, { projectOnly: false, onToggle() {} }));
  assert.match(html, /<span id="tw-project-memory-label">仅用本项目记忆<\/span><button type="button" role="switch" aria-checked="false" aria-labelledby="tw-project-memory-label" class="tw-switch">/);
  assert.doesNotMatch(html, /隔离|沙箱/);
  assert.match(source('./ProjectPage.tsx'), /<ProjectMemoryToggle projectId=\{project\.id\}/);
});

// ---- CP7 composer 无痕 ----
test('t237 C6c CP7: composer 无痕 — outline toggle, one-line status, memory_mode on create; Chinese only', () => {
  const { IncognitoToggle, IncognitoStatus, MEMORY_MODE_INCOGNITO } = require('./MemorySettings.tsx');
  assert.equal(MEMORY_MODE_INCOGNITO, 'incognito');
  const off = renderToStaticMarkup(h(IncognitoToggle, { on: false, onChange() {} }));
  assert.match(off, /^<button type="button" class="tw-incognito" aria-pressed="false" data-testid="tw-incognito">[^]*无痕<\/button>$/);
  assert.doesNotMatch(off, /tw-primary/);
  const on = renderToStaticMarkup(h(IncognitoToggle, { on: true, onChange() {} }));
  assert.match(on, /aria-pressed="true"/);
  assert.equal(renderToStaticMarkup(h(IncognitoStatus, { on: true, paused: true })), '<p class="tw-incognito-status" role="status">无痕 · 不读不写记忆</p>');
  assert.equal(renderToStaticMarkup(h(IncognitoStatus, { on: false, paused: false })), '');
  for (const html of [off, on]) assert.doesNotMatch(visible(html), ENGLISH_UI);
  const incog = rule('.task-workspace .tw-incognito');
  assert.match(incog, /border:1px solid var\(--tw-edge\)/);
  assert.match(incog, /background:transparent/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-incognito\{min-height:44px/);
  const session = source('./TaskSession.tsx');
  assert.match(session, /\.\.\.\(incognito \? \{ memory_mode: MEMORY_MODE_INCOGNITO \} : \{\}\)/);
  assert.match(session, /<IncognitoToggle on=\{incognito\}/);
  assert.match(session, /<IncognitoStatus on=\{incognito\}/);
  assert.doesNotMatch(session, /confirm\(|window\.confirm/);
});

// ---- 用到的记忆 ----
test('t237 C6c H7-5: 用到的记忆 · N collapsed under the answer, ≤40 chars, no ids; none when 0 / incognito / off', () => {
  const m = require('./memory.ts');
  assert.equal(m.memoryUsedLabel(2), '用到的记忆 · 2');
  assert.equal([...m.memoryUsedText('长'.repeat(80))].length, 40);
  assert.equal(m.memoryUsedText('短句'), '短句');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const used = [{ id: 'memory_' + '1'.repeat(32), text: '喜欢简洁回答' }, { id: 'memory_' + '2'.repeat(32), text: '长'.repeat(80) }];
  const run = extra => ({ id: 'run_used', status: 'completed', answer: '好的', message_history: history(), memory_used: used, memory_mode: 'on', ...extra });
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run() }));
  assert.match(html, /<details class="tw-memory-used" data-testid="tw-memory-used"><summary>用到的记忆 · 2<\/summary><ul><li>喜欢简洁回答<\/li><li>长{39}…<\/li><\/ul><\/details>/);
  assert.match(html, /<article class="tw-message tw-assistant(?: tw-msg-answer)?"[^>]*>[^]*?tw-memory-used[^]*?<\/article>/);
  assert.doesNotMatch(html, /memory_1{32}/);
  for (const extra of [{ memory_used: [] }, { memory_mode: 'incognito' }, { memory_mode: 'off' }])
    assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run(extra) })), /用到的记忆/, JSON.stringify(extra));
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run(), memoryEnabled: false })), /用到的记忆/);
  // Distinct from 已记住.
  const both = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: run({ memory_saved: [{ id: 'memory_' + '3'.repeat(32), text: '新事实' }] }) }));
  assert.match(both, /已记住/); assert.match(both, /用到的记忆 · 2/);
});

test('t238 C7e-R1: internal tool names never show in the tenant-visible 用到的记忆 lines (技术详情 only)', () => {
  const m = require('./memory.ts');
  assert.equal(m.memoryUsedText('受管 GPU 训练通道（managed_tensor_train）用于训练'), '受管 GPU 训练通道用于训练');
  assert.equal(m.memoryUsedText('用 hpc_execute_solution 和 tenant_tensor_train 训练'), '用 和 训练');
  assert.equal(m.memoryUsedText('喜欢简洁回答'), '喜欢简洁回答');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const run = { id: 'run_used', status: 'completed', answer: '好的', message_history: history(), memory_mode: 'on',
    memory_used: [{ id: 'memory_' + '1'.repeat(32), text: '受管 GPU 训练通道（managed_tensor_train）…' }] };
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run }));
  assert.match(html, /用到的记忆 · 1/);
  assert.doesNotMatch(html, /managed_tensor_train|tenant_tensor_train|hpc_execute_solution/);
});

test('t237 C6c3 H6-5: memory-row ⋯ is in-flow beside pin (no overlap), hit targets intact', () => {
  const { MemoryView } = require('./MemorySettings.tsx');
  const html = renderToStaticMarkup(h(MemoryView, { enabled: true, paused: false, now: NOW, tab: 'all', query: '', memories: [mem('1', '喜欢简洁')],
    onToggle() {}, onPause() {}, onClearAll() {}, onEdit() {}, onDelete() {}, onPin() {}, onTab() {}, onQuery() {} }));
  // The selector must really match the DOM: li.tw-memory-row > div.tw-memory-side > (pin, ⋯).
  assert.match(html, /<li [^>]*class="tw-memory-row"[^>]*>[^]*<div class="tw-memory-side">[^]*<button [^>]*class="tw-memory-pin"[^]*<button [^>]*class="tw-row-more"/);
  const fix = rule('.task-workspace .tw-memory-side .tw-row-more');
  assert.match(fix, /position:static/); assert.match(fix, /transform:none/); assert.match(fix, /flex:none/);
  assert.doesNotMatch(fix, /position:absolute/);
  assert.match(rule('.task-workspace .tw-memory-side'), /display:flex/);
  // Hover/focus still reveals ⋯ (opacity), and 44px targets at ≤760 for both pin and ⋯.
  assert.match(css, /\.tw-memory-row:hover \.tw-row-more[^{]*\{opacity:1\}/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-memory-pin\{width:44px;height:44px\}/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-row-more\{width:44px;height:44px/);
  // Specificity: the fix (0,3,0) must outrank the shared absolute rule (0,2,0).
  assert.ok(css.indexOf('.task-workspace .tw-memory-side .tw-row-more{') > 0);
});

test('t237 C6c3: run selector option never prints Invalid Date', () => {
  const { runOptionLabel } = presentation();
  assert.equal(typeof runOptionLabel, 'function');
  assert.equal(runOptionLabel(undefined, '执行已完成'), '执行已完成');
  assert.equal(runOptionLabel('', '执行已完成'), '执行已完成');
  assert.equal(runOptionLabel('not-a-date', '执行已完成'), '执行已完成');
  assert.equal(runOptionLabel(null, '执行已完成'), '执行已完成');
  const good = runOptionLabel('2026-10-28T12:00:00Z', '执行已完成');
  assert.doesNotMatch(good, /Invalid/); assert.match(good, / · 执行已完成$/);
  assert.doesNotMatch(source('./TaskSession.tsx'), /new Date\(item\.created_at \?\? ''\)\.toLocaleString/);
  assert.match(source('./TaskSession.tsx'), /runOptionLabel\(item\.created_at, taskStatus\(item\.status\)\)/);
});

test('t237 C6c4 H6-5: 清空 confirm is danger-styled — outranks the dialog-actions outline rule, both themes', () => {
  const { ClearAllDialog } = require('./MemorySettings.tsx');
  const html = renderToStaticMarkup(h(ClearAllDialog, { count: 3, onCancel() {}, onConfirm() {} }));
  assert.match(html, /<button type="button" class="tw-button tw-danger-solid">清空<\/button>/);
  assert.match(html, /<button type="button" class="tw-button">取消<\/button>/);
  // `.task-workspace .tw-dialog-actions .tw-button` is (0,3,0); the danger rule must be at least as specific and come later.
  const danger = rule('.task-workspace .tw-dialog-actions .tw-button.tw-danger-solid');
  assert.match(danger, /background:var\(--tw-danger-strong\)/);
  assert.match(danger, /border-color:var\(--tw-danger-strong\)/);
  assert.match(danger, /color:var\(--tw-on-accent\)/);
  assert.ok(css.indexOf('.task-workspace .tw-dialog-actions .tw-button.tw-danger-solid{') > css.indexOf('.task-workspace .tw-dialog-actions .tw-button{'), 'danger rule comes after the outline rule');
  // Tokens exist in light and dark themes.
  const dark = css.slice(css.indexOf('[data-theme="dark"] .task-workspace{'));
  assert.match(css.slice(0, css.indexOf('[data-theme="dark"] .task-workspace{')), /--tw-danger-strong:#B42318/);
  assert.match(dark, /--tw-danger-strong:#FCA5A5/); assert.match(dark, /--tw-on-accent:#0E1A13/);
});

test('t237 C6c4 H7-7: 393 composer — picker keeps name + chevron ≥44px, toolbar wraps ≤2 rows, 立即发送 icon-only', () => {
  const mobile = css.slice(css.lastIndexOf('@media(max-width:760px){.task-workspace .tw-composer-bottom{flex-wrap:wrap'));
  assert.ok(mobile.length > 0, 'C6c4 mobile composer block exists');
  assert.match(mobile, /\.tw-composer-bottom\{[^}]*flex-wrap:wrap/);
  assert.match(mobile, /\.tw-composer-bottom>\.tw-save-status\{[^}]*margin-left:auto/); assert.match(mobile, /\.tw-composer-send\{[^}]*order:1/);
  assert.match(mobile, /\.tw-composer-send\{[^}]*flex:1 1 auto/);
  assert.match(mobile, /\.task-workspace \.tw-picker-trigger\{[^}]*min-width:64px/);
  assert.match(mobile, /\.tw-composer-send \.tw-picker:first-child\{[^}]*min-width:64px/);
  assert.match(mobile, /\.task-workspace \.tw-steer-btn\{[^}]*flex:none[^}]*width:44px;height:44px/);
  assert.match(mobile, /\.tw-steer-label\{[^}]*clip:rect\(0,0,0,0\)/);
  assert.match(mobile, /\.tw-send-icon\{width:44px;height:44px\}/);
  assert.match(rule('.tw-steer-icon'), /display:none/);
  assert.match(mobile, /\.tw-steer-icon\{display:inline-block\}/);
  // The sr-only label names the button; C6d backlog 3: no aria-label repeating it, title stays as the tooltip.
  const session = source('./TaskSession.tsx');
  assert.match(session, /data-testid="tw-steer" title="立即发送"/);
  assert.match(session, /<span className="tw-steer-label">立即发送<\/span>/);
  assert.match(session, /<Zap [^>]*className="tw-steer-icon"/);
});

test('t237 C6c4 m3: sticky Stop row has a solid theme background and a bottom fade', () => {
  const bar = rule('.task-workspace .tw-run-controls');
  assert.match(bar, /position:sticky/); assert.match(bar, /background:var\(--tw-bg\)/);
  const fade = rule('.task-workspace .tw-run-controls::after');
  assert.match(fade, /top:100%/); assert.match(fade, /linear-gradient\(to bottom,var\(--tw-bg\),transparent\)/); assert.match(fade, /pointer-events:none/);
});
