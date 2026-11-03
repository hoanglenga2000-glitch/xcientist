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
const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
const rule = selector => new RegExp(selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\{([^}]*)\\}').exec(css)?.[1] ?? '';
const NOW = Date.parse('2026-10-27T12:00:00+08:00');
const mem = (id, text, extra = {}) => ({ id: 'memory_' + id.repeat(32).slice(0, 32), text, source_task_id: 'utask_' + 'a'.repeat(32),
  source_task_title: '论文阅读', project_id: null, created_at: '2026-10-26T12:00:00Z', updated_at: '2026-10-26T12:00:00Z', ...extra });

test('t237 C3.8: memory helpers — default ON, 2000 clamp, dates same / cross year, source line', () => {
  const m = require('./memory.ts');
  assert.equal(m.MEMORY_TEXT_MAX, 2000);
  assert.equal(m.resolveMemoryEnabled(undefined), true);
  assert.equal(m.resolveMemoryEnabled(null), true);
  assert.equal(m.resolveMemoryEnabled({}), true);
  assert.equal(m.resolveMemoryEnabled({ memory_enabled: false }), false);
  assert.equal(m.resolveMemoryEnabled({ memory_enabled: true }), true);
  assert.deepEqual(m.clampMemoryText('字'.repeat(2001)), ['字'.repeat(2000), true]);
  assert.deepEqual(m.clampMemoryText('短'), ['短', false]);
  assert.equal(m.memoryDate('2026-10-26T12:00:00Z', NOW), '10月26日');
  assert.equal(m.memoryDate('2025-10-26T12:00:00Z', NOW), '2025年10月26日');
  assert.equal(m.memoryDate('2026-01-05T12:00:00Z', NOW), '1月5日');
  assert.equal(m.memoryDate('bogus', NOW), '');
  assert.equal(m.memorySource(mem('1', 'x')), '来自「论文阅读」');
  assert.equal(m.memorySource(mem('1', 'x', { source_task_id: null, source_task_title: null })), '来自已删除的对话');
  assert.equal(m.memorySourceHref(mem('1', 'x')), '/workspace?task=utask_' + 'a'.repeat(32));
  assert.equal(m.memorySourceHref(mem('1', 'x', { source_task_title: null })), null, 'deleted source conversation is not clickable');
  assert.equal(m.memorySourceHref(mem('1', 'x', { source_task_id: null })), null);
  const kept = m.mergeEditedMemory(mem('1', 'old'), mem('1', 'new', { source_task_title: null }));
  assert.equal(kept.text, 'new'); assert.equal(kept.source_task_title, '论文阅读', 'edit keeps the live source title the list resolved');
  // C6c CP6 (H6-1/H6-5/H6-6/H6-9) supersedes the C3 heading / clear line / empty copy.
  for (const [key, value] of Object.entries({ tab: '记忆', title: '记忆', toggle: '使用记忆', pause: '暂停记忆', reset: '重置记忆', limit: '最多 2000 字', clearTitle: '清空全部记忆？',
    clearLine: '清空后不能恢复。', empty: '还没有记忆', back: '返回对话', noMatch: '没有匹配的记忆', offEmpty: '记忆已关闭。', turnOn: '打开记忆', deleted: '已删除', undo: '撤销',
    edit: '编辑', remove: '删除', save: '保存', cancel: '取消', clear: '清空' }))
    assert.equal(m.MEMORY_COPY[key], value, key);
});

test('t237 C3.8: settings tabs 模型 · 记忆 · 计算资源与额度 · 账户与外观 · 已归档 and ?settings=memory opens the memory page', () => {
  const { WorkspaceSettings } = require('./WorkspaceSettings.tsx');
  const html = renderToStaticMarkup(h(WorkspaceSettings, { group: 'memory', returnTask: '', isolated: false }));
  const tabs = [...html.matchAll(/href="\/workspace\?settings=(\w+)"[^>]*>([^<]+)</g)].map(match => [match[1], match[2]]);
  assert.deepEqual(tabs, [['models', '模型'], ['memory', '记忆'], ['resources', '计算资源与额度'], ['account', '账户与外观'], ['archived', '已归档']]);
  assert.match(html, /<a aria-current="page" href="\/workspace\?settings=memory">记忆<\/a>/);
  assert.match(html, /使用记忆/);
  assert.doesNotMatch(html, /添加记忆|>添加</);
  assert.doesNotMatch(html, /还没有记忆/, 'no empty state before the list has loaded');
});

test('t237 C3.8: 使用记忆 switch — role=switch, aria-checked, aria-labelledby; 36×20, accent, knob 16', () => {
  const { MemoryView } = require('./MemorySettings.tsx');
  const base = { memories: [], now: NOW, onToggle() {}, onClearAll() {}, onEdit() {}, onDelete() {} };
  const on = renderToStaticMarkup(h(MemoryView, { ...base, enabled: true }));
  assert.match(on, /<span id="tw-memory-toggle-label">使用记忆<\/span><button type="button" role="switch" aria-checked="true" aria-labelledby="tw-memory-toggle-label" class="tw-switch">/);
  const off = renderToStaticMarkup(h(MemoryView, { ...base, enabled: false }));
  assert.match(off, /aria-checked="false"/);
  const sw = rule('.task-workspace .tw-switch');
  for (const value of ['width:36px', 'height:20px']) assert.ok(sw.includes(value), value);
  assert.match(rule(".task-workspace .tw-switch[aria-checked=\"true\"]"), /background:(?:#2F855A|var\(--tw-accent\))/);
  assert.match(rule('.task-workspace .tw-switch::after'), /width:16px;height:16px/);
});

test('t237 C4 soft→hard: switch OFF #E7EAE8, toggle padding 16/16, .tw-memory-row position:relative', () => {
  // 科研先知 H_toggle_off_track / H_switch_row_padding / H_memory_row_relative + Critiquito minors
  const sw = rule('.task-workspace .tw-switch');
  assert.match(sw, /background:(?:#E7EAE8|var\(--tw-switch-off\)|var\(--tw-selected\))/);
  assert.doesNotMatch(sw, /#C9CEC9|#C3CBC5/);
  const toggle = rule('.task-workspace .tw-memory-toggle');
  assert.match(toggle, /padding:16px 0 16px/);
  assert.doesNotMatch(toggle, /padding:8px 0 16px/);
  const row = rule('.task-workspace .tw-memory-row');
  assert.match(row, /position:relative/);
});

test('t237 C3.8 (+C6c CP6): empty states, list chrome, rows (2-line clamp, source · date, menu trigger), no add button', () => {
  const { MemoryView } = require('./MemorySettings.tsx');
  const base = { now: NOW, onToggle() {}, onClearAll() {}, onEdit() {}, onDelete() {} };
  const empty = renderToStaticMarkup(h(MemoryView, { ...base, enabled: true, memories: [] }));
  assert.match(empty, /<div class="tw-empty"><p>还没有记忆<\/p><a class="tw-button" href="\/workspace">返回对话<\/a><\/div>/);
  assert.doesNotMatch(empty, /重置记忆/);
  const offEmpty = renderToStaticMarkup(h(MemoryView, { ...base, enabled: false, memories: [] }));
  assert.match(offEmpty, /<div class="tw-empty"><p>记忆已关闭。<\/p><button type="button" class="tw-button">打开记忆<\/button><\/div>/);
  const rows = [mem('1', '我在做肺结节分割'), mem('2', '偏好 PyTorch', { source_task_id: null, source_task_title: null, created_at: '2025-03-02T12:00:00Z' })];
  const html = renderToStaticMarkup(h(MemoryView, { ...base, enabled: true, memories: rows }));
  assert.match(html, /<h2 class="tw-memory-title">记忆<\/h2>/);
  assert.match(html, /<div class="tw-memory-danger"><button type="button" class="tw-danger-button">重置记忆<\/button><\/div>/);
  assert.match(html, /<p class="tw-memory-text">我在做肺结节分割<\/p><p class="tw-memory-source">来自「<a href="\/workspace\?task=utask_a{32}">论文阅读<\/a>」 · <time dateTime="2026-10-26T12:00:00Z">10月26日<\/time><\/p>/);
  assert.match(html, /<p class="tw-memory-source">来自已删除的对话 · <time dateTime="2025-03-02T12:00:00Z">2025年3月2日<\/time>/);
  assert.match(html, /<button type="button" class="tw-row-more" aria-haspopup="menu" aria-expanded="false" aria-label="更多操作：我在做肺结节分割">/);
  assert.doesNotMatch(html, /添加/);
  // Still listed and editable when memory is off.
  assert.match(renderToStaticMarkup(h(MemoryView, { ...base, enabled: false, memories: rows })), /我在做肺结节分割/);
  assert.match(rule('.task-workspace .tw-memory-text'), /-webkit-line-clamp:2/);
  assert.match(rule('.task-workspace .tw-text-button'), /color:(?:#6B7280|var\(--tw-muted\)|var\(--tw-label\))/);
  assert.doesNotMatch(rule('.task-workspace .tw-text-button'), /#B42318|red/);
  assert.match(rule('.task-workspace .tw-memory-row.is-highlight'), /background:(?:#EEF1EF|var\(--tw-hover\))/);
});

test('t237 C3.8: row menu 编辑 · 删除 (red only on 删除); edit textarea clamps to 2000 with 最多 2000 字 tip', () => {
  const { MemoryRowMenu, MemoryEditor } = require('./MemorySettings.tsx');
  const menu = renderToStaticMarkup(h(MemoryRowMenu, { onEdit() {}, onDelete() {} }));
  assert.match(menu, /<button type="button" role="menuitem" tabindex="-1" class="tw-menu-item">[^]*?<span class="tw-menu-label">编辑<\/span><\/button>/);
  assert.match(menu, /class="tw-menu-item is-danger">[^]*?<span class="tw-menu-label">删除<\/span>/);
  assert.equal((menu.match(/is-danger/g) || []).length, 1);
  const editor = renderToStaticMarkup(h(MemoryEditor, { value: '字'.repeat(2000), truncated: true, onChange() {}, onSave() {}, onCancel() {} }));
  assert.match(editor, /<textarea aria-label="编辑记忆"[^>]*>/);
  assert.match(editor, /<p class="tw-muted" role="status">最多 2000 字<\/p>/);
  assert.match(editor, /<button type="button" class="tw-primary">保存<\/button><button type="button">取消<\/button>/);
  const blank = renderToStaticMarkup(h(MemoryEditor, { value: '  ', truncated: false, onChange() {}, onSave() {}, onCancel() {} }));
  assert.match(blank, /<button type="button" class="tw-primary" disabled="">保存<\/button>/);
  assert.doesNotMatch(blank, /最多 2000 字/);
});

test('t237 C3.8 (+C6c H6-5): clear-all alertdialog says 清空后不能恢复, 取消 first and focused, 清空 danger; no undo toast after clear', () => {
  const { ClearAllDialog } = require('./MemorySettings.tsx');
  const html = renderToStaticMarkup(h(ClearAllDialog, { count: 4, onCancel() {}, onConfirm() {} }));
  assert.match(html, /role="alertdialog" aria-modal="true" aria-labelledby="tw-clear-memory-title" aria-describedby="tw-clear-memory-line"/);
  assert.match(html, /<h2 id="tw-clear-memory-title">清空全部记忆？<\/h2><p id="tw-clear-memory-line">清空后不能恢复。<\/p>/);
  assert.match(html, /<div class="tw-dialog-actions"><button type="button" class="tw-button">取消<\/button><button type="button" class="tw-button tw-danger-solid">清空<\/button><\/div>/);
  const source = readFileSync(new URL('./MemorySettings.tsx', import.meta.url), 'utf8');
  const clear = /async function clearAll\(\)[^]*?\n  }/.exec(source)?.[0] ?? '';
  assert.ok(clear.includes('clearMemories('), 'clear-all calls the API');
  assert.doesNotMatch(clear, /action:|撤销|undo/, 'no undo toast after clear-all');
  assert.match(source, /message: MEMORY_COPY\.deleted, action: \{ label: MEMORY_COPY\.undo/);
  assert.match(source, /restoreMemory\(/);
});

test('t237 C3.8: memory API paths go through the C3.7 proxies; no create call', () => {
  const source = readFileSync(new URL('./memory.ts', import.meta.url), 'utf8');
  for (const path of ["'/api/assistant/user-memory'", "'/api/assistant/user-settings'", '/restore']) assert.ok(source.includes(path), path);
  assert.doesNotMatch(source, /jsonPost\('\/api\/assistant\/user-memory'|POST', '\/api\/assistant\/user-memory'/);
});

const saved = (n, text = '我在做肺结节分割') => Array.from({ length: n }, (_, i) => ({ id: 'memory_' + String(i + 1).repeat(32), text: i ? `${text}${i}` : text }));
const chipRun = extra => ({ id: 'run_' + 'd'.repeat(32), status: 'completed', prompt: '记住我在做肺结节分割', answer: '好的，已经记下。', ...extra });

test('t237 C3.9: 已记住 chip copy, aria-label, 120-char title and link to the matching memory row', () => {
  const m = require('./memory.ts');
  assert.equal(m.memoryChipLabel(saved(1)), '已记住');
  assert.equal(m.memoryChipLabel(saved(3)), '已记住 3 条');
  assert.equal(m.memoryChipTitle(saved(1, '长'.repeat(200))), '长'.repeat(119) + '…');
  assert.equal(m.memoryChipTitle(saved(1)), '我在做肺结节分割');
  const { MemoryChip } = require('./MemorySettings.tsx');
  const one = renderToStaticMarkup(h(MemoryChip, { saved: saved(1) }));
  assert.match(one, new RegExp(`^<a class="tw-chip tw-memory-chip" href="/workspace\\?settings=memory&amp;memory=memory_1{32}" title="我在做肺结节分割" aria-label="已记住：我在做肺结节分割。管理记忆"><svg[^>]*width="12" height="12"[^>]*lucide-check[^]*?</svg>已记住</a>$`));
  assert.match(renderToStaticMarkup(h(MemoryChip, { saved: saved(2) })), />已记住 2 条<\/a>$/);
  assert.equal(renderToStaticMarkup(h(MemoryChip, { saved: [] })), '');
  assert.match(rule('.task-workspace .tw-memory-chip'), /margin-top:8px/);
  assert.match(rule('.task-workspace .tw-memory-chip svg'), /color:(?:#6B7280|var\(--tw-muted\)|var\(--tw-icon\))/);
  assert.match(rule('.task-workspace .tw-memory-row.is-highlight'), /background:(?:#EEF1EF|var\(--tw-hover\))/);
  assert.equal(m.MEMORY_HIGHLIGHT_MS, 1500);
});

test('t237 C3.9: chip renders under the assistant answer from memory_saved; none when off or nothing saved', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: chipRun({ memory_saved: saved(1) }) }));
  assert.match(html, /<article class="tw-message tw-assistant(?: tw-msg-answer)?"[^>]*><div class="tw-speaker">[^]*?<small>DeepEvo<\/small><\/div>[^]*?好的，已经记下。[^]*?<a class="tw-chip tw-memory-chip"[^>]*>[^]*?已记住<\/a><\/article>/);
  assert.equal((html.match(/tw-memory-chip/g) || []).length, 1);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: chipRun({ memory_saved: saved(1) }), memoryEnabled: false })), /已记住/);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: chipRun({ memory_saved: [] }) })), /已记住/);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: chipRun({}) })), /已记住/);
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, /'memory_saved'/, 'the live stream re-reads the snapshot on memory_saved');
  assert.match(session, /memoryEnabled=\{memoryEnabled\}/);
});
