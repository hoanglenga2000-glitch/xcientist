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
const source = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const css = source('./workspace.css');
const rule = selector => { const start = css.indexOf(selector + '{'); return start < 0 ? '' : css.slice(start, css.indexOf('}', start) + 1); };
const presentation = () => require('./task-presentation.ts');
const RAW = /file_read|shell_exec|web_search|tool_call|call_[a-z0-9]|\/hpc2hdd|\{"/;

// ---- CP3: 进行中 vs 待确认 ----

test('t237 C6a L5: isLiveTask counts running work only; waiting_approval and plan_pending are 待确认', () => {
  const { isLiveTask, isAwaitingConfirm } = presentation();
  for (const status of ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing']) {
    assert.equal(isLiveTask(status), true, status);
    assert.equal(isAwaitingConfirm(status), false, status);
  }
  for (const status of ['waiting_approval', 'plan_pending']) {
    assert.equal(isLiveTask(status), false, status);
    assert.equal(isAwaitingConfirm(status), true, status);
  }
  assert.equal(isAwaitingConfirm({ status: 'running', plan_pending: true }), true);
  assert.equal(isAwaitingConfirm({ status: 'completed' }), false);
  for (const status of ['completed', 'failed', 'blocked', 'paused', 'draft']) assert.equal(isLiveTask(status) || isAwaitingConfirm(status), false, status);
});

test('t237 C6a H3-1: dual count text omits zero segments and the whole row when both are zero', () => {
  const { liveCountSegments } = presentation();
  const text = (a, b) => liveCountSegments(a, b).map(item => item.text).join(' · ');
  assert.equal(text(2, 1), '2 项进行中 · 1 项待确认');
  assert.equal(text(1, 0), '1 项进行中');
  assert.equal(text(0, 1), '1 项待确认');
  assert.deepEqual(liveCountSegments(0, 0), []);
  assert.deepEqual(liveCountSegments(2, 1).map(item => item.kind), ['live', 'confirm']);
});

test('t237 C6a H3-2/H3-4: sidebar counts are two links told apart by shape and text, waiting never in 进行中', () => {
  const { LiveCounts } = require('./LiveCounts.tsx');
  const tasks = [
    { id: 'utask_run', title: '跑实验', status: 'running', updated_at: '2026-10-28T02:00:00Z' },
    { id: 'utask_run2', title: '读论文', status: 'verifying', updated_at: '2026-10-28T03:00:00Z' },
    { id: 'utask_wait', title: '等确认', status: 'waiting_approval', updated_at: '2026-10-28T01:00:00Z' },
    { id: 'utask_done', title: '完成', status: 'completed', updated_at: '2026-10-28T04:00:00Z' },
  ];
  const html = renderToStaticMarkup(React.createElement(LiveCounts, { tasks }));
  assert.match(html, /^<div class="tw-live" data-testid="tw-live-counts">/);
  assert.match(html, /<a class="tw-live-seg is-live" data-live-kind="live" href="\/workspace\?task=utask_run2"><span class="tw-dot is-current" aria-hidden="true"><\/span>2 项进行中<\/a>/);
  assert.match(html, /<a class="tw-live-seg is-confirm" data-live-kind="confirm" href="\/workspace\?task=utask_wait"><span class="tw-confirm-mark" aria-hidden="true">!<\/span>1 项待确认<\/a>/);
  assert.match(html, /<span class="tw-live-sep" aria-hidden="true">·<\/span>/);
  assert.equal(renderToStaticMarkup(React.createElement(LiveCounts, { tasks: [tasks[3]] })), '');
  const onlyWait = renderToStaticMarkup(React.createElement(LiveCounts, { tasks: [tasks[2]] }));
  assert.doesNotMatch(onlyWait, /进行中|tw-live-sep/);
  assert.match(source('./TaskWorkspace.tsx'), /<LiveCounts tasks=\{present\} \/>/);
  assert.match(css, /\.task-workspace \.tw-live-seg\{display:inline-flex[^}]*min-height:24px/);
  assert.match(rule('.task-workspace .tw-live-seg:hover'), /background:var\(--tw-selected\)/);
  assert.ok(!/#[0-9A-Fa-f]{3,8}\b|rgba?\(/.test(rule('.tw-confirm-mark')), 'confirm mark uses tokens');
  assert.match(rule('.tw-confirm-mark'), /border:1px solid var\(--tw-pill-confirm-fg\)/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-live-seg\{min-height:44px/);
  // Row markers split too: a waiting task gets the confirm mark, not the blue live dot.
  assert.match(source('./SidebarHistory.tsx'), /isAwaitingConfirm\(task\.status\) && <span className="tw-confirm-mark tw-row-live" aria-hidden="true">!<\/span>/);
});

// ---- CP1: process line + timeline ----

test('t237 C6a H1-1/H1-7 (C6-answer): the live row clock reads Ns / 1m 05s (hidden under 1s); the done row has no m:ss', () => {
  const { formatElapsed, liveClock, thinkingRowState } = presentation();
  assert.equal(formatElapsed(0), '0:00');
  assert.equal(formatElapsed(42_000), '0:42');
  assert.equal(formatElapsed(72_400), '1:12');
  assert.equal(formatElapsed(3_723_000), '1:02:03');
  // C6-answer H_a13: <1s shows no seconds; ≥60s reads 1m 05s.
  assert.equal(liveClock(999), null);
  assert.equal(liveClock(1000), '1s');
  assert.equal(liveClock(42_000), '42s');
  assert.equal(liveClock(65_000), '1m 05s');
  assert.equal(liveClock(null), null);
  // H_a16: a completed row is 工作了 Ns · N 步 — never m:ss.
  const done = thinkingRowState({ id: 'r', status: 'completed', created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:01:12Z',
    activity: [{ kind: 'tool', label: '读取文件', status: 'completed', duration_ms: 100 }] });
  assert.equal(done.text, '工作了 72s · 1 步');
  assert.doesNotMatch(done.text, /\d+:\d{2}/);
});

test('t237 C6a H1-3/S1-2: timeline shows the last 3 rows, merges repeats and never shows a raw tool id', () => {
  const { timelineRows, activityLabel } = presentation();
  assert.equal(activityLabel('file_read'), '处理一步');
  assert.equal(activityLabel('读取文件'), '读取文件');
  const rows = [
    { kind: 'tool', label: '搜索资料', status: 'completed', duration_ms: 1000 },
    { kind: 'tool', label: '搜索资料', status: 'completed', duration_ms: 2000 },
    { kind: 'tool', label: '搜索资料', status: 'completed', duration_ms: 3000 },
    { kind: 'tool', label: '读取文件', status: 'failed', duration_ms: 250, error: '没有权限' },
    { kind: 'tool', label: 'shell_exec', status: 'completed', duration_ms: 10 },
    { kind: 'tool', label: '运行命令', status: 'running', started_at: '2026-10-28T10:00:00Z' },
  ];
  const all = timelineRows(rows, true);
  assert.deepEqual(all.map(row => row.label), ['搜索资料 ×3', '读取文件', '处理一步', '运行命令']);
  assert.equal(all[0].duration_ms, 6000);
  assert.deepEqual(timelineRows(rows, false).map(row => row.label), ['读取文件', '处理一步', '运行命令']);
});

function liveRun(extra = {}) {
  const started = new Date(Date.now() - 42_000).toISOString();
  return {
    id: 'run_c6a', status: 'running', created_at: started, prompt: '帮我整理资料',
    plan: { steps: [{ id: 'understand', label: '理解目标与输入', status: 'completed' }, { id: 'execute', label: '调用工具执行任务', status: 'pending' }] },
    activity: [
      { kind: 'tool', label: '读取文件', status: 'completed', started_at: started, duration_ms: 1500 },
      { kind: 'tool', label: '整理结果文件', status: 'completed', started_at: started, duration_ms: 400 },
      { kind: 'tool', label: '运行命令', status: 'failed', started_at: started, duration_ms: 250, error: '没有权限' },
      { kind: 'tool', label: '生成报告', status: 'completed', started_at: started, duration_ms: 900 },
      { kind: 'tool', label: '搜索资料', status: 'running', started_at: started, duration_ms: null },
    ],
    activity_total: 5,
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: '帮我整理资料' }, { id: 'a', role: 'assistant', content: '' }] },
    current_activity: { kind: 'tool', label: '搜索资料', status: 'running', started_at: started, index: 2, total: 4 },
    ...extra,
  };
}

test('t237 C6a CP1 (C6-answer): a live run shows one process row 「正在{动作}…」 + clock, open, with the last 3 timeline rows', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: liveRun() }));
  assert.equal((html.match(/data-testid="tw-thinking-row"/g) ?? []).length, 1);
  assert.match(html, /<span class="tw-thinking-text">正在搜索资料…<\/span><span class="tw-thinking-clock" aria-hidden="true"> · 4\ds<\/span>/);
  assert.match(html, /aria-expanded="true"/);
  assert.match(html, /data-testid="tw-activity-timeline"/);
  assert.equal((html.match(/data-testid="tw-activity-row"/g) ?? []).length, 3);
  assert.match(html, /data-activity-status="failed"[^]*?运行命令 · 没有权限/);
  assert.match(html, /data-activity-status="running"/);
  assert.doesNotMatch(html, RAW);
  // No progress card, no fake 理解/执行/核对/交付 skeleton (C6-answer H_a9 / H_a15).
  assert.doesNotMatch(html, /tw-progress-card|tw-phases|查看过程|理解目标|调用工具执行任务/);
});

test('t237 C6a H1-6: 思考中… is gone once streaming text exists', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: liveRun({ streaming_text: '第一段' }) }));
  assert.doesNotMatch(html, /思考中/);
  assert.match(html, /第一段/);
});

test('t237 C6a H1-7 (C6-answer): a completed run folds to 「工作了 Ns · N 步」 (no m:ss) with the timeline hidden', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const run = liveRun({ status: 'completed', answer: '整理好了', created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:01:12Z', current_activity: null,
    activity: liveRun().activity.map(row => ({ ...row, status: row.status === 'running' ? 'completed' : row.status, duration_ms: row.duration_ms ?? 100 })) });
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run }));
  assert.match(html, /<button type="button" class="tw-process-line" data-testid="tw-process-line" aria-expanded="false"[^>]*>[^]*?<span class="tw-thinking-text">工作了 72s · 5 步<\/span>/);
  assert.doesNotMatch(html, /已完成 · \d+ 步 · \d+:\d+/);
  assert.match(html, /class="tw-thinking-detail" hidden=""/);
  // N 步 = the expanded timeline's row count.
  assert.equal((html.match(/data-testid="tw-activity-row"/g) ?? []).length, 5);
  assert.doesNotMatch(html, /思考中|正在搜索资料/);
});

test('t237 C6a H1-2/H1-4/H1-5 (C6-answer): process row is muted, one size smaller, no card; marks are accent check / red', () => {
  const line = rule('.task-workspace .tw-thinking-row>.tw-process-line');
  assert.match(line, /font-size:13px/);
  assert.match(line, /color:var\(--tw-muted\)/);
  assert.match(line, /border:0/);
  assert.match(line, /background:transparent/);
  assert.doesNotMatch(line, /box-shadow/);
  assert.match(rule('.tw-thinking-text'), /white-space:nowrap/);
  assert.match(rule('.tw-thinking-text'), /text-overflow:ellipsis/);
  assert.match(rule('.tw-activity-time'), /font-variant-numeric:tabular-nums/);
  assert.match(rule('.tw-thinking-detail .tw-activity-row.is-completed .tw-activity-mark'), /color:var\(--tw-accent\)/);
  assert.match(rule('.tw-activity-row.is-failed .tw-activity-mark'), /background:var\(--tw-danger-strong\)/);
  for (const selector of ['.tw-thinking-row', '.tw-thinking-row>.tw-process-line', '.tw-activity', '.tw-activity-row', '.tw-activity-time', '.tw-thinking-text', '.tw-live-seg']) {
    const body = rule(selector) || rule('.task-workspace ' + selector);
    assert.ok(body, selector);
    assert.ok(!/#[0-9A-Fa-f]{3,8}\b|rgba?\(/.test(body), `${selector} hard-codes a colour: ${body}`);
  }
  assert.doesNotMatch(source('./TaskRunView.tsx') + source('./task-presentation.ts'), /正在为您|智能|降级|流式失败/);
});

test('t237 C6a: SSE tool/step events update the timeline incrementally (no full read per tick)', () => {
  const { applyActivityEvent } = presentation();
  let run = { id: 'run_c6a', status: 'running', activity: [], activity_total: 0, current_activity: null };
  run = applyActivityEvent(run, 'step_started', { step_id: 'execute', label: '调用工具执行任务', index: 2, total: 4 });
  assert.deepEqual(run.current_activity, { kind: 'step', label: '调用工具执行任务', status: 'running', started_at: run.current_activity.started_at, index: 2, total: 4 });
  run = applyActivityEvent(run, 'tool_started', { tool: 'file_read', tool_call_id: 'call_1', label: '读取文件', started_at: '2026-10-28T10:00:00Z' });
  assert.equal(run.activity.length, 1);
  assert.deepEqual([run.current_activity.label, run.current_activity.index, run.current_activity.total], ['读取文件', 2, 4]);
  run = applyActivityEvent(run, 'tool_completed', { tool_call_id: 'call_1', status: 'failed', duration_ms: 250, error_label: '没有权限', label: '读取文件' });
  assert.deepEqual([run.activity[0].status, run.activity[0].duration_ms, run.activity[0].error], ['failed', 250, '没有权限']);
  assert.equal(run.current_activity.kind, 'step');
  run = applyActivityEvent(run, 'tool_started', { tool: 'x_tool', tool_call_id: 'call_2', label: 'x_tool' });
  assert.equal(run.activity[1].label, '处理一步');
  assert.doesNotMatch(run.activity.map(row => row.label).join('|'), RAW);
  const session = source('./TaskSession.tsx');
  for (const event of ['tool_started', 'step_started', 'step_completed']) assert.match(session, new RegExp(`'${event}'`));
  assert.match(session, /applyActivityEvent\(current, type, payload\)/);
});

// ---- CP2: model enable / disable / delete ----

test('t237 C6a H2-1: model menu matrix — 启用 when disabled; 停用/测试/设为默认 when enabled; 删除 always, danger', () => {
  const { modelMenuItems } = require('./ModelSettings.tsx');
  const on = { id: 'model_' + 'a'.repeat(32), name: '我的模型', enabled: true };
  const off = { ...on, enabled: false };
  const labels = (profile, isolated) => modelMenuItems(profile, isolated).map(item => item.label);
  assert.deepEqual(labels(on, false), ['编辑', '停用', '测试连接', '设为默认', '删除']);
  assert.deepEqual(labels(on, true), ['编辑', '停用', '设为默认', '删除']);
  assert.deepEqual(labels(off, false), ['编辑', '启用', '删除']);
  assert.deepEqual(labels(off, true), ['编辑', '启用', '删除']);
  for (const profile of [on, off]) {
    const items = modelMenuItems(profile, false);
    assert.deepEqual(items.filter(item => item.danger).map(item => item.label), ['删除']);
    assert.equal(items.at(-1).action, 'delete');
  }
});

test('t237 C6a H2-2/H2-3: a disabled row is greyed, says exactly 已停用, and its switch turns it back on; no banners', () => {
  const { ModelRow } = require('./ModelSettings.tsx');
  const profile = { id: 'model_' + 'b'.repeat(32), version: 2, name: '停用的模型', provider: 'openai', base_url: '', model: 'fixture-chat', enabled: false, verified_at: '', used_at: '' };
  const html = renderToStaticMarkup(React.createElement(ModelRow, { profile, isDefault: false, busy: false, menuOpen: false, onToggle() {}, onMenu() {} }));
  assert.match(html, /class="tw-set-row is-disabled"/);
  assert.equal((html.match(/已停用/g) ?? []).length, 1);
  assert.match(html, /<span class="tw-status-pill tw-status-idle" data-testid="tw-model-disabled-label">已停用<\/span>/);
  assert.match(html, /role="switch" class="tw-switch" aria-checked="false" aria-label="启用"/);
  assert.doesNotMatch(html.match(/<button[^>]*role="switch"[^>]*>/)[0], /disabled/);
  const on = renderToStaticMarkup(React.createElement(ModelRow, { profile: { ...profile, enabled: true }, isDefault: true, busy: false, menuOpen: false, onToggle() {}, onMenu() {} }));
  assert.match(on, /aria-checked="true" aria-label="停用"/);
  assert.doesNotMatch(on, /已停用/);
  const settings = source('./ModelSettings.tsx');
  assert.doesNotMatch(settings, /!profile\.enabled\}|if \(profile\.enabled\) void command/);
  assert.doesNotMatch(settings, /已暂停使用|可随时重新启用|重新启用后需再设为默认|暂停|禁用/);
});

test('t237 C6a H2-4: delete dialog copy is exact and 取消 holds the first focus', () => {
  const { DeleteModelDialog, MODEL_DELETE_COPY } = require('./DeleteModelDialog.tsx');
  assert.equal(MODEL_DELETE_COPY.title('我的模型'), '删除我的模型？');
  assert.equal(MODEL_DELETE_COPY.line, '删除后不能再选用，历史任务仍保留名称。');
  const html = renderToStaticMarkup(React.createElement(DeleteModelDialog, { name: '我的模型', busy: false, onCancel() {}, onConfirm() {} }));
  assert.equal(html, '<div class="tw-dialog-backdrop"><div class="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-delete-model-title" aria-describedby="tw-delete-model-line" data-testid="tw-model-delete-dialog">'
    + '<h2 id="tw-delete-model-title">删除我的模型？</h2><p id="tw-delete-model-line">删除后不能再选用，历史任务仍保留名称。</p>'
    + '<div class="tw-dialog-actions"><button type="button" class="tw-button" data-testid="tw-model-delete-cancel">取消</button><button type="button" class="tw-button is-danger" data-testid="tw-model-delete-confirm">删除</button></div></div></div>');
  const dialog = source('./DeleteModelDialog.tsx');
  assert.match(dialog, /cancel\.current\?\.focus\(\)/);
  assert.match(dialog, /event\.key === 'Escape'/);
  assert.doesNotMatch(dialog, /不可恢复|无法恢复|永久/);
  assert.match(rule('.task-workspace .tw-dialog .tw-button.is-danger'), /color:var\(--tw-danger\)/);
});

test('t237 C6a H2-5 / L8: a 409 refreshes the list with one plain line; enable and DELETE reach the runtime', () => {
  const settings = source('./ModelSettings.tsx');
  assert.match(settings, /MODEL_STATUS_UPDATED = '状态已更新，已刷新列表'/);
  assert.match(settings, /cause\.status === 409/);
  assert.match(settings, /jsonRequest\('DELETE', \{ version: profile\.version \}\)/);
  const route = source('../../../app/api/assistant/model-profiles/[[...path]]/route.ts');
  assert.match(route, /\['default', 'disable', 'enable', 'test'\]/);
  assert.match(route, /export const DELETE = forward;/);
  assert.match(route, /request\.method === 'DELETE' && parts\.length !== 1/);
  assert.match(source('./types.ts'), /model_profile_deleted:/);
});

test('C6a VERIFY fix: live-seg 44px lands after the 24px base rule in the cascade', () => {
  const base = css.indexOf('.task-workspace .tw-live-seg{display:inline-flex');
  assert.ok(base >= 0, 'base live-seg rule exists');
  const tail = css.slice(base);
  const win = css.lastIndexOf('.task-workspace .tw-live-seg{min-height:44px}');
  assert.ok(win > base, 'the 44px live-seg rule follows the 24px base rule');
  assert.ok(css.lastIndexOf('@media', win) > base && css.slice(css.lastIndexOf('@media', win)).startsWith('@media(max-width:760px)'), '44px rule sits in a later max-width:760px media');
  assert.ok(tail.includes('min-height:44px'), 'tail keeps 44px');
  assert.equal(css.split('.task-workspace .tw-live-seg{min-height:44px}').length - 1, 1, 'single source of truth');
});

test('C6a VERIFY fix: model-row more trigger stays in-flow and visible', () => {
  const global = css.indexOf('.task-workspace .tw-row-more{position:absolute');
  const scoped = css.indexOf('.task-workspace .tw-set-row .tw-row-more{');
  assert.ok(global >= 0 && scoped > global, 'set-row override comes after the global absolute rule');
  const body = css.slice(scoped, css.indexOf('}', scoped) + 1);
  assert.match(body, /position:static/);
  assert.match(body, /transform:none/);
  assert.match(body, /flex:none/);
  assert.match(body, /opacity:1/);
  // Coarse-pointer 44px sizing (later media rule) still has to apply: it must not be overridden to something smaller.
  assert.doesNotMatch(body, /(?:^|[;{])(?:width|height):/);
});
