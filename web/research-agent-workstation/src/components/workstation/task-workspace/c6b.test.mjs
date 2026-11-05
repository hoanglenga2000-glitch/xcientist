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
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const ENGLISH_UI = /Send now|\bsteer\b|>Stop<|>Pause<|exception|stack trace|TypeError|HTTP\/\d/i;

test('t237 C6b1 H4-2: 停止 is primary; 暂停 is secondary text — never same weight', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: { id: 'run_live', status: 'running' } }));
  assert.match(html, /data-testid="tw-stop"[^>]*>停止</);
  assert.match(html, /class="[^"]*tw-primary[^"]*tw-stop/);
  assert.match(html, /data-testid="tw-pause"[^>]*>暂停</);
  assert.match(html, /class="[^"]*tw-text-button[^"]*tw-pause/);
  assert.doesNotMatch(html, /暂停任务/);
  assert.doesNotMatch(html, ENGLISH_UI);
  // Pause must not also carry tw-primary.
  const pause = html.match(/<button[^>]*data-testid="tw-pause"[^>]*>/)?.[0] || '';
  assert.doesNotMatch(pause, /tw-primary/);
  assert.match(css, /\.task-workspace \.tw-stop\{[^}]*min-height:36px/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-stop\{min-height:44px/);
});

test('t237 C6b1 H4-6 (C6-answer H_a19): cancelled process row is 「已取消 · 用时 Ns」, never 已交付 or 工作了', () => {
  const { thinkingRowState, streamPhaseState } = presentation();
  assert.equal(thinkingRowState({ id: 'r', status: 'cancelled', created_at: '2026-10-29T00:00:00Z', completed_at: '2026-10-29T00:01:05Z' }).text, '已取消 · 用时 65s');
  const phase = streamPhaseState({ status: 'cancelled', answer: '半成品' });
  assert.equal(phase.done, false);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation', title: '分析',
    run: {
      id: 'run_c', status: 'cancelled', created_at: '2026-10-29T00:00:00Z', completed_at: '2026-10-29T00:01:05Z',
      answer: '半成品回答', partial_answer: '半成品回答',
      activity: [{ kind: 'tool', label: '整理', status: 'completed', duration_ms: 1000 }], activity_total: 2,
      message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [
        { id: 'u1', role: 'user', content: '写分析', kind: 'original_request' },
        { id: 'a1', role: 'assistant', content: '半成品回答', kind: 'stored_reply' },
      ] },
    },
  }));
  assert.match(html, /<span class="tw-thinking-text">已取消 · 用时 65s<\/span>/);
  assert.doesNotMatch(html, /已交付|工作了/);
  assert.match(html, /半成品回答/);
  assert.doesNotMatch(html, ENGLISH_UI);
});

test('t237 C6b1 L13/H5-1: stuck banner only when freshness=stale; amber wash, Chinese actions', () => {
  const { isStuckBanner, stuckSecondsLabel } = presentation();
  assert.equal(isStuckBanner({ freshness: 'fresh', heartbeat_age_seconds: 10 }, 'running'), false);
  assert.equal(isStuckBanner({ freshness: 'stale', heartbeat_age_seconds: 52 }, 'running'), true);
  assert.equal(isStuckBanner({ freshness: 'stale' }, 'cancelled'), false);
  assert.equal(stuckSecondsLabel(52), '52 秒无进展');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const fresh = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation',
    run: { id: 'run_f', status: 'running', progress: { freshness: 'fresh', heartbeat_age_seconds: 8 }, message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [{ id: 'u', role: 'user', content: 'x', kind: 'original_request' }, { id: 'a', role: 'assistant', content: '', kind: 'pending_response' }] } },
  }));
  assert.doesNotMatch(fresh, /tw-stuck-banner|秒无进展/);
  const stale = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation', onStop() {}, onRetry() {},
    run: { id: 'run_s', status: 'running', progress: { freshness: 'stale', heartbeat_age_seconds: 67 }, message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [{ id: 'u', role: 'user', content: 'x', kind: 'original_request' }, { id: 'a', role: 'assistant', content: '', kind: 'pending_response' }] } },
  }));
  assert.match(stale, /data-testid="tw-stuck-banner"/);
  assert.match(stale, /67 秒无进展/);
  assert.match(stale, />停止</);
  assert.match(stale, />再试一次</);
  assert.doesNotMatch(stale, /可能已崩溃|响应超时|联系管理员|Send now|steer/);
  assert.match(rule('.tw-stuck-banner'), /background:var\(--tw-pill-confirm-bg\)/);
  assert.match(rule('.tw-stuck-banner'), /border:1px solid var\(--tw-pill-confirm-fg\)/);
  assert.doesNotMatch(rule('.tw-stuck-banner'), /#(?:[fF]{2}|[eE]{2})[0-9a-fA-F]{0,4}\b|danger/);
});

test('t237 C6b1 H5-4: fail card primary 换个模型继续 + secondary 再试一次; tech under 技术详情', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation',
    run: {
      id: 'run_m', user_task_id: 'utask_m', status: 'blocked', error_class: 'model_identity_unconfirmed',
      error_message: 'model_identity_unconfirmed: served x != requested y',
      answer: 'Model execution is unavailable: LLMError/model_identity_unconfirmed.',
    },
  }));
  const card = html.match(/data-testid="tw-model-fail"[^>]*>[\s\S]*?<\/details><\/div>/)?.[0] || html;
  assert.match(card, /换个模型继续/);
  assert.match(card, />再试一次</); assert.doesNotMatch(card, /重试此步/);
  assert.match(card, /href="\/workspace\?settings=models&amp;return_task=utask_m"/);
  assert.match(card, /<summary>技术详情<\/summary>/);
  assert.doesNotMatch(card, /检查模型配置/);
  assert.doesNotMatch(card, /\bSend now\b|\bsteer\b/);
  // English / codes stay inside details only.
  const outside = card.replace(/<details[\s\S]*?<\/details>/, '');
  assert.doesNotMatch(outside, /model_identity_unconfirmed|LLMError|TypeError|exception/i);
});

test('t237 C6b1 H5-6 (C6-answer): the thinking summary lives in the process row — folded on settled runs; sanitized of paths/keys', () => {
  const { sanitizeThinkingSummary } = presentation();
  assert.equal(sanitizeThinkingSummary('先核对目标再拆步骤'), '先核对目标再拆步骤');
  assert.equal(sanitizeThinkingSummary('see /workspace/secret/key.txt and sk-abc1234567890'), '');
  assert.equal(sanitizeThinkingSummary('call_abc123 did tool_call'), '');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const pending = { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [{ id: 'u', role: 'user', content: 'x', kind: 'original_request' }, { id: 'a', role: 'assistant', content: '', kind: 'pending_response' }] };
  const none = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { id: 'run_t0', status: 'running', message_history: pending } }));
  assert.doesNotMatch(none, /tw-thinking-summary|思考摘要/);
  const done = { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [{ id: 'u', role: 'user', content: 'x', kind: 'original_request' }, { id: 'a', role: 'assistant', content: '好了', kind: 'stored_reply' }] };
  const html = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation',
    run: { id: 'run_t', status: 'completed', answer: '好了', created_at: '2026-10-29T00:00:00Z', completed_at: '2026-10-29T00:00:04Z',
      reasoning_summary: '先拆成两步核对\n/workspace/outputs/x.md\nsk-leak', message_history: done },
  }));
  assert.match(html, /<div id="[^"]+" class="tw-thinking-detail" hidden="">[^]*data-testid="tw-thinking-summary"[^]*先拆成两步核对/);
  assert.match(html, /aria-expanded="false"/);
  assert.doesNotMatch(html, /\/workspace\/outputs|sk-leak|<details[^>]*\sopen/);
});

test('t237 C6b1 copy: UI never shows Send now / steer / English mode names', () => {
  for (const file of ['TaskControls.tsx', 'TaskRunView.tsx', 'TaskSession.tsx']) {
    const body = source(`./${file}`);
    // Ban UI English; API token may be built as 'st'+'eer' (no contiguous quoted 'steer').
    assert.doesNotMatch(body, /Send now|"steer"|'steer'/);
  }
});

test('t237 C6b2 CP4: live composer unlock + queue preview + 立即发送 (no English)', () => {
  const body = source('./TaskSession.tsx');
  assert.match(body, /placeholder="说说你要推进的事"/);
  assert.match(body, /立即发送/);
  assert.match(body, /data-testid="tw-steer"/);
  assert.match(body, /data-testid="tw-queue-preview"/);
  assert.match(body, /排队 ·/);
  assert.match(body, /aria-label="取消排队"/);
  assert.match(body, /LIVE_MSG_MODE/);
  assert.match(body, /'st'\s*\+\s*'eer'/);
  assert.doesNotMatch(body, /Send now|"steer"|'steer'/);
  assert.doesNotMatch(body, /AI 正在|请稍候我们/);
  // Send no longer disabled solely because a run is active (composer unlocked).
  assert.doesNotMatch(body, /disabled=\{busy[^}]*Boolean\(run && active\(run\.status\)\)\}/);
  assert.match(rule('.tw-queue-preview'), /overflow:hidden|text-overflow:ellipsis|white-space:nowrap|display:flex/);
  assert.match(css, /\.tw-queue-preview\{/);
  assert.match(css, /\.task-workspace \.tw-steer-btn\{/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-steer-btn\{min-height:36px/);
});

test('t237 C6b2 L11: user_steered activity row is 你补充了… ≤40 chars', () => {
  const { applyActivityEvent } = presentation();
  const base = { id: 'run_s', status: 'running', activity: [], activity_total: 0 };
  const next = applyActivityEvent(base, 'user_steered', { summary: '你补充了：改成按月份汇总结果' });
  assert.equal(next.activity.length, 1);
  assert.equal(next.activity[0].kind, 'steer');
  assert.match(next.activity[0].label, /^你补充了/);
  assert.ok(next.activity[0].label.length <= 40);
  const long = applyActivityEvent(base, 'user_steered', { summary: '你补充了：' + '甲'.repeat(80) });
  assert.ok(long.activity[0].label.length <= 40);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(h(TaskRunView, {
    view: 'conversation',
    run: {
      id: 'run_steer', status: 'running', created_at: '2026-10-29T00:00:00Z',
      activity: [{ kind: 'steer', label: '你补充了：改成按月份汇总', status: 'completed', duration_ms: null }],
      activity_total: 1,
      message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
        { id: 'u', role: 'user', content: '分析', kind: 'original_request' },
        { id: 'a', role: 'assistant', content: '', kind: 'pending_response' },
      ] },
    },
  }));
  assert.match(html, /你补充了：改成按月份汇总/);
  assert.doesNotMatch(html, ENGLISH_UI);
  assert.doesNotMatch(html, /tool_[A-Za-z0-9]|call_[A-Za-z0-9]/);
});

test('t237 C6b2 L12: SSE auto-reconnect uses after=seq and Chinese restore copy', () => {
  const body = source('./TaskSession.tsx');
  assert.match(body, /events\?after=\$\{sequence\}/);
  assert.match(body, /setReconnect\(value => value \+ 1\)/);
  assert.match(body, /实时连接已断开，正在自动恢复…/);
  assert.match(body, /user_steered/);
  assert.match(body, /reasoning_summary|message_queued|message_drained/);
});

test('t237 C6b2 messages route proxies queue|steer|cancel_queue', () => {
  const route = readFileSync(new URL('../../../app/api/assistant/runs/[runId]/messages/route.ts', import.meta.url), 'utf8');
  assert.match(route, /\/v1\/runs\//);
  assert.match(route, /idempotency_key/);
  assert.match(route, /mode/);
});
