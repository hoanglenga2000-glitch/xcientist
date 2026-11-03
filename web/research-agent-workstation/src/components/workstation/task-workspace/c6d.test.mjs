import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// t237 C6d: chip/row design softs (a–c), FE minors (AbortSignal.any, answerStarted, pausing), backlog (meta zh, picker, aria).
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
const history = answer => ({ schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [
  { id: 'u1', role: 'user', content: '做一份拆分方案' }, { id: 'a1', role: 'assistant', content: answer }] });
const STAMPS = { created_at: '2026-10-29T10:00:00Z', started_at: '2026-10-29T10:00:30Z', completed_at: '2026-10-29T10:00:37Z' };
const ago = seconds => new Date(Date.now() - seconds * 1000).toISOString();
const view = r => renderToStaticMarkup(h(require('./TaskRunView.tsx').TaskRunView, { view: 'conversation', run: r }));
const file = { id: 'artifact_1', run_id: 'run_d', name: 'split_plan.md', bytes: 3200, sha256: 'a'.repeat(64), preview_kind: 'markdown', source_tool_call: 'artifact_publish' };

// ---- (a) file sources disclosure ----
test('C6d (a): 「已通过校验 · 查看来源」 — 校验 once; › chevron like the process row, native marker hidden', () => {
  const html = renderToStaticMarkup(h(require('./AttachmentChip.tsx').FileSources, { files: [file] }));
  const summary = (html.match(/<summary[^>]*>([^]*?)<\/summary>/) ?? [])[1] ?? '';
  const text = summary.replace(/<[^>]+>/g, '');
  assert.match(text, /^已通过校验 · 查看来源›$/);
  assert.equal((text.match(/校验/g) ?? []).length, 1);
  assert.match(summary, /<span class="tw-thinking-chevron" aria-hidden="true">›<\/span>/);
  assert.match(rule('.tw-file-sources>summary'), /list-style:none/);
  assert.match(css, /\.tw-file-sources>summary::-webkit-details-marker\{display:none\}/);
  assert.match(rule('.tw-file-sources[open]>summary .tw-thinking-chevron'), /transform:rotate\(90deg\)/);
  // reduced motion already drops the chevron transition for every .tw-thinking-chevron
  assert.match(css, /@media\(prefers-reduced-motion:reduce\)\{[^\n]*\.tw-thinking-chevron\{transition:none\}/);
});

// ---- (b) no red dot on failure ----
test('C6d (b): the failed process row has no trailing red dot; failure stays in text, colour and the sr status', () => {
  const failed = view({ id: 'run_f', status: 'failed', answer: '', ...STAMPS, completed_at: '2026-10-29T10:00:53Z', error_class: 'model_error', error_message: '模型调用失败' });
  assert.doesNotMatch(failed, /tw-thinking-dot/);
  assert.match(failed, /<span class="tw-thinking-text">这一步没有完成 · 用时 23s<\/span>/);
  assert.match(failed, /class="tw-sr-only" role="status"/);
  assert.doesNotMatch(css + source('./ThinkingRow.tsx'), /tw-thinking-dot/);
});

// ---- (c) step seconds sit next to the label ----
test('C6d (c): each step\'s seconds sit inline right after its label (baseline, small gap, tabular, muted)', () => {
  const html = view({ id: 'run_c', status: 'completed', answer: '好了', ...STAMPS, message_history: history('好了'),
    activity: [{ kind: 'tool', label: '读取资料', status: 'completed', duration_ms: 1500 }] });
  assert.match(html, /<span class="tw-activity-text"><span class="tw-activity-label">读取资料<\/span><span class="tw-activity-time">1s<\/span><\/span>/);
  const text = rule('.tw-activity-text');
  assert.match(text, /display:flex/); assert.match(text, /align-items:baseline/); assert.match(text, /gap:6px/); assert.match(text, /min-width:0/);
  assert.doesNotMatch(rule('.tw-activity-time'), /margin-left:auto/);
  assert.match(rule('.tw-activity-time'), /font-variant-numeric:tabular-nums/);
  assert.match(rule('.tw-activity-time'), /color:var\(--tw-muted\)/);
  assert.doesNotMatch(rule('.tw-activity-label'), /flex:1\b/);
  // no width-dependent override pushes it right again (393 and wide share the rule)
  assert.doesNotMatch(css, /\.tw-activity-time\{[^}]*margin-left:auto/);
});

// ---- minor 1: AbortSignal.any fallback ----
test('C6d minor 1: timeoutSignal works without AbortSignal.any / AbortSignal.timeout and never throws', async () => {
  const any = AbortSignal.any; const timeout = AbortSignal.timeout;
  try {
    delete AbortSignal.any;
    const { timeoutSignal } = require('./abort-signal.ts');
    const parent = new AbortController();
    const chained = timeoutSignal(60000, parent.signal);
    assert.equal(chained.aborted, false);
    parent.abort();
    assert.equal(chained.aborted, true);
    delete AbortSignal.timeout;
    const timed = timeoutSignal(5, new AbortController().signal);
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(timed.aborted, true);
    const alone = timeoutSignal(5);
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(alone.aborted, true);
    const already = new AbortController(); already.abort();
    assert.equal(timeoutSignal(1000, already.signal).aborted, true);
  } finally { AbortSignal.any = any; AbortSignal.timeout = timeout; }
  for (const name of ['./AttachmentChip.tsx', './DownloadFile.tsx']) {
    assert.doesNotMatch(source(name), /AbortSignal\.(?:any|timeout)\(/, name);
    assert.match(source(name), /timeoutSignal\(/, name);
  }
  const typesSrc = source('./types.ts');
  assert.match(typesSrc, /timeoutSignal\(15000/);
  assert.doesNotMatch(typesSrc.replace(/export function timeoutSignal[\s\S]*?^}/m, ''), /AbortSignal\.(?:any|timeout)\(/);
});

// ---- minor 2: answerStarted ----
test('C6d minor 2: a live continuation run with a stale run.answer has not started its answer — row stays open', () => {
  const { answerStarted } = presentation();
  const live = { id: 'run_cont', status: 'running', created_at: ago(20), started_at: ago(5), answer: '上一轮的答案', streaming_text: '', message_history: history('上一轮的答案') };
  assert.equal(answerStarted(live), false);
  assert.equal(answerStarted({ ...live, streaming_text: '本轮' }), true);
  assert.equal(answerStarted({ ...live, status: 'completed' }), true);
  const html = view(live);
  assert.match(html, /data-testid="tw-process-line" aria-expanded="true"/);
  assert.match(html, /data-brand-loader/);
  assert.match(source('./TaskRunView.tsx'), /answerStarted=\{answerStarted\(run\)\}/);
});

// ---- minor 3: pausing is live everywhere ----
test('C6d minor 3: one shared live set — pausing is live for the row, the thinking state, the task list, the session and Stop', () => {
  const p = presentation();
  assert.deepEqual([...p.LIVE_RUN_STATUSES].sort(), ['pausing', 'planning', 'queued', 'recovering', 'running', 'verifying']);
  assert.deepEqual([...p.STOPPABLE_RUN_STATUSES].sort(), [...p.LIVE_RUN_STATUSES, 'waiting_approval'].sort());
  for (const status of p.LIVE_RUN_STATUSES) {
    assert.equal(p.isLiveTask(status), true, status);
    assert.equal(p.thinkingRowState({ id: 'r', status, started_at: ago(3) }).live, true, status);
    assert.match(view({ id: 'r_' + status, status, created_at: ago(9), started_at: ago(3) }), /tw-message tw-assistant is-live/, status);
  }
  for (const name of ['./ThinkingRow.tsx', './TaskRunView.tsx', './TaskSession.tsx', './TaskControls.tsx', './TaskWorkspace.tsx'])
    assert.doesNotMatch(source(name), /\[\s*'queued',[^\]]*'pausing'|'recovering'\s*\]/, `${name} keeps its own live list`);
});

// ---- backlog 1: Chinese meta ----
test('C6d backlog 1: page / OG description is Chinese; the brand name stays', () => {
  const layout = readFileSync(app('app/layout.tsx'), 'utf8');
  assert.doesNotMatch(layout, /Auditable scientific research/);
  const description = (layout.match(/const DESCRIPTION = "([^"]+)"/) ?? [])[1] ?? '';
  assert.match(description, /^[^A-Za-z]*[一-鿿][^A-Za-z]*$/);
  assert.equal((layout.match(/description: DESCRIPTION/g) ?? []).length, 2, 'page + openGraph description');
  assert.match(layout, /title: "DeepEvo"/);
});

// ---- backlog 2: picker keeps the model name's first word ----
test('C6d backlog 2: ≤760px drops only the Claude vendor before a tier name — other names keep their first word', () => {
  const { Picker } = require('./ComposerPicker.tsx');
  const pick = text => renderToStaticMarkup(h(Picker, { id: 'p', label: '本次使用的模型', value: 'x', text, groups: [], onChange: () => undefined }));
  assert.match(pick('Claude Sonnet 5.5'), /<span class="tw-picker-vendor">Claude <\/span>Sonnet/);
  for (const text of ['Grok Code Fast 1', 'Kimi K2 0905', '我的 助手 2', 'Gemini Pro 2.5']) {
    const html = pick(text);
    assert.doesNotMatch(html, /tw-picker-vendor/, text);
    assert.match(html.replace(/<[^>]+>/g, ''), new RegExp('^' + text.replace(/\./g, '\\.')), text);
  }
});

// ---- backlog 3: one accessible name ----
test('C6d backlog 3: no doubled accessible names — picker name not repeated, no aria-label equal to visible text', () => {
  const { Picker, IntensityPicker } = require('./ComposerPicker.tsx');
  const intensity = renderToStaticMarkup(h(IntensityPicker, { id: 'i', value: '中', onChange: () => undefined }));
  const label = (intensity.match(/<button[^>]*aria-label="([^"]*)"/) ?? [])[1];
  assert.equal(label, '强度 · 中');
  const model = renderToStaticMarkup(h(Picker, { id: 'p', label: '本次使用的模型', value: 'x', text: 'Claude Sonnet 5.5', groups: [], onChange: () => undefined }));
  assert.equal((model.match(/<button[^>]*aria-label="([^"]*)"/) ?? [])[1], '本次使用的模型：Claude Sonnet 5.5');
  // 立即发送 has visible (or sr-only) text; it carries no aria-label duplicating it
  assert.doesNotMatch(source('./TaskSession.tsx'), /aria-label="立即发送"/);
  assert.match(source('./TaskSession.tsx'), /<span className="tw-steer-label">立即发送<\/span>/);
});
