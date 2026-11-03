import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// t237 C6-answer (SPEC 6558ff9b / CP 42c41971): ThinkingRow, borderless answer, attachment chips, delivery_mode.
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
const brandCss = readFileSync(app('components/workstation/deepevo-brand.css'), 'utf8');
const rule = selector => { const start = css.indexOf(selector + '{'); return start < 0 ? '' : css.slice(start, css.indexOf('}', start) + 1); };
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const presentation = () => require('./task-presentation.ts');

const history = (answer = '', extra = []) => ({ schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [
  { id: 'u1', role: 'user', content: '解释一下 Parquet' }, ...extra, { id: 'a1', role: 'assistant', content: answer }] });
const ago = seconds => new Date(Date.now() - seconds * 1000).toISOString();
const STAMPS = { created_at: '2026-10-29T10:00:00Z', started_at: '2026-10-29T10:00:30Z', completed_at: '2026-10-29T10:00:37Z' };
const TOOLS = [
  { kind: 'tool', label: '读取资料', status: 'completed', duration_ms: 1500 },
  { kind: 'tool', label: '整理要点', status: 'completed', duration_ms: 1700 },
  { kind: 'tool', label: '撰写回答', status: 'completed', duration_ms: 1900 },
];
const run = extra => ({ id: 'run_ans', status: 'running', created_at: ago(20), started_at: ago(8), message_history: history(), ...extra });
const done = extra => run({ status: 'completed', answer: 'Parquet 是列式存储格式。', ...STAMPS, message_history: history('Parquet 是列式存储格式。'), ...extra });
const view = r => renderToStaticMarkup(h(require('./TaskRunView.tsx').TaskRunView, { view: 'conversation', run: r }));
const rowText = html => (html.match(/<span class="tw-thinking-text">([^<]*)</) ?? [])[1] ?? null;
const visible = html => html.replace(/<[^>]+>/g, ' ');
const file = (index, extra = {}) => ({ id: `artifact_${index}`, run_id: 'run_ans', name: `kxxz_c6_part${index}.md`, bytes: 3200, sha256: 'a'.repeat(64), preview_kind: 'markdown', source_tool_call: 'artifact_publish', ...extra });

// ---- CP-A3 ThinkingRow copy ----
test('C6-answer H_a13: live copy is 思考中… / 正在{动作}… (≤14 chars) + a live Ns clock; under 1s shows no seconds', () => {
  const { thinkingRowState } = presentation();
  const now = Date.parse('2026-10-29T10:00:38Z');
  const base = { id: 'r', status: 'running', created_at: '2026-10-29T10:00:00Z', started_at: '2026-10-29T10:00:30Z' };
  assert.deepEqual(thinkingRowState(base, now), { phase: 'thinking', text: '思考中…', clock: '8s', steps: 0, live: true });
  const acting = thinkingRowState({ ...base, current_activity: { kind: 'tool', label: '读取记忆', status: 'running' } }, now);
  assert.equal(acting.text, '正在读取记忆…');
  const long = thinkingRowState({ ...base, current_activity: { kind: 'tool', label: '比对二十个公开数据集的许可证条款', status: 'running' } }, now);
  assert.ok(long.text.length <= 14, long.text);
  assert.match(long.text, /^正在.+…$/);
  // The runtime's fixed step labels are not 「动作」: no fake skeleton text while thinking.
  assert.equal(thinkingRowState({ ...base, current_activity: { kind: 'step', label: '调用工具执行任务', status: 'running' } }, now).text, '思考中…');
  assert.equal(thinkingRowState({ ...base, current_activity: { kind: 'tool', label: 'file_read', status: 'running' } }, now).text, '正在处理一步…');
  assert.equal(thinkingRowState(base, Date.parse('2026-10-29T10:00:30.900Z')).clock, null);
  assert.equal(thinkingRowState(base, Date.parse('2026-10-29T10:01:35Z')).clock, '1m 05s');
  // First answer character ⇒ 正在整理答案….
  assert.equal(thinkingRowState({ ...base, streaming_text: '你' }, now).text, '正在整理答案…');
  const html = view(run({ current_activity: { kind: 'tool', label: '读取记忆', status: 'running' } }));
  assert.match(html, /<span class="tw-thinking-text">正在读取记忆…<\/span><span class="tw-thinking-clock" aria-hidden="true"> · [89]s<\/span>/);
});

test('C6-answer H_a21: the clock starts at started_at, not created_at; queued shows 排队中 with no clock', () => {
  const { thinkingRowState, runStartedAt } = presentation();
  const now = Date.parse('2026-10-29T10:00:38Z');
  assert.equal(thinkingRowState({ id: 'r', status: 'running', created_at: '2026-10-29T10:00:00Z', started_at: '2026-10-29T10:00:30Z' }, now).clock, '8s');
  // Old snapshots without started_at fall back to created_at.
  assert.equal(thinkingRowState({ id: 'r', status: 'running', created_at: '2026-10-29T10:00:00Z' }, now).clock, '38s');
  assert.equal(runStartedAt({ created_at: 'x', started_at: '' }), null);
  const queued = thinkingRowState({ id: 'r', status: 'queued', created_at: '2026-10-29T10:00:00Z', started_at: '' }, now);
  assert.deepEqual([queued.text, queued.clock], ['排队中', null]);
  const html = view(run({ status: 'queued', started_at: '' }));
  assert.equal(rowText(html), '排队中');
  assert.doesNotMatch(html, /tw-thinking-clock|data-brand-loader/);
  // Completed time = started_at → completed_at (7s), not the 37s since creation.
  assert.equal(rowText(view(done({ activity: TOOLS, activity_total: 3 }))), '工作了 7s · 3 步');
});

test('C6-answer H_a15 / H_b15′: done ⇒ 「工作了 Ns · N 步」 (tools) or 「思考了 Ns」; <1s ⇒ 不到 1s; N 步 = expanded timeline rows', () => {
  const { thinkingRowState } = presentation();
  assert.equal(thinkingRowState(done({ activity: TOOLS })).text, '工作了 7s · 3 步');
  assert.equal(thinkingRowState(done()).text, '思考了 7s');
  const quick = { ...STAMPS, completed_at: '2026-10-29T10:00:30.600Z' };
  assert.equal(thinkingRowState(done({ ...quick })).text, '思考了不到 1s');
  assert.equal(thinkingRowState(done({ ...quick, activity: TOOLS })).text, '工作了不到 1s · 3 步');
  // Merged repeats count once: the header N equals the rendered rows.
  const repeats = [...TOOLS, { kind: 'tool', label: '撰写回答', status: 'completed', duration_ms: 100 }];
  const html = view(done({ activity: repeats, activity_total: 4 }));
  const rows = (html.match(/data-testid="tw-activity-row"/g) ?? []).length;
  assert.equal(rows, 3);
  assert.equal(rowText(html), `工作了 7s · ${rows} 步`);
  assert.match(html, /撰写回答 ×2/);
  for (const text of [rowText(html), rowText(view(done()))]) assert.doesNotMatch(text, /(^|\D)0s|\.\d+s|\d+:\d{2}/);
  // No fake 理解/执行/核对/交付 skeleton anywhere.
  assert.doesNotMatch(visible(html), /理解目标|调用工具执行任务|验证结果与文件|发布可预览下载产物/);
});

test('C6-answer H_a16: one time display on screen — no footer .tw-worked, no 「已完成 · … · m:ss」', () => {
  const html = view(done({ activity: TOOLS, activity_total: 3, message_history: history('第二次', [{ id: 'a0', role: 'assistant', content: '第一次' }, { id: 'u2', role: 'user', content: '再来' }]) }));
  const times = visible(html).match(/工作了|思考了|用时|\d+:\d{2}|\d+s\b/g) ?? [];
  // 工作了 7s plus the per-step durations inside the folded timeline (1s ×3), which are hidden.
  const shown = visible(html.replace(/<div id="[^"]+" class="tw-thinking-detail" hidden="">[^]*?<\/ol>/, '')).match(/工作了|思考了|用时|\d+:\d{2}|\d+s\b/g) ?? [];
  assert.deepEqual(shown, ['工作了', '7s']);
  assert.ok(times.length > shown.length, 'timeline durations exist but stay folded');
  assert.doesNotMatch(html, /tw-worked|已完成 · \d+ 步 · \d+:\d+/);
  assert.equal((html.match(/data-testid="tw-thinking-row"/g) ?? []).length, 1);
});

test('C6-answer H_a19 / H_b17: stopped 「已取消 · 用时 Ns」, failed 「这一步没有完成 · 用时 Ns」 + the fail card; never 工作了', () => {
  const stopped = view(run({ status: 'cancelled', answer: '已产出的部分……', ...STAMPS, completed_at: '2026-10-29T10:00:48Z', message_history: history('已产出的部分……') }));
  assert.equal(rowText(stopped), '已取消 · 用时 18s');
  assert.match(stopped, /已产出的部分/);
  assert.doesNotMatch(stopped, /工作了|思考了|data-brand-loader/);
  const failed = view(run({ status: 'failed', answer: '', ...STAMPS, completed_at: '2026-10-29T10:00:53Z', error_class: 'model_error', error_message: '模型调用失败' }));
  assert.equal(rowText(failed), '这一步没有完成 · 用时 23s');
  assert.doesNotMatch(failed, /tw-thinking-dot/);  // C6d (b): no trailing red dot
  assert.doesNotMatch(failed, /工作了|思考了/);
  // Model failure keeps its card under the red row.
  const model = view(run({ status: 'blocked', error_class: 'model_unavailable', answer: 'Model execution is unavailable: x', ...STAMPS }));
  assert.match(model, /data-testid="tw-model-fail"/);
  assert.ok(model.indexOf('tw-thinking-row') < model.indexOf('tw-model-fail'));
  assert.match(rowText(model), /^这一步没有完成/);
  assert.match(rule('.tw-thinking-row.is-failed .tw-thinking-text'), /color:var\(--tw-danger-strong\)/);
});

test('C6-answer H_a20: no English Thinking / Working / Thought for and no explainer copy in the row', () => {
  for (const r of [run(), run({ streaming_text: 'x' }), run({ status: 'queued' }), done(), done({ activity: TOOLS }), run({ status: 'cancelled', ...STAMPS }), run({ status: 'failed', ...STAMPS })]) {
    const text = visible(view(r));
    assert.doesNotMatch(text, /\b(Thinking|Working|Thought for|worked for)\b/i);
    assert.doesNotMatch(text, /说明|提示：|点击展开/);
  }
});

// ---- H_a17 auto-collapse + H_a18 a11y ----
test('C6-answer H_a17: live rows start open, history starts folded; first char folds after ≤800ms; a user toggle wins', () => {
  const { AUTO_COLLAPSE_MS, thinkingRowOpen } = presentation();
  assert.ok(AUTO_COLLAPSE_MS >= 400 && AUTO_COLLAPSE_MS <= 800, String(AUTO_COLLAPSE_MS));
  assert.equal(AUTO_COLLAPSE_MS, 600);
  assert.equal(thinkingRowOpen(null, true), true);
  assert.equal(thinkingRowOpen(null, false), false);
  assert.equal(thinkingRowOpen(true, false), true, 'manual open survives the auto fold');
  assert.equal(thinkingRowOpen(false, true), false, 'manual close survives a live run');
  assert.match(view(run({ activity: TOOLS })), /aria-expanded="true"/);
  assert.match(view(run({ streaming_text: '你' })), /aria-expanded="false"/);
  assert.match(view(done({ activity: TOOLS })), /aria-expanded="false"/);
  const row = source('./ThinkingRow.tsx');
  assert.match(row, /window\.setTimeout\(\(\) => setAutoOpen\(false\), AUTO_COLLAPSE_MS\)/);
  assert.match(row, /useState\(\(\) => live && !answerStarted\)/);
  assert.match(row, /onClick=\{\(\) => setManual\(!open\)\}/);
});

test('C6-answer H_a18: a real <button aria-expanded aria-controls>; polite status names the phase only; the clock is aria-hidden', () => {
  const html = view(run({ current_activity: { kind: 'tool', label: '读取记忆', status: 'running' }, activity: TOOLS }));
  const id = html.match(/<button type="button" class="tw-process-line" data-testid="tw-process-line" aria-expanded="true" aria-controls="([^"]+)"/)?.[1];
  assert.ok(id);
  assert.match(html, new RegExp(`<div id="${id}" class="tw-thinking-detail"`));
  const status = html.match(/<span class="tw-sr-only" role="status" aria-live="polite">([^<]*)<\/span>/)?.[1];
  assert.equal(status, '开始思考');
  assert.doesNotMatch(status, /\d/);
  assert.match(html, /<span class="tw-thinking-clock" aria-hidden="true">/);
  const { thinkingAnnouncement } = presentation();
  assert.deepEqual(['thinking', 'answering', 'done'].map(thinkingAnnouncement), ['开始思考', '正在整理答案', '已完成']);
  assert.match(css, /\.task-workspace :focus-visible\{outline:2px solid var\(--tw-focus-ring\)/);
});

test('C6-answer H_a14: loader = mark-small rotate only (1.8s ease-in-out, 72°); shimmer on the text only; reduced motion stops both', () => {
  const html = view(run());
  assert.match(html, /class="deepevo-mark deepevo-loader is-small tw-thinking-loader"/);
  assert.match(brandCss, /@keyframes deepevo-loader-small\{from\{transform:rotate\(0deg\)\}to\{transform:rotate\(72deg\)\}\}/);
  assert.match(rule('.deepevo-loader') || brandCss, /1\.8s ease-in-out infinite/);
  const shimmer = rule('.tw-thinking-row.is-live:not(.is-queued) .tw-thinking-text');
  assert.match(shimmer, /tw-shimmer 1\.8s linear infinite/);
  assert.match(shimmer, /var\(--tw-label\)[^;]*var\(--tw-ink\)/);
  assert.doesNotMatch(rule('.tw-thinking-icon'), /animation/);
  const reduced = css.match(/@media\(prefers-reduced-motion:reduce\)\{\.tw-thinking-text.*\}\}/)?.[0] ?? '';
  assert.match(reduced, /animation:none/);
  assert.match(reduced, /color:var\(--tw-ink\)/);
  assert.match(brandCss, /@media \(prefers-reduced-motion: reduce\)\{\.deepevo-loader,\.deepevo-loader\.is-small\{animation:none\}\}/);
  // SPEC §5.2: the streaming cursor blinks in accent, and holds still under reduced motion.
  assert.match(rule('.tw-message.is-live .tw-md>p:last-child::after'), /background:var\(--tw-accent\)[^}]*animation:tw-blink/);
  assert.match(reduced, /\.tw-message\.is-live \.tw-md>p:last-child::after\{animation:none\}/);
});

// ---- CP-A2 answer + attachments ----
test('C6-answer H_a7 / H_a9: the answer is borderless Markdown; no progress card, phase strip or 本次实际模型', () => {
  const html = view(done({ activity: TOOLS, model: 'gpt-4o-mini', model_provider: 'openai' }));
  assert.match(html, /<div class="tw-md tw-msg-answer" data-testid="tw-answer"><p>Parquet 是列式存储格式。<\/p><\/div>/);
  assert.doesNotMatch(html, /tw-progress-card|tw-phases|本次实际模型/);
  for (const selector of ['.tw-md', '.tw-message']) assert.doesNotMatch(rule(selector), /border:|box-shadow|background/, selector);
  assert.doesNotMatch(rule('.tw-assistant small'), /border|background/);
  assert.doesNotMatch(css, /\.tw-assistant\{[^}]*(border|background|box-shadow)/);
});

test('C6-answer H_a8: the last assistant message is never hidden; a bare file name is not process chatter', () => {
  const { isProcessLeak } = presentation();
  const answer = 'Your secret code is stored in secret_code.txt: 9X-1';
  assert.equal(isProcessLeak(answer), false);
  assert.equal(isProcessLeak("I'll read inputs/a.md first."), true);
  assert.equal(isProcessLeak("I'll read inputs/a.md first.", { last: true }), false);
  const html = view(done({ answer, message_history: history(answer) }));
  assert.match(html, /<article class="tw-message tw-assistant[^"]*"[^]*Your secret code is stored in secret_code\.txt: 9X-1/);
  const chatter = "I'll check outputs/report.md now.";
  const last = view(done({ answer: chatter, message_history: history(chatter, [{ id: 'a0', role: 'assistant', content: "Let me look at inputs/x.csv first." }]) }));
  assert.match(last, /<div class="tw-md tw-msg-answer" data-testid="tw-answer"><p>I&#x27;ll check outputs\/report\.md now\.<\/p><\/div>/);
  assert.match(last, /<p class="tw-process-note">Let me look at inputs\/x\.csv first\.<\/p>/);
});

test('C6-answer H_a10: inline ⇒ zero file elements; deliveryMode infers old snapshots', () => {
  const { deliveryMode } = presentation();
  const html = view(done({ delivery_mode: 'inline', artifacts: [] }));
  assert.doesNotMatch(html, /tw-filechip|tw-attachments|tw-file\b|查看来源/);
  assert.equal(deliveryMode({ artifacts: [] }), 'inline');
  assert.equal(deliveryMode({ artifacts: [file(1)], answer: '摘要' }), 'inline+file');
  assert.equal(deliveryMode({ artifacts: [file(1)], answer: '' }), 'file_only');
  assert.equal(deliveryMode({ delivery_mode: 'inline', artifacts: [file(1)] }), 'inline+file', 'a delivered file upgrades inline');
  assert.equal(deliveryMode({ delivery_mode: 'file_only', artifacts: [file(1)] }), 'file_only');
});

test('C6-answer H_a11 / H_a11b: borderless chip, one-line name, keyboard-reachable 预览/下载, ≥44px on 393; ≤3 then 「还有 N 个文件」', () => {
  const html = view(done({ delivery_mode: 'inline+file', answer: '拆分方案要点：……', message_history: history('拆分方案要点：……'), artifacts: [1, 2, 3, 4, 5].map(index => file(index)) }));
  assert.equal((html.match(/data-testid="tw-filechip"/g) ?? []).length, 5);
  const top = html.slice(0, html.indexOf('tw-attachments-more'));
  assert.equal((top.match(/data-testid="tw-filechip"/g) ?? []).length, 3);
  assert.match(html, /<details class="tw-details tw-attachments-more"><summary>还有 2 个文件<\/summary>/);
  assert.match(html, /<button type="button" class="tw-filechip-action is-preview" aria-expanded="false">预览<\/button>/);
  assert.match(html, /<a class="tw-filechip-action" href="\/api\/assistant\/artifacts\/artifact_1\?download=1"[^>]*>下载<\/a>/);
  assert.match(html, /<span class="tw-filechip-name" title="kxxz_c6_part1\.md">kxxz_c6_part1\.md<\/span><small>Markdown · 3 KB<\/small>/);
  const chip = rule('.tw-filechip');
  assert.match(chip, /border:0/);
  assert.match(chip, /background:var\(--tw-wash\)/);
  assert.match(chip, /border-radius:10px/);
  assert.doesNotMatch(chip, /box-shadow|border:1px/);
  const name = rule('.tw-filechip-name');
  for (const part of [/white-space:nowrap/, /overflow:hidden/, /text-overflow:ellipsis/]) assert.match(name, part);
  assert.match(rule('.tw-filechip-icon'), /width:32px;height:32px/);
  assert.match(css, /@media\(max-width:760px\)\{[^\n]*\.tw-filechip-actions\{flex-basis:100%;justify-content:flex-end\}\.task-workspace \.tw-filechip-action\{min-height:44px/);
  assert.match(css, /@media\(max-width:760px\)\{\.task-workspace \.tw-thinking-row>\.tw-process-line\{min-height:44px\}/);
});

test('C6-answer H_a12: process row → inline summary → chips → 查看来源 → actions; an answer-less file run gets 「已生成 …」', () => {
  const html = view(done({ delivery_mode: 'inline+file', answer: '已生成拆分方案，要点如下：……', message_history: history('已生成拆分方案，要点如下：……'), activity: TOOLS, artifacts: [file(1)] }));
  const order = ['tw-thinking-row', '已生成拆分方案', 'data-testid="tw-filechip"', '已通过校验 · 查看来源', 'tw-msg-actions'].map(mark => html.indexOf(mark));
  assert.ok(order.every(index => index >= 0), String(order));
  assert.deepEqual([...order].sort((a, b) => a - b), order);
  assert.doesNotMatch(html, /<details class="tw-details tw-file-sources" open/);
  const bare = view(done({ delivery_mode: 'file_only', answer: '', message_history: history(''), artifacts: [file(1), file(2)] }));
  assert.match(bare, /<div class="tw-md tw-msg-answer" data-testid="tw-answer"><p>已生成 <code>kxxz_c6_part1\.md<\/code>、<code>kxxz_c6_part2\.md<\/code>。<\/p><\/div>/);
  assert.ok(bare.indexOf('已生成') < bare.indexOf('data-testid="tw-filechip"'));
  const { filesSummaryLine } = presentation();
  assert.equal(filesSummaryLine(['a.md', 'b.md', 'c.md', 'd.md']), '已生成 `a.md`、`b.md`、`c.md` 等 4 个文件。');
  assert.equal(filesSummaryLine([]), '');
});

// ---- CP-A4 layout ----
test('C6-answer H_a22 / H_a24: reading width ≤760; row and answer share one left edge; nothing widens the 393 page', () => {
  assert.match(css, /--tw-read:760px/);
  assert.match(css, /\.tw-chat\{max-width:var\(--tw-read\)/);
  // Same parent, no indent: the row and the Markdown both start at the bubble's content edge.
  const html = view(done({ activity: TOOLS }));
  assert.match(html, /<article class="tw-message tw-assistant"[^>]*><div class="tw-speaker">[^]*?<\/div><div class="tw-thinking-row is-done"[^]*?<\/div><\/div><div class="tw-md tw-msg-answer" data-testid="tw-answer">/);
  assert.match(html, /data-testid="tw-answer"/);
  assert.match(html, /tw-msg-answer/);
  for (const selector of ['.tw-thinking-row', '.tw-md']) assert.doesNotMatch(rule(selector), /margin-left|padding-left|margin:\d+px \d+px \d+px [1-9]/, selector);
  assert.match(rule('.task-workspace .tw-thinking-row>.tw-process-line'), /padding:2px 8px 2px 0/);
  for (const selector of ['.tw-thinking-row', '.tw-filechip', '.tw-filechip-row', '.tw-filechip-text', '.tw-attachments', '.tw-thinking-detail']) assert.match(rule(selector), /min-width:0/, selector);
  assert.match(rule('.tw-thinking-row'), /max-width:100%/);
  assert.match(rule('.tw-thinking-detail'), /width:calc\(100% - 7px\)/);
  assert.match(rule('.tw-filechip-small'), /overflow:auto/);
  assert.match(rule('.tw-md-table'), /overflow-x:auto/);
});

test('C6-answer H_a23: light/dark muted + label tokens on the page background stay ≥4.5:1', () => {
  const lum = hex => {
    const [r, g, b] = [1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16) / 255).map(c => c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const token = (scope, name) => scope.match(new RegExp(`--tw-${name}:(#[0-9A-Fa-f]{6})`))?.[1];
  const light = css.slice(0, css.indexOf('}'));
  const dark = css.slice(css.indexOf('[data-theme="dark"] .task-workspace{'));
  for (const scope of [light, dark]) {
    const bg = token(scope, 'bg').replace('#ffffff', '#FFFFFF');
    for (const name of ['ink', 'muted', 'label', 'accent']) assert.ok(ratio(token(scope, name), bg) >= 4.5, `${name} ${token(scope, name)} on ${bg}`);
  }
  assert.equal(token(light, 'accent'), '#2F855A');
  assert.equal(token(dark, 'accent'), '#4FB37E');
});

// ---- softs ----
test('C6-answer S_a1: a small text deliverable (≤2KB / ≤40 lines) previews inline under its chip', () => {
  const { smallPreviewable, smallPreviewText } = presentation();
  assert.equal(smallPreviewable({ name: 'secret_code.txt', bytes: 51 }), true);
  assert.equal(smallPreviewable({ name: 'plan.md', bytes: 2048 }), true);
  assert.equal(smallPreviewable({ name: 'plan.md', bytes: 2049 }), false);
  assert.equal(smallPreviewable({ name: 'model.pkl', bytes: 100 }), false);
  assert.equal(smallPreviewText('a\nb\n'), 'a\nb');
  assert.equal(smallPreviewText(Array.from({ length: 41 }, (_, i) => String(i)).join('\n')), null);
  assert.match(source('./AttachmentChip.tsx'), /<pre className="tw-filechip-small" data-testid="tw-filechip-small">/);
});

test('C6-answer S_a2: no regenerate backend ⇒ the action bar keeps copy only (no dead button)', () => {
  const html = view(done());
  assert.match(html, /aria-label="复制"/);
  assert.doesNotMatch(html, /重新生成/);
});

test('C6-answer S_b2: dark 无痕 accent rule still wins the cascade', () => {
  const dark = '[data-theme="dark"] .task-workspace .tw-incognito[aria-pressed="true"]{border-color:var(--tw-accent);color:var(--tw-accent)}';
  const at = css.indexOf(dark);
  assert.ok(at >= 0);
  assert.doesNotMatch(css.slice(at + dark.length), /\.tw-incognito\[aria-pressed="true"\]/);
});
