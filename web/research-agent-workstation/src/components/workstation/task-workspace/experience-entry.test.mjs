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
const src = name => readFileSync(new URL(`./${name}`, import.meta.url), 'utf8');
const RID = 'run_' + 'd'.repeat(32);

test('t237 C3.11: 进化 tab entry — one line + .tw-primary 用经验库开发; empty adds 还没有经验。 + 开始对话; old placeholder gone', () => {
  const { ExperienceEntryView, EXPERIENCE_COPY } = require('./ExperiencePlan.tsx');
  // C4b E1/E3 supersede the exact markup (one state, no <h2>); see the C4b.4 test below.
  const full = renderToStaticMarkup(h(ExperienceEntryView, { count: 3 }));
  assert.match(full, /<p class="tw-experience-line">从已有经验开始一次模型开发。<\/p><a class="tw-primary" href="\/workspace\?new=1&amp;intent=experience">用经验库开发<\/a>/);
  const empty = renderToStaticMarkup(h(ExperienceEntryView, { count: 0 }));
  assert.match(empty, /<div class="tw-empty-hint"><p>还没有经验。<\/p><a class="tw-button" href="\/workspace\?new=1">开始对话<\/a><\/div><\/section>$/);
  // An unreadable count shows the 有经验 state, never an error on the entry page.
  assert.equal(renderToStaticMarkup(h(ExperienceEntryView, { count: null })), full);
  assert.equal(EXPERIENCE_COPY.chip, '经验库');
  const workspace = src('TaskWorkspace.tsx');
  assert.doesNotMatch(workspace, /在对话里说出想改进的目标/);
  assert.match(workspace, /tab === 'evolution' \? <ExperienceEntry \/>/);
  assert.match(src('ExperiencePlan.tsx'), /'\/api\/assistant\/experience\/summary'/);
  assert.doesNotMatch(full + empty, /Evolution|MCGS|dry-run|Gate|lineage|HPC|route|blocked/);
});

test('t237 C3.11: 经验库 chip (BookMarked, removable, aria-label 经验库，移出) beside the project chip; run carries intent=experience', () => {
  const { ExperienceChip } = require('./ExperiencePlan.tsx');
  const chip = renderToStaticMarkup(h(ExperienceChip, { onRemove() {} }));
  assert.match(chip, /^<span class="tw-chip tw-experience-chip"><svg[^>]*lucide-book-marked[^>]*>[^]*?<\/svg><span class="tw-chip-name">经验库<\/span><button type="button" aria-label="经验库，移出"><svg[^]*?<\/svg><\/button><\/span>$/);
  const { DraftProjectChip } = require('./ProjectDraft.tsx');
  const both = renderToStaticMarkup(h(DraftProjectChip, { project: { id: 'p', name: '肺结节' }, onRemove() {}, children: h(ExperienceChip, { onRemove() {} }) }));
  assert.match(both, /^<div class="tw-chips tw-draft-project"><span class="tw-chip">[^]*?肺结节[^]*?<\/span><span class="tw-chip tw-experience-chip">/);
  const { experienceIntent } = require('./ExperiencePlan.tsx');
  assert.equal(experienceIntent(new URLSearchParams('new=1&intent=experience')), 'experience');
  assert.equal(experienceIntent(new URLSearchParams('new=1&intent=train')), '');
  assert.equal(experienceIntent(new URLSearchParams('new=1')), '');
  const workspace = src('TaskWorkspace.tsx');
  assert.match(workspace, /<ExperienceChip onRemove=\{removeDraftIntent\} \/>/);
  assert.match(workspace, /intent=experience/, 'the intent survives task creation into the URL');
  const session = src('TaskSession.tsx');
  assert.match(session, /\.\.\.\(intent \? \{ intent \} : \{\}\)/);
  assert.match(session, /<ExperienceChip onRemove=\{removeIntent\} \/>/);
  assert.match(session, /'plan_proposed'/);
});

const plan = (extra = {}) => ({ references: [], steps: ['整理你的目标和已有条件', '对照相近的经验给出改进方案'], needs_compute: false, ...extra });

test('t237 C3.11: plan card copy — 计划 / 参考的经验 / 没有找到相近的经验。 / 步骤 / 需要的资源 / buttons', () => {
  const { PlanCard } = require('./ExperiencePlan.tsx');
  const html = renderToStaticMarkup(h(PlanCard, { plan: plan(), decision: '', onDecide() {} }));
  assert.equal(html, '<section class="tw-confirm tw-inline-confirm tw-plan-card" aria-label="计划"><h2>计划</h2><p class="tw-sr-only" role="status">计划已生成</p>'
    + '<h3>参考的经验</h3><p>没有找到相近的经验。</p><h3>步骤</h3><ol><li>整理你的目标和已有条件</li><li>对照相近的经验给出改进方案</li></ol>'
    + '<h3>需要的资源</h3><p>不使用算力</p><div class="tw-actions"><button type="button" class="tw-primary">确认计划</button><button type="button">取消</button></div></section>');
  const hits = renderToStaticMarkup(h(PlanCard, { plan: plan({ references: [{ title: 'U-Net 分割', summary: '多尺度更稳' }], needs_compute: true }), decision: '', onDecide() {} }));
  assert.match(hits, /<ul class="tw-plan-refs"><li><span>U-Net 分割<\/span><span class="tw-muted">多尺度更稳<\/span><\/li><\/ul>/);
  assert.match(hits, /<p>需要算力，确认后另行审批<\/p>/);
  const confirmed = renderToStaticMarkup(h(PlanCard, { plan: plan(), decision: 'confirm', onDecide() {} }));
  assert.match(confirmed, /<p class="tw-plan-status" role="status" tabindex="-1">已确认<\/p><\/section>$/);
  assert.doesNotMatch(confirmed, /确认计划|<button/);
  const cancelled = renderToStaticMarkup(h(PlanCard, { plan: plan(), decision: 'cancel', onDecide() {} }));
  assert.equal(cancelled, '<section class="tw-confirm tw-inline-confirm tw-plan-card" aria-label="计划"><p class="tw-plan-status" role="status" tabindex="-1">已取消计划。</p></section>');
  for (const out of [html, hits, confirmed]) assert.doesNotMatch(out, /run_|tool|route|HPC|train|训练|写入/);
});

test('t237 C3.11: plan decision POSTs {decision} to the run plan proxy; card renders under the answer; no train/HPC UI', () => {
  const source = src('ExperiencePlan.tsx');
  assert.match(source, /`\/api\/assistant\/runs\/\$\{encodeURIComponent\(runId\)\}\/plan`, jsonPost\(\{ decision \}\)/);
  assert.doesNotMatch(source, /approval|hpc|training|evolution_/i);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const run = { id: RID, status: 'completed', prompt: '用经验库', answer: '没有找到相近的经验。我先列了一个计划，你确认后我再继续。', intent: 'experience', plan_proposed: plan(), plan_decision: '' };
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run }));
  assert.match(html, /我先列了一个计划[^]*?<\/article><section class="tw-confirm tw-inline-confirm tw-plan-card" aria-label="计划">/);
  assert.match(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { ...run, plan_decision: 'confirm' } })), /已确认/);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { ...run, plan_proposed: undefined } })), /tw-plan-card/);
});


test('t237 C4: confirmActionLabel experience_compute is Chinese; no HPC/tool jargon in default plan HTML', () => {
  const { confirmActionLabel } = require('./task-presentation.ts');
  assert.equal(confirmActionLabel('experience_compute'), '使用算力继续');
  assert.equal(confirmActionLabel('file_write'), '写入这次任务的文件');
  const { PlanCard, EXPERIENCE_COPY } = require('./ExperiencePlan.tsx');
  const withCompute = renderToStaticMarkup(h(PlanCard, {
    plan: plan({ needs_compute: true, references: [{ title: 'U-Net 分割', summary: '多尺度更稳' }] }),
    decision: '', onDecide() {},
  }));
  assert.match(withCompute, /需要算力，确认后另行审批/);
  assert.doesNotMatch(withCompute, /Evolution|MCGS|dry-run|Gate|lineage|HPC|route|blocked|training_route|hpc_|experience_compute|\/hpc2hdd/);
  assert.equal(EXPERIENCE_COPY.compute, '需要算力，确认后另行审批');
  assert.equal(EXPERIENCE_COPY.noCompute, '不使用算力');
});

test('t237 C4b.8: summary count null (unknown) keeps 用经验库开发 under E1 XOR', () => {
  // Live summary() returns null until an owner-scoped store exists; must not look like count 0.
  const { ExperienceEntryView, EXPERIENCE_COPY } = require('./ExperiencePlan.tsx');
  const unknown = renderToStaticMarkup(h(ExperienceEntryView, { count: null }));
  assert.match(unknown, new RegExp(EXPERIENCE_COPY.start));
  assert.doesNotMatch(unknown, new RegExp(EXPERIENCE_COPY.empty));
  assert.doesNotMatch(unknown, new RegExp(EXPERIENCE_COPY.startChat));
});

test('t237 C4b.4 E1/E3: entry shows exactly one state and no extra 进化 heading under the tab', () => {
  const { ExperienceEntryView } = require('./ExperiencePlan.tsx');
  const full = renderToStaticMarkup(h(ExperienceEntryView, { count: 3 }));
  assert.equal(full, '<section class="tw-experience-entry" aria-label="进化"><p class="tw-experience-line">从已有经验开始一次模型开发。</p>'
    + '<a class="tw-primary" href="/workspace?new=1&amp;intent=experience">用经验库开发</a></section>');
  const empty = renderToStaticMarkup(h(ExperienceEntryView, { count: 0 }));
  assert.equal(empty, '<section class="tw-experience-entry" aria-label="进化"><div class="tw-empty-hint"><p>还没有经验。</p>'
    + '<a class="tw-button" href="/workspace?new=1">开始对话</a></div></section>');
  assert.doesNotMatch(empty, /用经验库开发|从已有经验开始/);
  assert.doesNotMatch(full, /还没有经验|开始对话/);
  assert.equal(renderToStaticMarkup(h(ExperienceEntryView, { count: null })), full);
  for (const html of [full, empty]) assert.doesNotMatch(html, /<h2/);
});

test('t237 C4b.4 E2: experience draft keeps the model picker closed and focuses the composer input', () => {
  // t238 C7c-A3 #1 rewrite (intent of b47fff2 H_e2_picker_focus): the picker is CLOSED on every draft, focus goes to the
  // composer input; it opens only on user action and never covers the composer error / training hint.
  const workspace = src('TaskWorkspace.tsx');
  assert.match(workspace, /const draftPickerAutoOpen = false;/);
  assert.doesNotMatch(workspace, /setDraftPickerAutoOpen|useState\(true\);\s*\n[^\n]*draftPicker/);
  assert.match(workspace, /<ModelPicker id="draft-model"[^>]*autoOpen=\{draftPickerAutoOpen\}[^>]*closeSignal=\{composerCloseSignal\}/);
  assert.match(workspace, /if \(creating && draftIntent\) document\.getElementById\('task-draft'\)\?\.focus\(\)/);
  const { ModelPicker } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [] };
  // first frame: collapsed, popup hidden — for a plain ?new=1 draft and an experience draft alike
  for (const autoOpen of [false, undefined]) {
    const closed = renderToStaticMarkup(h(ModelPicker, { id: 'draft-model', list, value: 'platform', autoOpen, onChange() {} }));
    assert.match(closed, /<button[^>]*aria-expanded="false"/);
    assert.match(closed, /<div class="tw-picker-pop" hidden="">/);
  }
  const picker = src('ComposerPicker.tsx');
  // opens on user action only (click toggles; arrow keys on the trigger), Esc closes and returns focus to the trigger
  assert.match(picker, /onClick=\{\(\) => open \? close\(\) : show\(\)\}/);
  assert.match(picker, /function close\(\) \{\s*setOpen\(false\);\s*trigger\.current\?\.focus\(\);/);
  assert.match(picker, /event\.key === 'Escape'\) \{ event\.preventDefault\(\); event\.stopPropagation\(\); close\(\); \}/);
  // outside click closes; an error / upload status / training hint closes it (never covers them)
  assert.match(picker, /document\.addEventListener\('pointerdown', outside\)/);
  assert.match(picker, /if \(closeSignal\) setOpen\(false\);/);
  assert.match(workspace, /const composerCloseSignal = \[homeUpload\.status, homeUpload\.error\]\.filter\(Boolean\)\.join\('\|'\);/);
  assert.match(src('TaskSession.tsx'), /<ModelPicker id="task-model"[^>]*closeSignal=\{\[composerUpload\.status, composerUpload\.error\]\.filter\(Boolean\)\.join\('\|'\)\}/);
  // the error bar is an alert and sits above the picker layer
  assert.match(src('ComposerAttach.tsx'), /<span className="tw-chip tw-chip-error" role="alert">/);
  const css = src('workspace.css');
  assert.match(css, /\.tw-picker-pop\{[^}]*z-index:30/);
  assert.match(css, /\.tw-chips \.tw-chip-error,\.tw-chips \.tw-chip-status\{position:relative;z-index:31\}/);
});

test('t237 C4b.5: pending plan card button row clears the composer (C5a L3 sibling pin + R5 scroll-margin)', () => {
  const css = src('workspace.css');
  // Composer is a flex sibling (not sticky/fixed overlay); scroll area keeps 16px bottom pad.
  assert.match(css, /\.tw-chat-scroll\{[^}]*padding-bottom:16px/);
  assert.match(css, /\.tw-chat>\.tw-composer\{[^}]*flex:none/);
  assert.doesNotMatch(css, /\.tw-chat>\.tw-composer\{[^}]*position:(sticky|fixed)/);
  assert.match(css, /\.task-workspace \.tw-plan-card \.tw-actions\{scroll-margin-bottom:20px\}/);
  const plan = src('ExperiencePlan.tsx');
  assert.match(plan, /<div ref=\{actions\} className="tw-actions">/);
  // Only a pending card scrolls; two rAF so layout (incl. composer height) is current.
  assert.match(plan, /if \(decision\) return;[^]*?requestAnimationFrame\(\(\) => \{ frame = requestAnimationFrame\(\(\) => actions\.current\?\.scrollIntoView\(\{ block: 'end' \}\)\); \}\)/);
  assert.match(plan, /cancelAnimationFrame\(frame\)/);
});

const pendingRun = (extra = {}) => ({ id: RID, status: 'completed', prompt: '用经验库', answer: '我先列了一个计划，你确认后我再继续。', intent: 'experience',
  plan_proposed: { references: [], steps: ['整理你的目标和已有条件'], needs_compute: true }, plan_decision: '', ...extra });

test('t237 C4b.6: pending plan ⇒ status 等你确认 (not 已交付); after confirm/cancel ⇒ 已完成 (C6c W3; C6-answer: no progress card)', () => {
  const { planPending, PLAN_PENDING_STATUS, runStatusWord } = require('./task-presentation.ts');
  const { taskStatus } = require('./types.ts');
  assert.equal(PLAN_PENDING_STATUS, '等你确认');
  assert.equal(planPending(pendingRun()), true);
  assert.equal(planPending(pendingRun({ plan_decision: 'confirm' })), false);
  assert.equal(planPending(pendingRun({ plan_decision: 'cancel' })), false);
  assert.equal(planPending({ status: 'completed' }), false);
  assert.equal(runStatusWord(pendingRun(), taskStatus), '等你确认');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const pending = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: pendingRun(), title: '经验库' }));
  assert.doesNotMatch(pending, /已交付|tw-progress-card/);
  for (const decision of ['confirm', 'cancel']) assert.equal(runStatusWord(pendingRun({ plan_decision: decision }), taskStatus), '已完成', decision);
  const session = src('TaskSession.tsx');
  // The title status chip follows the same rule.
  assert.match(session, /statusPillClass\(statusLabel\)|statusPillClass\(chatEyebrow/); // C4c.1 derives it from the run list before the snapshot arrives
  // A decision re-reads the run so plan_decision / a new approval reach TaskControls and the status copy.
  assert.match(src('TaskRunView.tsx'), /<ExperiencePlan key=\{run\.id\} runId=\{run\.id\} plan=\{run\.plan_proposed\} decision=\{run\.plan_decision\} onDecided=\{onRetry\} \/>/);
  assert.match(src('ExperiencePlan.tsx'), /setRecorded\([^;]*\);\s*onDecided\?\.\(\);/);
});

test('t237 C4b.6: experience_compute approval renders in-chat as 等你确认 with Chinese copy only', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: { id: RID, status: 'waiting_approval', approvals: [{
    id: 'approval_x', status: 'pending', tool_name: 'experience_compute', risk_level: 'low', reversible: true,
    normalized_arguments: { purpose: 'experience_plan_compute', output_root: '/hpc2hdd/home/aimslab/jinghw', estimated_cost: 0, budget: 0, steps: ['整理你的目标和已有条件'] },
    impact_scope: { personal_tool_boundary: false, output_root: '/hpc2hdd/home/aimslab/jinghw', experience_compute: true } }] } }));
  assert.match(html, /<section class="tw-confirm tw-inline-confirm" aria-label="等你确认"><h2>等你确认<\/h2>/);
  assert.match(html, /使用算力继续/);
  assert.match(html, /同意并继续/);
  assert.doesNotMatch(html, /待你确认：|<details|experience_compute|experience_plan_compute|hpc|jinghw|output_root/i);
  // Not auto-approved: nothing is checked or pre-submitted.
  assert.doesNotMatch(html, /checked=""/);
});


test('t237 C4b.7: plan labels 13px #6B7280; steps <ol> shows decimal numbers', () => {
  const css = src('workspace.css');
  assert.match(css, /\.task-workspace \.tw-plan-card h3\{[^}]*font-size:13px[^}]*color:(?:#6B7280|var\(--tw-muted\))/);
  assert.match(css, /\.task-workspace \.tw-plan-card ol\{list-style:decimal;padding-left:1\.5em/);
  const { PlanCard } = require('./ExperiencePlan.tsx');
  const html = renderToStaticMarkup(h(PlanCard, {
    plan: { references: [], steps: ['整理你的目标和已有条件', '对照相近的经验给出改进方案'], needs_compute: false },
    decision: '', onDecide() {},
  }));
  assert.match(html, /<h3>参考的经验<\/h3>/);
  assert.match(html, /<h3>步骤<\/h3><ol><li>/);
  assert.match(html, /<h3>需要的资源<\/h3>/);
});

// ---- C4c: three Critiquito minors on the live C4b tip ----
const pendingCompute = (args = {}) => ({ id: RID, status: 'waiting_approval', approvals: [{
  id: 'approval_c4c', status: 'pending', tool_name: 'experience_compute', risk_level: 'low', reversible: true,
  normalized_arguments: { purpose: 'experience_plan_compute', ...args }, impact_scope: { personal_tool_boundary: false, experience_compute: true } }] });

test('t237 C4c.1: title status is derived before first paint — never 已交付 while a plan/approval may be pending', () => {
  const { chatEyebrow, PLAN_PENDING_STATUS, STATUS_LOADING } = require('./task-presentation.ts');
  const { taskStatus } = require('./types.ts');
  const plan = { references: [], steps: ['a'], needs_compute: false };
  const base = { task: { status: 'delivered' }, taskStatus };
  // Loaded run: pending plan / pending approval ⇒ 等你确认.
  assert.equal(chatEyebrow({ ...base, run: { id: 'r1', status: 'completed', plan_proposed: plan }, runs: [], runId: 'r1' }), PLAN_PENDING_STATUS);
  assert.equal(chatEyebrow({ ...base, run: { id: 'r1', status: 'waiting_approval' }, runs: [], runId: 'r1' }), '等你确认');
  // Not loaded yet (first render: runId still '' ; then runId set, snapshot in flight): the run list already says what we know.
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed', plan_proposed: plan }], runId: '', restoring: true }), PLAN_PENDING_STATUS);
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'waiting_approval' }], runId: 'r1' }), PLAN_PENDING_STATUS);
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed', approvals: [{ id: 'a', status: 'pending' }] }], runId: 'r1' }), PLAN_PENDING_STATUS);
  // The task-detail summary says nothing is owed (plan_pending false) ⇒ the task's own status is safe on first paint.
  // C6c W3: a settled run reads 已完成 on the title chip, the same word as its progress card.
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed', plan_pending: false }], runId: '', restoring: true }), '已完成');
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed', plan_pending: true }], runId: 'r1' }), PLAN_PENDING_STATUS);
  // Unknown while loading ⇒ neutral, never the task's terminal 已交付.
  const neutral = chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed' }], runId: 'r1' });
  assert.equal(neutral, STATUS_LOADING);
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed' }], runId: '', restoring: true }), STATUS_LOADING);
  assert.doesNotMatch(neutral, /已交付/);
  // A requested run that is not in the list is not loading anything.
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed' }], runId: 'zzz' }), '已交付');
  // Restored with no run chosen (unlinked ?assistant_run): nothing is loading, so the task's own status stands.
  assert.equal(chatEyebrow({ ...base, run: null, runs: [{ id: 'r1', status: 'completed' }], runId: '' }), '已交付');
  // No runs at all (fresh task) keeps the task's own status.
  assert.equal(chatEyebrow({ task: { status: 'draft' }, taskStatus, run: null, runs: [], runId: '' }), '尚未开始');
  // Settled run, nothing pending ⇒ C6c W3 run word (decision made ⇒ 已完成, same as the progress card).
  assert.equal(chatEyebrow({ ...base, run: { id: 'r1', status: 'completed', plan_proposed: plan, plan_decision: 'confirm' }, runs: [], runId: 'r1' }), '已完成');
  assert.equal(chatEyebrow({ ...base, run: { id: 'r1', status: 'running' }, runs: [], runId: 'r1' }), '正在处理');
  // TaskSession routes the eyebrow through the helper (no inline taskStatus(task.status) on first paint).
  const session = src('TaskSession.tsx');
  assert.match(session, /statusPillClass\(statusLabel\)/); assert.match(session, /chatEyebrow\(\{ run, runs, runId, restoring: !restored, task, taskStatus \}\)/);
  assert.doesNotMatch(session, /planPending\(run\) \? PLAN_PENDING_STATUS : taskStatus\(workStatus\)/);
});

test('t237 C4c.2: approval cost line — never ¥0 / ¥0; hidden when estimate and cap are 0 or unknown', () => {
  const { confirmSpendLabel } = require('./task-presentation.ts');
  const { TaskControls } = require('./TaskControls.tsx');
  assert.equal(confirmSpendLabel({ normalized_arguments: { estimated_cost: 120, budget: 200 } }), '约 ¥120，上限 ¥200');
  assert.equal(confirmSpendLabel({ normalized_arguments: { estimated_cost: 120 } }), '约 ¥120');
  assert.equal(confirmSpendLabel({ normalized_arguments: { budget: 200 } }), '上限 ¥200');
  assert.equal(confirmSpendLabel({ normalized_arguments: { estimated_cost: 0, budget: 200 } }), '上限 ¥200');
  for (const args of [{ estimated_cost: 0, budget: 0 }, { estimated_cost: 0 }, { budget: 0 }, {}, { estimated_cost: null, budget: null }, { estimated_cost: '0' }, { estimated_cost: -1, budget: -1 }, { estimated_cost: NaN }]) {
    assert.equal(confirmSpendLabel({ normalized_arguments: args }), '', JSON.stringify(args));
  }
  assert.equal(confirmSpendLabel({}), '');
  // The real compute ask is 0 / 0: the in-chat card stays, the 花多少 row goes.
  const free = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: pendingCompute({ estimated_cost: 0, budget: 0 }) }));
  assert.doesNotMatch(free, /¥|花多少|约 |上限/);
  assert.match(free, /<h2>等你确认<\/h2>/);
  assert.match(free, /使用算力继续/);
  assert.match(free, /同意并继续/);
  assert.match(free, /不外发/);
  // Unknown spend keeps the existing fail-closed separate dialog (usesSeparateConfirm); it never prints ¥0 either.
  const unknown = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: pendingCompute({}) }));
  assert.doesNotMatch(unknown, /约 ¥0|¥0，上限/);
  const real = renderToStaticMarkup(h(TaskControls, { onChanged() {}, run: pendingCompute({ estimated_cost: 30, budget: 50 }) }));
  assert.match(real, /<p class="tw-confirm-label">花多少<\/p><p>约 ¥30，上限 ¥50<\/p>/);
});

test('t237 C4c.3: 等你确认 everywhere in the default view — 等待你确认 is gone', () => {
  const { taskStatus } = require('./types.ts');
  assert.equal(taskStatus('waiting_approval'), '等你确认');
  for (const name of ['types.ts', 'TaskSession.tsx', 'TaskRunView.tsx', 'TaskControls.tsx', 'TaskList.tsx', 'TaskWorkspace.tsx', 'ExperiencePlan.tsx', 'task-presentation.ts']) {
    assert.doesNotMatch(src(name), /等待你确认/, name);
  }
  const html = renderToStaticMarkup(h(TaskControls_(), { onChanged() {}, run: pendingCompute({}) }));
  assert.doesNotMatch(html, /等待你确认/);
  const list = renderToStaticMarkup(h(require('./TaskList.tsx').TaskList, { query: '', tasks: [{ id: 't1', title: '经验库', status: 'waiting_approval', updated_at: '2026-10-28T00:00:00Z' }] }));
  assert.match(list, /等你确认/);
  assert.doesNotMatch(list, /等待你确认/);
});
function TaskControls_() { return require('./TaskControls.tsx').TaskControls; }

test('t237 C4c.1: TaskSession first paint (SSR) — pending summary ⇒ 等你确认; unknown summary ⇒ 正在读取; never 已交付', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const eyebrow = runs => {
    const html = renderToStaticMarkup(h(TaskSession, { initial: {
      task: { id: 'utask_' + 'a'.repeat(32), title: '经验库', draft: '', version: 1, conversation_id: 'conversation_a', status: 'delivered', created_at: '', updated_at: '' },
      runs, files: [] } }));
    return html.match(/<p class="tw-status-pill[^"]*">([^<]*)<\/p>/)[1];
  };
  assert.equal(eyebrow([{ id: RID, status: 'completed', created_at: '2026-10-28T00:00:00Z', plan_pending: true }]), '等你确认');
  assert.equal(eyebrow([{ id: RID, status: 'waiting_approval', created_at: '2026-10-28T00:00:00Z', plan_pending: false }]), '等你确认');
  assert.equal(eyebrow([{ id: RID, status: 'completed', created_at: '2026-10-28T00:00:00Z' }]), '正在读取');
  assert.equal(eyebrow([{ id: RID, status: 'completed', created_at: '2026-10-28T00:00:00Z', plan_pending: false }]), '已完成');  // C6c W3
});

// ---- t237 C4e ----
const css = () => src('workspace.css');
const ruleOf = (sheet, selector) => {
  const at = sheet.indexOf(selector + '{');
  assert.ok(at >= 0, `rule ${selector}`);
  return sheet.slice(at + selector.length + 1, sheet.indexOf('}', at));
};

test('t237 C4e.1: the 已确认 line is no input-like block — ring only on :focus-visible, hugging the text', () => {
  const sheet = css();
  const base = ruleOf(sheet, '.task-workspace .tw-plan-status');
  assert.match(base, /width:fit-content/);
  assert.match(base, /outline:none/);
  const ring = ruleOf(sheet, '.task-workspace .tw-plan-status:focus-visible');
  assert.match(ring, /outline:2px solid var\(--tw-focus-ring\)/);
  // No other rule may paint a ring on the status line (a bare :focus rule would show it after a click).
  assert.ok(!/\.tw-plan-status:focus(?!-visible)/.test(sheet));
});

test('t237 C4e.1: focus moves to the status line only on a fresh decision, and asks for the ring only after keyboard activation', () => {
  const plan = src('ExperiencePlan.tsx');
  // A plan that was already decided when the card mounted (page load / reopen) never steals focus.
  assert.match(plan, /const decided = useRef\(decision\)/);
  assert.match(plan, /useEffect\(\(\) => \{ if \(decision && !decided\.current\) status\.current\?\.focus\(\{ focusVisible: viaKeyboard\.current \} as FocusOptions\); decided\.current = decision; \}, \[decision\]\);/);
  assert.match(plan, /const viaKeyboard = useRef\(false\)/);
  assert.match(plan, /onKeyDown=\{\(\) => \{ viaKeyboard\.current = true; \}\}/);
  assert.match(plan, /onPointerDown=\{\(\) => \{ viaKeyboard\.current = false; \}\}/);
  // The handlers sit on the card <section>, so both the pending card and the decided one carry them.
  assert.equal((plan.match(/onKeyDown=\{\(\) => \{ viaKeyboard\.current = true; \}\}/g) ?? []).length, 2);
  // Markup of the decided card is unchanged (the existing exact-markup tests still pin it).
  const { PlanCard } = require('./ExperiencePlan.tsx');
  const decided = renderToStaticMarkup(h(PlanCard, { plan: { references: [], steps: ['一步'], needs_compute: false }, decision: 'confirm', onDecide() {} }));
  assert.match(decided, /<p class="tw-plan-status" role="status" tabindex="-1">已确认<\/p><\/section>$/);
});

test('t237 C4e.5: a pending plan card ends ≥16px above the composer, not flush under it', () => {
  const sheet = css();
  const margin = ruleOf(sheet, '.task-workspace .tw-plan-card .tw-actions');
  const m = margin.match(/scroll-margin-bottom:(\d+)px/);
  assert.ok(m, 'scroll-margin-bottom on the actions row (the scrollIntoView target)');
  // actions bottom → card bottom = card padding 16 + border 1; the scroll padding already leaves 16 above the composer.
  const [desktop, mobile] = sheet.split('@media(max-width:760px)');
  const cardPadding = Number(desktop.match(/\.tw-confirm\{[^}]*padding:(\d+)px/)[1]);
  assert.ok(!/\.tw-confirm\{[^}]*padding:/.test(mobile) || Number(mobile.match(/\.tw-confirm\{[^}]*padding:(\d+)px/)[1]) <= cardPadding);
  assert.ok(Number(m[1]) + 16 - (cardPadding + 1) >= 16, `card bottom clearance ${Number(m[1]) + 16 - (cardPadding + 1)}px < 16px`);
  // The C4b test above keeps pinning scroll-padding-bottom / padding-bottom (desktop and mobile).
});
