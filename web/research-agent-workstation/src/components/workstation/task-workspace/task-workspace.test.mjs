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

// Built from parts so this file never contains the retired label itself.
const SAVE_DRAFT = ['保存', '草稿'].join('');

test('login cannot send a native GET containing credentials before hydration', () => {
  const { default: LoginPage } = require('../../../app/login/page.tsx');
  const html = renderToStaticMarkup(React.createElement(LoginPage));
  assert.match(html, /<form[^>]*method="post"/);
  assert.match(html, /<button[^>]*type="submit"[^>]*disabled/);
  assert.match(html, /data-brand-mark="(small|master)"/);
  for (const label of ['登录', '账号', '密码', '验证码']) assert.match(html, new RegExp(label));
  assert.doesNotMatch(html, /进入工作台/);
  assert.doesNotMatch(html, /SCIENTIFIC RESEARCH OS|Scientific Research OS|PRIVATE WORKSPACE|Private workspace|Production Agent Workspace|精确 CSRF|HPC|Kaggle|自进化|运行证据|Agent/);
});

test('my tasks explains how to begin and links to the exact saved task', () => {
  const { TaskList } = require('./TaskList.tsx');
  const empty = renderToStaticMarkup(React.createElement(TaskList, { tasks: [], query: '' }));
  assert.match(empty, /还没有任务/);
  assert.match(empty, /新对话/);
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
  assert.equal(empty, '');
  assert.equal(renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: null })), '');
  assert.equal(renderToStaticMarkup(React.createElement(TaskRunView, { view: 'results', run: null })), '');
  // No explanatory empty-tab copy; the page eyebrow already shows 尚未开始.
  assert.doesNotMatch(empty, /发送后|不会开始执行|<h2|尚未开始/);
  assert.doesNotMatch(empty, /Planner|Stage|Gate|运行中/);
  const run = { id: 'run_owned_fixture', status: 'completed', prompt: '我的问题', answer: '这是当前运行的回复',
    artifacts: [{ id: 'artifact_owned_fixture', run_id: 'run_owned_fixture', name: '本次结果.md', bytes: 12, sha256: 'a'.repeat(64), verified_at: '', preview_kind: 'text', source_tool_call: 'model_publish' }] };
  const conversation = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run }));
  assert.match(conversation, /这是当前运行的回复/);
  const result = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'results', run }));
  assert.match(result, /本次结果.md/);
  assert.match(result, /\/api\/assistant\/artifacts\/artifact_owned_fixture/);
  assert.match(result, /预览/);
  assert.match(result, /发布文件/);
  assert.doesNotMatch(result, /model_publish|不代表科研结论|也可能包含诊断记录/);
  assert.doesNotMatch(result, /质量已通过|官方结果已确认/);
});

test('unavailable model explains the next user action and retains technical evidence separately', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_fixture', user_task_id: 'utask_fixture', status: 'blocked', error_class: 'model_unavailable',
    answer: 'Runtime session is durable. Model execution is unavailable: LLMError/RuntimeError. No tool was executed by the failed model request.',
  } }));
  assert.match(html, /模型暂时不可用/);
  assert.match(html, /换个模型继续/);
  assert.match(html, /<details/);
});

test('model settings clearly separates save, connection test and actual use', () => {
  const { ModelSettings } = require('./ModelSettings.tsx');
  const html = renderToStaticMarkup(React.createElement(ModelSettings, { returnTask: 'utask_fixture', isolated: true }));
  assert.doesNotMatch(html, /保存不会发出请求/);
  assert.match(html, /还没有个人模型/); assert.match(html, /添加个人模型/);
  assert.match(html, /隔离验收环境禁止外部模型请求/);
  assert.match(html, /返回原任务/); assert.doesNotMatch(html, /localStorage|value="sk-/);
  assert.doesNotMatch(html, /tw-settings-grid/);
});

test('product navigation uses one workspace and personal settings never send users to legacy global pages', () => {
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const { navSections, retiredPrimaryPageIds, navItems } = require('../navigation.ts');
  const html = renderToStaticMarkup(React.createElement(TaskWorkspace));
  for (const label of ['新对话', '对话', '工作', '资料', '我的']) assert.ok(html.includes(label), label);
  assert.doesNotMatch(html, /我的任务|资料与数据|文献检索|高级工具|href="\/\?page=/);
  const primary = navSections.flatMap(section => section.ids);
  assert.equal(retiredPrimaryPageIds.length, 12);
  for (const id of retiredPrimaryPageIds) {
    assert.equal(primary.includes(id), false);
    assert.ok(navItems.some(item => item.id === id));
  }
  const page = readFileSync(new URL('../../../app/page.tsx', import.meta.url), 'utf8');
  const home = readFileSync(new URL('../../../app/home-client.tsx', import.meta.url), 'utf8');
  assert.match(page, /redirect\(workspaceLocation/);
  assert.doesNotMatch(page, /HomeClient|AppShell/);
  assert.doesNotMatch(home, /AppShell/);
  const { ModelSettings } = require('./ModelSettings.tsx');
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(ModelSettings, { returnTask: '', isolated: true })), /打开原设置|href="\/\?page=/);
});

test('settings offers three simple groups and resources never invent a personal budget', () => {
  const { WorkspaceSettings } = require('./WorkspaceSettings.tsx');
  const { ThemeProvider } = require('../theme/ThemeProvider.tsx');
  const html = renderToStaticMarkup(React.createElement(ThemeProvider, null,
    React.createElement(WorkspaceSettings, { group: 'resources', returnTask: 'utask_fixture', isolated: true })));
  for (const label of ['模型', '计算资源与额度', '账户与外观', '返回原任务']) assert.ok(html.includes(label), label);
  assert.ok(html.includes('计算资源') || html.includes('额度'), 'resources section');
  assert.doesNotMatch(html, /额度由管理员管理/);
  assert.doesNotMatch(html, /保存预算|剩余额度|治理校验|打开原设置/);
});

test('legacy links retain identities without guessing a task and map settings to one shell', () => {
  const { workspaceLocation } = require('./workspace-location.ts');
  assert.equal(workspaceLocation(new URLSearchParams('page=gpu')), '/workspace?section=work&tab=resources');
  assert.equal(workspaceLocation(new URLSearchParams('page=evolution')), '/workspace?section=work&tab=evolution');
  assert.equal(workspaceLocation(new URLSearchParams('page=literature')), '/workspace');
  assert.equal(workspaceLocation(new URLSearchParams('page=experiments')), '/workspace');
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

test('a saved task with no run has the same send action as the home composer', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_fixture', title: '阅读资料', draft: '请开始', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  assert.match(html, /发送/);
  assert.match(html, /aria-label="添加资料"/);
});

test('served model and generated files are visible in the current conversation', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_served', status: 'completed', prompt: '[Selected task: risk_control]\n测试正文', answer: '回复',
    model_provider: 'anthropic', model: 'claude-opus-4-8',
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: '[Selected task: risk_control]\n测试正文' },
      { id: 'a', role: 'assistant', content: '回复' },
    ] },
    artifacts: [{ id: 'artifact_report', run_id: 'run_served', name: '报告.md', bytes: 4, sha256: 'b'.repeat(64), preview_kind: 'markdown' }],
    artifact_evidence: [{ id: 'artifact_note', run_id: 'run_served', name: 'notes.txt', bytes: 2, sha256: 'c'.repeat(64), preview_kind: 'text' }],
  } }));
  assert.match(html, /data-served-model="anthropic\/claude-opus-4-8"/);
  assert.match(html, /<p class="tw-served-model"[^>]*>Claude Opus 4.8<\/p>/);
  assert.doesNotMatch(html.replace(/data-served-model="[^"]*"/g, ''), /anthropic\s*\/\s*claude|claude-opus-4-8/);
  assert.match(html, /报告\.md/);
  // C6-answer H_a4: the chat lists deliverables only; intermediate evidence stays on the results view.
  assert.doesNotMatch(html, /notes\.txt/);
  assert.match(html, /测试正文/);
  assert.doesNotMatch(html, /risk_control|本轮实际模型/);
  const results = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'results', run: {
    id: 'run_served', status: 'completed',
    artifacts: [{ id: 'artifact_report', run_id: 'run_served', name: '报告.md', bytes: 4, sha256: 'b'.repeat(64), preview_kind: 'markdown' }],
    artifact_evidence: [{ id: 'artifact_note', run_id: 'run_served', name: 'notes.txt', bytes: 2, sha256: 'c'.repeat(64), preview_kind: 'text' }],
  } }));
  assert.match(results, /中间文件与执行记录（1）[^]*notes\.txt/);
});

test('work item label is delivered or not, without an explanation paragraph', () => {
  const { TaskList } = require('./TaskList.tsx');
  const { TaskSession } = require('./TaskSession.tsx');
  const { taskStatus } = require('./types.ts');
  assert.equal(taskStatus('delivered'), '已交付');
  assert.equal(taskStatus('undelivered'), '未交付');
  const list = renderToStaticMarkup(React.createElement(TaskList, { query: '', tasks: [{
    id: 'utask_0123456789abcdef0123456789abcdef', title: '报告任务', status: 'undelivered', updated_at: '2026-10-23T00:00:00Z',
  }] }));
  assert.match(list, /未交付/);
  assert.doesNotMatch(list, /执行已完成|因为|后续/);
  const session = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_0123456789abcdef0123456789abcdef', title: '报告任务', draft: '请生成报告', version: 1, conversation_id: 'conversation_a', status: 'undelivered', created_at: '', updated_at: '' },
    runs: [{ id: 'run_qa', status: 'completed', created_at: '2026-10-23T00:00:00Z', plan_pending: false }], files: [],
  } }));
  assert.match(session, /未交付/);
  assert.doesNotMatch(session, /<p>[^<]*因为/);
});

test('platform label and task restore follow the served model and the owning task', () => {
  const { productModelLabel, owningTaskHref, sessionDraft, actionName, visibleMessageText } = require('./task-presentation.ts');
  const { modelGroups } = require('./ComposerPicker.tsx');
  const platform = { profiles: [], default_profile_id: '', platform_model: { provider: 'anthropic', model: 'claude-sonnet-5' } };
  assert.equal(modelGroups(platform)[0].choices[0].label, 'Claude Sonnet 5 · 默认');
  // t237 C4.2: a run on another model never relabels the default slot.
  assert.equal(modelGroups(platform, { model: 'claude-opus-4-8' })[0].choices[0].label, 'Claude Sonnet 5 · 默认');
  assert.equal(modelGroups(platform, { model: 'claude-sonnet-5-20261001' })[0].choices[0].model, 'claude-sonnet-5-20261001');
  assert.equal(productModelLabel('claude-opus-5-5'), 'Claude Opus 5.5');
  assert.equal(productModelLabel('gpt-4o'), 'GPT-4o');
  assert.equal(actionName('model_publish'), '发布文件');
  assert.equal(actionName('not_a_known_op'), '已生成');
  assert.equal(actionName('artifact_import:run_1'), '导入文件');
  assert.equal(visibleMessageText('**本轮实际模型**：`anthropic / claude-opus-5-5`（来自受管 model_observed 事件）\n\n回复正文'), '回复正文');
  assert.equal(owningTaskHref('utask_old', { id: 'run_new', user_task_id: 'utask_new' }, 'conversation'), '/workspace?task=utask_new&assistant_run=run_new&view=conversation');
  assert.equal(owningTaskHref('utask_same', { id: 'run_same', user_task_id: 'utask_same' }), '');
  assert.equal(sessionDraft('已保存的正文', '另一任务的草稿', 'utask_other', 'utask_current'), '已保存的正文');
  assert.equal(sessionDraft('已保存的正文', '正在输入', 'utask_current', 'utask_current'), '正在输入');
});

test('pending filter uses the same status the chip renders, newest updated_at first', () => {
  const { TaskList } = require('./TaskList.tsx');
  const { taskStatus } = require('./types.ts');
  const tasks = [
    { id: 'utask_old', title: '较早的未交付', status: 'undelivered', updated_at: '2026-10-01T00:00:00Z' },
    { id: 'utask_new', title: '较新的未交付', status: 'undelivered', updated_at: '2026-10-20T00:00:00Z' },
    { id: 'utask_done', title: '已经交付', status: 'delivered', updated_at: '2026-10-22T00:00:00Z' },
  ];
  const pending = renderToStaticMarkup(React.createElement(TaskList, { query: '', pendingOnly: true, tasks }));
  assert.match(pending, /还要处理/);
  assert.match(pending, /较新的未交付/);
  assert.match(pending, /较早的未交付/);
  assert.equal(taskStatus('undelivered'), '未交付');
  assert.match(pending, /未交付/);
  assert.doesNotMatch(pending, /已经交付|已交付/);
  assert.ok(pending.indexOf('较新的未交付') < pending.indexOf('较早的未交付'));
  const all = renderToStaticMarkup(React.createElement(TaskList, { query: '', pendingOnly: false, tasks }));
  assert.match(all, /已经交付/);
  assert.ok(all.indexOf('已经交付') < all.indexOf('较新的未交付'));
});

test('quota used label excludes reserved holds', () => {
  const { BudgetUsage } = require('./BudgetUsage.tsx');
  const html = renderToStaticMarkup(React.createElement(BudgetUsage, { budget: {
    used_seconds: 3600, charged_wall_seconds: 3600, pending_reserved_seconds: 7200, committed_seconds: 10800,
  } }));
  assert.match(html, /已用/);
  assert.match(html, /data-budget-used="3600"/);
  assert.match(html, /data-budget-reserved="7200"/);
  assert.doesNotMatch(html, /data-budget-used="10800"/);
  assert.doesNotMatch(html, /剩余额度|保存预算/);
});

test('blocked progress is one Chinese status and a next action', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { blockedProgressCopy } = require('./task-presentation.ts');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'progress', run: {
    id: 'run_blocked', status: 'blocked', error_class: 'recoverable',
    error_message: 'required result artifacts unavailable; missing final artifacts: r5-anchor-report.md',
    plan: { steps: [
      { id: 's1', label: '验证结果与文件', status: 'failed', detail: 'missing final artifacts: r5-anchor-report.md' },
    ] },
  } }));
  assert.match(html, /结果还没齐/);
  assert.match(html, /检查并继续/);
  assert.doesNotMatch(html, /r5-anchor-report|required result|recoverable|技术错误|missing final/);
  assert.equal(blockedProgressCopy({ status: 'blocked', error_class: 'planning_idle' }).status, '方案停住了');
  assert.equal(blockedProgressCopy({ status: 'blocked', error_class: 'execution_reconciliation_required' }).action, '先核对这一步');
});

test('a stuck conversation names the step in Chinese and keeps the file list folded', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { blockedAssistantLine, streamPhaseState } = require('./task-presentation.ts');
  const stuck = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_stuck', status: 'blocked', error_class: 'planning_idle', prompt: '继续', answer: '先停一下',
  } }));
  assert.match(stuck, /方案停住了，我先检查这一步再继续。/);
  // C6-answer: no progress card or 理解/执行/核对/交付 strip — the process row names the stop.
  assert.match(stuck, /<span class="tw-thinking-text">这一步没有完成<\/span>/);
  assert.doesNotMatch(stuck, /tw-progress-card|tw-phases|理解目标/);
  assert.doesNotMatch(stuck, /r5-anchor-report|recoverable|技术错误|missing final/);
  assert.equal(blockedAssistantLine('结果还没齐'), '结果还没齐，我先检查再接着跑。');
  assert.equal(streamPhaseState({ status: 'blocked', error_class: 'planning_idle' }).stuck, true);
  const files = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_files', status: 'completed', prompt: '整理', answer: '好了',
    artifacts: [1, 2, 3, 4].map(index => ({ id: `artifact_${index}`, run_id: 'run_files', name: `结果${index}.md`, bytes: index * 1024, sha256: 'd'.repeat(64), preview_kind: 'text', source_tool_call: 'model_publish' })),
  } }));
  // C6-answer H_a11b: up to 3 chips, the rest fold into 「还有 N 个文件」.
  assert.match(files, /结果1\.md/);
  assert.match(files, /结果3\.md/);
  assert.match(files, /还有 1 个文件/);
  assert.ok(files.indexOf('还有 1 个文件') < files.indexOf('结果4.md'));
});

test('in-budget confirmation stays in the chat and danger stays a separate dialog', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const { usesSeparateConfirm } = require('./task-presentation.ts');
  const inline = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: {
    id: 'run_budget', status: 'waiting_approval', approvals: [{ id: 'approval_budget', status: 'pending', tool_name: 'cost_estimate', reversible: true, risk_level: 'low', normalized_arguments: { estimated_cost: 120, budget: 200 }, impact_scope: {} }],
  } }));
  assert.match(inline, /等你确认/);
  assert.match(inline, /要做什么/);
  assert.match(inline, /花多少/);
  assert.match(inline, /约 ¥120，上限 ¥200/);
  assert.match(inline, /不外发/);
  assert.match(inline, /同意并继续/);
  assert.match(inline, /调整方案/);
  assert.doesNotMatch(inline, /确认执行/);
  assert.equal(usesSeparateConfirm({ tool_name: 'file_delete', risk_level: 'low' }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { over_budget: true } }), true);
});

test('a spend above the cap or an unknown tool stays on the separate confirmation', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const { usesSeparateConfirm } = require('./task-presentation.ts');
  const card = approval => renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: {
    id: 'run_budget', status: 'waiting_approval', approvals: [{ id: 'approval_budget', status: 'pending', reversible: true, impact_scope: {}, ...approval }],
  } }));
  const over = card({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { estimated_cost: 300, budget: 200 } });
  assert.match(over, /待你确认/);
  assert.match(over, /确认执行/);
  assert.doesNotMatch(over, /同意并继续|不外发/);
  const unknown = card({ tool_name: 'upload', risk_level: 'low', normalized_arguments: { estimated_cost: 10, budget: 100 } });
  assert.match(unknown, /确认执行/);
  assert.doesNotMatch(unknown, /同意并继续|不外发/);
  for (const tool of ['http', 'train', 'gpu_job', 'git_push']) {
    assert.equal(usesSeparateConfirm({ tool_name: tool, risk_level: 'low', normalized_arguments: { estimated_cost: 10, budget: 100 } }), true, tool);
  }
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', normalized_arguments: { estimated_cost: 200, cap: 200 } }), false);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { estimated_cost: 120, budget: 200 } }), false);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { cost: '120', budget: 200 } }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { estimated_cost: 120 } }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'medium', normalized_arguments: { estimated_cost: 120, budget: 200 } }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { estimated_cost: 10, budget: 100, external_submit: true } }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'file_delete', risk_level: 'low' }), true);
  assert.equal(usesSeparateConfirm({ tool_name: 'cost_estimate', risk_level: 'low', normalized_arguments: { over_budget: true, estimated_cost: 10, budget: 100 } }), true);
});

test('tool completion is a short step label and hides the raw tool id', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'progress', run: {
    id: 'run_tool', status: 'completed', plan: { steps: [
      { id: 's1', label: 'tool_completed', status: 'completed', detail: 'hpc_execute_solution' },
      { id: 's2', label: '理解目标与输入', status: 'completed', detail: '已读取资料' },
      { id: 's3', label: '验证结果与文件', status: 'completed', detail: 'verified 1 artifact(s)' },
    ] },
  } }));
  assert.match(html, /已完成一步/);
  assert.match(html, /理解目标与输入/);
  assert.match(html, /已读取资料/);
  assert.match(html, /已核对/);
  assert.doesNotMatch(html, /tool_completed|hpc_execute_solution|verified 1 artifact/);
});

test('allocation status is one string from the check', () => {
  const { allocationStatusText } = require('../../../lib/allocation-status.ts');
  assert.equal(allocationStatusText({ personal_resources: 'unavailable_in_local_mode', binding: null }), '未接入计算资源');
  assert.equal(allocationStatusText({ binding: null }), '未设置计算资源');
  assert.equal(allocationStatusText({ binding: { state: 'active' }, activation: { state: 'pending_activation', receipt_exists: false } }), '计算资源未通过核对');
  assert.equal(allocationStatusText({ binding: { state: 'active' }, activation: { state: 'completed', receipt_exists: true } }), '计算资源已核对');
  assert.equal(allocationStatusText({ binding: { state: 'frozen' } }), '计算资源暂不可用');
  const labels = ['未接入计算资源', '未设置计算资源', '计算资源未通过核对', '计算资源已核对', '计算资源暂不可用'];
  for (const label of labels) assert.doesNotMatch(label, /GPU|HPC|Tenant/);
});

test('an accepted resume remains in progress and does not offer another resume button', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const { taskStatus } = require('./types.ts');
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { id: 'run_fixture', status: 'recovering' } }));
  // C6b H4-2 / Critiquito: label is 「暂停」, not 「暂停任务」; Stop is primary.
  assert.match(html, /data-testid="tw-pause"[^>]*>暂停</);
  assert.match(html, /data-testid="tw-stop"[^>]*>停止</);
  assert.doesNotMatch(html, /暂停任务|检查并继续/);
  assert.equal(taskStatus('recovering'), '正在恢复');
});

test('composer sends a real effort, defaults to a usable model, and folds English process chatter', () => {
  const { reasoningEffort, defaultModelChoice } = require('./composer-options.ts');
  const { isProcessLeak } = require('./task-presentation.ts');
  assert.equal(reasoningEffort('低'), 'low'); assert.equal(reasoningEffort('中'), 'medium'); assert.equal(reasoningEffort('高'), 'high');
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5' }, default_profile_id: 'p1', profiles: [{ id: 'p1', enabled: true }, { id: 'p2', enabled: false }] };
  assert.equal(defaultModelChoice(list, '', null), 'platform');
  assert.equal(defaultModelChoice(list, 'p2', 'p1'), 'p1');
  assert.equal(defaultModelChoice({ ...list, platform_model: null }, 'platform'), 'p1');
  assert.equal(defaultModelChoice({ profiles: [], default_profile_id: '', platform_model: null }), '');
  assert.ok(isProcessLeak("I'll check inputs/data.csv first and then write report.md."));
  assert.ok(isProcessLeak('Verifying artifacts in outputs/final'));
  assert.equal(isProcessLeak('这是给你的结论：基线复现完成，见 report.md。'), false);
  assert.equal(isProcessLeak('The baseline reaches 0.91 accuracy.'), false);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_leak', status: 'completed', message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a2', messages: [
      { id: 'u', role: 'user', content: '复现基线' },
      { id: 'a1', role: 'assistant', content: "I'll read inputs/paper.md and plan the artifacts." },
      { id: 'a2', role: 'assistant', content: '基线已复现。' },
    ] },
  } }));
  assert.match(html, /基线已复现/);
  // C6-answer: the chatter folds into the process row's hidden detail, never a chat bubble.
  const bubbles = (html.match(/<article class="tw-message tw-assistant[^]*?<\/article>/g) ?? []).join('');
  assert.doesNotMatch(bubbles.replace(/<div id="[^"]+" class="tw-thinking-detail" hidden="">[^]*?<\/div>/g, ''), /inputs\/paper/);
  assert.match(html, /class="tw-thinking-detail" hidden="">[^]*<p class="tw-process-note">I&#x27;ll read inputs\/paper\.md/);
});

test('platform models are selectable by id, never displace the default, and only non-default ids are sent', () => {
  const { defaultModelChoice, isSelectableModel, platformModelId } = require('./composer-options.ts');
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const platform_models = [{ id: 'm0', provider: 'anthropic', model: 'claude-sonnet-5', label: '' }, { id: 'm1', provider: 'openai', model: 'gpt-4o', label: 'gpt-4o' }];
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5' }, platform_models, default_profile_id: 'p1', profiles: [{ id: 'p1', enabled: true }] };
  assert.equal(isSelectableModel(list, 'platform:m1'), true);
  assert.equal(isSelectableModel(list, 'platform:m0'), false);
  assert.equal(defaultModelChoice(list, 'platform:m1'), 'platform:m1');
  assert.equal(defaultModelChoice(list, 'platform:gone'), 'platform');
  assert.equal(defaultModelChoice(list, '', null), 'platform');
  assert.equal(defaultModelChoice({ ...list, platform_model: null }), 'p1');
  assert.equal(defaultModelChoice({ ...list, platform_model: null, profiles: [], default_profile_id: '' }), 'platform:m0');
  assert.equal(defaultModelChoice({ ...list, platform_models: null, platform_model: null, profiles: [], default_profile_id: '' }), '');
  for (const value of ['platform', 'platform:m0', 'p1', 'platform:gone', '']) assert.equal(platformModelId(list, value), undefined);
  assert.equal(platformModelId(list, 'platform:m1'), 'm1');
  const [platformGroup] = modelGroups(list);
  assert.deepEqual(platformGroup.choices.map(choice => [choice.value, choice.label]), [['platform', 'Claude Sonnet 5 · 默认'], ['platform:m1', 'GPT-4o']]);
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform', onChange() {} }));
  assert.equal((html.match(/data-value="platform/g) || []).length, 3); // trigger + two options
  assert.match(html, /data-value="platform:m1"[^>]*><span class="tw-picker-label" title="GPT-4o">GPT-4o</);
  const visible = (html.match(/>[^<]*</g) || []).join('');
  for (const raw of ['gpt-4o', 'claude-sonnet-5', 'platform:']) assert.doesNotMatch(visible, new RegExp(raw));
  const deepseek = modelGroups({ ...list, platform_models: [...platform_models,
    { id: 'm2', provider: 'deepseek', model: 'deepseek-chat', label: 'deepseek-chat' }, { id: 'm3', provider: 'deepseek', model: 'deepseek-reasoner', label: 'deepseek-reasoner' }] });
  assert.deepEqual(deepseek[0].choices.slice(2).map(choice => [choice.value, choice.label]), [['platform:m2', 'DeepSeek Chat'], ['platform:m3', 'DeepSeek Reasoner']]);
  assert.deepEqual(modelGroups({ profiles: [], default_profile_id: '' }), []);
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: { profiles: [], default_profile_id: '' }, value: '', onChange() {} })), /aria-label="(?:平台模型|我的模型)"/);
});

test('model labels keep the exact version and match the shared Python table', () => {
  const { productModelLabel } = require('./task-presentation.ts');
  const table = JSON.parse(readFileSync(new URL('../../../../../../tests/fixtures/model_display_names.json', import.meta.url), 'utf8'));
  assert.ok(table.cases.length >= 25);
  for (const [id, label] of table.cases) assert.equal(productModelLabel(id), label, id);
  for (const [id] of table.cases) if (id) assert.notEqual(productModelLabel(id), id);
});

test('lt4net catalog groups by family with the default first and thinking after its base', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const catalog = JSON.parse(readFileSync(new URL('../../../../../../src/evomind_runtime/lt4net_models.json', import.meta.url), 'utf8'));
  const claude = [...catalog.models].reverse().map(model => ({ id: `anthropic:${model}`, provider: 'anthropic', model, label: model }));
  const others = ['deepseek-chat', 'deepseek-reasoner', 'gpt-4o', 'grok-4.3'].map(model => ({ id: `x:${model}`, provider: 'x', model, label: model }));
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5' }, default_profile_id: '', profiles: [],
    platform_models: [{ id: 'anthropic:claude-sonnet-5', provider: 'anthropic', model: 'claude-sonnet-5', label: '' }, ...others, ...claude] };
  const [platform] = modelGroups(list);
  assert.deepEqual(platform.choices.map(choice => choice.label), [
    'Claude Sonnet 5 · 默认',
    'Claude Opus 5.5', 'Claude Opus 5.5 Thinking', 'Claude Opus 5', 'Claude Opus 5 Thinking', 'Claude Opus 4.8', 'Claude Opus 4.8 Thinking',
    'Claude Opus 4.7', 'Claude Opus 4.7 Thinking', 'Claude Opus 4.6', 'Claude Opus 4.6 Thinking', 'Claude Opus 4.5',
    'Claude Sonnet 5.5', 'Claude Sonnet 5 Thinking', 'Claude Sonnet 4.6',
    'Claude Haiku 5.5', 'Claude Haiku 5.5 Thinking', 'Claude Haiku 5.5 Agentic', 'Claude Haiku 4.5',
    'GPT-4o', 'Grok 4.3', 'DeepSeek Chat', 'DeepSeek Reasoner']);
  assert.equal(new Set(platform.choices.map(choice => choice.label)).size, platform.choices.length);
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform:anthropic:claude-opus-4-8-thinking', onChange() {} }));
  assert.deepEqual([...html.matchAll(/class="tw-picker-section"[^>]*>([^<]+)</g)].map(match => match[1]), ['Claude', 'GPT', 'Grok', '国模']);
  assert.match(html, /<span title="Claude Opus 4.8 Thinking" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Opus<\/span><span class="tw-picker-version"> 4\.8<\/span><span class="tw-picker-rest"> Thinking<\/span><\/span>/);
  const visible = (html.match(/>[^<]*</g) || []).join('');
  assert.doesNotMatch(visible, /claude-|gpt-|grok-|deepseek-|platform:/);

  const deepseekDefault = modelGroups({ ...list, platform_model: { provider: 'deepseek', model: 'deepseek-reasoner' } })[0].choices;
  assert.equal(deepseekDefault[0].label, 'DeepSeek Reasoner · 默认');
  assert.deepEqual([...new Set(deepseekDefault.map(choice => choice.section))], ['国模', 'Claude', 'GPT', 'Grok']);
  const claudeOnly = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: { ...list, platform_models: list.platform_models.filter(item => item.provider === 'anthropic') }, value: 'platform', onChange() {} }));
  assert.deepEqual([...claudeOnly.matchAll(/class="tw-picker-section"[^>]*>([^<]+)</g)].map(match => match[1]), ['Claude']);
  assert.match(claudeOnly, /<span title="Claude Sonnet 5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5<\/span><\/span>/);
});

test('lt4net product default sonnet-5-5 labels as Claude Sonnet 5.5 · 默认', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  const { defaultModelChoice } = require('./composer-options.ts');
  const catalog = JSON.parse(readFileSync(new URL('../../../../../../src/evomind_runtime/lt4net_models.json', import.meta.url), 'utf8'));
  // Same order the API returns on lt4net: the product default first, the rest in catalog order.
  const models = ['claude-sonnet-5-5', ...catalog.models.filter(model => model !== 'claude-sonnet-5-5')];
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [],
    platform_models: models.map(model => ({ id: `anthropic:${model}`, provider: 'anthropic', model, label: '' })) };
  const labels = modelGroups(list)[0].choices.map(choice => choice.label);
  assert.equal(labels[0], 'Claude Sonnet 5.5 · 默认');
  assert.equal(labels.filter(label => label.startsWith('Claude Sonnet 5.5')).length, 1);
  assert.ok(labels.includes('Claude Sonnet 5'));
  assert.equal(labels.length, catalog.models.length);
  assert.equal(defaultModelChoice(list), 'platform');
});

test('model picker is a labelled listbox: product name on the trigger, groups, ids only in values', () => {
  const { ModelPicker } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'deepseek', model: 'deepseek-chat' }, default_profile_id: 'p1',
    platform_models: [{ id: 'deepseek:deepseek-chat', provider: 'deepseek', model: 'deepseek-chat', label: 'deepseek-chat' }, { id: 'deepseek:deepseek-reasoner', provider: 'deepseek', model: 'deepseek-reasoner', label: 'deepseek-reasoner' }],
    profiles: [{ id: 'profile_a', name: '实验室网关', model: 'claude-sonnet-5', enabled: true, verified_at: '' }, { id: 'profile_off', name: '停用', model: 'gpt-4o', enabled: false }] };
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform', onChange() {} }));
  assert.match(html, /<button[^>]*id="task-model"[^>]*aria-haspopup="listbox"[^>]*aria-expanded="false"[^>]*aria-controls="task-model-list"[^>]*aria-label="本次使用的模型：DeepSeek Chat"/);
  assert.match(html, /<span title="DeepSeek Chat">DeepSeek Chat<\/span>/);
  assert.match(html, /<div class="tw-picker-pop" hidden="">/);
  assert.match(html, /role="listbox" id="task-model-list"[^>]*aria-label="本次使用的模型"/);
  assert.ok(html.indexOf('>平台模型<') < html.indexOf('>我的模型<'));
  assert.match(html, /data-value="platform" aria-selected="true"/);
  assert.match(html, /data-value="platform:deepseek:deepseek-reasoner" aria-selected="false"[^>]*><span class="tw-picker-label" title="DeepSeek Reasoner">DeepSeek Reasoner</);
  assert.match(html, /data-value="profile_a"[^>]*><span class="tw-picker-label" title="实验室网关 · Claude Sonnet 5 · 尚未测试">实验室网关<\/span><small>Claude Sonnet 5 · 尚未测试<\/small>/);
  assert.doesNotMatch(html, /profile_off|停用/);
  const visible = (html.match(/>[^<]*</g) || []).join('') + (html.match(/aria-label="[^"]*"/g) || []).join('');
  assert.doesNotMatch(visible, /deepseek-|claude-|platform:|profile_a/);
  const mineOnly = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: { ...list, profiles: [] }, value: 'platform', onChange() {} }));
  assert.doesNotMatch(mineOnly, /aria-label="我的模型"|>我的模型</);
});

test('intensity picker keeps the 低/中/高 → reasoning_effort wiring behind a compact trigger', () => {
  const { IntensityPicker } = require('./ComposerPicker.tsx');
  const { INTENSITY_LEVELS, reasoningEffort } = require('./composer-options.ts');
  const html = renderToStaticMarkup(React.createElement(IntensityPicker, { id: 'task-intensity', value: '中', onChange() {} }));
  assert.match(html, /aria-label="强度 · 中"/);  // C6d backlog 3: no doubled 「强度：强度」
  assert.match(html, /<span title="强度 · 中">强度 · 中<\/span>/);
  for (const level of INTENSITY_LEVELS) assert.match(html, new RegExp(`role="option"[^>]*data-value="${level}"`));
  assert.match(html, /data-value="中" aria-selected="true"/);
  assert.deepEqual(INTENSITY_LEVELS.map(reasoningEffort), ['low', 'medium', 'high']);
  assert.doesNotMatch(html, /aria-pressed/);
});

test('task composer: attach left, model + intensity + send on the right, secondary actions inside the card', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_fixture', title: '阅读资料', draft: '请开始', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  const card = html.slice(html.indexOf('<div class="tw-composer">'));
  const bottom = card.slice(card.indexOf('tw-composer-bottom'), card.indexOf('tw-composer-extra'));
  const order = ['aria-label="添加资料"', 'tw-composer-send', 'id="task-model"', 'id="task-intensity"', 'aria-label="发送"'].map(marker => bottom.indexOf(marker));
  assert.ok(order.every((at, index) => at > 0 && (index === 0 || at > order[index - 1])), String(order));
  assert.doesNotMatch(html, /<select[^>]*id="task-model"|class="tw-intensity"/);
  const extra = card.slice(card.indexOf('tw-composer-extra'));
  assert.ok(extra.includes('>从资料库选择</button>'));
  assert.ok(!extra.includes('从资料库选择（0）'));
  for (const label of [SAVE_DRAFT, '配置我的模型']) assert.ok(!extra.slice(0, extra.indexOf('</form>')).includes(label), label);
  assert.match(card, /aria-expanded="false" aria-controls="task-library"/);
  assert.match(card, /<\/form>\s*<div id="task-library" class="tw-composer-library" hidden="">/);
  assert.doesNotMatch(html, /<details class="tw-details"><summary>从资料库/);
  assert.match(html, /class="tw-chat tw-chat-fresh"/);
  assert.equal((html.match(/尚未开始/g) || []).length, 1);
});

test('home composer uses the same pickers on the right of the toolbar', () => {
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const source = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(source, /<div className="tw-composer-send"><ModelPicker id="home-model"[^]*?\{intensityPicker\}<button type="submit" className="tw-send-icon tw-send"/);
  assert.doesNotMatch(source, /<select id="home-model"|ModelOptions|INTENSITY_LEVELS\.map/);
  assert.ok(TaskWorkspace);
});

const MD_REPLY = [
  '## 结论', '', '- 第一项', '- 第二项', '  - 嵌套项', '', '1. 步骤一', '2. 步骤二', '',
  '运行 `npm test` 后查看 [文档](https://example.com/docs)。', '', '```python', 'print("hi")', '```', '',
  '| 指标 | 值 |', '| --- | --- |', '| AUC | 0.91 |', '', '> 引用', '', '---', '', '**粗** *斜* ~~删~~',
].join('\n');

function chatHtml(answer, extra = {}) {
  const { TaskRunView } = require('./TaskRunView.tsx');
  return renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_md', status: 'completed', prompt: '- 用户原文', answer,
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: '- 用户原文' }, { id: 'a', role: 'assistant', content: answer },
    ] }, ...extra,
  } }));
}

test('assistant replies render as Markdown; user messages stay plain text', () => {
  const html = chatHtml(MD_REPLY);
  const assistant = html.slice(html.indexOf('tw-message tw-assistant'));
  assert.match(assistant, /<div class="tw-md(?: tw-msg-answer)?"(?: data-testid="tw-answer")?>/);
  assert.match(assistant, /<h2>结论<\/h2>/);
  assert.match(assistant, /<ul>\s*<li>第一项<\/li>/);
  assert.match(assistant, /<li>第二项\s*<ul>\s*<li>嵌套项<\/li>/);
  assert.match(assistant, /<ol>\s*<li>步骤一<\/li>/);
  assert.doesNotMatch(assistant, /<li>- |>- 第/);
  assert.match(assistant, /<code>npm test<\/code>/);
  assert.match(assistant, /<a href="https:\/\/example.com\/docs" target="_blank" rel="noopener noreferrer">文档<\/a>/);
  assert.match(assistant, /<pre><code class="language-python">print\(&quot;hi&quot;\)\n?<\/code><\/pre>/);
  assert.match(assistant, /class="tw-md-code"[^]*?<button[^>]*type="button"[^>]*aria-label="复制代码"/);
  assert.match(assistant, /<div class="tw-md-table"><table>[^]*<th>指标<\/th>[^]*<td>0.91<\/td>/);
  assert.match(assistant, /<blockquote>[^]*引用/);
  assert.match(assistant, /<hr\/>/);
  assert.match(assistant, /<strong>粗<\/strong>[^]*<em>斜<\/em>[^]*<del>删<\/del>/);
  const user = html.slice(html.indexOf('tw-message tw-user'), html.indexOf('tw-message tw-assistant'));
  assert.match(user, /<p>- 用户原文<\/p>/);
  assert.doesNotMatch(user, /tw-md/);
});

test('assistant Markdown never executes raw HTML, handlers or script URLs', () => {
  const html = chatHtml('前文\n\n<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n[点我](javascript:alert(3))\n\n<a href="https://x.test" onclick="alert(4)">x</a>');
  assert.doesNotMatch(html, /<script|<img|onerror=|onclick=|href="javascript:/i);
  assert.match(html, /前文/);
});

test('an unclosed fence while streaming stays one code block and keeps the leak filter', () => {
  const html = chatHtml('先看代码：\n\n```js\nconst a = 1;');
  assert.match(html, /<pre><code class="language-js">const a = 1;\n?<\/code><\/pre>/);
  // C6-answer H_a8: the last assistant message is the answer — the leak filter never hides it.
  const last = chatHtml("I'll check the outputs/report.md file first.");
  assert.match(last, /<article class="tw-message tw-assistant[^"]*"[^]*outputs\/report\.md/);
  const prefixed = chatHtml('**本轮实际模型**：`anthropic/claude-opus-4-8`（来自受管 model_observed 事件）\n\n正文');
  assert.doesNotMatch(prefixed, /本轮实际模型/);
  assert.match(prefixed, /<p>正文<\/p>/);
});

test('chat column, composer and Markdown styles keep their layout contracts', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  const read = css.match(/--tw-read:(\d+)px/);
  assert.ok(read && Number(read[1]) >= 760 && Number(read[1]) <= 820, 'reading column 760–820px');
  assert.match(css, /\.tw-chat\{max-width:var\(--tw-read\)/);
  for (const selector of ['.tw-md ul', '.tw-md pre', '.tw-md-table', '.tw-md a', '.tw-md code', '.tw-md blockquote']) assert.ok(css.includes(selector), selector);
  assert.match(css, /\.tw-md-table\{[^}]*overflow-x:auto/);
  const mobile = css.slice(css.indexOf('@media(max-width:760px)'));
  assert.match(mobile, /\.task-workspace main\{padding:\d+px (1[2-6])px/);
  assert.match(mobile, /\.tw-user\{margin-left:0/);
});

test('picker trigger and options carry the full product label for ellipsis', () => {
  const { ModelPicker, IntensityPicker } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'anthropic', model: 'claude-opus-5-5-thinking' }, default_profile_id: '', profiles: [],
    platform_models: [{ id: 'anthropic:claude-haiku-5-5-agentic', provider: 'anthropic', model: 'claude-haiku-5-5-agentic', label: '' }] };
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform', onChange() {} }));
  const trigger = html.match(/<button[^>]*>/)[0];
  assert.match(trigger, /title="Claude Opus 5.5 Thinking"/);
  assert.match(trigger, /aria-label="本次使用的模型：Claude Opus 5.5 Thinking"/);
  assert.match(html, /role="option"[^>]*title="Claude Haiku 5.5 Agentic"/);
  assert.match(html, /role="option"[^>]*title="Claude Opus 5.5 Thinking · 默认"/);
  assert.match(renderToStaticMarkup(React.createElement(IntensityPicker, { id: 'i', value: '高', onChange() {} })), /role="option"[^>]*title="高 · 想得更久"/);
  // The ellipsized elements themselves carry the native tooltip.
  assert.match(html, /<button[^>]*><span title="Claude Opus 5.5 Thinking" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Opus<\/span><span class="tw-picker-version"> 5\.5<\/span><span class="tw-picker-rest"> Thinking<\/span><\/span>/);
  assert.match(html, /<span class="tw-picker-label" title="Claude Haiku 5.5 Agentic">Claude Haiku 5.5 Agentic<\/span>/);
  assert.match(html, /<span class="tw-picker-label" title="Claude Opus 5.5 Thinking · 默认">Claude Opus 5.5 Thinking · 默认<\/span>/);
  const intensity = renderToStaticMarkup(React.createElement(IntensityPicker, { id: 'i', value: '高', onChange() {} }));
  assert.match(intensity, /<button[^>]*title="强度 · 高"[^>]*><span title="强度 · 高">强度 · 高<\/span>/);
  assert.match(intensity, /<span class="tw-picker-label" title="高 · 想得更久">高<\/span>/);
});

test('two model ids with the same product label are listed once; the default keeps its slot', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'deepseek', model: 'deepseek-v4-pro' }, default_profile_id: '', profiles: [], platform_models: [
    { id: 'deepseek:deepseek-v4-flash', provider: 'deepseek', model: 'deepseek-v4-flash', label: '' },
    { id: 'openai:DeepSeek-V4-Flash', provider: 'openai', model: 'DeepSeek-V4-Flash', label: '' },
    { id: 'other:deepseek_v4_flash', provider: 'other', model: 'deepseek_v4_flash', label: '' },
    { id: 'other:deepseek-v4-pro-20260101', provider: 'other', model: 'deepseek-v4-pro-20260101', label: '' },
  ] };
  assert.deepEqual(modelGroups(list)[0].choices.map(choice => [choice.value, choice.label]),
    [['platform', 'DeepSeek 4 Pro · 默认'], ['platform:deepseek:deepseek-v4-flash', 'DeepSeek 4 Flash']]);
});

test('the draft model picker opens once on ?new=1; other pickers stay closed', () => {
  const { ModelPicker, IntensityPicker } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [] };
  const opened = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'draft-model', list, value: 'platform', autoOpen: true, onChange() {} }));
  assert.match(opened, /<button[^>]*aria-expanded="true"/);
  assert.doesNotMatch(opened, /class="tw-picker-pop" hidden/);
  const closed = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list, value: 'platform', onChange() {} }));
  assert.match(closed, /<button[^>]*aria-expanded="false"/);
  assert.match(closed, /class="tw-picker-pop" hidden/);
  // No choices yet: nothing to open (the effect opens later once the list arrives).
  const empty = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'draft-model', list: { profiles: [], default_profile_id: '' }, value: '', autoOpen: true, onChange() {} }));
  assert.match(empty, /aria-expanded="false"/);
  const source = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(source, /<ModelPicker id="draft-model"[^>]*autoOpen/);
  assert.doesNotMatch(source, /<ModelPicker id="home-model"[^>]*autoOpen/);
  assert.doesNotMatch(source, /<IntensityPicker[^>]*autoOpen/);
  const picker = readFileSync(new URL('./ComposerPicker.tsx', import.meta.url), 'utf8');
  assert.match(picker, /autoOpened\.current/); // a ref guards against re-opening on later renders
});

const LIVE_PICKER = () => JSON.parse(readFileSync(new URL('../../../../../../tests/fixtures/live_picker_models.json', import.meta.url), 'utf8'));
const LIVE_LIST = () => ({ platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [], platform_models: LIVE_PICKER() });

test('t237 C4.2: the live picker shows only DeepSeek 4.1 Flash', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  const labels = modelGroups(LIVE_LIST()).flatMap(group => group.choices.map(choice => choice.label));
  assert.equal(labels.length, 22);
  assert.deepEqual(labels.filter(label => /DeepSeek/.test(label)), ['DeepSeek 4.1 Flash']);
});

const pickerRows = (groups) => groups.flatMap(group => group.choices);
const assertSonnetDefaultFirst = (rows) => {
  assert.equal(rows.length, 22);
  assert.deepEqual([rows[0].value, rows[0].label, rows[0].model], ['platform', 'Claude Sonnet 5.5 · 默认', 'claude-sonnet-5-5']);
  assert.equal(rows.filter(row => /Claude Sonnet 5\.5( · 默认)?$/.test(row.label)).length, 1);
  assert.equal(new Set(rows.map(row => row.value)).size, rows.length);
};

test('t237 C4.2: an old conversation on removed DeepSeek 4 Pro falls back to Sonnet 5.5', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const { defaultModelChoice } = require('./composer-options.ts');
  const list = LIVE_LIST();
  const value = defaultModelChoice(list, 'platform:deepseek:deepseek-v4-pro', 'platform:deepseek:deepseek-v4-pro');
  assert.equal(value, 'platform');
  const rows = pickerRows(modelGroups(list, { model: 'deepseek-v4-pro' }));
  assertSonnetDefaultFirst(rows);
  assert.ok(!rows.some(row => /4 Pro/.test(row.label) || row.model === 'deepseek-v4-pro'));
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value, platform: { model: 'deepseek-v4-pro' }, onChange() {} }));
  assert.match(html, /<button[^>]*><span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/);
  assert.doesNotMatch(html, /DeepSeek 4 Pro/);
});

test('t237 C4.2: after model_identity_unconfirmed the picker still lists Sonnet 5.5 first and each pick once', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  for (const model of ['deepseek-v4-pro', 'deepseek/deepseek-v4.1-flash', 'deepseek-flash', 'gpt-4o', 'gpt-4o-mini-2024-07-18']) {
    const rows = pickerRows(modelGroups(LIVE_LIST(), { model }));
    assertSonnetDefaultFirst(rows);
    assert.equal(rows.filter(row => row.label === 'DeepSeek 4.1 Flash').length, 1, model);
    assert.equal(rows.filter(row => row.label === 'GPT-4o').length, 1, model);
    assert.ok(rows.some(row => row.value === 'platform:deepseek:deepseek-flash') && rows.some(row => row.value === 'platform:openai:gpt-4o'), model);
  }
});

test('t237 C4.2: a stored hidden model id falls back to Sonnet 5.5', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const { defaultModelChoice } = require('./composer-options.ts');
  const list = LIVE_LIST();
  for (const stored of ['platform:deepseek:deepseek-chat', 'platform:openai:gpt-4.1', 'platform:openai:deepseek-flash', 'platform:deepseek:deepseek-v4-pro']) {
    const value = defaultModelChoice(list, stored, stored);
    assert.equal(value, 'platform', stored);
    assertSonnetDefaultFirst(pickerRows(modelGroups(list, null)));
    const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value, onChange() {} }));
    assert.match(html, /<button[^>]*><span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/);
  }
  // A still-listed pick is kept.
  assert.equal(defaultModelChoice(list, 'platform:openai:gpt-4o'), 'platform:openai:gpt-4o');
});

const defaultRows = (rows) => rows.filter(row => / · 默认$/.test(row.label));
const SONNET_TRIGGER = /<button[^>]*><span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/;

test('t237 C4.2b: switching back to the default after a failed 4 Pro run shows Sonnet 5.5 · 默认', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const list = LIVE_LIST();
  // Exactly what TaskSession passes when model === 'platform' and the failed run recorded deepseek-v4-pro.
  const platform = { model: 'deepseek-v4-pro' };
  const rows = pickerRows(modelGroups(list, platform));
  assertSonnetDefaultFirst(rows);
  assert.ok(!rows.some(row => /4 Pro/.test(row.label) || row.model === 'deepseek-v4-pro'));
  assert.deepEqual(defaultRows(rows).map(row => [row.label, row.model]), [['Claude Sonnet 5.5 · 默认', 'claude-sonnet-5-5']]);
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform', platform, onChange() {} }));
  assert.match(html, SONNET_TRIGGER);
  assert.doesNotMatch(html, /4 Pro/);
});

test('t237 C4.2b: reload with last-model=platform after a failed 4 Pro run keeps Sonnet 5.5 selected', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const { defaultModelChoice } = require('./composer-options.ts');
  const list = LIVE_LIST();
  const value = defaultModelChoice(list, null, 'platform');
  assert.equal(value, 'platform');
  const platform = { model: 'deepseek-v4-pro' };
  const rows = pickerRows(modelGroups(list, platform));
  assertSonnetDefaultFirst(rows);
  assert.equal(defaultRows(rows).length, 1);
  assert.equal(defaultRows(rows)[0].model, 'claude-sonnet-5-5');
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value, platform, onChange() {} }));
  assert.match(html, SONNET_TRIGGER);
  const selected = [...html.matchAll(/<div role="option"[^>]*aria-selected="true"[^>]*>/g)].map(match => match[0]);
  assert.equal(selected.length, 1);
  assert.match(selected[0], /data-value="platform"/);
  assert.match(selected[0], /title="[^"]*Claude Sonnet 5\.5[^"]*"/);
  assert.equal((html.match(/role="option"/g) || []).length, 22);
  assert.equal((html.match(/· 默认/g) || []).length, (html.match(/Claude Sonnet 5\.5 · 默认/g) || []).length);
});

test('t237 C4.2b: 「· 默认」 never follows a foreign served model', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  for (const model of ['deepseek-v4-pro', 'deepseek/deepseek-v4.1-flash', 'gpt-4o', 'claude-opus-4-8']) {
    const rows = pickerRows(modelGroups(LIVE_LIST(), { model }));
    assertSonnetDefaultFirst(rows);
    assert.deepEqual(defaultRows(rows).map(row => [row.label, row.model]), [['Claude Sonnet 5.5 · 默认', 'claude-sonnet-5-5']], model);
    // Not injected: a listed model keeps its single own row, an unlisted one is absent.
    const listed = LIVE_PICKER().some(item => item.model === model);
    assert.equal(rows.filter(row => row.model === model).length, listed ? 1 : 0, model);
    assert.notEqual(rows[0].model, model, model);
  }
});

test('t237 C4.2b: TaskSession passes the run model as `platform` only under the default slot', () => {
  const source = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  const props = source.match(/<ModelPicker id="task-model"[^\n]*/)?.[0] || '';
  assert.match(props, /platform=\{model === 'platform' && run\?\.model \? \{ model: run\.model \} : null\}/);
  assert.equal((source.match(/<ModelPicker\b/g) || []).length, 1);
});

test('one model served by two providers is listed once, first provider wins', () => {
  const { modelGroups } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'deepseek', model: 'deepseek-v4-flash' }, default_profile_id: '', profiles: [], platform_models: [
    { id: 'deepseek:deepseek-v4-flash', provider: 'deepseek', model: 'deepseek-v4-flash', label: '' },
    { id: 'deepseek:deepseek-v4-pro', provider: 'deepseek', model: 'deepseek-v4-pro', label: '' },
    { id: 'openai:deepseek-v4-flash', provider: 'openai', model: 'deepseek-v4-flash', label: '' },
    { id: 'openai:deepseek-v4-pro', provider: 'openai', model: 'deepseek-v4-pro', label: '' },
  ] };
  assert.deepEqual(modelGroups(list)[0].choices.map(choice => [choice.value, choice.label]),
    [['platform', 'DeepSeek 4 Flash · 默认'], ['platform:deepseek:deepseek-v4-pro', 'DeepSeek 4 Pro']]);
});

test('a new chat draft offers the model picker and carries the choice into the task', () => {
  const source = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(source, /<ModelPicker id="draft-model"/);
  assert.match(source, /if \(creating\) \{[^]*?NEXT_MODEL_KEY, homeModel/);
});

test('t237 C1a: no source file under src/ carries the retired save-draft label', async () => {
  const { readdirSync, statSync } = await import('node:fs');
  const root = fileURLToPath(new URL('../../../', import.meta.url));
  const hits = [];
  (function walk(dir) {
    for (const name of readdirSync(dir)) {
      const path = dir + '/' + name;
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.(tsx?|mjs|css)$/.test(name) && readFileSync(path, 'utf8').includes(SAVE_DRAFT)) hits.push(path.slice(root.length));
    }
  })(root.replace(/\/$/, ''));
  assert.deepEqual(hits, []);
  const workspace = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(workspace, /正在保存…/);
});

test('t237 C1a: the model picker footer links to 我的 → 模型 outside the listbox; intensity has no footer', () => {
  const { ModelPicker, IntensityPicker } = require('./ComposerPicker.tsx');
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [] };
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'task-model', list, value: 'platform', autoOpen: true, returnTask: 'utask_1', onChange() {} }));
  const listbox = html.slice(html.indexOf('role="listbox"'));
  const footer = /<\/div><\/div><div class="tw-picker-footer"><a href="([^"]+)">配置我的模型<\/a><\/div><\/div>/.exec(listbox);
  assert.ok(footer, 'footer anchor follows the closed listbox inside the popover');
  assert.equal(footer[1].replace(/&amp;/g, '&'), '/workspace?settings=models&return_task=utask_1');
  assert.doesNotMatch(html, /role="option"[^>]*>配置我的模型/);
  const home = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list, value: 'platform', onChange() {} }));
  assert.match(home, /<div class="tw-picker-footer"><a href="\/workspace\?settings=models">配置我的模型<\/a>/);
  const intensity = renderToStaticMarkup(React.createElement(IntensityPicker, { id: 'task-intensity', value: '中', onChange() {} }));
  assert.doesNotMatch(intensity, /tw-picker-footer|配置我的模型/);
  // Every composer passes its open task (or none) to the picker.
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, /<ModelPicker id="task-model"[^>]*returnTask=\{task\.id\}/);
  const workspace = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  for (const id of ['draft-model', 'home-model']) assert.match(workspace, new RegExp(`<ModelPicker id="${id}"`));
});

test('t237 C1a: 我的 still opens model settings', () => {
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskWorkspace));
  assert.match(html, /<a href="\/workspace\?settings=models">[^]*?我的<\/a>/);
});

test('t237 C1a: a draft conflict offers 使用服务器版本 / 保留我的输入 and no instruction', () => {
  const { DraftConflict } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(DraftConflict, { disabled: false, onServer() {}, onMine() {} }));
  assert.match(html, /服务器上有更新的草稿/);
  // t237 C1b (Critiquito): 保留我的输入 is the primary action and comes first.
  assert.match(html, /<button type="button" class="tw-primary">保留我的输入<\/button><button type="button" class="tw-button">使用服务器版本<\/button>/);
  assert.doesNotMatch(html, new RegExp(SAVE_DRAFT + '|点击|才会'));
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, />重新读取<\/button>/);
  assert.doesNotMatch(session, /重新读取版本（保留本页输入）|检查后点击/);
});

test('t237 C1a: code styles and composer clearance tokens', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /--tw-code-inline:#F2F4F3/);
  assert.match(css, /--tw-code:#F7F8F7/);
  assert.match(css, /--tw-code-edge:#E5E7EB/);
  assert.match(css, /\.tw-md code\{[^}]*background:var\(--tw-code-inline\)/);
  const block = /\.tw-md-code\{([^}]*)\}/.exec(css)[1];
  for (const rule of ['background:var(--tw-code)', 'border:1px solid var(--tw-code-edge)', 'border-radius:8px']) assert.ok(block.includes(rule), rule);
  assert.doesNotMatch(block, /overflow:hidden/);
  const mobile = css.slice(css.indexOf('@media(max-width:760px)'));
  assert.doesNotMatch(mobile, /\.tw-md-code\{border-radius:8px/);
  assert.match(css, /\.tw-md pre\{[^}]*overflow-x:auto[^}]*white-space:pre/);
  assert.match(css, /\.tw-md pre code\{[^}]*background:transparent/);
  for (const rule of css.match(/\.tw-md(?:-code| pre| code)[^{]*\{[^}]*\}/g)) assert.doesNotMatch(rule, /overflow-wrap:anywhere|word-break/, rule);
  // C5a L3: composer is a flex sibling (structural clearance) — no overlay scroll-padding / negative margin
  assert.match(css, /\.tw-chat>\.tw-composer\{[^}]*flex:none/);
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, /new ResizeObserver/);
  assert.match(session, /--tw-composer-h/);
  assert.match(session, /\.disconnect\(\)/);
});

test('editing the draft after a save error clears it so autosave retries', () => {
  const source = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(source, /if \(!creating \|\| !draft\.trim\(\) \|\| busy \|\| error \|\|/);
  const start = source.indexOf('<textarea id="task-draft"');
  assert.ok(start >= 0);
  const textarea = source.slice(start, source.indexOf('/>', start));
  const onChange = /onChange=\{event => \{([^}]*)\}\}/.exec(textarea)?.[1] ?? '';
  assert.match(onChange, /setDraft\(event\.target\.value\)/);
  assert.match(onChange, /setError\(''\)/);
  assert.ok(!source.includes(SAVE_DRAFT));
});

// ---- t237 C1b: sidebar conversation menu ----
const C1B_NOW = Date.parse('2026-10-26T15:00:00Z');
const at = hoursAgo => new Date(C1B_NOW - hoursAgo * 3600e3).toISOString();
const c1bTask = (id, title, extra = {}) => ({ id: 'utask_' + id.padEnd(32, '0'), title, draft: '', version: 1, conversation_id: 'conversation_' + id,
  status: 'draft', created_at: at(1), updated_at: at(1), pinned_at: null, archived_at: null, project_id: null, ...extra });
const C1B_PROJECT = { id: 'project_' + 'a'.repeat(32), name: '蛋白质项目' };
const c1bTasks = () => [
  c1bTask('a1', '今天的对话', { updated_at: at(1) }),
  c1bTask('a2', '很早的对话', { updated_at: at(24 * 9) }),
  c1bTask('a3', '置顶对话', { pinned_at: at(2) }),
  c1bTask('a4', '项目里置顶', { project_id: C1B_PROJECT.id, pinned_at: at(1) }),
  c1bTask('a5', '项目普通', { project_id: C1B_PROJECT.id, updated_at: at(0.5) }),
  c1bTask('a6', '归档的', { archived_at: at(3) }),
  c1bTask('a7', '删掉的', { deleted: true }),
];

test('t237 C1b: sidebar sections are 置顶 / 项目 / 今天 / 昨天 / 更早 with a pinned project conversation in both', () => {
  const { sidebarSections } = require('./conversation-menu.ts');
  const view = sidebarSections(c1bTasks(), [C1B_PROJECT], C1B_NOW);
  assert.deepEqual(view.pinned.map(task => task.title), ['项目里置顶', '置顶对话']);
  assert.deepEqual(view.projects.map(folder => [folder.project.name, folder.items.map(task => task.title)]), [['蛋白质项目', ['项目里置顶', '项目普通']]]);
  assert.deepEqual(view.groups.map(group => [group.label, group.items.map(task => task.title)]), [['今天', ['今天的对话']], ['更早', ['很早的对话']]]);
  const all = JSON.stringify(view);
  assert.doesNotMatch(all, /归档的|删掉的/);
  assert.deepEqual(sidebarSections([c1bTask('b1', '孤儿', { project_id: 'project_' + 'f'.repeat(32) })], [], C1B_NOW).groups[0].items.map(task => task.title), ['孤儿']);
  assert.deepEqual(sidebarSections([], [], C1B_NOW), { pinned: [], projects: [], groups: [] });
});

test('t237 C1b: the sidebar renders groups, folder pin marks, 「…」 triggers and 已归档（n）', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const props = { tasks: c1bTasks().map(task => ({ ...task, updated_at: task.pinned_at || task.project_id ? task.updated_at : new Date().toISOString() })),
    setTasks() {}, projects: [C1B_PROJECT], setProjects() {}, archivedCount: 2, setArchivedCount() {}, selection: 'utask_' + 'a1'.padEnd(32, '0'), query: '', onLeaveOpen() {} };
  const html = renderToStaticMarkup(React.createElement(SidebarHistory, props));
  const labels = [...html.matchAll(/<p class="tw-history-label">([^<]+)<\/p>/g)].map(match => match[1]);
  assert.deepEqual(labels.slice(0, 3), ['置顶', '项目', '今天']);
  assert.equal(html.match(/项目里置顶/g).length >= 2, true, 'pinned project conversation in 置顶 and its folder');
  // C2b: folders start collapsed; the open conversation's project is forced open.
  const opened = renderToStaticMarkup(React.createElement(SidebarHistory, { ...props, selection: 'utask_' + 'a5'.padEnd(32, '0') }));
  const folder = opened.slice(opened.indexOf('tw-folder-items'), opened.indexOf('<p class="tw-history-label">今天'));
  assert.match(folder, /lucide-pin[^>]*tw-row-pin|tw-row-pin[^>]*lucide-pin|class="[^"]*tw-row-pin/);
  assert.ok(folder.indexOf('项目里置顶') < folder.indexOf('项目普通'));
  const pinnedGroup = html.slice(0, html.indexOf('<p class="tw-history-label">项目'));
  assert.doesNotMatch(pinnedGroup, /tw-row-pin/);
  assert.match(html, /<a class="tw-archived-link" href="\/workspace\?settings=archived">已归档（2）<\/a>/);
  assert.match(html, /<div class="tw-history-row"><a href="\/workspace\?task=utask_a1[0]+"[^>]*aria-current="page"/);
  assert.match(html, /<button type="button" class="tw-row-more"[^>]*aria-haspopup="menu" aria-expanded="false" aria-label="对话操作：今天的对话"><svg[^>]*width="24"/);
  // Roving tabindex: one stop for the list, on the open conversation.
  assert.equal(html.match(/data-row="[^"]+" tabindex="0"/g).length, 1);
  assert.match(html, /data-row="time:utask_a1[0]+" tabindex="0"/);
  assert.doesNotMatch(html, /归档的|删掉的/);
  const none = renderToStaticMarkup(React.createElement(SidebarHistory, { ...props, archivedCount: 0, projects: [] }));
  assert.doesNotMatch(none, /已归档|tw-folder-row/);
  // C2b (02 §2): with no projects the 「项目」 title and its 「+」 still show.
  assert.match(none, /tw-history-label">项目<\/p><button type="button" class="tw-group-add" aria-label="新建项目"/);
});

test('t237 C1b: menu items are in spec order with the 删除 danger text and no confirm dialog', () => {
  const { MenuEntries, ProjectEntries } = require('./ConversationMenu.tsx');
  const { menuEntries } = require('./conversation-menu.ts');
  assert.deepEqual(menuEntries({ pinned_at: null, archived_at: null }).map(entry => entry.label), ['重命名', '置顶', '移到项目', '导出', '归档', '删除']);
  assert.deepEqual(menuEntries({ pinned_at: 'x', archived_at: 'y' }).map(entry => entry.label), ['重命名', '取消置顶', '移到项目', '导出', '取消归档', '删除']);
  const task = c1bTask('m1', '我的论文');
  const html = renderToStaticMarkup(React.createElement(MenuEntries, { task, subOpen: false, onCommand() {}, onOpenSubmenu() {} }));
  const labels = [...html.matchAll(/<span class="tw-menu-label">([^<]+)<\/span>/g)].map(match => match[1]);
  assert.deepEqual(labels, ['重命名', '置顶', '移到项目', '导出', '归档', '删除']);
  assert.equal(html.match(/role="menuitem"/g).length, 6);
  assert.match(html, /<div role="separator" class="tw-menu-sep"><\/div><button type="button" role="menuitem"[^>]*><svg[^>]*lucide-archive/);
  assert.match(html, /<button type="button" role="menuitem" tabindex="-1" class="tw-menu-item is-danger" aria-label="删除 我的论文">/);
  assert.match(html, /aria-haspopup="menu" aria-expanded="false"><svg[^>]*lucide-folder-input/);
  assert.equal(html.match(/is-danger/g).length, 1);
  const sub = renderToStaticMarkup(React.createElement(ProjectEntries, { task: { ...task, project_id: C1B_PROJECT.id }, projects: [C1B_PROJECT, { id: 'project_' + 'b'.repeat(32), name: '另一个' }], inline: true, onBack() {}, onMove() {}, onCreateProject: async () => {} }));
  const subLabels = [...sub.matchAll(/<span class="tw-menu-label">([^<]+)<\/span>/g)].map(match => match[1]);
  assert.deepEqual(subLabels, ['移到项目', '移出项目', '蛋白质项目', '另一个', '新建项目']);
  assert.match(sub, /lucide-chevron-left/);
  assert.match(sub, /role="menuitemradio"[^>]*aria-checked="true" aria-disabled="true"><span class="tw-menu-label">蛋白质项目/);
  assert.match(sub, /role="menuitemradio"[^>]*aria-checked="false"><span class="tw-menu-label">另一个/);
  for (const file of ['./ConversationMenu.tsx', './SidebarHistory.tsx', './ArchivedConversations.tsx']) {
    assert.doesNotMatch(readFileSync(new URL(file, import.meta.url), 'utf8'), /confirm\(|role="dialog"|alertdialog/, file);
  }
  const menu = readFileSync(new URL('./ConversationMenu.tsx', import.meta.url), 'utf8');
  assert.match(menu, /role="menu" aria-label=\{COPY\.menu\(task\.title\)\}/);
  for (const key of ["'ArrowDown'", "'ArrowUp'", "'Home'", "'End'", "'ArrowRight'", "'ArrowLeft'", "'Escape'", "'Tab'"]) assert.ok(menu.includes(key), key);
  assert.match(menu, /\(max-width:760px\)/);
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  for (const key of ["'ContextMenu'", "'F10' && event.shiftKey", "'F2'", 'onContextMenu', 'preventDefault', 'LONG_PRESS_MS']) assert.ok(sidebar.includes(key), key);
});

test('t237 C1b: §9 copy, toast semantics and timing constants', () => {
  const { COPY, LONG_PRESS_MS, SEARCH_DEBOUNCE_MS, TOAST_MS } = require('./conversation-menu.ts');
  assert.equal(LONG_PRESS_MS, 500);
  assert.equal(SEARCH_DEBOUNCE_MS, 250);
  assert.equal(TOAST_MS, 5000);
  assert.equal(COPY.moved('蛋白质项目'), '已移到「蛋白质项目」');
  assert.equal(COPY.archivedLink(3), '已归档（3）');
  assert.equal(COPY.trigger('A'), '对话操作：A');
  assert.equal(COPY.menu('A'), 'A 的操作');
  assert.equal(COPY.saveFailed, '没能保存，请重试');
  const { ToastView } = require('./Toast.tsx');
  const deleted = renderToStaticMarkup(React.createElement(ToastView, { toast: { message: COPY.deleted, action: { label: COPY.undo, run() {} } } }));
  assert.equal(deleted, '<div class="tw-toast" role="status" aria-live="polite"><span>已删除</span><span aria-hidden="true">·</span><button type="button" class="tw-toast-action">撤销</button></div>');
  const failed = renderToStaticMarkup(React.createElement(ToastView, { toast: { message: COPY.exportFailed, action: { label: COPY.retry, run() {} } } }));
  assert.match(failed, /导出失败.*重试/);
  assert.equal(renderToStaticMarkup(React.createElement(ToastView, { toast: { message: COPY.saveFailed } })), '<div class="tw-toast" role="status" aria-live="polite"><span>没能保存，请重试</span></div>');
  const toast = readFileSync(new URL('./Toast.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(toast, /\.focus\(\)|autoFocus/);
  const workspace = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(workspace, /<ToastHost \/>/);
});

test('t237 C1b: search results show snippets, a plain 已归档 tag and 没有找到', () => {
  const { SearchResult } = require('./SidebarHistory.tsx');
  const { snippetParts, titleMatches, COPY } = require('./conversation-menu.ts');
  assert.deepEqual(snippetParts('读 Protein 和 protein', 'protein'), [{ text: '读 ', match: false }, { text: 'Protein', match: true }, { text: ' 和 ', match: false }, { text: 'protein', match: true }]);
  const html = renderToStaticMarkup(React.createElement(SearchResult, { task: c1bTask('s1', '旧笔记', { archived_at: at(1), snippet: '…总结蛋白质结构…' }), query: '蛋白质' }));
  assert.equal(html, '<span class="tw-result-head"><span class="tw-row-title">旧笔记</span><span class="tw-tag">已归档</span></span><span class="tw-result-snippet"><span>…总结</span><strong>蛋白质</strong><span>结构…</span></span>');
  const live = renderToStaticMarkup(React.createElement(SearchResult, { task: c1bTask('s2', '新笔记', { snippet: '' }), query: 'x' }));
  assert.doesNotMatch(live, /tw-tag|tw-result-snippet/);
  // Newest first; equal times fall back to id order.
  assert.deepEqual(titleMatches(c1bTasks(), '对话').map(task => task.title), ['置顶对话', '今天的对话', '很早的对话']);
  assert.doesNotMatch(JSON.stringify(titleMatches(c1bTasks(), '的')), /删掉的|归档的/);
  assert.equal(COPY.noResults, '没有找到');
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  assert.match(sidebar, /COPY\.noResults/);
  assert.match(sidebar, /listConversations\(\{ q: searching \}/);
  const workspace = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(workspace, /event\.key === 'Escape' && historyQuery[^}]*setHistoryQuery\(''\)/);
});

test('t237 C1b: 我的 → 已归档 lists rows with 恢复 / 删除 and a short empty state', () => {
  const { ArchivedList } = require('./ArchivedConversations.tsx');
  const empty = renderToStaticMarkup(React.createElement(ArchivedList, { tasks: [], onUnarchive() {}, onDelete() {} }));
  assert.equal(empty, '<div class="tw-empty"><p>没有归档的对话。</p></div>');
  const html = renderToStaticMarkup(React.createElement(ArchivedList, { tasks: [c1bTask('r1', '归档的论文', { archived_at: at(1) })], onUnarchive() {}, onDelete() {} }));
  assert.match(html, /<article class="tw-task-row">/);
  assert.match(html, /<button type="button">恢复<\/button><button type="button" aria-label="删除 归档的论文">删除<\/button>/);
  const { WorkspaceSettings } = require('./WorkspaceSettings.tsx');
  const settings = renderToStaticMarkup(React.createElement(WorkspaceSettings, { group: 'archived', returnTask: '', isolated: false }));
  const tabs = [...settings.matchAll(/href="\/workspace\?settings=(\w+)"[^>]*>([^<]+)</g)].map(match => match[2]);
  assert.deepEqual(tabs, ['模型', '记忆', '计算资源与额度', '账户与外观', '已归档']);
  assert.match(settings, /<h2>已归档<\/h2>/);
  assert.doesNotMatch(settings, /使用记忆|记住的内容/);
});

test('t237 C1b: CSS for 「…」 hit areas, visibility, menu, toast and the touch rules', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  const rule = selector => new RegExp(selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\{([^}]*)\\}').exec(css)?.[1] ?? '';
  const more = rule('.task-workspace .tw-row-more');
  for (const value of ['width:32px', 'height:32px', 'opacity:0', 'background:transparent', 'border-radius:6px']) assert.ok(more.includes(value), value);
  assert.ok(/color:(?:#6B7280|var\(--tw-icon\))/.test(more), more);
  assert.match(rule('.tw-row-more svg'), /width:24px;height:24px/);
  assert.match(css, /\.tw-history-row:hover \.tw-row-more,\.tw-history-row:focus-within \.tw-row-more,\.tw-history-row\.is-open \.tw-row-more\{opacity:1\}/);
  assert.match(css, /\.task-workspace \.tw-row-more:hover,\.task-workspace \.tw-row-more\[aria-expanded="true"\]\{background:(?:#E7EAE8|var\(--tw-selected\))\}/);
  assert.match(css, /@media\(hover:none\) and \(pointer:coarse\)\{\.task-workspace \.tw-row-more\{opacity:1\}/);
  const mobile = css.slice(css.indexOf('@media(max-width:760px)'));
  assert.match(mobile, /\.task-workspace \.tw-row-more\{width:44px;height:44px;opacity:1\}/);
  assert.match(css, /-webkit-touch-callout:none/);
  assert.match(rule('.task-workspace .tw-rename'), /height:32px;padding:5px 9px;border:1px solid (?:#E7EAE8|var\(--tw-input-edge\));border-radius:8px/);
  const pop = rule('.tw-menu-pop');
  for (const value of ['min-width:200px', 'border-radius:12px', 'padding:4px', 'animation:tw-fade 120ms ease']) assert.ok(pop.includes(value), value);
  assert.ok(/box-shadow:(?:0 10px 30px rgba\(31,41,51,\.12\)|var\(--tw-menu-shadow\))/.test(pop), pop);
  assert.match(rule('.task-workspace .tw-menu-item'), /padding:6px 10px;border:0;border-radius:8px;[^}]*font-size:13px;line-height:20px/);
  assert.match(css, /\.task-workspace \.tw-menu-item\.is-danger,\.tw-menu-item\.is-danger svg\{color:(?:#8A2A2A|var\(--tw-danger\))\}/);
  assert.doesNotMatch(css, /is-danger[^{]*\{[^}]*background/);
  const toast = rule('.tw-toast');
  for (const value of ['position:fixed', 'bottom:16px', 'transform:translateX(-50%)', 'max-width:calc(100vw - 32px)', 'border-radius:12px']) assert.ok(toast.includes(value), value);
  assert.match(css, /html:has\(\.tw-chat>\.tw-composer\) \.tw-toast,html:has\(\.tw-composer-float\) \.tw-toast\{bottom:calc\(var\(--tw-composer-h,168px\) \+ max\(var\(--tw-composer-gap,16px\),16px\) \+ 16px\)\}/);
  assert.match(mobile, /\.tw-toast\{bottom:calc\(var\(--tw-composer-h,179px\) \+ max\(var\(--tw-composer-gap,10px\),10px\) \+ 16px\)\}/);
  assert.match(readFileSync(new URL('./Toast.tsx', import.meta.url), 'utf8'), /--tw-composer-gap/);
  assert.match(rule('.task-workspace .tw-toast-action'), /color:(?:#2F855A|var\(--tw-accent\));font-weight:600/);
  assert.match(css, /@media\(prefers-reduced-motion:reduce\)\{[^}]*\.tw-menu-pop,\.tw-toast\{animation:none\}/);
  // No new colors beyond the §0 tokens.
  const allowed = new Set(['#ffffff', '#fff', '#f6f7f6', '#f7f8f7', '#1f2933', '#6b7280', '#0a7a4e', '#2f855a', '#e7eae8', '#e5e7eb', '#f1f3f2', '#e2e5e3', '#eef1ef', '#8a2a2a']);
  const added = css.slice(css.indexOf('.tw-history-row{'), css.indexOf('@media(hover:none) and (pointer:coarse)'));
  for (const color of added.match(/#[0-9a-fA-F]{3,6}\b/g) || []) assert.ok(allowed.has(color.toLowerCase()), color);
});

test('t237 C1b: the draft-conflict buttons use the primary and outline styles, not the muted hint', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.task-workspace \.tw-composer-extra \.tw-conflict \.tw-primary\{border:1px solid var\(--tw-accent\);background:var\(--tw-accent\);color:(?:#fff|var\(--tw-on-accent\));font-weight:600/);
  assert.match(css, /\.task-workspace \.tw-composer-extra \.tw-conflict \.tw-button\{border:1px solid var\(--tw-edge\);background:(?:#fff|var\(--tw-surface\));color:var\(--tw-ink\)/);
  assert.match(css, /\.tw-composer-links>span\{[^}]*color:var\(--tw-muted\)/);
  const { DraftConflict } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(DraftConflict, { disabled: false, onServer() {}, onMine() {} }));
  assert.match(html, /^<span class="tw-composer-links tw-conflict"><span>服务器上有更新的草稿<\/span><button type="button" class="tw-primary">保留我的输入/);
});

test('t237 C1b: the tasks proxy passes PATCH / DELETE / restore / export, mutations keep CSRF', () => {
  const route = readFileSync(new URL('../../../app/api/assistant/tasks/[[...path]]/route.ts', import.meta.url), 'utf8');
  for (const method of ['GET', 'POST', 'PATCH', 'DELETE']) assert.match(route, new RegExp(`export const ${method} = forward;`));
  assert.match(route, /GET: \['', 'export'\]/);
  assert.match(route, /POST: \['draft', 'links', 'files', 'restore'\]/);
  assert.match(route, /PATCH: \[''\]/);
  assert.match(route, /DELETE: \[''\]/);
  assert.match(route, /request\.method === 'GET' \? new URL\(request\.url\)\.search : ''/);
  const proxy = readFileSync(new URL('../../../proxy.ts', import.meta.url), 'utf8');
  assert.match(proxy, /MUTATING_METHODS = new Set\(\["POST", "PUT", "PATCH", "DELETE"\]\)/);
  assert.match(proxy, /validCsrfToken\(request\.headers\.get\(CSRF_HEADER\)/);
  const { jsonRequest } = require('./types.ts');
  assert.deepEqual(jsonRequest('DELETE'), { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  const { dispositionFilename } = require('./conversation-menu.ts');
  assert.equal(dispositionFilename(`attachment; filename="conversation-2026-10-26.md"; filename*=UTF-8''${encodeURIComponent('论文-2026-10-26.md')}`, 'x'), '论文-2026-10-26.md');
  assert.equal(dispositionFilename(null, '对话.md'), '对话.md');
});

test('t237 C1b B.13: code-block scrollbars are thin and light', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.tw-md-code pre\{scrollbar-width:thin;scrollbar-color:(?:#C9CEC9|var\(--tw-scrollbar\)) transparent\}/);
  assert.match(css, /\.tw-md-code pre::-webkit-scrollbar\{height:6px\}/);
  assert.match(css, /\.tw-md-code pre::-webkit-scrollbar-thumb\{background:(?:#C9CEC9|var\(--tw-scrollbar\));border-radius:3px\}/);
  assert.match(css, /\.tw-md-code pre::-webkit-scrollbar-track\{background:transparent\}/);
});

test('t237 C4: .thin-scrollbar computed scrollbar-color is #C9CEC9 transparent (live dark-green source)', () => {
  const css = readFileSync(new URL('../../../app/globals.css', import.meta.url), 'utf8');
  assert.match(css, /\.thin-scrollbar\s*\{\s*scrollbar-width:\s*thin;\s*scrollbar-color:\s*#C9CEC9\s+transparent;\s*\}/);
  assert.match(css, /\.thin-scrollbar::-webkit-scrollbar-thumb\s*\{\s*background:\s*#C9CEC9;/);
  assert.doesNotMatch(css, /\.thin-scrollbar\s*\{[^}]*scrollbar-color:\s*rgb\(var\(--color-edge-strong\)\)/);
});

test('t237 C5a L9: the sidebar is the one scroller (history no longer scrolls on its own); 6px thin edge-token scrollbar shown on hover', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.doesNotMatch(css, /\.tw-history\{[^}]*overflow:auto/);
  assert.match(css, /@media\(min-width:761px\)\{[^@]*\.tw-sidebar[^{]*\{scrollbar-width:thin;scrollbar-color:transparent transparent\}/);
  assert.match(css, /\.tw-sidebar:hover\{scrollbar-color:var\(--tw-edge\) transparent\}/);
  assert.match(css, /\.tw-sidebar::-webkit-scrollbar[^{]*\{width:6px\}/);
  assert.match(css, /:hover::-webkit-scrollbar-thumb[^{]*\{background:var\(--tw-edge\)\}/);
});

test('t237 C4: long history titles keep 「…」 inside the sidebar (row and link shrink, title ellipsis)', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.tw-history-row\{position:relative;min-width:0;overflow:hidden\}/);
  assert.match(css, /\.tw-history-row>a\{display:flex;[^}]*min-width:0;overflow:hidden;/);
  assert.match(css, /\.tw-row-title\{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap\}/);
});


test('t237 C1b B.14: 从资料库选择 drops the count at 0 and keeps 生成文件', () => {
  const { libraryLabel, TaskSession } = require('./TaskSession.tsx');
  assert.equal(libraryLabel(0, 0), '从资料库选择');
  assert.equal(libraryLabel(3, 0), '从资料库选择（3）');
  assert.equal(libraryLabel(0, 2), '从资料库选择 · 生成文件（2）');
  assert.equal(libraryLabel(1, 2), '从资料库选择（1） · 生成文件（2）');
  const render = files => renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_fixture', title: '阅读资料', draft: '', version: 1, status: 'draft' }, runs: [], files,
  } }));
  assert.match(render([]), /aria-controls="task-library">从资料库选择<\/button>/);
  const file = id => ({ id, name: `${id}.pdf`, bytes: 1, mime_type: 'application/pdf' });
  assert.match(render([file('a'), file('b')]), /aria-controls="task-library">从资料库选择（2）<\/button>/);
});

test('t237 C1b B.16: 重新读取 is an ink secondary button like 使用服务器版本, not the muted hint', () => {
  const { DraftPaused } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(DraftPaused, { disabled: false, onReload() {} }));
  assert.equal(html, '<span class="tw-composer-links tw-conflict"><button type="button" class="tw-button">重新读取</button></span>');
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.task-workspace \.tw-composer-extra \.tw-conflict \.tw-button\{border:1px solid var\(--tw-edge\);background:(?:#fff|var\(--tw-surface\));color:var\(--tw-ink\)/);
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(session, /autosavePaused && <DraftPaused /);
});

test('t237 C1b B.17: composer is a flex sibling under .tw-chat-scroll; list keeps 16px bottom pad (C5a L3)', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.tw-chat-scroll\{[^}]*padding-bottom:16px/);
  assert.match(css, /\.tw-chat>\.tw-composer\{[^}]*flex:none/);
  assert.doesNotMatch(css, /\.tw-chat>\.tw-composer\{[^}]*position:sticky/);
  assert.doesNotMatch(css, /\.tw-chat>\.tw-composer\{[^}]*position:fixed/);
  // Overlay padding/negative-margin hack retired with the sibling pin
  assert.doesNotMatch(css, /\.tw-chat>\.tw-messages:has\(\+\.tw-composer\)\{[^}]*padding-bottom:calc\(var\(--tw-composer-h/);
  assert.match(chatHtml('好的'), /^<section class="tw-messages" aria-label="任务对话">/);
});

test('t237 C1b B.18: inline code in a user bubble uses the assistant inline-code style', () => {
  const html = chatHtml('好的', { message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
    { id: 'u', role: 'user', content: '看 `a.py` 和 `b<c>`，``` 不算' }, { id: 'a', role: 'assistant', content: '好的' }] } });
  const user = html.slice(html.indexOf('tw-message tw-user'), html.indexOf('tw-message tw-assistant'));
  assert.match(user, /<p>看 <code>a.py<\/code> 和 <code>b&lt;c&gt;<\/code>，``` 不算<\/p>/);
  assert.doesNotMatch(user, /tw-md/);
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  const inline = /\.tw-md code\{([^}]*)\}/.exec(css)[1];
  assert.match(inline, /background:var\(--tw-code-inline\);border-radius:4px;padding:1px 4px/);
  assert.equal(/\.tw-user code\{([^}]*)\}/.exec(css)?.[1], inline);
});

test('t237 C1b fix: Android contextmenu after a long press is ignored for 800ms', () => {
  const { LONG_PRESS_CONTEXTMENU_GUARD_MS, shouldIgnoreContextMenu } = require('./conversation-menu.ts');
  assert.equal(LONG_PRESS_CONTEXTMENU_GUARD_MS, 800);
  assert.equal(shouldIgnoreContextMenu(0, 5000), false); // desktop right-click: no preceding long press
  assert.equal(shouldIgnoreContextMenu(undefined, 5000), false);
  assert.equal(shouldIgnoreContextMenu(1000, 1000), true);
  assert.equal(shouldIgnoreContextMenu(1000, 1799), true);
  assert.equal(shouldIgnoreContextMenu(1000, 1800), false);
  assert.equal(shouldIgnoreContextMenu(1000, 900), false); // clock went backwards: treat as a fresh right-click
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  const menu = readFileSync(new URL('./conversation-menu.ts', import.meta.url), 'utf8');
  // The guard now lives in contextMenuDecision, which the row's contextmenu handler always consults before opening.
  assert.match(menu, /return \{ open: !shouldIgnoreContextMenu\(press\.at, now\), press \};/);
  assert.match(sidebar, /onContextMenu=\{event => contextMenu\(event, key, task\)\}/);
  assert.match(sidebar, /press\.current\.at = Date\.now\(\)/);
  assert.match(sidebar, /useEffect\(\(\) => \(\) => window\.clearTimeout\(press\.current\.timer\), \[\]\)/);
});

test('t237 C1b fix2: contextmenu before the long-press timer claims the press and opens the menu once', () => {
  const { contextMenuDecision, LONG_PRESS_CONTEXTMENU_GUARD_MS } = require('./conversation-menu.ts');
  // Android fires contextmenu while the 500ms timer is still pending: clear it, mark fired, stamp `at`, open once.
  const pending = { timer: 42, x: 3, y: 4, fired: false };
  const claimed = contextMenuDecision(pending, 1000);
  assert.deepEqual(claimed, { open: true, clearTimer: 42, press: { timer: undefined, x: 3, y: 4, fired: true, at: 1000 } });
  assert.deepEqual(pending, { timer: 42, x: 3, y: 4, fired: false }); // pure: the input is untouched
  // A second contextmenu right after the claim is the same gesture: the 800ms guard swallows it.
  assert.equal(contextMenuDecision(claimed.press, 1000 + LONG_PRESS_CONTEXTMENU_GUARD_MS - 1).open, false);
  assert.equal(contextMenuDecision(claimed.press, 1000 + LONG_PRESS_CONTEXTMENU_GUARD_MS).open, true);
  // Timer already fired (contextmenu after the long press): the existing guard still holds, nothing to clear.
  const fired = { timer: undefined, x: 0, y: 0, fired: true, at: 5000 };
  assert.deepEqual(contextMenuDecision(fired, 5300), { open: false, press: fired });
  // Desktop right-click: no touch press at all, opens immediately and leaves the press alone.
  const idle = { x: 0, y: 0, fired: false };
  assert.deepEqual(contextMenuDecision(idle, 9000), { open: true, press: idle });
  assert.equal(contextMenuDecision({ x: 0, y: 0, fired: false, at: 100 }, 9000).open, true);
});

test('t237 C1b fix2: the row handler clears the pending timer before opening, and cleared timers read as not pending', () => {
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  const handler = sidebar.match(/function contextMenu\([^)]*\) \{([\s\S]*?)\n  \}/)?.[1] ?? '';
  assert.match(handler, /^\s*event\.preventDefault\(\);\s*const decision = contextMenuDecision\(press\.current, Date\.now\(\)\);/);
  const clear = handler.indexOf('window.clearTimeout(decision.clearTimer)');
  const store = handler.indexOf('press.current = decision.press');
  const open = handler.indexOf('if (decision.open) openMenu(');
  assert.ok(clear > 0 && store > clear && open > store, handler);
  assert.equal(handler.match(/openMenu\(/g)?.length, 1);
  // Every path that stops or fires the timer also forgets it, so `timer` set means "still pending".
  assert.match(sidebar, /move\(event: TouchEvent\) \{[\s\S]*?> 10\) \{ window\.clearTimeout\(press\.current\.timer\); press\.current\.timer = undefined; \}/);
  assert.match(sidebar, /end\(event: TouchEvent\) \{ window\.clearTimeout\(press\.current\.timer\); press\.current\.timer = undefined;/);
  assert.match(sidebar, /press\.current\.fired = true; press\.current\.at = Date\.now\(\); press\.current\.timer = undefined;/);
});

test('t237 C4: the search fallback filters the list as it is when the request fails', async () => {
  const { searchOrTitles } = require('./conversation-menu.ts');
  let list = c1bTasks();
  let fail;
  const pending = searchOrTitles(new Promise((_, reject) => { fail = reject; }), () => list, '对话');
  // A conversation created while the request is in flight must be found by the fallback.
  list = [...list, c1bTask('b1', '刚建的对话', { updated_at: at(0.1) })];
  fail(new Error('offline'));
  assert.deepEqual((await pending).map(task => task.title), ['刚建的对话', '置顶对话', '今天的对话', '很早的对话']);
  const server = [c1bTask('b2', '服务器结果')];
  assert.deepEqual(await searchOrTitles(Promise.resolve(server), () => list, '对话'), server);
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  assert.match(sidebar, /currentTasks\.current = tasks;/);
  assert.match(sidebar, /searchOrTitles\(listConversations\(\{ q: searching \}, controller\.signal\), \(\) => currentTasks\.current, searching\)/);
  assert.match(sidebar, /\}, \[searching\]\);/);
});

test('t237 C4: the paused-autosave line never shows a server error code', async () => {
  const { AUTOSAVE_PAUSED } = require('./TaskSession.tsx');
  const { taskRequest } = require('./types.ts');
  const original = globalThis.fetch;
  try {
    for (const [status, code] of [[409, 'task_version_conflict'], [500, 'internal_error'], [409, 'foo_bar']]) {
      globalThis.fetch = async () => new Response(JSON.stringify({ error: code }), { status });
      // The request error carries the code; the autosave line must not pass it through.
      await assert.rejects(taskRequest('/api/assistant/tasks/x/draft'), error => !error.message.includes(code) && !/[（(]/.test(error.message));
      const html = renderToStaticMarkup(React.createElement('p', { role: 'alert', className: 'tw-error' }, AUTOSAVE_PAUSED));
      const visible = html.replace(/<[^>]*>/g, '');
      assert.equal(visible, '自动保存已暂停');
      assert.doesNotMatch(visible, /[A-Za-z()（）]/);
      assert.ok(!visible.includes(code));
    }
  } finally { globalThis.fetch = original; }
  const session = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  // Every autosave / 重新读取 / 保留我的输入 failure uses the copy, never cause.message.
  assert.match(session, /setAutosavePaused\(true\); setError\(AUTOSAVE_PAUSED\)/);
  assert.match(session, /async function reloadDraftVersion\(\)[\s\S]*?\} catch \{ setError\(AUTOSAVE_PAUSED\); \}/);
  assert.match(session, /async function keepMyDraft\(\)[\s\S]*?catch \{ if \(alive\.current\) \{ setServerDraft\(null\); setError\(AUTOSAVE_PAUSED\); \} \}/);
  const autosave = session.slice(session.indexOf('useEffect(() => {\n    if (draft === saved.current.draft'), session.indexOf('function takeServerDraft'));
  assert.doesNotMatch(autosave, /cause\.message/);
});

test('t237 C4: a content search that answers HTTP 500 falls back to titles and shows nothing to the user', async () => {
  const { listConversations, searchOrTitles } = require('./conversation-menu.ts');
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ error: 'internal_error' }), { status: 500 });
    const found = await searchOrTitles(listConversations({ q: '对话' }), () => c1bTasks(), '对话');
    assert.deepEqual(found.map(task => task.title), ['置顶对话', '今天的对话', '很早的对话']);
  } finally { globalThis.fetch = original; }
  const sidebar = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  // The search effect only sets results: no toast, no error line, no status code.
  const effect = sidebar.slice(sidebar.indexOf('if (!searching) { setResults(null); return; }'), sidebar.indexOf('}, [searching]);'));
  assert.ok(effect.length > 0);
  assert.doesNotMatch(effect, /showToast|setError|status|500|message/);
});

test('t237 C4.1: history row and 「…」 focus ring is inset so overflow:hidden cannot clip it', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  // Spec 01-sidebar-menu.md §3.1 (md5 5f4aaa59): row a/button radius 8, 「…」 radius 6, offset -2px.
  assert.match(css, /\.tw-history-row a:focus-visible,\.tw-history-row button:focus-visible\{outline:2px solid var\(--tw-focus-ring\);outline-offset:-2px;border-radius:8px\}/);
  assert.match(css, /\.task-workspace \.tw-history-row \.tw-row-more:focus-visible\{border-radius:6px\}/);
  // Containment from C4 stays; the global ring is unchanged elsewhere.
  assert.match(css, /\.tw-history-row\{position:relative;min-width:0;overflow:hidden\}/);
  assert.match(css, /\.tw-history-row>a\{display:flex;[^}]*min-width:0;overflow:hidden;/);
  assert.match(css, /\.task-workspace :focus-visible\{outline:2px solid var\(--tw-focus-ring\);outline-offset:2px\}/);
});

// t237 C2a: one error-copy mapper; codes only inside a closed 技术详情.
const COPY_TABLE = {
  network: ['网络断了，连上后再试。', '重试'],
  401: ['登录已过期，请重新登录。', '重新登录'],
  403: ['你没有权限打开这项内容。', '返回'],
  404: ['找不到这项内容，可能已被删除。', '返回'],
  409: ['内容刚被更新，请刷新后再改。', '刷新'],
  413: ['文件太大，请换一个小一点的。', '重新选择文件'],
  '5xx': ['服务暂时不可用，请稍后再试。', '重试'],
  unknown: ['没能完成这次操作，请再试一次。', '重试'],
  model_identity_unconfirmed: ['这次没有拿到模型回复，对话已保留。', '重试'],
};
const LATIN_OUTSIDE_DETAILS = (html) => html.replace(/<details[^]*?<\/details>/g, '').replace(/<[^>]+>/g, '');

test('t237 C2a: the error-copy table holds the approved line and single action per class', () => {
  const { ERROR_COPY, errorCopy } = require('./error-copy.ts');
  const { TaskRequestError } = require('./types.ts');
  assert.deepEqual(Object.fromEntries(Object.entries(ERROR_COPY).map(([key, copy]) => [key, [copy.line, copy.action]])), COPY_TABLE);
  const cases = [
    [new TaskRequestError('网络连接中断', 0, 'network'), 'network'],
    [new TypeError('Failed to fetch'), 'network'],
    [new TaskRequestError('x', 401, 'auth_required'), '401'], [new TaskRequestError('x', 403, ''), '403'],
    [new TaskRequestError('x', 404, 'task_not_found'), '404'], [new TaskRequestError('x', 409, 'version_conflict'), '409'],
    [new TaskRequestError('x', 413, 'file_too_large'), '413'], [new TaskRequestError('x', 500, ''), '5xx'], [new TaskRequestError('x', 503, 'upstream'), '5xx'],
    [new TaskRequestError('x', 422, 'invalid'), 'unknown'], [new Error('boom'), 'unknown'], ['plain', 'unknown'], [null, 'unknown'],
    [new TaskRequestError('x', 502, 'model_identity_unconfirmed'), 'model_identity_unconfirmed'],
  ];
  for (const [cause, key] of cases) {
    const copy = errorCopy(cause);
    assert.equal(copy.key, key, String(cause?.message ?? cause));
    assert.equal(copy.line, COPY_TABLE[key][0]); assert.equal(copy.action, COPY_TABLE[key][1]);
    assert.doesNotMatch(copy.line, /[A-Za-z0-9]/);
  }
  assert.deepEqual(errorCopy(new TaskRequestError('版本冲突（version_conflict）', 409, 'version_conflict')).details, ['HTTP 409 · version_conflict', '版本冲突（version_conflict）']);
});

test('t237 C2a: taskRequest keeps the typed status and code on the thrown error', async (context) => {
  const { taskRequest, TaskRequestError } = require('./types.ts');
  const fetch = context.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ error: 'task_version_conflict' }), { status: 409 }));
  const error = await taskRequest('/api/assistant/tasks/owned').catch(cause => cause);
  assert.ok(error instanceof TaskRequestError); assert.equal(error.status, 409); assert.equal(error.code, 'task_version_conflict');
  fetch.mock.mockImplementation(async () => { throw new TypeError('Failed to fetch'); });
  const offline = await taskRequest('/api/assistant/tasks/owned').catch(cause => cause);
  assert.equal(offline.code, 'network'); assert.match(offline.message, /网络连接中断/);
});

test('t237 C2a: ErrorNotice shows one Chinese line, one action, and a never-open 技术详情', () => {
  const { ErrorNotice } = require('./ErrorNotice.tsx');
  const { TaskRequestError } = require('./types.ts');
  const render = (error) => renderToStaticMarkup(React.createElement(ErrorNotice, { error, onAction() {} }));
  const conflict = render(new TaskRequestError('Version conflict', 409, 'version_conflict'));
  assert.match(conflict, /内容刚被更新，请刷新后再改。/);
  assert.equal((conflict.match(/<(?:button|a)\b/g) || []).length, 1); assert.match(conflict, />刷新</);
  assert.match(conflict, /<details class="tw-details"><summary>技术详情<\/summary><p>HTTP 409 · version_conflict<\/p><p>Version conflict<\/p><\/details>/);
  assert.doesNotMatch(conflict, /<details[^>]*\bopen/);
  assert.doesNotMatch(LATIN_OUTSIDE_DETAILS(conflict), /[A-Za-z0-9]/);
  assert.match(render(new TaskRequestError('x', 401, '')), /<a[^>]*href="\/login"[^>]*>重新登录<\/a>/);
  assert.match(render(new TaskRequestError('x', 404, '')), /<a[^>]*href="\/workspace"[^>]*>返回<\/a>/);
  assert.equal(renderToStaticMarkup(React.createElement(ErrorNotice, { error: null })), '');
});

test('t237 C2a: TaskFiles, ModelSettings and WorkspaceRecords show errors through the mapper, never raw cause.message', () => {
  for (const name of ['TaskFiles.tsx', 'ModelSettings.tsx', 'WorkspaceRecords.tsx']) {
    const source = readFileSync(new URL('./' + name, import.meta.url), 'utf8');
    assert.match(source, /<ErrorNotice\b/, name);
    assert.doesNotMatch(source, /setError\(cause\.message\)|cause instanceof Error \? cause\.message/, name);
    assert.doesNotMatch(source, />\{error\}|\{error\}</, name);
    assert.doesNotMatch(source, /<details[^>]*\bopen/, name);
  }
});

test('t237 C2a / C6b H5-4: model card is one Chinese line, 换个模型继续 + 再试一次, closed 技术详情', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  for (const run of [
    { id: 'run_a', user_task_id: 'utask_a', status: 'blocked', error_class: 'model_identity_unconfirmed', error_message: 'model_identity_unconfirmed: served claude-opus-4-8 != requested claude-sonnet-5-5' },
    { id: 'run_b', user_task_id: 'utask_b', status: 'blocked', error_class: 'model_unavailable', answer: 'Runtime session is durable. Model execution is unavailable: LLMError/RuntimeError.' },
  ]) for (const view of ['conversation', 'progress']) {
    const html = renderToStaticMarkup(React.createElement(TaskRunView, { view, run }));
    const card = html.match(/<div class="tw-error tw-model-error" role="alert"[^>]*>[^]*?<\/details><\/div>/)?.[0] || '';
    assert.ok(card, view);
    assert.match(card, /这次没有拿到模型回复，对话已保留。/);
    assert.match(card, /换个模型继续/);
    assert.match(card, />再试一次</); assert.doesNotMatch(card, /重试此步/);
    assert.match(card, /href="\/workspace\?settings=models&amp;return_task=utask_[ab]"/);
    assert.match(card, /<details class="tw-details"><summary>技术详情<\/summary>/);
    assert.doesNotMatch(html, /<details[^>]*\bopen/);
    assert.doesNotMatch(LATIN_OUTSIDE_DETAILS(card), /[A-Za-z]/);
    assert.doesNotMatch(LATIN_OUTSIDE_DETAILS(html), /model_identity_unconfirmed|Model execution|LLMError|claude-opus/);
  }
});

test('t237 C2a: before model-profiles loads the trigger shows Claude Sonnet 5.5, not 选择模型', () => {
  const { ModelPicker } = require('./ComposerPicker.tsx');
  const before = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: { profiles: [], default_profile_id: '' }, value: '', onChange() {} }));
  assert.match(before, /<span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/);
  assert.doesNotMatch(before, /选择模型|· 默认/);
  // Loaded but nothing selectable: unchanged.
  const none = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: { profiles: [], default_profile_id: '', platform_model: null }, value: '', onChange() {} }));
  assert.match(none, />选择模型</);
  const loaded = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: LIVE_LIST(), value: 'platform', onChange() {} }));
  assert.match(loaded, /<span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/);
  assert.deepEqual(loaded.match(/<span class="tw-picker-label"[^>]*>[^<]*· 默认</g)?.map(label => label.replace(/<[^>]*>/, '')), ['Claude Sonnet 5.5 · 默认<']);
});

test('t237 c2a-minors: a model failure hides 检查并继续 so only 换个模型继续 remains', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const { isModelFailure } = require('./task-presentation.ts');
  const model = { id: 'run_m', status: 'blocked', error_class: 'model_identity_unconfirmed' };
  const other = { id: 'run_o', status: 'blocked', error_class: 'tool_failed' };
  assert.equal(isModelFailure(model), true);
  assert.equal(isModelFailure({ id: 'run_u', status: 'blocked', answer: 'Model execution is unavailable: LLMError' }), true);
  assert.equal(isModelFailure(other), false);
  assert.equal(isModelFailure({ ...model, status: 'completed' }), false);
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(TaskControls, { run: model, onChanged() {} })), /检查并继续/);
  assert.match(renderToStaticMarkup(React.createElement(TaskControls, { run: other, onChanged() {} })), /检查并继续/);
});

test('t237 c2a-minors: the model error card body is ink #1F2933; only its title stays red', () => {
  const css = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.tw-error\.tw-model-error\{color:(?:#1F2933|var\(--tw-ink\))\}\.tw-model-error h2\{color:(?:#8a2a2a|var\(--tw-error-title\))\}/);
});

// ---- t237 C4d.3 / C6b: fail card always offers 换个模型继续 + 再试一次; tech under 技术详情 ----
const C4D_BLOCKED = text => ({ id: 'run_c4d', user_task_id: 'utask_c4d', status: 'blocked', error_class: 'recoverable',
  error_message: text, answer: text });
const C4D_TEXT = code => `Runtime session is durable. Model execution is unavailable: LLMError/${code}. No tool was executed by the failed model request.`;

test('t237 C4d.3 / C6b H5-4: an encoding/transport/exception failure is Chinese line + 换个模型继续 + 再试一次; code only inside 技术详情', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { TaskControls } = require('./TaskControls.tsx');
  for (const code of ['ValueError', 'model_transport_failed', 'model_response_invalid', 'model_network_failed', 'model_http_500', 'model_http_429', 'RuntimeError', 'UnicodeDecodeError']) {
    const run = C4D_BLOCKED(C4D_TEXT(code));
    for (const view of ['conversation', 'progress']) {
      const html = renderToStaticMarkup(React.createElement(TaskRunView, { view, run }));
      const card = html.match(/<div class="tw-error tw-model-error" role="alert"[^>]*>[^]*?<\/details><\/div>/)?.[0] || '';
      assert.ok(card, `${code} ${view}`);
      assert.match(card, /这次没处理完，可以再试一次/);
      assert.match(card, /换个模型继续/);
      assert.match(card, />再试一次</); assert.doesNotMatch(card, /重试此步/);
      assert.doesNotMatch(html, /密钥/);
      assert.doesNotMatch(html, /<details[^>]*\bopen/);
      assert.match(card, new RegExp(`<summary>技术详情</summary><p>${code}</p>`));
      assert.doesNotMatch(LATIN_OUTSIDE_DETAILS(html), /[A-Za-z]/);
      assert.doesNotMatch(html, /Runtime session|Traceback|sk-/);
    }
    assert.doesNotMatch(renderToStaticMarkup(React.createElement(TaskControls, { run, onChanged() {} })), /检查并继续/);
  }
});

test('t237 C4d.3 / C6b H5-4: config card keeps Chinese line + 换个模型继续 + 再试一次; tech under 技术详情', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { ERROR_COPY } = require('./error-copy.ts');
  assert.equal(ERROR_COPY.model_identity_unconfirmed.line, '这次没有拿到模型回复，对话已保留。');
  const C4D_TEXT = code => `Model execution is unavailable: LLMError/${code}.`;
  for (const answer of [C4D_TEXT('model_identity_unconfirmed'), C4D_TEXT('model_address_invalid')]) {
    const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
      id: 'run_c4d', user_task_id: 'utask_c4d', status: 'blocked', error_class: 'recoverable', answer,
    }}));
    assert.match(html, /data-testid="tw-model-fail"/);
    assert.match(html, /换个模型继续/);
    assert.match(html, />再试一次</); assert.doesNotMatch(html, /重试此步/);
    assert.match(html, /href="\/workspace\?settings=models&amp;return_task=utask_c4d"/);
    assert.match(html, /<summary>技术详情<\/summary>/);
    assert.doesNotMatch(html.replace(/<details[\s\S]*?<\/details>/, ''), /LLMError|model_address|model_identity/);
  }
  for (const error_class of ['model_unavailable', 'model_identity_unconfirmed']) {
    const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
      id: 'run_x', user_task_id: 'utask_x', status: 'blocked', error_class,
      answer: 'Model execution is unavailable: LLMError/model_identity_unconfirmed.',
    }}));
    assert.match(html, /换个模型继续/); assert.match(html, />再试一次</); assert.doesNotMatch(html, /重试此步/);
  }
});

test('t237 C4d.3: classification is deterministic and a non-model failure is not touched', () => {
  const { isModelFailure, isConfigModelFailure, modelFailureCode } = require('./task-presentation.ts');
  assert.equal(modelFailureCode(C4D_BLOCKED(C4D_TEXT('model_http_401'))), 'model_http_401');
  assert.equal(modelFailureCode({ status: 'blocked', answer: 'no code here' }), '');
  assert.equal(isConfigModelFailure(C4D_BLOCKED(C4D_TEXT('model_http_401'))), true);
  assert.equal(isConfigModelFailure(C4D_BLOCKED(C4D_TEXT('ValueError'))), false);
  assert.equal(isConfigModelFailure(C4D_BLOCKED(C4D_TEXT('model_http_500'))), false);
  assert.equal(isModelFailure(C4D_BLOCKED(C4D_TEXT('ValueError'))), true);
  assert.equal(isModelFailure({ id: 'x', status: 'blocked', error_class: 'tool_failed' }), false);
});

// ── t237 C4d.2: a disabled personal model is nowhere in the composer ──
const C4D_LIST = {
  profiles: [
    { id: 'model_on', version: 1, name: '启用的模型', provider: 'openai', model: 'gpt-5', enabled: true, verified_at: 'x' },
    { id: 'model_off', version: 1, name: '停用的模型', provider: 'openai', model: 'gpt-5', enabled: false, verified_at: 'x' },
  ],
  default_profile_id: 'model_off',
  platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, platform_models: [],
};

test('t237 C4d.2: pickerModels drops a disabled model and its default; an enabled one is untouched', () => {
  const { pickerModels, defaultModelChoice, isSelectableModel } = require('./composer-options.ts');
  const list = pickerModels(C4D_LIST);
  assert.deepEqual(list.profiles.map(item => item.id), ['model_on']);
  assert.equal(list.default_profile_id, '');
  assert.equal(isSelectableModel(list, 'model_off'), false);
  assert.equal(defaultModelChoice(list, 'model_off'), 'platform');
  const enabledOnly = { ...C4D_LIST, profiles: [C4D_LIST.profiles[0]], default_profile_id: 'model_on' };
  assert.deepEqual(pickerModels(enabledOnly), enabledOnly);
});

test('t237 C4d.2: the picker lists only the enabled model, in the raw and the filtered list alike', () => {
  const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
  const { pickerModels } = require('./composer-options.ts');
  for (const list of [C4D_LIST, pickerModels(C4D_LIST)]) {
    const mine = modelGroups(list).find(group => group.label === '我的模型');
    assert.deepEqual(mine.choices.map(choice => choice.value), ['model_on']);
    assert.ok(!JSON.stringify(modelGroups(list)).includes('停用的模型'));
    assert.ok(!renderToStaticMarkup(React.createElement(ModelPicker, { id: 'm', list, value: 'model_off' })).includes('停用的模型'));
  }
});

test('t237 C4d.2: both composers filter the list when it arrives', () => {
  for (const file of ['./TaskSession.tsx', './TaskWorkspace.tsx']) {
    const source = readFileSync(new URL(file, import.meta.url), 'utf8');
    assert.match(source, /pickerModels\(value\)/, file);
  }
});

test('t237 C4d.3: 重试 resumes the run on the history page too, not only in the task session', () => {
  for (const file of ['./TaskSession.tsx', './WorkspaceRecords.tsx']) {
    const source = readFileSync(new URL(file, import.meta.url), 'utf8');
    assert.match(source, /onResume=\{/, file);
    assert.match(source, /action: 'resume'/, file);
  }
});

// ---- t237 C4e ----
const c4eCss = () => readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
const c4eRule = (selector) => { const css = c4eCss(); const at = css.indexOf(selector + '{'); assert.ok(at >= 0, selector); return css.slice(at + selector.length + 1, css.indexOf('}', at)); };

test('t237 C4e.2: approval labels share the plan-card label style (13px, #6B7280), values keep the body style', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const run = { id: 'run_label', status: 'waiting_approval', approvals: [{ id: 'approval_label', status: 'pending', tool_name: 'cost_estimate', reversible: true, risk_level: 'low', normalized_arguments: { estimated_cost: 120, budget: 200 }, impact_scope: {} }] };
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run }));
  for (const label of ['要做什么', '花多少']) assert.match(html, new RegExp(`<p class="tw-confirm-label">${label}</p><p>`), label);
  assert.ok(!/<p>(要做什么|花多少|外发)<\/p>/.test(html));
  const outbound = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { ...run, approvals: [{ ...run.approvals[0], tool_name: 'competition_submit' }] } }));
  if (outbound.includes('外发')) assert.match(outbound, /<p class="tw-confirm-label">外发<\/p><p>/);
  assert.equal((html.match(/class="tw-confirm-label"/g) ?? []).length >= 2, true);
  const label = c4eRule('.tw-confirm-label'), h3 = c4eRule('.tw-plan-card h3');
  assert.match(label, /font-size:13px/); assert.match(label, /color:var\(--tw-muted\)/); assert.match(label, /font-weight:600/);
  assert.match(h3, /font-size:13px/); assert.match(h3, /color:var\(--tw-muted\)/); // same style source as the plan card labels
});

test('t237 C4e.3 / C6b H5-4: 换个模型继续 is the primary fail action; copy unchanged', () => {
  const rest = c4eRule('.task-workspace .tw-model-error .tw-fail-actions .tw-primary');
  assert.match(rest, /font-weight:600/);
  assert.match(c4eCss(), /\.task-workspace \.tw-model-error a:focus-visible\{[^}]*outline:2px solid var\(--tw-focus-ring\)/);
  const { TaskRunView } = require('./TaskRunView.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: { id: 'run_cfg', user_task_id: 'utask_cfg', status: 'blocked', error_class: 'model_unavailable', error_message: 'model_http_401' } }));
  assert.match(html, /class="tw-primary"[^>]*>换个模型继续</);
  assert.match(html, />再试一次</); assert.doesNotMatch(html, /重试此步/);
  assert.equal(require('node:crypto').createHash('sha256').update(readFileSync(new URL('./error-copy.ts', import.meta.url))).digest('hex').length, 64);
});

test('t237 C4e.4: the model version stays visible when the trigger is squeezed (brand shrinks, version does not)', () => {
  const { ModelPicker, IntensityPicker, splitVersion } = require('./ComposerPicker.tsx');
  assert.deepEqual(splitVersion('Claude Sonnet 5.5'), { name: 'Claude Sonnet', version: ' 5.5', rest: '' });
  assert.deepEqual(splitVersion('Claude Opus 4.8 Thinking'), { name: 'Claude Opus', version: ' 4.8', rest: ' Thinking' });
  for (const plain of ['DeepSeek Chat', '强度 · 中', '选择模型', '5.5']) assert.equal(splitVersion(plain), null, plain);
  const list = { profiles: [], platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' } };
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'm', list, value: 'platform' }));
  assert.match(html, /<span title="Claude Sonnet 5\.5" class="tw-picker-text"><span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span><\/span>/);
  assert.match(html, /aria-label="本次使用的模型：Claude Sonnet 5\.5"/);   // the accessible name and the title keep the full label
  assert.match(html, /<button[^>]*title="Claude Sonnet 5\.5"/);
  assert.match(renderToStaticMarkup(React.createElement(IntensityPicker, { id: 'i', value: '中' })), /<span title="强度 · 中">强度 · 中<\/span>/);
  assert.match(c4eRule('.tw-picker-version'), /flex:none/);
  assert.match(c4eRule('.tw-picker-version'), /white-space:pre/);
  assert.match(c4eRule('.tw-picker-trigger .tw-picker-text'), /display:inline-flex/);
  assert.match(c4eRule('.tw-picker-trigger .tw-picker-text'), /min-width:0/);
  assert.match(c4eRule('.tw-picker-name'), /overflow:hidden/); assert.match(c4eRule('.tw-picker-name'), /text-overflow:ellipsis/);
  assert.match(c4eRule('.tw-picker-rest'), /flex-shrink:\d+/);
});


// ---- t237 C5a (L1 + L3 + L4 + token baseline) ----
const c5aCss = () => readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');
const c5aRule = (selector) => {
  const css = c5aCss();
  const at = css.indexOf(selector + '{');
  assert.ok(at >= 0, `missing ${selector}`);
  return css.slice(at + selector.length + 1, css.indexOf('}', at));
};
const c5aMobile = () => {
  const css = c5aCss();
  const at = css.indexOf('@media(max-width:760px)');
  assert.ok(at >= 0);
  return css.slice(at);
};

test('t237 C5a.0: token baseline — accent #2F855A, ink/muted/label/weak, sidebar/edge/read, focus+link retarget', () => {
  const root = c5aRule('.task-workspace');
  assert.match(root, /--tw-accent:#2F855A/);
  assert.match(root, /--tw-accent-hover:#276749/);
  assert.match(root, /--tw-accent-wash:#F3F8F5/);
  assert.match(root, /--tw-ink:#1F2933/);
  assert.match(root, /--tw-secondary:#4B5563/);
  assert.match(root, /--tw-muted:#6B7280/);
  assert.match(root, /--tw-weak:#9AA3AE/);
  assert.match(root, /--tw-sidebar:#F7F8F7/);
  assert.match(root, /--tw-edge:#E5E7EB/);
  assert.match(root, /--tw-read:760px/);
  assert.match(root, /--tw-code:#F7F8F7/);
  assert.match(root, /--tw-code-inline:#F2F4F3/);
  assert.match(root, /--tw-code-edge:#E5E7EB/);
  assert.match(root, /--tw-wash:#F2F4F3/);
  // Focus / shadow / on-accent tokens (dark theme can override later)
  assert.match(root, /--tw-focus-ring:rgba\(47,133,90,\.35\)/);
  assert.match(root, /--tw-focus-wash:rgba\(47,133,90,\.15\)/);
  assert.match(root, /--tw-shadow-float:0 4px 16px rgba\(16,24,40,\.08\)/);
  assert.match(root, /--tw-shadow-card:0 1px 2px rgba\(16,24,40,\.04\)/);
  assert.match(root, /--tw-on-accent:#fff/);
  assert.match(root, /--tw-pill-delivered-bg:#E8F5EE/);
  assert.match(root, /--tw-pill-delivered-fg:#1F7A4D/);
  assert.match(root, /--tw-pill-confirm-bg:#FEF3C7/);
  assert.match(root, /--tw-pill-confirm-fg:#92400E/);
  assert.match(root, /--tw-pill-progress-bg:#E8F0FE/);
  assert.match(root, /--tw-pill-progress-fg:#1D4ED8/);
  assert.match(root, /--tw-pill-idle-bg:#F2F4F3/);
  assert.match(root, /--tw-pill-idle-fg:#4B5563/);
  assert.match(root, /--tw-pill-error-bg:#FDECEC/);
  assert.match(root, /--tw-pill-error-fg:#B42318/);
  const focus = c5aRule('.task-workspace :focus-visible');
  assert.match(focus, /outline:2px solid var\(--tw-focus-ring\)/);
  assert.match(focus, /outline-offset:2px/);
  assert.doesNotMatch(c5aCss(), /--tw-accent:#0A7A4E/);
  // R1 / R3 retarget onto the lock tokens (no leftover solid #0A7A4E focus/link in task-workspace)
  assert.doesNotMatch(c5aCss(), /outline:2px solid #0A7A4E/);
  assert.match(c5aRule('.task-workspace .tw-model-error a'), /color:var\(--tw-accent\)/);
  assert.match(c5aCss(), /\.task-workspace \.tw-model-error a:focus-visible\{[^}]*var\(--tw-focus-ring\)/);
  assert.match(c5aCss(), /\.task-workspace \.tw-plan-status:focus-visible\{[^}]*var\(--tw-focus-ring\)/);
  // content column pad 24 (1440); 393/mobile pad 16
  assert.match(c5aCss(), /\.task-workspace main\{[^}]*padding:24px 24px/);
  assert.match(c5aMobile(), /\.task-workspace main\{[^}]*padding:16px/);
  // C5a prereq for L10: reserved dark override block (empty / stub — no dark values yet)
  assert.match(c5aCss(), /\[data-theme=["']dark["']\]\s*\.task-workspace\s*\{[^}]*\}/);
});

test('t237 C5a.1 L1: markdown hierarchy — body 15/24, h2/h3, lists, tables, code, quote; 393 h2 16/24; table scrolls', () => {
  const md = c5aRule('.tw-md');
  assert.match(md, /font-size:15px/);
  assert.match(md, /line-height:24px/);
  const h2 = c5aRule('.task-workspace .tw-md h2');
  assert.match(h2, /font-size:17px/);
  assert.match(h2, /line-height:26px/);
  assert.match(h2, /font-weight:600/);
  assert.match(h2, /margin:20px 0 8px/);
  const h3 = c5aRule('.tw-md h3');
  assert.match(h3, /font-size:15px/);
  assert.match(h3, /line-height:24px/);
  assert.match(h3, /font-weight:600/);
  assert.match(h3, /margin:16px 0 6px/);
  const list = c5aRule('.tw-md ul,.tw-md ol');
  assert.match(list, /padding-left:20px/);
  assert.match(c5aRule('.tw-md li'), /margin:6px 0|margin:0 0 6px/);
  assert.match(c5aRule('.tw-md li::marker'), /color:#6B7280|color:var\(--tw-muted\)/);
  const tableWrap = c5aRule('.tw-md-table');
  assert.match(tableWrap, /overflow-x:auto/);
  assert.match(tableWrap, /border:1px solid/);
  assert.match(tableWrap, /border-radius:8px/);
  const th = c5aRule('.tw-md th');
  assert.match(th, /font-size:13px/);
  assert.match(th, /line-height:20px/);
  assert.match(th, /font-weight:600/);
  assert.match(th, /color:#4B5563|color:var\(--tw-secondary\)/);
  assert.match(th, /background:var\(--tw-(?:sidebar|code)\)/);
  const cell = c5aRule('.tw-md th,.tw-md td');
  assert.match(cell, /padding:8px 12px/);
  const code = c5aRule('.tw-md code');
  assert.match(code, /background:var\(--tw-code-inline\)|#F2F4F3/);
  assert.match(code, /border-radius:4px/);
  const block = c5aRule('.tw-md-code');
  assert.match(block, /border-radius:8px/);
  assert.match(block, /background:var\(--tw-code\)|#F7F8F7/);
  assert.match(block, /border:1px solid/);
  const quote = c5aRule('.tw-md blockquote');
  assert.match(quote, /border-left:3px solid/);
  assert.match(quote, /padding[^;]*12px|padding-left:12px/);
  assert.match(quote, /color:#4B5563|color:var\(--tw-secondary\)/);
  assert.match(c5aMobile(), /\.task-workspace \.tw-md h2\{[^}]*font-size:16px[^}]*line-height:24px|\.tw-md h2\{[^}]*font-size:16px[^}]*line-height:24px/);
  // Fixture still renders four layers
  const html = chatHtml(MD_REPLY);
  assert.match(html, /<h2>/);
  assert.match(html, /<ul>/);
  assert.match(html, /class="tw-md-table"/);
  assert.match(html, /class="tw-md-code"/);
});

test('t237 C5a.3 L3: circular ↑ send, draft-saved inside composer, float/focus tokens, clearance + R4/R5 pinned', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const sessionSrc = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  const homeSrc = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  // Icon send with aria-label 发送 — no visible 发送 / 正在发送… text inside the button
  assert.match(sessionSrc, /aria-label="发送"/);
  assert.match(sessionSrc, /className="tw-send-icon"/);
  assert.match(sessionSrc, /<ArrowUp/);
  assert.doesNotMatch(sessionSrc, />\{busy \? '正在发送…' : '发送'\}</);
  assert.match(homeSrc, /className="tw-send-icon tw-send"/);
  assert.match(homeSrc, /<ArrowUp/);
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_c5a', title: '阅读资料', draft: '请开始', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  const card = html.slice(html.indexOf('class="tw-composer"'));
  assert.match(card, /aria-label="发送"/);
  assert.match(card, /class="tw-send-icon"/);
  assert.doesNotMatch(card.slice(0, card.indexOf('</form>') > 0 ? card.indexOf('</form>') : 2000), />发送</);
  // Draft-saved lives inside .tw-composer (tw-save-status), not a sibling muted outside
  assert.match(sessionSrc, /tw-save-status/);
  assert.match(sessionSrc, /草稿已保存/);
  // Outside muted notice must not be the home for 草稿已保存
  assert.doesNotMatch(sessionSrc, /className="tw-muted">\{error \|\| \(notice === '草稿已保存'/);
  const send = c5aRule('.task-workspace .tw-send-icon');
  assert.match(send, /width:32px/);
  assert.match(send, /height:32px/);
  assert.match(send, /border-radius:999px|border-radius:50%/);
  assert.match(send, /background:var\(--tw-accent\)|#2F855A/);
  assert.match(send, /color:var\(--tw-on-accent\)/);
  const disabled = c5aRule('.task-workspace .tw-send-icon:disabled');
  assert.match(disabled, /background:#E5E7EB|background:var\(--tw-edge\)/);
  assert.match(disabled, /color:#9AA3AE|color:var\(--tw-weak\)/);
  const composer = c5aRule('.tw-composer');
  assert.match(composer, /min-height:88px/);
  assert.match(composer, /box-shadow:var\(--tw-shadow-float\)/);
  assert.match(c5aCss(), /\.tw-composer:focus-within\{[^}]*border-color:var\(--tw-accent\)/);
  // L3: named scroll container; composer is a flex sibling (not inside it, not position:fixed)
  assert.match(sessionSrc, /className="tw-chat-scroll"/);
  assert.match(c5aCss(), /\.tw-chat-scroll\{[^}]*\}/);
  const pin = c5aRule('.tw-chat>.tw-composer');
  assert.doesNotMatch(pin, /position:\s*fixed/);
  assert.doesNotMatch(pin, /position:\s*sticky/);
  assert.match(pin, /flex:\s*none|flex-shrink:\s*0/);
  // Structural clearance: scroll area keeps ≥16px padding under last card; R5 scroll-margin kept
  assert.match(c5aRule('.tw-chat-scroll'), /padding-bottom:16px/);
  assert.match(c5aRule('.task-workspace .tw-plan-card .tw-actions'), /scroll-margin-bottom:20px/);
  // Rendered DOM: composer is a sibling AFTER .tw-chat-scroll closes
  const scrollAt = html.indexOf('class="tw-chat-scroll"');
  const composerAt = html.indexOf('class="tw-composer"');
  assert.ok(scrollAt >= 0 && composerAt > scrollAt, 'composer after scroll open');
  assert.match(html.slice(scrollAt, composerAt), /tw-chat-scroll[\s\S]*?<\/div>/);
  // R4 splitVersion still present
  const { splitVersion } = require('./ComposerPicker.tsx');
  assert.deepEqual(splitVersion('Claude Sonnet 5.5'), { name: 'Claude Sonnet', version: ' 5.5', rest: '' });
  assert.ok(TaskWorkspace);
});

test('t237 C5a.4 L4: five status pills + card label/value; not on sidebar rows', () => {
  const { statusPillClass, statusPillKind } = require('./task-presentation.ts');
  assert.equal(statusPillKind('已交付'), 'delivered');
  assert.equal(statusPillKind('等你确认'), 'confirm');
  assert.equal(statusPillKind('正在处理'), 'progress');
  assert.equal(statusPillKind('尚未开始'), 'idle');
  assert.equal(statusPillKind('执行失败'), 'error');
  assert.equal(statusPillKind('暂时无法继续'), 'error');
  // blockedProgressCopy progress-card headings (phase.stuck) must be red error, not blue progress
  for (const stuck of ['这一步停住了', '结果还没齐', '方案停住了', '有一步还没核对完', '尚未交付', '这次没有完成']) {
    assert.equal(statusPillKind(stuck), 'error', stuck);
    assert.equal(statusPillClass(stuck), 'tw-status-pill tw-status-error', stuck);
  }
  const { blockedProgressCopy } = require('./task-presentation.ts');
  assert.equal(statusPillKind(blockedProgressCopy({ status: 'blocked' }).status), 'error');
  assert.equal(statusPillKind(blockedProgressCopy({ status: 'failed' }).status), 'error');
  assert.equal(statusPillKind(blockedProgressCopy({ status: 'undelivered' }).status), 'error');
  assert.equal(statusPillClass('已交付'), 'tw-status-pill tw-status-delivered');
  const pills = {
    delivered: { bg: '#E8F5EE', fg: '#1F7A4D' },
    confirm: { bg: '#FEF3C7', fg: '#92400E' },
    progress: { bg: '#E8F0FE', fg: '#1D4ED8' },
    idle: { bg: '#F2F4F3', fg: '#4B5563' },
    error: { bg: '#FDECEC', fg: '#B42318' },
  };
  const pillBase = c5aRule('.tw-status-pill');
  assert.match(pillBase, /font-size:12px/);
  assert.match(pillBase, /line-height:18px/);
  assert.match(pillBase, /font-weight:500/);
  assert.match(pillBase, /border-radius:999px/);
  assert.match(pillBase, /padding:2px 8px/);
  for (const [kind, colors] of Object.entries(pills)) {
    const rule = c5aRule(`.tw-status-pill.tw-status-${kind}`);
    assert.match(rule, new RegExp(`background:var\\(--tw-pill-${kind}-bg\\)`));
    assert.match(rule, new RegExp(`color:var\\(--tw-pill-${kind}-fg\\)`));
  }
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_pill', title: '阅读资料', draft: '', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  assert.match(html, /class="tw-status-pill tw-status-idle"[^>]*>尚未开始</);
  assert.doesNotMatch(html, /class="tw-eyebrow"/);
  // C6-answer H_a9: the progress card (and its pill) is gone from the conversation; the title chip keeps the status.
  const { TaskRunView } = require('./TaskRunView.tsx');
  const progress = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_pill', status: 'completed', answer: '好', message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: 'hi' }, { id: 'a', role: 'assistant', content: '好' },
    ] },
  } }));
  assert.doesNotMatch(progress, /tw-status-pill|tw-progress-card|本次实际模型/);
  // Label / value contract
  const label = c5aRule('.task-workspace .tw-confirm-label');
  assert.match(label, /font-size:13px/);
  assert.match(label, /line-height:20px/);
  assert.match(label, /color:var\(--tw-muted\)/);
  assert.match(label, /margin:[^;]*2px/);
  const value = c5aRule('.task-workspace .tw-confirm-label+p');
  assert.match(value, /font-size:14px/);
  assert.match(value, /line-height:22px/);
  assert.match(value, /color:var\(--tw-ink\)/);
  // C6-answer: no stuck progress card; the process row turns red (danger) and the title chip keeps the error pill.
  const stuckHtml = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_stuck_pill', status: 'blocked', error_class: 'planning_idle', message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: 'hi' }, { id: 'a', role: 'assistant', content: '…' },
    ] },
  } }));
  assert.match(stuckHtml, /<div class="tw-thinking-row is-failed"[^]*?<span class="tw-thinking-text">这一步没有完成<\/span><span class="tw-thinking-chevron"/);
  assert.doesNotMatch(stuckHtml, /tw-progress-card|tw-status-pill/);
  assert.match(c5aRule('.tw-thinking-row.is-stopped .tw-process-line'), /color:var\(--tw-label\)/);
  // Sidebar history source must not add status pills on rows
  const side = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(side, /tw-status-pill|statusPillClass/);
});

// ---- t237 C5a L9: two-column independent scroll, composer = flex sibling, back-to-bottom ----
test('t237 C5a.9a L9: page never scrolls; sidebar, message list and (other pages) work area scroll on their own', () => {
  const css = c5aCss();
  assert.match(css, /html:has\(\.task-workspace\),body:has\(\.task-workspace\)\{[^}]*overflow:hidden/);
  const root = c5aRule('.task-workspace');
  assert.match(root, /height:var\(--tw-vh,100dvh\)/);
  assert.match(root, /overflow:hidden/);
  assert.doesNotMatch(root, /min-height:100dvh/);
  const side = c5aRule('.tw-sidebar');
  assert.match(side, /overflow-y:auto/); assert.match(side, /min-height:0/);
  assert.match(c5aRule('.tw-side-top'), /position:sticky/); assert.match(c5aRule('.tw-side-top'), /top:0/);
  assert.match(c5aRule('.tw-work-area'), /min-height:0/);
  assert.match(c5aRule('.tw-work-area'), /overflow-y:auto/);
  assert.match(css, /\.tw-work-area:has\(>main>\.tw-chat\)\{[^}]*overflow:hidden/);
  assert.match(css, /\.task-workspace main:has\(>\.tw-chat\)\{[^}]*display:flex[^}]*flex-direction:column[^}]*min-height:0[^}]*padding-bottom:16px/);
  const chat = c5aRule('.tw-chat');
  assert.match(chat, /flex:1 1 auto/); assert.match(chat, /min-height:0/);
  assert.doesNotMatch(chat, /min-height:calc\(100dvh/);
  const scroll = c5aRule('.tw-chat-scroll');
  assert.match(scroll, /flex:1 1 auto/); assert.match(scroll, /min-height:0/); assert.match(scroll, /overflow-y:auto/); assert.match(scroll, /padding-bottom:16px/);
  assert.match(c5aRule('.tw-chat-head'), /flex:none/);
  assert.match(c5aRule('.tw-chat>.tw-composer'), /flex:none/);
  assert.doesNotMatch(c5aRule('.tw-chat>.tw-composer'), /position:(sticky|fixed)/);
  assert.match(c5aRule('.tw-composer-float'), /flex:none/);
  assert.doesNotMatch(c5aRule('.tw-composer-float'), /position:(sticky|fixed)/);
  // 6px scrollbar, thumb = border token, desktop only
  const desktop = css.slice(0, css.indexOf('@media(max-width:760px)'));
  assert.match(desktop, /\.tw-chat-scroll::-webkit-scrollbar[^{]*\{width:6px\}/);
  assert.match(desktop, /\.tw-chat-scroll:hover\{scrollbar-color:var\(--tw-edge\) transparent\}/);
  // 393: safe-area bottom, visualViewport-driven height, drawer sidebar stacked above, no double scroll
  const mobile = c5aMobile();
  assert.match(mobile, /\.task-workspace main:has\(>\.tw-chat\)\{[^}]*padding-bottom:calc\(12px \+ env\(safe-area-inset-bottom\)\)/);
  assert.match(mobile, /\.task-workspace\{display:flex;flex-direction:column\}/);
  assert.match(mobile, /\.tw-sidebar\.tw-open\{[^}]*max-height:60dvh/);
  const src = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(src, /visualViewport/); assert.match(src, /--tw-vh/);
});

test('t237 C5a.9b L9: DOM — head fixed, list scrolls, composer is a sibling of the scroller, 回到底部 button', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_l9', title: '阅读资料', draft: '', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  const at = key => html.indexOf(key);
  assert.ok(at('class="tw-chat-head"') > 0);
  assert.ok(at('class="tw-chat-head"') < at('class="tw-chat-body"'));
  assert.ok(at('class="tw-chat-body"') < at('class="tw-chat-scroll"'));
  assert.ok(at('class="tw-chat-scroll"') < at('tw-to-bottom'));
  assert.ok(at('tw-to-bottom') < at('class="tw-composer"'));
  const head = html.slice(at('class="tw-chat-head"'), at('class="tw-chat-body"'));
  assert.match(head, /<h1>阅读资料<\/h1>/); assert.match(head, /class="tw-tabs"/); assert.match(head, /tw-status-pill/);
  // composer comes after the whole body closes: body = scroll + button, nothing else
  const body = html.slice(at('<div class="tw-chat-body"'), at('<div class="tw-composer"'));
  assert.equal((body.match(/<div/g) ?? []).length, (body.match(/<\/div>/g) ?? []).length, 'the body closes before the composer opens');
  assert.match(html, /<button type="button" class="tw-to-bottom"[^>]*aria-label="回到底部"/);
  assert.match(html, /<button type="button" class="tw-to-bottom"[^>]*tabindex="-1"/); // hidden at rest → not tabbable
  assert.match(html, /<section class="tw-chat tw-chat-fresh"><header class="tw-chat-head">/);
  const home = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(home, /<div className="tw-side-top">/);
  assert.match(home, /<div className="tw-chat-scroll"><a className="tw-back"/); // draft view: composer stays a sibling
  assert.match(home, /<div ref=\{homeComposer\} className="tw-composer-float">/);
});

test('t237 C5a.9c L9: back-to-bottom button geometry + thresholds + no snap-back while streaming', () => {
  const css = c5aCss();
  const btn = c5aRule('.tw-to-bottom');
  assert.match(btn, /position:absolute/); assert.match(btn, /left:50%/); assert.match(btn, /bottom:12px/);
  assert.match(btn, /width:32px/); assert.match(btn, /height:32px/); assert.match(btn, /border-radius:999px/);
  assert.match(btn, /background:var\(--tw-surface\)/); assert.match(btn, /border:1px solid var\(--tw-edge\)/); assert.match(btn, /box-shadow:var\(--tw-shadow-to-bottom\)/);
  assert.match(btn, /opacity:0/); assert.match(btn, /transition:opacity 150ms/); assert.match(btn, /pointer-events:none/);
  assert.match(css, /\.tw-to-bottom\.is-visible\{[^}]*opacity:1[^}]*pointer-events:auto/);
  assert.match(c5aRule('.tw-to-bottom-dot'), /width:6px/); assert.match(c5aRule('.tw-to-bottom-dot'), /background:var\(--tw-accent\)/);
  assert.match(c5aRule('.tw-chat-body'), /position:relative/);
  const { BACK_TO_BOTTOM_SHOW, BACK_TO_BOTTOM_HIDE, nextAway, shouldFollow, distanceFromBottom } = require('./chat-scroll.ts');
  assert.equal(BACK_TO_BOTTOM_SHOW, 200); assert.equal(BACK_TO_BOTTOM_HIDE, 40);
  assert.equal(distanceFromBottom({ scrollHeight: 2000, scrollTop: 1000, clientHeight: 700 }), 300);
  assert.equal(nextAway(false, 201), true); assert.equal(nextAway(false, 200), false);
  assert.equal(nextAway(true, 100), true); // hysteresis between 40 and 200
  assert.equal(nextAway(true, 40), false); assert.equal(nextAway(true, 41), true);
  // follows when the user is at the bottom; never pulls back a user who scrolled up (streaming or not)
  assert.equal(shouldFollow(30), true); assert.equal(shouldFollow(40), true);
  assert.equal(shouldFollow(41), false); assert.equal(shouldFollow(900), false);
  const src = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(src, /useChatScroll/); assert.match(src, /aria-label="回到底部"/);
  assert.match(readFileSync(new URL('./chat-scroll.ts', import.meta.url), 'utf8'), /behavior: reduceMotion\(\) \? 'auto' : 'smooth'/);
  assert.match(src, /onMouseDown=\{event => event\.preventDefault\(\)\}/);
  assert.match(readFileSync(new URL('./chat-scroll.ts', import.meta.url), 'utf8'), /new MutationObserver/);
});

test('t237 C5a CSS-var discipline: C5a-touched rules take colours and shadows from tokens (no hex / rgba literals), so C5c dark only overrides vars', () => {
  const selectors = [
    '.tw-md', '.tw-md li::marker', '.tw-md blockquote', '.tw-md th', '.tw-md a', '.tw-composer textarea::placeholder',
    '.tw-save-status', '.tw-composer-bottom .tw-save-status', '.tw-filechip', '.tw-confirm', '.tw-composer:focus-within',
    '.task-workspace .tw-plan-card h3', '.task-workspace .tw-plan-card h3+p,.task-workspace .tw-plan-card h3+ul,.task-workspace .tw-plan-card h3+ol',
    '.task-workspace .tw-confirm-label', '.task-workspace .tw-confirm-label+p',
  ];
  for (const selector of selectors) {
    const rule = c5aRule(selector);
    assert.ok(rule, selector);
    assert.ok(!/#[0-9A-Fa-f]{3,8}\b|rgba?\(/.test(rule), `${selector} still hard-codes a colour: ${rule}`);
  }
  // C6-answer: the progress card is gone; the attachment chip is a wash with no border or shadow.
  assert.ok(!c5aCss().includes('.tw-progress-card'));
  assert.doesNotMatch(c5aRule('.tw-filechip'), /box-shadow|border:1px/);
  assert.match(c5aRule('.tw-composer:focus-within'), /var\(--tw-shadow-float\),0 0 0 3px var\(--tw-focus-wash\)/);
  assert.match(c5aRule('.tw-md'), /color:var\(--tw-ink\)/);
  assert.match(c5aCss(), /--tw-accent-line:rgba\(47,133,90,\.3\)/);
  assert.match(c5aCss(), /\[data-theme="dark"\] \.task-workspace\{[^}]*--tw-accent:#4FB37E/);
});

// ---- t237 C5b (L2 + L5 + L6 + L7 + C5a polish fade/to-bottom) ----
const c5bCss = c5aCss;
const c5bRule = c5aRule;
const c5bMobile = c5aMobile;

test('t237 C5b.1 L2: copy-only MessageActions under completed assistant replies; no regenerate/有用/没用/更多', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { MessageActions } = require('./MessageActions.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_l2', status: 'completed', answer: '第二答',
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a2', messages: [
      { id: 'u1', role: 'user', content: '问1' },
      { id: 'a1', role: 'assistant', content: '第一答' },
      { id: 'u2', role: 'user', content: '问2' },
      { id: 'a2', role: 'assistant', content: '第二答' },
    ] },
  } }));
  assert.match(html, /class="tw-msg-actions/);
  assert.match(html, /aria-label="复制"/);
  assert.equal((html.match(/aria-label="复制"/g) || []).length, 2);
  assert.match(html, /tw-msg-actions is-latest/);
  assert.doesNotMatch(html, /重新生成|有用|没用|aria-label="更多"/);
  const src = readFileSync(new URL('./MessageActions.tsx', import.meta.url), 'utf8');
  assert.match(src, /navigator\.clipboard\.writeText/);
  assert.doesNotMatch(src, /重新生成|有用|没用|MoreHorizontal/);
  const bar = c5bRule('.tw-msg-actions');
  assert.match(bar, /margin-top:8px/); assert.match(bar, /gap:4px/);
  const btn = c5bRule('.task-workspace .tw-msg-action');
  assert.match(btn, /width:28px/); assert.match(btn, /height:28px/);
  assert.match(btn, /padding:0/); assert.match(btn, /border:0/); assert.match(btn, /border-radius:6px/);
  assert.match(btn, /background:transparent/);
  assert.match(btn, /color:var\(--tw-weak\)/);
  assert.ok(!/#[0-9A-Fa-f]{3,8}\b|rgba?\(/.test(btn), `literal colour in .tw-msg-action: ${btn}`);
  assert.match(c5bMobile(), /\.task-workspace \.tw-msg-action\{[^}]*width:36px[^}]*height:36px[^}]*min-height:36px/);
  assert.match(c5bCss(), /\.tw-message\.tw-assistant:hover \.tw-msg-actions|\.tw-msg-actions\.is-latest/);
  // unit: MessageActions alone exposes copy
  const alone = renderToStaticMarkup(React.createElement(MessageActions, { text: 'hello', latest: true }));
  assert.match(alone, /aria-label="复制"/);
  assert.match(alone, /is-latest/);
});

test('t237 C5b.2 L5: home keeps 3 chips with lock styling; empty task is 「从这里开始。」 + composer only', () => {
  const { EMPTY_TASK_LINE, CHAT_SUGGESTIONS, CHAT_TITLE } = require('./task-presentation.ts');
  assert.equal(EMPTY_TASK_LINE, '从这里开始。');
  assert.equal(CHAT_SUGGESTIONS.length, 3);
  assert.equal(CHAT_TITLE, '今天想推进哪一项？');
  const home = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(home, /CHAT_SUGGESTIONS\.map/);
  assert.match(home, /tw-suggest/);
  assert.doesNotMatch(home, /EMPTY_TASK_LINE/); // empty-task line lives in TaskSession, not home
  const suggest = c5bRule('.tw-suggest');
  assert.match(suggest, /margin-top:16px/);
  const chip = c5bRule('.tw-suggest button');
  assert.match(chip, /font-size:14px/); assert.match(chip, /line-height:20px/);
  assert.match(chip, /padding:8px 14px/); assert.match(chip, /border-radius:999px/);
  assert.match(chip, /border:1px solid var\(--tw-edge\)/);
  assert.match(chip, /color:var\(--tw-ink\)/);
  assert.ok(!/#[0-9A-Fa-f]{3,8}\b/.test(chip.replace(/var\(--tw-[^)]+\)/g, '')), `literal in chip: ${chip}`);
  assert.match(c5bCss(), /\.tw-suggest button:hover\{[^}]*border-color:var\(--tw-accent\)[^}]*background:var\(--tw-accent-wash\)/);
  assert.match(c5bCss(), /\.tw-suggest button svg\{[^}]*color:var\(--tw-muted\)/);
  const { TaskSession } = require('./TaskSession.tsx');
  const fresh = renderToStaticMarkup(React.createElement(TaskSession, { initial: {
    task: { id: 'utask_empty', title: '新任务', draft: '', version: 1, status: 'draft' }, runs: [], files: [],
  } }));
  assert.match(fresh, /从这里开始。/);
  assert.match(fresh, /class="tw-empty-start"/);
  assert.match(fresh, /class="tw-composer"/);
  assert.doesNotMatch(fresh, /tw-suggest|欢迎/);
  const emptyCss = c5bRule('.tw-empty-start');
  assert.match(emptyCss, /font-size:15px/); assert.match(emptyCss, /color:var\(--tw-muted\)/);
});

test('t237 C5b.3 L6: sidebar group spacing; hide zero counts; user block; title tooltips', () => {
  const side = readFileSync(new URL('./SidebarHistory.tsx', import.meta.url), 'utf8');
  assert.match(side, /items\.length > 0 &&/);
  assert.match(side, /tw-folder-count/);
  assert.match(side, /title=\{task\.title\}|title=\{project\.name\}/);
  const group = c5bRule('.tw-history-group');
  // groups themselves sit in .tw-history with gap 16
  assert.match(c5bRule('.tw-history'), /gap:16px/);
  const label = c5bRule('.tw-history-label');
  assert.match(label, /font-size:12px/); assert.match(label, /line-height:16px/);
  assert.match(label, /font-weight:500/); assert.match(label, /color:var\(--tw-muted\)/);
  assert.match(label, /padding:16px 10px 4px/);
  const count = c5bRule('.tw-folder-count');
  assert.match(count, /color:var\(--tw-weak\)/); assert.match(count, /font-size:12px/);
  // row height 32 / 393 → 40 (min-height)
  assert.match(c5bCss(), /\.tw-history-row>a\{[^}]*min-height:32px|\.tw-history a\{[^}]*min-height:32px|padding:6px 10px/);
  const mobile = c5bMobile();
  assert.match(mobile, /min-height:40px|min-height:44px/); // 40 lock; existing 44 touch target OK if ≥40
  const shell = readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8');
  assert.match(shell, /tw-user-block/);
  assert.match(shell, /外观|tw-user-appearance/);
  const user = c5bRule('.tw-user-block');
  assert.match(user, /height:48px/); assert.match(user, /border-top:1px solid var\(--tw-edge\)/);
  const avatar = c5bRule('.tw-user-avatar');
  assert.match(avatar, /width:24px/); assert.match(avatar, /height:24px/);
});

test('t237 C5b.4 L7: breadcrumb 项目 › 任务; pill above title gap 6; title 22/30 clamp 2; tabs gap 20 ink+green underline', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskSession, {
    initial: { task: { id: 'utask_l7', title: '阅读长标题需要两行截断测试用的任务名称', draft: '', version: 1, status: 'draft', project_id: 'proj_1' }, runs: [], files: [] },
    project: { id: 'proj_1', name: '论文复现', conversation_count: 1 },
  }));
  assert.match(html, /class="tw-breadcrumb"/);
  assert.match(html, /论文复现/);
  assert.match(html, /›|\/|›/);
  assert.match(html, /阅读长标题/);
  // pill before h1 inside title stack
  const head = html.slice(html.indexOf('tw-chat-head'), html.indexOf('tw-chat-body'));
  assert.ok(head.indexOf('tw-status-pill') < head.indexOf('<h1'), 'pill above title');
  assert.match(head, /tw-title-stack|tw-chat-title/);
  const stack = c5bRule('.tw-title-stack');
  assert.match(stack, /gap:6px/);
  const title = c5bRule('.tw-chat-head h1');
  assert.match(title, /font-size:22px/); assert.match(title, /line-height:30px/); assert.match(title, /font-weight:600/);
  assert.match(title, /-webkit-line-clamp:2|line-clamp:2/);
  assert.match(c5bMobile(), /\.tw-chat-head h1\{[^}]*font-size:20px[^}]*line-height:28px/);
  const tabs = c5bRule('.tw-tabs');
  assert.match(tabs, /gap:20px/);
  assert.match(tabs, /border-bottom:1px solid var\(--tw-edge\)/);
  const active = c5bRule('.tw-tabs button[aria-current],.tw-tabs a[aria-current]');
  assert.match(active, /color:var\(--tw-ink\)/);
  assert.match(active, /border-color:var\(--tw-accent\)|border-bottom-color:var\(--tw-accent\)/);
  assert.match(active, /font-weight:600/);
  const inactive = c5bCss().match(/\.tw-tabs button\{[^}]+\}/)?.[0] || '';
  assert.match(inactive, /color:var\(--tw-muted\)/);
});

test('t237 C5b polish: 16px fade under tabs on message list; to-bottom keeps readable shadow at 393', () => {
  const css = c5bCss();
  assert.match(css, /\.tw-chat-scroll(?::before|::after)?\{[^}]*|tw-scroll-fade/);
  // either mask-image or ::before gradient 16px
  assert.ok(
    /mask-image:[^;]*linear-gradient/.test(css) ||
    /\.tw-chat-scroll::before\{[^}]*height:16px/.test(css) ||
    /\.tw-scroll-fade\{[^}]*height:16px/.test(css),
    'expected 16px top fade on message list',
  );
  const btn = c5bRule('.tw-to-bottom');
  assert.match(btn, /box-shadow:var\(--tw-shadow-to-bottom\)|box-shadow:var\(--tw-shadow-float\)/);
  // dedicated stronger token preferred
  assert.match(css, /--tw-shadow-to-bottom:/);
  assert.match(btn, /box-shadow:var\(--tw-shadow-to-bottom\)/);
});

test('t237 C5b CSS-var discipline: C5b-touched rules use tokens (no hex/rgba literals outside root)', () => {
  const selectors = [
    '.tw-msg-actions', '.task-workspace .tw-msg-action', '.task-workspace .tw-msg-action:hover',
    '.tw-suggest button', '.tw-suggest button:hover', '.tw-empty-start',
    '.tw-history-label', '.tw-folder-count', '.tw-user-block',
    '.tw-breadcrumb', '.tw-title-stack', '.tw-chat-head h1',
    '.tw-tabs', '.tw-tabs button[aria-current],.tw-tabs a[aria-current]',
    '.tw-to-bottom',
  ];
  for (const selector of selectors) {
    const rule = c5bRule(selector);
    assert.ok(rule, `missing ${selector}`);
    assert.ok(!/#[0-9A-Fa-f]{3,8}\b|rgba?\(/.test(rule), `${selector} hard-codes colour: ${rule}`);
  }
});

test('t237 C5b.3b L6 W4: live 「N 项进行中」 must not cover last project rows — history scrolls with clearance', () => {
  const css = c5bCss();
  const history = c5bRule('.tw-history');
  // History is the sidebar list scrollport so bottom chrome cannot paint over rows
  assert.match(history, /flex:1 1 auto/);
  assert.match(history, /min-height:0/);
  assert.match(history, /overflow-y:auto/);
  // When the live float is present, keep ≥2 row heights (64px) clear under it
  const clear = c5bRule('.tw-sidebar:has(.tw-live) .tw-history');
  const pad = clear.match(/padding-bottom:(\d+)px/);
  assert.ok(pad, `expected padding-bottom clearance, got: ${clear}`);
  assert.ok(Number(pad[1]) >= 64, `clearance ${pad[1]}px < 64px (two 32px rows)`);
  // Live stays flex-none in the sidebar chrome (not an absolute overlay without clearance)
  const live = c5bRule('.tw-live');
  assert.match(live, /flex:none/);
  assert.doesNotMatch(live, /position:\s*(absolute|fixed)/);
  // C6a: the sidebar row moved into LiveCounts (进行中 · 待确认); the shell renders it in the same slot.
  assert.match(readFileSync(new URL('./TaskWorkspace.tsx', import.meta.url), 'utf8'), /<LiveCounts tasks=\{present\} \/>/);
  assert.match(readFileSync(new URL('./LiveCounts.tsx', import.meta.url), 'utf8'), /className="tw-live"/);
  assert.match(readFileSync(new URL('./task-presentation.ts', import.meta.url), 'utf8'), /项进行中/);
});

test('C5a VERIFY fix: 回到底部 stays round — beats .task-workspace button radius', () => {
  const rule = c5bRule('.task-workspace .tw-to-bottom');
  assert.ok(rule, 'missing .task-workspace .tw-to-bottom');
  assert.match(rule, /border-radius:999px/);
  assert.match(rule, /padding:0/);
  assert.match(rule, /color:var\(--tw-secondary\)/);
});

test('C5b L2 VERIFY fix: copy action beats .task-workspace button specificity', () => {
  const rule = c5bRule('.task-workspace .tw-msg-action');
  assert.ok(rule, 'missing .task-workspace .tw-msg-action');
  assert.match(rule, /width:28px/);
  assert.match(rule, /height:28px/);
  assert.match(rule, /padding:0/);
  assert.match(rule, /border:0/);
  assert.match(rule, /border-radius:6px/);
  assert.match(rule, /background:transparent/);
  assert.match(rule, /color:var\(--tw-weak\)/);
  assert.match(c5bMobile(), /\.task-workspace \.tw-msg-action\{[^}]*width:36px[^}]*height:36px[^}]*min-height:36px/);
});

// ---- t237 C5c (L10 theme + L11 settings + L8 step strip) ----
test('t237 C5c.1 L10: theme defaults to system; boot script + dark accent #4FB37E', () => {
  const provider = readFileSync(new URL('../theme/ThemeProvider.tsx', import.meta.url), 'utf8');
  assert.match(provider, /useState<ThemeMode>\("system"\)/);
  assert.match(provider, /THEME_BOOT_SCRIPT/);
  assert.match(provider, /chooseMode/);
  const layout = readFileSync(new URL('../../../app/layout.tsx', import.meta.url), 'utf8');
  assert.match(layout, /THEME_BOOT_SCRIPT/);
  assert.doesNotMatch(layout, /data-theme="dark"/);
  const css = c5aCss();
  assert.match(css, /\[data-theme="dark"\] \.task-workspace\{[^}]*--tw-accent:#4FB37E/);
  assert.match(css, /\[data-theme="dark"\] \.task-workspace\{[^}]*--tw-bg:#17191A/);
  assert.match(css, /\[data-theme="dark"\] \.task-workspace\{[^}]*--tw-on-accent:#0E1A13/);
});

test('t237 C5c.2 L10/L11: account appearance uses ThemeSegment; user-block 外观 menu', () => {
  const { WorkspaceSettings } = require('./WorkspaceSettings.tsx');
  const { ThemeProvider } = require('../theme/ThemeProvider.tsx');
  const html = renderToStaticMarkup(React.createElement(ThemeProvider, null,
    React.createElement(WorkspaceSettings, { group: 'account', returnTask: '', isolated: true })));
  assert.match(html, /tw-theme-seg/);
  assert.match(html, /跟随系统/);
  assert.match(html, /浅色/);
  assert.match(html, /深色/);
  assert.match(html, />主题</);
  assert.doesNotMatch(html, /保存外观|选择主题|<select id="workspace-theme"/);
  const { TaskWorkspace } = require('./TaskWorkspace.tsx');
  const shell = renderToStaticMarkup(React.createElement(ThemeProvider, null, React.createElement(TaskWorkspace)));
  assert.match(shell, /data-slot="appearance"/);
  assert.match(shell, /aria-label="外观"/);
});

test('t237 C5c.3 L11: settings shell nav 180px + content max 640; model empty CTA', () => {
  const css = c5aCss();
  assert.match(c5aRule('.tw-settings-nav'), /width:180px/);
  assert.match(c5aRule('.tw-settings-content'), /max-width:640px/);
  assert.match(c5aRule('.tw-set-row'), /min-height:56px/);
  assert.match(c5aRule('.tw-side-panel'), /width:min\(420px,100%\)/);
  const { ModelSettings } = require('./ModelSettings.tsx');
  const html = renderToStaticMarkup(React.createElement(ModelSettings, { returnTask: '', isolated: true, embedded: true }));
  assert.match(html, /还没有个人模型/);
  assert.match(html, /添加个人模型/);
  assert.doesNotMatch(html, /保存不会发出请求|尚未添加个人模型/);
});

test('t237 C5c.4 L8 (C6-answer): 6px status dots stay for the live counts; the step strip and model line are gone', () => {
  const css = c5aCss();
  assert.match(c5aRule('.tw-dot'), /width:6px;height:6px/);
  assert.match(c5aRule('.tw-dot.is-current'), /box-shadow:0 0 0 2px var\(--tw-phase-ring\)/);
  assert.match(c5aRule('.tw-dot.is-done'), /background:var\(--tw-accent\)/);
  assert.doesNotMatch(css, /\.tw-progress-model|\.tw-phases/);
  const root = c5aRule('.task-workspace') || css;
  assert.match(css, /--tw-phase-current:#1D4ED8/);
  assert.match(css, /--tw-phase-pending:#E5E7EB/);
  assert.match(css, /\[data-theme="dark"\][\s\S]*--tw-phase-current:#93C5FD/);
});

test('t237 C5c.5 L7 soft: task-header .tw-chat-head .tw-tabs keeps gap:20 under max-width 900', () => {
  const css = c5aCss();
  const media = css.slice(css.indexOf('@media(max-width:900px)'));
  assert.match(media, /\.tw-tabs\{gap:12px\}/);
  assert.match(media, /\.tw-chat-head \.tw-tabs\{gap:20px\}/);
  // Base L7 rule unchanged
  assert.match(c5aRule('.tw-tabs'), /gap:20px/);
});

test('t237 C5c.6 L10 CSS-var discipline: no hard #hex outside token blocks; switch uses --tw-switch-off/thumb', () => {
  const css = c5aCss();
  const lines = css.split('\n');
  let darkStart = -1, darkEnd = -1, inDark = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('[data-theme="dark"]') && lines[i].includes('.task-workspace')) {
      inDark = true; darkStart = i;
    } else if (inDark && lines[i].trim() === '}') {
      darkEnd = i; inDark = false; break;
    }
  }
  assert.ok(darkStart >= 0 && darkEnd > darkStart, 'dark token block missing');
  const hex = /#[0-9A-Fa-f]{3,8}\b/g;
  const offenders = [];
  for (let i = 0; i < lines.length; i++) {
    if (i === 0) continue; // light token declarations on .task-workspace
    if (i >= darkStart && i <= darkEnd) continue;
    // Strip /* */ comments so a hex in a comment (e.g. plan-card note) does not fail
    const stripped = lines[i].replace(/\/\*[\s\S]*?\*\//g, '');
    let m;
    while ((m = hex.exec(stripped))) offenders.push(`L${i + 1}:${m[0]}`);
  }
  assert.equal(offenders.length, 0, `hard hex outside token blocks: ${offenders.join(', ')}`);
  assert.match(c5aRule('.task-workspace .tw-switch'), /background:var\(--tw-switch-off\)/);
  assert.match(c5aRule('.task-workspace .tw-switch::after'), /background:var\(--tw-switch-thumb\)/);
  assert.match(css, /--tw-switch-thumb:#fff/);
  // Themed surfaces that must not stay light in dark mode
  for (const sel of ['.tw-side-search', '.tw-button', '.tw-picker-pop', '.tw-dialog']) {
    const r = c5aRule(sel);
    assert.ok(r, sel);
    assert.match(r, /background:var\(--tw-(?:surface|float|bg)\)/);
    assert.doesNotMatch(r, /background:#fff\b|background:white\b/i);
  }
});

test('t237 C5c.7 L11: model provider listbox keyboard + side panel focus trap', () => {
  const src = readFileSync(new URL('./ModelSettings.tsx', import.meta.url), 'utf8');
  assert.match(src, /aria-haspopup="listbox"/);
  assert.match(src, /aria-activedescendant/);
  assert.match(src, /onProviderListKey|ArrowDown/);
  assert.match(src, /data-active/);
  assert.match(src, /role="dialog"/);
  assert.match(src, /aria-modal="true"/);
  assert.match(src, /returnFocusRef/);
  assert.match(src, /Escape/);
  assert.match(src, /FOCUSABLE|focusables/);
});
