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
  assert.match(html, /data-ui-deepevo-glyph/);
  for (const label of ['登录', '账号', '密码', '进入工作台']) assert.match(html, new RegExp(label));
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
  assert.match(empty, /尚未开始/);
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
  for (const label of ['模型', '计算资源与额度', '账户与外观', '额度由管理员管理', '返回原任务']) assert.ok(html.includes(label), label);
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
  assert.match(html, /<p class="tw-served-model"[^>]*>Claude Opus<\/p>/);
  assert.doesNotMatch(html.replace(/data-served-model="[^"]*"/g, ''), /anthropic\s*\/\s*claude|claude-opus-4-8/);
  assert.match(html, /报告\.md/);
  assert.match(html, /notes\.txt/);
  assert.match(html, /测试正文/);
  assert.doesNotMatch(html, /risk_control|本轮实际模型/);
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
    runs: [{ id: 'run_qa', status: 'completed', created_at: '2026-10-23T00:00:00Z' }], files: [],
  } }));
  assert.match(session, /未交付/);
  assert.doesNotMatch(session, /<p>[^<]*因为/);
});

test('platform label and task restore follow the served model and the owning task', () => {
  const { platformOptionLabel, productModelLabel, owningTaskHref, sessionDraft, actionName, visibleMessageText } = require('./task-presentation.ts');
  assert.equal(platformOptionLabel({ provider: 'anthropic', model: 'claude-sonnet-5' }, null), '平台模型 · Claude Sonnet');
  assert.equal(
    platformOptionLabel({ provider: 'anthropic', model: 'claude-sonnet-5' }, { provider: 'anthropic', model: 'claude-opus-4-8' }),
    '平台模型 · Claude Opus',
  );
  assert.equal(platformOptionLabel(null, null), '平台模型');
  assert.equal(productModelLabel('claude-opus-5-5'), 'Claude Opus');
  assert.equal(productModelLabel('claude-sonnet-5'), 'Claude Sonnet');
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
  assert.match(stuck, /方案停住了/);
  assert.match(stuck, /检查并继续/);
  assert.match(stuck, /class="is-stuck"/);
  assert.match(stuck, /理解目标/);
  assert.doesNotMatch(stuck, /r5-anchor-report|recoverable|技术错误|missing final/);
  assert.equal(blockedAssistantLine('结果还没齐'), '结果还没齐，我先检查再接着跑。');
  assert.equal(streamPhaseState({ status: 'blocked', error_class: 'planning_idle' }).stuck, true);
  const files = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'conversation', run: {
    id: 'run_files', status: 'completed', prompt: '整理', answer: '好了',
    artifacts: [1, 2, 3].map(index => ({ id: `artifact_${index}`, run_id: 'run_files', name: `结果${index}.md`, bytes: index * 1024, sha256: 'd'.repeat(64), preview_kind: 'text', source_tool_call: 'model_publish' })),
  } }));
  assert.match(files, /结果1\.md/);
  assert.match(files, /结果2\.md/);
  assert.match(files, /还有 1 个文件/);
  assert.ok(files.indexOf('还有 1 个文件') < files.indexOf('结果3.md'));
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
  assert.match(html, /暂停任务/);
  assert.doesNotMatch(html, /检查并继续/);
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
  assert.doesNotMatch(html.split('<details')[0], /inputs\/paper/);
});

test('platform models are selectable by id, never displace the default, and only non-default ids are sent', () => {
  const { defaultModelChoice, isSelectableModel, platformModelId } = require('./composer-options.ts');
  const { ModelOptions } = require('./ModelOptions.tsx');
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
  const html = renderToStaticMarkup(React.createElement('select', null, React.createElement(ModelOptions, { list, value: 'platform', platformLabel: '平台模型 · Claude Sonnet' })));
  assert.equal((html.match(/<option value="platform/g) || []).length, 2);
  assert.match(html, /value="platform:m1">GPT</);
  const visible = (html.match(/>[^<]*</g) || []).join('');
  for (const raw of ['gpt-4o', 'claude-sonnet-5', 'platform:']) assert.doesNotMatch(visible, new RegExp(raw));
  const deepseek = renderToStaticMarkup(React.createElement('select', null, React.createElement(ModelOptions, { list: { ...list, platform_models: [...platform_models, { id: 'm2', provider: 'deepseek', model: 'deepseek-chat', label: 'deepseek-chat' }] }, value: 'platform', platformLabel: '平台模型 · Claude Sonnet' })));
  assert.match(deepseek, /value="platform:m2">DeepSeek</);
  assert.doesNotMatch((deepseek.match(/>[^<]*</g) || []).join(''), /deepseek-chat/);
  assert.doesNotMatch(renderToStaticMarkup(React.createElement('select', null, React.createElement(ModelOptions, { list: { profiles: [], default_profile_id: '' }, value: '', platformLabel: '' }))), /平台模型/);
});
