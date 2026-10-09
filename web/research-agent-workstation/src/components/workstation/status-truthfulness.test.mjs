import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// Render the real TSX components without a browser or a running backend.
const require = createRequire(import.meta.url);
const resolve = Module._resolveFilename;
Module._resolveFilename = function (name, ...args) {
  return resolve.call(this, name.startsWith('@/') ? path.resolve('src', name.slice(2)) : name, ...args);
};
for (const extension of ['.ts', '.tsx']) {
  Module._extensions[extension] = (module, filename) => {
    const { outputText } = ts.transpileModule(readFileSync(filename, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
      fileName: filename,
    });
    module._compile(outputText, filename);
  };
}
const render = (component, props) => renderToStaticMarkup(React.createElement(component, { locale: 'zh-CN', selectedTask: 'example-task', ...props }));
const text = (markup) => markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');

test('unexecuted control run never claims a completed DAG or passed evolution', () => {
  const { AiControlConsole } = require('./AiControlConsole.tsx');
  const markup = render(AiControlConsole, { summary: { runtime: {
    current_run: { run_id: 'wr_waiting', task_id: 'example-task', status: 'WAIT_PLAN_GATE' },
    task_graph: { nodes: [], edges: [] }, artifact_manifest: { artifacts: [] },
  } } });
  assert.doesNotMatch(text(markup), /7 \/ 7 completed|EXP001|EXP002|\bpassed\b/);
  assert.match(text(markup), /暂无任务图/);
  assert.match(text(markup), /未核验/);
});

test('gate list and context bar agree for equivalent task IDs without exposing other tasks', () => {
  const { GatesScreen } = require('./screens/GatesScreen.tsx');
  const { RunContextBar } = require('./layout/RunContextBar.tsx');
  const summary = { gates: [
    { task_id: 'example_task', gate_id: 'gate-visible', name: 'Visible review', decision: 'pending' },
    { task_id: 'other_task', gate_id: 'gate-private', name: 'Other review', decision: 'pending' },
  ] };
  const page = text(render(GatesScreen, { summary }));
  assert.match(page, /Visible review/);
  assert.doesNotMatch(page, /Other review|无闸门记录/);
  assert.match(text(render(RunContextBar, { summary })), /1 门禁待审/);
});

test('stale HPC proof cannot display a ready connector even when configured state says ready', () => {
  const { GpuHpcScreen } = require('./screens/GpuHpcScreen.tsx');
  const markup = text(render(GpuHpcScreen, { summary: { connector_status: { local_hpc: {
    configured: true, state: 'ready', current_gate_ready: false, live_probe_fresh: false,
    job_container_verified: true, designated_proxy_path_verified: true, identity_samples_passed: 5,
    proof_expires_at: '2000-01-01T00:00:00Z',
  } } } }));
  assert.doesNotMatch(markup, /就绪|已连接|连接已核验/);
  assert.match(markup, /待.*核验/);
});

test('waiting gate is counted by both the context bar and task list', () => {
  const { GatesScreen } = require('./screens/GatesScreen.tsx');
  const { RunContextBar } = require('./layout/RunContextBar.tsx');
  const summary = { gates: [{ task_id: 'example-task', gate_id: 'waiting-review', decision: ' waiting ' }] };
  assert.match(text(render(GatesScreen, { summary })), /待审 1/);
  assert.match(text(render(RunContextBar, { summary })), /1 门禁待审/);
});

test('control progress uses actual nodes and requires reviewed comparison metrics to pass', () => {
  const { AiControlConsole } = require('./AiControlConsole.tsx');
  const runtime = {
    current_run: { run_id: 'wr_reviewed', task_id: 'example-task', status: 'completed' },
    task_graph: { nodes: [{ task_id: 'a', status: 'completed' }, { task_id: 'b', status: 'completed' }] },
    review: { status: 'passed' },
    runtime_snapshot: { metrics: { self_evolution: {
      status: 'passed', parent_exp_id: 'BASE', child_exp_id: 'CHILD', before: { pr_auc: 0.7 }, after: { pr_auc: 0.8 },
    } } },
  };
  const markup = text(render(AiControlConsole, { summary: { runtime } }));
  assert.match(markup, /2 \/ 2 已完成/);
  assert.match(markup, /BASE → CHILD passed/);
  runtime.review.status = 'pending';
  assert.match(text(render(AiControlConsole, { summary: { runtime } })), /BASE → CHILD 未核验/);
});

test('unselected task never inherits another task header or global gate count', () => {
  const { GatesScreen } = require('./screens/GatesScreen.tsx');
  const { RunContextBar } = require('./layout/RunContextBar.tsx');
  const summary = {
    tasks: [{ id: 'example-task', name: 'Example research' }],
    runtime: { current_run: { task_id: 'example-task', status: 'completed' } },
    gates: [{ task_id: 'example-task', gate_id: 'gate-1', decision: 'pending' }],
  };
  const header = text(render(RunContextBar, { selectedTask: '', summary }));
  assert.doesNotMatch(header, /Example research|已完成|1 门禁待审/);
  assert.match(header, /未选择任务/);
  const page = render(GatesScreen, { selectedTask: '', summary });
  assert.match(page, /<select[^>]*aria-label="选择任务"/);
  assert.match(page, /<option value="example-task">Example research/);
});
