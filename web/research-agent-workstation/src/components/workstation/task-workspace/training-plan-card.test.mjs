// 2026-11-03: the training consent card shows the plan (scenario, algorithm, epochs, estimated on-card time) before any GPU use.
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
const FORBIDDEN = /受控操作|managed_tensor_train|GPU|显存|lease|租约|ProgramData|sha256|哈希|C:\\/;
const PLAN = { task: 'classification', task_label: '分类（2 类）', target_column: 'diagnosis', data_name: 'breast_cancer.csv', rows: 569, train_rows: 456,
  holdout_rows: 113, features: 30, network: '全连接神经网络（MLP）', hidden_sizes: [64, 32], epochs: 200, batch_size: 256, learning_rate: 0.001,
  steps: 400, metrics: ['Accuracy', 'Macro-F1', 'ROC-AUC'], estimated_gpu_seconds: { low: 15, high: 30 }, billing_note: '只按显卡上实际训练的时间计费；环境准备和排队不计费。' };

test('trainingPlanRows turns the attached plan into plain rows', () => {
  const { trainingPlanRows } = require('./task-presentation.ts');
  const rows = trainingPlanRows({ impact_scope: { training_plan: PLAN } });
  const byLabel = Object.fromEntries(rows.map(row => [row.label, row.value]));
  assert.match(byLabel['数据'], /breast_cancer\.csv（569 行 × 30 个特征）/);
  assert.match(byLabel['训练场景'], /分类（2 类）.*diagnosis/);
  assert.match(byLabel['推荐算法'], /64 → 32/);
  assert.match(byLabel['训练参数'], /200 轮，批大小 256，学习率 0.001/);
  assert.match(byLabel['预计训练时间'], /约 15–30 秒/);
  assert.match(byLabel['计费'], /不计费/);
  assert.deepEqual(trainingPlanRows({ impact_scope: {} }), []);
  assert.match(trainingPlanRows({ impact_scope: { training_plan: { ...PLAN, estimated_gpu_seconds: { low: 200, high: 600 } } } }).find(r => r.label === '预计训练时间').value, /约 3–10 分钟/);
});

test('the consent card renders the plan and still never shows ids, paths or hashes', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const html = renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { id: 'run_fixture', status: 'waiting_approval', approvals: [{
    id: 'approval_fixture', status: 'pending', reversible: true, risk_level: 'high', tool_name: 'managed_tensor_train',
    normalized_arguments: { configuration_path: 'C:\\ProgramData\\EvoMind\\x\\work\\c.json', configuration_sha256: 'ab'.repeat(32) },
    impact_scope: { workspace_root: 'C:\\ProgramData\\EvoMind\\x', training_plan: PLAN } }] } }));
  assert.match(html, /aria-label="训练方案"/);
  assert.match(html, /推荐算法/); assert.match(html, /预计训练时间/); assert.match(html, /200 轮/);
  assert.doesNotMatch(html, FORBIDDEN);
});
