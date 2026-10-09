import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const require = createRequire(import.meta.url);
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};

test('task detail offers labelled conversation, progress and task-specific result without implying real execution', () => {
  const { TaskDetail } = require('../components/TaskDetail.tsx');
  const task = {
    id: 'sample-public-test', title: '公开界面验收任务', sample: true, version: 2, status: 'completed', updated: '',
    messages: [{ role: 'user', text: '只记录我的需求' }], steps: [{ title: '保存需求', state: 'done', detail: '未执行实验' }],
    result: { name: 'request-sample.md', content: '这份文件仅属于公开界面验收任务' },
  };
  const props = { task, tab: 'conversation', draft: '', busy: false, error: '', setDraft() {}, send() {}, reload() {} };
  const conversation = renderToStaticMarkup(React.createElement(TaskDetail, props));
  assert.match(conversation, /公开界面验收任务/);
  assert.match(conversation, /对话/);
  assert.match(conversation, /进度/);
  assert.match(conversation, /文件与结果/);
  assert.match(conversation, /输入需求/);
  assert.match(conversation, /查看样例回复/);
  assert.match(conversation, /未调用模型/);
  const results = renderToStaticMarkup(React.createElement(TaskDetail, { ...props, tab: 'results' }));
  assert.match(results, /这份文件仅属于公开界面验收任务/);
  assert.match(results, /\/api\/preview\/tasks\/sample-public-test\/artifact/);
  assert.doesNotMatch(results, /质量已通过/);
});
