// 2026-11-03 tri_vendor_v1: the platform picker shows Claude / GPT / 国模 with Claude Opus 5.5 as the default.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, Module } from 'node:module';
import test from 'node:test';
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
const { modelGroups, ModelPicker } = require('./ComposerPicker.tsx');
const { defaultModelChoice, PRODUCT_DEFAULT_MODEL } = require('./composer-options.ts');

const TRI = JSON.parse(readFileSync(new URL('../../../../../../tests/fixtures/tri_vendor_picker_models.json', import.meta.url), 'utf8'));
const LIST = { platform_model: { provider: 'anthropic', model: 'claude-opus-5-5' }, default_profile_id: '', profiles: [], platform_models: TRI };

test('three groups in order Claude, GPT, 国模 with Opus 5.5 · 默认 first', () => {
  const [platform] = modelGroups(LIST);
  assert.deepEqual(platform.choices.map(choice => choice.label), [
    'Claude Opus 5.5 · 默认', 'Claude Sonnet 5.5', 'Claude Haiku 5.5',
    'GPT-6 Astra', 'GPT-6 Luna', 'GPT-6 Sol', 'GPT-6.1 Sol',
    'DeepSeek 4.1 Flash', 'DeepSeek 4 Pro', 'Qwen3.8 Max', 'Qwen3.8 Flash', 'GLM 5.3',
  ]);
  assert.deepEqual([...new Set(platform.choices.map(choice => choice.section))], ['Claude', 'GPT', '国模']);
  const html = renderToStaticMarkup(React.createElement(ModelPicker, { id: 'home-model', list: LIST, value: 'platform', onChange() {} }));
  assert.deepEqual([...html.matchAll(/class="tw-picker-section"[^>]*>([^<]+)</g)].map(match => match[1]), ['Claude', 'GPT', '国模']);
  assert.equal((html.match(/role="option"/g) || []).length, 12);
  assert.doesNotMatch((html.match(/>[^<]*</g) || []).join(''), /claude-|gpt-|deepseek-|qwen|glm-|platform:/);
});

test('every listed model is selectable and a pick survives reload', () => {
  assert.equal(PRODUCT_DEFAULT_MODEL, 'claude-opus-5-5');
  assert.equal(defaultModelChoice(LIST), 'platform');
  for (const item of TRI.slice(1)) assert.equal(defaultModelChoice(LIST, `platform:${item.id}`), `platform:${item.id}`, item.id);
  // a stored id that is no longer offered falls back to the default
  assert.equal(defaultModelChoice(LIST, 'platform:openai:gpt-4o'), 'platform');
});
