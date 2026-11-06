// Run supervision v1: the model_unavailable card suggests the failed model's group default, and a run that waits
// for a model slot shows its place in line.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

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

const LIST = {
  profiles: [], default_profile_id: '',
  platform_model: { provider: 'anthropic', model: 'claude-opus-5-5' },
  platform_models: [
    { id: 'anthropic:claude-opus-5-5', provider: 'anthropic', model: 'claude-opus-5-5', label: 'x' },
    { id: 'openai:gpt-6-astra', provider: 'openai', model: 'gpt-6-astra', label: 'x' },
    { id: 'openai:gpt-6-sol', provider: 'openai', model: 'gpt-6-sol', label: 'x' },
    { id: 'deepseek:deepseek-v4.1-flash', provider: 'deepseek', model: 'deepseek-v4.1-flash', label: 'x' },
  ],
};
const blockedOn = (failed, extra = {}) => ({
  id: 'run_x', status: 'blocked', error_class: 'recoverable',
  model_error: { kind: 'unavailable', code: 'empty_model_response', explicit: true, platform_model_id: failed }, ...extra,
});
const card = (target) => ({ kind: 'model_unavailable', options: [{ action: 'resume' }, { action: 'new_run_with_model', platform_model_id: target }] });

test('run supervision v1: the card suggests the same-group default ahead of the product default', () => {
  const { fallbackChoice } = require('./model-failure.ts');
  const choice = fallbackChoice(LIST, blockedOn('openai:gpt-6-sol', { attention: card('openai:gpt-6-astra') }));
  assert.equal(choice.value, 'platform:openai:gpt-6-astra');
  // a suggestion that is the product default maps to the `platform` choice
  assert.equal(fallbackChoice(LIST, blockedOn('openai:gpt-6-sol', { attention: card('anthropic:claude-opus-5-5') })).value, 'platform');
  // the user's own pre-selected fallback still wins
  assert.equal(fallbackChoice(LIST, blockedOn('openai:gpt-6-sol', { attention: card('openai:gpt-6-astra'), fallback_model_id: 'deepseek:deepseek-v4.1-flash' })).value,
    'platform:deepseek:deepseek-v4.1-flash');
  // an unlisted suggestion is ignored (never offer a hidden model)
  assert.equal(fallbackChoice(LIST, blockedOn('openai:gpt-6-sol', { attention: card('openai:gpt-4.1') })).value, 'platform');
});

test('run supervision v1: a queued run names its place in line', () => {
  const { thinkingRowState } = require('./task-presentation.ts');
  assert.equal(thinkingRowState({ id: 'r', status: 'queued', queue_position: 3, activity: [] }).text, '排队中（第 3 位）');
  assert.equal(thinkingRowState({ id: 'r', status: 'queued', activity: [] }).text, '排队中');
});
