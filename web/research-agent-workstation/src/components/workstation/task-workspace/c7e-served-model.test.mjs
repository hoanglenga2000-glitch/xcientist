// t238 C7e-FE-BTN F-2: the model line under the title shows a registered display name only.
//   `deepseek/deepseek-v4.1-flash` (the id the gateway serves) -> 「DeepSeek 4.1 Flash」; an id with no registered name hides the line
//   and the id (raw, title-cased or split) is visible nowhere in the UI.
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
const { productModelLabel, servedModelLabel } = require('./task-presentation.ts');
const { MODEL_DISPLAY_NAMES } = require('./model-display-names.ts');
const { TaskRunView } = require('./TaskRunView.tsx');
const { modelGroups } = require('./ComposerPicker.tsx');
const { fallbackChoice, fallbackGroups } = require('./model-failure.ts');
const FIXTURES = new URL('../../../../../../', import.meta.url);

const render = (view, provider, model) => renderToStaticMarkup(React.createElement(TaskRunView, { view, run: {
  id: 'run_f2', status: 'completed', prompt: '测试正文', answer: '回复', model_provider: provider, model,
  message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
    { id: 'u', role: 'user', content: '测试正文' }, { id: 'a', role: 'assistant', content: '回复' }] },
} }));

test('F-2: the served id and the mock id render 「DeepSeek 4.1 Flash」, not "Deepseek/deepseek V4.1 Flash"', () => {
  for (const [provider, model] of [['deepseek', 'deepseek/deepseek-v4.1-flash'], ['', 'deepseek/deepseek-v4.1-flash'], ['deepseek', 'deepseek-v4.1-flash'],
    ['deepseek', 'deepseek-flash'], ['openai', 'DeepSeek/DeepSeek-V4.1-Flash'], ['deepseek', 'deepseek_v4_flash']]) {
    const html = render('conversation', provider, model);
    const expected = model.endsWith('v4_flash') ? 'DeepSeek 4 Flash' : 'DeepSeek 4.1 Flash';
    assert.match(html, new RegExp(`<p class="tw-served-model"[^>]*>${expected}</p>`), `${provider}|${model}`);
    assert.doesNotMatch(html.replace(/data-served-model="[^"]*"/g, ''), /Deepseek|deepseek\s*\/|V4\.1/, `${provider}|${model}`);
    assert.match(render('progress', provider, model), new RegExp(`<dt>实际模型</dt><dd>${expected}</dd>`));
  }
  assert.equal(servedModelLabel('deepseek', 'deepseek/deepseek-v4.1-flash'), 'DeepSeek 4.1 Flash');
  assert.equal(productModelLabel('deepseek-flash'), 'DeepSeek 4.1 Flash');
});

test('F-2: existing mappings are unchanged (shared fixture) and every lt4net catalog id + alias has a registered name', () => {
  const table = JSON.parse(readFileSync(new URL('tests/fixtures/model_display_names.json', FIXTURES), 'utf8'));
  for (const [id, label] of table.cases) assert.equal(productModelLabel(id), label, id);
  const catalog = JSON.parse(readFileSync(new URL('src/evomind_runtime/lt4net_models.json', FIXTURES), 'utf8'));
  for (const id of [...catalog.models, ...Object.keys(catalog.aliases)]) assert.ok(productModelLabel(id), `unregistered: ${id}`);
  for (const [id, label] of Object.entries(MODEL_DISPLAY_NAMES)) { assert.equal(id, id.toLowerCase()); assert.ok(label && label !== id, id); }
});

test('F-2 (HARD): an id with no registered name hides the model line; the raw, title-cased or split id is visible nowhere', () => {
  for (const [provider, model] of [['acme', 'acme/secret-model-9'], ['', 'zeta-quark_7b'], ['deepseek', 'deepseek/deepseek-v9-ultra'], ['x', 'unknown']]) {
    assert.equal(productModelLabel(model), '', model);
    assert.equal(servedModelLabel(provider, model), '', model);
    for (const view of ['conversation', 'progress', 'results']) {
      const html = render(view, provider, model);
      assert.doesNotMatch(html, /tw-served-model/, `${view}|${model}`);
      assert.doesNotMatch(html, /data-served-model/, `${view}|${model}`);
      // not the raw id, not a provider/model pair, not any title-cased or split derivation of it
      const words = model.split(/[/\-_]+/).filter(word => word.length > 2 && !/^\d/.test(word));
      for (const word of [model, ...words, ...words.map(word => word[0].toUpperCase() + word.slice(1))])
        assert.ok(!html.includes(word), `${view}|${model}: "${word}" is visible`);
    }
  }
  assert.match(render('progress', 'acme', 'acme/secret-model-9'), /<dt>实际模型<\/dt><dd>平台模型<\/dd>/);
  assert.match(render('progress', '', ''), /尚无记录/);
});

test('F-2 (HARD): the picker never shows an unregistered id either (generic 「平台模型」 label)', () => {
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [],
    platform_models: [{ id: 'acme:secret-model-9', provider: 'acme', model: 'secret-model-9', label: '' }] };
  const labels = modelGroups(list).flatMap(group => group.choices.map(choice => choice.label));
  assert.ok(labels.length && labels.every(label => !/secret|Secret|model-9|Model 9/.test(label)), labels.join('|'));
});

test('F-2 (HARD): a server-supplied label for an unregistered model is never shown (failure card + 备选模型 list)', () => {
  const list = { platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' }, default_profile_id: '', profiles: [],
    platform_models: [
      { id: 'acme:secret-model-9', provider: 'acme', model: 'secret-model-9', label: 'Secret Model 9' },
      { id: 'deepseek:deepseek-flash', provider: 'deepseek', model: 'deepseek-flash', label: 'Deepseek Flash (server)' }] };
  const run = (fallback) => ({ model_error: { explicit: true, platform_model_id: 'anthropic:claude-sonnet-5-5' }, fallback_model_id: fallback, model_profile: 'x' });
  assert.equal(fallbackChoice(list, run('acme:secret-model-9')).label, '备选模型');
  assert.equal(fallbackChoice(list, run('deepseek:deepseek-flash')).label, 'DeepSeek 4.1 Flash');
  const shown = fallbackGroups(list, 'platform:deepseek:deepseek-flash')?.flatMap(group => group.choices.map(choice => choice.label)) ?? [];
  const all = [...shown, ...(fallbackGroups(list, 'platform:other')?.flatMap(group => group.choices.map(choice => choice.label)) ?? [])].join('|');
  assert.ok(!/Secret|secret|model-9|\(server\)/.test(all), all);
  assert.ok(all.includes('备选模型') && all.includes('DeepSeek 4.1 Flash'), all);
});
