// t238 C7e A1/A2: actionable Chinese copy for explicit-model failures, 「换到备选模型」 (hidden models never offered),
// codes only under 技术详情, and the visible 「已切换到备选模型「X」」 notice.
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
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const mf = () => require('./model-failure.ts');
const CODES = /http_\d{3}|LLMError|model_[a-z_]+|rate_limited|HTTP \d{3}|\b(?:429|503|401|403)\b/;

// The BE already drops PICKER_HIDDEN_MODEL_IDS (deepseek:deepseek-chat, ...) from platform_models.
const LIST = {
  profiles: [], default_profile_id: '',
  platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' },
  platform_models: [
    { id: 'anthropic:claude-sonnet-5-5', provider: 'anthropic', model: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' },
    { id: 'anthropic:claude-opus-5-5', provider: 'anthropic', model: 'claude-opus-5-5', label: 'Claude Opus 5.5' },
    { id: 'deepseek:deepseek-flash', provider: 'deepseek', model: 'deepseek-flash', label: 'DeepSeek Flash' },
  ],
};
const blocked = (kind, code, extra = {}) => ({
  id: 'run_x', user_task_id: 'utask_x', status: 'blocked', error_class: 'recoverable',
  answer: `Runtime session is durable. Model execution is unavailable: LLMError/${code}.`,
  model_error: { kind, code, explicit: true, platform_model_id: 'anthropic:claude-opus-5-5' }, ...extra,
});

test('t238 C7e A2: one short Chinese line per failure kind, with a next action and no codes', () => {
  const { MODEL_ERROR_COPY, explicitModelErrorLine } = mf();
  for (const [kind, line] of Object.entries(MODEL_ERROR_COPY)) {
    assert.match(line, /[一-鿿]/); assert.doesNotMatch(line, CODES); assert.doesNotMatch(line, /您/);
    assert.ok(line.length <= 40, kind);
  }
  assert.match(MODEL_ERROR_COPY.rate_limited, /请求太多.*稍等/);
  assert.match(MODEL_ERROR_COPY.unavailable, /暂时连不上/);
  assert.match(MODEL_ERROR_COPY.timeout, /响应太慢/);
  assert.match(MODEL_ERROR_COPY.auth, /检查模型配置/);
  assert.equal(explicitModelErrorLine(blocked('rate_limited', 'http_429')), MODEL_ERROR_COPY.rate_limited);
  assert.equal(explicitModelErrorLine({ model_error: { kind: 'unavailable', code: 'http_503', explicit: false } }), null);
  assert.equal(explicitModelErrorLine({ model_error: { kind: 'bogus', code: 'x', explicit: true } }), null);
});

test('t238 C7e A2: 「换到备选模型」 offers the run\'s own fallback, else the default, never the failed or a hidden model', () => {
  const { fallbackChoice } = mf();
  assert.deepEqual(fallbackChoice(LIST, blocked('unavailable', 'http_503', { fallback_model_id: 'deepseek:deepseek-flash' })),
    { value: 'platform:deepseek:deepseek-flash', label: 'DeepSeek 4.1 Flash' });
  assert.deepEqual(fallbackChoice(LIST, blocked('unavailable', 'http_503')), { value: 'platform', label: 'Claude Sonnet 5.5' });
  // A hidden fallback id is not listed, so it is never offered; the default is offered instead.
  assert.deepEqual(fallbackChoice(LIST, blocked('unavailable', 'http_503', { fallback_model_id: 'deepseek:deepseek-chat' })),
    { value: 'platform', label: 'Claude Sonnet 5.5' });
  const onlyFailed = { ...LIST, platform_model: null, platform_models: [LIST.platform_models[1]] };
  assert.equal(fallbackChoice(onlyFailed, blocked('unavailable', 'http_503')), null);
  assert.equal(fallbackChoice(LIST, { model_error: { kind: 'unavailable', code: 'http_503', explicit: false } }), null);
});

test('t238 C7e: an unlabelled alternate shows a product name, never the raw model id', () => {
  const { fallbackChoice, fallbackGroups } = mf();
  const bare = { ...LIST, platform_models: [], platform_model: { provider: 'anthropic', model: 'claude-sonnet-5-5' } };
  const choice = fallbackChoice(bare, blocked('unavailable', 'http_503'));
  assert.ok(choice && !/claude-sonnet-5-5|anthropic:/.test(choice.label), choice && choice.label);
  const groups = fallbackGroups(bare, 'platform:deepseek:deepseek-flash');
  for (const item of groups?.[0]?.choices ?? []) assert.ok(!/[a-z]+-[a-z0-9]+-[0-9]/.test(item.label) || item.value === '', item.label);
});

test('t238 C7e A2: the fail card shows the line + 换到备选模型「X」 + 再试一次; codes only inside 技术详情', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  for (const [kind, code, line] of [['rate_limited', 'http_429', /请求太多/], ['unavailable', 'http_503', /暂时连不上/],
    ['timeout', 'model_total_deadline_exceeded', /响应太慢/], ['auth', 'http_401', /访问权限/]]) {
    const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: blocked(kind, code),
      fallback: { label: 'Claude Sonnet 5.5', onSwitch() {} } }));
    const card = html.match(/data-testid="tw-model-fail"[\s\S]*?<\/details><\/div>/)?.[0] || '';
    assert.match(card, line);
    assert.match(card, /data-testid="tw-switch-fallback"[^>]*>换到备选模型「Claude Sonnet 5\.5」</);
    assert.match(card, />再试一次</); assert.doesNotMatch(card, /重试此步/);
    const outside = card.replace(/<details[\s\S]*?<\/details>/, '');
    assert.doesNotMatch(outside, CODES, kind);
    assert.match(card.match(/<details[\s\S]*?<\/details>/)[0], new RegExp(code));
  }
  // Without an alternate the card keeps the existing 换个模型继续 link (C6b).
  const plain = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: blocked('unavailable', 'http_503'), fallback: null }));
  assert.match(plain, /换个模型继续/); assert.doesNotMatch(plain, /tw-switch-fallback/);
});

test('t238 C7e A1: the switch notice is visible on the answer and absent otherwise', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const base = { id: 'run_ok', status: 'completed', answer: '你好', message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [{ id: 'a1', role: 'assistant', content: '你好' }] } };
  const switched = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: { ...base, model_failover: { notice: '已切换到备选模型「备用线路 1」', label: '备用线路 1', explicit: false } } }));
  assert.match(switched, /data-testid="tw-failover-notice"[^>]*>已切换到备选模型「备用线路 1」</);
  assert.doesNotMatch(renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run: base })), /tw-failover-notice/);
  assert.equal(mf().failoverNotice({ model_failover: { notice: 'https://evil' } }), null);
});

test('t238 C7e A1: 备选模型 picker only for an explicit pick, never the pick itself, 「不设置」 first', () => {
  const { fallbackGroups, fallbackModelId } = mf();
  assert.equal(fallbackGroups(LIST, 'platform'), null);
  assert.equal(fallbackGroups(LIST, ''), null);
  const groups = fallbackGroups(LIST, 'platform:anthropic:claude-opus-5-5');
  assert.deepEqual(groups[0].choices.map(item => item.value), ['', 'anthropic:claude-sonnet-5-5', 'deepseek:deepseek-flash']);
  assert.equal(fallbackModelId(LIST, 'platform:anthropic:claude-opus-5-5', 'deepseek:deepseek-flash'), 'deepseek:deepseek-flash');
  assert.equal(fallbackModelId(LIST, 'platform:anthropic:claude-opus-5-5', 'anthropic:claude-opus-5-5'), undefined);
  assert.equal(fallbackModelId(LIST, 'platform:anthropic:claude-opus-5-5', 'deepseek:deepseek-chat'), undefined); // hidden
  assert.equal(fallbackModelId(LIST, 'platform', 'deepseek:deepseek-flash'), undefined);
  const source = readFileSync(new URL('./TaskSession.tsx', import.meta.url), 'utf8');
  assert.match(source, /fallback_model_id: fallbackModelId\(models, model, fallbackPick\)/);
  assert.match(source, /备选模型（这个模型连不上时改用它，会提示你）/);
});

// t238 C7e-R1 (Critiquito): an alternate with no registered display name is never shown as 换到备选模型「备选模型」.
test('t238 C7e-R1: unregistered fallback -> primary button is exactly 换个模型继续; registered -> 换到备选模型「<name>」', () => {
  const { TaskRunView } = require('./TaskRunView.tsx');
  const { fallbackChoice } = mf();
  const unregistered = { ...LIST, platform_model: { provider: 'acme', model: 'zz-unlisted-9' },
    platform_models: [{ id: 'acme:zz-unlisted-9', provider: 'acme', model: 'zz-unlisted-9', label: 'Zz Unlisted 9' }] };
  const run = blocked('unavailable', 'http_503');
  // the chooser labels an unregistered alternate with the neutral placeholder (FE tip F-2 contract) ...
  assert.equal(fallbackChoice(unregistered, run)?.label, '备选模型');
  // ... which the card must never turn into 换到备选模型「备选模型」.
  const placeholder = fallbackChoice(unregistered, run);
  assert.ok(placeholder);
  const phHtml = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run, fallback: { label: placeholder.label, onSwitch() {} } }));
  assert.match(phHtml, /<a class="tw-primary" href="[^"]*">换个模型继续<\/a>/);
  assert.doesNotMatch(phHtml, /tw-switch-fallback|换到备选模型|「备选模型」/);
  // no fallback / nameless / placeholder: only the link button.
  for (const fallback of [null, undefined, { label: '', onSwitch() {} }, { label: '备选模型', onSwitch() {} }]) {
    const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run, fallback }));
    const card = html.match(/data-testid="tw-model-fail"[\s\S]*?<\/details><\/div>/)?.[0] || '';
    assert.ok(card);
    assert.match(card, /<a class="tw-primary" href="[^"]*">换个模型继续<\/a>/);
    assert.doesNotMatch(card, /tw-switch-fallback|换到备选模型|「备选模型」/);
    assert.match(card, />再试一次</);
  }
  // registered alternate keeps the existing label
  const choice = fallbackChoice(LIST, run);
  assert.equal(choice?.label, 'Claude Sonnet 5.5');
  const html = renderToStaticMarkup(h(TaskRunView, { view: 'conversation', run, fallback: { label: choice.label, onSwitch() {} } }));
  assert.match(html, /data-testid="tw-switch-fallback"[^>]*>换到备选模型「Claude Sonnet 5\.5」</);
  assert.doesNotMatch(html, /<a class="tw-primary"/);
});
