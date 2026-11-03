import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// t237 C6d-pin: H_c_pin_more_no_overlap — every memory row's pin is hittable in the first 393×852 fold; softs (AbortSignal.any, brand aria).
const require = createRequire(import.meta.url);
const app = path => fileURLToPath(new URL('../../../' + path, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const source = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const css = source('./workspace.css');
const rule = selector => { const start = css.indexOf(selector + '{'); return start < 0 ? '' : css.slice(start, css.indexOf('}', start) + 1); };
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const ENGLISH_UI = /\b(?:Pin|Unpin|More|Delete|Edit|Memory|Search|Preference|Project|Fact|Other)\b/;
const visible = html => [...html.replace(/<[^>]+>/g, ' ').split(/\s+/), ...[...html.matchAll(/(?:aria-label|title|placeholder)="([^"]*)"/g)].map(match => match[1])].join(' ');
const NOW = Date.parse('2026-10-30T12:00:00Z');
const mem = (i, text, kind, pinned = false) => ({ id: 'memory_' + String(i).padStart(32, '0'), text, kind, pinned, scope: 'user', project_id: null, source_task_id: null, source_task_title: null,
  created_at: new Date(NOW - 86400000 * (i + 1)).toISOString(), updated_at: new Date(NOW).toISOString(), last_used_at: null, use_count: i });
/** The VERIFY forge (c6c_lib.forge_rows): 7 rows, one long enough to clamp at two lines. */
const SEVEN = [mem(1, '喜欢简洁直接的回答，少用客套话', 'preference', true), mem(2, '我是做医学影像的研究生', 'profile'),
  mem(3, '当前在做胸片分类，数据在内部集群上，训练用三折交叉验证并固定随机种子，需要保留每一折的混淆矩阵和曲线图', 'project'),
  mem(4, '实验室周三下午开组会', 'fact'), mem(5, '默认用中文回复', 'preference'), mem(6, '偶尔提到的零散信息', 'other'), mem(7, '不要在回答里出现英文缩写解释', 'preference', true)];
const memoryView = () => renderToStaticMarkup(h(require('./MemorySettings.tsx').MemoryView, { enabled: true, memories: SEVEN, now: NOW, onToggle() {}, onClearAll() {} }));

/** The ≤760 compact block this fix owns (first rule is the toggle). */
const COMPACT_HEAD = '@media(max-width:760px){.task-workspace .tw-memory-toggle{';
const compactBlock = () => {
  const start = css.indexOf(COMPACT_HEAD);
  if (start < 0) return '';
  let depth = 0;
  for (let i = start; i < css.length; i++) {
    if (css[i] === '{') depth++;
    else if (css[i] === '}' && --depth === 0) return css.slice(start, i + 1);
  }
  return '';
};
const decl = (block, selector, prop) => {
  const start = block.indexOf(selector + '{');
  if (start < 0) return null;
  const body = block.slice(start + selector.length + 1, block.indexOf('}', start));
  const match = body.match(new RegExp(`(?:^|;)${prop}:([^;]+)`));
  return match ? match[1].trim() : null;
};
const px = value => Number(String(value).match(/^(-?\d+(?:\.\d+)?)px/)?.[1] ?? NaN);
/** `padding:A B` / `padding:A` → vertical px. */
const padY = value => px(String(value).split(/\s+/)[0]);

// ---- HARD: H_c_pin_more_no_overlap ----
test('C6d-pin: 7 memory rows render 7 pins + 7 ⋯ in .tw-memory-side (pin then ⋯), kind pill outside the side, Chinese only', () => {
  const html = memoryView();
  const rows = html.match(/<li [^>]*class="tw-memory-row[^"]*"[^>]*>[^]*?<\/li>/g) ?? [];
  assert.equal(rows.length, 7);
  for (const row of rows) {
    const side = row.match(/<div class="tw-memory-side">([^]*?)<\/div><\/li>$/);
    assert.ok(side, 'side is the last child of the row');
    // side = pin + ⋯ only; nothing else competes for the 393 width.
    assert.match(side[1], /^<button type="button" class="tw-memory-pin"[^>]*>[^]*?<\/button><button type="button" class="tw-row-more"[^>]*>[^]*?<\/button>$/);
    assert.doesNotMatch(side[1], /tw-memory-kind/);
    // kind pill lives in the text column, after the source line.
    assert.match(row, /<div class="tw-memory-body"><p class="tw-memory-text">[^<]*<\/p><p class="tw-memory-source">[^]*?<\/p><span class="tw-memory-kind">[^<]+<\/span><\/div>/);
  }
  assert.equal((html.match(/class="tw-memory-pin"/g) ?? []).length, 7);
  assert.equal((html.match(/class="tw-row-more"/g) ?? []).length, 7);
  for (const label of ['偏好', '关于我', '项目', '事实', '其他']) assert.match(html, new RegExp(`<span class="tw-memory-kind">${label}</span>`));
  assert.doesNotMatch(visible(html), ENGLISH_UI);
});


test('C6d-pin Critiquito: pin+⋯ fixed right column — body cannot cover pin center after wrap', () => {
  const body = rule('.task-workspace .tw-memory-body');
  assert.match(body, /flex:\s*1/);
  assert.match(body, /min-width:\s*0/);
  const side = rule('.task-workspace .tw-memory-side');
  assert.match(side, /display:\s*flex/);
  assert.match(side, /flex:\s*none/);
  assert.match(side, /flex-shrink:\s*0/);
  assert.match(side, /position:\s*relative/);
  assert.match(side, /z-index:\s*[1-9]/);
  assert.match(side, /overflow:\s*visible/);
  assert.match(side, /margin-left:\s*auto/);
  const pin = rule('.task-workspace .tw-memory-pin');
  assert.match(pin, /position:\s*relative/);
  assert.match(pin, /z-index:\s*[1-9]/);
  // more stays in-flow beside pin (no absolute overlay onto pin center)
  const more = rule('.task-workspace .tw-memory-side .tw-row-more');
  assert.match(more, /position:\s*static/);
  assert.match(more, /transform:\s*none/);
});

test('C6d-pin: kind pill sits at the end of the source line (body grid), not on a new line', () => {
  const body = rule('.task-workspace .tw-memory-body');
  assert.match(body, /display:grid/);
  assert.match(body, /grid-template-columns:minmax\(0,auto\) 1fr/);
  assert.match(rule('.task-workspace .tw-memory-body>.tw-memory-text'), /grid-column:1\/-1/);
  const kind = rule('.task-workspace .tw-memory-body>.tw-memory-kind');
  assert.match(kind, /justify-self:start/);
  assert.match(kind, /align-self:end/);
});

test('C6d-pin: ⋯ stays in-flow beside the pin; nothing clips them; 44×44 at ≤760', () => {
  const fix = rule('.task-workspace .tw-memory-side .tw-row-more');
  assert.match(fix, /position:static/); assert.match(fix, /transform:none/); assert.match(fix, /flex:none/);
  assert.doesNotMatch(css, /\.tw-memory-(?:side|row)[^{]*\.tw-row-more\{[^}]*position:(?:absolute|fixed)/);
  assert.match(rule('.task-workspace .tw-memory-side'), /display:flex/);
  for (const selector of ['.tw-memory-side', '.tw-memory-row', '.tw-memory-list']) {
    for (const block of css.matchAll(new RegExp(`\\.task-workspace \\${selector}\\{([^}]*)\\}`, 'g')))
      assert.doesNotMatch(block[1], /overflow(?:-[xy])?:(?:hidden|clip)/, selector);
  }
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-memory-pin\{width:44px;height:44px\}/);
  assert.match(css, /@media\(max-width:760px\)\{[^]*\.task-workspace \.tw-row-more\{width:44px;height:44px/);
  assert.match(rule('.task-workspace .tw-memory-side'), /flex:none/);
});

test('C6d-pin: ≤760 height budget — chrome + 7 VERIFY rows put the last pin centre inside the 393×852 first fold', () => {
  const block = compactBlock();
  assert.ok(block, 'compact ≤760 memory block exists');
  // Compact numbers the fix relies on (re-inflating any of them fails here).
  const toggleMin = px(decl(block, '.task-workspace .tw-memory-toggle', 'min-height'));
  const togglePad = padY(decl(block, '.task-workspace .tw-memory-toggle', 'padding'));
  const rowPad = padY(decl(block, '.task-workspace .tw-memory-row', 'padding'));
  const sourceGap = px(decl(block, '.task-workspace .tw-memory-source', 'margin-top'));
  const listGap = px(decl(block, '.task-workspace .tw-memory-list', 'gap'));
  const backGap = px(decl(block, '.tw-settings>.tw-back', 'margin-bottom'));
  const titleGap = px(decl(block, '.tw-settings>h1', 'margin-bottom'));
  assert.equal(toggleMin, 44, 'toggle row keeps a 44px hit row');
  assert.ok(togglePad <= 8, `toggle padding ${togglePad} ≤ 8`);
  assert.ok(rowPad <= 4, `row padding ${rowPad} ≤ 4`);
  assert.ok(sourceGap <= 2, `source gap ${sourceGap} ≤ 2`);
  assert.ok(listGap <= 2, `list gap ${listGap} ≤ 2`);
  assert.ok(backGap <= 8 && titleGap <= 8, 'settings chrome margins ≤ 8');
  assert.equal(decl(block, '.task-workspace .tw-memory-row', 'align-items'), 'center');
  // Row/pin geometry: 44px pin fits inside the compact row without clipping.
  const LINE = 22, SOURCE_LINE = 20, PIN = 44;
  const rowH = lines => Math.max(44, 2 * rowPad + lines * LINE + sourceGap + SOURCE_LINE);
  assert.ok(rowH(1) >= PIN, 'row min-height holds the 44px pin');
  // Fixed 393 chrome (Chrome-measured on the VERIFY forge; matches c6d_dom_evidence within 1px):
  // topbar + main top + back line + h1 line = 134, settings nav 37, shell gap 12, memory title 26 + 4.
  const CHROME = 134 + backGap + titleGap + 37 + 12 + 30;
  const toggle = Math.max(toggleMin, 2 * togglePad + LINE + 1);
  const TOOLS = 8 + 44 + 8 + 45 + 4; // margin, search, gap, tabs (44 + 1px border), margin
  // VERIFY rows: with the kind pill out of the side every row is one line except the 54-char one (clamped to 2).
  const heights = [1, 1, 2, 1, 1, 1, 1].map(rowH);
  const lastTop = CHROME + 2 * toggle + TOOLS + heights.slice(0, -1).reduce((a, b) => a + b + listGap, 0);
  const lastPinCentre = lastTop + heights.at(-1) / 2;
  assert.ok(lastPinCentre + PIN / 2 <= 852 - 24, `last pin bottom ${lastPinCentre + PIN / 2} leaves ≥24px of the 852 fold`);
});

test('C6d-pin: compact rules come after the base memory rules (cascade wins) and stay scoped to ≤760', () => {
  const at = css.indexOf(COMPACT_HEAD);
  assert.ok(at > css.indexOf('.task-workspace .tw-memory-row{position:relative'), 'after base row');
  assert.ok(at > css.indexOf('.task-workspace .tw-memory-toggle{display:flex'), 'after base toggle');
  assert.ok(at > css.indexOf('.tw-settings-title,.tw-settings>h1{'), 'after base settings title');
  // Desktop keeps the roomy rhythm.
  assert.match(rule('.task-workspace .tw-memory-toggle'), /padding:16px 0 16px/);
  assert.match(rule('.task-workspace .tw-memory-row'), /padding:10px 8px/);
});

// ---- SOFT s_src_abort_signal_any ----
test('C6d-pin soft: AbortSignal.any is spelled only in abort-signal.ts; types.ts imports the helper', async () => {
  const types = source('./types.ts');
  assert.doesNotMatch(types, /AbortSignal\.any\(/);
  assert.doesNotMatch(types, /export function timeoutSignal/);
  assert.match(types, /import \{ timeoutSignal \} from '\.\/abort-signal\.ts';/);
  assert.match(types, /timeoutSignal\(15000/);
  const helper = source('./abort-signal.ts');
  assert.match(helper, /export function timeoutSignal\(ms: number, parent\?: AbortSignal\): AbortSignal/);
  assert.match(helper, /typeof AbortSignal\.any === 'function'/);
  assert.doesNotMatch(helper, /from '\.\/types/);
  // Native strip-types (task-request.test.ts path) still loads types.ts → abort-signal.ts.
  const native = await import('./types.ts');
  assert.equal(typeof native.taskRequest, 'function');
  const any = AbortSignal.any;
  try {
    delete AbortSignal.any;
    const { timeoutSignal } = await import('./abort-signal.ts');
    const parent = new AbortController();
    const chained = timeoutSignal(60000, parent.signal);
    parent.abort();
    assert.equal(chained.aborted, true);
  } finally { AbortSignal.any = any; }
});

// ---- SOFT s_f_brand_aria_once ----
test('C6d-pin soft: the workspace brand link is announced once (lockup inside is decorative)', () => {
  const { DeepEvoLockup } = require('../DeepEvoBrand.tsx');
  const silent = renderToStaticMarkup(h(DeepEvoLockup, { width: 168, decorative: true }));
  assert.match(silent, /(?:^|(?:<link[^>]*>)*)<span aria-hidden="true" data-ui-deepevo-lockup(?:="[^"]*")? class="deepevo-lockup"/);
  assert.doesNotMatch(silent, /aria-label|role="img"/);
  // Standalone lockup still names itself.
  assert.match(renderToStaticMarkup(h(DeepEvoLockup, { width: 168 })), /role="img" aria-label="DeepEvo 深度进化"/);
  const workspace = source('./TaskWorkspace.tsx');
  const link = workspace.match(/<a className="tw-brand"[^]*?<\/a>/)?.[0] ?? '';
  assert.match(link, /aria-label="DeepEvo 深度进化"/);
  assert.match(link, /<DeepEvoLockup width=\{168\} className="tw-brand-lockup" decorative \/>/);
  assert.equal((link.match(/DeepEvo 深度进化/g) ?? []).length, 1);
  // The mark inside stays aria-hidden (no label prop).
  assert.match(link, /<DeepEvoMark size=\{32\} className="tw-brand-mark" \/>/);
});
