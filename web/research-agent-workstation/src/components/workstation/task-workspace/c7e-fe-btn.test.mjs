// t238 C7e-FE-BTN: every action link/button inside an error / notice / empty-state card must be readable (computed-style
// contrast >= 4.5:1 and text colour != background) in the light and the dark theme. Root cause of the original defect:
// `.task-workspace .tw-model-error a{color:var(--tw-accent)}` (0,2,1) beat `.task-workspace .tw-primary` (0,2,0), so the
// 「换个模型继续」 label was accent-green on an accent-green fill. A real browser (headless Chrome via playwright) resolves the cascade.
//   Needs: playwright (PLAYWRIGHT_REQUIRE_DIR, default /workspace/login-mocks/build/) + Chrome (CHROME_BIN, default /usr/bin/google-chrome).
//   Without them the browser tests SKIP, unless EVOMIND_REQUIRE_BROWSER_TESTS=1 (then they fail).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { existsSync, readFileSync, appendFileSync } from 'node:fs';
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
const CSS = readFileSync(new URL('./workspace.css', import.meta.url), 'utf8');

const blocked = () => ({
  id: 'run_x', user_task_id: 'utask_x', status: 'blocked', error_class: 'recoverable',
  answer: 'Runtime session is durable. Model execution is unavailable: LLMError/model_empty_response.',
  model_error: { kind: 'unavailable', code: 'model_empty_response', explicit: false },
});
const real = {
  'model-error (换个模型继续 link)': () => renderToStaticMarkup(h(require('./TaskRunView.tsx').TaskRunView, { view: 'conversation', run: blocked(), fallback: null })),
  'model-error (换到备选模型 button)': () => renderToStaticMarkup(h(require('./TaskRunView.tsx').TaskRunView, { view: 'conversation', run: blocked(), fallback: { label: 'DeepSeek 4.1 Flash', onSwitch() {} } })),
  'ErrorNotice 重新登录': () => renderToStaticMarkup(h(require('./ErrorNotice.tsx').ErrorNotice, { error: { status: 401 } })),
  'ErrorNotice 返回': () => renderToStaticMarkup(h(require('./ErrorNotice.tsx').ErrorNotice, { error: { status: 404 } })),
  'ErrorNotice 重试 (button)': () => renderToStaticMarkup(h(require('./ErrorNotice.tsx').ErrorNotice, { error: { status: 500 } })),
  'ProjectFilesNotice total (button)': () => renderToStaticMarkup(h(require('./ProjectFilesNotice.tsx').ProjectFilesNotice, { notice: { kind: 'total', files: [] } })),
  'ProjectFilesNotice partial link': () => renderToStaticMarkup(h(require('./ProjectFilesNotice.tsx').ProjectFilesNotice, { notice: { kind: 'partial', files: [{ name: 'a.csv', error_class: 'x' }] }, projectId: 'p1' })),
  'composer chip-error 管理文件': () => renderToStaticMarkup(h(require('./ComposerAttach.tsx').AttachmentChips, { files: [], error: '存储空间已用完。删掉不用的文件后再上传。', errorCode: 'quota_exceeded', onRemove() {} })),
};
// Markup the components render only behind async/hooks state: same class chains as in the .tsx (line refs in the audit list of HANDOFF).
const synthetic = {
  'stuck-banner 停止/再试一次': '<div class="tw-stuck-banner" role="status"><span>x</span><span class="tw-stuck-actions"><button type="button" class="tw-text-button">停止</button><button type="button" class="tw-text-button">再试一次</button></span></div>',
  'TaskList empty a.tw-primary': '<div class="tw-empty"><h2>还没有任务</h2><p>先描述</p><a class="tw-primary" href="/x">新对话</a></div>',
  'ExperiencePlan a.tw-primary': '<div class="tw-experience"><p class="tw-experience-line">x</p><a class="tw-primary" href="/x">开始</a></div>',
  'ExperiencePlan empty a.tw-button': '<div class="tw-empty-hint"><p>x</p><a class="tw-button" href="/x">新对话</a></div>',
  'MemorySettings empty a.tw-button': '<div class="tw-empty"><p>x</p><a class="tw-button" href="/x">返回</a></div>',
  'WorkspaceRecords empty a.tw-button': '<section><p>还没有运行记录。<a class="tw-button" href="/x">新对话</a></p></section>',
  'ProjectPage empty a.tw-button': '<section class="tw-project"><p class="tw-empty-hint">x</p><a class="tw-button" href="/x">新对话</a></section>',
  'ProjectPage header a.tw-primary': '<section class="tw-project"><div><a class="tw-primary tw-project-new" href="/x"><span class="tw-project-new-label">新对话</span></a></div></section>',
  'ProjectPage notice a.tw-button': '<section class="tw-project"><div class="tw-error"><p>x</p><a class="tw-button tw-nowrap" href="/x">管理文件</a></div></section>',
  'picker footer a.tw-primary': '<div class="tw-picker-footer"><a class="tw-primary" href="/x">管理模型</a></div>',
  'memory source a.tw-primary': '<p class="tw-memory-source"><a class="tw-primary" href="/x">来源</a></p>',
  'DownloadFile error + a.tw-button': '<div><a class="tw-button" href="/x">下载</a><p role="alert" class="tw-error">下载未完成</p></div>',
};
const ITEMS = [
  ...Object.entries(real).map(([name, render]) => ({ name, html: render() })),
  ...Object.entries(synthetic).map(([name, html]) => ({ name, html })),
];

let chromium = null;
try { chromium = createRequire(process.env.PLAYWRIGHT_REQUIRE_DIR || '/workspace/login-mocks/build/')('playwright').chromium; } catch { /* skipped below */ }
const CHROME = process.env.CHROME_BIN || '/usr/bin/google-chrome';
const REQUIRED = process.env.EVOMIND_REQUIRE_BROWSER_TESTS === '1';
const skip = !REQUIRED && (!chromium || !existsSync(CHROME)) ? 'playwright/Chrome unavailable (set EVOMIND_REQUIRE_BROWSER_TESTS=1 to require)' : false;

// Runs in the page: the visible text colour / effective background (alpha-composited down the ancestor chain) and WCAG ratio of every
// action element (a.*, button) in the fixture.
const MEASURE = () => {
  const parse = value => { const m = value.match(/rgba?\(([^)]+)\)/); if (!m) return [0, 0, 0, 0]; const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const over = (top, bottom) => { const a = top[3] + bottom[3] * (1 - top[3]); return a === 0 ? [0, 0, 0, 0] : [0, 1, 2].map(i => (top[i] * top[3] + bottom[i] * bottom[3] * (1 - top[3])) / a).concat(a); };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
  const effectiveBackground = el => {
    const chain = []; for (let n = el; n; n = n.parentElement) chain.push(parse(getComputedStyle(n).backgroundColor));
    let out = [255, 255, 255, 1]; for (let i = chain.length - 1; i >= 0; i--) out = over(chain[i], out); return out;
  };
  const measure = el => {
    const bg = effectiveBackground(el); const fg = over(parse(getComputedStyle(el).color), bg);
    const [hi, lo] = [lum(fg), lum(bg)].sort((a, b) => b - a);
    return { text: el.textContent.trim().slice(0, 24), tag: el.tagName.toLowerCase(), cls: el.className, color: getComputedStyle(el).color, background: `rgb(${bg.slice(0, 3).map(Math.round).join(', ')})`, ratio: Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100, same: [0, 1, 2].every(i => Math.round(fg[i]) === Math.round(bg[i])) };
  };
  return [...document.querySelectorAll('#fx a, #fx button')].map(measure);
};

async function withPage(fn) {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  try { const page = await browser.newPage({ viewport: { width: 1200, height: 800 } }); await fn(page); } finally { await browser.close(); }
}
const setup = async (page, html, theme) => {
  await page.setContent(`<!doctype html><html${theme === 'dark' ? ' data-theme="dark"' : ''}><head><meta charset="utf-8"><style>${CSS}</style></head><body><div class="task-workspace"><main id="fx">${html}</main></div></body></html>`);
};

for (const theme of ['light', 'dark']) {
  test(`t238 C7e-FE-BTN (${theme}): every action in the error / notice / empty cards has text colour != background and contrast >= 4.5:1`, { skip }, async () => {
    const failures = [];
    await withPage(async page => {
      for (const item of ITEMS) {
        await setup(page, item.html, theme);
        const rows = await page.evaluate(MEASURE);
        if (!rows.length) failures.push(`${item.name}: no action element rendered (fixture drifted)`);
        if (process.env.C7E_FE_BTN_DUMP) appendFileSync(process.env.C7E_FE_BTN_DUMP, rows.map(row => JSON.stringify({ theme, item: item.name, ...row })).join('\n') + '\n');
        for (const row of rows) if (row.same || row.ratio < 4.5) failures.push(`${item.name} <${row.tag} class="${row.cls}"> 「${row.text}」 ${row.color} on ${row.background} = ${row.ratio}:1`);
      }
    });
    assert.deepEqual(failures, [], failures.join('\n'));
  });

  test(`t238 C7e-FE-BTN (${theme}): 「换个模型继续」 stays readable on hover and keyboard focus`, { skip }, async () => {
    await withPage(async page => {
      await setup(page, ITEMS.find(item => item.name.startsWith('model-error (换个模型继续')).html, theme);
      const link = page.locator('#fx a.tw-primary');
      const assertReadable = async state => {
        const [row] = (await page.evaluate(MEASURE)).filter(r => r.tag === 'a');
        assert.ok(!row.same && row.ratio >= 4.5, `${theme}/${state}: ${row.color} on ${row.background} = ${row.ratio}:1`);
      };
      await link.hover(); await page.waitForTimeout(250); await assertReadable('hover');
      await setup(page, ITEMS.find(item => item.name.startsWith('model-error (换个模型继续')).html, theme); // fresh page: Tab starts at the top
      for (let i = 0; i < 6 && await page.evaluate(() => document.activeElement?.tagName !== 'A'); i++) await page.keyboard.press('Tab');
      await page.waitForTimeout(250);
      const active = await page.evaluate(() => document.activeElement?.outerHTML.slice(0, 120)); assert.match(active, /^<a /, active);
      await assertReadable('focus-visible');
    });
  });
}

for (const theme of ['light', 'dark']) {
  test(`t238 C7e-FE-BTN (${theme}): an a.tw-primary inside the picker footer / memory source stays readable on hover and keyboard focus`, { skip }, async () => {
    await withPage(async page => {
      for (const name of ['picker footer a.tw-primary', 'memory source a.tw-primary']) {
        const html = ITEMS.find(item => item.name === name).html;
        const readable = async state => {
          const [row] = await page.evaluate(MEASURE);
          assert.ok(!row.same && row.ratio >= 4.5, `${name} ${theme}/${state}: ${row.color} on ${row.background} = ${row.ratio}:1`);
        };
        await setup(page, html, theme); await page.locator('#fx a').hover(); await page.waitForTimeout(250); await readable('hover');
        await setup(page, html, theme); await page.keyboard.press('Tab'); await page.waitForTimeout(250);
        assert.equal(await page.evaluate(() => document.activeElement?.tagName), 'A'); await readable('focus-visible');
      }
    });
  });
}

test('t238 C7e-FE-BTN: the picker-footer and memory-source a.tw-primary overrides cover :visited/:hover/:focus-visible', () => {
  for (const card of ['tw-picker-footer', 'tw-memory-source']) for (const state of ['', ':visited', ':hover', ':focus-visible'])
    assert.match(CSS, new RegExp(String.raw`\.task-workspace \.${card} a\.tw-primary${state.replace(/[:()-]/g, '\\$&')}[,{][^}]*color:var\(--tw-on-accent\)`), `${card} a.tw-primary${state}`);
  assert.match(CSS, /\.tw-picker-footer a\.tw-primary:hover,[^{]*\{background:var\(--tw-accent\)/);
});

test('t238 C7e-FE-BTN: the override covers :visited/:hover/:focus-visible and the global accent tokens are untouched', () => {
  for (const state of ['', ':hover', ':focus-visible', ':visited'])
    assert.match(CSS, new RegExp(String.raw`\.task-workspace \.tw-model-error a\.tw-primary${state.replace(/[:()-]/g, '\\$&')}[,{][^}]*color:var\(--tw-on-accent\)`), `a.tw-primary${state}`);
  assert.match(CSS, /--tw-accent:#2F855A;--tw-accent-hover:#276749;/);
  assert.match(CSS, /--tw-accent:#4FB37E;--tw-accent-hover:#63C08F;/);
  assert.match(CSS, /--tw-on-accent:#0E1A13;/);
});

test('t238 C7e-FE-BTN: no card-scoped bare `<card> a{color}` rule can out-rank a filled .tw-primary/.tw-button action', () => {
  const CARDS = /\.(?:tw-error|tw-model-error|tw-empty|tw-empty-hint|tw-chip-error|tw-stuck-banner|tw-project-files-notice|tw-fail-actions)\b/;
  const hits = [];
  for (const [, selectors, body] of CSS.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (!/(^|[;\s])color:/.test(body)) continue;
    for (const selector of selectors.replace(/\/\*[\s\S]*?\*\//g, '').split(',').map(part => part.trim())) {
      if (CARDS.test(selector) && /\sa(?::[\w-]+)?$/.test(selector)) hits.push(selector);
    }
  }
  // The one bare rule (plain links inside the model-error card) is only allowed together with the a.tw-primary override asserted above.
  assert.deepEqual([...new Set(hits)].sort(), ['.task-workspace .tw-model-error a', '.task-workspace .tw-model-error a:hover'].sort());
});
