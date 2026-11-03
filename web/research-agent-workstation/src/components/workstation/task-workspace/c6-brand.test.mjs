import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const require = createRequire(import.meta.url);
const app = path => fileURLToPath(new URL('../../../' + path, import.meta.url));
const pkg = path => fileURLToPath(new URL('../../../../' + path, import.meta.url));
const repo = path => fileURLToPath(new URL('../../../../../../' + path, import.meta.url));
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
const brandCss = readFileSync(app('components/workstation/deepevo-brand.css'), 'utf8');
const rule = (text, selector) => { const start = text.indexOf(selector + '{'); return start < 0 ? '' : text.slice(start, text.indexOf('}', start) + 1); };
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const md5 = path => createHash('md5').update(readFileSync(path)).digest('hex');
const brand = () => require('../DeepEvoBrand.tsx');

// ---- CP-B1 assets: the kit ships bit-identical; nothing primary points at the old crop ----
// md5 pins = /workspace/verify-t237-c6-brand-READY/assets_pin (kit Iris refined v2).
const KIT_PINS = {
  'public/brand/mark.svg': 'd78ec9746a9f682d492f3c7eb7ee8233',
  'public/brand/mark-small.svg': '2db1c9cb54524b11c14d24640fa36096',
  'public/brand/mark-loader.svg': '1e2326be78a08618c7c9590153214493',
  'public/favicon.svg': '07ad58f89cad3024367d0757bc9fa29a',
  'public/favicon.ico': '42aab02f81c7c8354a6055f4cbc49e74',
  'public/apple-touch-icon.png': 'f0bcc4b3bf2de6debf8f55b770aede36',
  'public/og-image.png': 'd690b4199534dfbd8006556b5e4ea39f',
};
const KIT_PNGS = ['apple-touch-icon', 'icon-16', 'icon-32', 'icon-48', 'icon-192', 'icon-512', 'icon-maskable-512', 'og-image'];
const KIT_PRESENT = [
  'public/brand/mark-black.svg', 'public/brand/mark-white.svg',
  ...KIT_PNGS.map(name => `public/${name}.png`),
  'public/brand/lockup-horizontal-black.svg', 'public/brand/lockup-horizontal-white.svg',
  'public/brand/lockup-horizontal-black-on-white.svg', 'public/brand/lockup-horizontal-white-on-black.svg',
  'public/brand/lockup-stacked-black-on-white.svg', 'public/brand/lockup-stacked-white-on-black.svg',
];

test('t237 C6-brand H_assets: the Iris kit ships bit-identical under public/ and the PNGs are not git-ignored', () => {
  for (const [path, hash] of Object.entries(KIT_PINS)) assert.equal(md5(pkg(path)), hash, path);
  for (const path of KIT_PRESENT) assert.ok(existsSync(pkg(path)), path);
  // The root .gitignore drops *.png; each kit PNG needs its own exception or a clean checkout ships no icons.
  const ignore = readFileSync(repo('.gitignore'), 'utf8');
  for (const name of KIT_PNGS) assert.match(ignore, new RegExp(`^!web/research-agent-workstation/public/${name}\\.png$`, 'm'), name);
});

test('t237 C6-brand H_no_old_brand: no primary ref to the lockup crop / orbit / evomind mark / old app icon', () => {
  const files = ['components/workstation/DeepEvoBrand.tsx', 'components/workstation/deepevo-brand.css', 'app/globals.css', 'app/layout.tsx', 'app/login/page.tsx',
    'components/workstation/Sidebar.tsx', 'components/workstation/screens/AssistantScreen.tsx'];
  const all = [...files.map(file => readFileSync(app(file), 'utf8')), css, source('./TaskWorkspace.tsx'), source('./TaskRunView.tsx')].join('\n');
  assert.doesNotMatch(all, /deepevo-lockup\.png|evomind-mark|deepevo-orbit|DeepEvoGlyph/);
  assert.ok(!existsSync(pkg('public/brand/deepevo-lockup.png')));
  // Next app-dir icon files would emit their own <link rel="icon"> ahead of the kit favicon.
  assert.deepEqual(readdirSync(app('app')).filter(name => /^(icon|apple-icon|favicon)\./.test(name)), []);
});

// ---- H_b1 size cut ----
test('t237 C6-brand H_b1: ≥48 CSS px → master 7-seam; 16–47 → small cut 5-seam (never the master scaled down)', () => {
  const { markCut, markSrc } = brand();
  for (const size of [16, 20, 24, 28, 32, 47]) assert.equal(markCut(size), 'small', String(size));
  for (const size of [48, 64, 96]) assert.equal(markCut(size), 'master', String(size));
  assert.equal(markSrc(16), '/brand/mark-small.svg');
  assert.equal(markSrc(32), '/brand/mark-small.svg');
  assert.equal(markSrc(48), '/brand/mark.svg');
});

test('t237 C6-brand H_b1/H_b2: DeepEvoMark masks the kit file with currentColor; no tint / tilt / gradient', () => {
  const { DeepEvoMark } = brand();
  const small = renderToStaticMarkup(h(DeepEvoMark, { size: 20 }));
  const master = renderToStaticMarkup(h(DeepEvoMark, { size: 48, label: 'DeepEvo 深度进化' }));
  assert.match(small, /data-brand-mark="small"/);
  assert.match(small, /\/brand\/mark-small\.svg/);
  assert.doesNotMatch(small, /\/brand\/mark\.svg/);
  assert.match(master, /data-brand-mark="master"/);
  assert.match(master, /\/brand\/mark\.svg/);
  assert.match(master, /role="img" aria-label="DeepEvo 深度进化"/);
  assert.match(small, /aria-hidden="true"/);
  for (const html of [small, master]) assert.doesNotMatch(html, /rotate|gradient|filter|hue|opacity/);
  const mark = rule(brandCss, '.deepevo-mark');
  assert.match(mark, /background-color:currentColor/);
  assert.match(mark, /mask:var\(--deepevo-mark-src\) center\/contain no-repeat/);
  assert.match(mark, /-webkit-mask:var\(--deepevo-mark-src\) center\/contain no-repeat/);
  assert.doesNotMatch(mark, /transform|animation|gradient|#[0-9a-f]{3,8}/i);
});

// ---- H_b3/H_b4/H_b6 lockup ----
test('t237 C6-brand H_b3/H_b4: lockup is the kit image (black / white ink by theme), not system text', () => {
  const { DeepEvoLockup } = brand();
  const html = renderToStaticMarkup(h(DeepEvoLockup, { width: 168 }));
  assert.match(html, /<img[^>]*src="\/brand\/lockup-horizontal-black\.svg"/);
  assert.match(html, /<img[^>]*src="\/brand\/lockup-horizontal-white\.svg"/);
  assert.match(html, /role="img" aria-label="DeepEvo 深度进化"/);
  // The kit SVG keeps its own aspect ratio (viewBox 1303.21 × 420.61) — no hand-made crop.
  assert.match(html, /style="width:168px;height:54px"/);
  // Only the alt/aria name says DeepEvo — no visible text wordmark.
  assert.doesNotMatch(html.replace(/<[^>]+>/g, ''), /DeepEvo|深度进化/);
  assert.match(brandCss, /\[data-theme="dark"\] \.deepevo-lockup-light\{display:none\}/);
  assert.match(brandCss, /\[data-theme="dark"\] \.deepevo-lockup-dark\{display:block\}/);
  assert.match(rule(brandCss, '.deepevo-lockup-dark'), /display:none/);
  assert.doesNotMatch(brandCss, /font-family|@font-face/);
});

test('t237 C6-brand H_b6: sidebar = horizontal lockup ≥160px; ≤760px falls back to mark only (never 「Deep…」)', () => {
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /<a className="tw-brand" href="\/workspace" aria-label="DeepEvo 深度进化"><DeepEvoLockup width=\{168\} className="tw-brand-lockup" decorative \/><DeepEvoMark size=\{32\} className="tw-brand-mark" \/><\/a>/);
  assert.doesNotMatch(workspace, /DeepEvo<small>深度进化<\/small>/);
  assert.match(rule(css, '.tw-brand .tw-brand-mark'), /display:none/);
  assert.match(css, /@media\(max-width:760px\)\{\.tw-brand \.tw-brand-lockup\{display:none\}\.tw-brand \.tw-brand-mark\{display:block\}\}/);
  assert.doesNotMatch(rule(css, '.tw-brand'), /text-overflow|ellipsis/);
  // Legacy shell: lockup at the kit minimum, never narrower.
  const sidebar = readFileSync(app('components/workstation/Sidebar.tsx'), 'utf8');
  for (const [, width] of sidebar.matchAll(/<DeepEvoLockup width=\{(\d+)\}/g)) assert.ok(Number(width) >= 160, width);
});

test('t237 C6-brand: marks in the workspace are pure black / white (not accent, not muted)', () => {
  assert.match(css, /\.task-workspace\{--tw-brand-ink:#000;/);
  assert.match(css, /\[data-theme="dark"\] \.task-workspace\{\n--tw-brand-ink:#fff;/);
  for (const selector of ['.tw-brand', '.tw-loading']) assert.match(rule(css, selector), /color:var\(--tw-brand-ink\)/, selector);
});

// ---- H_b5 favicon / PWA / og ----
test('t237 C6-brand H_b5: favicon.ico + favicon.svg + apple-touch-icon + manifest + og:image wired in layout', () => {
  const layout = readFileSync(app('app/layout.tsx'), 'utf8');
  assert.match(layout, /\{ url: "\/favicon\.ico", sizes: "16x16 32x32 48x48" \}/);
  assert.match(layout, /\{ url: "\/favicon\.svg", type: "image\/svg\+xml" \}/);
  assert.match(layout, /apple: \[\{ url: "\/apple-touch-icon\.png", sizes: "180x180" \}\]/);
  assert.match(layout, /manifest: "\/manifest\.webmanifest"/);
  assert.match(layout, /images: \[\{ url: "\/og-image\.png", width: 1200, height: 630/);
  // og:image must be absolute on the public origin, not Next's localhost fallback.
  assert.match(layout, /metadataBase: new URL\(origin\)/);
  assert.match(layout, /import "@\/components\/workstation\/deepevo-brand\.css";/);
  const manifest = JSON.parse(readFileSync(pkg('public/manifest.webmanifest'), 'utf8'));
  assert.deepEqual(manifest.icons.map(icon => [icon.src, icon.sizes, icon.purpose ?? 'any']), [
    ['/icon-192.png', '192x192', 'any'], ['/icon-512.png', '512x512', 'any'], ['/icon-maskable-512.png', '512x512', 'maskable']]);
  assert.equal(manifest.background_color, '#000000');
  // Signed-out browsers must still fetch the manifest and favicon without a login redirect.
  const proxy = readFileSync(app('proxy.ts'), 'utf8');
  assert.match(proxy, /favicon\.ico\|manifest\.webmanifest\|/);
});

// ---- CP-B2 loader ----
test('t237 C6-brand H_b8/H_b9: loader keyframes rotate only, 1.8s ease-in-out, 360/7 master / 72° small', () => {
  assert.match(brandCss, /@keyframes deepevo-loader-master\{from\{transform:rotate\(0deg\)\}to\{transform:rotate\(51\.4286deg\)\}\}/);
  assert.match(brandCss, /@keyframes deepevo-loader-small\{from\{transform:rotate\(0deg\)\}to\{transform:rotate\(72deg\)\}\}/);
  const loader = rule(brandCss, '.deepevo-loader');
  assert.match(loader, /animation:deepevo-loader-master 1\.8s ease-in-out infinite/);
  assert.doesNotMatch(loader, /scale|opacity:0|translate/);
  assert.match(rule(brandCss, '.deepevo-loader.is-small'), /animation-name:deepevo-loader-small/);
  const { DeepEvoLoader } = brand();
  const big = renderToStaticMarkup(h(DeepEvoLoader, { size: 48 }));
  const small = renderToStaticMarkup(h(DeepEvoLoader, { size: 20 }));
  assert.match(big, /class="deepevo-mark deepevo-loader"/);
  assert.match(big, /\/brand\/mark\.svg/);
  assert.match(small, /class="deepevo-mark deepevo-loader is-small"/);
  assert.match(small, /\/brand\/mark-small\.svg/);
  for (const html of [big, small]) assert.match(html, /data-brand-loader/);
});

test('t237 C6-brand H_b10: an inactive loader stops + fades; reduced motion holds both cuts still', () => {
  const done = rule(brandCss, '.deepevo-loader.is-done');
  assert.match(done, /animation-play-state:paused/);
  assert.match(done, /opacity:0/);
  assert.match(rule(brandCss, '.deepevo-loader'), /transition:opacity 240ms ease/);
  assert.match(brandCss, /@media \(prefers-reduced-motion: reduce\)\{\.deepevo-loader,\.deepevo-loader\.is-small\{animation:none\}\}/);
  const { DeepEvoLoader } = brand();
  assert.match(renderToStaticMarkup(h(DeepEvoLoader, { size: 20, active: false })), /class="deepevo-mark deepevo-loader is-small is-done"/);
});

test('t237 C6-brand H_b11: the workspace loading state uses the 48px master loader in brand ink', () => {
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /<p role="status" className="tw-loading"><DeepEvoLoader size=\{48\} \/><span>正在读取你的任务…<\/span><\/p>/);
});

// ---- CP-B2 / CP-B3 in the conversation: avatar, live bubble, streaming ----
/** What a user reads: text nodes plus aria-labels / titles. */
const visible = html => [...html.replace(/<[^>]+>/g, ' ').split(/\s+/), ...[...html.matchAll(/(?:aria-label|title|placeholder|alt)="([^"]*)"/g)].map(match => match[1])].join(' ');
const ENGLISH_STATUS = /\b(Thinking|Streaming|Generating|worked for|took \d|elapsed)\b/i;
const history = (answer = '', extra = []) => ({ schema: 'evomind.assistant_message_history.v1', active_message_id: 'a1', messages: [
  { id: 'u1', role: 'user', content: '帮我分析' }, ...extra, { id: 'a1', role: 'assistant', content: answer }] });
const run = extra => ({ id: 'run_1', status: 'running', created_at: '2026-10-28T10:00:00Z', message_history: history(), ...extra });
const view = r => renderToStaticMarkup(h(require('./TaskRunView.tsx').TaskRunView, { view: 'conversation', run: r }));
const assistants = html => html.match(/<article class="tw-message tw-assistant[^"]*"/g) ?? [];

test('t237 C6-brand H_b7: assistant bubbles carry the small-cut mark avatar in brand ink; user bubbles do not', () => {
  const html = view(run({ status: 'completed', answer: '好了', completed_at: '2026-10-28T10:00:05Z', message_history: history('好了') }));
  assert.equal((html.match(/class="tw-avatar"/g) ?? []).length, 1);
  assert.match(html, /<article class="tw-message tw-assistant(?: tw-msg-answer)?"(?: data-testid="tw-answer")?><div class="tw-speaker"><span class="tw-avatar" aria-hidden="true"><span[^>]*data-brand-mark="small"[^>]*class="deepevo-mark tw-avatar-mark"/);
  assert.match(html, /<small>DeepEvo<\/small><\/div>/);
  assert.match(rule(css, '.tw-avatar'), /color:var\(--tw-brand-ink\)/);
  // No circular crop over the mark: the kit disc keeps its hollow centre.
  assert.doesNotMatch(rule(css, '.tw-avatar'), /border-radius|overflow:hidden|background/);
});

test('t237 C6-brand H_b10 (C6-answer): one loader per run — it spins in the process row while live, the avatar stays the static mark', () => {
  // C6-answer SPEC §5.1: the ThinkingRow carries the 16px small-cut loader; the speaker avatar no longer doubles it.
  const live = view(run({ streaming_text: '一段' }));
  assert.equal((live.match(/data-brand-loader/g) ?? []).length, 1);
  assert.match(live, /<span class="tw-thinking-icon" aria-hidden="true"><span[^>]*data-brand-loader[^>]*class="deepevo-mark deepevo-loader is-small tw-thinking-loader"/);
  assert.match(live, /<span class="tw-avatar" aria-hidden="true"><span[^>]*class="deepevo-mark tw-avatar-mark"/);
  // Settled: the row shows the static mark; nothing spins.
  const done = view(run({ status: 'completed', answer: '一段', completed_at: '2026-10-28T10:00:05Z', message_history: history('一段') }));
  assert.doesNotMatch(done, /data-brand-loader/);
  assert.match(done, /<span class="tw-thinking-icon" aria-hidden="true"><span[^>]*class="deepevo-mark tw-thinking-mark"/);
  // Older answers in the same conversation never carry a process row or a loader.
  const two = view(run({ streaming_text: '二', message_history: history('', [{ id: 'a0', role: 'assistant', content: '一' }, { id: 'u2', role: 'user', content: '再来' }]) }));
  assert.equal((two.match(/data-brand-loader/g) ?? []).length, 1);
  assert.equal((two.match(/data-testid="tw-thinking-row"/g) ?? []).length, 1);
});

test('t237 C6-brand H_b12 (C6-answer): before the first token the answer bubble holds the process row 「思考中…」, then text in the same bubble', () => {
  const before = view(run());
  assert.equal(assistants(before).length, 1);
  assert.match(before, /<article class="tw-message tw-assistant is-live(?: tw-msg-answer)?"/);
  assert.match(before, /<span class="tw-thinking-text">思考中…<\/span>/);
  assert.equal((before.match(/思考中…/g) ?? []).length, 1);
  assert.doesNotMatch(before, /tw-msg-actions|工作了|tw-md/);
  const during = view(run({ streaming_text: '第一段' }));
  assert.equal(assistants(during).length, 1);
  assert.match(during, /<article class="tw-message tw-assistant is-live(?: tw-msg-answer)?"/);
  assert.match(during, /第一段/);
  assert.match(during, /正在整理答案…/);
  assert.doesNotMatch(during, /思考中|tw-msg-actions|工作了/);
  // The process row sits above the answer inside the same bubble.
  assert.ok(during.indexOf('tw-thinking-row') < during.indexOf('第一段'));
});

test('t237 C6-brand H_b12: a run without message history (or a missing active row) still streams into one bubble', () => {
  const html = view(run({ message_history: undefined, prompt: '帮我分析', streaming_text: '正在写的第一段' }));
  assert.equal(assistants(html).length, 1);
  assert.match(html, /正在写的第一段/);
  const orphan = view(run({ message_history: { ...history(), active_message_id: 'gone' } }));
  assert.equal(assistants(orphan).length, 1);
  assert.equal((orphan.match(/思考中…/g) ?? []).length, 1);
});

test('t237 C6-brand H_b13: settled with no text ⇒ no empty bubble, no 「…」 placeholder, no loader', () => {
  for (const status of ['completed', 'failed', 'cancelled', 'waiting_approval']) {
    const html = view(run({ status, completed_at: '2026-10-28T10:00:05Z' }));
    // A failed run keeps its existing one-line failure bubble; nothing else may appear for an empty answer.
    assert.equal(assistants(html).length, status === 'failed' ? 1 : 0, status);
    assert.doesNotMatch(html, /<\/small><\/div><\/article>|思考中|data-brand-loader|>…</, status);
  }
  // Model failure owns the turn: no live bubble beside the fail card.
  const failed = view(run({ status: 'blocked', error_class: 'model_unavailable', answer: 'Model execution is unavailable: x', message_history: history('Model execution is unavailable: x') }));
  assert.equal(assistants(failed).length, 0);
  assert.match(failed, /data-testid="tw-model-fail"/);
  // Non-stream fallback: the whole answer arrives at once in a settled bubble.
  const once = view(run({ status: 'completed', answer: '完整答案', completed_at: '2026-10-28T10:00:05Z', message_history: history('完整答案') }));
  assert.equal(assistants(once).length, 1);
  assert.match(once, /<article class="tw-message tw-assistant(?: tw-msg-answer)?"/);
});

test('t237 C6-brand H_b14: no English Thinking / Streaming / Generating anywhere the user reads', () => {
  for (const r of [run(), run({ streaming_text: '一段' }), run({ status: 'completed', answer: '好', completed_at: '2026-10-28T10:00:03Z', message_history: history('好') })]) {
    assert.doesNotMatch(visible(view(r)), ENGLISH_STATUS);
  }
  assert.doesNotMatch(source('./TaskRunView.tsx'), />\s*(Thinking|Streaming|Generating)/);
});

// ---- CP-B4 「工作了 Ns」 ----
const presentation = () => require('./task-presentation.ts');

test('t237 C6-brand H_b15′ (C6-answer): whole seconds; under 1s ⇒ 「不到 1s」; never 0s, decimals or a rounded-up 1s', () => {
  const { secondsLabel, thinkingRowState } = presentation();
  assert.equal(secondsLabel(0), '不到 1s');
  assert.equal(secondsLabel(999), '不到 1s');
  assert.equal(secondsLabel(1000), '1s');
  assert.equal(secondsLabel(1999), '1s');
  assert.equal(secondsLabel(12_345), '12s');
  assert.equal(secondsLabel(3_725_000), '3725s');
  for (const ms of [0, 400, 1000, 1500, 59_999, 61_000]) assert.doesNotMatch(secondsLabel(ms), /(^|\D)0s|\.\d|worked|took|elapsed/);
  const at = ms => ({ id: 'r', status: 'completed', started_at: '2026-10-28T10:00:00.000Z', completed_at: new Date(Date.parse('2026-10-28T10:00:00.000Z') + ms).toISOString() });
  const tool = [{ kind: 'tool', label: '读取文件', status: 'completed', duration_ms: 300 }];
  assert.equal(thinkingRowState(at(400)).text, '思考了不到 1s');
  assert.equal(thinkingRowState({ ...at(400), activity: tool }).text, '工作了不到 1s · 1 步');
  assert.equal(thinkingRowState(at(999)).text, '思考了不到 1s');
  assert.equal(thinkingRowState(at(7_400)).text, '思考了 7s');
});

test('t237 C6-brand H_b15: workedMs — started_at (else created_at) → completed_at; bad stamps ⇒ null', () => {
  const { workedMs } = presentation();
  assert.equal(workedMs({ created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:01:12Z' }), 72_000);
  assert.equal(workedMs({ created_at: '2026-10-28T09:59:00Z', started_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:01:12Z' }), 72_000);
  assert.equal(workedMs({ created_at: '2026-10-28T10:00:00.000Z', completed_at: '2026-10-28T10:00:00.400Z' }), 400);
  assert.equal(workedMs({ created_at: '2026-10-28T10:00:00Z', completed_at: '' }), null);
  assert.equal(workedMs({ created_at: '2026-10-28T10:00:00Z' }), null);
  assert.equal(workedMs({ created_at: 'garbage', completed_at: '2026-10-28T10:01:12Z' }), null);
  // Clock skew never shows as 0s: a completion before the start is unknown, not instant.
  assert.equal(workedMs({ created_at: '2026-10-28T10:01:12Z', completed_at: '2026-10-28T10:00:00Z' }), null);
});

const processLineText = html => (html.match(/<span class="tw-thinking-text">([^<]*)</) ?? [])[1] ?? null;
const DUAL_TIME = /已完成 · \d+ 步 · \d+:\d+/;

test('t237 C6-brand H_b15′/H_b16′ (C6-answer): completed ⇒ 「工作了 Ns · N 步」 on the process row ABOVE the answer; no footer .tw-worked', () => {
  const stamps = { created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:01:12Z' };
  const rows = [
    { kind: 'tool', label: '搜索资料', status: 'completed', duration_ms: 3000 },
    { kind: 'tool', label: '读取文件', status: 'completed', duration_ms: 42_000 },
  ];
  for (const status of ['completed', 'delivered']) for (const extra of [{}, { activity: rows, activity_total: 2 }]) {
    const html = view(run({ status, answer: '第二次', ...stamps, ...extra,
      message_history: history('第二次', [{ id: 'a0', role: 'assistant', content: '第一次' }, { id: 'u2', role: 'user', content: '再来' }]) }));
    const label = `${status}${extra.activity ? ' +timeline' : ''}`;
    assert.equal(processLineText(html), extra.activity ? '工作了 72s · 2 步' : '思考了 72s', label);
    assert.equal((html.match(/工作了|思考了/g) ?? []).length, 1, label);
    assert.doesNotMatch(html, /tw-worked/, label);
    assert.doesNotMatch(html, DUAL_TIME, label);
    assert.doesNotMatch(processLineText(html), /\d+:\d+/, label);
    // Above the answer, inside the answer's bubble; the older answer has no row.
    assert.ok(html.indexOf('第一次') < html.indexOf('tw-thinking-row'), label);
    assert.ok(html.indexOf('tw-thinking-row') < html.indexOf('第二次'), label);
    assert.ok(html.indexOf('第二次') < html.lastIndexOf('tw-msg-actions'), label);
  }
  // H_b16′: muted, secondary size, expandable.
  const line = rule(css, '.task-workspace .tw-thinking-row>.tw-process-line');
  assert.match(line, /color:var\(--tw-muted\)/);
  assert.match(line, /font-size:13px/);
  assert.doesNotMatch(line, /font-weight:[6-9]|border:1px|box-shadow/);
  const html = view(run({ status: 'completed', answer: '好了', ...stamps, activity: rows, activity_total: 2, message_history: history('好了') }));
  assert.match(html, /<button type="button" class="tw-process-line" data-testid="tw-process-line" aria-expanded="false" aria-controls="[^"]+"/);
  // Copy takes the answer only.
  assert.match(source('./TaskRunView.tsx'), /<MessageActions text=\{content\}/);
});

test('t237 C6-brand H_b17: no 「工作了」 while streaming, failed, cancelled, waiting or without timestamps', () => {
  const stamps = { created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:00:09Z' };
  for (const status of ['running', 'queued', 'planning', 'verifying', 'recovering', 'failed', 'cancelled', 'waiting_approval', 'blocked', 'paused']) {
    const html = view(run({ status, answer: '部分', partial_answer: '部分', streaming_text: '部分', ...stamps, message_history: history('部分') }));
    assert.doesNotMatch(html, /工作了|思考了/, status);
  }
  assert.doesNotMatch(view(run({ status: 'completed', answer: '好', created_at: '', completed_at: '', message_history: history('好') })), /工作了|思考了/);
  // The model-failure card owns a blocked model turn.
  assert.doesNotMatch(view(run({ status: 'blocked', error_class: 'model_unavailable', answer: 'Model execution is unavailable: x', ...stamps })), /工作了/);
});

test('t237 C6-brand H_b18: copy stays Chinese — no worked for / took / elapsed in source or markup', () => {
  const html = view(run({ status: 'completed', answer: '好', created_at: '2026-10-28T10:00:00Z', completed_at: '2026-10-28T10:00:03Z', message_history: history('好') }));
  assert.match(html, /思考了 3s/);
  assert.doesNotMatch(visible(html), ENGLISH_STATUS);
  // Whole words only: identifiers such as formatElapsed / elapsedMs are code, not copy.
  const copy = source('./task-presentation.ts') + source('./TaskRunView.tsx') + source('./ThinkingRow.tsx');
  assert.doesNotMatch(copy, /['`"][^'`"\n]*\b(worked for|took \d|took \$\{|elapsed)\b[^'`"\n]*['`"]/i);
  // C6-answer H_a20: no English Thinking / Working / Thought for as copy (identifiers like thinkingRowState are code).
  assert.doesNotMatch(copy, /['`"][^'`"\n]*\b(Thought for|Thinking|Working)\b[^'`"\n]*['`"]/);
});

// ---- softs ----
test('t237 C6-brand S_b1: ≤760px the model picker drops the vendor — 「Sonnet 5.5」, version never cut', () => {
  const { Picker } = require('./ComposerPicker.tsx');
  const html = renderToStaticMarkup(h(Picker, { id: 'p', label: '本次使用的模型', value: 'platform', text: 'Claude Sonnet 5.5', groups: [], onChange: () => {} }));
  assert.match(html, /<span class="tw-picker-name"><span class="tw-picker-vendor">Claude <\/span>Sonnet<\/span><span class="tw-picker-version"> 5\.5<\/span>/);
  assert.match(html, /aria-label="本次使用的模型：Claude Sonnet 5\.5"/);
  assert.match(css, /@media\(max-width:760px\)\{\.tw-picker-vendor\{display:none\}\}/);
  // One-word names have no vendor to drop.
  const other = renderToStaticMarkup(h(Picker, { id: 'q', label: '模型', value: 'x', text: 'GPT 5', groups: [], onChange: () => {} }));
  assert.doesNotMatch(other, /tw-picker-vendor/);
});

test('t237 C6-brand S_b2: dark 无痕 on-state uses the accent, not brightness alone', () => {
  assert.match(css, /\[data-theme="dark"\] \.task-workspace \.tw-incognito\[aria-pressed="true"\]\{border-color:var\(--tw-accent\);color:var\(--tw-accent\)\}/);
});

test('t237 C6-brand S_b2 (C6br4): the dark accent rule wins the cascade; the toggle renders aria-pressed; light stays ink', () => {
  const { IncognitoToggle } = require('./MemorySettings.tsx');
  for (const on of [true, false]) assert.match(renderToStaticMarkup(h(IncognitoToggle, { on, onChange: () => {} })), new RegExp(`class="tw-incognito" aria-pressed="${on}"`));
  const base = '.task-workspace .tw-incognito[aria-pressed="true"]{border-color:var(--tw-ink);color:var(--tw-ink)}';
  const dark = '[data-theme="dark"] .task-workspace .tw-incognito[aria-pressed="true"]{border-color:var(--tw-accent);color:var(--tw-accent)}';
  const baseAt = css.indexOf(base);
  const darkAt = css.indexOf(dark);
  assert.ok(baseAt >= 0 && darkAt > baseAt, 'dark rule after the base pressed rule');
  // Nothing after the dark rule re-colors the pressed toggle.
  const later = css.slice(darkAt + dark.length);
  for (const block of later.matchAll(/([^{}]*\.tw-incognito\[aria-pressed="true"\][^{}]*)\{([^}]*)\}/g)) assert.doesNotMatch(block[2], /(^|;)\s*(border-color|color|border)\s*:/, block[1]);
  assert.match(css, /\[data-theme="dark"\] \.task-workspace\{[^}]*--tw-accent:#4FB37E;/);
});
