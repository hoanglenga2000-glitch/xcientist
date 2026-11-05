import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// t237 C6e G14 (CP10): terminal-run notifications — copy, dedupe, foreground suppress, return sync, bell.
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
const h = (type, props, ...children) => React.createElement(type, props, ...children);
const N = require('./notifications.ts');

const BANNED = /Agent|agent|Deep research|Please review|智能助手|正在思考您的|[!！]|\p{Extended_Pictographic}/u;
const NOW = Date.parse('2026-10-31T12:00:00Z');
const TASK = 'utask_' + 'a'.repeat(32);
const OTHER = 'utask_' + 'b'.repeat(32);
const ev = (extra = {}) => ({ runId: 'run_1', taskId: TASK, taskTitle: '文献综述', status: 'completed', at: NOW, ...extra });
const live = { activeTaskId: '', enabled: true, permission: 'granted', historical: false };
const memory = () => { const data = new Map(); return { getItem: k => data.get(k) ?? null, setItem: (k, v) => data.set(k, String(v)), removeItem: k => data.delete(k), data }; };

test('H10-4 notificationCopy: exact Chinese table, ≤18-char titles, no AI-flavoured words', () => {
  assert.deepEqual(N.notificationCopy('completed'), { title: '任务已完成', body: '可以查看结果' });
  assert.deepEqual(N.notificationCopy('awaiting', '文献综述'), { title: '需要你确认', body: '打开后继续' });
  assert.deepEqual(N.notificationCopy('failed', '文献综述'), { title: '这次没有完成', body: '打开后查看原因' });
  assert.deepEqual(N.notificationCopy('stopped', '文献综述'), { title: '已停止', body: '' });
  assert.equal(N.notificationCopy('completed', '文献综述').title, '文献综述已完成');
  const long = N.notificationCopy('completed', '一个非常非常长的关于蛋白质结构预测的任务名称').title;
  assert.ok([...long].length <= 18, long);
  assert.ok(long.endsWith('已完成'));
  assert.equal(N.notificationCopy('completed', '   ').title, '任务已完成');
  for (const kind of ['completed', 'awaiting', 'failed', 'stopped']) {
    const copy = N.notificationCopy(kind, '综述');
    assert.ok([...copy.title].length <= 18);
    assert.ok([...copy.body].length <= 40);
    assert.doesNotMatch(copy.title + copy.body, BANNED);
  }
  assert.equal(N.NOTIF_EMPTY, '还没有通知');
});

test('H10-1 triggers: only completed / awaiting / failed / stopped; never queued, running, thinking', () => {
  assert.equal(N.notificationKind('completed'), 'completed');
  assert.equal(N.notificationKind('delivered'), 'completed');
  assert.equal(N.notificationKind('waiting_approval'), 'awaiting');
  assert.equal(N.notificationKind('completed', true), 'awaiting');
  assert.equal(N.notificationKind('failed'), 'failed');
  assert.equal(N.notificationKind('blocked'), 'failed');
  assert.equal(N.notificationKind('cancelled'), 'stopped');
  for (const status of ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing', 'draft', '', 'thinking'])
    assert.equal(N.notificationKind(status), null, status);
  // TaskSession reports only a change out of an active status, never the first snapshot of an old run.
  assert.equal(N.isTerminalTransition('running', 'completed'), true);
  assert.equal(N.isTerminalTransition('running', 'waiting_approval'), true);
  assert.equal(N.isTerminalTransition('waiting_approval', 'completed'), true);
  assert.equal(N.isTerminalTransition('', 'completed'), false);
  assert.equal(N.isTerminalTransition('completed', 'completed'), false);
  assert.equal(N.isTerminalTransition('queued', 'running'), false);
});

test('H10-2 dedupe: same runId + kind twice → one entry, one browser attempt', () => {
  let state = { ...N.emptyState(), seeded: true };
  const first = N.ingest(state, [ev()], live);
  assert.equal(first.state.items.length, 1);
  assert.equal(first.browser?.key, 'run_1:completed');
  const again = N.ingest(first.state, [ev(), ev()], live);
  assert.equal(again.state.items.length, 1);
  assert.equal(again.browser, null);
  // A different terminal kind for the same run is its own item.
  const confirm = N.ingest(again.state, [ev({ status: 'waiting_approval' })], live);
  assert.equal(confirm.state.items.length, 2);
  // Several new terminals at once: at most one system notification, never a chain.
  state = { ...N.emptyState(), seeded: true };
  const batch = N.ingest(state, [ev({ runId: 'a', at: NOW - 2000 }), ev({ runId: 'b', taskId: OTHER, at: NOW })], live);
  assert.equal(batch.state.items.length, 2);
  assert.equal(batch.browser?.runId, 'b');
  assert.deepEqual(batch.state.items.map(item => item.runId), ['b', 'a']);
});

test('H10-2 foreground suppress: active task gets no browser notification, badge still counts', () => {
  const state = { ...N.emptyState(), seeded: true };
  const out = N.ingest(state, [ev()], { ...live, activeTaskId: TASK });
  assert.equal(out.browser, null);
  assert.equal(N.unreadCount(out.state), 1);
  const item = out.state.items[0];
  assert.equal(N.shouldBrowserNotify(item, { ...live, activeTaskId: TASK }), false);
  assert.equal(N.shouldBrowserNotify(item, { ...live, activeTaskId: OTHER }), true);
  // Default off; permission must be granted.
  assert.equal(N.shouldBrowserNotify(item, { ...live, enabled: false }), false);
  assert.equal(N.shouldBrowserNotify(item, { ...live, permission: 'default' }), false);
  assert.equal(N.shouldBrowserNotify(item, { ...live, permission: 'denied' }), false);
  assert.equal(N.loadBrowserPref(memory()), false);
});

test('H10-3 list routing: row opens the owning task, no run id / internals in visible copy', () => {
  const item = N.ingest({ ...N.emptyState(), seeded: true }, [ev()], live).state.items[0];
  const href = N.notificationHref(item);
  const url = new URL(href, 'http://x');
  assert.equal(url.pathname, '/workspace');
  assert.equal(url.searchParams.get('task'), TASK);
  assert.equal(url.searchParams.get('assistant_run'), 'run_1');
  assert.equal(N.relativeTime(NOW - 10e3, NOW), '刚刚');
  assert.equal(N.relativeTime(NOW - 5 * 60e3, NOW), '5 分钟前');
  assert.equal(N.relativeTime(NOW - 3 * 3600e3, NOW), '3 小时前');
  assert.equal(N.relativeTime(NOW - 30 * 3600e3, NOW), '昨天');
  assert.match(N.relativeTime(NOW - 5 * 86400e3, NOW), /^\d+月\d+日$/);
});

test('H10-6 return sync: first load seeds silently; later reconcile lists unread but never replays browser', () => {
  const tasks = [
    { id: TASK, title: '综述', status: 'delivered', latest_run: { id: 'r_old', status: 'completed', updated_at: new Date(NOW).toISOString() } },
    { id: OTHER, title: '实验', status: 'running', latest_run: { id: 'r_live', status: 'running' } },
  ];
  // First ever load: existing history becomes the baseline — no list flood, no browser.
  const seeded = N.reconcileTasks(N.emptyState(), tasks, { ...live, historical: true });
  assert.equal(seeded.state.seeded, true);
  assert.equal(seeded.state.items.length, 0);
  assert.equal(seeded.browser, null);
  // Left the page; the live run completed server-side. Return (historical) → unread on bell, no system toast.
  const done = tasks.map(task => task.id === OTHER ? { ...task, status: 'delivered', latest_run: { id: 'r_live', status: 'completed' } } : task);
  const back = N.reconcileTasks(seeded.state, done, { ...live, historical: true });
  assert.equal(back.state.items.length, 1);
  assert.equal(back.state.items[0].taskId, OTHER);
  assert.equal(N.unreadCount(back.state), 1);
  assert.equal(back.browser, null);
  // Reload / reconnect: already recorded → nothing new, even when live.
  const reload = N.reconcileTasks(back.state, done, live);
  assert.equal(reload.state.items.length, 1);
  assert.equal(reload.browser, null);
  // Persisted keys survive a refresh.
  const storage = memory();
  N.saveState(storage, back.state);
  const restored = N.loadState(storage);
  assert.equal(restored.seeded, true);
  assert.equal(N.reconcileTasks(restored, done, live).browser, null);
  assert.equal(N.loadState(memory()).seeded, false);
});

test('S10-2 mark read / mark all read clears the badge', () => {
  let state = { ...N.emptyState(), seeded: true };
  state = N.ingest(state, [ev({ runId: 'a' }), ev({ runId: 'b' }), ev({ runId: 'c', status: 'failed' })], live).state;
  assert.equal(N.unreadCount(state), 3);
  state = N.markRead(state, 'a:completed');
  assert.equal(N.unreadCount(state), 2);
  state = N.markAllRead(state);
  assert.equal(N.unreadCount(state), 0);
  assert.equal(state.items.length, 3);
});

test('store: browser notifier fires once per key, pref off by default, persists to storage', () => {
  const storage = memory();
  const sent = [];
  const store = N.createNotificationStore({ storage, notify: item => sent.push(item.key) });
  store.reconcile([], { activeTaskId: '', historical: true, permission: 'granted' });
  store.report(ev(), { activeTaskId: '', permission: 'granted' });
  assert.equal(sent.length, 0, 'pref default off');
  assert.equal(store.get().items.length, 1);
  store.setBrowserPref(true);
  store.report(ev({ runId: 'r2' }), { activeTaskId: '', permission: 'granted' });
  store.report(ev({ runId: 'r2' }), { activeTaskId: '', permission: 'granted' });
  assert.deepEqual(sent, ['r2:completed']);
  const again = N.createNotificationStore({ storage, notify: item => sent.push(item.key) });
  again.report(ev({ runId: 'r2' }), { activeTaskId: '', permission: 'granted' });
  assert.equal(sent.length, 1);
  assert.equal(again.get().items.length, 2);
  assert.equal(again.browserPref(), true);
});

test('H10-3 render: bell + panel testids, empty copy, badge, no internals', () => {
  const { NotificationsBellView } = require('./NotificationsBell.tsx');
  const empty = renderToStaticMarkup(h(NotificationsBellView, { items: [], open: true, now: NOW, browserPref: false, onToggle() {}, onOpen() {}, onMarkAll() {}, onPref() {} }));
  for (const id of ['tw-notif-bell', 'tw-notif-panel', 'tw-notif-empty']) assert.match(empty, new RegExp(`data-testid="${id}"`));
  assert.match(empty, />还没有通知</);
  assert.doesNotMatch(empty, /tw-notif-badge/);
  assert.match(empty, /完成后通知我/);
  const items = N.ingest({ ...N.emptyState(), seeded: true }, [ev({ at: NOW - 120e3 }), ev({ runId: 'run_2', status: 'waiting_approval', at: NOW - 60e3 })], live).state.items;
  const full = renderToStaticMarkup(h(NotificationsBellView, { items, open: true, now: NOW, browserPref: true, onToggle() {}, onOpen() {}, onMarkAll() {}, onPref() {} }));
  assert.equal(full.match(/data-testid="tw-notif-row"/g)?.length, 2);
  assert.match(full, /data-testid="tw-notif-badge"[^>]*>2</);
  assert.match(full, /data-testid="tw-notif-mark-all"/);
  assert.match(full, /需要你确认/);
  assert.match(full, /文献综述已完成/);
  assert.match(full, /2 分钟前/);
  assert.doesNotMatch(full.replace(/href="[^"]*"/g, ''), /run_1|run_2|utask_|tenant/);
  assert.match(full, /href="\/workspace\?task=utask_a+&amp;assistant_run=run_1/);
  const closed = renderToStaticMarkup(h(NotificationsBellView, { items, open: false, now: NOW, browserPref: false, onToggle() {}, onOpen() {}, onMarkAll() {}, onPref() {} }));
  assert.doesNotMatch(closed, /data-testid="tw-notif-panel"/);
  assert.match(closed, /aria-label="通知，2 条未读"/);
});

test('wiring + S10-3: bell in topbar, TaskSession reports transitions without a second EventSource, 44px @393', () => {
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /<NotificationsBell[\s\S]*?\/>[\s\S]*?账户<\/a><\/header>/);
  assert.match(workspace, /reconcileNotifications\(/);
  assert.match(workspace, /visibilitychange/);
  const session = source('./TaskSession.tsx');
  assert.equal(session.match(/new EventSource/g)?.length, 1);
  assert.match(session, /isTerminalTransition\(/);
  const css = source('./workspace.css');
  assert.match(css, /\.tw-notif-bell\{[^}]*min-width:44px[^}]*min-height:44px/);
  assert.match(css, /\.tw-notif-row\{[^}]*min-height:44px/);
  assert.match(css, /\.tw-notif-badge\{[^}]*background:var\(--tw-accent\)/);
});
