/**
 * t237 C6e G14 (CP10): what reached a terminal state while you were elsewhere.
 * Only 已完成 / 需要你确认 / 这次没有完成 / 已停止; one item per run per terminal kind, kept in localStorage so a
 * refresh or reconnect never fires again. Browser Notification is off by default and never for the task in front of you.
 */
import { isAwaitingConfirm, owningTaskHref, STOPPABLE_RUN_STATUSES } from './task-presentation';

export type NotificationKind = 'completed' | 'awaiting' | 'failed' | 'stopped';
export type AppNotification = { key: string; runId: string; taskId: string; kind: NotificationKind; title: string; body: string; at: number; read: boolean };
export type NotificationState = { items: AppNotification[]; pushed: string[]; seeded: boolean };
/** One observed run status. */
export type RunEvent = { runId: string; taskId: string; taskTitle?: string; status: string; planPending?: boolean; at?: number };
export type BrowserContext = { activeTaskId: string; enabled: boolean; permission: string; historical: boolean };
type TaskRow = { id: string; title?: string; latest_run?: { id?: string; status?: string; plan_pending?: boolean; updated_at?: string } | null };
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem'>;

export const NOTIF_EMPTY = '还没有通知';
export const NOTIF_PREF_LABEL = '完成后通知我';
export const NOTIF_MARK_ALL = '全部标为已读';
const STATE_KEY = 'tw-notifications-v1';
const PREF_KEY = 'tw-notifications-browser';
const ITEM_LIMIT = 50;
const PUSHED_LIMIT = 500;
const TITLE_MAX = 18;
const DONE_SUFFIX = '已完成';

const COPY: Record<NotificationKind, { title: string; body: string }> = {
  completed: { title: '任务已完成', body: '可以查看结果' },
  awaiting: { title: '需要你确认', body: '打开后继续' },
  failed: { title: '这次没有完成', body: '打开后查看原因' },
  stopped: { title: '已停止', body: '' },
};

/** H10-1: the only statuses that notify. Queued / running / thinking never do. */
export function notificationKind(status: string, planPending = false): NotificationKind | null {
  if (isAwaitingConfirm({ status, plan_pending: planPending })) return 'awaiting';
  if (status === 'completed' || status === 'delivered') return 'completed';
  if (status === 'failed' || status === 'blocked') return 'failed';
  if (status === 'cancelled' || status === 'stopped') return 'stopped';
  return null;
}

/** A change out of an active status into a notifying one; the first snapshot of an old run is not a transition. */
export function isTerminalTransition(previous: string, next: string, planPending = false) {
  return Boolean(previous) && previous !== next && STOPPABLE_RUN_STATUSES.includes(previous) && notificationKind(next, planPending) !== null;
}

/** H10-4: 任务已完成 or {短任务名}已完成; titles stay ≤18 characters. */
export function notificationCopy(kind: NotificationKind, taskTitle = '') {
  const copy = COPY[kind];
  const name = [...taskTitle.replace(/\s+/g, ' ').trim()];
  if (kind !== 'completed' || !name.length) return { ...copy };
  const room = TITLE_MAX - DONE_SUFFIX.length;
  const short = name.length > room ? `${name.slice(0, room - 1).join('')}…` : name.join('');
  return { title: `${short}${DONE_SUFFIX}`, body: copy.body };
}

export const notificationKey = (runId: string, kind: NotificationKind) => `${runId}:${kind}`;
export const emptyState = (): NotificationState => ({ items: [], pushed: [], seeded: false });
export const unreadCount = (state: NotificationState) => state.items.filter(item => !item.read).length;
export const markRead = (state: NotificationState, key: string): NotificationState =>
  ({ ...state, items: state.items.map(item => item.key === key ? { ...item, read: true } : item) });
export const markAllRead = (state: NotificationState): NotificationState => ({ ...state, items: state.items.map(item => ({ ...item, read: true })) });
/** Same deep link the run card uses for another task's run. */
export const notificationHref = (item: Pick<AppNotification, 'runId' | 'taskId'>) => owningTaskHref('', { id: item.runId, user_task_id: item.taskId });

/** H10-2: preference on, permission granted, live (not a return replay) and not the task already in front of you. */
export function shouldBrowserNotify(item: AppNotification, context: BrowserContext) {
  return context.enabled && context.permission === 'granted' && !context.historical && context.activeTaskId !== item.taskId;
}

/** New terminal events become unread items; at most one (the newest) is offered to the browser per batch. */
export function ingest(state: NotificationState, events: RunEvent[], context: BrowserContext) {
  const pushed = new Set(state.pushed);
  const fresh: AppNotification[] = [];
  for (const event of events) {
    const kind = notificationKind(event.status, event.planPending);
    if (!kind || !event.runId || !event.taskId) continue;
    const key = notificationKey(event.runId, kind);
    if (pushed.has(key)) continue;
    pushed.add(key);
    fresh.push({ key, runId: event.runId, taskId: event.taskId, kind, ...notificationCopy(kind, event.taskTitle), at: event.at ?? Date.now(), read: false });
  }
  if (!fresh.length) return { state, browser: null };
  fresh.sort((a, b) => b.at - a.at);
  const items = [...fresh, ...state.items].sort((a, b) => b.at - a.at).slice(0, ITEM_LIMIT);
  const browser = fresh.find(item => shouldBrowserNotify(item, context)) ?? null;
  return { state: { ...state, items, pushed: [...pushed].slice(-PUSHED_LIMIT) }, browser };
}

/** H10-6: the task list's latest runs. The first-ever read is the silent baseline, so old history never floods the list. */
export function reconcileTasks(state: NotificationState, tasks: TaskRow[], context: BrowserContext) {
  const events: RunEvent[] = tasks.flatMap(task => task.latest_run?.id ? [{
    runId: task.latest_run.id, taskId: task.id, taskTitle: task.title, status: String(task.latest_run.status || ''),
    planPending: task.latest_run.plan_pending === true, at: Date.parse(task.latest_run.updated_at || '') || Date.now(),
  }] : []);
  if (state.seeded) return ingest(state, events, context);
  const baseline = ingest(state, events, { ...context, historical: true }).state;
  return { state: { ...baseline, items: state.items, seeded: true }, browser: null };
}

export function relativeTime(at: number, now = Date.now()) {
  const minutes = Math.floor(Math.max(0, now - at) / 60e3);
  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes} 分钟前`;
  if (minutes < 24 * 60) return `${Math.floor(minutes / 60)} 小时前`;
  if (minutes < 48 * 60) return '昨天';
  const date = new Date(at);
  return `${date.getMonth() + 1}月${date.getDate()}日`;
}

export function loadState(storage: Storage | null | undefined): NotificationState {
  try {
    const value = JSON.parse(storage?.getItem(STATE_KEY) || 'null') as Partial<NotificationState> | null;
    if (!value || !Array.isArray(value.items) || !Array.isArray(value.pushed)) return emptyState();
    return { items: value.items, pushed: value.pushed, seeded: value.seeded === true };
  } catch { return emptyState(); }
}
export function saveState(storage: Storage | null | undefined, state: NotificationState) {
  try { storage?.setItem(STATE_KEY, JSON.stringify(state)); } catch { /* Private mode / quota: the in-memory list still works. */ }
}
export const loadBrowserPref = (storage: Storage | null | undefined) => { try { return storage?.getItem(PREF_KEY) === '1'; } catch { return false; } };

type ReportContext = { activeTaskId: string; permission: string };
/** One shared store for the bell, the task list reconcile and TaskSession's run stream. */
export function createNotificationStore({ storage, notify }: { storage?: Storage | null; notify: (item: AppNotification) => void }) {
  let state = loadState(storage);
  let pref = loadBrowserPref(storage);
  const listeners = new Set<() => void>();
  const commit = (next: NotificationState) => {
    if (next === state) return;
    state = next; saveState(storage, state); listeners.forEach(listener => listener());
  };
  const apply = (result: { state: NotificationState; browser: AppNotification | null }) => {
    commit(result.state);
    if (result.browser) notify(result.browser);
  };
  return {
    get: () => state,
    browserPref: () => pref,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    /** A live transition seen in this tab (TaskSession stream or a list poll). */
    report(event: RunEvent, context: ReportContext) {
      apply(ingest(state, [event], { ...context, enabled: pref, historical: false }));
    },
    reconcile(tasks: TaskRow[], context: ReportContext & { historical: boolean }) {
      apply(reconcileTasks(state, tasks, { ...context, enabled: pref }));
    },
    markRead(key: string) { commit(markRead(state, key)); },
    markAllRead() { commit(markAllRead(state)); },
    setBrowserPref(value: boolean) {
      pref = value;
      try { storage?.setItem(PREF_KEY, value ? '1' : '0'); } catch { /* Kept for this tab only. */ }
      listeners.forEach(listener => listener());
    },
  };
}
