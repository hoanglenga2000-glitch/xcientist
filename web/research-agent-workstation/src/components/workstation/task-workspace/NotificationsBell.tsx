'use client';
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Bell } from 'lucide-react';
import {
  createNotificationStore, NOTIF_EMPTY, NOTIF_MARK_ALL, NOTIF_PREF_LABEL, notificationHref, relativeTime, unreadCount,
  type AppNotification, type NotificationState, type RunEvent,
} from './notifications';
import type { UserTask } from './types';

type Store = ReturnType<typeof createNotificationStore>;
let store: Store | null = null;
let opener: ((href: string) => void) | null = null;
const permission = () => typeof Notification === 'undefined' ? 'unsupported' : Notification.permission;
/** The task in front of you; a hidden tab has none, so its own run may notify. */
const activeTaskId = () => document.visibilityState === 'visible' ? new URLSearchParams(window.location.search).get('task') ?? '' : '';
function open(href: string) {
  if (opener) opener(href); else window.location.assign(href);
}

/** H10-5: title/body from the copy table; tag dedupes at the OS too; no custom sound, so system 勿扰 applies. */
function browserNotify(item: AppNotification) {
  if (typeof Notification === 'undefined') return;
  try {
    const shown = new Notification(item.title, { body: item.body, tag: item.key });
    shown.onclick = () => { window.focus(); store?.markRead(item.key); open(notificationHref(item)); shown.close(); };
  } catch { /* Some mobile browsers only allow service-worker notifications; the bell still has it. */ }
}

function notificationStore(): Store {
  store ??= createNotificationStore({ storage: typeof window === 'undefined' ? null : window.localStorage, notify: browserNotify });
  return store;
}

/** TaskSession: a live run of the open task changed into a notifying status. */
export function reportRunStatus(event: RunEvent) {
  notificationStore().report(event, { activeTaskId: activeTaskId(), permission: permission() });
}
/** Task list read. `historical` (page load / return): list only, never a browser replay. */
export function reconcileNotifications(tasks: UserTask[], historical: boolean) {
  notificationStore().reconcile(tasks, { activeTaskId: activeTaskId(), permission: permission(), historical });
}

type ViewProps = {
  items: AppNotification[]; open: boolean; now: number; browserPref: boolean;
  onToggle: () => void; onOpen: (item: AppNotification) => void; onMarkAll: () => void; onPref: (value: boolean) => void;
};
/** Presentational part, rendered statically in tests. */
export function NotificationsBellView({ items, open: expanded, now, browserPref, onToggle, onOpen, onMarkAll, onPref }: ViewProps) {
  const unread = unreadCount({ items } as NotificationState);
  return <div className="tw-notif">
    <button type="button" className="tw-notif-bell" data-testid="tw-notif-bell" aria-label={unread ? `通知，${unread} 条未读` : '通知'} title="通知"
      aria-haspopup="dialog" aria-expanded={expanded} aria-controls="tw-notif-panel" onClick={onToggle}>
      <Bell size={18} aria-hidden="true" />
      {unread > 0 && <span className="tw-notif-badge" data-testid="tw-notif-badge" aria-hidden="true">{unread > 99 ? '99+' : unread}</span>}
    </button>
    {expanded && <div id="tw-notif-panel" className="tw-notif-panel" data-testid="tw-notif-panel" role="dialog" aria-label="通知">
      <div className="tw-notif-head"><span>通知</span>
        {unread > 0 && <button type="button" className="tw-text-button" data-testid="tw-notif-mark-all" onClick={onMarkAll}>{NOTIF_MARK_ALL}</button>}</div>
      {items.length ? <ul className="tw-notif-list">{items.map(item => <li key={item.key}>
        <a className="tw-notif-row" data-testid="tw-notif-row" data-unread={item.read ? undefined : 'true'} href={notificationHref(item)}
          onClick={event => { event.preventDefault(); onOpen(item); }}>
          <span className="tw-notif-title">{item.title}</span><time dateTime={new Date(item.at).toISOString()}>{relativeTime(item.at, now)}</time>
        </a></li>)}</ul> : <p className="tw-notif-empty" data-testid="tw-notif-empty">{NOTIF_EMPTY}</p>}
      <label className="tw-notif-pref"><input type="checkbox" checked={browserPref} onChange={event => onPref(event.target.checked)} />{NOTIF_PREF_LABEL}</label>
    </div>}
  </div>;
}

const emptyServer = { items: [], pushed: [], seeded: false } satisfies NotificationState;

export function NotificationsBell({ onNavigate }: { onNavigate: (href: string) => void }) {
  const notifications = notificationStore();
  const state = useSyncExternalStore(notifications.subscribe, notifications.get, () => emptyServer);
  const browserPref = useSyncExternalStore(notifications.subscribe, notifications.browserPref, () => false);
  const [expanded, setExpanded] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => { opener = onNavigate; return () => { if (opener === onNavigate) opener = null; }; }, [onNavigate]);
  useEffect(() => {
    if (!expanded) return;
    setNow(Date.now());
    const close = (event: Event) => {
      if (event instanceof KeyboardEvent ? event.key === 'Escape' : !root.current?.contains(event.target as Node)) {
        setExpanded(false);
        if (event instanceof KeyboardEvent) root.current?.querySelector<HTMLElement>('.tw-notif-bell')?.focus();
      }
    };
    document.addEventListener('keydown', close); document.addEventListener('pointerdown', close);
    return () => { document.removeEventListener('keydown', close); document.removeEventListener('pointerdown', close); };
  }, [expanded]);
  /** S10-1: turning it on is the user gesture that shows the system permission prompt. */
  function choosePref(value: boolean) {
    if (!value || typeof Notification === 'undefined') { notifications.setBrowserPref(false); return; }
    if (Notification.permission === 'granted') { notifications.setBrowserPref(true); return; }
    void Notification.requestPermission().then(result => notifications.setBrowserPref(result === 'granted')).catch(() => notifications.setBrowserPref(false));
  }
  return <div ref={root} className="tw-notif-wrap">
    <NotificationsBellView items={state.items} open={expanded} now={now} browserPref={browserPref} onToggle={() => setExpanded(value => !value)}
      onMarkAll={notifications.markAllRead} onPref={choosePref}
      onOpen={item => { notifications.markRead(item.key); setExpanded(false); onNavigate(notificationHref(item)); }} />
  </div>;
}
