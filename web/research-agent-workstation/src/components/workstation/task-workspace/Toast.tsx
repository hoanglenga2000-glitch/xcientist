'use client';
import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from 'react';
import { TOAST_MS } from './conversation-menu';

export type ToastValue = { message: string; action?: { label: string; run: () => void } };
type Entry = ToastValue & { id: number };

// One toast at a time: a new action replaces the old one, whose action then stands (spec §5).
let current: Entry | null = null;
let serial = 0;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(listener => listener());
export function showToast(value: ToastValue) { current = { ...value, id: ++serial }; emit(); }
export function dismissToast(id?: number) { if (current && (id === undefined || current.id === id)) { current = null; emit(); } }
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };

/** Space under the composer (page padding keeps it above its sticky offset); the CSS adds --tw-composer-h + 16px. */
function measureComposerGap() {
  const composer = document.querySelector('.tw-chat>.tw-composer, .tw-composer-float');
  const gap = composer ? Math.max(0, Math.round(window.innerHeight - composer.getBoundingClientRect().bottom)) : 0;
  document.documentElement.style.setProperty('--tw-composer-gap', `${gap}px`);
}

type Hold = 'hover' | 'focus';
/** Presentational part, rendered statically in tests. Never takes focus; 撤销 is reachable with Tab. */
export function ToastView({ toast, onAction, onPause, onResume }: { toast: ToastValue; onAction?: () => void; onPause?: (hold: Hold) => void; onResume?: (hold: Hold) => void }) {
  return <div className="tw-toast" role="status" aria-live="polite" onMouseEnter={() => onPause?.('hover')} onMouseLeave={() => onResume?.('hover')}
    onFocus={() => onPause?.('focus')} onBlur={() => onResume?.('focus')}>
    <span>{toast.message}</span>
    {toast.action && <><span aria-hidden="true">·</span><button type="button" className="tw-toast-action" onClick={onAction}>{toast.action.label}</button></>}
  </div>;
}

export function ToastHost() {
  const toast = useSyncExternalStore(subscribe, () => current, () => null);
  const left = useRef(TOAST_MS);
  const started = useRef(0);
  const timer = useRef<number | undefined>(undefined);
  const holds = useRef(new Set<Hold>());
  const pause = (hold: Hold) => {
    if (!holds.current.size) { window.clearTimeout(timer.current); left.current -= Date.now() - started.current; }
    holds.current.add(hold);
  };
  const resume = (hold?: Hold) => {
    if (hold) holds.current.delete(hold);
    if (!toast || holds.current.size) return;
    const id = toast.id;
    started.current = Date.now();
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => dismissToast(id), Math.max(0, left.current));
  };
  useLayoutEffect(() => {
    if (!toast) return;
    measureComposerGap();
    window.addEventListener('scroll', measureComposerGap, { passive: true });
    window.addEventListener('resize', measureComposerGap);
    return () => { window.removeEventListener('scroll', measureComposerGap); window.removeEventListener('resize', measureComposerGap); };
  }, [toast?.id]);
  useEffect(() => {
    left.current = TOAST_MS;
    holds.current.clear();
    resume();
    return () => window.clearTimeout(timer.current);
  }, [toast?.id]);
  if (!toast) return null;
  return <ToastView key={toast.id} toast={toast} onPause={pause} onResume={resume}
    onAction={() => { dismissToast(toast.id); toast.action?.run(); }} />;
}
