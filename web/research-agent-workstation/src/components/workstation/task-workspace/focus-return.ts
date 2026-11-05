// t238 C7e M-3 (Critiquito): closing a modal dialog (Esc, 取消, click outside, or the confirm button) returns focus to the
// button that opened it. Without this the dialog's focused button unmounts and focus falls to the page body.
// The decision logic is plain functions over DOM-like objects so it is node-testable without a browser.
// HARD (VERIFY 86ebf36): every tw-dialog-backdrop onMouseDown MUST event.preventDefault() before closing. Otherwise
// useReturnFocus restores to the trigger during mousedown, then Chrome's default mousedown focus fixup moves focus to
// <main>. jsdom/node cannot see that; the gate is ui/esc_live_real.mjs (Esc/取消/scrim × 2 dialogs = 6/6).
import { useEffect, useState } from 'react';

export type FocusLike = {
  focus: (options?: { preventScroll?: boolean }) => void;
  isConnected?: boolean;
  contains?: (other: unknown) => boolean;
  closest?: (selector: string) => FocusLike | null;
};
export type DocLike = { activeElement: FocusLike | null; body: FocusLike | null; documentElement?: FocusLike | null };

const ACTIVATOR = 'button, a[href], [role="menuitem"], [role="option"], [tabindex]';
let lastActivator: FocusLike | null = null;

/** A pointer click on a button does not focus it in every browser (Safari), so the last activated control is remembered. */
export function rememberActivator(target: unknown): void {
  const node = target as FocusLike | null;
  lastActivator = node && typeof node.closest === 'function' ? node.closest(ACTIVATOR) : null;
}

function isPageLevel(doc: DocLike, element: FocusLike | null): boolean {
  return !element || element === doc.body || element === (doc.documentElement ?? null);
}

/** The control that has focus (or was just activated) when a dialog opens; null when only the page body does. */
export function captureTrigger(doc: DocLike): FocusLike | null {
  if (!isPageLevel(doc, doc.activeElement)) return doc.activeElement;
  return lastActivator && lastActivator.isConnected !== false ? lastActivator : null;
}

/**
 * Give focus back to the opener after the dialog is gone. The first connected target wins (an explicit target the caller
 * knows, e.g. a row's 更多 button, before the captured one). Focus is only moved when it is lost (on the page body, on a
 * node that left the page, or still inside the dialog): a caller that already moved focus somewhere deliberate keeps it.
 */
export function restoreFocus(doc: DocLike, targets: Array<FocusLike | null | undefined>, dialog?: FocusLike | null): boolean {
  const target = targets.find(item => item && item.isConnected !== false);
  if (!target) return false;
  const active = doc.activeElement;
  const lost = isPageLevel(doc, active) || active?.isConnected === false || Boolean(dialog && active && dialog.contains?.(active));
  if (!lost) return false;
  target.focus({ preventScroll: true });
  return true;
}

/** Call in a dialog component: remembers the opener on mount and focuses it again on unmount (every way the dialog closes). */
export function useReturnFocus(explicit?: () => HTMLElement | null): void {
  const [captured] = useState<FocusLike | null>(() => (typeof document === 'undefined' ? null : captureTrigger(document as unknown as DocLike)));
  useEffect(() => () => {
    if (typeof document === 'undefined') return;
    restoreFocus(document as unknown as DocLike, [explicit?.() as FocusLike | null | undefined, captured]);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
}

/** Installed once in the browser; capture phase so it runs before the click handler that opens the dialog. */
if (typeof document !== 'undefined') document.addEventListener('click', event => rememberActivator(event.target), true);
