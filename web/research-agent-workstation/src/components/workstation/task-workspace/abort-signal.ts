/**
 * t237 C6d: `AbortSignal.any` is missing before Safari 17.4 (and `AbortSignal.timeout` before 16).
 * The only place the guarded call is spelled; types.ts (taskRequest) imports it with an explicit .ts extension. Never throws.
 */
export function timeoutSignal(ms: number, parent?: AbortSignal): AbortSignal {
  const timeout = typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(ms) : null;
  if (timeout && !parent) return timeout;
  if (timeout && parent && typeof AbortSignal.any === 'function') return AbortSignal.any([parent, timeout]);
  const controller = new AbortController();
  const abort = () => { if (!controller.signal.aborted) controller.abort(); };
  if (parent?.aborted) { abort(); return controller.signal; }
  parent?.addEventListener('abort', abort, { once: true });
  if (timeout) timeout.addEventListener('abort', abort, { once: true });
  else {
    const timer = setTimeout(abort, ms);
    controller.signal.addEventListener('abort', () => clearTimeout(timer), { once: true });
  }
  return controller.signal;
}
