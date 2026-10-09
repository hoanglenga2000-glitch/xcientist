const KEY = /^[A-Za-z0-9_-]{8,128}$/;

function pending(sessionId: string, ownerKey?: string | null): string | null {
  const url = new URL(window.location.href);
  if (url.searchParams.get("creation_conversation") === sessionId) {
    const value = url.searchParams.get("creation_request");
    if (value && KEY.test(value)) return value;
  }
  if (ownerKey) {
    try {
      const value = JSON.parse(window.sessionStorage.getItem(`${ownerKey}:creation`) ?? "null") as { key?: string; conversation?: string } | null;
      if (value?.conversation === sessionId && value.key && KEY.test(value.key)) return value.key;
    } catch { /* The URL is the durable fallback; never store request bodies. */ }
  }
  return null;
}

export function ensureCreationKey(sessionId: string, ownerKey?: string | null): string {
  const key = pending(sessionId, ownerKey) ?? `request_${crypto.randomUUID()}`;
  const url = new URL(window.location.href);
  url.searchParams.set("creation_request", key);
  url.searchParams.set("creation_conversation", sessionId);
  url.searchParams.set("conversation", sessionId);
  window.history.replaceState(null, "", url);
  if (ownerKey) {
    try { window.sessionStorage.setItem(`${ownerKey}:creation`, JSON.stringify({ key, conversation: sessionId })); } catch { /* URL fallback remains. */ }
  }
  return key;
}

export function finishCreation(sessionId: string, ownerKey?: string | null): void {
  const url = new URL(window.location.href);
  if (url.searchParams.get("creation_conversation") === sessionId) {
    url.searchParams.delete("creation_request");
    url.searchParams.delete("creation_conversation");
    window.history.replaceState(null, "", url);
  }
  if (ownerKey) {
    try { window.sessionStorage.removeItem(`${ownerKey}:creation`); } catch { /* restricted storage */ }
  }
}

export async function recoverCreation<T extends { id: string }>(sessionId: string, ownerKey?: string | null, signal?: AbortSignal): Promise<T | null> {
  const key = pending(sessionId, ownerKey);
  if (!key) return null;
  const response = await fetch(`/api/assistant/run-requests/${encodeURIComponent(key)}`, { cache: "no-store", signal });
  if (response.status === 404) return null;
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(body.error ?? "run_request_recovery_unavailable");
  }
  const run = await response.json() as T;
  if (!run.id) throw new Error("run_request_recovery_invalid");
  return run;
}
