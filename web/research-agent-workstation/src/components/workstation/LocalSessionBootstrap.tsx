"use client";

import { useEffect, useState } from "react";

const CSRF_STORAGE_KEY = "evomind.local.csrf.v1";

function bootstrapTokenFromFragment() {
  const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const token = fragment.get("bootstrap") ?? "";
  if (token) {
    fragment.delete("bootstrap");
    const remaining = fragment.toString();
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}${remaining ? `#${remaining}` : ""}`);
  }
  return token;
}

function installAuthenticatedFetch(csrfToken: string) {
  window.sessionStorage.setItem(CSRF_STORAGE_KEY, csrfToken);
  if ((window as typeof window & { __evomindFetchInstalled?: boolean }).__evomindFetchInstalled) return;
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const requestUrl = new URL(input instanceof Request ? input.url : String(input), window.location.href);
    const method = (init.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
    if (requestUrl.origin === window.location.origin && !["GET", "HEAD", "OPTIONS"].includes(method)) {
      const headers = new Headers(input instanceof Request ? input.headers : undefined);
      new Headers(init.headers).forEach((value, key) => headers.set(key, value));
      const inheritedBody = input instanceof Request ? input.body : null;
      const body = init.body === undefined && inheritedBody === null ? "{}" : init.body;
      if (!headers.has("content-type") && typeof body === "string") {
        headers.set("Content-Type", "application/json");
      }
      const latestCsrf = window.sessionStorage.getItem(CSRF_STORAGE_KEY) || csrfToken;
      headers.set("x-evomind-csrf", latestCsrf);
      init = { ...init, body, headers, credentials: "same-origin", cache: "no-store", redirect: "error" };
    }
    return nativeFetch(input, init);
  };
  (window as typeof window & { __evomindFetchInstalled?: boolean }).__evomindFetchInstalled = true;
}

export function LocalSessionBootstrap({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<"checking" | "ready" | "failed">("checking");

  useEffect(() => {
    let active = true;
    async function establish() {
      let csrf = window.sessionStorage.getItem(CSRF_STORAGE_KEY) ?? "";
      const status = await fetch("/api/session/status", { cache: "no-store", credentials: "same-origin" }).catch(() => null);
      if (status?.ok) {
        const payload = (await status.json()) as { csrf_token?: string };
        csrf = payload.csrf_token ?? csrf;
      } else {
        const token = bootstrapTokenFromFragment();
        if (!token) throw new Error("缺少一次性本地启动令牌，请通过 evomind open 重新打开工作站。");
        const response = await fetch("/api/session/bootstrap", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
          credentials: "same-origin",
        });
        const payload = (await response.json().catch(() => ({}))) as { csrf_token?: string; code?: string };
        if (!response.ok || !payload.csrf_token) throw new Error(payload.code ?? "本地会话初始化失败");
        csrf = payload.csrf_token;
      }
      if (!csrf) throw new Error("本地会话未返回 CSRF 令牌");
      installAuthenticatedFetch(csrf);
      if (active) setState("ready");
    }
    void establish().catch(() => active && setState("failed"));
    return () => { active = false; };
  }, []);

  if (state === "checking") {
    return <main className="flex min-h-screen items-center justify-center bg-surface text-sm text-ink-secondary">正在建立本地安全会话…</main>;
  }
  if (state === "failed") {
    return (
      <main className="flex min-h-screen items-center justify-center bg-surface p-6 text-ink">
        <section className="max-w-lg rounded-xl border border-warning/40 bg-surface-raised p-6 shadow-2xl">
          <h1 className="text-lg font-semibold">本地会话未建立</h1>
          <p className="mt-3 text-sm leading-6 text-ink-secondary">请关闭此标签页并运行 <code>evomind open</code>。工作站不会接受来自其他网页的本地 API 请求。</p>
        </section>
      </main>
    );
  }
  return children;
}
