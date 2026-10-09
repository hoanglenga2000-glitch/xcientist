import { ensureEvoMindRuntime } from "@/lib/server/evomind-runtime";

const RESPONSE_HEADERS = [
  "content-type",
  "content-disposition",
  "cache-control",
  "x-artifact-bytes",
  "x-artifact-sha256",
  "x-content-type-options",
  "content-security-policy",
] as const;

export async function assistantRuntimeFetch(pathname: string, init: RequestInit = {}) {
  const runtime = await ensureEvoMindRuntime();
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${runtime.token}`);
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  return fetch(`${runtime.baseUrl}${pathname}`, {
    ...init,
    headers,
    cache: "no-store",
    redirect: "error",
  });
}

export async function proxyAssistantRuntime(request: Request, pathname: string) {
  try {
    const headers = new Headers();
    for (const name of ["content-type", "accept", "x-chunk-sha256"] as const) {
      const value = request.headers.get(name);
      if (value) headers.set(name, value);
    }
    const response = await assistantRuntimeFetch(pathname, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : await request.arrayBuffer(),
      signal: request.signal,
    });
    const responseHeaders = new Headers();
    for (const name of RESPONSE_HEADERS) {
      const value = response.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set("Cache-Control", responseHeaders.get("Cache-Control") ?? "private, no-store");
    return new Response(response.body, { status: response.status, headers: responseHeaders });
  } catch {
    return Response.json(
      { ok: false, error: "runtime_not_ready" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}

export async function runtimeJson<T>(pathname: string, init: RequestInit = {}): Promise<T> {
  const response = await assistantRuntimeFetch(pathname, init);
  const payload = await response.json().catch(() => ({})) as T & { error?: string; message?: string };
  if (!response.ok) throw new Error(payload.message ?? payload.error ?? `runtime_${response.status}`);
  return payload;
}
import { scopedRuntimeFetch as fetch } from "./scoped-runtime-fetch";
