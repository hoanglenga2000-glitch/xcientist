import { headers as requestHeaders } from "next/headers";

import { isLoopbackHostHeader } from "@/lib/security/request-boundary";
import {
  LOCAL_AUTOMATION_VERIFIED_HEADER,
  SESSION_COOKIE,
  sessionPrincipal,
} from "./local-session";
import { principalScopeHeaders, sessionCookieFromHeaders } from "./runtime-principal";

async function runtimeFetch(
  input: Parameters<typeof globalThis.fetch>[0],
  init: Parameters<typeof globalThis.fetch>[1],
  readOnlyProxy: boolean,
): Promise<Response> {
  const destination = new URL(input instanceof Request ? input.url : String(input));
  if (
    destination.protocol !== "http:"
    || !["127.0.0.1", "localhost", "[::1]"].includes(destination.hostname)
    || destination.username || destination.password
    || !destination.pathname.startsWith("/v1/")
  ) throw new Error("runtime_destination_rejected");

  const inbound = await requestHeaders();
  const verifiedAutomation = inbound.get(LOCAL_AUTOMATION_VERIFIED_HEADER) === "1";
  const loopback = isLoopbackHostHeader(inbound.get("host"));
  const principal = verifiedAutomation && loopback ? null : sessionPrincipal(
    sessionCookieFromHeaders(new Headers(inbound), SESSION_COOKIE),
  );
  const forwarded = new Headers(input instanceof Request ? input.headers : undefined);
  new Headers(init?.headers).forEach((value, key) => forwarded.set(key, value));
  const scoped = principalScopeHeaders(principal, { verified: verifiedAutomation, loopback }, forwarded);
  const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  if (readOnlyProxy && scoped.has("x-evomind-access-scope") && !["GET", "HEAD"].includes(method)) {
    throw new Error("managed_assistant_endpoint_required");
  }
  // A redirect must never forward the runtime bearer or user identity elsewhere.
  return globalThis.fetch(input, { ...init, headers: scoped, redirect: "error" });
}

export const scopedRuntimeFetch: typeof globalThis.fetch = (input, init) => runtimeFetch(input, init, false);
export const scopedRuntimeReadFetch: typeof globalThis.fetch = (input, init) => runtimeFetch(input, init, true);
