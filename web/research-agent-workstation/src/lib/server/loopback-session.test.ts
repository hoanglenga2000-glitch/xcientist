import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error The runtime helper is intentionally plain ESM for Node contract scripts.
import { createLoopbackSession, LoopbackSessionError } from "../../../scripts/lib/loopback-session.mjs";

const baseUrl = "http://127.0.0.1:18088";
const bootstrapToken = "bootstrap-token-with-at-least-24-characters";
const csrfToken = "csrf-token-with-at-least-24-characters";

function hasSessionErrorCode(error: unknown, code: string) {
  return error instanceof LoopbackSessionError
    && typeof error === "object"
    && error !== null
    && "code" in error
    && (error as { code?: unknown }).code === code;
}

test("loopback helper establishes a strict cookie session and keeps auth state in its closure", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetchImpl = async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    if (url.endsWith("/api/session/bootstrap")) {
      return new Response(JSON.stringify({ ok: true, csrf_token: csrfToken }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Set-Cookie": "evomind_local_session=session-value; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200",
        },
      });
    }
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("cookie"), "evomind_local_session=session-value");
    if (url.endsWith("/api/session/status")) {
      return new Response(JSON.stringify({ ok: true, authenticated: true, csrf_token: csrfToken }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  const session = await createLoopbackSession({ baseUrl, bootstrapToken, fetchImpl });
  assert.deepEqual(Object.keys(session).sort(), ["close", "origin", "request"]);
  assert.doesNotMatch(JSON.stringify(session), /session-value|csrf-token|bootstrap-token/);
  assert.equal((await session.request("/api/protected", { cache: "force-cache", redirect: "follow" })).status, 200);
  assert.equal(calls.at(-1)?.init?.cache, "no-store", "callers must not weaken authenticated cache handling");
  assert.equal(calls.at(-1)?.init?.redirect, "error", "callers must not permit authenticated redirects");
  assert.equal((await session.request("/api/mutation", { method: "POST" })).status, 200);
  assert.equal(calls.at(-1)?.init?.body, "{}", "empty mutations must receive a bounded JSON envelope");
  assert.equal(new Headers(calls.at(-1)?.init?.headers).get("content-type"), "application/json");
  const callsBeforeClose = calls.length;
  session.close();
  await assert.rejects(
    () => session.request("/api/mutation", { method: "POST", body: "{}" }),
    (error: unknown) => hasSessionErrorCode(error, "csrf_unavailable"),
  );
  assert.equal(calls.length, callsBeforeClose, "closed mutation must fail before fetch");
});

test("loopback helper rejects a bootstrap cookie without strict attributes", async () => {
  const fetchImpl = async () => new Response(JSON.stringify({ ok: true, csrf_token: csrfToken }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": "evomind_local_session=session-value; SameSite=Lax; Path=/",
    },
  });
  await assert.rejects(
    createLoopbackSession({ baseUrl, bootstrapToken, fetchImpl }),
    (error: unknown) => hasSessionErrorCode(error, "cookie_missing_httponly"),
  );
});
