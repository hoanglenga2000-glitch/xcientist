import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import net from "node:net";

const baseUrl = new URL(process.env.REPORT_TEST_BASE_URL ?? "http://127.0.0.1:18088");
assert.equal(baseUrl.protocol, "http:");
assert.equal(baseUrl.hostname, "127.0.0.1");
assert.match(baseUrl.port, /^\d+$/);

let bootstrapToken = process.env.REPORT_TEST_BOOTSTRAP_TOKEN ?? "";
delete process.env.REPORT_TEST_BOOTSTRAP_TOKEN;
assert.match(bootstrapToken, /^[A-Za-z0-9_-]{24,256}$/);

const origin = baseUrl.origin;
const host = baseUrl.host;
const checks = [];
const record = (name, evidence = {}) => checks.push({ name, passed: true, ...evidence });

async function responseJson(response, expectedStatus, expectedCode = undefined) {
  assert.equal(response.status, expectedStatus);
  assert.match(response.headers.get("cache-control") ?? "", /no-store/i);
  const payload = await response.json();
  if (expectedCode !== undefined) assert.equal(payload.code, expectedCode);
  return payload;
}

function rawHttp(request, { label = "unlabeled", allowNoHandshake = false } = {}) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host: "127.0.0.1", port: Number(baseUrl.port) });
    let response = Buffer.alloc(0);
    let settled = false;
    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      if (error) reject(error);
      else resolve(value);
    };
    socket.setTimeout(7000, () => {
      if (allowNoHandshake) finish(null, { status: null, headers: {}, transport: "no_handshake" });
      else finish(new Error(`raw HTTP probe timed out at ${label}`));
    });
    socket.on("error", (error) => finish(error));
    socket.on("connect", () => socket.write(request));
    socket.on("data", (chunk) => {
      response = Buffer.concat([response, chunk]);
      if (response.length > 1024 * 1024) return finish(new Error("raw HTTP response exceeded evidence bound"));
      const headerEnd = response.indexOf("\r\n\r\n");
      if (headerEnd < 0) return;
      const headerText = response.subarray(0, headerEnd).toString("latin1");
      const lines = headerText.split("\r\n");
      const statusMatch = lines[0].match(/^HTTP\/1\.[01]\s+(\d{3})\b/);
      if (!statusMatch) return finish(new Error("raw HTTP response has no status line"));
      const headers = Object.fromEntries(lines.slice(1).flatMap((line) => {
        const index = line.indexOf(":");
        return index > 0 ? [[line.slice(0, index).trim().toLowerCase(), line.slice(index + 1).trim()]] : [];
      }));
      finish(null, { status: Number(statusMatch[1]), headers });
    });
    socket.on("end", () => {
      if (settled) return;
      if (allowNoHandshake) finish(null, { status: null, headers: {}, transport: "closed_without_handshake" });
      else finish(new Error(`raw HTTP connection ended before headers at ${label}`));
    });
  });
}

function rawPost(pathname, headers, body = "", label = "raw_post") {
  const headerLines = Object.entries(headers).map(([key, value]) => `${key}: ${value}`).join("\r\n");
  return rawHttp(`POST ${pathname} HTTP/1.1\r\n${headerLines}\r\nConnection: close\r\n\r\n${body}`, { label });
}

function authHeaders(cookie, extra = {}) {
  return { Cookie: cookie, ...extra };
}

// Public shell and health are the only intentionally unauthenticated surfaces.
const shell = await fetch(new URL("/?page=assistant", baseUrl), { cache: "no-store", redirect: "error" });
assert.equal(shell.status, 200);
const shellHeaders = {
  csp: shell.headers.get("content-security-policy") ?? "",
  referrerPolicy: shell.headers.get("referrer-policy") ?? "",
  permissionsPolicy: shell.headers.get("permissions-policy") ?? "",
  nosniff: shell.headers.get("x-content-type-options") ?? "",
  frame: shell.headers.get("x-frame-options") ?? "",
  coop: shell.headers.get("cross-origin-opener-policy") ?? "",
  corp: shell.headers.get("cross-origin-resource-policy") ?? "",
  cache: shell.headers.get("cache-control") ?? "",
};
assert.match(shellHeaders.csp, /default-src 'self'/);
assert.match(shellHeaders.csp, /frame-ancestors 'none'/);
assert.match(shellHeaders.csp, /object-src 'none'/);
assert.match(shellHeaders.csp, /script-src 'self' 'nonce-[^']+'/);
assert.doesNotMatch(shellHeaders.csp, /script-src[^;]*'unsafe-inline'/);
assert.equal(shellHeaders.referrerPolicy, "no-referrer");
assert.match(shellHeaders.permissionsPolicy, /camera=\(\)/);
assert.equal(shellHeaders.nosniff, "nosniff");
assert.equal(shellHeaders.frame, "DENY");
assert.equal(shellHeaders.coop, "same-origin");
assert.equal(shellHeaders.corp, "same-origin");
assert.match(shellHeaders.cache, /no-store/);
assert.equal(shell.headers.get("access-control-allow-origin"), null);
const shellHtml = await shell.text();
const staticPath = shellHtml.match(/\bsrc="(\/_next\/static\/[^"]+\.js)"/)?.[1] ?? "";
assert.ok(staticPath);
const staticResponse = await fetch(new URL(staticPath, baseUrl), { cache: "no-store" });
assert.equal(staticResponse.status, 200);
assert.match(staticResponse.headers.get("cache-control") ?? "", /immutable/);
assert.equal(staticResponse.headers.get("x-content-type-options"), "nosniff");
record("shell_security_headers_and_static_cache", { status: shell.status, static_status: staticResponse.status });

const health = await fetch(new URL("/api/healthz", baseUrl), { cache: "no-store" });
const healthPayload = await health.json();
assert.equal(health.status, 200);
assert.equal(healthPayload.status, "ready");
assert.equal(healthPayload.version, "0.3.0");
assert.equal(health.headers.get("access-control-allow-origin"), null);
record("minimal_public_health", { status: health.status, version: healthPayload.version });

for (const endpoint of ["/api/workstation-summary", "/api/runtime/health", "/api/scientist/stream/events"]) {
  const response = await fetch(new URL(endpoint, baseUrl), { cache: "no-store", redirect: "error" });
  await responseJson(response, 401, "session_required");
}
record("protected_api_runtime_and_sse_require_session", { endpoints: 3 });

const unauthenticatedUpgrade = await rawHttp(
  `GET /api/workstation-summary HTTP/1.1\r\nHost: ${host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ZXZvbWluZC1xYS1wcm9iZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n`,
  { label: "unauthenticated_websocket", allowNoHandshake: true },
);
assert.notEqual(unauthenticatedUpgrade.status, 101);
record("websocket_upgrade_not_unauthenticated", { status: unauthenticatedUpgrade.status });

const bootstrapEndpoint = new URL("/api/session/bootstrap", baseUrl);
const bootstrapBody = JSON.stringify({ token: bootstrapToken });

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: bootstrapBody,
}), 403, "origin_rejected");
record("bootstrap_missing_origin_rejected");

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json", Origin: "http://evil.invalid" }, body: bootstrapBody,
}), 403, "origin_rejected");
record("bootstrap_wrong_origin_rejected_without_consuming_token");

const wrongHostBootstrap = await rawPost("/api/session/bootstrap", {
  Host: "evil.invalid",
  Origin: origin,
  "Content-Type": "application/json",
  "Content-Length": Buffer.byteLength(bootstrapBody),
}, bootstrapBody, "bootstrap_wrong_host");
assert.equal(wrongHostBootstrap.status, 403);
record("bootstrap_wrong_host_rejected", { status: wrongHostBootstrap.status });

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "text/plain", Origin: origin }, body: bootstrapBody,
}), 415, "unsupported_content_type");
record("bootstrap_json_only");

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json", Origin: origin }, body: "{",
}), 400, "invalid_json");
record("bootstrap_malformed_json_rejected");

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json", Origin: origin }, body: JSON.stringify({ token: bootstrapToken, padding: "x".repeat(4096) }),
}), 413, "body_too_large");
record("bootstrap_body_limit_enforced");

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json", Origin: origin }, body: JSON.stringify({ token: "test-invalid-bootstrap-token-value-000000000000" }),
}), 401, "invalid_bootstrap");
record("invalid_bootstrap_token_rejected_without_consuming_real_token");

const bootstrap = await fetch(bootstrapEndpoint, {
  method: "POST",
  headers: { "Content-Type": "application/json", Origin: origin },
  body: bootstrapBody,
  cache: "no-store",
  redirect: "error",
});
assert.equal(bootstrap.status, 200);
assert.match(bootstrap.headers.get("cache-control") ?? "", /no-store/);
const setCookie = bootstrap.headers.get("set-cookie") ?? "";
assert.match(setCookie, /^evomind_local_session=/i);
assert.match(setCookie, /;\s*HttpOnly/i);
assert.match(setCookie, /;\s*SameSite=Strict/i);
assert.match(setCookie, /;\s*Path=\//i);
assert.match(setCookie, /;\s*Max-Age=43200/i);
assert.doesNotMatch(setCookie, /;\s*Domain=/i);
const sessionCookie = setCookie.split(";", 1)[0];
const sessionCookieValue = sessionCookie.split("=", 2)[1] ?? "";
assert.match(sessionCookieValue, /^[A-Za-z0-9_-]{24,256}$/);
const bootstrapPayload = await bootstrap.json();
const csrf = String(bootstrapPayload.csrf_token ?? "");
assert.match(csrf, /^[A-Za-z0-9_-]{24,256}$/);
assert.equal(bootstrapPayload.expires_in, 43200);
record("bootstrap_sets_strict_host_only_session", { status: bootstrap.status, max_age: bootstrapPayload.expires_in });

await responseJson(await fetch(bootstrapEndpoint, {
  method: "POST", headers: { "Content-Type": "application/json", Origin: origin }, body: bootstrapBody,
}), 409, "bootstrap_consumed");
record("bootstrap_token_single_use");

const tampered = await fetch(new URL("/api/session/status", baseUrl), {
  headers: { Cookie: "evomind_local_session=tampered-session-value-000000000000" }, cache: "no-store",
});
await responseJson(tampered, 401, "session_required");
record("tampered_session_cookie_rejected");

const sessionStatus = await fetch(new URL("/api/session/status", baseUrl), {
  headers: authHeaders(sessionCookie), cache: "no-store",
});
assert.equal(sessionStatus.status, 200);
const sessionStatusPayload = await sessionStatus.json();
assert.equal(sessionStatusPayload.authenticated, true);
assert.equal(sessionStatusPayload.csrf_token, csrf);
record("authenticated_session_status");

const summary = await fetch(new URL("/api/workstation-summary", baseUrl), {
  headers: authHeaders(sessionCookie), cache: "no-store",
});
assert.equal(summary.status, 200);
assert.match(summary.headers.get("cache-control") ?? "", /no-store/);
const summaryText = await summary.text();
const summaryPayload = JSON.parse(summaryText);
const summaryStrings = [];
const collectSummaryStrings = (value) => {
  if (typeof value === "string") summaryStrings.push(value);
  else if (Array.isArray(value)) value.forEach(collectSummaryStrings);
  else if (value && typeof value === "object") Object.values(value).forEach(collectSummaryStrings);
};
collectSummaryStrings(summaryPayload);
const hasLocalAbsolutePath = summaryStrings.some((value) => (
  /(?:^|[\s"'`=(])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])/.test(value)
  || /(?:^|[\s"'`=(])\/(?:home|Users|root|hpc2hdd|mnt)\//i.test(value)
));
if (hasLocalAbsolutePath) throw new Error("workstation summary contains a local absolute path");
if (/(?:47\.97\.124\.121|43\.130\.57\.88|119\.3\.164\.120|10\.120\.\d+\.\d+)/.test(summaryText)) {
  throw new Error("workstation summary contains a private infrastructure address");
}
if (/(?:host_uuid|gpu_uuid|ssh_route|proxy_route|dpapi|private_key|authorization)["']?\s*:/i.test(summaryText)) {
  throw new Error("workstation summary contains a private infrastructure field");
}
if (/\bBearer\s+[A-Za-z0-9._~+/-]{16,}/i.test(summaryText)) {
  throw new Error("workstation summary contains an authorization value");
}
record("workstation_summary_client_safe_projection", { bytes: Buffer.byteLength(summaryText) });

const runtimeHealth = await fetch(new URL("/api/runtime/health", baseUrl), {
  headers: authHeaders(sessionCookie), cache: "no-store",
});
assert.equal(runtimeHealth.status, 200);
const runtimeHealthText = await runtimeHealth.text();
assert.doesNotMatch(runtimeHealthText, /runtime\.token|Bearer|Authorization/i);
assert.equal(JSON.parse(runtimeHealthText).status, "ready");
record("authenticated_runtime_proxy_redacts_bearer", { status: runtimeHealth.status });

const traversal = await fetch(new URL("/api/evolution/state?task_id=..%2F..%2Fescape", baseUrl), {
  headers: authHeaders(sessionCookie), cache: "no-store",
});
assert.ok([400, 403, 404].includes(traversal.status));
record("path_traversal_rejected", { status: traversal.status });

const probePath = "/api/__evomind_security_probe_missing__";
async function mutation(extraHeaders = {}, body = "{}") {
  return fetch(new URL(probePath, baseUrl), {
    method: "POST",
    headers: authHeaders(sessionCookie, { "Content-Type": "application/json", Origin: origin, "x-evomind-csrf": csrf, ...extraHeaders }),
    body,
    redirect: "error",
  });
}

await responseJson(await mutation({ Origin: "http://evil.invalid" }), 403, "origin_rejected");
record("mutation_wrong_origin_rejected");
await responseJson(await mutation({ Origin: "" }), 403, "origin_rejected");
record("mutation_missing_origin_rejected");
await responseJson(await mutation({ "x-evomind-csrf": "invalid-csrf-value-000000000000" }), 403, "csrf_rejected");
record("mutation_wrong_csrf_rejected");
await responseJson(await mutation({ "x-evomind-csrf": "" }), 403, "csrf_rejected");
record("mutation_missing_csrf_rejected");
await responseJson(await mutation({ "Content-Type": "text/plain" }, "probe"), 415, "unsupported_content_type");
record("mutation_json_only");

const wrongHostMutationBody = "{}";
const wrongHostMutation = await rawPost(probePath, {
  Host: "evil.invalid",
  Origin: origin,
  Cookie: sessionCookie,
  "x-evomind-csrf": csrf,
  "Content-Type": "application/json",
  "Content-Length": Buffer.byteLength(wrongHostMutationBody),
}, wrongHostMutationBody, "mutation_wrong_host");
assert.equal(wrongHostMutation.status, 403);
record("mutation_wrong_host_rejected", { status: wrongHostMutation.status });

const missingLengthMutation = await rawPost(probePath, {
  Host: host,
  Origin: origin,
  Cookie: sessionCookie,
  "x-evomind-csrf": csrf,
  "Content-Type": "application/json",
}, "", "mutation_missing_content_length");
assert.equal(missingLengthMutation.status, 411);
record("mutation_content_length_required", { status: missingLengthMutation.status });

const oversizedBody = JSON.stringify({ padding: "x".repeat(16 * 1024 * 1024) });
const oversizedMutation = await mutation({}, oversizedBody);
await responseJson(oversizedMutation, 413, "body_too_large");
record("mutation_body_limit_enforced_before_route", { status: oversizedMutation.status });

const validEnvelope = await mutation();
assert.equal(validEnvelope.status, 404);
assert.equal(validEnvelope.headers.get("access-control-allow-origin"), null);
record("valid_mutation_envelope_reaches_route", { status: validEnvelope.status });

const preflightUnauth = await fetch(new URL(probePath, baseUrl), { method: "OPTIONS" });
await responseJson(preflightUnauth, 401, "session_required");
const preflightAuth = await fetch(new URL(probePath, baseUrl), { method: "OPTIONS", headers: authHeaders(sessionCookie) });
assert.equal(preflightAuth.status, 404);
assert.equal(preflightAuth.headers.get("access-control-allow-origin"), null);
record("cors_disabled_and_options_authenticated", { unauthenticated: preflightUnauth.status, authenticated: preflightAuth.status });

const sseController = new AbortController();
const sse = await fetch(new URL("/api/scientist/stream/events", baseUrl), {
  headers: authHeaders(sessionCookie, { Accept: "text/event-stream" }),
  cache: "no-store",
  signal: sseController.signal,
});
assert.equal(sse.status, 200);
assert.match(sse.headers.get("content-type") ?? "", /^text\/event-stream/i);
const sseReader = sse.body.getReader();
const firstSseChunk = await Promise.race([
  sseReader.read(),
  new Promise((_, reject) => setTimeout(() => reject(new Error("authenticated SSE emitted no bounded first event")), 7000)),
]);
assert.equal(firstSseChunk.done, false);
assert.match(new TextDecoder().decode(firstSseChunk.value), /event:\s*(?:snapshot|heartbeat|scientist_event)/);
sseController.abort();
await sseReader.cancel().catch(() => undefined);
record("authenticated_sse_stream_and_abort", { status: sse.status });

const authenticatedUpgrade = await rawHttp(
  `GET /api/workstation-summary HTTP/1.1\r\nHost: ${host}\r\nCookie: ${sessionCookie}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ZXZvbWluZC1xYS1wcm9iZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n`,
  { label: "authenticated_websocket", allowNoHandshake: true },
);
assert.notEqual(authenticatedUpgrade.status, 101);
record("websocket_upgrade_has_no_uncontrolled_channel", { status: authenticatedUpgrade.status });

const runtimeDir = process.env.WORKSTATION_RUNTIME_DIR ?? "";
assert.ok(runtimeDir);
const logNames = (await fs.readdir(runtimeDir)).filter((name) => name.endsWith(".log"));
let logText = "";
for (const name of logNames) {
  const file = `${runtimeDir}/${name}`;
  const stat = await fs.stat(file);
  assert.ok(stat.size <= 16 * 1024 * 1024);
  logText += await fs.readFile(file, "utf8");
}
assert.equal(logText.includes(bootstrapToken), false);
assert.equal(logText.includes(sessionCookieValue), false);
assert.equal(logText.includes(csrf), false);
record("runtime_logs_do_not_contain_session_secrets", { files_scanned: logNames.length });

bootstrapToken = "";

console.log(JSON.stringify({
  ok: true,
  schema: "evomind.localhost_security_contract.v1",
  base_url: origin,
  checks: checks.length,
  results: checks,
  secret_values_recorded: false,
  mutations_with_product_side_effects: 0,
  run_created: false,
  grader_invoked: false,
}, null, 2));
