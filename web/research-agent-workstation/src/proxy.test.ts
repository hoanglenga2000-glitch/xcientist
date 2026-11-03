import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { createRequire, Module } from "node:module";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(new URL("./proxy.ts", import.meta.url), "utf8");

test("public health and login routes remain explicitly available on the trusted public origin", () => {
  const publicSet = source.indexOf("const PUBLIC_API_PATHS = new Set([");
  const publicPassThrough = source.indexOf("if (PUBLIC_API_PATHS.has(pathname)) {");
  const trustedHostBoundary = source.indexOf("if (!isTrustedHostHeader(host))");

  assert.ok(publicSet >= 0, "public route allowlist must remain explicit");
  assert.match(source, /"\/api\/healthz"/);
  assert.match(source, /"\/api\/auth\/login"/);
  const allowlist = source.slice(publicSet, source.indexOf("]);", publicSet));
  assert.match(allowlist, /"\/api\/auth\/captcha"/, "logged-out users must be able to fetch the login captcha");
  assert.ok(publicPassThrough >= 0, "public routes must pass through");
  assert.ok(trustedHostBoundary >= 0, "all traffic must retain the trusted-host boundary");
  assert.match(source, /pathname === "\/api\/session\/bootstrap" && !isLoopbackHostHeader\(host\)/);
});

test("public pages redirect unauthenticated sessions while protected APIs require a host-valid session", () => {
  assert.match(source, /pathname === "\/login"/);
  assert.match(source, /principal\.authentication !== "public_password"/);
  assert.match(source, /NextResponse\.redirect\(loginUrl, 307\)/);
  assert.match(source, /isPublicHostHeader\(host\)[\s\S]*principal\?\.authentication === "public_password"/);
  assert.match(source, /validSessionCookie\(session\)/);
  assert.match(source, /isLoopbackHostHeader\(host\)[\s\S]*validLocalAutomationToken/);
});

// t238 C7b H1: behavioural checks — the real proxy and the real /api/healthz route, chained.
// Only the runtime fetch and the version identity are faked (they would need a live runtime).
const AUTOMATION_TOKEN = "automation-token-for-proxy-test";
process.env.WORKSTATION_SESSION_SECRET = "s".repeat(48);
process.env.WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH = createHash("sha256").update(AUTOMATION_TOKEN).digest("hex");
delete process.env.WORKSTATION_ACCOUNTS_DB;
delete process.env.WORKSTATION_PUBLIC_ORIGIN;

type Loader = { _resolveFilename: (...args: unknown[]) => string; _extensions: Record<string, (m: { _compile: (code: string, file: string) => void }, file: string) => void> };
const loader = Module as unknown as Loader;
const cjsRequire = createRequire(import.meta.url);
const ts = cjsRequire("typescript");
const srcPath = (p: string) => fileURLToPath(new URL(`./${p}`, import.meta.url));
const originalResolve = loader._resolveFilename;
loader._resolveFilename = function (name: unknown, ...rest: unknown[]) {
  const target = typeof name === "string" && name.startsWith("@/") ? srcPath(name.slice(2)) : name;
  return originalResolve.call(this, target, ...rest);
};
loader._extensions[".ts"] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, "utf8"), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename }).outputText, filename);
};
const fake = (p: string, exports: unknown) => {
  const file = srcPath(p);
  cjsRequire.cache[file] = { id: file, filename: file, loaded: true, exports } as unknown as NodeJS.Module;
};
const runtimeCalls: string[] = [];
fake("lib/server/assistant-runtime.ts", {
  assistantRuntimeFetch: async (pathname: string) => {
    runtimeCalls.push(pathname);
    return Response.json({ status: "ok", failed: [], components: { disk: { ok: true, free_bytes: 1 } } });
  },
});
fake("lib/server/runtime-version.ts", {
  runtimeVersionIdentity: async () => ({ ready: true, status: "ready", frontend_version: "1.0", commit_hash: "c", source_revision: "r",
    source_commit: "s", source_tree_sha256: "t", source_provenance: "p", build_id: "b", failures: [] }),
});
const { NextRequest } = cjsRequire("next/server");
const { proxy } = cjsRequire("./proxy.ts");
const healthz = cjsRequire("./app/api/healthz/route.ts");
const { expectedSessionCookie } = cjsRequire("./lib/server/local-session.ts");

const FORWARDED = "x-middleware-request-";
async function throughProxy(url: string, headers: Record<string, string>) {
  const request = new NextRequest(url, { headers: { host: new URL(url).host, ...headers } });
  const response: Response = await proxy(request);
  if (response.headers.get("x-middleware-next") !== "1") return { blocked: response, forwarded: null };
  // Rebuild what the route receives: the proxy's override set, or the client's own headers if none.
  const override = response.headers.get("x-middleware-override-headers");
  if (override === null) return { blocked: null, forwarded: new Request(url, { headers: request.headers }) };
  const forwarded = new Headers();
  for (const name of override.split(",").filter(Boolean)) forwarded.set(name, response.headers.get(FORWARDED + name) ?? "");
  return { blocked: null, forwarded: new Request(url, { headers: forwarded }) };
}
const DEEP_KEYS = ["components", "failed", "disk", "queue", "free_bytes"];

test("t238 C7b H1: a forged verified header on public /api/healthz?deep=1 is stripped and gets 401 without a runtime probe", async () => {
  runtimeCalls.length = 0;
  for (const host of ["127.0.0.1:8088", "localhost:8088"]) {
    const { blocked, forwarded } = await throughProxy(`http://${host}/api/healthz?deep=1`, {
      "x-evomind-local-automation-verified": "1",
      "x-evomind-tenant-id": "tenant_" + "a".repeat(24),
      "x-evomind-local-automation": "wrong-token",
    });
    assert.equal(blocked, null);
    assert.equal(forwarded!.headers.get("x-evomind-local-automation-verified"), null);
    assert.equal(forwarded!.headers.get("x-evomind-tenant-id"), null);
    assert.equal(forwarded!.headers.get("x-evomind-local-automation"), null);
    const response = await healthz.GET(forwarded);
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { ok: false, code: "session_required" });
  }
  // Defence in depth: even if a front proxy let the forged header reach the route, it is not enough.
  const direct = await healthz.GET(new Request("http://127.0.0.1:8088/api/healthz?deep=1", {
    headers: { host: "127.0.0.1:8088", "x-evomind-local-automation-verified": "1" } }));
  assert.equal(direct.status, 401);
  assert.deepEqual(runtimeCalls, []);
});

test("t238 C7b H1: other public paths never receive derived or raw automation headers", async () => {
  const { forwarded } = await throughProxy("http://127.0.0.1:8088/api/system/version", {
    "x-evomind-local-automation-verified": "1", "x-evomind-local-automation": AUTOMATION_TOKEN, "x-evomind-principal-id": "p" });
  for (const name of ["x-evomind-local-automation-verified", "x-evomind-local-automation", "x-evomind-principal-id"]) {
    assert.equal(forwarded!.headers.get(name), null);
  }
});

test("t238 C7b H1: legitimate loopback automation and the operator session still get the deep report", async () => {
  runtimeCalls.length = 0;
  const automation = await throughProxy("http://127.0.0.1:8088/api/healthz?deep=1", { "x-evomind-local-automation": AUTOMATION_TOKEN });
  assert.equal(automation.forwarded!.headers.get("x-evomind-local-automation-verified"), null);
  const viaToken = await healthz.GET(automation.forwarded);
  assert.equal(viaToken.status, 200);
  assert.equal((await viaToken.json()).status, "ok");
  const operator = await throughProxy("http://127.0.0.1:8088/api/healthz?deep=1", { cookie: `evomind_local_session=${expectedSessionCookie()}` });
  assert.equal((await healthz.GET(operator.forwarded)).status, 200);
  assert.deepEqual(runtimeCalls, ["/v1/health/deep", "/v1/health/deep"]);
  // A valid token from a non-loopback host is not automation.
  const spoofed = await healthz.GET(new Request("http://evil.example/api/healthz?deep=1", {
    headers: { host: "evil.example", "x-evomind-local-automation": AUTOMATION_TOKEN } }));
  assert.equal(spoofed.status, 401);
  // Non-public paths still receive the derived header exactly as before.
  const legacy = await throughProxy("http://127.0.0.1:8088/api/gpu/jobs", { "x-evomind-local-automation": AUTOMATION_TOKEN });
  assert.equal(legacy.forwarded!.headers.get("x-evomind-local-automation-verified"), "1");
  assert.equal(legacy.forwarded!.headers.get("x-evomind-local-automation"), null);
  const forgedLegacy = await throughProxy("http://127.0.0.1:8088/api/gpu/jobs", { "x-evomind-local-automation-verified": "1" });
  assert.equal(forgedLegacy.blocked!.status, 401);
});

test("t238 C7b H1: shallow /api/healthz stays 200 unauthenticated and leaks no deep fields", async () => {
  runtimeCalls.length = 0;
  const { forwarded } = await throughProxy("http://127.0.0.1:8088/api/healthz", { "x-evomind-local-automation-verified": "1" });
  const response = await healthz.GET(forwarded);
  assert.equal(response.status, 200);
  const body = await response.json();
  for (const key of DEEP_KEYS) assert.equal(key in body, false, key);
  assert.deepEqual(runtimeCalls, []);
});
