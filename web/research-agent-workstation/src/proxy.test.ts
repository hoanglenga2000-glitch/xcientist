import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("./proxy.ts", import.meta.url), "utf8");

test("public health and login routes remain explicitly available on the trusted public origin", () => {
  const publicSet = source.indexOf("const PUBLIC_API_PATHS = new Set([");
  const publicPassThrough = source.indexOf("if (PUBLIC_API_PATHS.has(pathname)) return NextResponse.next();");
  const trustedHostBoundary = source.indexOf("if (!isTrustedHostHeader(host))");

  assert.ok(publicSet >= 0, "public route allowlist must remain explicit");
  assert.match(source, /"\/api\/healthz"/);
  assert.match(source, /"\/api\/auth\/login"/);
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
