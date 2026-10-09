import assert from "node:assert/strict";
import test from "node:test";

import { principalScopeHeaders, sessionCookieFromHeaders } from "./runtime-principal.ts";

const alice = { username: "alice", tenantId: `tenant_${"a".repeat(24)}`, authentication: "public_password" };
const ordinary = { verified: false, loopback: false };

test("runtime identity is server-derived, not copied from caller headers", () => {
  const headers = principalScopeHeaders(alice, ordinary, {
    "x-evomind-access-scope": "admin",
    "x-evomind-tenant-id": `tenant_${"b".repeat(24)}`,
    "x-evomind-principal-id": "bob",
    "Authorization": "Bearer fixture-service-token",
  });
  assert.equal(headers.get("x-evomind-access-scope"), "user.v1");
  assert.equal(headers.get("x-evomind-tenant-id"), alice.tenantId);
  assert.equal(headers.get("x-evomind-principal-id"), "alice");
  assert.equal(headers.get("Authorization"), "Bearer fixture-service-token");
});

test("missing or invalid principals cannot become unscoped service callers", () => {
  assert.throws(() => principalScopeHeaders(null, ordinary), /session_required/);
  assert.throws(() => principalScopeHeaders({ ...alice, tenantId: null }, ordinary), /invalid_principal_scope/);
  assert.throws(() => principalScopeHeaders({ ...alice, username: "a/b" }, ordinary), /invalid_principal_scope/);
  assert.throws(() => principalScopeHeaders(null, { verified: true, loopback: false }), /session_required/);
});

test("only verified loopback automation preserves administrative service calls", () => {
  const headers = principalScopeHeaders(null, { verified: true, loopback: true }, {
    "x-evomind-access-scope": "user.v1", "x-evomind-principal-id": "forged",
  });
  assert.equal(headers.has("x-evomind-access-scope"), false);
  assert.equal(headers.has("x-evomind-principal-id"), false);
});

test("local browser sessions remain scoped, not globally privileged", () => {
  const headers = principalScopeHeaders({ username: "local", tenantId: null, authentication: "local_bootstrap" }, ordinary);
  assert.equal(headers.get("x-evomind-access-scope"), "user.v1");
  assert.equal(headers.get("x-evomind-tenant-id"), "local");
});

test("session cookie extraction rejects ambiguity and ignores similarly named cookies", () => {
  assert.equal(sessionCookieFromHeaders(new Headers({ cookie: "other=1; session=abc.def; session_copy=bad" }), "session"), "abc.def");
  assert.equal(sessionCookieFromHeaders(new Headers(), "session"), undefined);
  assert.throws(() => sessionCookieFromHeaders(new Headers({ cookie: "session=a; session=b" }), "session"), /ambiguous_session_cookie/);
});
