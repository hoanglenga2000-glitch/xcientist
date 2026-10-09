import assert from "node:assert/strict";
import test from "node:test";

process.env.WORKSTATION_SESSION_SECRET = "test-public-session-secret-that-is-at-least-thirty-two-bytes";
process.env.WORKSTATION_ADMIN_USERNAME = "admin";

// @ts-ignore Node's strip-types test runner requires explicit TypeScript suffixes.
import { issuePublicSessionCookie, sessionPrincipal } from "./local-session.ts";
// @ts-ignore Node's strip-types test runner requires explicit TypeScript suffixes.
import { createPasswordHash, safePostLoginPath, verifyPublicCredentials } from "./public-auth.ts";

test("public password login issues a signed tenant-bound session", async () => {
  process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT = createPasswordHash("correct-password", Buffer.alloc(16, 7));
  const verified = await verifyPublicCredentials("ADMIN", "correct-password", "success-key");
  assert.equal(verified.ok, true);
  if (!verified.ok) return;
  assert.equal(verified.tenantId, "tenant_ba0ef9d3767f2fb385b856e0");

  const session = issuePublicSessionCookie(verified.username, verified.tenantId, 1_000_000);
  const principal = sessionPrincipal(session, 1_000_001);
  assert.equal(principal?.username, "admin");
  assert.equal(principal?.tenantId, verified.tenantId);
  assert.equal(principal?.authentication, "public_password");
  assert.equal(sessionPrincipal(`${session}x`, 1_000_001), null);
});

test("post-login redirect stays on the configured origin", () => {
  assert.equal(safePostLoginPath("/?page=assistant"), "/?page=assistant");
  assert.equal(safePostLoginPath("//attacker.example/path"), "/?page=assistant");
  assert.equal(safePostLoginPath("https://attacker.example/"), "/?page=assistant");
  assert.equal(safePostLoginPath("/safe\\escape"), "/?page=assistant");
});
