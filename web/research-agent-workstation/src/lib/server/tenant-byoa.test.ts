import assert from "node:assert/strict";
import test from "node:test";
import { assertTenantOwner, tenantIdForUsername, tenantPrincipal } from "./tenant-identity.mjs";

test("public users receive stable distinct tenant identities", () => {
  const a = tenantIdForUsername("admin"); const b = tenantIdForUsername("researcher2");
  assert.match(a, /^tenant_[a-f0-9]{24}$/); assert.notEqual(a, b);
  assert.deepEqual(tenantPrincipal({ username: "admin", tenantId: a, authentication: "public_password" }), { username: "admin", tenantId: a });
});
test("session tenant drift and cross-tenant resources fail closed", () => {
  const a = { username: "alice", tenantId: tenantIdForUsername("alice") };
  assert.throws(() => tenantPrincipal({ username: "alice", tenantId: tenantIdForUsername("bob"), authentication: "public_password" }), /tenant_session_mismatch/);
  assert.throws(() => assertTenantOwner({ tenant_id: tenantIdForUsername("bob"), owner_principal_id: "bob" }, a), /cross_tenant/);
  assert.doesNotThrow(() => assertTenantOwner({ tenant_id: a.tenantId, owner_principal_id: a.username }, a));
});

