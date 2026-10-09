import { createHash } from "node:crypto";

export function tenantIdForUsername(username) {
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(username)) throw new Error("invalid_username");
  return `tenant_${createHash("sha256").update(`evomind.tenant.v1:${username.toLowerCase()}`).digest("hex").slice(0, 24)}`;
}

export function tenantPrincipal(principal) {
  if (!principal || principal.authentication !== "public_password") throw new Error("public_session_required");
  const tenantId = tenantIdForUsername(principal.username);
  if (principal.tenantId !== tenantId) throw new Error("tenant_session_mismatch");
  return { username: principal.username, tenantId };
}

export function assertTenantOwner(value, identity) {
  if (value.tenant_id !== identity.tenantId || value.owner_principal_id !== identity.username) {
    throw new Error("cross_tenant_resource_rejected");
  }
}
