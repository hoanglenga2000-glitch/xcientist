export type TenantIdentity = { username: string; tenantId: string };
export function tenantIdForUsername(username: string): string;
export function tenantPrincipal(principal: { username: string; tenantId: string | null; authentication: string } | null): TenantIdentity;
export function assertTenantOwner(value: Record<string, unknown>, identity: TenantIdentity): void;
