export type RuntimePrincipal = {
  username: string;
  tenantId: string | null;
  authentication: "local_bootstrap" | "public_password";
};

export const RUNTIME_SCOPE_HEADERS = [
  "x-evomind-access-scope",
  "x-evomind-tenant-id",
  "x-evomind-principal-id",
] as const;

export function sessionCookieFromHeaders(headers: Headers, cookieName: string): string | undefined {
  const matches = (headers.get("cookie") ?? "").split(";")
    .map((value) => value.trim())
    .filter((value) => value.startsWith(`${cookieName}=`));
  if (matches.length > 1) throw new Error("ambiguous_session_cookie");
  return matches[0]?.slice(cookieName.length + 1);
}

export function principalScopeHeaders(
  principal: RuntimePrincipal | null,
  automation: { verified: boolean; loopback: boolean },
  initial?: HeadersInit,
): Headers {
  const headers = new Headers(initial);
  // Caller-supplied identity headers are never authoritative.
  for (const name of RUNTIME_SCOPE_HEADERS) headers.delete(name);
  if (automation.verified && automation.loopback) return headers;
  if (!principal) throw new Error("session_required");
  const local = principal.authentication === "local_bootstrap";
  const tenant = local ? "local" : principal.tenantId ?? "";
  if (
    (local && principal.username !== "local")
    || (!local && !/^tenant_[a-f0-9]{24}$/.test(tenant))
    || !/^[A-Za-z0-9._-]{1,64}$/.test(principal.username)
  ) throw new Error("invalid_principal_scope");
  headers.set(RUNTIME_SCOPE_HEADERS[0], "user.v1");
  headers.set(RUNTIME_SCOPE_HEADERS[1], tenant);
  headers.set(RUNTIME_SCOPE_HEADERS[2], principal.username);
  return headers;
}
