import { NextResponse } from "next/server";
import {
  LOCAL_AUTOMATION_HEADER,
  LOCAL_AUTOMATION_VERIFIED_HEADER,
  SESSION_COOKIE,
  sessionPrincipal,
  validLocalAutomationToken,
  validSessionCookie,
} from "@/lib/server/local-session";
import { isLoopbackHostHeader } from "@/lib/security/request-boundary";
import { sessionCookieFromHeaders } from "@/lib/server/runtime-principal";

// t238 C7b S6: legacy `/api/gpu/jobs*` records (workspace/gpu/jobs/*.json) carry no tenant
// attribution, so they cannot be filtered per tenant. Only the operator — verified local
// release automation or the single-user local bootstrap session — may use them; a tenant
// (public password) session gets 403, no session 401. Tenants use the scoped assistant APIs.
export const LEGACY_GPU_TENANT_MESSAGE = "这里的旧版 GPU 任务列表只对管理员开放。你的任务在「任务」页查看。";

export type LegacyGpuAccess = { ok: true } | { ok: false; status: 401 | 403; code: string };

type Principal = { authentication: string } | null;

export function legacyGpuAccess(
  request: Request,
  resolve: (cookie: string | undefined) => Principal = (cookie) => (validSessionCookie(cookie) ? sessionPrincipal(cookie) : null),
): LegacyGpuAccess {
  if (request.headers.get(LOCAL_AUTOMATION_VERIFIED_HEADER) === "1") return { ok: true };
  let principal: Principal;
  try {
    principal = resolve(sessionCookieFromHeaders(request.headers, SESSION_COOKIE));
  } catch {
    principal = null;
  }
  if (!principal) return { ok: false, status: 401, code: "session_required" };
  if (principal.authentication === "local_bootstrap") return { ok: true };
  return { ok: false, status: 403, code: "legacy_gpu_operator_only" };
}

// t238 C7b H1: for a route reachable on a public path (`/api/healthz?deep=1`) the derived
// verified header is never trusted, even though the proxy strips it there. Operator access
// is the raw loopback automation token re-verified here, or a local bootstrap session.
export function verifiedOperatorAccess(
  request: Request,
  resolve?: (cookie: string | undefined) => Principal,
): LegacyGpuAccess {
  if (isLoopbackHostHeader(request.headers.get("host"))
    && validLocalAutomationToken(request.headers.get(LOCAL_AUTOMATION_HEADER))) return { ok: true };
  const headers = new Headers(request.headers);
  headers.delete(LOCAL_AUTOMATION_VERIFIED_HEADER);
  return legacyGpuAccess(new Request(request.url, { headers }), resolve);
}

export function legacyGpuDenied(access: Exclude<LegacyGpuAccess, { ok: true }>) {
  return NextResponse.json(
    {
      ok: false,
      code: access.code,
      error: access.code,
      ...(access.status === 403 ? { message: LEGACY_GPU_TENANT_MESSAGE } : {}),
    },
    { status: access.status, headers: { "Cache-Control": "no-store" } },
  );
}
