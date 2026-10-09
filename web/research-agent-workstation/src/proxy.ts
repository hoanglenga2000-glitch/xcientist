import { NextRequest, NextResponse } from "next/server";
import { accountRegistryEnabled } from "@/lib/server/account-registry.mjs";

import {
  CSRF_HEADER,
  LOCAL_AUTOMATION_VERIFIED_HEADER,
  SESSION_COOKIE,
  localOrigin,
  sessionPrincipal,
  validCsrfToken,
  validLocalAutomationToken,
  validSessionCookie,
} from "@/lib/server/local-session";
import {
  configuredPublicOrigin,
  isAllowedMutationSource,
  isLoopbackHostHeader,
  isPublicHostHeader,
  isTrustedHostHeader,
  normalizeTaskId,
} from "@/lib/security/request-boundary";

const PUBLIC_API_PATHS = new Set([
  "/api/healthz",
  "/api/system/version",
  "/api/session/bootstrap",
  "/api/auth/login",
]);
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
const MAX_BODY_BYTES = 16 * 1024 * 1024;
const LOCAL_AUTOMATION_HEADER = "x-evomind-local-automation";

function pageContentSecurityPolicy(nonce: string) {
  return [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}'`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "frame-src 'self' blob:",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

function pageResponse(request: NextRequest) {
  const nonce = crypto.randomUUID().replaceAll("-", "");
  const contentSecurityPolicy = pageContentSecurityPolicy(nonce);
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", contentSecurityPolicy);
  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", contentSecurityPolicy);
  response.headers.set("Cache-Control", "private, no-store, max-age=0, must-revalidate");
  return response;
}

function jsonError(status: number, code: string) {
  return NextResponse.json({ ok: false, code }, { status, headers: { "Cache-Control": "no-store" } });
}

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const host = request.headers.get("host");
  if (!isTrustedHostHeader(host)) return jsonError(403, "untrusted_host");

  if (!pathname.startsWith("/api/")) {
    if (pathname === "/login") return pageResponse(request);
    if (isPublicHostHeader(host) || accountRegistryEnabled()) {
      try {
        const principal = sessionPrincipal(request.cookies.get(SESSION_COOKIE)?.value);
        if (!principal || principal.authentication !== "public_password") {
          const publicOrigin = isLoopbackHostHeader(host) ? request.nextUrl.origin : configuredPublicOrigin();
          if (!publicOrigin) return jsonError(503, "public_origin_not_configured");
          const loginUrl = new URL("/login", publicOrigin);
          loginUrl.searchParams.set("next", `${request.nextUrl.pathname}${request.nextUrl.search}`);
          return NextResponse.redirect(loginUrl, 307);
        }
      } catch {
        return jsonError(503, "session_not_configured");
      }
    }
    return pageResponse(request);
  }

  if (pathname === "/api/session/bootstrap" && !isLoopbackHostHeader(host)) {
    return jsonError(404, "not_found");
  }

  if (
    MUTATING_METHODS.has(request.method)
    && !isAllowedMutationSource(request.headers.get("origin"), host, request.headers.get("sec-fetch-site"))
  ) {
    return jsonError(403, "origin_rejected");
  }

  const taskPrefix = "/api/tasks/";
  if (pathname.startsWith(taskPrefix)) {
    const taskSegment = pathname.slice(taskPrefix.length).split("/", 1)[0];
    try {
      normalizeTaskId(taskSegment);
    } catch {
      return jsonError(400, "invalid_task_id");
    }
  }

  if (MUTATING_METHODS.has(request.method)) {
    const rawLength = request.headers.get("content-length");
    if (rawLength === null) return jsonError(411, "content_length_required");
    if (!/^\d+$/.test(rawLength) || Number(rawLength) > MAX_BODY_BYTES) return jsonError(413, "body_too_large");
    const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase() ?? "";
    const allowsMultipart = pathname === "/api/literature/import";
    const allowsBinaryChunk = /^\/api\/assistant\/uploads\/[A-Za-z0-9_.-]+\/chunks\/\d+$/.test(pathname);
    if (
      contentType !== "application/json"
      && !(allowsMultipart && contentType === "multipart/form-data")
      && !(allowsBinaryChunk && contentType === "application/octet-stream")
    ) return jsonError(415, "unsupported_content_type");
  }

  if (PUBLIC_API_PATHS.has(pathname)) return NextResponse.next();

  const session = request.cookies.get(SESSION_COOKIE)?.value;
  const localAutomation = isLoopbackHostHeader(host)
    && validLocalAutomationToken(request.headers.get(LOCAL_AUTOMATION_HEADER));
  const forwardedHeaders = new Headers(request.headers);
  forwardedHeaders.delete("x-evomind-access-scope");
  forwardedHeaders.delete("x-evomind-tenant-id");
  forwardedHeaders.delete("x-evomind-principal-id");
  forwardedHeaders.delete(LOCAL_AUTOMATION_HEADER);
  forwardedHeaders.delete(LOCAL_AUTOMATION_VERIFIED_HEADER);
  if (localAutomation) forwardedHeaders.set(LOCAL_AUTOMATION_VERIFIED_HEADER, "1");
  try {
    const principal = sessionPrincipal(session);
    const validForHost = isPublicHostHeader(host)
      ? principal?.authentication === "public_password"
      : principal !== null;
    if (!localAutomation && (!validSessionCookie(session) || !validForHost)) return jsonError(401, "session_required");
    // Legacy projections and actions have no trustworthy personal ownership.
    // Even a managed administrator must use scoped APIs; verified local release
    // automation is a separate boundary, never a browser role escalation.
    if (!localAutomation && accountRegistryEnabled()) {
      const personal = /^\/api\/assistant\/(tasks|projects|runs|run-requests|uploads|files|artifacts|approvals|model-profiles|execution-budget|run-secrets)(\/|$)/.test(pathname);
      const sessionRoute = ["/api/session/status", "/api/auth/logout", "/api/settings"].includes(pathname)
        || (request.method === "GET" && pathname === "/api/hpc/byoa/binding");
      if (!personal && !sessionRoute) return jsonError(403, "personal_workspace_required");
    }
  } catch {
    return jsonError(503, "session_not_configured");
  }

  if (!SAFE_METHODS.has(request.method)) {
    if (pathname.startsWith("/api/runtime/") && !localAutomation) {
      return jsonError(403, "managed_assistant_endpoint_required");
    }
    const origin = request.headers.get("origin");
    const expectedOrigin = isLoopbackHostHeader(host) ? localOrigin() : configuredPublicOrigin();
    if (!expectedOrigin || origin !== expectedOrigin || host !== new URL(expectedOrigin).host) {
      return jsonError(403, "origin_rejected");
    }
    if (!localAutomation && !validCsrfToken(request.headers.get(CSRF_HEADER), session!)) {
      return jsonError(403, "csrf_rejected");
    }
  }

  const response = NextResponse.next({ request: { headers: forwardedHeaders } });
  response.headers.set("Cache-Control", "private, no-store, max-age=0, must-revalidate");
  return response;
}

export const config = {
  matcher: [
    "/api/:path*",
    "/((?!api/|_next/static|_next/image|favicon.ico|icon.png|robots.txt|sitemap.xml|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
