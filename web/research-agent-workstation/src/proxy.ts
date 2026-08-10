import { NextRequest, NextResponse } from "next/server";
import {
  CSRF_HEADER,
  LOCAL_AUTOMATION_VERIFIED_HEADER,
  SESSION_COOKIE,
  localOrigin,
  validCsrfToken,
  validLocalAutomationToken,
  validSessionCookie,
} from "@/lib/server/local-session";
import {
  isAllowedMutationSource,
  isLoopbackHostHeader,
  normalizeTaskId,
} from "@/lib/security/request-boundary";

const PUBLIC_API_PATHS = new Set(["/api/healthz", "/api/system/version", "/api/session/bootstrap"]);
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
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

function pageResponse(request: NextRequest) {
  // Next.js' App Router emits inline bootstrap/Flight scripts. A fresh nonce
  // keeps those scripts executable without permitting arbitrary inline code.
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
  if (!pathname.startsWith("/api/")) return pageResponse(request);
  const host = request.headers.get("host");
  if (!isLoopbackHostHeader(host)) {
    return jsonError(403, "loopback_required");
  }

  if (
    MUTATING_METHODS.has(request.method)
    && !isAllowedMutationSource(
      request.headers.get("origin"),
      host,
      request.headers.get("sec-fetch-site"),
    )
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

  if (PUBLIC_API_PATHS.has(pathname)) return NextResponse.next();

  const session = request.cookies.get(SESSION_COOKIE)?.value;
  const localAutomation = validLocalAutomationToken(request.headers.get(LOCAL_AUTOMATION_HEADER));
  const forwardedHeaders = new Headers(request.headers);
  forwardedHeaders.delete(LOCAL_AUTOMATION_VERIFIED_HEADER);
  if (localAutomation) forwardedHeaders.set(LOCAL_AUTOMATION_VERIFIED_HEADER, "1");
  try {
    if (!localAutomation && !validSessionCookie(session)) return jsonError(401, "session_required");
  } catch {
    return jsonError(503, "local_session_not_configured");
  }

  if (!SAFE_METHODS.has(request.method)) {
    const expectedOrigin = localOrigin();
    const origin = request.headers.get("origin");
    if (origin !== expectedOrigin || host !== new URL(expectedOrigin).host) {
      return jsonError(403, "origin_rejected");
    }
    if (!localAutomation && !validCsrfToken(request.headers.get(CSRF_HEADER), session!)) {
      return jsonError(403, "csrf_rejected");
    }
    const rawLength = request.headers.get("content-length");
    if (rawLength === null) {
      // Every mutation must carry an authenticated, bounded representation.
      // This also rejects chunked/HTTP2 bodies whose size cannot be enforced
      // before the route allocates or parses them.
      return jsonError(411, "content_length_required");
    }
    if (!/^\d+$/.test(rawLength) || Number(rawLength) > MAX_BODY_BYTES) {
      return jsonError(413, "body_too_large");
    }
    const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase() ?? "";
    const allowsMultipart = pathname === "/api/literature/import";
    if (contentType !== "application/json" && !(allowsMultipart && contentType === "multipart/form-data")) {
      return jsonError(415, "unsupported_content_type");
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
