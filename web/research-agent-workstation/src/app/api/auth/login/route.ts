import { createHash } from "node:crypto";
import { NextResponse } from "next/server";

import { cookieOptions, expectedCsrfToken, issuePublicSessionCookie, SESSION_COOKIE } from "@/lib/server/local-session";
import { safePostLoginPath, verifyPublicCredentials } from "@/lib/server/public-auth";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null) as { username?: unknown; password?: unknown; next?: unknown } | null;
  const username = typeof body?.username === "string" ? body.username.trim() : "";
  const password = typeof body?.password === "string" ? body.password : "";
  if (!username || !password || username.length > 64 || password.length > 256) {
    return NextResponse.json({ ok: false, code: "invalid_credentials", message: "账号或密码不正确" }, { status: 401 });
  }
  const forwarded = request.headers.get("x-forwarded-for")?.trim() ?? "unknown";
  const clientAddress = /^[A-Fa-f0-9:.]{2,64}$/.test(forwarded) ? forwarded.toLowerCase() : "unknown";
  const rateKey = createHash("sha256").update(clientAddress, "utf8").digest("hex");
  let result;
  try {
    result = await verifyPublicCredentials(username, password, rateKey);
  } catch {
    return NextResponse.json({ ok: false, code: "auth_not_configured", message: "登录服务尚未完成配置" }, { status: 503 });
  }
  if (!result.ok) {
    const headers = result.retryAfterSeconds ? { "Retry-After": String(result.retryAfterSeconds) } : undefined;
    return NextResponse.json(
      { ok: false, code: result.code, message: result.code === "rate_limited" ? "尝试次数过多，请稍后再试" : "账号或密码不正确" },
      { status: result.code === "rate_limited" ? 429 : 401, headers },
    );
  }
  const session = issuePublicSessionCookie(result.username, result.tenantId, undefined, result.sessionVersion);
  const response = NextResponse.json({
    ok: true,
    username: result.username,
    tenant_id: result.tenantId,
    csrf_token: expectedCsrfToken(session),
    redirect_to: safePostLoginPath(body?.next),
  });
  response.cookies.set(SESSION_COOKIE, session, cookieOptions());
  response.headers.set("Cache-Control", "no-store");
  return response;
}
