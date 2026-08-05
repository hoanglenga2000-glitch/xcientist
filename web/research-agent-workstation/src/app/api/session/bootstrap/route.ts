import { NextResponse } from "next/server";
import {
  SESSION_COOKIE,
  cookieOptions,
  expectedCsrfToken,
  expectedSessionCookie,
  localOrigin,
  validBootstrapToken,
} from "@/lib/server/local-session";

export const dynamic = "force-dynamic";

const CONSUMED_KEY = Symbol.for("evomind.local.bootstrap.consumed.v1");
type BootstrapGlobal = typeof globalThis & { [CONSUMED_KEY]?: boolean };
const MAX_BOOTSTRAP_BODY_BYTES = 2 * 1024;

function errorResponse(status: number, code: string) {
  return NextResponse.json({ ok: false, code }, { status, headers: { "Cache-Control": "no-store" } });
}

async function readBootstrapBody(request: Request): Promise<{ token?: unknown } | null | "too_large"> {
  const declaredLength = request.headers.get("content-length");
  if (declaredLength !== null) {
    if (!/^\d+$/.test(declaredLength)) return null;
    if (Number(declaredLength) > MAX_BOOTSTRAP_BODY_BYTES) return "too_large";
  }
  if (!request.body) return null;

  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = "";
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      bytes += chunk.value.byteLength;
      if (bytes > MAX_BOOTSTRAP_BODY_BYTES) {
        await reader.cancel().catch(() => undefined);
        return "too_large";
      }
      text += decoder.decode(chunk.value, { stream: true });
    }
    text += decoder.decode();
    const payload = JSON.parse(text) as unknown;
    return payload !== null && typeof payload === "object" && !Array.isArray(payload)
      ? payload as { token?: unknown }
      : null;
  } catch {
    return null;
  } finally {
    reader.releaseLock();
  }
}

export async function POST(request: Request) {
  const expectedOrigin = localOrigin();
  if (
    request.headers.get("origin") !== expectedOrigin
    || request.headers.get("host") !== new URL(expectedOrigin).host
  ) {
    return errorResponse(403, "origin_rejected");
  }
  const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (contentType !== "application/json") {
    return errorResponse(415, "unsupported_content_type");
  }
  const body = await readBootstrapBody(request);
  if (body === "too_large") return errorResponse(413, "body_too_large");
  if (!body) return errorResponse(400, "invalid_json");

  const globalState = globalThis as BootstrapGlobal;
  if (globalState[CONSUMED_KEY]) {
    return errorResponse(409, "bootstrap_consumed");
  }
  const token = typeof body?.token === "string" ? body.token : "";
  if (!token || !validBootstrapToken(token)) {
    return errorResponse(401, "invalid_bootstrap");
  }

  globalState[CONSUMED_KEY] = true;
  const session = expectedSessionCookie();
  const response = NextResponse.json(
    { ok: true, csrf_token: expectedCsrfToken(session), expires_in: cookieOptions().maxAge },
    { headers: { "Cache-Control": "no-store" } },
  );
  response.cookies.set(SESSION_COOKIE, session, cookieOptions());
  return response;
}
