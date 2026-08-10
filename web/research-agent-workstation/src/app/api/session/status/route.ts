import { cookies, headers } from "next/headers";
import { NextResponse } from "next/server";
import {
  LOCAL_AUTOMATION_VERIFIED_HEADER,
  SESSION_COOKIE,
  expectedCsrfToken,
  validSessionCookie,
} from "@/lib/server/local-session";

export const dynamic = "force-dynamic";

export async function GET() {
  if ((await headers()).get(LOCAL_AUTOMATION_VERIFIED_HEADER) === "1") {
    return NextResponse.json(
      {
        ok: true,
        csrf_token: "local-automation",
        authenticated: true,
        authentication: "local_automation",
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  }
  const value = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!validSessionCookie(value)) {
    return NextResponse.json({ ok: false, code: "session_required" }, { status: 401 });
  }
  return NextResponse.json(
    { ok: true, csrf_token: expectedCsrfToken(value), authenticated: true },
    { headers: { "Cache-Control": "no-store" } },
  );
}
