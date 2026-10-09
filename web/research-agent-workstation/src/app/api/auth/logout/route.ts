import { NextResponse } from "next/server";

import { cookieOptions, SESSION_COOKIE } from "@/lib/server/local-session";

export const dynamic = "force-dynamic";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, "", { ...cookieOptions(), maxAge: 0, expires: new Date(0) });
  response.headers.set("Cache-Control", "no-store");
  return response;
}
