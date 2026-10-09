import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";
import { sessionCookieFromHeaders } from "@/lib/server/runtime-principal";
import { preferenceKey, validatePreferences, mergePreferences } from "@/lib/server/user-preferences";

export const dynamic = "force-dynamic";

function keyFor(request: Request) {
  return preferenceKey(sessionPrincipal(sessionCookieFromHeaders(request.headers, SESSION_COOKIE)));
}
function decode(value: string | null | undefined): unknown {
  try { return JSON.parse(value ?? "{}"); } catch { return {}; }
}
function response(settings: unknown) {
  return NextResponse.json({ ok: true, settings }, { headers: { "Cache-Control": "private, no-store" } });
}
export async function GET(request: Request) {
  let key: string;
  try { key = keyFor(request); } catch { return NextResponse.json({ ok: false, error: "session_required" }, { status: 401 }); }
  try {
    const row = await prisma.setting.findUnique({ where: { key } });
    return response(mergePreferences(decode(row?.valueJson)));
  } catch { return NextResponse.json({ ok: false, error: "preferences_unavailable" }, { status: 503 }); }
}
export async function PATCH(request: Request) {
  let key: string;
  try { key = keyFor(request); } catch { return NextResponse.json({ ok: false, error: "session_required" }, { status: 401 }); }
  let patch;
  try {
    const body = await request.json();
    patch = validatePreferences(body?.settings);
  } catch { return NextResponse.json({ ok: false, error: "invalid_preferences" }, { status: 400 }); }
  try {
    const settings = await prisma.$transaction(async (tx) => {
      const row = await tx.setting.findUnique({ where: { key } });
      const next = mergePreferences(decode(row?.valueJson), patch);
      const valueJson = JSON.stringify(next);
      await tx.setting.upsert({ where: { key }, create: { key, valueJson }, update: { valueJson } });
      return next;
    });
    return response(settings);
  } catch { return NextResponse.json({ ok: false, error: "preferences_save_failed" }, { status: 503 }); }
}
