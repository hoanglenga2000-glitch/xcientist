import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { decodeJson, encodeJson } from "@/lib/server/json";

export const dynamic = "force-dynamic";

const SENSITIVE_KEY_PATTERN = /(api[_-]?key|token|secret|password|cookie|credential|private[_-]?key|access[_-]?token|refresh[_-]?token)/i;

function redactSettingsValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactSettingsValue);
  if (!value || typeof value !== "object") return value;

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => {
      if (SENSITIVE_KEY_PATTERN.test(key)) {
        const lowerKey = key.toLowerCase();
        if (lowerKey.endsWith("_status") || lowerKey === "status" || lowerKey === "token_status" || lowerKey === "credential_status") {
          return [key, typeof item === "string" ? item : item ? "configured" : "not_configured"];
        }
        const configured = typeof item === "string" ? item.trim().length > 0 : Boolean(item);
        return [key, configured ? "hidden_configured" : "not_configured"];
      }
      return [key, redactSettingsValue(item)];
    })
  );
}

function redactSettingsMap(settings: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(settings).map(([key, value]) => [key, redactSettingsValue(value)])
  );
}

export async function GET() {
  await ensureWorkstationSeeded();
  const settings = await prisma.setting.findMany({ orderBy: { key: "asc" } });
  const rawSettings = Object.fromEntries(settings.map((item) => [item.key, decodeJson(item.valueJson) ?? {}]));
  return NextResponse.json({
    ok: true,
    settings: redactSettingsMap(rawSettings)
  });
}

export async function PATCH(request: Request) {
  await ensureWorkstationSeeded();
  const body = await request.json().catch(() => ({}));
  const settings = body.settings && typeof body.settings === "object" ? body.settings as Record<string, unknown> : {};

  if (!Object.keys(settings).length) {
    return NextResponse.json({ ok: false, error: "settings payload is required" }, { status: 400 });
  }

  for (const [key, value] of Object.entries(settings)) {
    const valueJson = encodeJson(value) ?? "{}";
    await prisma.setting.upsert({
      where: { key },
      update: { valueJson },
      create: { key, valueJson }
    });
  }

  await logAction({
    action: "save_settings",
    message: "Settings saved to SQLite.",
    artifactPath: null,
    metadata: { keys: Object.keys(settings) }
  });

  return GET();
}
