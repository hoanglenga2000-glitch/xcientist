import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { tenantByoaRoot } from "@/lib/server/tenant-byoa";
export const dynamic = "force-dynamic";
export async function GET() {
  try {
    const record = JSON.parse(await fs.readFile(path.join(tenantByoaRoot(), "enrollment-keys", "current-public-key.json"), "utf8")) as Record<string, unknown>;
    if (record.schema !== "evomind.tenant_enrollment_key.v1" || record.algorithm !== "RSA-OAEP-256+A256GCM" || typeof record.key_id !== "string" || !record.rsa_jwk || Date.parse(String(record.not_before_utc)) > Date.now() || Date.parse(String(record.expires_at_utc)) <= Date.now()) throw new Error("key_unavailable");
    return NextResponse.json({ ok: true, schema: record.schema, algorithm: record.algorithm, key_id: record.key_id, rsa_jwk: record.rsa_jwk, expires_at_utc: record.expires_at_utc }, { headers: { "Cache-Control": "no-store" } });
  } catch { return NextResponse.json({ ok: false, code: "enrollment_key_unavailable" }, { status: 503, headers: { "Cache-Control": "no-store" } }); }
}

