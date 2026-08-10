import { NextResponse } from "next/server";
import { runtimeVersionIdentity } from "@/lib/server/runtime-version";

export const dynamic = "force-dynamic";

export async function GET() {
  const identity = await runtimeVersionIdentity();
  return NextResponse.json(
    {
      ok: identity.ready,
      status: identity.status,
      service: "evomind-workstation",
      version: identity.frontend_version ?? "unknown",
      commit_hash: identity.commit_hash,
      build_id: identity.build_id,
      failures: identity.failures,
    },
    {
      status: identity.ready ? 200 : 503,
      headers: { "Cache-Control": "no-store" },
    },
  );
}
