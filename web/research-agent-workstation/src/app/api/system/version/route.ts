import { NextResponse } from "next/server";

import { runtimeVersionIdentity } from "@/lib/server/runtime-version";

export const dynamic = "force-dynamic";

export async function GET() {
  const identity = await runtimeVersionIdentity();
  return NextResponse.json(identity, {
    status: identity.ready ? 200 : 503,
    headers: { "Cache-Control": "no-store" },
  });
}
