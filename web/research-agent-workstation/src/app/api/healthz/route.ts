import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(
    {
      ok: true,
      status: "ready",
      service: "evomind-workstation",
      version: process.env.npm_package_version ?? "unknown",
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
