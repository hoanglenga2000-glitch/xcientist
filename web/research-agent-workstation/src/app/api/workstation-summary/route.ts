import { NextResponse } from "next/server";
import { getWorkstationSummary } from "@/lib/server/summary";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(await getWorkstationSummary());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown summary error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
