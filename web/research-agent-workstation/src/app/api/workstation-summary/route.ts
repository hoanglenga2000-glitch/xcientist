import { NextResponse } from "next/server";
import { sanitizeWorkstationSummary } from "@/lib/server/json";
import { getWorkstationSummary } from "@/lib/server/summary";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(sanitizeWorkstationSummary(await getWorkstationSummary()));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown summary error";
    return NextResponse.json(sanitizeWorkstationSummary({ ok: false, error: message }), { status: 500 });
  }
}
