import { NextResponse } from "next/server";
import { getWorkstationSummary } from "@/lib/server/summary";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const summary = await getWorkstationSummary({ full: true, force: true });
  const connectors = summary.connector_status ?? {};
  return NextResponse.json({
    schema: "evomind.connector_health_registry.v1",
    states: ["READY", "DEGRADED", "OFFLINE", "NOT_CONFIGURED"],
    connectors,
    checked_at: new Date().toISOString(),
  });
}
