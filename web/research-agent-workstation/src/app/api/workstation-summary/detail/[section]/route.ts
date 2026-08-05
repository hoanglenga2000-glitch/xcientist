import { NextRequest, NextResponse } from "next/server";
import { sanitizeWorkstationSummary } from "@/lib/server/json";
import { getWorkstationSummary } from "@/lib/server/summary";

export const dynamic = "force-dynamic";

const allowedSections = new Set([
  "assistant", "overview", "tasks", "data", "gpu", "evidence", "literature",
  "workflow", "code", "runtime", "experiments", "evolution", "report", "gates",
  "settings", "control"
]);

export async function GET(request: NextRequest, context: { params: Promise<{ section: string }> }) {
  const { section } = await context.params;
  if (!allowedSections.has(section)) {
    return NextResponse.json({ ok: false, error: "Unknown summary detail section" }, { status: 404 });
  }
  try {
    const force = request.nextUrl.searchParams.get("fresh") === "1";
    return NextResponse.json(sanitizeWorkstationSummary(await getWorkstationSummary({ detail: section, force })));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown summary detail error";
    return NextResponse.json(sanitizeWorkstationSummary({ ok: false, error: message }), { status: 500 });
  }
}
