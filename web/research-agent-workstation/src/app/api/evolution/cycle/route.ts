import { NextResponse } from "next/server";
import { runEvolutionCycle } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

/**
 * Full evolution loop: plan -> human-approval gate -> REAL training -> ingest.
 * Real training only runs when the body carries { approve: true }; otherwise the
 * route returns the plan and stops. official_submit_allowed is forced false.
 */
export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const taskId = typeof body.task_id === "string" ? body.task_id : "";
  try {
    const payload = await runEvolutionCycle({ ...body, official_submit_allowed: false });
    return NextResponse.json({ ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution cycle failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
