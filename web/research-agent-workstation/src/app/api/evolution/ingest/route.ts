import { NextResponse } from "next/server";
import { ingestEvolutionResult } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

/**
 * Backfill bridge: ingest a REAL training result (cv_score + on-disk artifacts)
 * into the search graph + shared memory and apply the promotion gate. The brain
 * only attaches artifacts that actually exist under experiments/<task>/<run_id>/.
 */
export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const taskId = typeof body.task_id === "string" ? body.task_id : "";
  try {
    const payload = await ingestEvolutionResult({ ...body, official_submit_allowed: false });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution ingest failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
