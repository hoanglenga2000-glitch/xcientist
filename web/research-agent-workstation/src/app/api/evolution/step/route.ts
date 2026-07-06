import { NextResponse } from "next/server";
import { runEvolutionStep } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const taskId = typeof body.task_id === "string" ? body.task_id : "";
  // dry_run defaults to true; only an explicit false attempts a real step
  // (which the engine blocks in favor of the workstation orchestrator).
  const dryRun = body.dry_run !== false;
  try {
    const payload = await runEvolutionStep({ ...body, dry_run: dryRun, official_submit_allowed: false });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution step failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
