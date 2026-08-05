import { NextResponse } from "next/server";
import { getEvolutionExperience } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const runId = url.searchParams.get("run_id") ?? "";
  try {
    const payload = await getEvolutionExperience(taskId, runId);
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "private, no-store, max-age=0" }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution experience failed";
    return NextResponse.json(
      { ok: false, task_id: taskId, run_id: runId || null, error: message },
      { status: 400, headers: { "Cache-Control": "private, no-store, max-age=0" } }
    );
  }
}
