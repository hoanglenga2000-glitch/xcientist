import { NextResponse } from "next/server";
import { getEvolutionState } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const taskType = url.searchParams.get("task_type") ?? "";
  const metricDirection = url.searchParams.get("metric_direction") ?? "maximize";
  try {
    const payload = await getEvolutionState(taskId, { task_type: taskType, metric_direction: metricDirection });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution state failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
