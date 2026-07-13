import { NextResponse } from "next/server";
import { getEvolutionGraph } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const metricDirection = url.searchParams.get("metric_direction") ?? "maximize";
  try {
    const payload = await getEvolutionGraph(taskId, { metric_direction: metricDirection });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution graph failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
