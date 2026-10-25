import { NextResponse } from "next/server";
import { evolutionErrorBody } from "@/lib/evolution-cli-path";
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
    const failure = evolutionErrorBody(error, { task_id: taskId }, "evolution state failed");
    return NextResponse.json(failure.body, { status: failure.status });
  }
}
