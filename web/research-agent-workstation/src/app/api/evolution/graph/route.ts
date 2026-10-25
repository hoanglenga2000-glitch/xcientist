import { NextResponse } from "next/server";
import { evolutionErrorBody } from "@/lib/evolution-cli-path";
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
    const failure = evolutionErrorBody(error, { task_id: taskId }, "evolution graph failed");
    return NextResponse.json(failure.body, { status: failure.status });
  }
}
