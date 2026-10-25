import { NextResponse } from "next/server";
import { evolutionErrorBody } from "@/lib/evolution-cli-path";
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
    const failure = evolutionErrorBody(error, { task_id: taskId, run_id: runId || null }, "evolution experience failed");
    return NextResponse.json(failure.body, {
      status: failure.status,
      headers: { "Cache-Control": "private, no-store, max-age=0" },
    });
  }
}
