import { NextResponse } from "next/server";
import { evolutionErrorBody } from "@/lib/evolution-cli-path";
import { getEvolutionMemory } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const taskType = url.searchParams.get("task_type") ?? "";
  try {
    const payload = await getEvolutionMemory(taskId, { task_type: taskType });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const failure = evolutionErrorBody(error, { task_id: taskId }, "evolution memory failed");
    return NextResponse.json(failure.body, { status: failure.status });
  }
}
