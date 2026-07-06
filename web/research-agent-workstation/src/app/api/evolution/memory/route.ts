import { NextResponse } from "next/server";
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
    const message = error instanceof Error ? error.message : "evolution memory failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
