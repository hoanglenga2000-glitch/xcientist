import { NextResponse } from "next/server";
import { runLocalExperiment } from "@/lib/server/runs";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  try {
    return NextResponse.json(await runLocalExperiment(taskId));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown run error";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 500 });
  }
}
