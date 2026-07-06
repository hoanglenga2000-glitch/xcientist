import { NextResponse } from "next/server";
import { runLocalExperiment } from "@/lib/server/runs";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: { taskId: string } }) {
  try {
    return NextResponse.json(await runLocalExperiment(params.taskId));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown run error";
    return NextResponse.json({ ok: false, task_id: params.taskId, error: message }, { status: 500 });
  }
}
