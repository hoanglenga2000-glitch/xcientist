import { NextResponse } from "next/server";
import { runMCGSExperiment } from "@/lib/server/runs";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  try {
    const { taskId } = await params;
    const body = await _request.json().catch(() => ({}));
    return NextResponse.json(await runMCGSExperiment(taskId, {
      budgetNodes: body.budgetNodes ?? 8,
      fast: body.fast ?? false,
    }));
  } catch (error) {
    const { taskId } = await params;
    const message = error instanceof Error ? error.message : "Unknown MCGS error";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 500 });
  }
}
