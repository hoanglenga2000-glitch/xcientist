import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// t238 C7c-6: cancel the caller's own queued or running GPU training (idempotent; other tenants' runs → 404).
export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  if (!/^[A-Za-z0-9_-]{1,120}$/.test(runId)) return Response.json({ error: "invalid_run_id" }, { status: 400 });
  return proxyAssistantRuntime(request, `/v1/gpu/runs/${encodeURIComponent(runId)}/cancel`);
}
