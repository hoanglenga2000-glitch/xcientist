import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// t237 C3.7: {decision: "confirm" | "cancel"} on an experience plan card; the runtime only records it.
export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  if (!/^run_[A-Za-z0-9_-]{1,160}$/.test(runId)) return Response.json({ error: 'not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/runs/${runId}/plan`);
}
