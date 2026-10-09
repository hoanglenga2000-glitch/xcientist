import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request, context: { params: Promise<{ approvalId: string }> }) {
  const { approvalId } = await context.params;
  return proxyAssistantRuntime(request, `/v1/approvals/${encodeURIComponent(approvalId)}/decision`);
}
