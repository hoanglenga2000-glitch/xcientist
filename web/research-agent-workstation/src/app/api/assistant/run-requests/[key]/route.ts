import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ key: string }> }) {
  const { key } = await context.params;
  return proxyAssistantRuntime(request, `/v1/run-requests/${encodeURIComponent(key)}`);
}
