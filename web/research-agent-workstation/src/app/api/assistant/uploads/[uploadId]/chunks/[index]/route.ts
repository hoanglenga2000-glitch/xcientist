import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function PUT(request: Request, context: { params: Promise<{ uploadId: string; index: string }> }) {
  const { uploadId, index } = await context.params;
  return proxyAssistantRuntime(request, `/v1/uploads/${encodeURIComponent(uploadId)}/chunks/${encodeURIComponent(index)}`);
}
