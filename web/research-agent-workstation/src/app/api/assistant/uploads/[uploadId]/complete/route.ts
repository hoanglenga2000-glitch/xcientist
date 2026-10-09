import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request, context: { params: Promise<{ uploadId: string }> }) {
  const { uploadId } = await context.params;
  return proxyAssistantRuntime(request, `/v1/uploads/${encodeURIComponent(uploadId)}/complete`);
}
