import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await context.params;
  if (!/^project_[a-f0-9]{32}$/.test(projectId)) return Response.json({ error: "project_not_found" }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/projects/${encodeURIComponent(projectId)}`);
}
