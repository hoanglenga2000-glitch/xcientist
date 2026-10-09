import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ artifactId: string }> }) {
  const { artifactId } = await context.params;
  const incoming = new URL(request.url);
  if (incoming.searchParams.get("preview") === "1") {
    return proxyAssistantRuntime(request, `/v1/artifacts/${encodeURIComponent(artifactId)}/preview`);
  }
  const query = new URLSearchParams();
  if (incoming.searchParams.get("download") === "1") query.set("download", "1");
  const derived = incoming.searchParams.get("derived");
  if (derived === "pdf" || derived === "png") query.set("derived", derived);
  const suffix = query.size ? `?${query}` : "";
  return proxyAssistantRuntime(request, `/v1/artifacts/${encodeURIComponent(artifactId)}${suffix}`);
}
