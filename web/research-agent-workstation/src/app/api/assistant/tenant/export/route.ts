import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// t238 C7b D6: streaming ZIP of the caller's files + manifest.json (the runtime streams; no buffering here).
export async function GET(request: Request) {
  return proxyAssistantRuntime(request, "/v1/tenant/export");
}
