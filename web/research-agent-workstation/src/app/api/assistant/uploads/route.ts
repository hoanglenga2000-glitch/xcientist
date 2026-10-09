import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  return proxyAssistantRuntime(request, "/v1/uploads");
}
