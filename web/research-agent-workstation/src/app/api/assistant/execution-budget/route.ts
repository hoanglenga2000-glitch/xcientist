import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export function GET(request: Request) {
  return proxyAssistantRuntime(request, "/v1/execution-budget");
}

export function POST(request: Request) {
  return proxyAssistantRuntime(request, "/v1/execution-budget");
}
