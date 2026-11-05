import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// t238 C7b D6: delete all of the caller's uploaded data; body must be exactly {"confirm":"DELETE"}.
export async function DELETE(request: Request) {
  return proxyAssistantRuntime(request, "/v1/tenant/data");
}
