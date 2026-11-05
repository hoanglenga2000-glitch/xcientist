import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// t238 C7b D6: delete one owned upload (non-owner → 404; active run → 409 attachment_in_use).
export async function DELETE(request: Request, { params }: { params: Promise<{ uploadId: string }> }) {
  const { uploadId } = await params;
  if (!/^upload_[a-f0-9]{32}$/.test(uploadId)) return Response.json({ error: "not_found" }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/uploads/${uploadId}`);
}
