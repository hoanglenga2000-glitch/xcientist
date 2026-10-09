import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  const reportId = new URL(request.url).searchParams.get("report_id");
  const endpoint = `/v1/runs/${encodeURIComponent(runId)}/reports${reportId ? `/${encodeURIComponent(reportId)}` : ""}`;
  const response = await assistantRuntimeFetch(endpoint, { signal: request.signal });
  return new Response(response.body, { status: response.status, headers: { "Content-Type": "application/json", "Cache-Control": "private, no-store" } });
}

export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  try {
    const body = await request.json() as Record<string, unknown>;
    const argumentsValue = body.report_id ? { report_id: body.report_id } : {
      title: body.title, summary: body.summary, artifact_ids: body.artifact_ids, report_kind: body.report_kind,
      language: body.language ?? "zh-CN", formats: body.formats ?? ["markdown", "html", "docx", "pdf"],
    };
    const response = await assistantRuntimeFetch(`/v1/runs/${encodeURIComponent(runId)}/reports`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...argumentsValue,
        idempotency_key: typeof body.idempotency_key === "string" ? body.idempotency_key : "" }),
      signal: request.signal,
    });
    return new Response(response.body, { status: response.status, headers: { "Content-Type": "application/json", "Cache-Control": "private, no-store" } });
  } catch {
    return Response.json({ error: "report_request_failed" }, { status: 400 });
  }
}
