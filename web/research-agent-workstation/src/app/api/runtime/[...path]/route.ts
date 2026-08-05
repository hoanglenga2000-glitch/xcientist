import { ensureEvoMindRuntime } from "@/lib/server/evomind-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

async function proxy(request: Request, context: { params: Promise<{ path: string[] }> }) {
  let local: Awaited<ReturnType<typeof ensureEvoMindRuntime>>;
  try {
    local = await ensureEvoMindRuntime();
  } catch {
    return Response.json(
      { ok: false, code: "runtime_not_ready" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
  const segments = (await context.params).path.map(encodeURIComponent).join("/");
  const incoming = new URL(request.url);
  const headers: Record<string, string> = { Authorization: `Bearer ${local.token}`, Accept: request.headers.get("accept") ?? "application/json" };
  const contentType = request.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;
  const response = await fetch(`${local.baseUrl}/v1/${segments}${incoming.search}`, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
    cache: "no-store",
  });
  return new Response(response.body, { status: response.status, headers: { "Content-Type": response.headers.get("content-type") ?? "application/json", "Cache-Control": "no-store" } });
}

export const GET = proxy;
export const POST = proxy;
