import { assistantRuntimeFetch, runtimeJson } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

type RuntimeEvent = {
  seq: number;
  event_type: string;
  payload: Record<string, unknown>;
};

const terminalEvents = new Set(["run_completed", "run_blocked", "run_failed"]);

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

export async function GET(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  // Reject before opening a 200 event stream. A transport/authorization failure
  // is not evidence that somebody's execution failed.
  try {
    const access = await assistantRuntimeFetch(`/v1/runs/${encodeURIComponent(runId)}`, { signal: request.signal });
    await access.body?.cancel();
    if (!access.ok) return Response.json({ error: access.status === 404 ? 'not_found' : 'stream_unavailable' },
      { status: access.status, headers: { 'Cache-Control': 'private, no-store' } });
  } catch {
    return Response.json({ error: 'stream_unavailable' }, { status: 503, headers: { 'Cache-Control': 'private, no-store' } });
  }
  const incoming = new URL(request.url);
  let after = Number(request.headers.get("last-event-id") ?? incoming.searchParams.get("after") ?? "0");
  if (!Number.isSafeInteger(after) || after < 0) after = 0;
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      void (async () => {
        const started = Date.now();
        let lastKeepalive = 0;
        try {
          while (!request.signal.aborted && Date.now() - started < 295_000) {
            const payload = await runtimeJson<{ events: RuntimeEvent[] }>(
              `/v1/runs/${encodeURIComponent(runId)}/events?after=${after}`,
            );
            for (const event of payload.events ?? []) {
              after = Math.max(after, Number(event.seq) || after);
              controller.enqueue(encoder.encode(
                `id: ${event.seq}\nevent: ${event.event_type}\ndata: ${JSON.stringify(event.payload)}\n\n`,
              ));
              if (terminalEvents.has(event.event_type)) {
                controller.close();
                return;
              }
            }
            if (Date.now() - lastKeepalive >= 15_000) {
              controller.enqueue(encoder.encode(`: keepalive ${Date.now()}\n\n`));
              lastKeepalive = Date.now();
            }
            await sleep(500, request.signal);
          }
          controller.close();
        } catch (error) {
          if (!request.signal.aborted) {
            controller.enqueue(encoder.encode(`event: stream_error\ndata: ${JSON.stringify({
              message: "event_stream_unavailable",
              recoverable: true,
            })}\n\n`));
          }
          controller.close();
        }
      })();
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
