import { runtimeJson } from "@/lib/server/assistant-runtime";
import { currentAssistantManagedHpcIdentity } from "@/lib/server/assistant-managed-hpc";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

type AssistantRequest = {
  prompt?: unknown;
  session_id?: unknown;
  selected_task?: unknown;
};

type RunSnapshot = {
  id: string;
  status: string;
  prompt: string;
};

type RuntimeEvent = {
  seq: number;
  event_type: string;
  payload: Record<string, unknown>;
};

function cleanIdentifier(value: unknown, fallback: string) {
  const candidate = String(value ?? "").trim();
  return /^[a-zA-Z0-9_.-]{1,160}$/.test(candidate) ? candidate : fallback;
}

function sse(event: string, payload: Record<string, unknown>) {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

export async function POST(request: Request) {
  let body: AssistantRequest;
  try {
    body = await request.json() as AssistantRequest;
  } catch {
    return Response.json({ ok: false, error: "invalid_json" }, { status: 400 });
  }
  const prompt = String(body.prompt ?? "").trim();
  if (!prompt || prompt.length > 20_000) {
    return Response.json({ ok: false, error: "invalid_prompt" }, { status: 400 });
  }
  const sessionId = cleanIdentifier(body.session_id, `chat_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`);
  const selectedTask = cleanIdentifier(body.selected_task, "");
  let run: RunSnapshot;
  try {
    run = await runtimeJson<RunSnapshot>("/v1/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: selectedTask ? `[Selected task: ${selectedTask}]\n${prompt}` : prompt,
        conversation_id: sessionId,
        selected_task: selectedTask,
        attachment_ids: [],
        managed_hpc_identity: await currentAssistantManagedHpcIdentity(),
      }),
    });
  } catch (error) {
    return Response.json(
      { ok: false, error: "runtime_not_ready", message: error instanceof Error ? error.message : "runtime_not_ready" },
      { status: 503 },
    );
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      void (async () => {
        let after = 0;
        let observedProvider = "";
        let observedModel = "";
        let inputTokens = 0;
        let outputTokens = 0;
        let nativeToolCalls = 0;
        const toolNames = new Set<string>();
        const emit = (event: string, payload: Record<string, unknown>) => controller.enqueue(encoder.encode(sse(event, { session_id: sessionId, ...payload })));
        emit("session", { ready: true, run_id: run.id, workspace: "assistant_task" });
        emit("context", { current_task: Boolean(selectedTask), task_label: selectedTask, run_status: "queued", memory_available: true, tools_available: true });
        emit("route", { route: "agent", label: "持久任务 Agent", detail: "受管运行时将持续执行并发布真实产物" });
        try {
          while (!request.signal.aborted) {
            const response = await runtimeJson<{ events: RuntimeEvent[] }>(`/v1/runs/${encodeURIComponent(run.id)}/events?after=${after}`);
            for (const item of response.events ?? []) {
              after = Math.max(after, Number(item.seq) || after);
              const payload = item.payload ?? {};
              if (item.event_type === "tool_started") {
                nativeToolCalls += 1;
                const tool = String(payload.tool ?? payload.tool_name ?? "");
                if (tool) toolNames.add(tool);
                emit("tool_started", payload);
              }
              else if (item.event_type === "tool_completed") emit("tool_completed", payload);
              else if (item.event_type === "model_observed") {
                observedProvider = String(payload.provider ?? observedProvider);
                observedModel = String(payload.model ?? observedModel);
                const observedInput = Number(payload.input_tokens ?? 0);
                const observedOutput = Number(payload.output_tokens ?? 0);
                if (Number.isFinite(observedInput) && observedInput > 0) inputTokens += observedInput;
                if (Number.isFinite(observedOutput) && observedOutput > 0) outputTokens += observedOutput;
                for (const name of Array.isArray(payload.tool_names) ? payload.tool_names : []) {
                  const tool = String(name ?? "");
                  if (tool) toolNames.add(tool);
                }
                emit("model", payload);
              }
              else if (["plan_updated", "step_started", "verification_completed", "retry_scheduled", "artifact_published"].includes(item.event_type)) {
                emit("thinking_status", { ...payload, status: "running", label: String(payload.label ?? item.event_type) });
              } else if (item.event_type === "approval_required") {
                emit("thinking_status", { ...payload, status: "blocked", label: "等待精确审批" });
              } else if (item.event_type === "run_completed") {
                const answer = String(payload.answer ?? "");
                if (answer) emit("answer_delta", { delta: answer });
                emit("answer_completed", {
                  answer,
                  run_id: run.id,
                  artifacts: payload.artifacts ?? [],
                  provider: observedProvider,
                  model: observedModel,
                  native_tool_calls: nativeToolCalls,
                  tool_calls_total: nativeToolCalls,
                  tool_names: [...toolNames],
                  input_tokens: inputTokens,
                  output_tokens: outputTokens,
                });
                controller.close();
                return;
              } else if (["run_blocked", "run_failed"].includes(item.event_type)) {
                emit("error", { ...payload, code: item.event_type, message: String(payload.message ?? "任务已保留，可直接继续。") });
                controller.close();
                return;
              }
            }
            await sleep(500, request.signal);
          }
          controller.close();
        } catch (error) {
          if (!request.signal.aborted) emit("error", { code: "assistant_runtime_stream_failed", message: error instanceof Error ? error.message : "任务事件流中断，可以继续同一 Run。", run_id: run.id });
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
