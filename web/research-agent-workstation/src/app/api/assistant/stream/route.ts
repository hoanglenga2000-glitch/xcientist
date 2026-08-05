import path from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { workspaceRoot } from "@/lib/server/paths";
import { CSRF_HEADER, SESSION_COOKIE, localOrigin } from "@/lib/server/local-session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

type AssistantMessage = {
  role?: unknown;
  content?: unknown;
};

type AssistantRequest = {
  prompt?: unknown;
  session_id?: unknown;
  selected_task?: unknown;
  history?: unknown;
};

function cleanHistory(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.slice(-20).flatMap((item) => {
    const message = item as AssistantMessage;
    const role = String(message?.role ?? "");
    const content = String(message?.content ?? "").trim();
    if (!content || (role !== "user" && role !== "assistant")) return [];
    return [{ role, content: content.slice(0, 8000) }];
  });
}

function cleanIdentifier(value: unknown, fallback: string) {
  const candidate = String(value ?? "").trim();
  return /^[a-zA-Z0-9_.-]{1,160}$/.test(candidate) ? candidate : fallback;
}

function sse(event: string, payload: Record<string, unknown>) {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function assistantToolSessionEnv(request: Request) {
  const cookieHeader = request.headers.get("cookie") ?? "";
  const prefix = `${SESSION_COOKIE}=`;
  const sessionCookie = cookieHeader
    .split(";")
    .map((item) => item.trim())
    .find((item) => item.startsWith(prefix));
  const csrf = request.headers.get(CSRF_HEADER) ?? "";
  if (!sessionCookie || !csrf) return {};
  return {
    EVOMIND_INTERNAL_SESSION_COOKIE: sessionCookie,
    EVOMIND_INTERNAL_CSRF: csrf,
    EVOMIND_INTERNAL_ORIGIN: localOrigin(),
  };
}

export async function POST(request: Request) {
  let body: AssistantRequest;
  try {
    body = await request.json() as AssistantRequest;
  } catch {
    return Response.json({ ok: false, error: "invalid_json" }, { status: 400 });
  }

  const prompt = String(body.prompt ?? "").trim();
  if (!prompt || prompt.length > 20000) {
    return Response.json({ ok: false, error: "invalid_prompt" }, { status: 400 });
  }

  const sessionId = cleanIdentifier(body.session_id, `chat_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`);
  const selectedTask = cleanIdentifier(body.selected_task, "");
  const payload = {
    prompt,
    session_id: sessionId,
    selected_task: selectedTask,
    history: cleanHistory(body.history),
  };
  const encoder = new TextEncoder();
  const toolSessionEnv = assistantToolSessionEnv(request);
  let child: ChildProcessWithoutNullStreams | null = null;
  let closed = false;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const python = process.env.EVOMIND_PYTHON ?? process.env.PYTHON ?? "python";
      const sourceRoot = path.join(workspaceRoot, "src");
      child = spawn(python, ["-X", "utf8", "-m", "xsci.assistant_stream"], {
        cwd: workspaceRoot,
        windowsHide: true,
        shell: false,
        env: {
          ...process.env,
          ...toolSessionEnv,
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
          PYTHONPATH: [sourceRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        },
      });

      let stdoutBuffer = "";
      let stderrSize = 0;
      let completed = false;

      const push = (event: string, data: Record<string, unknown>) => {
        if (closed) return;
        controller.enqueue(encoder.encode(sse(event, data)));
      };
      const close = () => {
        if (closed) return;
        closed = true;
        controller.close();
      };

      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        stdoutBuffer += chunk;
        const lines = stdoutBuffer.split(/\r?\n/);
        stdoutBuffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line) as Record<string, unknown>;
            const event = String(data.type ?? "message");
            if (event === "answer_completed") completed = true;
            push(event, data);
          } catch {
            // Python stdout is an NDJSON contract. Non-contract diagnostics stay hidden.
          }
        }
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderrSize = Math.min(65536, stderrSize + chunk.length);
      });
      child.on("error", () => {
        push("error", { type: "error", code: "assistant_process_start_failed", message: "助手服务启动失败，请重试。", session_id: sessionId });
        close();
      });
      child.on("close", (code) => {
        if (!completed && !request.signal.aborted) {
          push("error", {
            type: "error",
            code: code === 0 ? "assistant_stream_incomplete" : "assistant_process_failed",
            message: "本轮回复中断，状态已保留，可以直接重试。",
            session_id: sessionId,
            diagnostics_available: stderrSize > 0,
          });
        }
        close();
      });

      request.signal.addEventListener("abort", () => {
        if (child && !child.killed) child.kill();
        close();
      }, { once: true });

      child.stdin.end(JSON.stringify(payload));
    },
    cancel() {
      closed = true;
      if (child && !child.killed) child.kill();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
