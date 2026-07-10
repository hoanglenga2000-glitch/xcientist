import { promises as fs } from "node:fs";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const stepTracePath = ".xsci/scientist_step_trace.jsonl";
const turnsPath = ".xsci/scientist_turns.jsonl";
const statusPath = ".xsci/scientist_autopilot_status.json";
const terminalTurnPath = ".xsci/scientist_terminal_turn.json";
const pollMs = 1200;
const heartbeatMs = 5000;

type TraceEvent = Record<string, unknown>;

async function readJsonlLines(relativePath: string) {
  const text = await fs.readFile(resolveWorkspacePath(relativePath), "utf-8").catch(() => "");
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function parseJsonLine(line: string): TraceEvent | null {
  try {
    const parsed = JSON.parse(line) as TraceEvent;
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeTraceEvent(event: TraceEvent, index: number) {
  return {
    event_id: String(event.event_id ?? event.id ?? `trace-${index}`),
    ts: String(event.ts ?? event.timestamp ?? event.created_at ?? ""),
    source: String(event.source ?? "scientist"),
    phase: String(event.phase ?? "step"),
    step_id: String(event.step_id ?? ""),
    tool: String(event.tool ?? ""),
    status: String(event.status ?? "info"),
    message: String(event.message ?? "").slice(0, 360),
    artifact_path: String(event.artifact_path ?? event.artifact ?? ""),
    gate: String(event.gate ?? ""),
    evidence: Array.isArray(event.evidence) ? event.evidence.slice(0, 8).map(String) : [],
    no_training_started: event.no_training_started !== false,
  };
}

async function readJsonlTail(relativePath: string, limit: number) {
  const lines = await readJsonlLines(relativePath);
  const parsed = lines
    .slice(-limit)
    .map(parseJsonLine)
    .filter(Boolean) as TraceEvent[];
  return {
    present: parsed.length > 0,
    artifact_path: relativePath,
    count: lines.length,
    latest: parsed.at(-1) ?? null,
    recent: sanitizeClientJson(parsed),
  };
}

async function readAutopilotStatus() {
  const payload = await readJsonFile(resolveWorkspacePath(statusPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: statusPath,
      running: false,
      status: "not_started",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
    };
  }
  return { present: true, artifact_path: statusPath, ...(sanitizeClientJson(payload) as TraceEvent) };
}

async function readTerminalTurn() {
  const payload = await readJsonFile(resolveWorkspacePath(terminalTurnPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: terminalTurnPath,
      tool: "scientist_terminal_turn",
      selected_task: null,
      user_goal: "",
      autonomy_level: "not_run",
      executed_tools: [],
      execution_ready: false,
      execution_blocked: true,
      blocking_gates: ["No Scientist terminal turn has been generated yet."],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
    };
  }
  return { present: true, artifact_path: terminalTurnPath, ...(sanitizeClientJson(payload) as TraceEvent) };
}

async function buildSnapshot() {
  const [stepTrace, turns, status, terminalTurn] = await Promise.all([
    readJsonlTail(stepTracePath, 80),
    readJsonlTail(turnsPath, 20),
    readAutopilotStatus(),
    readTerminalTurn(),
  ]);
  const recent = Array.isArray(stepTrace.recent) ? (stepTrace.recent as TraceEvent[]) : [];
  const normalized = recent.map(normalizeTraceEvent);
  const statusText = String((status as TraceEvent).status ?? "not_started");
  const running = Boolean((status as TraceEvent).running) || statusText === "running";
  const latest = normalized.at(-1) ?? null;
  return {
    ok: true,
    action: "scientist_stream_sse_snapshot",
    generated_at: new Date().toISOString(),
    cursor: stepTrace.count,
    scientist_stream: {
      present: normalized.length > 0 || Boolean((status as TraceEvent).present),
      generated_at: new Date().toISOString(),
      running,
      status: running ? "running" : statusText,
      heartbeat: latest?.ts || String((status as TraceEvent).finished_at ?? (status as TraceEvent).started_at ?? ""),
      artifact_path: stepTracePath,
      event_count: stepTrace.count,
      latest_event: latest,
      recent_events: normalized.slice(-30),
      latest_turn: (turns.latest ?? null) as TraceEvent | null,
      latest_terminal_turn: terminalTurn,
      turns_count: turns.count,
      autopilot_status: status,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      transport: "sse",
    },
    scientist_step_trace: {
      present: stepTrace.present,
      artifact_path: stepTrace.artifact_path,
      count: stepTrace.count,
      latest,
      recent: normalized.slice(-30),
    },
    scientist_turns: {
      present: turns.present,
      artifact_path: turns.artifact_path,
      count: turns.count,
      latest: turns.latest ?? null,
      recent: [],
    },
    scientist_terminal_turn: terminalTurn,
    scientist_autopilot_status: status,
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval",
  };
}

function sseMessage(event: string, data: unknown, id?: string | number) {
  const lines = [`event: ${event}`];
  if (id != null) lines.push(`id: ${id}`);
  lines.push(`data: ${JSON.stringify(sanitizeClientJson(data))}`);
  return `${lines.join("\n")}\n\n`;
}

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true }
    );
  });
}

export async function GET(request: Request) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      let cursor = 0;
      let lastHeartbeat = 0;
      const write = (event: string, data: unknown, id?: string | number) => {
        try {
          controller.enqueue(encoder.encode(sseMessage(event, data, id)));
        } catch {
          // The browser may have closed the stream between loop iterations.
        }
      };

      const snapshot = await buildSnapshot();
      cursor = snapshot.cursor;
      write("snapshot", snapshot, "snapshot");

      while (!request.signal.aborted) {
        const lines = await readJsonlLines(stepTracePath);
        if (lines.length > cursor) {
          for (let index = cursor; index < lines.length; index += 1) {
            const parsed = parseJsonLine(lines[index]);
            if (!parsed) continue;
            const event = normalizeTraceEvent(parsed, index + 1);
            write(
              "scientist_event",
              {
                ok: true,
                action: "scientist_stream_event",
                generated_at: new Date().toISOString(),
                event_index: index + 1,
                event_count: lines.length,
                artifact_path: stepTracePath,
                event,
                no_training_started: true,
                official_submit: "blocked_until_explicit_human_approval",
              },
              event.event_id
            );
          }
          cursor = lines.length;
        }

        const now = Date.now();
        if (now - lastHeartbeat >= heartbeatMs) {
          lastHeartbeat = now;
          write("heartbeat", {
            ok: true,
            generated_at: new Date().toISOString(),
            cursor,
            artifact_path: stepTracePath,
            no_training_started: true,
            official_submit: "blocked_until_explicit_human_approval",
          });
        }
        await sleep(pollMs, request.signal);
      }

      try {
        controller.close();
      } catch {
        // Already closed.
      }
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
