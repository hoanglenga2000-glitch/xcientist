import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const stepTracePath = ".xsci/scientist_step_trace.jsonl";
const turnsPath = ".xsci/scientist_turns.jsonl";
const statusPath = ".xsci/scientist_autopilot_status.json";
const terminalTurnPath = ".xsci/scientist_terminal_turn.json";

async function readJsonlTail(relativePath: string, limit = 80) {
  const text = await fs.readFile(resolveWorkspacePath(relativePath), "utf-8").catch(() => "");
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const parsed = lines.slice(-limit).map((line) => {
    try {
      return JSON.parse(line) as Record<string, unknown>;
    } catch {
      return null;
    }
  }).filter(Boolean) as Array<Record<string, unknown>>;
  return {
    present: parsed.length > 0,
    artifact_path: relativePath,
    count: lines.length,
    latest: parsed.at(-1) ?? null,
    recent: sanitizeClientJson(parsed)
  };
}

function normalizeTraceEvent(event: Record<string, unknown>, index: number) {
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

async function readAutopilotStatus() {
  const payload = await readJsonFile(resolveWorkspacePath(statusPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: statusPath,
      running: false,
      status: "not_started",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: statusPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
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
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: terminalTurnPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

export async function GET() {
  const [stepTrace, turns, status, terminalTurn] = await Promise.all([
    readJsonlTail(stepTracePath, 80),
    readJsonlTail(turnsPath, 20),
    readAutopilotStatus(),
    readTerminalTurn(),
  ]);
  const recent = Array.isArray(stepTrace.recent) ? stepTrace.recent as Array<Record<string, unknown>> : [];
  const normalized = recent.map(normalizeTraceEvent);
  const statusText = String((status as Record<string, unknown>).status ?? "not_started");
  const running = Boolean((status as Record<string, unknown>).running) || statusText === "running";
  const latest = normalized.at(-1) ?? null;
  return NextResponse.json({
    ok: true,
    action: "scientist_stream",
    scientist_stream: {
      present: normalized.length > 0 || Boolean((status as Record<string, unknown>).present),
      generated_at: new Date().toISOString(),
      running,
      status: running ? "running" : statusText,
      heartbeat: latest?.ts || String((status as Record<string, unknown>).finished_at ?? (status as Record<string, unknown>).started_at ?? ""),
      artifact_path: stepTracePath,
      event_count: stepTrace.count,
      latest_event: latest,
      recent_events: normalized.slice(-30),
      latest_turn: (turns.latest ?? null) as Record<string, unknown> | null,
      latest_terminal_turn: terminalTurn,
      turns_count: turns.count,
      autopilot_status: status,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
    },
    scientist_step_trace: stepTrace,
    scientist_turns: turns,
    scientist_terminal_turn: terminalTurn,
    scientist_autopilot_status: status,
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  });
}
