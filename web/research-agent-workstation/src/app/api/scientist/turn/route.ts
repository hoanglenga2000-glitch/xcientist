import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const terminalTurnPath = ".xsci/scientist_terminal_turn.json";
const contextPacketPath = ".xsci/scientist_context_packet.json";
const strategyPath = ".xsci/scientist_strategy_optimizer.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

function normalizePrompt(value: unknown) {
  const prompt = typeof value === "string" ? value.trim() : "";
  return (prompt || "Analyze the current EvoMind workstation state and propose the next safe research step.").slice(0, 4000);
}

function normalizeMaxTools(value: unknown) {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return 4;
  return Math.max(1, Math.min(8, Math.trunc(parsed)));
}

async function readTerminalTurnArtifact() {
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
      next_safe_command: "evomind ask \"analyze the current state\"",
      execution_ready: false,
      execution_blocked: true,
      blocking_gates: ["No Scientist terminal turn has been generated yet."],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Run a Scientist Turn to generate this artifact."
    };
  }
  return { present: true, artifact_path: terminalTurnPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? (sanitizeClientJson(payload) as Record<string, unknown>) : null;
}

async function readJsonlTail(relativePath: string, limit = 50) {
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

async function buildPayload(action: string, ok = true, error?: string, cliPayload?: Record<string, unknown> | null) {
  const scientistTerminalTurn = await readTerminalTurnArtifact();
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    ...(cliPayload ? { cli_result: sanitizeClientJson(cliPayload) } : {}),
    scientist_terminal_turn: scientistTerminalTurn,
    scientist_turn: scientistTerminalTurn,
    scientist_context_packet: await readJsonArtifact(contextPacketPath),
    scientist_strategy_optimizer: await readJsonArtifact(strategyPath),
    scientist_action_queue: await readJsonArtifact(actionQueuePath),
    scientist_step_trace: await readJsonlTail(stepTracePath, 50),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_terminal_turn_status"));
}

export async function POST(request: Request) {
  let body: Record<string, unknown> = {};
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    body = {};
  }

  const prompt = normalizePrompt(body.prompt ?? body.goal ?? body.user_goal);
  const maxTools = normalizeMaxTools(body.max_tools ?? body.maxTools);

  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const { stdout } = await execFileAsync(
      pythonExecutable(),
      ["-m", "xsci.kaggle", "ask", "--json", "--max-tools", String(maxTools), prompt],
      {
        cwd: workspaceRoot,
        timeout: 90_000,
        maxBuffer: 2 * 1024 * 1024,
        windowsHide: true,
        env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
      }
    );
    let cliPayload: Record<string, unknown> | null = null;
    try {
      cliPayload = JSON.parse(stdout) as Record<string, unknown>;
    } catch {
      cliPayload = { raw_stdout_preview: stdout.slice(0, 1200) };
    }
    return NextResponse.json(await buildPayload("scientist_terminal_turn", true, undefined, cliPayload));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist terminal turn error";
    return NextResponse.json(await buildPayload("scientist_terminal_turn", false, message), { status: 500 });
  }
}
