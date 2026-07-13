import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const continuationPath = ".xsci/scientist_continuation.json";
const continuationStatusPath = ".xsci/scientist_continuation_status.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readJsonArtifact(relativePath: string, tool: string, fallback: Record<string, unknown> = {}) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      ...fallback
    };
  }
  return { present: true, artifact_path: relativePath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readContinuationStatusArtifact() {
  return readJsonArtifact(continuationStatusPath, "scientist_continuation_status", {
    status: "no_continuation",
    selected_task: null,
    completion_ratio: 0,
    total_required_tools: 0,
    completed_required_tools: 0,
    remaining_count: 0,
    remaining_safe_tools: [],
    executed_or_completed_tools: [],
    progress_history: [],
    next_safe_action_command: "evomind turn",
    message: "No Scientist continuation status artifact yet. Run a Scientist turn or loop first."
  });
}

async function readContinuationArtifact() {
  return readJsonArtifact(continuationPath, "scientist_continuation", {
    status: "not_found",
    required_safe_tools: [],
    completed_safe_tools: [],
    progress_history: []
  });
}

async function readScientistStepTrace(limit = 50) {
  const text = await fs.readFile(resolveWorkspacePath(stepTracePath), "utf-8").catch(() => "");
  const events = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as Record<string, unknown>;
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .slice(-limit);
  return {
    present: events.length > 0,
    artifact_path: stepTracePath,
    count: events.length,
    latest: events.at(-1) ?? null,
    recent: sanitizeClientJson(events)
  };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_continuation_status: await readContinuationStatusArtifact(),
    scientist_continuation: await readContinuationArtifact(),
    scientist_action_queue: await readJsonArtifact(actionQueuePath, "scientist_action_queue"),
    scientist_step_trace: await readScientistStepTrace(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_continuation_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "continuation-status"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_continuation_status_refresh"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist continuation-status error";
    return NextResponse.json(await buildPayload("scientist_continuation_status_refresh", false, message), { status: 500 });
  }
}
