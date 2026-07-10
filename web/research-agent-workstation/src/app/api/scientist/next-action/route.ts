import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const nextActionPath = ".xsci/scientist_next_action.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readNextActionArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(nextActionPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: nextActionPath,
      tool: "scientist_next_action",
      status: "not_run",
      selected_action: null,
      message: "No safe-next artifact yet.",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: nextActionPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readActionQueueArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(actionQueuePath));
  if (!payload) {
    return {
      present: false,
      artifact_path: actionQueuePath,
      tool: "scientist_action_queue",
      selected_task: null,
      actions: [
        {
          id: "run_autopilot_first",
          title: "Run AI Scientist Autopilot",
          status: "ready",
          command: "evomind autopilot",
          gate: "read_only",
          why: "Create the first action queue from current system, task, data, memory, and gate evidence.",
          risk: "none; read-only diagnosis",
          rollback_condition: "stay in planner mode until action queue exists",
          expected_artifacts: [".xsci/scientist_autopilot.json", actionQueuePath],
          evidence: ["scientist_autopilot", "scientist_step_trace"],
          autonomy: "read_only"
        }
      ],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: actionQueuePath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readScientistStepTrace(limit = 50) {
  const fs = await import("node:fs/promises");
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

export async function GET() {
  return NextResponse.json({
    ok: true,
    action: "scientist_next_action_status",
    scientist_next_action: await readNextActionArtifact(),
    scientist_action_queue: await readActionQueueArtifact(),
    scientist_step_trace: await readScientistStepTrace(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  });
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "next"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json({
      ok: true,
      action: "scientist_next_action",
      scientist_next_action: await readNextActionArtifact(),
      scientist_action_queue: await readActionQueueArtifact(),
      scientist_step_trace: await readScientistStepTrace(),
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist next-action error";
    return NextResponse.json({
      ok: false,
      action: "scientist_next_action",
      error: message,
      scientist_next_action: await readNextActionArtifact(),
      scientist_action_queue: await readActionQueueArtifact(),
      scientist_step_trace: await readScientistStepTrace(),
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: 500 });
  }
}
