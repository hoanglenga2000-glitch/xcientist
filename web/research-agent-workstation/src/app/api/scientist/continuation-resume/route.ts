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
const continuationResumePath = ".xsci/scientist_continuation_resume.json";
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
    scientist_continuation_resume: await readJsonArtifact(continuationResumePath, "scientist_continuation_resume", {
      status: "not_run",
      steps_executed: 0,
      remaining_safe_tools: []
    }),
    scientist_continuation_status: await readJsonArtifact(continuationStatusPath, "scientist_continuation_status", {
      status: "no_continuation",
      remaining_safe_tools: [],
      executed_or_completed_tools: [],
      progress_history: []
    }),
    scientist_continuation: await readJsonArtifact(continuationPath, "scientist_continuation"),
    scientist_action_queue: await readJsonArtifact(actionQueuePath, "scientist_action_queue"),
    scientist_step_trace: await readScientistStepTrace(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_continuation_resume_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "resume-continuation"], {
      cwd: workspaceRoot,
      timeout: 120_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_continuation_resume"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist continuation-resume error";
    return NextResponse.json(await buildPayload("scientist_continuation_resume", false, message), { status: 500 });
  }
}
