import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const loopPath = ".xsci/scientist_loop.json";
const lessonsPath = ".xsci/scientist_loop_lessons.jsonl";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";
const recoveryPath = ".xsci/scientist_recovery_snapshot.json";
const memoryConsolidationPath = ".xsci/scientist_memory_consolidation.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

function parseJsonl(text: string, limit = 20) {
  return text
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
}

async function readLoopArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(loopPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: loopPath,
      tool: "scientist_loop",
      mode: "not_run",
      stop_reason: "not_run",
      selected_task: null,
      trace_run_id: null,
      steps: [],
      final_autopilot: null,
      final_next_action: null,
      lesson: null,
      lessons_path: lessonsPath,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: loopPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readLessonsArtifact() {
  const text = await fs.readFile(resolveWorkspacePath(lessonsPath), "utf-8").catch(() => "");
  const lessons = parseJsonl(text, 20);
  return {
    present: lessons.length > 0,
    artifact_path: lessonsPath,
    count: lessons.length,
    latest: lessons.at(-1) ?? null,
    recent: sanitizeClientJson(lessons)
  };
}

async function readJsonArtifact(relativePath: string, tool: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  if (!payload) {
    return { present: false, artifact_path: relativePath, tool };
  }
  return { present: true, artifact_path: relativePath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readStepTrace(limit = 50) {
  const text = await fs.readFile(resolveWorkspacePath(stepTracePath), "utf-8").catch(() => "");
  const events = parseJsonl(text, limit);
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
    scientist_loop: await readLoopArtifact(),
    scientist_loop_lessons: await readLessonsArtifact(),
    scientist_action_queue: await readJsonArtifact(actionQueuePath, "scientist_action_queue"),
    scientist_step_trace: await readStepTrace(),
    scientist_recovery: await readJsonArtifact(recoveryPath, "scientist_recovery"),
    scientist_memory_consolidation: await readJsonArtifact(memoryConsolidationPath, "scientist_memory_consolidation"),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_loop_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "loop"], {
      cwd: workspaceRoot,
      timeout: 120_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_loop"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist loop error";
    return NextResponse.json(await buildPayload("scientist_loop", false, message), { status: 500 });
  }
}
