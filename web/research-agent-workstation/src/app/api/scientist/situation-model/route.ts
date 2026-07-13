import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_situation_model.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readSituationModelArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      ok: true,
      artifact_path: artifactPath,
      tool: "scientist_situation_model",
      selected_task: null,
      situation_status: "not_run",
      situation_model: null,
      readiness_score: 0,
      blockers: [],
      next_safe_commands: ["evomind situation"],
      source_artifacts: [],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist situation model has not been generated yet."
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_situation_model: await readSituationModelArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_situation_model_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "situation"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_situation_model"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist situation model error";
    return NextResponse.json(await buildPayload("scientist_situation_model", false, message), { status: 500 });
  }
}
