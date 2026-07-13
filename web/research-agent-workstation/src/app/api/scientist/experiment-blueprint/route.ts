import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_experiment_blueprint.json";
const reviewPath = ".xsci/scientist_hypothesis_review.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readExperimentBlueprintArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      source_review_path: reviewPath,
      tool: "scientist_experiment_blueprint",
      selected_task: null,
      blueprint_status: "not_run",
      selected_hypothesis: null,
      experiment_blueprint: null,
      gate_summary: {},
      next_safe_commands: ["evomind blueprint"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_experiment_blueprint: await readExperimentBlueprintArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_experiment_blueprint_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "blueprint"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_experiment_blueprint"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist experiment blueprint error";
    return NextResponse.json(await buildPayload("scientist_experiment_blueprint", false, message), { status: 500 });
  }
}
