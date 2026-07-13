import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_turn_plan.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readTurnPlanArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      ok: true,
      artifact_path: artifactPath,
      tool: "scientist_turn_plan",
      selected_task: null,
      intent: { kind: "not_run" },
      autonomy_level: "not_run",
      selected_tools: [],
      tool_sequence: [],
      expected_artifacts: [],
      stop_conditions: [],
      next_safe_command: "evomind turn-plan",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist turn plan has not been generated yet."
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_turn_plan: await readTurnPlanArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_turn_plan_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "turn-plan"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_turn_plan"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist turn plan error";
    return NextResponse.json(await buildPayload("scientist_turn_plan", false, message), { status: 500 });
  }
}
