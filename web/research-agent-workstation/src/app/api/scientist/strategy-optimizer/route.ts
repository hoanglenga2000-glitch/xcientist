import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_strategy_optimizer.json";
const markdownPath = ".xsci/scientist_strategy_optimizer.md";
const readinessPath = ".xsci/scientist_readiness_report.json";
const causalPath = ".xsci/scientist_causal_diagnosis.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? (sanitizeClientJson(payload) as Record<string, unknown>) : null;
}

async function readStrategyArtifact() {
  const payload = await readJsonArtifact(artifactPath);
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_strategy_optimizer",
      strategy_posture: "not_run",
      selected_strategy: null,
      intervention_ranking: [],
      decision_matrix: { candidate_count: 0, source_presence: {} },
      next_safe_command: "evomind strategy",
      claim_boundary: {
        rank_or_medal: "blocked_without_kaggle_response_artifact",
        official_submit: "blocked_until_explicit_human_approval"
      },
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: artifactPath, ...(payload as Record<string, unknown>) };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_strategy_optimizer: await readStrategyArtifact(),
    scientist_readiness_report: await readJsonArtifact(readinessPath),
    scientist_causal_diagnosis: await readJsonArtifact(causalPath),
    scientist_action_queue: await readJsonArtifact(actionQueuePath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_strategy_optimizer_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "strategy"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_strategy_optimizer"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist strategy optimizer error";
    return NextResponse.json(await buildPayload("scientist_strategy_optimizer", false, message), { status: 500 });
  }
}
