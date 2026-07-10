import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_causal_diagnosis.json";
const markdownPath = ".xsci/scientist_causal_diagnosis.md";
const readinessPath = ".xsci/scientist_readiness_report.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? (sanitizeClientJson(payload) as Record<string, unknown>) : null;
}

async function readCausalArtifact() {
  const payload = await readJsonArtifact(artifactPath);
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_causal_diagnosis",
      posture: "not_run",
      symptoms: [],
      root_causes: [],
      interventions: [],
      causal_graph: { nodes: [], edges: [] },
      next_safe_command: "evomind causal-diagnosis",
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
    scientist_causal_diagnosis: await readCausalArtifact(),
    scientist_readiness_report: await readJsonArtifact(readinessPath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_causal_diagnosis_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "causal-diagnosis"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_causal_diagnosis"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist causal-diagnosis error";
    return NextResponse.json(await buildPayload("scientist_causal_diagnosis", false, message), { status: 500 });
  }
}
