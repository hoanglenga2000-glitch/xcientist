import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_readiness_report.json";
const markdownPath = ".xsci/scientist_readiness_report.md";
const selfAuditPath = ".xsci/scientist_self_audit.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? (sanitizeClientJson(payload) as Record<string, unknown>) : null;
}

async function readReadinessReportArtifact() {
  const payload = await readJsonArtifact(artifactPath);
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_readiness_report",
      overall_score: 0,
      capability_readiness: "not_run",
      launch_readiness: "not_run",
      claim_readiness: {
        training_readiness_claim: "not_run",
        rank_or_medal_claim: "blocked_without_kaggle_response_artifact",
        official_submit_claim: "blocked_until_explicit_human_approval"
      },
      readiness_matrix: [],
      recommended_next_commands: ["evomind readiness-report", "evomind self-audit"],
      artifact_evidence: [],
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
    scientist_readiness_report: await readReadinessReportArtifact(),
    scientist_self_audit: await readJsonArtifact(selfAuditPath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_readiness_report_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "readiness-report"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_readiness_report"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist readiness-report error";
    return NextResponse.json(await buildPayload("scientist_readiness_report", false, message), { status: 500 });
  }
}
