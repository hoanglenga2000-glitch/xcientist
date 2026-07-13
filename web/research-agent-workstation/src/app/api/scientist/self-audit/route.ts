import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_self_audit.json";
const backlogPath = ".xsci/scientist_upgrade_backlog.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readSelfAuditArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      backlog_artifact_path: backlogPath,
      tool: "scientist_self_audit",
      overall_score: 0,
      launch_readiness: "not_run",
      capabilities: [],
      gaps: [],
      upgrade_backlog: [
        {
          id: "run_self_audit_first",
          title: "Run EvoMind self-audit",
          priority: "P0",
          status: "ready",
          safe_next_command: "evomind self-audit",
          gate: "read_only"
        }
      ],
      evidence_sources: {},
      next_safe_commands: ["evomind self-audit"],
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
    scientist_self_audit: await readSelfAuditArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_self_audit_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "self-audit"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_self_audit"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist self-audit error";
    return NextResponse.json(await buildPayload("scientist_self_audit", false, message), { status: 500 });
  }
}
