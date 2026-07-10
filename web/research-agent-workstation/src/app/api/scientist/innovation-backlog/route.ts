import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_innovation_backlog.json";
const innovationLogPath = ".xsci/innovation_log.json";
const selfAuditPath = ".xsci/scientist_self_audit.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readInnovationBacklogArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      innovation_log_path: innovationLogPath,
      tool: "scientist_innovation_backlog",
      selected_task: null,
      memory_summary: {},
      innovation_hypotheses: [],
      next_safe_commands: ["evomind innovate-plan"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readSelfAuditArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(selfAuditPath));
  return payload ? { present: true, artifact_path: selfAuditPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) } : null;
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_innovation_backlog: await readInnovationBacklogArtifact(),
    scientist_self_audit: await readSelfAuditArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_innovation_backlog_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "innovate-plan"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_innovation_backlog"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist innovation backlog error";
    return NextResponse.json(await buildPayload("scientist_innovation_backlog", false, message), { status: 500 });
  }
}
