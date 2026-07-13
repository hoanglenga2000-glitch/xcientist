import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_upgrade_plan.json";
const backlogPath = ".xsci/scientist_upgrade_backlog.json";
const selfAuditPath = ".xsci/scientist_self_audit.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readUpgradePlanArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      source_backlog_path: backlogPath,
      source_self_audit_path: selfAuditPath,
      tool: "scientist_upgrade_plan",
      readiness: "not_run",
      open_backlog_count: 0,
      planned_steps: [],
      execution_policy: {
        mode: "engineering_plan_only",
        training: "blocked",
        submit: "blocked_until_explicit_user_approval"
      },
      next_safe_commands: ["evomind self-audit", "evomind upgrade-plan"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist upgrade plan has not been generated yet."
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
    scientist_upgrade_plan: await readUpgradePlanArtifact(),
    scientist_self_audit: await readSelfAuditArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_upgrade_plan_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "upgrade-plan"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_upgrade_plan"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist upgrade-plan error";
    return NextResponse.json(await buildPayload("scientist_upgrade_plan", false, message), { status: 500 });
  }
}
