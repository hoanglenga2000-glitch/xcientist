import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_self_upgrade_loop.json";
const workOrderPath = ".xsci/scientist_self_upgrade_work_order.json";
const actionQueuePath = ".xsci/scientist_self_upgrade_action_queue.json";
const trialsPath = ".xsci/scientist_self_upgrade_trials.jsonl";
const upgradePlanPath = ".xsci/scientist_upgrade_plan.json";
const selfAuditPath = ".xsci/scientist_self_audit.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readSelfUpgradeArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      work_order_path: workOrderPath,
      action_queue_path: actionQueuePath,
      trials_path: trialsPath,
      source_upgrade_plan_path: upgradePlanPath,
      source_self_audit_path: selfAuditPath,
      tool: "scientist_self_upgrade_loop",
      status: "not_run",
      selected_backlog_id: "",
      open_backlog_count: 0,
      work_order: null,
      action_queue: null,
      loop_phases: [],
      next_safe_commands: ["evomind self-audit", "evomind upgrade-plan", "evomind self-upgrade"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist self-upgrade loop has not been generated yet."
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? { present: true, artifact_path: relativePath, ...(sanitizeClientJson(payload) as Record<string, unknown>) } : null;
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_self_upgrade_loop: await readSelfUpgradeArtifact(),
    scientist_upgrade_plan: await readJsonArtifact(upgradePlanPath),
    scientist_self_audit: await readJsonArtifact(selfAuditPath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_self_upgrade_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "self-upgrade"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_self_upgrade"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist self-upgrade error";
    return NextResponse.json(await buildPayload("scientist_self_upgrade", false, message), { status: 500 });
  }
}
