import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_patch_work_order.json";
const actionQueuePath = ".xsci/scientist_patch_action_queue.json";
const trialsPath = ".xsci/scientist_patch_trials.jsonl";
const terminalTurnPath = ".xsci/scientist_terminal_turn.json";
const parityLoopPath = ".xsci/scientist_latest_parity_loop.json";
const repairPlanPath = ".xsci/scientist_repair_plan.json";
const selfAuditPath = ".xsci/scientist_self_audit.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readPatchWorkOrderArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      action_queue_path: actionQueuePath,
      trials_path: trialsPath,
      tool: "scientist_patch_work_order",
      status: "not_run",
      selected_issue_id: "",
      selected_title: "",
      work_order: null,
      action_queue: null,
      next_safe_commands: ["evomind patch-order", "evomind ask --json \"复核当前系统智能体闭环，不启动训练\"", "evomind self-audit"],
      source_artifacts: [terminalTurnPath, parityLoopPath, repairPlanPath, selfAuditPath],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist patch work order has not been generated yet."
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
    scientist_patch_work_order: await readPatchWorkOrderArtifact(),
    scientist_action_queue: await readJsonArtifact(actionQueuePath),
    scientist_terminal_turn: await readJsonArtifact(terminalTurnPath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_patch_work_order_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "patch-order"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_patch_work_order"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist patch work order error";
    return NextResponse.json(await buildPayload("scientist_patch_work_order", false, message), { status: 500 });
  }
}
