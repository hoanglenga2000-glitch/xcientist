import { spawn } from "node:child_process";
import path from "node:path";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, stamp, workspaceRoot, writeJsonArtifact } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const statusPath = ".xsci/scientist_autopilot_status.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";
const repairPlanPath = ".xsci/scientist_repair_plan.json";
const executionContractPath = ".xsci/scientist_execution_contract.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readScientistStepTrace(limit = 50) {
  const fs = await import("node:fs/promises");
  const text = await fs.readFile(resolveWorkspacePath(stepTracePath), "utf-8").catch(() => "");
  const events = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as Record<string, unknown>;
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .slice(-limit);
  return {
    present: events.length > 0,
    artifact_path: stepTracePath,
    count: events.length,
    latest: events.at(-1) ?? null,
    recent: sanitizeClientJson(events)
  };
}

async function readActionQueueArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(actionQueuePath));
  if (!payload) {
    return {
      present: false,
      artifact_path: actionQueuePath,
      tool: "scientist_action_queue",
      selected_task: null,
      actions: [
        {
          id: "run_autopilot_first",
          title: "Run AI Scientist Autopilot",
          status: "ready",
          command: "evomind autopilot",
          gate: "read_only",
          why: "Create the first action queue from current system, task, data, memory, and gate evidence.",
          risk: "none; read-only diagnosis",
          rollback_condition: "stay in planner mode until action queue exists",
          expected_artifacts: [".xsci/scientist_autopilot.json", actionQueuePath],
          evidence: ["scientist_autopilot", "scientist_step_trace"],
          autonomy: "read_only"
        }
      ],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: actionQueuePath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readRepairPlanArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(repairPlanPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: repairPlanPath,
      mode: "not_run",
      diagnosis: [],
      root_causes: [],
      repair_steps: [],
      safe_next_command: "Run Scientist Autopilot or `evomind repair` to create the first repair plan.",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: repairPlanPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readExecutionContractArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(executionContractPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: executionContractPath,
      go_no_go: "not_run",
      agent_session_ready: false,
      model_training_ready: false,
      data_contract_status: "unknown",
      required_artifacts: [],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: executionContractPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function writeStatus(payload: Record<string, unknown>) {
  await writeJsonArtifact(statusPath, {
    artifact_path: statusPath,
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval",
    ...payload
  });
}

export async function POST() {
  const runId = `ui_autopilot_${stamp()}`;
  const startedAt = new Date().toISOString();
  const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);

  try {
    await writeStatus({
      present: true,
      running: true,
      status: "starting",
      run_id: runId,
      pid: null,
      started_at: startedAt,
      finished_at: null,
      exit_code: null,
      signal: null,
      message: "Scientist Autopilot is starting in the background."
    });

    const child = spawn(pythonExecutable(), ["-m", "xsci.kaggle", "autopilot"], {
      cwd: workspaceRoot,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" },
      stdio: "ignore"
    });

    child.on("exit", (code, signal) => {
      void writeStatus({
        present: true,
        running: false,
        status: code === 0 ? "completed" : "failed",
        run_id: runId,
        pid: child.pid ?? null,
        started_at: startedAt,
        finished_at: new Date().toISOString(),
        exit_code: code,
        signal: signal ?? null,
        message: code === 0 ? "Scientist Autopilot completed." : "Scientist Autopilot failed."
      });
    });

    child.on("error", (error) => {
      void writeStatus({
        present: true,
        running: false,
        status: "failed",
        run_id: runId,
        pid: child.pid ?? null,
        started_at: startedAt,
        finished_at: new Date().toISOString(),
        error: error.message,
        message: "Scientist Autopilot failed to start."
      });
    });

    child.unref();

    return NextResponse.json({
      ok: true,
      action: "scientist_autopilot_start",
      run_id: runId,
      pid: child.pid ?? null,
      status_artifact: statusPath,
      scientist_autopilot_status: {
        present: true,
        artifact_path: statusPath,
        running: true,
        status: "starting",
        run_id: runId,
        pid: child.pid ?? null,
        started_at: startedAt,
        finished_at: null,
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      },
      scientist_action_queue: await readActionQueueArtifact(),
      scientist_step_trace: await readScientistStepTrace(),
      scientist_repair_plan: await readRepairPlanArtifact(),
      scientist_execution_contract: await readExecutionContractArtifact(),
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist Autopilot start error";
    await writeStatus({
      present: true,
      running: false,
      status: "failed",
      run_id: runId,
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      error: message,
      message: "Scientist Autopilot failed before spawning."
    });
    return NextResponse.json({
      ok: false,
      action: "scientist_autopilot_start",
      error: message,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: 500 });
  }
}
