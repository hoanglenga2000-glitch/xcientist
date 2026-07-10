import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_autopilot.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";
const workplanPath = ".xsci/scientist_workplan.json";
const repairPlanPath = ".xsci/scientist_repair_plan.json";
const executionContractPath = ".xsci/scientist_execution_contract.json";
const stepTracePath = ".xsci/scientist_step_trace.jsonl";
const statusPath = ".xsci/scientist_autopilot_status.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readAutopilotArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      mode: "not_run",
      summary_lines: [],
      tool_trace: [],
      next_actions: ["Run Scientist Autopilot to create the first diagnosis artifact."],
      blockers: [],
      human_gate: {
        official_kaggle_submit: "blocked_until_explicit_user_approval",
        rank_or_medal_claims: "blocked_without_kaggle_response_artifact"
      }
    };
  }
  return { present: true, artifact_path: artifactPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
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
          expected_artifacts: [artifactPath, actionQueuePath],
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

async function readWorkplanArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(workplanPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: workplanPath,
      mode: "not_run",
      current_focus: null,
      summary: { steps_total: 0, completed: 0, ready: 0, pending: 0, blocked: 0 },
      steps: [],
      resume_commands: ["Run Scientist Autopilot or `evomind workplan` to create the first workplan artifact."],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: workplanPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
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
      official_submit: "blocked_until_explicit_human_approval",
      claim_boundary: "No execution-contract evidence has been generated yet."
    };
  }
  return { present: true, artifact_path: executionContractPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

async function readScientistTurns(limit = 10) {
  const fs = await import("node:fs/promises");
  const turnsPath = resolveWorkspacePath(".xsci/scientist_turns.jsonl");
  const text = await fs.readFile(turnsPath, "utf-8").catch(() => "");
  const turns = text
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
    present: turns.length > 0,
    artifact_path: ".xsci/scientist_turns.jsonl",
    count: turns.length,
    latest: turns.at(-1) ?? null,
    recent: sanitizeClientJson(turns)
  };
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

async function readAutopilotStatus() {
  const payload = await readJsonFile(resolveWorkspacePath(statusPath));
  if (!payload) {
    return {
      present: false,
      artifact_path: statusPath,
      running: false,
      status: "not_started",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: statusPath, ...(sanitizeClientJson(payload) as Record<string, unknown>) };
}

export async function GET() {
  return NextResponse.json({
    ok: true,
    scientist_autopilot: await readAutopilotArtifact(),
    scientist_action_queue: await readActionQueueArtifact(),
    scientist_workplan: await readWorkplanArtifact(),
    scientist_repair_plan: await readRepairPlanArtifact(),
    scientist_execution_contract: await readExecutionContractArtifact(),
    scientist_turns: await readScientistTurns(),
    scientist_step_trace: await readScientistStepTrace(),
    scientist_autopilot_status: await readAutopilotStatus()
  });
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "autopilot"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json({
      ok: true,
      action: "scientist_autopilot",
      scientist_autopilot: await readAutopilotArtifact(),
      scientist_action_queue: await readActionQueueArtifact(),
      scientist_workplan: await readWorkplanArtifact(),
      scientist_repair_plan: await readRepairPlanArtifact(),
      scientist_execution_contract: await readExecutionContractArtifact(),
      scientist_turns: await readScientistTurns(),
      scientist_step_trace: await readScientistStepTrace(),
      scientist_autopilot_status: await readAutopilotStatus()
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist Autopilot error";
    return NextResponse.json({
      ok: false,
      action: "scientist_autopilot",
      error: message,
      scientist_autopilot: await readAutopilotArtifact(),
      scientist_action_queue: await readActionQueueArtifact(),
      scientist_workplan: await readWorkplanArtifact(),
      scientist_repair_plan: await readRepairPlanArtifact(),
      scientist_execution_contract: await readExecutionContractArtifact(),
      scientist_turns: await readScientistTurns(),
      scientist_step_trace: await readScientistStepTrace(),
      scientist_autopilot_status: await readAutopilotStatus()
    }, { status: 500 });
  }
}
