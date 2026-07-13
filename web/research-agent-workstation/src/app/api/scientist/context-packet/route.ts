import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_context_packet.json";
const markdownPath = ".xsci/scientist_context_packet.md";
const strategyPath = ".xsci/scientist_strategy_optimizer.json";
const readinessPath = ".xsci/scientist_readiness_report.json";
const actionQueuePath = ".xsci/scientist_action_queue.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readJsonArtifact(relativePath: string) {
  const payload = await readJsonFile(resolveWorkspacePath(relativePath));
  return payload ? (sanitizeClientJson(payload) as Record<string, unknown>) : null;
}

async function readContextPacketArtifact() {
  const payload = await readJsonArtifact(artifactPath);
  if (!payload) {
    return {
      present: false,
      artifact_path: artifactPath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_context_packet",
      schema: "evomind.ai_scientist.context_packet.v1",
      context_quality: {
        score: 0,
        present_artifacts: 0,
        missing_sources: ["scientist_context_packet.json"],
        interpretation: "not_run"
      },
      readiness: {
        can_execute: false,
        blocking_gates: []
      },
      active_strategy: {
        present: false,
        selected_command: "evomind briefing",
        gate_status: "safe_read_only"
      },
      memory_digest: {
        retrospective_records: 0,
        recent_lessons: []
      },
      next_safe_command: "evomind briefing",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return { present: true, artifact_path: artifactPath, markdown_artifact_path: markdownPath, ...(payload as Record<string, unknown>) };
}

async function buildPayload(action: string, ok = true, error?: string) {
  return {
    ok,
    action,
    ...(error ? { error } : {}),
    scientist_context_packet: await readContextPacketArtifact(),
    scientist_strategy_optimizer: await readJsonArtifact(strategyPath),
    scientist_readiness_report: await readJsonArtifact(readinessPath),
    scientist_action_queue: await readJsonArtifact(actionQueuePath),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  };
}

export async function GET() {
  return NextResponse.json(await buildPayload("scientist_context_packet_status"));
}

export async function POST() {
  try {
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    await execFileAsync(pythonExecutable(), ["-m", "xsci.kaggle", "briefing"], {
      cwd: workspaceRoot,
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    return NextResponse.json(await buildPayload("scientist_context_packet"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Scientist context packet error";
    return NextResponse.json(await buildPayload("scientist_context_packet", false, message), { status: 500 });
  }
}
