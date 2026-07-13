import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const workspaceRoot = path.resolve(process.cwd(), "..", "..");

async function readJson(filePath: string) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8"));
  } catch {
    return {};
  }
}

function buildInstructions(agentLabel: string, agentId: string, taskId: string, outputDir?: string | null) {
  return `# Instructions for ${agentLabel}

- Current task: ${taskId}
- Target coding agent: ${agentId}
- Data and experiment outputs stay inside the local workspace.
- Current output dir: ${outputDir ?? "not available"}
- Current code package: current_code/
- Allowed modification scope: workspace/tasks/${taskId}/code/**, generated scripts, validation helpers.
- Forbidden modification scope: original data files, credentials, Docker secrets, production database files.
- Do not overwrite original data.
- Output a patch, not destructive edits.
- Keep runtime logs.
- Generate metrics.json and artifacts_manifest.json when a run changes behavior.
- Save experiment records.
- Pass submission check before requesting submission approval.
- Bind claims to evidence artifacts.
- Explain why every code change is needed.
`;
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  const body = await request.json().catch(() => ({}));
  const targetAgent = String(body.target_agent ?? body.source_agent ?? "claude_code");
  const summaryPath = path.join(workspaceRoot, "workspace", "workstation_summary.json");
  const summary = await readJson(summaryPath);
  const run = (summary.runs ?? []).find((item: { task_id: string }) => item.task_id === taskId);
  const contextDir = path.join(workspaceRoot, "workspace", "tasks", taskId, "code_agent_context");
  await fs.mkdir(path.join(contextDir, "current_code"), { recursive: true });
  await fs.mkdir(path.join(contextDir, "current_errors"), { recursive: true });

  const taskProfile = {
    task_id: taskId,
    current_goal: "Improve the research task through patch-based code generation.",
    output_dir: run?.output_dir ?? null,
    source_policy: "Do not overwrite original data or pipeline files directly."
  };
  await fs.writeFile(path.join(contextDir, "task_profile.json"), JSON.stringify(taskProfile, null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "scaffold.json"), JSON.stringify({ task_id: taskId, stages: summary.stages ?? [], workspace_contract: "patch_only" }, null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "experiment_history.json"), JSON.stringify((summary.runs ?? []).filter((item: { task_id: string }) => item.task_id === taskId), null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "evidence_index.json"), JSON.stringify((summary.evidence ?? []).filter((item: { task_id: string }) => item.task_id === taskId), null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "eda_summary.json"), JSON.stringify(run?.validation_gate ?? {}, null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "metrics.json"), JSON.stringify(run?.best_metrics ?? {}, null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "current_metrics.json"), JSON.stringify(run?.best_metrics ?? {}, null, 2), "utf-8");
  await fs.writeFile(path.join(contextDir, "current_errors", "error_log.txt"), "", "utf-8");
  await fs.writeFile(path.join(contextDir, "current_code", "README.md"), "Place generated or modified code files here only as patch-backed artifacts.\n", "utf-8");
  await fs.writeFile(
    path.join(contextDir, "experiment_plan.md"),
    "Use the local template baseline first. External Code Agent work must return a patch and preserve evidence bindings.",
    "utf-8"
  );

  await fs.writeFile(path.join(contextDir, "instructions_for_codex.md"), buildInstructions("Codex", "codex", taskId, run?.output_dir), "utf-8");
  await fs.writeFile(path.join(contextDir, "instructions_for_claude_code.md"), buildInstructions("Claude Code", "claude_code", taskId, run?.output_dir), "utf-8");

  const context_dir = path.relative(workspaceRoot, contextDir);
  await logAction({ action: "export_code_agent_context", taskId, message: `Code Agent context exported for ${targetAgent}: ${context_dir}`, artifactPath: context_dir, metadata: { target_agent: targetAgent } });

  return NextResponse.json({
    ok: true,
    task_id: taskId,
    context_dir,
    target_agent: targetAgent
  });
}
