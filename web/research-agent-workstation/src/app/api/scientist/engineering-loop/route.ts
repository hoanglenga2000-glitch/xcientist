import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const artifactPath = ".xsci/scientist_engineering_loop.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

async function readArtifact() {
  const payload = await readJsonFile(resolveWorkspacePath(artifactPath));
  if (!payload) {
    return {
      present: false,
      tool: "scientist_engineering_loop",
      status: "not_run",
      changed_files: [],
      acceptance_checks: [],
      main_worktree_modified: false,
      merge_ready: false,
      artifact_path: artifactPath,
      next_safe_command: "evomind patch-order",
      human_gate: "review_candidate_before_merge",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: artifactPath,
    ...(payload as Record<string, unknown>)
  });
}

export async function GET() {
  return NextResponse.json({
    ok: true,
    action: "scientist_engineering_loop_status",
    scientist_engineering_loop: await readArtifact(),
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  });
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({})) as Record<string, unknown>;
  const args = ["-m", "xsci.kaggle", "engineer", "--json"];
  if (body.generate_patch === true) args.push("--generate");
  if (typeof body.patch_path === "string" && body.patch_path.trim()) {
    args.push("--patch", body.patch_path.trim());
  }
  if (typeof body.work_order_path === "string" && body.work_order_path.trim()) {
    args.push("--work-order", body.work_order_path.trim());
  }
  if (typeof body.dashboard_url === "string" && body.dashboard_url.trim()) {
    args.push("--dashboard-url", body.dashboard_url.trim());
  }
  const timeoutSeconds = Math.max(30, Math.min(900, Number(body.timeout_seconds ?? 240) || 240));
  args.push("--timeout", String(timeoutSeconds));

  const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  let cliResult: Record<string, unknown> | null = null;
  let cliError = "";
  try {
    const { stdout } = await execFileAsync(pythonExecutable(), args, {
      cwd: workspaceRoot,
      timeout: (timeoutSeconds + 60) * 1000,
      maxBuffer: 4 * 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    });
    cliResult = JSON.parse(stdout) as Record<string, unknown>;
  } catch (error) {
    cliError = error instanceof Error ? error.message : "Engineering loop CLI failed";
  }

  const artifact = await readArtifact();
  const ok = Boolean((artifact as Record<string, unknown>).ok);
  return NextResponse.json({
    ok,
    action: "scientist_engineering_loop",
    ...(cliResult ? { cli_result: sanitizeClientJson(cliResult) } : {}),
    ...(cliError ? { cli_error: cliError } : {}),
    scientist_engineering_loop: artifact,
    no_training_started: true,
    official_submit: "blocked_until_explicit_human_approval"
  });
}
