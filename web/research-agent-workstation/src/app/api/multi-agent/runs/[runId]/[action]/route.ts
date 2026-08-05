import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { cancelRunningJob, runningJob, runManagedCommand } from "@/lib/server/job-registry";
import { resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";
import { generateScientificReport } from "@/lib/server/scientific-report";

export const dynamic = "force-dynamic";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || (process.platform === "win32" ? "C:\\codex-python\\python.exe" : "python3");
}

export async function POST(_request: Request, { params }: { params: Promise<{ runId: string; action: string }> }) {
  const { runId, action } = await params;
  if (!/^[A-Za-z0-9_-]{8,160}$/.test(runId) || !["pause", "resume", "cancel"].includes(action)) {
    return NextResponse.json({ ok: false, error: "invalid run control request" }, { status: 400 });
  }
  const runDir = resolveWorkspacePath(`workspace/evomind_runs/${runId}`);
  if (!await fs.stat(runDir).then((value) => value.isDirectory()).catch(() => false)) {
    return NextResponse.json({ ok: false, error: "run not found" }, { status: 404 });
  }
  if (action === "resume") {
    const activeJob = runningJob("multi-agent", runId);
    if (activeJob) {
      return NextResponse.json({
        ok: false,
        error: "already_resuming",
        run_id: runId,
        action,
        status: "already_resuming",
        active_job: activeJob,
      }, { status: 409 });
    }
    const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const refinement = await fs.readFile(`${runDir}/refinement.json`, "utf-8")
      .then((value) => JSON.parse(value) as Record<string, unknown>)
      .catch(() => null);
    const reportTaskId = typeof refinement?.task_id === "string" && /^[A-Za-z0-9_-]{1,160}$/.test(refinement.task_id)
      ? refinement.task_id
      : null;
    void runManagedCommand({
      command: pythonExecutable(),
      args: ["-m", "xsci.multi_agent_cli", "resume", "--run-id", runId],
      cwd: workspaceRoot,
      env: { ...process.env, PYTHONPATH: pythonPath },
      timeout: 45 * 60 * 1000,
      taskId: "multi-agent",
      runId
    }).then(async () => {
      if (reportTaskId) await generateScientificReport(reportTaskId, runId, { automated: true });
    }).catch(() => undefined);
    return NextResponse.json({ ok: true, run_id: runId, action, status: "resuming" }, { status: 202 });
  }
  await fs.writeFile(`${runDir}/control.json`, JSON.stringify({ schema: "evomind.multi_agent.control.v1", action }, null, 2) + "\n", "utf-8");
  const cancellation = action === "cancel" ? cancelRunningJob("multi-agent", runId) : null;
  return NextResponse.json({
    ok: true,
    run_id: runId,
    action,
    status: action === "pause" ? "pause_requested" : "cancel_requested",
    cancellation
  });
}
