import path from "node:path";
import { NextResponse } from "next/server";
import { runManagedCommand } from "@/lib/server/job-registry";
import { resolveWorkspacePath, stamp, workspaceRoot, writeTextArtifact } from "@/lib/server/paths";
import {
  HpcExecutionContractError,
  parseHpcExecutionContract,
  taskRequiresHpcExecutionContract,
} from "@/lib/server/hpc-execution-contract";

export const dynamic = "force-dynamic";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || (process.platform === "win32" ? "C:\\codex-python\\python.exe" : "python3");
}

function normalizeObjective(value: unknown) {
  return typeof value === "string" ? value.trim().slice(0, 6000) : "";
}

export async function POST(request: Request) {
  let body: Record<string, unknown> = {};
  try {
    body = await request.json() as Record<string, unknown>;
  } catch {
    return NextResponse.json({ ok: false, error: "Request body must be JSON." }, { status: 400 });
  }
  const objective = normalizeObjective(body.objective ?? body.prompt ?? body.goal);
  if (!objective) return NextResponse.json({ ok: false, error: "objective is required" }, { status: 400 });
  const taskId = typeof body.task_id === "string" ? body.task_id : "multi-agent";
  const contractInput = body.hpc_execution_contract ?? body.compute_policy ?? body;
  let hpcContract;
  try {
    hpcContract = parseHpcExecutionContract(contractInput, {
      required: taskRequiresHpcExecutionContract(taskId, body)
        || /\b(?:siim|isic)\b/i.test(objective),
    });
  } catch (error) {
    if (error instanceof HpcExecutionContractError) {
      return NextResponse.json({
        ok: false,
        code: error.code,
        error: error.message,
        missing_fields: error.missingFields,
      }, { status: error.statusCode });
    }
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Invalid HPC execution contract." },
      { status: 422 }
    );
  }
  const normalizedStamp = stamp().replace(/[^0-9TZ-]/g, "").replaceAll("-", "");
  const runId = `evomind_run_${normalizedStamp}_${crypto.randomUUID().replaceAll("-", "").slice(0, 6)}`;
  const requestPath = `workspace/evomind_requests/${runId}.txt`;
  await writeTextArtifact(requestPath, objective);
  const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  const args = ["-m", "xsci.multi_agent_cli", "run", "--request-file", resolveWorkspacePath(requestPath), "--run-id", runId];
  if (hpcContract) {
    args.push(
      "--hpc-job-id",
      String(hpcContract.job_id),
      "--hpc-credential-profile",
      hpcContract.credential_profile,
      "--hpc-resource-profile",
      hpcContract.resource_profile,
      "--execution-backend",
      hpcContract.execution_backend,
    );
  }
  void runManagedCommand({
    command: pythonExecutable(),
    args,
    cwd: workspaceRoot,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      ...(hpcContract ? {
        EVOMIND_SIIM_HPC_JOB_ID: String(hpcContract.job_id),
        EVOMIND_HPC_CREDENTIAL_PROFILE: hpcContract.credential_profile,
        EVOMIND_HPC_RESOURCE_PROFILE: hpcContract.resource_profile,
        EVOMIND_EXECUTION_BACKEND: hpcContract.execution_backend,
      } : {}),
    },
    timeout: 100 * 60 * 1000,
    taskId: "multi-agent",
    runId
  }).catch(() => {
    // Durable failure details are written by the Python run ledger.
  });
  return NextResponse.json({
    ok: true,
    task_id: "multi-agent",
    run_id: runId,
    status: "starting",
    snapshot_url: `/api/multi-agent/runs/${runId}`,
    events_url: `/api/multi-agent/runs/${runId}/events?after_seq=0`,
    current_run_url: "/api/workstation-summary",
    irreversible_actions: "human_gate",
    compute_policy: hpcContract,
  }, { status: 202 });
}
