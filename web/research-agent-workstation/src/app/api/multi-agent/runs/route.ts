import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { runManagedCommand } from "@/lib/server/job-registry";
import { resolveWorkspacePath, stamp, workspaceRoot, writeTextArtifact } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || (process.platform === "win32" ? "C:\\codex-python\\python.exe" : "python3");
}

function normalizeObjective(value: unknown) {
  return typeof value === "string" ? value.trim().slice(0, 6000) : "";
}

type HpcAllocation = { jobId: number; credentialProfile: string };

async function resolveHpcAllocation(body: Record<string, unknown>): Promise<HpcAllocation | null> {
  const policy = body.compute_policy && typeof body.compute_policy === "object"
    ? body.compute_policy as Record<string, unknown>
    : {};
  let memoryBinding: Record<string, unknown> = {};
  try {
    const memory = JSON.parse(
      await readFile(resolveWorkspacePath("configs/hpc_connection_memory_core.json"), "utf8")
    ) as Record<string, unknown>;
    if (memory.current_binding && typeof memory.current_binding === "object") {
      memoryBinding = memory.current_binding as Record<string, unknown>;
    }
  } catch {
    // Environment or an explicit request policy may still provide the durable binding.
  }
  const rawJob = policy.job_id
    ?? body.hpc_job_id
    ?? process.env.EVOMIND_SIIM_HPC_JOB_ID
    ?? memoryBinding.job_id;
  const rawProfile = policy.credential_profile
    ?? body.hpc_credential_profile
    ?? process.env.EVOMIND_HPC_CREDENTIAL_PROFILE
    ?? memoryBinding.credential_profile;
  if (rawJob == null && rawProfile == null) return null;
  const jobId = Number(rawJob);
  const credentialProfile = typeof rawProfile === "string" ? rawProfile.trim() : "";
  if (!Number.isSafeInteger(jobId) || jobId <= 0 || credentialProfile !== `job${jobId}`) {
    throw new Error("HPC job_id and credential_profile must form one canonical job binding.");
  }
  return { jobId, credentialProfile };
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
  let allocation: HpcAllocation | null;
  try {
    allocation = await resolveHpcAllocation(body);
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Invalid HPC allocation." },
      { status: 400 }
    );
  }
  const normalizedStamp = stamp().replace(/[^0-9TZ-]/g, "").replaceAll("-", "");
  const runId = `evomind_run_${normalizedStamp}_${crypto.randomUUID().replaceAll("-", "").slice(0, 6)}`;
  const requestPath = `workspace/evomind_requests/${runId}.txt`;
  await writeTextArtifact(requestPath, objective);
  const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  const args = ["-m", "xsci.multi_agent_cli", "run", "--request-file", resolveWorkspacePath(requestPath), "--run-id", runId];
  if (allocation) {
    args.push(
      "--hpc-job-id",
      String(allocation.jobId),
      "--hpc-credential-profile",
      allocation.credentialProfile
    );
  }
  void runManagedCommand({
    command: pythonExecutable(),
    args,
    cwd: workspaceRoot,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      ...(allocation ? {
        EVOMIND_SIIM_HPC_JOB_ID: String(allocation.jobId),
        EVOMIND_HPC_CREDENTIAL_PROFILE: allocation.credentialProfile,
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
    compute_policy: allocation ? {
      backend: "hpc",
      job_id: allocation.jobId,
      credential_profile: allocation.credentialProfile,
    } : null,
  }, { status: 202 });
}
