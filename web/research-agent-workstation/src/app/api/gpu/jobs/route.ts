import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";
import { submitGpuJob } from "@/lib/server/gpu-ssh-gateway";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

function summarizeGpuJob(payload: unknown, relativePath: string) {
  const record = payload && typeof payload === "object" && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {};
  return {
    job_id: typeof record.job_id === "string" ? record.job_id : path.basename(relativePath, ".json"),
    task_id: typeof record.task_id === "string" ? record.task_id : null,
    status: typeof record.status === "string" ? record.status : "unknown",
    provider: typeof record.provider === "string" ? record.provider : "ssh_gateway",
    template: typeof record.template === "string" ? record.template : typeof record.command_template === "string" ? record.command_template : null,
    workstation_run_id: typeof record.workstation_run_id === "string" ? record.workstation_run_id : null,
    agent_id: typeof record.agent_id === "string" ? record.agent_id : null,
    gate_id: typeof record.gate_id === "string" ? record.gate_id : null,
    artifact_path: relativePath,
    job_manifest_path: typeof record.job_manifest_path === "string" ? record.job_manifest_path : null,
    stdout_artifact: typeof record.stdout_artifact === "string" ? record.stdout_artifact : null,
    stderr_artifact: typeof record.stderr_artifact === "string" ? record.stderr_artifact : null,
    metrics_artifact: typeof record.metrics_artifact === "string" ? record.metrics_artifact : null,
    submission_artifact: typeof record.submission_artifact === "string" ? record.submission_artifact : null,
    created_at: typeof record.created_at === "string" ? record.created_at : null,
    updated_at: typeof record.updated_at === "string" ? record.updated_at : null,
    error: typeof record.error === "string" ? record.error : null
  };
}

export async function GET() {
  const jobsRoot = resolveWorkspacePath(path.join("workspace", "gpu", "jobs"));
  const entries = await fs.readdir(jobsRoot, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json") && !entry.name.endsWith("_manifest.json") && !entry.name.endsWith("_cancel.json") && !entry.name.endsWith("_artifact_manifest.json"))
    .map(async (entry) => {
      const relativePath = path.join("workspace", "gpu", "jobs", entry.name).replaceAll("\\", "/");
      const absolutePath = path.join(jobsRoot, entry.name);
      const [stat, payload] = await Promise.all([
        fs.stat(absolutePath).catch(() => null),
        readJsonFile(absolutePath)
      ]);
      return { ...summarizeGpuJob(payload, relativePath), mtime_ms: stat?.mtimeMs ?? 0 };
    }));
  const jobs = candidates.sort((a, b) => b.mtime_ms - a.mtime_ms).slice(0, 50).map(({ mtime_ms: _mtimeMs, ...job }) => job);
  return NextResponse.json({ ok: true, jobs });
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const taskId = typeof body.task_id === "string" ? body.task_id : "house_prices";
  const template = typeof body.template === "string" ? body.template : undefined;
  const runId = typeof body.run_id === "string" ? body.run_id : undefined;
  const agentId = typeof body.agent_id === "string" ? body.agent_id : undefined;
  const gateId = typeof body.gate_id === "string" ? body.gate_id : undefined;
  const resourceRequest = body.resource_request && typeof body.resource_request === "object" ? body.resource_request : undefined;
  return NextResponse.json(await submitGpuJob({ taskId, template, runId, agentId, gateId, resourceRequest }));
}
