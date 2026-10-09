import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

const SAFE_ID = /^[A-Za-z0-9_.-]{1,180}$/;
const workspaceRoot = path.resolve(process.env.WORKSTATION_ROOT ?? path.resolve(process.cwd(), "..", ".."));

function resolveWorkspacePath(relativePath: string) {
  const target = path.resolve(workspaceRoot, relativePath);
  const boundary = path.relative(workspaceRoot, target);
  if (boundary.startsWith("..") || path.isAbsolute(boundary)) throw new Error("HPC lineage path escapes workspace.");
  return target;
}

async function readJson(filePath: string): Promise<Record<string, unknown> | null> {
  try { return JSON.parse(await fs.readFile(filePath, "utf-8")) as Record<string, unknown>; }
  catch { return null; }
}

async function atomicJson(filePath: string, payload: unknown) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  await fs.rename(temporary, filePath);
}

function safe(value: string, label: string) {
  if (!SAFE_ID.test(value)) throw new Error(`Invalid ${label} for HPC Evidence Identity.`);
  return value;
}

export type HpcJobIdentity = {
  schema: "evomind.hpc_job_identity.v1";
  identity_kind: "execution_job";
  run_id: string;
  task_id: string;
  job_id: string;
  cluster: string;
  created_time: string;
  owner: string;
  status: string;
  credential_profile: string;
  resource_profile: string;
  execution_backend: string;
  dispatch_contract_path: string | null;
  remote_receipt_path: string | null;
  updated_at: string;
};

export async function bindHpcJobIdentity(input: {
  runId: string;
  taskId: string;
  jobId: string | number;
  cluster: string;
  owner: string;
  status: string;
  credentialProfile: string;
  resourceProfile: string;
  executionBackend: string;
  createdTime?: string;
  dispatchContractPath?: string | null;
  remoteReceiptPath?: string | null;
}): Promise<HpcJobIdentity> {
  const runId = safe(input.runId, "run_id");
  const taskId = safe(input.taskId, "task_id");
  const jobId = safe(String(input.jobId), "job_id");
  const cluster = safe(input.cluster, "cluster");
  const root = resolveWorkspacePath("workspace/hpc_job_lineage");
  const jobPath = path.join(root, "jobs", `${cluster}__${jobId}.json`);
  const runPath = path.join(root, "runs", `${runId}.json`);
  const [existingJob, existingRun] = await Promise.all([readJson(jobPath), readJson(runPath)]);
  if (existingJob && existingJob.run_id !== runId) {
    throw new Error(`HPC job ${cluster}/${jobId} is already bound to run ${String(existingJob.run_id)}.`);
  }
  if (existingRun && (existingRun.job_id !== jobId || existingRun.cluster !== cluster)) {
    throw new Error(`Run ${runId} is already bound to HPC job ${String(existingRun.cluster)}/${String(existingRun.job_id)}.`);
  }
  const updatedAt = new Date().toISOString();
  const identity: HpcJobIdentity = {
    schema: "evomind.hpc_job_identity.v1",
    identity_kind: "execution_job",
    run_id: runId,
    task_id: taskId,
    job_id: jobId,
    cluster,
    created_time: input.createdTime ?? (typeof existingJob?.created_time === "string" ? existingJob.created_time : updatedAt),
    owner: input.owner,
    status: input.status,
    credential_profile: input.credentialProfile,
    resource_profile: input.resourceProfile,
    execution_backend: input.executionBackend,
    dispatch_contract_path: input.dispatchContractPath ?? (typeof existingJob?.dispatch_contract_path === "string" ? existingJob.dispatch_contract_path : null),
    remote_receipt_path: input.remoteReceiptPath ?? (typeof existingJob?.remote_receipt_path === "string" ? existingJob.remote_receipt_path : null),
    updated_at: updatedAt,
  };
  await fs.mkdir(path.join(root, "jobs"), { recursive: true });
  await fs.mkdir(path.join(root, "runs"), { recursive: true });
  await atomicJson(jobPath, identity);
  await atomicJson(runPath, identity);
  await fs.appendFile(path.join(root, "events.jsonl"), `${JSON.stringify({
    schema: "evomind.hpc_job_lineage_event.v1",
    event_id: randomUUID(),
    run_id: runId,
    task_id: taskId,
    job_id: jobId,
    cluster,
    status: input.status,
    created_at: updatedAt,
  })}\n`, "utf-8");
  return identity;
}

export async function recordHpcProfileIdentity(input: {
  profile: string;
  jobId: string | number;
  cluster: string;
  owner: string;
  status: "PROFILE_ONLY_NOT_EXECUTION_JOB" | "RETIRED";
  evidencePath: string;
  createdTime?: string;
}) {
  const profile = safe(input.profile, "profile");
  const jobId = safe(String(input.jobId), "job_id");
  const cluster = safe(input.cluster, "cluster");
  const profilePath = resolveWorkspacePath(`workspace/hpc_job_lineage/profiles/${profile}.json`);
  const previous = await readJson(profilePath);
  const updatedAt = new Date().toISOString();
  const identity = {
    schema: "evomind.hpc_profile_identity.v1",
    identity_kind: "allocation_profile",
    run_id: null,
    task_id: null,
    job_id: jobId,
    cluster,
    profile,
    created_time: input.createdTime ?? (typeof previous?.created_time === "string" ? previous.created_time : updatedAt),
    owner: input.owner,
    status: input.status,
    evidence_path: input.evidencePath,
    execution_eligible: false,
    updated_at: updatedAt,
  };
  await atomicJson(profilePath, identity);
  return identity;
}

export async function listHpcJobLineage(): Promise<Array<Record<string, unknown>>> {
  const jobsRoot = resolveWorkspacePath("workspace/hpc_job_lineage/jobs");
  const profilesRoot = resolveWorkspacePath("workspace/hpc_job_lineage/profiles");
  const [entries, profileEntries] = await Promise.all([
    fs.readdir(jobsRoot, { withFileTypes: true }).catch(() => []),
    fs.readdir(profilesRoot, { withFileTypes: true }).catch(() => []),
  ]);
  const identities = await Promise.all([...entries.map((entry) => ({ entry, root: jobsRoot })), ...profileEntries.map((entry) => ({ entry, root: profilesRoot }))]
    .filter(({ entry }) => entry.isFile() && entry.name.endsWith(".json"))
    .map(({ entry, root }) => readJson(path.join(root, entry.name))));
  const valid = identities.filter((entry): entry is Record<string, unknown> => Boolean(entry));
  const executionJobs = new Set(valid
    .filter((entry) => entry.identity_kind === "execution_job")
    .map((entry) => `${String(entry.cluster)}::${String(entry.job_id)}`));
  return valid
    .filter((entry) => entry.identity_kind !== "allocation_profile"
      || !executionJobs.has(`${String(entry.cluster)}::${String(entry.job_id)}`))
    .sort((a, b) => String(b.updated_at ?? "").localeCompare(String(a.updated_at ?? "")));
}
