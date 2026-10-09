import { promises as fs } from "node:fs";
import path from "node:path";

type JsonObject = Record<string, unknown>;

const MAX_JSON_BYTES = 1024 * 1024;
const MAX_TENANT_DIRECTORIES = 128;
const TENANT_ID_PATTERN = /^tenant_[a-f0-9]{24}$/;
const KAGGLE_EVIDENCE_MAX_AGE_MS = 24 * 60 * 60 * 1000;

function isObject(value: unknown): value is JsonObject {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function safeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function parseTimestamp(value: unknown): number | null {
  if (typeof value !== "string" || !value.trim()) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function isContained(root: string, candidate: string) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function trustedDirectory(rawPath: string | null | undefined): Promise<string | null> {
  if (!rawPath || !path.isAbsolute(rawPath)) return null;
  const candidate = path.resolve(rawPath);
  try {
    const stat = await fs.lstat(candidate);
    if (!stat.isDirectory() || stat.isSymbolicLink()) return null;
    return await fs.realpath(candidate);
  } catch {
    return null;
  }
}

async function trustedChild(root: string, ...segments: string[]): Promise<string | null> {
  const candidate = path.resolve(root, ...segments);
  if (!isContained(root, candidate)) return null;
  try {
    const real = await fs.realpath(candidate);
    return isContained(root, real) ? real : null;
  } catch {
    return null;
  }
}

async function readTrustedJson(root: string, ...segments: string[]): Promise<JsonObject | null> {
  const candidate = await trustedChild(root, ...segments);
  if (!candidate) return null;
  try {
    const stat = await fs.lstat(candidate);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0 || stat.size > MAX_JSON_BYTES) return null;
    const parsed = JSON.parse((await fs.readFile(candidate, "utf-8")).replace(/^\uFEFF/, ""));
    return isObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function regularFilePresent(root: string, ...segments: string[]) {
  const candidate = await trustedChild(root, ...segments);
  if (!candidate) return false;
  try {
    const stat = await fs.lstat(candidate);
    return stat.isFile() && !stat.isSymbolicLink() && stat.size > 0 && stat.size <= MAX_JSON_BYTES;
  } catch {
    return false;
  }
}

type ManagedHpcBinding = {
  jobId: number;
  generation: number;
  allocationBindingId: string;
  healthVerified: boolean;
  healthFresh: boolean;
  checkedAt: string | null;
  expiresAt: string | null;
};

function validAllocationBindingId(value: unknown) {
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,192}$/.test(value);
}

function validCredentialProfile(tenantId: string, jobId: number, generation: number, value: unknown) {
  return value === `${tenantId}_job${jobId}_g${generation}`;
}

export async function projectManagedTenantHpcConnector(
  rawRoot: string | null | undefined = process.env.WORKSTATION_BYOA_STATE_ROOT,
  nowMs: number = Date.now(),
): Promise<JsonObject | null> {
  const root = await trustedDirectory(rawRoot);
  if (!root) return null;
  const tenantsRoot = await trustedChild(root, "tenants");
  if (!tenantsRoot) return null;

  const tenantEntries = await fs.readdir(tenantsRoot, { withFileTypes: true }).catch(() => []);
  const candidates = tenantEntries
    .filter((entry) => entry.isDirectory() && !entry.isSymbolicLink() && TENANT_ID_PATTERN.test(entry.name))
    .slice(0, MAX_TENANT_DIRECTORIES);
  const active: ManagedHpcBinding[] = [];
  let recognizedBindingCount = 0;

  for (const entry of candidates) {
    const tenantRoot = await trustedChild(tenantsRoot, entry.name);
    if (!tenantRoot) continue;
    const binding = await readTrustedJson(tenantRoot, "hpc-binding.json");
    if (binding?.schema !== "evomind.tenant_hpc_binding.v1") continue;
    const jobId = safeInteger(binding.job_id);
    const generation = safeInteger(binding.allocation_generation);
    if (!jobId || !generation) continue;
    if (!validAllocationBindingId(binding.allocation_binding_id)) continue;
    if (!validCredentialProfile(entry.name, jobId, generation, binding.credential_profile)) continue;
    recognizedBindingCount += 1;
    if (binding.state !== "active") continue;

    const health = await readTrustedJson(tenantRoot, "runtime", "gpu-health.json");
    const checkedMs = parseTimestamp(health?.checked_at_utc);
    const expiresMs = parseTimestamp(health?.expires_at_utc);
    const healthMatches = Boolean(
      health?.schema === "evomind.tenant_gpu_runtime_health.v1"
      && health?.status === "ready"
      && health?.code === "five_of_five_identity_verified"
      && health?.job_id === jobId
      && health?.allocation_generation === generation
      && health?.allocation_binding_id === binding.allocation_binding_id
      && health?.designated_proxy_path_verified === true
      && health?.job_container_verified === true
      && health?.identity_samples_passed === 5
      && health?.probe_remote_write_performed === false
      && checkedMs !== null
      && expiresMs !== null,
    );
    const healthFresh = Boolean(
      healthMatches
      && checkedMs! <= nowMs + 2 * 60 * 1000
      && expiresMs! >= nowMs,
    );
    active.push({
      jobId,
      generation,
      allocationBindingId: String(binding.allocation_binding_id),
      healthVerified: healthMatches,
      healthFresh,
      checkedAt: checkedMs === null ? null : new Date(checkedMs).toISOString(),
      expiresAt: expiresMs === null ? null : new Date(expiresMs).toISOString(),
    });
  }

  if (active.length === 0) {
    if (recognizedBindingCount === 0) return null;
    return {
      name: "Tenant HPC GPU",
      state: "HPC Offline: no active tenant allocation binding",
      configured: true,
      current_allocation_blocked: true,
      current_gate_ready: false,
      active_binding_count: 0,
      recognized_binding_count: recognizedBindingCount,
      source: "managed_tenant_binding_projection",
      notes: "Managed tenant bindings exist, but none is active.",
    };
  }

  active.sort((left, right) =>
    Number(right.healthVerified) - Number(left.healthVerified)
    || right.generation - left.generation
    || right.jobId - left.jobId,
  );
  const selected = active[0];
  return {
    name: "Tenant HPC GPU",
    state: selected.healthVerified
      ? selected.healthFresh ? "Tenant HPC Ready" : "Tenant HPC Active Binding Verified"
      : "Tenant HPC Active Binding Configured",
    configured: true,
    current_allocation_blocked: false,
    current_gate_ready: selected.healthFresh,
    active_binding_count: active.length,
    recognized_binding_count: recognizedBindingCount,
    job_id: selected.jobId,
    allocation_generation: selected.generation,
    allocation_binding_id: selected.allocationBindingId,
    designated_proxy_path_verified: selected.healthVerified,
    job_container_verified: selected.healthVerified,
    identity_samples_passed: selected.healthVerified ? 5 : 0,
    live_probe_fresh: selected.healthFresh,
    last_verified_at: selected.checkedAt,
    proof_expires_at: selected.expiresAt,
    source: "managed_tenant_binding_projection",
    notes: selected.healthFresh
      ? "The active tenant binding has a current 5-of-5 designated-proxy and job-container identity proof."
      : selected.healthVerified
        ? "The active tenant binding has persisted 5-of-5 designated-proxy and job-container identity proof; every GPU action still reruns the live identity gate before execution."
        : "The tenant allocation binding is active; a live identity proof is required before GPU execution.",
  };
}

function verifiedKaggleReport(report: JsonObject | null) {
  if (!report) return false;
  const realApiSmoke = isObject(report.real_api_smoke) ? report.real_api_smoke : null;
  const method = String(report.verification_method ?? "");
  return Boolean(
    (report.schema === "evomind.kaggle_dpapi_readiness.v2" || report.schema === undefined)
    && report.status === "passed"
    && report.credential_status === "authenticated_real_api"
    && report.credential_installed === true
    && report.authenticated === true
    && report.human_gate_required_for_submission === true
    && ["dpapi_real_api", "dpapi_status_and_real_api_smoke"].includes(method)
    && (!realApiSmoke || (realApiSmoke.status === "passed" && realApiSmoke.real_external_called === true)),
  );
}

export async function projectManagedKaggleConnector(
  rawWorkspaceRoot: string,
  rawAppData: string | null | undefined = process.env.APPDATA,
  nowMs: number = Date.now(),
  environmentConfigured: boolean = Boolean(
    process.env.KAGGLE_API_TOKEN
    || process.env.KAGGLE_API_TOKEN_FILE
    || (process.env.KAGGLE_USERNAME && (process.env.KAGGLE_KEY || process.env.KAGGLE_KEY_FILE)),
  ),
): Promise<JsonObject | null> {
  const workspace = await trustedDirectory(rawWorkspaceRoot);
  if (!workspace) return null;
  const report = await readTrustedJson(workspace, "docs", "kaggle_dpapi_readiness.json");
  const appData = await trustedDirectory(rawAppData);
  const credentialPresent = appData
    ? await regularFilePresent(appData, "ResearchAgentWorkstation", "kaggle_api_token.xml")
    : false;
  const configured = credentialPresent || environmentConfigured;
  const authenticatedEvidence = configured && verifiedKaggleReport(report);
  const generatedMs = parseTimestamp(report?.generated_at);
  const evidenceAgeMs = generatedMs === null ? null : Math.max(0, nowMs - generatedMs);
  const evidenceFresh = authenticatedEvidence
    && evidenceAgeMs !== null
    && evidenceAgeMs <= KAGGLE_EVIDENCE_MAX_AGE_MS;
  const toolStatus = isObject(report?.tool_status) ? report.tool_status : null;
  const toolchainReady = Boolean(toolStatus?.python_package_installed === true || toolStatus?.python_package_version);

  if (!configured && !report) return null;
  return {
    name: "Kaggle",
    state: authenticatedEvidence
      ? "Kaggle DPAPI Real API Verified"
      : configured
        ? "Kaggle DPAPI Configured"
        : "Not Configured",
    configured,
    authenticated: authenticatedEvidence,
    credential_status: authenticatedEvidence
      ? "authenticated_real_api"
      : configured ? "configured_dpapi_unverified" : "not_configured",
    credential_storage: credentialPresent ? "windows_dpapi_current_user" : environmentConfigured ? "managed_environment" : "none",
    toolchain_ready: toolchainReady,
    verification_method: authenticatedEvidence ? String(report?.verification_method) : null,
    last_verified_at: generatedMs === null ? null : new Date(generatedMs).toISOString(),
    evidence_fresh: evidenceFresh,
    evidence_age_seconds: evidenceAgeMs === null ? null : Math.floor(evidenceAgeMs / 1000),
    human_gate_required_for_submission: true,
    source: "managed_kaggle_projection",
    notes: authenticatedEvidence
      ? "A managed DPAPI credential exists and a real read-only Kaggle API smoke passed; each external action revalidates authentication and official submission remains human-gated."
      : configured
        ? "A managed Kaggle credential exists; a real read-only API smoke is required before treating authentication as current."
        : "No managed Kaggle credential is installed.",
  };
}
