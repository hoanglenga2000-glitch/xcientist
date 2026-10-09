import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import type { TenantBinding } from "./tenant-byoa";

const execFileAsync = promisify(execFile);
const SHA256 = /^[a-f0-9]{64}$/;
const REASON = /^[a-z][a-z0-9._-]{2,63}$/;
type TerminalState = "frozen" | "retired";

export type TenantProfileTransitionResult = {
  schema: "evomind.tenant_hpc_profile_transition.v1";
  status: "completed";
  tenant_id: string;
  owner_principal_id: string;
  credential_profile: string;
  job_id: number;
  allocation_binding_id: string;
  allocation_generation: number;
  profile_instance_id: string;
  profile_state: TerminalState;
  reason: string;
  lifecycle_revision: number;
  tombstone: string;
  tombstone_sha256: string;
  idempotent_replay: boolean;
  credential_decrypted: false;
  network_accessed: false;
};

function transitionScriptPath() {
  const manifest = process.env.EVOMIND_RUNTIME_BUILD_MANIFEST?.trim();
  if (!manifest || !path.isAbsolute(manifest)) throw new Error("tenant_profile_transition_runtime_manifest_missing");
  const overlayRoot = path.dirname(path.resolve(manifest));
  const supportRoot = path.join(overlayRoot, "support");
  const script = path.resolve(supportRoot, "transition_tenant_hpc_profile.py");
  if (path.dirname(script) !== path.resolve(supportRoot)) throw new Error("tenant_profile_transition_script_escape");
  return script;
}

function parseResult(stdout: string) {
  const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
  try {
    const value = JSON.parse(line) as unknown;
    if (value && typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>;
  } catch {
    // Converted to the stable fail-closed error below.
  }
  throw new Error("tenant_profile_transition_result_invalid");
}

export function assertExactTenantProfileTransition(
  binding: TenantBinding,
  targetState: TerminalState,
  reason: string,
  value: Record<string, unknown>,
): TenantProfileTransitionResult {
  if (
    value.schema !== "evomind.tenant_hpc_profile_transition.v1"
    || value.status !== "completed"
    || value.tenant_id !== binding.tenant_id
    || value.owner_principal_id !== binding.username
    || value.credential_profile !== binding.credential_profile
    || value.job_id !== binding.job_id
    || value.allocation_binding_id !== binding.allocation_binding_id
    || value.allocation_generation !== binding.allocation_generation
    || value.profile_instance_id !== binding.profile_instance_id
    || value.profile_state !== targetState
    || value.reason !== reason
    || !Number.isSafeInteger(value.lifecycle_revision) || Number(value.lifecycle_revision) < 1
    || typeof value.tombstone !== "string" || value.tombstone !== `hpc_profile_${targetState}.tombstone.json`
    || typeof value.tombstone_sha256 !== "string" || !SHA256.test(value.tombstone_sha256)
    || typeof value.idempotent_replay !== "boolean"
    || value.credential_decrypted !== false
    || value.network_accessed !== false
  ) throw new Error("tenant_profile_transition_identity_mismatch");
  return value as TenantProfileTransitionResult;
}

export async function transitionTenantHpcProfile(
  binding: TenantBinding,
  targetState: TerminalState,
  reason: string,
) {
  if (!REASON.test(reason)) throw new Error("invalid_lifecycle_reason");
  const python = process.env.WORKSTATION_PYTHON?.trim() || process.env.EVOMIND_PYTHON?.trim();
  if (!python) throw new Error("tenant_profile_transition_python_missing");
  let stdout = "";
  try {
    const completed = await execFileAsync(python, [
      "-X", "utf8", transitionScriptPath(),
      "--tenant-id", binding.tenant_id,
      "--owner-principal-id", binding.username,
      "--profile", binding.credential_profile,
      "--job-id", String(binding.job_id),
      "--allocation-binding-id", binding.allocation_binding_id,
      "--allocation-generation", String(binding.allocation_generation),
      "--profile-instance-id", binding.profile_instance_id,
      "--target-state", targetState,
      "--reason", reason,
    ], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: 256 * 1024,
      env: process.env,
    });
    stdout = String(completed.stdout ?? "");
  } catch (error) {
    const candidate = error as NodeJS.ErrnoException & { stdout?: string | Buffer };
    const payload = String(candidate.stdout ?? "").trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
    try {
      const parsed = JSON.parse(payload) as Record<string, unknown>;
      if (typeof parsed.code === "string" && /^[a-z0-9_:-]{3,160}$/.test(parsed.code)) throw new Error(parsed.code);
    } catch (parseError) {
      if (parseError instanceof Error && parseError.message !== "Unexpected end of JSON input") throw parseError;
    }
    throw new Error("tenant_profile_transition_failed_closed");
  }
  return assertExactTenantProfileTransition(binding, targetState, reason, parseResult(stdout));
}

