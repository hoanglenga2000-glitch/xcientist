import { latestPassedCodeQualityGate, passedCodeQualityGateByBinding } from "@/lib/server/code-quality-gate";
import { decodeJson } from "@/lib/server/json";
import { resolveWorkspacePath } from "@/lib/server/paths";
import { readStableRegularTextFile } from "@/lib/server/stable-file";

export type HpcGateRecord = {
  id: string;
  taskId: string;
  runId: string | null;
  gateType: string;
  decision: string;
  evidenceJson: string | null;
  reviewer?: string | null;
  decidedAt?: Date | null;
};

type CodeDependency = {
  quality_gate_path?: unknown;
  quality_gate_sha256?: unknown;
  quality_gate_action_id?: unknown;
  patch_path?: unknown;
  patch_sha256?: unknown;
};

type HpcApprovalEvidence = {
  gate_id?: unknown;
  task_id?: unknown;
  run_id?: unknown;
  reviewer?: unknown;
  reason?: unknown;
  artifact_path?: unknown;
  approved_at?: unknown;
  manifest_path?: unknown;
  manifest_sha256?: unknown;
  quality_gate_action_id?: unknown;
};

function stringField(value: unknown) {
  return typeof value === "string" ? value.replaceAll("\\", "/") : "";
}

function dependencyFields(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const dependency = value as CodeDependency;
  const fields = {
    qualityArtifactPath: stringField(dependency.quality_gate_path),
    qualityArtifactSha256: stringField(dependency.quality_gate_sha256).toLowerCase(),
    actionId: stringField(dependency.quality_gate_action_id),
    patchPath: stringField(dependency.patch_path),
    patchSha256: stringField(dependency.patch_sha256).toLowerCase()
  };
  return Object.values(fields).every(Boolean) ? fields : null;
}

function approvalFields(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const approval = value as HpcApprovalEvidence;
  const fields = {
    gateId: stringField(approval.gate_id),
    taskId: stringField(approval.task_id),
    runId: stringField(approval.run_id),
    reviewer: stringField(approval.reviewer),
    reason: stringField(approval.reason),
    artifactPath: stringField(approval.artifact_path),
    approvedAt: stringField(approval.approved_at),
    manifestPath: stringField(approval.manifest_path),
    manifestSha256: stringField(approval.manifest_sha256).toLowerCase(),
    qualityGateActionId: stringField(approval.quality_gate_action_id)
  };
  return fields.gateId
    && fields.taskId
    && fields.runId
    && fields.reviewer
    && fields.reason
    && fields.approvedAt
    && fields.manifestPath
    && fields.manifestSha256
    && fields.qualityGateActionId
    ? fields
    : null;
}

function isoDate(value: Date | null | undefined) {
  return value instanceof Date && Number.isFinite(value.getTime()) ? value.toISOString() : "";
}

export function buildHpcApprovalEvidence(
  gate: HpcGateRecord,
  input: { reviewer: string; reason: string; artifactPath?: string; decidedAt: Date }
) {
  if (gate.decision !== "pending") throw new Error("hpc_gate_not_pending");
  const evidence = decodeJson<Record<string, unknown>>(gate.evidenceJson);
  const dependency = dependencyFields(evidence?.code_agent_dependency);
  const reviewer = input.reviewer.trim();
  const reason = input.reason.trim();
  const approvedAt = isoDate(input.decidedAt);
  const manifestPath = stringField(evidence?.manifest_path);
  const manifestSha256 = stringField(evidence?.manifest_sha256).toLowerCase();
  if (
    !evidence
    || !gate.runId
    || !reviewer
    || !reason
    || !approvedAt
    || !manifestPath
    || !/^[0-9a-f]{64}$/.test(manifestSha256)
    || !dependency
  ) {
    throw new Error("hpc_gate_approval_binding_invalid");
  }
  return {
    ...evidence,
    approval: {
      gate_id: gate.id,
      task_id: gate.taskId,
      run_id: gate.runId,
      reviewer,
      reason,
      artifact_path: input.artifactPath ?? null,
      approved_at: approvedAt,
      manifest_path: manifestPath,
      manifest_sha256: manifestSha256,
      quality_gate_action_id: dependency.actionId
    }
  };
}

export async function validateHpcExecutionGate(
  gate: HpcGateRecord | null,
  input: { taskId: string; runId: string; template: string; requireApproved?: boolean }
) {
  const reasons: string[] = [];
  if (!gate) return { ok: false, reasons: ["hpc_gate_missing"] };
  if (gate.id !== `${input.runId}_hpc_execution_approval`) reasons.push("hpc_gate_id_mismatch");
  if (gate.taskId !== input.taskId) reasons.push("hpc_gate_task_mismatch");
  if (gate.runId !== input.runId) reasons.push("hpc_gate_run_mismatch");
  if (gate.gateType !== "hpc_execution_approval") reasons.push("hpc_gate_type_mismatch");
  const requireApproved = input.requireApproved ?? true;
  if (requireApproved) {
    if (gate.decision !== "approved") reasons.push("hpc_gate_not_approved");
  } else if (gate.decision !== "pending") {
    reasons.push("hpc_gate_not_pending");
  }

  const evidence = decodeJson<Record<string, unknown>>(gate.evidenceJson);
  const expectedManifestPath = `workspace/workstation_runs/${input.taskId}/${input.runId}/hpc_execution_gate_manifest.json`;
  const manifestPath = stringField(evidence?.manifest_path);
  const manifestSha256 = stringField(evidence?.manifest_sha256).toLowerCase();
  const requestedTemplate = stringField(evidence?.requested_template);
  const dependency = dependencyFields(evidence?.code_agent_dependency);
  if (manifestPath !== expectedManifestPath) reasons.push("hpc_gate_manifest_path_mismatch");
  if (!/^[0-9a-f]{64}$/.test(manifestSha256)) reasons.push("hpc_gate_manifest_hash_missing");
  if (requestedTemplate !== input.template) reasons.push("hpc_gate_template_mismatch");
  if (!dependency) reasons.push("hpc_gate_code_dependency_missing");
  if (!requireApproved && (gate.reviewer || gate.decidedAt || evidence?.approval)) {
    reasons.push("hpc_gate_pending_audit_dirty");
  }

  let manifest: Record<string, unknown> | null = null;
  if (manifestPath === expectedManifestPath && /^[0-9a-f]{64}$/.test(manifestSha256)) {
    const runRoot = resolveWorkspacePath(`workspace/workstation_runs/${input.taskId}/${input.runId}`);
    const stableManifest = await readStableRegularTextFile(resolveWorkspacePath(manifestPath), {
      allowedRoot: runRoot,
      maxBytes: 1_000_000
    }).catch(() => null);
    if (!stableManifest || stableManifest.sha256 !== manifestSha256) {
      reasons.push("hpc_gate_manifest_hash_mismatch");
    } else {
      try {
        const parsed = JSON.parse(stableManifest.text);
        manifest = parsed && typeof parsed === "object" && !Array.isArray(parsed)
          ? parsed as Record<string, unknown>
          : null;
      } catch {
        manifest = null;
      }
      if (!manifest) reasons.push("hpc_gate_manifest_invalid");
    }
  }

  if (manifest) {
    if (manifest.schema !== "academic_research_os.hpc_execution_gate.v1") reasons.push("hpc_gate_manifest_schema_mismatch");
    if (manifest.task_id !== input.taskId) reasons.push("hpc_gate_manifest_task_mismatch");
    if (manifest.workstation_run_id !== input.runId) reasons.push("hpc_gate_manifest_run_mismatch");
    if (manifest.requested_template !== input.template) reasons.push("hpc_gate_manifest_template_mismatch");
    if (manifest.status !== "pending_approval") reasons.push("hpc_gate_manifest_status_mismatch");
    if (manifest.resource_policy !== "whitelist_templates_only") reasons.push("hpc_gate_manifest_resource_policy_mismatch");
    if (manifest.remote_training_allowed !== false) reasons.push("hpc_gate_manifest_remote_training_policy_mismatch");
    if (manifest.approval_required_before !== "POST /api/gpu/jobs") reasons.push("hpc_gate_manifest_approval_boundary_mismatch");
    const rawManifestDependency = manifest.code_agent_dependency as Record<string, unknown> | null;
    if (
      !rawManifestDependency
      || typeof rawManifestDependency !== "object"
      || Array.isArray(rawManifestDependency)
      || rawManifestDependency.status !== "code_quality_passed"
      || rawManifestDependency.required_before_training !== true
    ) reasons.push("hpc_gate_manifest_dependency_policy_mismatch");
    const manifestDependency = dependencyFields(manifest.code_agent_dependency);
    if (!manifestDependency || !dependency || Object.keys(dependency).some(
      (key) => manifestDependency[key as keyof typeof manifestDependency] !== dependency[key as keyof typeof dependency]
    )) reasons.push("hpc_gate_manifest_dependency_mismatch");
  }

  let codeGate = null;
  if (dependency) {
    codeGate = await passedCodeQualityGateByBinding(input.taskId, dependency);
    if (!codeGate) {
      reasons.push("hpc_gate_code_dependency_invalid");
    } else {
      const latest = await latestPassedCodeQualityGate(input.taskId);
      if (!latest || latest.actionId !== codeGate.actionId) reasons.push("hpc_gate_code_dependency_stale");
    }
  }
  if (requireApproved) {
    const decidedAt = isoDate(gate.decidedAt);
    const reviewer = typeof gate.reviewer === "string" ? gate.reviewer.trim() : "";
    const approval = approvalFields(evidence?.approval);
    if (!reviewer || !decidedAt || !approval) {
      reasons.push("hpc_gate_approval_audit_missing");
    } else if (
      approval.gateId !== gate.id
      || approval.taskId !== gate.taskId
      || approval.runId !== gate.runId
      || approval.reviewer !== reviewer
      || approval.approvedAt !== decidedAt
      || approval.manifestPath !== manifestPath
      || approval.manifestSha256 !== manifestSha256
      || approval.qualityGateActionId !== dependency?.actionId
    ) {
      reasons.push("hpc_gate_approval_binding_mismatch");
    }
  }

  return {
    ok: reasons.length === 0,
    reasons,
    evidence,
    manifest,
    codeGate
  };
}
