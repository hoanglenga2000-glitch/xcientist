import path from "node:path";

import { prisma } from "@/lib/db";
import { decodeJson } from "@/lib/server/json";
import { resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";
import { readStableRegularTextFile } from "@/lib/server/stable-file";

export type PassedCodeQualityGate = {
  actionId: string;
  actionCreatedAt: Date;
  relativePath: string;
  qualityArtifactSha256: string;
  patchPath: string;
  patchSha256: string;
  affectedFiles: string[];
  patchPythonSyntaxCheck: unknown;
  payload: Record<string, unknown>;
};

type GateBinding = {
  actionId?: string;
  qualityArtifactPath?: string;
  qualityArtifactSha256?: string;
  patchPath?: string;
  patchSha256?: string;
};

function normalizedArtifactPath(value: unknown) {
  return typeof value === "string" ? value.replaceAll("\\", "/") : "";
}

function isBoundArtifactPath(value: string, prefix: string, filePattern: RegExp) {
  return value.startsWith(prefix) && filePattern.test(path.posix.basename(value));
}

async function validateReviewAction(
  taskId: string,
  record: {
    id: string;
    metadataJson: string | null;
    createdAt: Date;
  },
  expected: GateBinding = {}
): Promise<PassedCodeQualityGate | null> {
  const metadata = decodeJson<Record<string, unknown>>(record.metadataJson);
  if (!metadata || metadata.overall_status !== "passed") return null;

  const patchPrefix = `workspace/tasks/${taskId}/code/patches/`;
  const relativePath = normalizedArtifactPath(metadata.quality_artifact);
  const qualityArtifactSha256 = normalizedArtifactPath(metadata.quality_artifact_sha256).toLowerCase();
  const metadataPatchSha256 = normalizedArtifactPath(metadata.patch_sha256).toLowerCase();
  if (
    !isBoundArtifactPath(relativePath, patchPrefix, /^code_quality_check_[A-Za-z0-9._-]+\.json$/)
    || !/^[0-9a-f]{64}$/.test(qualityArtifactSha256)
    || !/^[0-9a-f]{64}$/.test(metadataPatchSha256)
  ) return null;
  if (
    (expected.actionId && record.id !== expected.actionId)
    || (expected.qualityArtifactPath && relativePath !== expected.qualityArtifactPath)
    || (expected.qualityArtifactSha256 && qualityArtifactSha256 !== expected.qualityArtifactSha256)
    || (expected.patchSha256 && metadataPatchSha256 !== expected.patchSha256)
  ) return null;

  const patchDir = resolveWorkspacePath(patchPrefix.slice(0, -1));
  const absoluteQualityPath = resolveWorkspacePath(relativePath);
  if (toRelativePath(absoluteQualityPath)?.replaceAll("\\", "/") !== relativePath) return null;
  const stableQuality = await readStableRegularTextFile(absoluteQualityPath, {
    allowedRoot: patchDir,
    maxBytes: 1_000_000
  }).catch(() => null);
  if (!stableQuality || stableQuality.sha256 !== qualityArtifactSha256) return null;

  let payload: Record<string, unknown>;
  try {
    const parsed = JSON.parse(stableQuality.text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    payload = parsed as Record<string, unknown>;
  } catch {
    return null;
  }
  const patchPath = normalizedArtifactPath(payload.patch_path);
  const patchSha256 = normalizedArtifactPath(payload.patch_sha256).toLowerCase();
  if (
    payload.task_id !== taskId
    || payload.overall_status !== "passed"
    || payload.original_data_check !== "passed"
    || payload.command_risk_check !== "passed"
    || payload.credential_check !== "passed"
    || payload.human_gate_required !== true
    || !isBoundArtifactPath(patchPath, patchPrefix, /^[A-Za-z0-9._-]+\.diff$/)
    || patchSha256 !== metadataPatchSha256
    || (expected.patchPath && patchPath !== expected.patchPath)
  ) return null;

  const absolutePatchPath = resolveWorkspacePath(patchPath);
  if (toRelativePath(absolutePatchPath)?.replaceAll("\\", "/") !== patchPath) return null;
  const stablePatch = await readStableRegularTextFile(absolutePatchPath, {
    allowedRoot: patchDir,
    maxBytes: 2_000_000
  }).catch(() => null);
  if (!stablePatch || stablePatch.sha256 !== patchSha256) return null;

  return {
    actionId: record.id,
    actionCreatedAt: record.createdAt,
    relativePath,
    qualityArtifactSha256,
    patchPath,
    patchSha256,
    affectedFiles: Array.isArray(payload.affected_files)
      ? payload.affected_files.filter((item): item is string => typeof item === "string")
      : [],
    patchPythonSyntaxCheck: payload.patch_python_syntax_check ?? null,
    payload
  };
}

export async function latestPassedCodeQualityGate(taskId: string) {
  const actionLogs = await prisma.actionLog.findMany({
    where: { taskId, action: "review_agent_patch" },
    orderBy: [{ createdAt: "desc" }, { id: "desc" }],
    take: 2
  });
  const latest = actionLogs[0];
  if (!latest) return null;

  // ActionLog has millisecond precision and no monotonic sequence. Never guess
  // which review is authoritative when two records share the newest timestamp.
  if (actionLogs[1]?.createdAt.getTime() === latest.createdAt.getTime()) return null;
  return validateReviewAction(taskId, latest);
}

export async function passedCodeQualityGateByBinding(taskId: string, binding: GateBinding) {
  if (!binding.actionId) return null;
  const record = await prisma.actionLog.findUnique({ where: { id: binding.actionId } });
  if (!record || record.taskId !== taskId || record.action !== "review_agent_patch") return null;
  return validateReviewAction(taskId, record, binding);
}
