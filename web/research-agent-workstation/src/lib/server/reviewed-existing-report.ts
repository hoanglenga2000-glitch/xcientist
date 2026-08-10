type JsonRecord = Record<string, unknown>;

function text(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

export function reviewedExistingReportEligible(input: {
  taskId: string;
  runId: string;
  run: JsonRecord;
  review: JsonRecord;
  artifactManifest: JsonRecord;
  pointer: JsonRecord;
}) {
  const { taskId, runId, run, review, artifactManifest, pointer } = input;
  if (text(run.run_id) !== runId || text(run.status).toLowerCase() !== "completed") return false;
  const explicitTask = text(run.task_id) || text(artifactManifest.task_id);
  if (explicitTask && explicitTask !== taskId) return false;
  const pointerBound = text(pointer.task_id) === taskId && text(pointer.run_id) === runId;
  if (!explicitTask && !pointerBound) return false;
  if (text(review.status).toLowerCase() !== "passed") return false;
  const claimAudit = review.claim_audit && typeof review.claim_audit === "object"
    ? review.claim_audit as JsonRecord
    : {};
  if (text(claimAudit.status).toLowerCase() !== "passed") return false;
  if (
    text(artifactManifest.schema) !== "evomind.artifact_manifest.v1"
    || text(artifactManifest.run_id) !== runId
    || !text(artifactManifest.selected_solution)
    || !Array.isArray(artifactManifest.artifacts)
    || artifactManifest.artifacts.length === 0
  ) return false;
  return true;
}
