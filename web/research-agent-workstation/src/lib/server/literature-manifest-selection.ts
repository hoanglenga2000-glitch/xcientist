type JsonRecord = Record<string, unknown>;

export type LiteratureManifestCandidate = {
  absolutePath: string;
  expectedTaskId: string;
  mtimeMs: number;
  payload: unknown;
};

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function externalVerifiedCount(payload: JsonRecord) {
  const integrity = isRecord(payload.integrity) ? payload.integrity : {};
  const value = Number(integrity.external_verified ?? 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

export function selectLatestValidLiteratureManifest(
  candidates: LiteratureManifestCandidate[]
): LiteratureManifestCandidate | null {
  const valid = candidates.filter((candidate) => {
    if (!isRecord(candidate.payload)) return false;
    if (String(candidate.payload.task_id ?? "") !== candidate.expectedTaskId) return false;
    return Array.isArray(candidate.payload.papers) && candidate.payload.papers.length > 0;
  });

  return valid.sort((left, right) => {
    const leftExternallyVerified = externalVerifiedCount(left.payload as JsonRecord) > 0 ? 1 : 0;
    const rightExternallyVerified = externalVerifiedCount(right.payload as JsonRecord) > 0 ? 1 : 0;
    return rightExternallyVerified - leftExternallyVerified
      || right.mtimeMs - left.mtimeMs
      || right.absolutePath.localeCompare(left.absolutePath);
  })[0] ?? null;
}
