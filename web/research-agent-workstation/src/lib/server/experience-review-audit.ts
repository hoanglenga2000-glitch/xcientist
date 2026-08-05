import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import type {
  EvolutionCandidateFreezeSummary,
  EvolutionClaimAuditSummary,
  EvolutionIndependentReviewSummary,
  EvolutionReviewChainIntegrity
} from "../api/types";

const REVIEW_ARTIFACT_LIMIT = 8 * 1024 * 1024;
const SHA256_RE = /^[a-f0-9]{64}$/i;

type ExactArtifactName = "candidate-freeze.json" | "independent-review.json" | "claim-audit.json";

type ExactJsonArtifact = {
  name: ExactArtifactName;
  present: boolean;
  payload: Record<string, unknown> | null;
  sha256: string | null;
  error: string | null;
};

export type EvolutionReviewAuditProjection = {
  candidate_freeze: EvolutionCandidateFreezeSummary;
  independent_review: EvolutionIndependentReviewSummary;
  claim_audit: EvolutionClaimAuditSummary;
  review_chain: EvolutionReviewChainIntegrity;
  artifact_names: ExactArtifactName[];
};

function recordOf(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown, maxLength = 500): string | null {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, maxLength)
    : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function positiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function normalizeSha256(value: unknown): string | null {
  const candidate = typeof value === "string" ? value.trim().toLowerCase() : "";
  return SHA256_RE.test(candidate) ? candidate : null;
}

function artifactWithinRoot(runRoot: string, artifactPath: string): boolean {
  const relative = path.relative(runRoot, artifactPath);
  return Boolean(relative) && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function readExactJsonArtifact(runRootInput: string, name: ExactArtifactName): Promise<ExactJsonArtifact> {
  const runRoot = await fs.realpath(runRootInput).catch(() => path.resolve(runRootInput));
  const configured = path.resolve(runRoot, name);
  const entry = await fs.lstat(configured).catch(() => null);
  if (!entry) return { name, present: false, payload: null, sha256: null, error: null };

  const real = await fs.realpath(configured).catch(() => null);
  if (!real || !artifactWithinRoot(runRoot, real)) {
    return { name, present: true, payload: null, sha256: null, error: `${name} escapes the exact run root.` };
  }
  const stat = await fs.stat(real).catch(() => null);
  if (!stat?.isFile()) {
    return { name, present: true, payload: null, sha256: null, error: `${name} is not a regular file.` };
  }
  if (stat.size > REVIEW_ARTIFACT_LIMIT) {
    return { name, present: true, payload: null, sha256: null, error: `${name} exceeds the bounded projection limit.` };
  }
  const bytes = await fs.readFile(real).catch(() => null);
  if (!bytes) {
    return { name, present: true, payload: null, sha256: null, error: `${name} could not be read.` };
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  try {
    const payload = recordOf(JSON.parse(bytes.toString("utf-8").replace(/^\uFEFF/, "")));
    if (!payload) {
      return { name, present: true, payload: null, sha256, error: `${name} must contain a JSON object.` };
    }
    return { name, present: true, payload, sha256, error: null };
  } catch {
    return { name, present: true, payload: null, sha256, error: `${name} contains invalid JSON.` };
  }
}

function stageStatus(present: boolean, errors: string[]): "verified" | "failed" | "not_present" {
  if (!present) return "not_present";
  return errors.length ? "failed" : "verified";
}

function artifactStatus(artifact: ExactJsonArtifact): string | null {
  return stringValue(artifact.payload?.status, 80);
}

/**
 * Load only the three governance artifacts from one already-selected run root.
 * Hashes always come from the exact file bytes; payload-declared hashes are used
 * only as links and never as substitutes for an artifact's own digest.
 */
export async function loadExperienceReviewAudit(
  runRootInput: string,
  taskId: string
): Promise<EvolutionReviewAuditProjection> {
  const runRoot = await fs.realpath(runRootInput).catch(() => path.resolve(runRootInput));
  const runRootName = path.basename(runRoot);
  const [freezeArtifact, reviewArtifact, claimArtifact] = await Promise.all([
    readExactJsonArtifact(runRoot, "candidate-freeze.json"),
    readExactJsonArtifact(runRoot, "independent-review.json"),
    readExactJsonArtifact(runRoot, "claim-audit.json")
  ]);

  const freezeErrors: string[] = [];
  const reviewErrors: string[] = [];
  const claimErrors: string[] = [];
  const freeze = freezeArtifact.payload;
  const review = reviewArtifact.payload;
  const claim = claimArtifact.payload;

  if (freezeArtifact.present) {
    if (freezeArtifact.error) freezeErrors.push(freezeArtifact.error);
    if (!normalizeSha256(freezeArtifact.sha256)) freezeErrors.push("candidate-freeze.json has no valid computed SHA-256.");
    if (!freeze) {
      freezeErrors.push("candidate-freeze.json payload is unavailable.");
    } else {
      if (freeze.status !== "frozen_before_independent_review") {
        freezeErrors.push("Candidate Freeze status is not frozen_before_independent_review.");
      }
      if (freeze.task_id !== taskId || freeze.run_id !== runRootName) {
        freezeErrors.push("Candidate Freeze task/run binding does not match the exact run root.");
      }
      if (freeze.official_submission_performed !== false || freeze.private_grader_used !== false) {
        freezeErrors.push("Candidate Freeze state is not pre-review and submission-free.");
      }
      if (!stringValue(freeze.best_exp_id, 100)) freezeErrors.push("Candidate Freeze has no best_exp_id.");
      const files = Array.isArray(freeze.files) ? freeze.files : [];
      if (!files.length) {
        freezeErrors.push("Candidate Freeze has no frozen file hashes.");
      } else if (files.some((entry) => {
        const item = recordOf(entry);
        return !item || !stringValue(item.path, 300) || !normalizeSha256(item.sha256);
      })) {
        freezeErrors.push("Candidate Freeze contains an invalid frozen file hash record.");
      }
    }
  }

  const runBindingValid = freezeArtifact.present
    ? Boolean(freeze && freeze.task_id === taskId && freeze.run_id === runRootName)
    : null;
  const freezeSha = normalizeSha256(freezeArtifact.sha256);
  const reviewFreezeSha = normalizeSha256(review?.candidate_freeze_sha256);
  const reviewCandidateFreezeHashValid = reviewArtifact.present
    ? Boolean(freezeArtifact.present && freezeSha && reviewFreezeSha === freezeSha)
    : null;

  if (reviewArtifact.present) {
    if (reviewArtifact.error) reviewErrors.push(reviewArtifact.error);
    if (!normalizeSha256(reviewArtifact.sha256)) reviewErrors.push("independent-review.json has no valid computed SHA-256.");
    if (!freezeArtifact.present) reviewErrors.push("Independent Review has no Candidate Freeze in the exact run root.");
    if (freezeErrors.length) reviewErrors.push("Independent Review depends on an unverified Candidate Freeze.");
    if (!review) {
      reviewErrors.push("independent-review.json payload is unavailable.");
    } else {
      if (review.status !== "passed") reviewErrors.push("Independent Review status is not passed.");
      if (!reviewCandidateFreezeHashValid) reviewErrors.push("Independent Review Candidate Freeze SHA-256 binding mismatches.");
      if (!stringValue(review.reviewer, 160)) reviewErrors.push("Independent Review reviewer is missing.");
      if (!stringValue(review.reviewed_at, 80)) reviewErrors.push("Independent Review timestamp is missing.");
      if (!positiveInteger(review.rows)) reviewErrors.push("Independent Review rows must be a positive integer.");
      if (!stringValue(review.metric, 100) || finiteNumber(review.score) === null) {
        reviewErrors.push("Independent Review metric/score is invalid.");
      }
      if (!normalizeSha256(review.submission_sha256) || !normalizeSha256(review.sealed_labels_sha256)) {
        reviewErrors.push("Independent Review evidence hashes are invalid.");
      }
      if (!stringValue(review.source, 160)) reviewErrors.push("Independent Review source is missing.");
      if (review.model_reload_passed !== true || review.official_submission_performed !== false) {
        reviewErrors.push("Independent Review state checks did not pass.");
      }

      const bestExpId = stringValue(freeze?.best_exp_id, 100);
      const frozenFiles = Array.isArray(freeze?.files) ? freeze.files : [];
      const frozenSubmission = bestExpId
        ? frozenFiles.map(recordOf).find((entry) => entry?.path === `${bestExpId}/out/submission.csv`)
        : null;
      const frozenSubmissionSha = normalizeSha256(frozenSubmission?.sha256);
      if (!frozenSubmissionSha || frozenSubmissionSha !== normalizeSha256(review.submission_sha256)) {
        reviewErrors.push("Independent Review submission SHA-256 is not bound to the frozen candidate.");
      }
    }
  }

  const reviewSha = normalizeSha256(reviewArtifact.sha256);
  const evidence = recordOf(claim?.evidence);
  const claimFreezeSha = normalizeSha256(evidence?.candidate_freeze_sha256);
  const claimReviewSha = normalizeSha256(evidence?.independent_review_sha256)
    ?? normalizeSha256(evidence?.review_sha256);
  const claimCandidateFreezeHashValid = claimArtifact.present
    ? Boolean(freezeArtifact.present && freezeSha && claimFreezeSha === freezeSha)
    : null;
  const claimIndependentReviewHashValid = claimArtifact.present
    ? Boolean(reviewArtifact.present && reviewSha && claimReviewSha === reviewSha)
    : null;

  if (claimArtifact.present) {
    if (claimArtifact.error) claimErrors.push(claimArtifact.error);
    if (!normalizeSha256(claimArtifact.sha256)) claimErrors.push("claim-audit.json has no valid computed SHA-256.");
    if (!freezeArtifact.present || !reviewArtifact.present) {
      claimErrors.push("Claim Audit does not have the complete Freeze to Review chain in the exact run root.");
    }
    if (freezeErrors.length || reviewErrors.length) claimErrors.push("Claim Audit depends on an unverified upstream stage.");
    if (!claim) {
      claimErrors.push("claim-audit.json payload is unavailable.");
    } else {
      if (claim.status !== "passed") claimErrors.push("Claim Audit status is not passed.");
      if (!claimCandidateFreezeHashValid) claimErrors.push("Claim Audit Candidate Freeze SHA-256 binding mismatches.");
      if (!claimIndependentReviewHashValid) claimErrors.push("Claim Audit Independent Review SHA-256 binding mismatches.");
      if (!Array.isArray(claim.supported_claims) || !Array.isArray(claim.unsupported_claims)) {
        claimErrors.push("Claim Audit claim lists are invalid.");
      }
      if (!stringValue(claim.claim_boundary, 1_000)) claimErrors.push("Claim Audit boundary is missing.");
    }
  }

  const freezeStage = stageStatus(freezeArtifact.present, freezeErrors);
  const reviewStage = stageStatus(reviewArtifact.present, reviewErrors);
  const claimStage = stageStatus(claimArtifact.present, claimErrors);
  const errors = [...freezeErrors, ...reviewErrors, ...claimErrors].slice(0, 32);
  const anyPresent = freezeArtifact.present || reviewArtifact.present || claimArtifact.present;
  const status: EvolutionReviewChainIntegrity["status"] = !anyPresent
    ? "not_present"
    : errors.length
      ? "failed"
      : "verified";
  const artifactHashesValid = anyPresent
    ? [freezeArtifact, reviewArtifact, claimArtifact]
      .filter((artifact) => artifact.present)
      .every((artifact) => Boolean(normalizeSha256(artifact.sha256)))
    : null;

  const freezeTrusted = freezeStage === "verified";
  const reviewTrusted = reviewStage === "verified";
  const claimTrusted = claimStage === "verified";
  const frozenFiles = Array.isArray(freeze?.files) ? freeze.files : [];

  return {
    candidate_freeze: {
      present: freezeArtifact.present,
      status: artifactStatus(freezeArtifact),
      candidate_id: freezeTrusted ? stringValue(freeze?.best_exp_id ?? freeze?.candidate_id, 100) : null,
      frozen_at: freezeTrusted ? stringValue(freeze?.frozen_at, 80) : null,
      sha256: freezeArtifact.sha256,
      artifact_count: freezeTrusted ? frozenFiles.length : null,
      source: "local_candidate_freeze",
      artifact: freezeArtifact.present ? freezeArtifact.name : null
    },
    independent_review: {
      present: reviewArtifact.present,
      status: artifactStatus(reviewArtifact),
      reviewer: reviewTrusted ? stringValue(review?.reviewer, 160) : null,
      reviewed_at: reviewTrusted ? stringValue(review?.reviewed_at, 80) : null,
      rows: reviewTrusted ? positiveInteger(review?.rows) : null,
      metric: reviewTrusted ? stringValue(review?.metric, 100) : null,
      score: reviewTrusted ? finiteNumber(review?.score) : null,
      sha256: reviewArtifact.sha256,
      candidate_freeze_sha256: reviewTrusted ? reviewFreezeSha : null,
      review_source: reviewTrusted ? stringValue(review?.source, 160) : null,
      artifact: reviewArtifact.present ? "independent-review.json" : null
    },
    claim_audit: {
      present: claimArtifact.present,
      status: artifactStatus(claimArtifact),
      sha256: claimArtifact.sha256,
      supported_claim_count: claimTrusted && Array.isArray(claim?.supported_claims) ? claim.supported_claims.length : null,
      unsupported_claim_count: claimTrusted && Array.isArray(claim?.unsupported_claims) ? claim.unsupported_claims.length : null,
      claim_boundary: claimTrusted ? stringValue(claim?.claim_boundary, 1_000) : null,
      artifact: claimArtifact.present ? "claim-audit.json" : null
    },
    review_chain: {
      status,
      stages: {
        candidate_freeze: freezeStage,
        independent_review: reviewStage,
        claim_audit: claimStage
      },
      artifact_hashes_valid: artifactHashesValid,
      run_binding_valid: runBindingValid,
      review_candidate_freeze_hash_valid: reviewCandidateFreezeHashValid,
      claim_candidate_freeze_hash_valid: claimCandidateFreezeHashValid,
      claim_independent_review_hash_valid: claimIndependentReviewHashValid,
      errors
    },
    artifact_names: [freezeArtifact, reviewArtifact, claimArtifact]
      .filter((artifact) => artifact.present)
      .map((artifact) => artifact.name)
  };
}
