import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { loadExperienceReviewAudit } from "./experience-review-audit.ts";

const TASK_ID = "fixture_task";
const SUBMISSION_SHA = "a".repeat(64);
const LABELS_SHA = "b".repeat(64);

async function writeJson(runRoot: string, name: string, payload: Record<string, unknown>) {
  const bytes = Buffer.from(`${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  await fs.writeFile(path.join(runRoot, name), bytes);
  return createHash("sha256").update(bytes).digest("hex");
}

async function writeValidChain(
  runRoot: string,
  options: {
    reviewStatus?: string;
    claimStatus?: string;
    claimReviewSha?: string;
    unsupportedClaims?: string[];
  } = {}
) {
  await fs.mkdir(runRoot, { recursive: true });
  const freeze: Record<string, unknown> = {
    schema: "evomind.demo.candidate_freeze.v1",
    task_id: TASK_ID,
    run_id: path.basename(runRoot),
    best_exp_id: "EXP003",
    status: "frozen_before_independent_review",
    frozen_at: "2026-08-02T09:28:33.266468+00:00",
    files: [
      { path: "EXP003/out/submission.csv", bytes: 123, sha256: SUBMISSION_SHA },
      { path: "EXP003/out/metrics.json", bytes: 45, sha256: "c".repeat(64) }
    ],
    official_submission_performed: false,
    private_grader_used: false
  };
  const freezeSha = await writeJson(runRoot, "candidate-freeze.json", freeze);
  const review: Record<string, unknown> = {
    schema: "evomind.demo.independent_review.v1",
    status: options.reviewStatus ?? "passed",
    reviewer: "IndependentFixtureReviewer",
    reviewed_at: "2026-08-02T09:29:00.000000+00:00",
    candidate_freeze_sha256: freezeSha,
    submission_sha256: SUBMISSION_SHA,
    sealed_labels_sha256: LABELS_SHA,
    metric: "roc_auc",
    score: 0.91,
    rows: 600,
    model_reload_passed: true,
    source: "sealed_fixture_holdout",
    official_submission_performed: false
  };
  const reviewSha = await writeJson(runRoot, "independent-review.json", review);
  const claim: Record<string, unknown> = {
    schema: "evomind.demo.claim_audit.v1",
    status: options.claimStatus ?? "passed",
    supported_claims: ["The frozen candidate was independently reviewed."],
    unsupported_claims: options.unsupportedClaims ?? [],
    evidence: {
      candidate_freeze_sha256: freezeSha,
      independent_review_sha256: options.claimReviewSha ?? reviewSha
    },
    claim_boundary: "Fixture evidence only; no external result is inferred."
  };
  const claimSha = await writeJson(runRoot, "claim-audit.json", claim);
  return { freeze, review, claim, freezeSha, reviewSha, claimSha };
}

test("projects the exact Freeze to Review to Claim chain with computed artifact hashes", async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "experience-review-valid-"));
  const runRoot = path.join(temp, "run_exact");
  try {
    const fixture = await writeValidChain(runRoot, { unsupportedClaims: ["Unsupported fixture claim"] });
    const projection = await loadExperienceReviewAudit(runRoot, TASK_ID);

    assert.equal(projection.review_chain.status, "verified");
    assert.deepEqual(projection.review_chain.stages, {
      candidate_freeze: "verified",
      independent_review: "verified",
      claim_audit: "verified"
    });
    assert.equal(projection.candidate_freeze.sha256, fixture.freezeSha);
    assert.equal(projection.independent_review.sha256, fixture.reviewSha);
    assert.equal(projection.claim_audit.sha256, fixture.claimSha);
    assert.equal(projection.independent_review.reviewer, "IndependentFixtureReviewer");
    assert.equal(projection.independent_review.rows, 600);
    assert.equal(projection.claim_audit.unsupported_claim_count, 1);
    assert.match(projection.claim_audit.claim_boundary ?? "", /Fixture evidence only/);
  } finally {
    await fs.rm(temp, { recursive: true, force: true });
  }
});

test("fails closed when the exact Candidate Freeze bytes change and never reads a sibling run", async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "experience-review-freeze-tamper-"));
  const selectedRoot = path.join(temp, "selected_run");
  const siblingRoot = path.join(temp, "newer_valid_run");
  try {
    const selected = await writeValidChain(selectedRoot);
    await writeValidChain(siblingRoot);
    await writeJson(selectedRoot, "candidate-freeze.json", {
      ...selected.freeze,
      frozen_at: "2026-08-02T10:00:00.000000+00:00"
    });

    const projection = await loadExperienceReviewAudit(selectedRoot, TASK_ID);
    assert.equal(projection.review_chain.status, "failed");
    assert.equal(projection.review_chain.stages.independent_review, "failed");
    assert.equal(projection.review_chain.stages.claim_audit, "failed");
    assert.equal(projection.review_chain.review_candidate_freeze_hash_valid, false);
    assert.equal(projection.independent_review.reviewer, null);
    assert.equal(projection.independent_review.rows, null);
    assert.equal(projection.claim_audit.claim_boundary, null);
    assert.match(projection.review_chain.errors.join(" "), /Candidate Freeze SHA-256 binding mismatches/);
  } finally {
    await fs.rm(temp, { recursive: true, force: true });
  }
});

test("keeps a verified Independent Review but suppresses a Claim Audit with a broken review hash", async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "experience-review-claim-tamper-"));
  const runRoot = path.join(temp, "run_exact");
  try {
    await writeValidChain(runRoot, { claimReviewSha: "0".repeat(64) });
    const projection = await loadExperienceReviewAudit(runRoot, TASK_ID);

    assert.equal(projection.review_chain.status, "failed");
    assert.equal(projection.review_chain.stages.independent_review, "verified");
    assert.equal(projection.independent_review.reviewer, "IndependentFixtureReviewer");
    assert.equal(projection.review_chain.stages.claim_audit, "failed");
    assert.equal(projection.review_chain.claim_independent_review_hash_valid, false);
    assert.equal(projection.claim_audit.unsupported_claim_count, null);
    assert.equal(projection.claim_audit.claim_boundary, null);
  } finally {
    await fs.rm(temp, { recursive: true, force: true });
  }
});

test("does not project review or claim details when an upstream status is not passed", async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "experience-review-status-"));
  const runRoot = path.join(temp, "run_exact");
  try {
    await writeValidChain(runRoot, { reviewStatus: "failed" });
    const projection = await loadExperienceReviewAudit(runRoot, TASK_ID);

    assert.equal(projection.independent_review.present, true);
    assert.equal(projection.independent_review.status, "failed");
    assert.equal(projection.independent_review.reviewer, null);
    assert.equal(projection.review_chain.stages.independent_review, "failed");
    assert.equal(projection.review_chain.stages.claim_audit, "failed");
    assert.equal(projection.claim_audit.claim_boundary, null);
  } finally {
    await fs.rm(temp, { recursive: true, force: true });
  }
});
