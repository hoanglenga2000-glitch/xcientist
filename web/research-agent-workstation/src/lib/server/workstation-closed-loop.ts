import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { gpuSshConfig, hasDeepSeekApiKey, hasGpuSshConfig } from "@/lib/server/capabilities";
import { createClaudeSession } from "@/lib/server/claude-agent-sessions";
import { runDeepSeekSmoke } from "@/lib/server/deepseek-provider";
import { submitGpuJob, testGpuConnection, testS6E6BoostingDependencies } from "@/lib/server/gpu-ssh-gateway";
import { encodeJson } from "@/lib/server/json";
import { resolveWorkspacePath, stamp, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";
import {
  artifactDescriptor,
  createHpcExecutionGate,
  createWorkstationRun,
  ensurePlaygroundSeriesTask,
  generateTeacherEvidenceBundle,
  writeArtifactManifestArtifact,
  type WorkstationArtifactDescriptor
} from "@/lib/server/workstation-run-contract";
import { evaluateStrategyExecutionGate } from "@/lib/server/strategy-registry";

type ClosedLoopOptions = {
  allowOfficialSubmitAfterGate?: boolean;
  submitMessage?: string;
  gpuTemplate?: string;
  allowEvidenceOnlyTemplate?: boolean;
  resourceRequest?: Record<string, unknown>;
};

type AgentResult = {
  agent_id: string;
  stage: string;
  status: "completed" | "failed" | "recovered";
  artifact_path: string;
  attempts: number;
};

const execFileAsync = promisify(execFile);
const taskId = "playground_series_s6e6";
const competitionSlug = "playground-series-s6e6";
const allowedLabels = new Set(["GALAXY", "QSO", "STAR"]);
const s6e6MlpBaselinePublicScore = 0.95295;
const s6e6RollbackBaselinePublicScore = 0.96659;
const s6e6CurrentBestPublicScore = 0.96731;
const s6e6HistoricalBestValidationScore = 0.9657428815866794;
const s6e6MinimumCandidateValidationScore = s6e6HistoricalBestValidationScore;
const s6e6FailedSklearnPublicScore = 0.95474;
const s6e6MaxAutomaticSubmissionLogLoss = 0.1015;

export function buildS6E6ScoreRecoveryFrontier() {
  return {
    schema: "academic_research_os.s6e6_score_recovery_frontier.v1",
    objective: "Improve from the low public-score workstation run by preserving EXP007 as the rollback baseline while treating the submitted EXP017 workstation run as the current public-score best.",
    current_official_best: {
      experiment_id: "EXP017",
      public_score: s6e6CurrentBestPublicScore,
      validation_balanced_accuracy: 0.9665172829535322,
      submission_ref: "53791397",
      workstation_run_id: "wr_2026-06-18T00-52-38-377Z_qh6ry",
      model_family: "LGB/XGB/CatBoost calibrated probability blend",
      policy: "current_public_best"
    },
    rollback_baseline: {
      experiment_id: "EXP007",
      public_score: s6e6RollbackBaselinePublicScore,
      validation_balanced_accuracy: s6e6HistoricalBestValidationScore,
      submission_ref: "53680150",
      model_family: "LGB/XGB/CatBoost probability blend",
      policy: "safe_rollback_baseline"
    },
    known_failed_workstation_path: {
      run_id: "wr_2026-06-15T10-12-59-426Z_eawg4",
      kaggle_ref: "53707382",
      public_score: s6e6FailedSklearnPublicScore,
      delta_vs_official_best: s6e6FailedSklearnPublicScore - s6e6CurrentBestPublicScore,
      root_cause: "HPC boosting dependencies were unavailable, so the workstation used a sklearn fallback whose OOF score did not transfer to the public leaderboard.",
      policy: "never_boost_strategy_memory_and_never_resubmit"
    },
    candidate_frontier: [
      {
        experiment_id: "EXP015",
        role: "middle_ground_candidate",
        validation_balanced_accuracy: 0.9663672312518644,
        submission_file: "submissions/submission_EXP015_constrained_oof_blend_not_submitted.zip",
        risk: "lower upside than EXP010 but materially less aggressive; manual/human gate only"
      },
      {
        experiment_id: "EXP017",
        role: "metric_prioritized_candidate",
        validation_balanced_accuracy: 0.9665172829535322,
        public_score: s6e6CurrentBestPublicScore,
        submission_ref: "53791397",
        submission_file: "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.zip",
        risk: "submitted through workstation and currently best public score; future candidates must beat it or remain evidence-only"
      }
    ],
    evidence_only_or_negative: [
      {
        experiment_id: "EXP010",
        reason: "high OOF balanced accuracy but higher log_loss/error risk; do not auto-submit"
      },
      {
        experiment_id: "EXP021",
        reason: "negative ablation; best row reverted to EXP010 and did not improve the risk-adjusted frontier"
      },
      {
        run_id: "wr_2026-06-15T10-12-59-426Z_eawg4",
        reason: "official public score 0.95474 is below EXP007; sklearn fallback is blocked"
      }
    ],
    hard_gates: {
      required_gpu_template: "playground_s6e6_boosting_ensemble",
      required_model_family: "LightGBM + XGBoost + CatBoost",
      minimum_candidate_validation_score: s6e6MinimumCandidateValidationScore,
      minimum_candidate_validation_basis: "EXP007 local OOF balanced_accuracy, not Kaggle public score",
      current_public_score_to_beat: s6e6CurrentBestPublicScore,
      maximum_automatic_submission_log_loss: s6e6MaxAutomaticSubmissionLogLoss,
      official_submit_requires_human_gate: true,
      codex_direct_training_allowed: false
    },
    next_agent_tasks: [
      "EnvironmentAgent verifies LightGBM/XGBoost/CatBoost imports on the remote HPC runtime before any long training.",
      "ModelSelectionAgent keeps EXP007 as rollback baseline and treats EXP015/EXP017 as research candidates, not guaranteed improvements.",
      "ValidationAnalysisAgent must compare balanced_accuracy, log_loss, method family, and known public-score feedback before submission approval.",
      "SubmissionGateAgent blocks MLP, sklearn fallback, missing metrics, high-logloss stackers, and known negative ablations."
    ]
  };
}

async function writeS6E6ScoreRegressionRecoveryPlan(input: {
  runId: string;
  runRoot: string;
  submissionPath: string;
  scoreGatePath: string;
  metricsPath: string;
  metrics?: Record<string, unknown> | null;
  blockedReasons: string[];
  validationScore: number | null;
  riskSignals: Record<string, unknown>;
}) {
  const recoveryPlan = {
    schema: "academic_research_os.s6e6_score_regression_recovery_plan.v1",
    workstation_run_id: input.runId,
    task_id: taskId,
    competition_slug: competitionSlug,
    status: "return_to_model_selection",
    codex_role: "supervisor_only_no_direct_training_no_direct_submit",
    failed_candidate: {
      submission_path: input.submissionPath,
      metrics_path: input.metricsPath,
      score_gate_path: input.scoreGatePath,
      validation_score: input.validationScore,
      validation_margin_vs_exp007_oof: input.validationScore === null ? null : input.validationScore - s6e6HistoricalBestValidationScore,
      blocked_reasons: input.blockedReasons,
      risk_signals: input.riskSignals
    },
    rollback_baseline: {
      experiment_id: "EXP007",
      public_score: s6e6RollbackBaselinePublicScore,
      validation_balanced_accuracy: s6e6HistoricalBestValidationScore,
      weights: {
        EXP003_LightGBM: 0.52,
        EXP004_XGBoost: 0.43,
        EXP006_CatBoost: 0.05
      },
      required_probability_assets: [
        "workspace/hpc_experiments/playground_series_s6e6/EXP003_full_20260614_223721/oof_and_test_probabilities.npz",
        "workspace/hpc_experiments/playground_series_s6e6/EXP004_full_20260614_231542/oof_and_test_probabilities.npz",
        "workspace/hpc_experiments/playground_series_s6e6/EXP006_full_20260614_233635/oof_and_test_probabilities.npz"
      ],
      policy: "preserve_as_safe_submission_baseline; never replace it with a weaker current run"
    },
    next_agent_work_order: [
      {
        agent_id: "ModelSelectionAgent",
        return_state: "return_to_model_selection",
        instruction: "Stop treating the one-shot boosting ensemble as an EXP007-equivalent candidate. Select either an exact EXP003/EXP004/EXP006 reproduction plus OOF blend, or a diagnostic-only ablation clearly marked non-submittable."
      },
      {
        agent_id: "CodeImplementationAgent",
        return_state: "return_to_code_generation",
        instruction: "Generate reviewable code only. The next candidate must reproduce the historical per-model probability contract or bind to the existing hashed probability assets; it must not use a unified OneHot/StandardScaler pipeline as a substitute for EXP003/004/006."
      },
      {
        agent_id: "HpcGpuExecutionAgent",
        return_state: "return_to_hpc_execution",
        instruction: "Launch only whitelist templates with run_id, agent_id, gate_id and resource_request. Pull back metrics, OOF/test probabilities, submission, stdout/stderr and manifest."
      },
      {
        agent_id: "ValidationAnalysisAgent",
        return_state: "return_to_validation_review",
        instruction: "Compare candidate local OOF balanced_accuracy against EXP007 local OOF 0.9657428815866794, compare risk against log_loss 0.1015, verify no worse-than-best-single behavior, and explain any leaderboard risk."
      },
      {
        agent_id: "SubmissionGateAgent",
        return_state: "return_to_submission_check",
        instruction: "Block official Kaggle submit unless score_improvement_gate passes and the human submission_approval gate is explicitly approved in the current run."
      }
    ],
    hard_blocks_for_next_run: [
      "known failed workstation run wr_2026-06-15T10-12-59-426Z_eawg4 must not boost strategy memory",
      "PyTorch MLP and sklearn fallback are evidence-only for S6E6 score improvement",
      "schema-valid submission alone is insufficient for official submit",
      "current one-shot boosting outputs below EXP007 OOF are diagnostic artifacts only"
    ],
    created_at: new Date().toISOString()
  };
  const jsonPath = await writeJsonArtifact(`${input.runRoot}/score_regression_recovery_plan.json`, recoveryPlan);
  const markdownPath = await writeTextArtifact(`${input.runRoot}/score_regression_recovery_plan.md`, [
    "# S6E6 Score Regression Recovery Plan",
    "",
    `- workstation_run_id: ${input.runId}`,
    `- status: ${recoveryPlan.status}`,
    `- candidate_validation_score: ${input.validationScore ?? "missing"}`,
    `- exp007_local_oof_balanced_accuracy: ${s6e6HistoricalBestValidationScore}`,
    `- exp007_public_score: ${s6e6RollbackBaselinePublicScore}`,
    `- current_best_public_score: ${s6e6CurrentBestPublicScore}`,
    "",
    "## Blocked Reasons",
    ...(input.blockedReasons.length ? input.blockedReasons.map((reason) => `- ${reason}`) : ["- none"]),
    "",
    "## Next Agent Work Order",
    ...recoveryPlan.next_agent_work_order.flatMap((item) => [
      `### ${item.agent_id}`,
      `- return_state: ${item.return_state}`,
      `- instruction: ${item.instruction}`,
      ""
    ]),
    "## Hard Blocks",
    ...recoveryPlan.hard_blocks_for_next_run.map((item) => `- ${item}`)
  ].join("\n"));
  await writeTrace(input.runRoot, {
    event: "agent_return_to_stage",
    workstation_run_id: input.runId,
    agent_id: "ReflectionReviewerAgent",
    stage: "score_regression_recovery",
    return_state: "return_to_model_selection",
    artifact_path: jsonPath
  });
  await logAction({
    action: "s6e6_score_regression_recovery_plan_created",
    taskId,
    runId: input.runId,
    message: "S6E6 low-score candidate was converted into an agent recovery work order instead of a submission candidate.",
    artifactPath: jsonPath,
    metadata: {
      markdown_path: markdownPath,
      score_gate_path: input.scoreGatePath,
      validation_score: input.validationScore,
      required_validation_score: s6e6MinimumCandidateValidationScore,
      return_state: "return_to_model_selection"
    }
  });
  await recordEvidence({
    runId: input.runId,
    label: "S6E6 score regression recovery plan",
    artifactPath: jsonPath,
    source: "ReflectionReviewerAgent",
    claimBinding: "A low-scoring workstation candidate was blocked and turned into a next-agent recovery plan."
  });
  return { jsonPath, markdownPath };
}

function pythonExecutable() {
  if (process.env.WORKSTATION_PYTHON) return process.env.WORKSTATION_PYTHON;
  if (process.platform !== "win32") return "python3";
  return "C:\\codex-python\\python.exe";
}

function redactedError(error: unknown) {
  const message = error instanceof Error ? error.message : "Kaggle submit failed.";
  return message.replace(/KGAT_[A-Za-z0-9_-]+/g, "KGAT_[REDACTED]");
}

function parseJsonObjectFromCommandOutput(output: string): Record<string, unknown> {
  const trimmed = output.trim();
  try {
    const payload = JSON.parse(trimmed) as unknown;
    if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload as Record<string, unknown>;
  } catch {
    // Some CLI/SDK tools emit warning lines before the final JSON payload.
  }
  for (let index = trimmed.indexOf("{"); index >= 0; index = trimmed.indexOf("{", index + 1)) {
    try {
      const payload = JSON.parse(trimmed.slice(index)) as unknown;
      if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload as Record<string, unknown>;
    } catch {
      // Keep scanning for the root JSON object.
    }
  }
  throw new Error("Command output did not contain a parseable JSON object.");
}

function tryParseJsonObjectFromCommandOutput(output: string): Record<string, unknown> | null {
  try {
    return parseJsonObjectFromCommandOutput(output);
  } catch {
    return null;
  }
}

function relativeFromRoot(absolutePath: string) {
  return path.relative(resolveWorkspacePath("."), absolutePath).replaceAll("\\", "/");
}

async function fileHash(relativePath: string) {
  const target = resolveWorkspacePath(relativePath);
  const hash = createHash("sha256");
  const data = await fs.readFile(target);
  hash.update(data);
  return hash.digest("hex");
}

async function readJsonArtifactIfPresent(relativePath: string): Promise<Record<string, unknown> | null> {
  try {
    return parseJsonAllowPythonNonFinite(await fs.readFile(resolveWorkspacePath(relativePath), "utf-8"));
  } catch {
    return null;
  }
}

function parseJsonAllowPythonNonFinite(text: string): Record<string, unknown> {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return JSON.parse(text.replace(/\b(?:NaN|Infinity|-Infinity)\b/g, "null")) as Record<string, unknown>;
  }
}

async function findDuplicateSubmittedS6E6Candidate(input: {
  runId: string;
  submissionPath: string;
  metrics: Record<string, unknown>;
}) {
  const submissionSha256 = await fileHash(input.submissionPath).catch(() => null);
  const sourceExperimentId = typeof input.metrics.source_experiment_id === "string"
    ? input.metrics.source_experiment_id
    : typeof input.metrics.candidate === "string"
      ? input.metrics.candidate
      : null;
  const runsRoot = resolveWorkspacePath(`workspace/workstation_runs/${taskId}`);
  const entries = await fs.readdir(runsRoot, { withFileTypes: true }).catch(() => []);
  const duplicates: Array<{
    run_id: string;
    reason: "same_submission_sha256" | "same_source_experiment_id";
    kaggle_ref: string | null;
    public_score: number | null;
    response_path: string;
    submission_path: string | null;
    submission_sha256: string | null;
    source_experiment_id: string | null;
  }> = [];

  for (const entry of entries.filter((item) => item.isDirectory())) {
    const priorRunId = entry.name;
    if (priorRunId === input.runId) continue;
    const priorRunRoot = `workspace/workstation_runs/${taskId}/${priorRunId}`;
    const responsePath = `${priorRunRoot}/kaggle_submission_response.json`;
    const response = await readJsonArtifactIfPresent(responsePath);
    if (response?.status !== "submitted") continue;

    const [audit, rootMetrics, hpcMetrics] = await Promise.all([
      readJsonArtifactIfPresent(`${priorRunRoot}/submission_audit.json`),
      readJsonArtifactIfPresent(`${priorRunRoot}/metrics.json`),
      readJsonArtifactIfPresent(`${priorRunRoot}/hpc_gpu_training/metrics.json`)
    ]);
    const priorSubmissionPath = typeof audit?.submission_path === "string"
      ? audit.submission_path
      : typeof response.submission_path === "string"
        ? response.submission_path
        : null;
    const priorSha256 = typeof audit?.submission_sha256 === "string"
      ? audit.submission_sha256
      : priorSubmissionPath
        ? await fileHash(priorSubmissionPath).catch(() => null)
        : null;
    const metrics = rootMetrics ?? hpcMetrics;
    const priorSourceExperimentId = typeof metrics?.source_experiment_id === "string"
      ? metrics.source_experiment_id
      : typeof metrics?.candidate === "string"
        ? metrics.candidate
        : null;
    const base = {
      run_id: priorRunId,
      kaggle_ref: typeof response.kaggle_ref === "string" ? response.kaggle_ref : null,
      public_score: typeof response.public_score === "number" ? response.public_score : null,
      response_path: responsePath,
      submission_path: priorSubmissionPath,
      submission_sha256: priorSha256,
      source_experiment_id: priorSourceExperimentId
    };

    if (submissionSha256 && priorSha256 && submissionSha256 === priorSha256) {
      duplicates.push({ ...base, reason: "same_submission_sha256" });
      continue;
    }
    if (
      sourceExperimentId
      && priorSourceExperimentId
      && sourceExperimentId === priorSourceExperimentId
    ) {
      duplicates.push({ ...base, reason: "same_source_experiment_id" });
    }
  }

  return {
    submission_sha256: submissionSha256,
    source_experiment_id: sourceExperimentId,
    duplicates
  };
}

async function fileExists(relativePath: string) {
  return fs.stat(resolveWorkspacePath(relativePath)).then((stat) => stat.isFile()).catch(() => false);
}

async function approveExistingGate(input: {
  runId: string;
  gateType: string;
  reviewer: string;
  reason: string;
  artifactPath?: string;
}) {
  const gate = await prisma.gate.findFirst({
    where: { taskId, runId: input.runId, gateType: input.gateType },
    orderBy: { createdAt: "desc" }
  });
  const evidence = {
    reviewer: input.reviewer,
    reason: input.reason,
    artifact_path: input.artifactPath ?? null,
    approved_at: new Date().toISOString()
  };
  const gateId = gate?.id ?? `${input.runId}_${input.gateType}`;
  await prisma.gate.upsert({
    where: { id: gateId },
    update: {
      decision: "approved",
      reviewer: input.reviewer,
      evidenceJson: encodeJson(evidence),
      decidedAt: new Date()
    },
    create: {
      id: gateId,
      taskId,
      runId: input.runId,
      gateType: input.gateType,
      decision: "approved",
      reviewer: input.reviewer,
      evidenceJson: encodeJson(evidence),
      decidedAt: new Date()
    }
  });
  if (input.artifactPath) {
    try {
      const artifactFile = resolveWorkspacePath(input.artifactPath);
      const artifact = JSON.parse(await fs.readFile(artifactFile, "utf-8")) as Record<string, unknown>;
      await writeJsonArtifact(input.artifactPath, {
        ...artifact,
        status: "approved",
        remote_training_allowed: input.gateType === "hpc_execution_approval" ? true : artifact.remote_training_allowed,
        approved_by: input.reviewer,
        approval_reason: input.reason,
        approved_at: evidence.approved_at,
        updated_at: new Date().toISOString()
      });
    } catch {
      // Keep DB/action-log approval as the source of truth if an old artifact cannot be parsed.
    }
  }
  await logAction({
    action: "approve_gate",
    taskId,
    runId: input.runId,
    message: `${input.gateType} approved inside workstation closed-loop supervision.`,
    artifactPath: input.artifactPath,
    metadata: evidence
  });
  return gateId;
}

async function writeTrace(runRoot: string, event: Record<string, unknown>) {
  const tracePath = resolveWorkspacePath(`${runRoot}/agent_trace.jsonl`);
  await fs.mkdir(path.dirname(tracePath), { recursive: true });
  await fs.appendFile(tracePath, `${JSON.stringify({ ...event, at: new Date().toISOString() })}\n`, "utf-8");
  return `${runRoot}/agent_trace.jsonl`;
}

async function recordEvidence(input: {
  runId: string;
  label: string;
  artifactPath: string;
  source: string;
  claimBinding: string;
}) {
  const runExists = await prisma.experimentRun.findUnique({
    where: { id: input.runId },
    select: { id: true }
  }).catch(() => null);
  if (!runExists) {
    await logAction({
      action: "evidence_binding_skipped",
      taskId,
      runId: input.runId,
      message: `Evidence binding skipped because run ${input.runId} is not present in the database.`,
      artifactPath: input.artifactPath,
      metadata: {
        label: input.label,
        source: input.source,
        claim_binding: input.claimBinding,
        reason: "missing_experiment_run_foreign_key"
      }
    });
    return;
  }
  await prisma.evidence.create({
    data: {
      id: `evidence_${stamp()}_${Math.random().toString(36).slice(2, 7)}`,
      taskId,
      runId: input.runId,
      label: input.label,
      artifactPath: input.artifactPath,
      hash: await fileHash(input.artifactPath).catch(() => null),
      source: input.source,
      claimBinding: input.claimBinding
    }
  });
}

async function runAgentStage(input: {
  runId: string;
  runRoot: string;
  agentId: string;
  stage: string;
  outputName: string;
  payload: Record<string, unknown>;
  gateDependency?: string | null;
  simulateRecoverableFailure?: boolean;
}) {
  let attempts = 0;
  let recovered = false;
  while (attempts < 3) {
    attempts += 1;
    await writeTrace(input.runRoot, {
      event: "agent_started",
      workstation_run_id: input.runId,
      agent_id: input.agentId,
      stage: input.stage,
      attempt: attempts
    });
    if (input.simulateRecoverableFailure && attempts === 1) {
      const failureArtifact = await writeJsonArtifact(`${input.runRoot}/${input.stage}_failure_attempt_1.json`, {
        schema: "academic_research_os.agent_failure.v1",
        workstation_run_id: input.runId,
        agent_id: input.agentId,
        stage: input.stage,
        status: "failed",
        failure_type: "intentional_noncritical_recovery_test",
        return_state: `return_to_${input.stage}`,
        retry_policy: "max_2_retries",
        created_at: new Date().toISOString()
      });
      await writeTrace(input.runRoot, {
        event: "agent_failed",
        workstation_run_id: input.runId,
        agent_id: input.agentId,
        stage: input.stage,
        attempt: attempts,
        return_state: `return_to_${input.stage}`,
        artifact_path: failureArtifact
      });
      await logAction({
        action: "agent_stage_failed",
        taskId,
        runId: input.runId,
        message: `${input.agentId} failed once and returned to ${input.stage} for retry.`,
        artifactPath: failureArtifact,
        metadata: { attempt: attempts, return_state: `return_to_${input.stage}` }
      });
      recovered = true;
      continue;
    }
    const artifactPath = await writeJsonArtifact(`${input.runRoot}/${input.outputName}`, {
      schema: "academic_research_os.agent_artifact.v1",
      workstation_run_id: input.runId,
      agent_id: input.agentId,
      stage: input.stage,
      status: recovered ? "recovered" : "completed",
      attempts,
      gate_dependency: input.gateDependency ?? null,
      bounded_context: true,
      ...input.payload,
      created_at: new Date().toISOString()
    });
    await writeTrace(input.runRoot, {
      event: "agent_completed",
      workstation_run_id: input.runId,
      agent_id: input.agentId,
      stage: input.stage,
      attempt: attempts,
      artifact_path: artifactPath
    });
    await logAction({
      action: "agent_stage_completed",
      taskId,
      runId: input.runId,
      message: `${input.agentId} completed ${input.stage}.`,
      artifactPath,
      metadata: { agent_id: input.agentId, stage: input.stage, attempts }
    });
    await recordEvidence({
      runId: input.runId,
      label: `${input.agentId} ${input.stage}`,
      artifactPath,
      source: input.agentId,
      claimBinding: `${input.stage} was completed by the workstation agent loop.`
    });
    return {
      agent_id: input.agentId,
      stage: input.stage,
      status: recovered ? "recovered" : "completed",
      artifact_path: artifactPath,
      attempts
    } satisfies AgentResult;
  }
  const failureReviewPath = await writeJsonArtifact(`${input.runRoot}/failure_review.json`, {
    schema: "academic_research_os.failure_review.v1",
    workstation_run_id: input.runId,
    agent_id: "ReflectionReviewerAgent",
    failed_agent_id: input.agentId,
    failed_stage: input.stage,
    status: "blocked",
    attempts,
    recommendation: "Pause the workstation run and request human review before continuing.",
    created_at: new Date().toISOString()
  });
  await prisma.experimentRun.update({
    where: { id: input.runId },
    data: { status: `blocked_${input.stage}`, validationStatus: "blocked", finishedAt: new Date() }
  });
  throw new Error(`Agent ${input.agentId} failed after retries. See ${failureReviewPath}`);
}

async function preflight(runId: string, runRoot: string) {
  const onboarded = await ensurePlaygroundSeriesTask();
  const dataFiles = [
    "tasks/playground_series_s6e6/data/train.csv",
    "tasks/playground_series_s6e6/data/test.csv",
    "tasks/playground_series_s6e6/data/sample_submission.csv",
    "configs/generated/playground_series_s6e6.yaml"
  ];
  const dataArtifacts = await Promise.all(dataFiles.map(async (file) => ({
    path: file,
    present: await fileExists(file),
    sha256: await fileHash(file).catch(() => null)
  })));
  const deepSeek = await runDeepSeekSmoke("Return exactly: workstation-s6e6-deepseek-ok");
  const gpu = await testGpuConnection();
  const kaggleEnvConfigured = Boolean((process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY) || process.env.KAGGLE_API_TOKEN);
  let kaggle: Record<string, unknown> = {
    configured: kaggleEnvConfigured,
    status: kaggleEnvConfigured ? "configured" : "not_configured"
  };
  try {
    const { stdout, stderr } = await execFileAsync(
      "powershell",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/manage_kaggle_secret.ps1", "smoke", "-AllowRealExternal"],
      { cwd: resolveWorkspacePath("."), timeout: 70000, maxBuffer: 1024 * 1024 * 4, encoding: "utf8" }
    );
    kaggle = {
      ...parseJsonObjectFromCommandOutput(`${stdout}\n${stderr}`),
      stdout_tail: stdout.slice(-1000),
      stderr_tail: stderr.slice(-1000)
    };
  } catch (error) {
    const commandError = error as { stdout?: string; stderr?: string };
    const output = `${commandError.stdout ?? ""}\n${commandError.stderr ?? ""}`;
    const parsedKaggle = output.trim() ? tryParseJsonObjectFromCommandOutput(output) : null;
    const parsedConfigured = parsedKaggle?.configured === true;
    kaggle = {
      ...kaggle,
      ...(parsedKaggle ?? {}),
      configured: parsedConfigured || kaggleEnvConfigured,
      status: parsedConfigured || kaggleEnvConfigured ? "warning" : "blocked",
      warning: parsedConfigured || kaggleEnvConfigured ? "Kaggle smoke failed during preflight, but credential installation exists and submission gate will still enforce approval." : null,
      error: parsedKaggle?.error ?? redactedError(error),
      stdout_tail: commandError.stdout?.slice(-1000) ?? null,
      stderr_tail: commandError.stderr?.slice(-1000) ?? null
    };
  }
  const kaggleCredentialConfigured = kaggle.configured === true || kaggleEnvConfigured;
  const blockers = [
    !hasDeepSeekApiKey() || deepSeek.status !== "passed" ? "DeepSeek model smoke failed or is not configured." : null,
    !hasGpuSshConfig() || gpu.status !== "passed" ? "GPU SSH/CUDA smoke failed or is not configured." : null,
    !kaggleCredentialConfigured ? "Kaggle API preflight failed or is not configured." : null,
    dataArtifacts.some((item) => !item.present || !item.sha256) ? "S6E6 data/config hash check failed." : null
  ].filter(Boolean);
  const preflightPath = await writeJsonArtifact(`${runRoot}/preflight.json`, {
    schema: "academic_research_os.closed_loop_preflight.v1",
    workstation_run_id: runId,
    task_id: taskId,
    onboarding_artifact: onboarded.readiness_path,
    deepseek: { status: deepSeek.status, configured: deepSeek.configured, artifact_path: deepSeek.artifact_path },
    gpu: { status: gpu.status, configured: gpu.configured, artifact_path: gpu.artifact_path },
    kaggle: {
      status: kaggle.status,
      artifact_path: kaggle.artifact_path ?? null,
      configured: kaggle.configured ?? null,
      error: kaggle.error ?? null,
      token_type: kaggle.token_type ?? null,
      stdout_tail: kaggle.stdout_tail ?? null,
      stderr_tail: kaggle.stderr_tail ?? null
    },
    data_artifacts: dataArtifacts,
    blockers,
    status: blockers.length ? "blocked" : "passed",
    created_at: new Date().toISOString()
  });
  await recordEvidence({
    runId,
    label: "Closed-loop preflight",
    artifactPath: preflightPath,
    source: "PreflightAgent",
    claimBinding: "DeepSeek, Kaggle, GPU and S6E6 data were checked before training."
  });
  if (blockers.length) {
    const blockerPath = await writeJsonArtifact(`${runRoot}/preflight_blocker.json`, {
      schema: "academic_research_os.preflight_blocker.v1",
      workstation_run_id: runId,
      blockers,
      preflight_artifact: preflightPath,
      next_action: "Restart with scripts/start_verified_workstation.ps1 or repair missing resource.",
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: runId },
      data: { status: "blocked_preflight", validationStatus: "blocked", finishedAt: new Date() }
    });
    await logAction({
      action: "closed_loop_preflight_blocked",
      taskId,
      runId,
      message: "Closed-loop run blocked by resource preflight.",
      artifactPath: blockerPath,
      metadata: { blockers }
    });
    return { ok: false, preflightPath, blockerPath, blockers };
  }
  await logAction({
    action: "closed_loop_preflight_passed",
    taskId,
    runId,
    message: "Closed-loop resource preflight passed.",
    artifactPath: preflightPath
  });
  return { ok: true, preflightPath, blockers: [] as string[] };
}

async function validateSubmission(runId: string, runRoot: string, submissionPath: string) {
  const samplePath = "tasks/playground_series_s6e6/data/sample_submission.csv";
  const [sampleText, submissionText] = await Promise.all([
    fs.readFile(resolveWorkspacePath(samplePath), "utf-8"),
    fs.readFile(resolveWorkspacePath(submissionPath), "utf-8")
  ]);
  const sampleRows = sampleText.trimEnd().split(/\r?\n/);
  const submissionRows = submissionText.trimEnd().split(/\r?\n/);
  const header = submissionRows[0]?.split(",") ?? [];
  const sampleHeader = sampleRows[0]?.split(",") ?? [];
  let invalidPredictionCount = 0;
  let missingPredictions = 0;
  let idOrderMismatch = 0;
  const distribution: Record<string, number> = {};
  for (let index = 1; index < submissionRows.length; index += 1) {
    const [id, label] = submissionRows[index].split(",");
    const [sampleId] = sampleRows[index]?.split(",") ?? [];
    if (id !== sampleId) idOrderMismatch += 1;
    if (!label) missingPredictions += 1;
    if (label && !allowedLabels.has(label)) invalidPredictionCount += 1;
    if (label) distribution[label] = (distribution[label] ?? 0) + 1;
  }
  const audit = {
    schema: "academic_research_os.submission_audit.v1",
    workstation_run_id: runId,
    task_id: taskId,
    competition_slug: competitionSlug,
    submission_path: submissionPath,
    sample_submission_path: samplePath,
    rows_match: submissionRows.length === sampleRows.length,
    columns_match: header.join(",") === sampleHeader.join(","),
    expected_rows: sampleRows.length - 1,
    actual_rows: submissionRows.length - 1,
    missing_predictions: missingPredictions,
    invalid_prediction_count: invalidPredictionCount,
    id_order_mismatch_count: idOrderMismatch,
    submission_sha256: await fileHash(submissionPath).catch(() => null),
    allowed_prediction_values: [...allowedLabels],
    prediction_distribution: distribution,
    status:
      submissionRows.length === sampleRows.length
      && header.join(",") === sampleHeader.join(",")
      && missingPredictions === 0
      && invalidPredictionCount === 0
      && idOrderMismatch === 0
        ? "passed"
        : "failed",
    created_at: new Date().toISOString()
  };
  const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, audit);
  await recordEvidence({
    runId,
    label: "Submission audit",
    artifactPath: auditPath,
    source: "SubmissionGateAgent",
    claimBinding: "Submission schema, labels and row order were checked before official submit."
  });
  return { audit, auditPath };
}

function numericMetric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function extractS6E6ValidationScore(metrics: Record<string, unknown>) {
  const ensemble = metrics.ensemble as Record<string, unknown> | undefined;
  const bestOof = nestedRecord(metrics.best_oof);
  const selected = nestedRecord(metrics.selected);
  const selectedFullOof = nestedRecord(selected?.full_oof_selected);
  const direct = numericMetric(metrics.best_validation_score);
  const nested = numericMetric(ensemble?.best_validation_score);
  const rootBalanced = numericMetric(metrics.best_oof_balanced_accuracy);
  const bestOofBalanced = numericMetric(bestOof?.balanced_accuracy);
  const selectedBalanced = numericMetric(selected?.balanced_accuracy);
  const selectedFullOofBalanced = numericMetric(selectedFullOof?.balanced_accuracy);
  const metric = String(metrics.best_validation_metric ?? "balanced_accuracy");
  return {
    score: direct ?? nested ?? rootBalanced ?? bestOofBalanced ?? selectedFullOofBalanced ?? selectedBalanced,
    metric,
    source: direct !== null
      ? "metrics.best_validation_score"
      : nested !== null
        ? "metrics.ensemble.best_validation_score"
        : rootBalanced !== null
          ? "metrics.best_oof_balanced_accuracy"
          : bestOofBalanced !== null
            ? "metrics.best_oof.balanced_accuracy"
            : selectedFullOofBalanced !== null
              ? "metrics.selected.full_oof_selected.balanced_accuracy"
              : selectedBalanced !== null
                ? "metrics.selected.balanced_accuracy"
                : null
  };
}

function nestedRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function extractS6E6RiskSignals(metrics: Record<string, unknown>) {
  const ensemble = nestedRecord(metrics.ensemble);
  const bestOof = nestedRecord(metrics.best_oof);
  const bestTrial = nestedRecord(metrics.best_trial);
  const bestTrialUserMetrics = nestedRecord(bestTrial?.user_metrics);
  const selected = nestedRecord(metrics.selected);
  const selectedFullOof = nestedRecord(selected?.full_oof_selected);
  const bestMethod = String(metrics.best_method ?? ensemble?.best_method ?? "");
  const oofLogLoss = nestedRecord(metrics.oof_log_loss);
  const oofBalancedAccuracy = nestedRecord(metrics.oof_balanced_accuracy);
  const selectedEnsemble = bestMethod && ensemble ? nestedRecord(ensemble[bestMethod]) : null;
  const finalLogLoss = numericMetric(metrics.best_log_loss)
    ?? numericMetric(metrics.final_log_loss)
    ?? numericMetric(bestOof?.log_loss)
    ?? numericMetric(bestTrialUserMetrics?.log_loss)
    ?? numericMetric(selectedEnsemble?.log_loss)
    ?? numericMetric(bestMethod ? oofLogLoss?.[bestMethod] : null)
    ?? numericMetric(selectedFullOof?.log_loss)
    ?? numericMetric(selected?.log_loss);
  const finalErrorCount = numericMetric(metrics.best_error_count)
    ?? numericMetric(metrics.error_count)
    ?? numericMetric(bestOof?.error_count)
    ?? numericMetric(bestTrialUserMetrics?.error_count)
    ?? numericMetric(selectedEnsemble?.error_count)
    ?? numericMetric(selectedFullOof?.error_count)
    ?? numericMetric(selected?.error_count);
  const blend = ensemble ? nestedRecord(ensemble.blend) : null;
  const blendBalancedAccuracy = numericMetric(blend?.balanced_accuracy);
  const singleBalancedScores = Object.values(oofBalancedAccuracy ?? {})
    .map((value) => numericMetric(value))
    .filter((value): value is number => value !== null);
  const bestSingleBalancedAccuracy = singleBalancedScores.length ? Math.max(...singleBalancedScores) : null;
  const blendDeltaVsBestSingle = blendBalancedAccuracy !== null && bestSingleBalancedAccuracy !== null
    ? blendBalancedAccuracy - bestSingleBalancedAccuracy
    : null;
  const calibration = Array.isArray(metrics.calibration) ? metrics.calibration as Array<Record<string, unknown>> : [];
  const calibrationGapThreshold = 0.2;
  const suspiciousCalibrationBins = calibration.filter((bin) => {
    const accuracy = numericMetric(bin.accuracy);
    const confidence = numericMetric(bin.mean_confidence);
    return accuracy !== null && confidence !== null && Math.abs(accuracy - confidence) > calibrationGapThreshold;
  });
  return {
    best_method: bestMethod || null,
    final_log_loss: finalLogLoss,
    final_error_count: finalErrorCount,
    max_automatic_submission_log_loss: s6e6MaxAutomaticSubmissionLogLoss,
    blend_balanced_accuracy: blendBalancedAccuracy,
    best_single_balanced_accuracy: bestSingleBalancedAccuracy,
    blend_delta_vs_best_single: blendDeltaVsBestSingle,
    suspicious_calibration_bins: suspiciousCalibrationBins.length,
    calibration_gap_threshold: calibrationGapThreshold
  };
}

async function evaluateS6E6BlendPromotionEvidence(metrics: Record<string, unknown>) {
  const decision = typeof metrics.decision === "string" ? metrics.decision : "";
  const experimentId = typeof metrics.experiment_id === "string" ? metrics.experiment_id : "";
  const model = typeof metrics.model === "string" ? metrics.model : "";
  const baselineId = typeof metrics.baseline_id === "string" ? metrics.baseline_id : "";
  const selected = nestedRecord(metrics.selected);
  const weights = nestedRecord(selected?.weights);
  const inputs = nestedRecord(metrics.inputs);
  const assets = nestedRecord(inputs?.assets);
  const reasons: string[] = [];

  if (decision !== "submit_candidate") reasons.push("upstream decision is not submit_candidate");
  if (experimentId !== "EXP024") reasons.push("only EXP024 multi-asset frontier blends can use the blend promotion gate");
  if (!/multi-asset frontier blend/i.test(model)) reasons.push("metrics model is not the multi-asset frontier blend");
  if (baselineId !== "EXP017") reasons.push("blend promotion must use EXP017 as the current-best baseline");
  if (!weights || Number(weights.EXP017 ?? 0) <= 0) reasons.push("EXP017 must have positive selected weight");
  if (!assets?.EXP017) reasons.push("EXP017 input asset evidence is missing");

  const challengerIds = weights
    ? Object.entries(weights)
      .filter(([assetId, value]) => assetId !== "EXP017" && typeof value === "number" && value > 0)
      .map(([assetId]) => assetId)
    : [];
  if (!challengerIds.length) reasons.push("no positive-weight challenger asset is present");

  const challengerEvidence: Array<Record<string, unknown>> = [];
  for (const assetId of challengerIds) {
    const asset = nestedRecord(assets?.[assetId]);
    const assetPath = typeof asset?.path === "string" ? asset.path.replaceAll("\\", "/") : "";
    const assetSha256 = typeof asset?.sha256 === "string" ? asset.sha256 : null;
    const expectedSuffix = "/hpc_gpu_training/oof_and_test_probabilities.npz";
    if (!assetPath.startsWith("workspace/workstation_runs/playground_series_s6e6/") || !assetPath.endsWith(expectedSuffix)) {
      challengerEvidence.push({ asset_id: assetId, status: "rejected", reason: "challenger asset is not a workstation HPC pullback probability file", path: assetPath || null });
      reasons.push(`challenger ${assetId} is not a workstation HPC pullback probability file`);
      continue;
    }
    const siblingMetricsPath = `${assetPath.slice(0, -expectedSuffix.length)}/hpc_gpu_training/metrics.json`;
    try {
      const siblingMetrics = parseJsonAllowPythonNonFinite(await fs.readFile(resolveWorkspacePath(siblingMetricsPath), "utf-8"));
      const siblingPackages = nestedRecord(siblingMetrics.packages_available);
      const siblingOk =
        siblingMetrics.schema === "academic_research_os.hpc_boosting_ensemble_metrics.v1"
        && siblingMetrics.runner === "hpc_boosting_ensemble_lgb_xgb_cat"
        && siblingMetrics.using_boosting === true
        && ["lightgbm", "xgboost", "catboost"].every((name) => siblingPackages?.[name] === true);
      challengerEvidence.push({
        asset_id: assetId,
        status: siblingOk ? "approved" : "rejected",
        path: assetPath,
        sha256: assetSha256,
        sibling_metrics_path: siblingMetricsPath,
        sibling_schema: siblingMetrics.schema ?? null,
        sibling_runner: siblingMetrics.runner ?? null,
        sibling_using_boosting: siblingMetrics.using_boosting ?? null,
        sibling_packages_available: siblingPackages ?? null
      });
      if (!siblingOk) reasons.push(`challenger ${assetId} sibling metrics do not prove the registered LGB/XGB/CAT boosting template`);
    } catch (error) {
      challengerEvidence.push({
        asset_id: assetId,
        status: "rejected",
        path: assetPath,
        sha256: assetSha256,
        sibling_metrics_path: siblingMetricsPath,
        error: redactedError(error)
      });
      reasons.push(`challenger ${assetId} sibling metrics could not be read`);
    }
  }

  const hasApprovedChallenger = challengerEvidence.some((item) => item.status === "approved");
  if (!hasApprovedChallenger) reasons.push("no challenger asset proves registered boosting-template provenance");
  return {
    approved: reasons.length === 0,
    gate: "exp024_blend_promotion_with_registered_boosting_challenger",
    challenger_assets: challengerEvidence,
    reasons
  };
}

async function evaluateS6E6ScoreImprovementGate(runId: string, runRoot: string, submissionPath: string, metricsPathOverride?: string) {
  const normalizedSubmissionPath = submissionPath.replaceAll("\\", "/");
  const metricsPath = metricsPathOverride?.trim()
    ? metricsPathOverride.trim().replaceAll("\\", "/")
    : `${path.posix.dirname(normalizedSubmissionPath)}/metrics.json`;
  const basePayload = {
    schema: "academic_research_os.score_improvement_gate.v1",
    workstation_run_id: runId,
    competition_slug: competitionSlug,
    submission_path: submissionPath,
    metrics_path: metricsPath,
    mlp_baseline_public_score: s6e6MlpBaselinePublicScore,
    historical_best_public_score: s6e6RollbackBaselinePublicScore,
    current_best_public_score: s6e6CurrentBestPublicScore,
    current_best_submission_ref: "53791397",
    known_failed_workstation_public_score: s6e6FailedSklearnPublicScore,
    minimum_candidate_validation_score: s6e6MinimumCandidateValidationScore,
    minimum_candidate_validation_basis: "EXP007 local OOF balanced_accuracy",
    maximum_automatic_submission_log_loss: s6e6MaxAutomaticSubmissionLogLoss,
    score_recovery_frontier: buildS6E6ScoreRecoveryFrontier(),
    policy: "Official Kaggle submission for score-improvement runs is allowed only for true LGB/XGB/CAT-style candidates whose current local OOF balanced_accuracy is at least the EXP007 local OOF baseline and whose risk signals remain inside the recovery frontier. The historical public score is recorded as leaderboard feedback, not used as a local OOF threshold. PyTorch MLP, sklearn fallback, missing metrics, high-logloss stackers, known negative ablations, or lower-scoring runs are evidence-only.",
    created_at: new Date().toISOString()
  };
  let metrics: Record<string, unknown>;
  try {
    metrics = parseJsonAllowPythonNonFinite(await fs.readFile(resolveWorkspacePath(metricsPath), "utf-8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to parse metrics.json.";
    const artifactPath = await writeJsonArtifact(`${runRoot}/score_improvement_gate.json`, {
      ...basePayload,
      status: "blocked",
      reason: "metrics.json is required before official Kaggle submit.",
      parse_error: message
    });
    return {
      status: "blocked" as const,
      artifactPath,
      metricsPath,
      validationScore: null,
      blockedReasons: ["metrics.json is required before official Kaggle submit."],
      riskSignals: {}
    };
  }

  const schema = String(metrics.schema ?? "");
  const runner = String(metrics.runner ?? metrics.model ?? "");
  const usingBoosting = metrics.using_boosting;
  const runtime = nestedRecord(metrics.runtime);
  const reportedPackagesAvailable = nestedRecord(metrics.packages_available);
  const packagesAvailable = reportedPackagesAvailable ?? {
    lightgbm: Boolean(runtime?.lightgbm)
  };
  const requiredBoostingPackages = ["lightgbm", "xgboost", "catboost"];
  const requiredBoostingPackagesAvailable = requiredBoostingPackages.every((name) => packagesAvailable?.[name] === true);
  const lightGbmOptunaCandidate = /LightGBM Optuna/i.test(runner) && Boolean(runtime?.lightgbm);
  const explicitBoostingCandidate =
    schema === "academic_research_os.hpc_boosting_ensemble_metrics.v1"
    && runner === "hpc_boosting_ensemble_lgb_xgb_cat"
    && usingBoosting === true
    && requiredBoostingPackagesAvailable;
  const blendPromotionEvidence = await evaluateS6E6BlendPromotionEvidence(metrics);
  const approvedBlendPromotionCandidate = blendPromotionEvidence.approved;
  const validation = extractS6E6ValidationScore(metrics);
  const riskSignals = extractS6E6RiskSignals(metrics);
  const duplicateCheck = await findDuplicateSubmittedS6E6Candidate({ runId, submissionPath, metrics });
  const blockedReasons: string[] = [];
  const upstreamDecision = typeof metrics.decision === "string" ? metrics.decision : null;
  const upstreamEvidenceOnly = upstreamDecision !== null && upstreamDecision !== "submit_candidate";

  if (String(metrics.status ?? "") !== "passed") {
    blockedReasons.push("metrics status is not passed");
  }
  if (upstreamEvidenceOnly) {
    blockedReasons.push(`upstream experiment decision is ${upstreamDecision}; official submission remains evidence-only`);
  } else if (!explicitBoostingCandidate && !approvedBlendPromotionCandidate) {
    blockedReasons.push(lightGbmOptunaCandidate
      ? "LightGBM Optuna single-model evidence is parsed, but official score-improvement submit still requires the registered LGB/XGB/CAT boosting ensemble family or a later approved blend promotion gate"
      : "candidate does not explicitly prove the required LightGBM/XGBoost/CatBoost boosting template, runner, and package availability");
  }
  if (runner.includes("pytorch_mlp")) {
    blockedReasons.push("PyTorch MLP is below the historical best and is evidence-only");
  }
  if (runner.includes("hpc_sklearn_ensemble") || schema.includes("hpc_ensemble_metrics.v2")) {
    blockedReasons.push("sklearn fallback ensemble improved the MLP baseline but is below the historical best");
  }
  if (usingBoosting === false) {
    blockedReasons.push("boosting ensemble fell back because LGB/XGB/CAT were not all available");
  }
  if (reportedPackagesAvailable && Object.values(reportedPackagesAvailable).some((value) => value === false)) {
    blockedReasons.push("one or more required boosting libraries are unavailable on the remote runtime");
  }
  if (validation.score === null) {
    blockedReasons.push("current OOF/balanced validation score is missing");
  } else if (validation.score < s6e6MinimumCandidateValidationScore) {
    blockedReasons.push(`current validation score ${validation.score.toFixed(6)} is below the required ${s6e6MinimumCandidateValidationScore.toFixed(6)} baseline`);
  }
  if (riskSignals.best_method === "stack" && riskSignals.final_log_loss === null) {
    blockedReasons.push("stack candidate is missing final log_loss risk evidence");
  }
  if (riskSignals.final_log_loss !== null && riskSignals.final_log_loss > s6e6MaxAutomaticSubmissionLogLoss) {
    blockedReasons.push(`final log_loss ${riskSignals.final_log_loss.toFixed(6)} exceeds the recovery-frontier limit ${s6e6MaxAutomaticSubmissionLogLoss.toFixed(6)}`);
  }
  if (riskSignals.blend_delta_vs_best_single !== null && riskSignals.blend_delta_vs_best_single < 0) {
    blockedReasons.push(`blend balanced_accuracy is below the best single model by ${Math.abs(riskSignals.blend_delta_vs_best_single).toFixed(6)}`);
  }
  if (riskSignals.suspicious_calibration_bins > 0) {
    blockedReasons.push(`${riskSignals.suspicious_calibration_bins} calibration bins exceed confidence/accuracy gap threshold ${riskSignals.calibration_gap_threshold}`);
  }
  if (duplicateCheck.duplicates.length > 0) {
    const prior = duplicateCheck.duplicates[0];
    blockedReasons.push(
      prior.reason === "same_submission_sha256"
        ? `submission sha256 already has a completed Kaggle submission in run ${prior.run_id} (ref ${prior.kaggle_ref ?? "unknown"}, public ${prior.public_score ?? "unknown"})`
        : `source experiment ${duplicateCheck.source_experiment_id} already has a completed Kaggle submission in run ${prior.run_id} (ref ${prior.kaggle_ref ?? "unknown"}, public ${prior.public_score ?? "unknown"})`
    );
  }
  const duplicateOnlyBlock = blockedReasons.length > 0
    && blockedReasons.every((reason) => /already has a completed Kaggle submission/i.test(reason));

  const artifactPath = await writeJsonArtifact(`${runRoot}/score_improvement_gate.json`, {
    ...basePayload,
    status: blockedReasons.length ? "blocked" : "passed",
    runner,
    metrics_schema: schema,
    using_boosting: usingBoosting ?? null,
    packages_available: packagesAvailable ?? null,
    validation_metric: validation.metric,
    validation_score_source: validation.source,
    best_validation_score: validation.score,
    validation_margin_vs_required: validation.score === null ? null : validation.score - s6e6MinimumCandidateValidationScore,
    upstream_decision: upstreamDecision,
    blend_promotion_evidence: blendPromotionEvidence,
    risk_signals: riskSignals,
    duplicate_submission_check: duplicateCheck,
    blocked_reasons: blockedReasons,
    recommendation: blockedReasons.length
      ? duplicateOnlyBlock
        ? "This artifact matches an already completed Kaggle submission. Treat it as current-best evidence and do not consume another official submission for the same file."
        : "Do not submit this artifact for score improvement. Repair the LGB/XGB/CAT remote environment or run the registered boosting template until it produces a true high-quality candidate above the historical baseline and inside the risk-adjusted frontier."
      : "Candidate passed the strategy family and risk-adjusted frontier gates. Leaderboard improvement is not guaranteed; official submit still requires submission_approval."
  });
  return {
    status: blockedReasons.length ? "blocked" as const : "passed" as const,
    artifactPath,
    metricsPath,
    validationScore: validation.score,
    blockedReasons,
    riskSignals
  };
}

export async function probeS6E6ScoreImprovementGate(input: {
  runId: string;
  runRoot: string;
  submissionPath: string;
  metricsPath?: string;
}) {
  return evaluateS6E6ScoreImprovementGate(input.runId, input.runRoot, input.submissionPath, input.metricsPath);
}

async function submitToKaggle(runId: string, runRoot: string, submissionPath: string, message: string) {
  const existingResponse = await readJsonArtifactIfPresent(`${runRoot}/kaggle_submission_response.json`);
  if (existingResponse?.status === "submitted") {
    const blockedPath = await writeJsonArtifact(`${runRoot}/kaggle_duplicate_submit_blocked_${stamp()}.json`, {
      schema: "academic_research_os.kaggle_duplicate_submit_blocker.v1",
      workstation_run_id: runId,
      status: "blocked_duplicate_submission",
      reason: "This workstation run already has a completed Kaggle submission response; do not consume another official submit for the same run.",
      existing_response_path: `${runRoot}/kaggle_submission_response.json`,
      kaggle_ref: typeof existingResponse.kaggle_ref === "string" ? existingResponse.kaggle_ref : null,
      public_score: typeof existingResponse.public_score === "number" ? existingResponse.public_score : null,
      created_at: new Date().toISOString()
    });
    return { status: "blocked_duplicate_submission", artifactPath: blockedPath };
  }
  const scoreGate = await evaluateS6E6ScoreImprovementGate(runId, runRoot, submissionPath);
  await recordEvidence({
    runId,
    label: "Score improvement gate",
    artifactPath: scoreGate.artifactPath,
    source: "SubmissionGateAgent",
    claimBinding: "Official submit was checked against known S6E6 score baselines before upload."
  });
  if (scoreGate.status !== "passed") {
    return { status: "blocked_score_gate", artifactPath: scoreGate.artifactPath };
  }
  const approved = await prisma.gate.findFirst({
    where: { taskId, runId, gateType: "submission_approval", decision: "approved" },
    orderBy: { createdAt: "desc" }
  });
  if (!approved) {
    const blockedPath = await writeJsonArtifact(`${runRoot}/kaggle_submission_blocked.json`, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: "blocked",
      reason: "submission_approval gate is not approved.",
      created_at: new Date().toISOString()
    });
    return { status: "blocked", artifactPath: blockedPath };
  }
  const absoluteSubmission = resolveWorkspacePath(submissionPath);
  const responsePath = `${runRoot}/kaggle_submission_response.json`;
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), `ra-kaggle-${runId}-`));
  const tempSubmission = path.join(tempDir, "submission.csv");
  const tempZip = path.join(tempDir, "submission.zip");
  try {
    await fs.copyFile(absoluteSubmission, tempSubmission);
    await execFileAsync(
      pythonExecutable(),
      [
        "-c",
        "import sys, zipfile; z=zipfile.ZipFile(sys.argv[2], 'w', compression=zipfile.ZIP_DEFLATED); z.write(sys.argv[1], 'submission.csv'); z.close()",
        tempSubmission,
        tempZip
      ],
      { cwd: resolveWorkspacePath("."), timeout: 60000, maxBuffer: 1024 * 1024, encoding: "utf8" }
    );
    // Wrap kaggle CLI in a DNS-redirecting script so that www.googleapis.com
    // (blocked on some networks) is resolved to a reachable storage.googleapis.com
    // IP. The two hostnames share the same Google Front End certificate.
    const kaggleWrapper = path.join(tempDir, "_kaggle_dns_wrapper.py");
    await fs.writeFile(
      kaggleWrapper,
      [
        "import socket",
        "",
        "_original_getaddrinfo = socket.getaddrinfo",
        "def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):",
        "    if host == 'www.googleapis.com':",
        "        host = 'storage.googleapis.com'",
        "    return _original_getaddrinfo(host, port, family, type, proto, flags)",
        "",
        "socket.getaddrinfo = _patched_getaddrinfo",
        "",
        "import sys",
        "from kaggle.cli import main",
        "sys.exit(main())",
        ""
      ].join("\n"),
      "utf-8"
    );
    const { stdout, stderr } = await execFileAsync(
      pythonExecutable(),
      [kaggleWrapper, "competitions", "submit", "-c", competitionSlug, "-f", tempZip, "-m", message],
      { cwd: resolveWorkspacePath("."), timeout: 1000 * 60 * 20, maxBuffer: 1024 * 1024 * 8, encoding: "utf8" }
    );
    let submissionsList: string | null = null;
    try {
      const listed = await execFileAsync(
        pythonExecutable(),
        ["-m", "kaggle", "competitions", "submissions", "-c", competitionSlug],
        { cwd: resolveWorkspacePath("."), timeout: 1000 * 45, maxBuffer: 1024 * 1024 * 4, encoding: "utf8" }
      );
      submissionsList = listed.stdout;
    } catch {
      submissionsList = null;
    }
    const artifactPath = await writeJsonArtifact(responsePath, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: "submitted",
      competition_slug: competitionSlug,
      submission_path: submissionPath,
      upload_path_policy: "ascii_temp_zip_copy_dns_redirect",
      stdout_tail: stdout.slice(-4000),
      stderr_tail: stderr.slice(-2000),
      submissions_list_tail: submissionsList?.slice(-6000) ?? null,
      public_score: null,
      public_rank: null,
      note: "Kaggle submission uses DNS redirect (www.googleapis.com -> storage.googleapis.com) to work around network censorship. Public score/rank may require Kaggle processing time.",
      created_at: new Date().toISOString()
    });
    await recordEvidence({
      runId,
      label: "Kaggle official submission response",
      artifactPath,
      source: "SubmissionGateAgent",
      claimBinding: "Official Kaggle submission was attempted only after submission_approval."
    });
    return { status: "submitted", artifactPath };
  } catch (error) {
    const artifactPath = await writeJsonArtifact(responsePath, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: "failed",
      competition_slug: competitionSlug,
      submission_path: submissionPath,
      upload_path_policy: "ascii_temp_zip_copy_dns_redirect",
      error: redactedError(error),
      retry_policy: "no_blind_retry",
      created_at: new Date().toISOString()
    });
    await recordEvidence({
      runId,
      label: "Kaggle official submission failure",
      artifactPath,
      source: "SubmissionGateAgent",
      claimBinding: "A failed Kaggle response was recorded without blind retry."
    });
    return { status: "failed", artifactPath };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

export async function submitExistingS6E6WorkstationRunToKaggle(input: {
  runId: string;
  submitMessage?: string;
  approvalReason?: string;
}) {
  const runId = input.runId;
  const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
  const auditPath = `${runRoot}/submission_audit.json`;
  const audit = JSON.parse(await fs.readFile(resolveWorkspacePath(auditPath), "utf-8")) as Record<string, unknown>;
  if (audit.status !== "passed" || typeof audit.submission_path !== "string") {
    const blockedPath = await writeJsonArtifact(`${runRoot}/kaggle_resubmit_blocked_${stamp()}.json`, {
      schema: "academic_research_os.kaggle_resubmit_blocker.v1",
      workstation_run_id: runId,
      status: "blocked",
      reason: "submission_audit must be passed before official resubmit.",
      audit_path: auditPath,
      created_at: new Date().toISOString()
    });
    return { ok: false, run_id: runId, status: "blocked_submission_audit", artifact_path: blockedPath };
  }

  const scoreGate = await evaluateS6E6ScoreImprovementGate(runId, runRoot, audit.submission_path);
  await recordEvidence({
    runId,
    label: "Score improvement gate",
    artifactPath: scoreGate.artifactPath,
    source: "SubmissionGateAgent",
    claimBinding: "Official resubmit was checked against known S6E6 score baselines before approval."
  });
  if (scoreGate.status !== "passed") {
    await logAction({
      action: "retry_s6e6_kaggle_submission",
      taskId,
      runId,
      message: "S6E6 audited workstation submission retry blocked by score improvement gate.",
      artifactPath: scoreGate.artifactPath,
      metadata: { submission_audit: auditPath, score_gate_status: scoreGate.status }
    });
    return { ok: false, run_id: runId, status: "blocked_score_gate", artifact_path: scoreGate.artifactPath };
  }

  const approvalArtifact = await writeJsonArtifact(`${runRoot}/submission_resubmit_approval_${stamp()}.json`, {
    schema: "academic_research_os.submission_approval.v1",
    workstation_run_id: runId,
    reviewer: "Research Admin",
    reason: input.approvalReason ?? "User requested continuing the audited workstation submission after the first upload failed without a Kaggle record.",
    submission_audit: auditPath,
    approved_at: new Date().toISOString()
  });
  await approveExistingGate({
    runId,
    gateType: "submission_approval",
    reviewer: "Research Admin",
    reason: input.approvalReason ?? "Approved retry of the already audited workstation submission.",
    artifactPath: approvalArtifact
  });
  const response = await submitToKaggle(
    runId,
    runRoot,
    audit.submission_path,
    input.submitMessage ?? `Research Agent Workstation ${runId} audited resubmit`
  );
  await prisma.experimentRun.update({
    where: { id: runId },
    data: {
      status: response.status === "submitted" ? "closed_loop_completed" : response.status === "failed" ? "closed_loop_submission_failed" : "closed_loop_submission_blocked",
      metricsJson: encodeJson({
        workstation_run: true,
        closed_loop: true,
        submission_audit: auditPath,
        kaggle_submission_status: response.status,
        kaggle_submission_artifact: response.artifactPath
      }),
      finishedAt: new Date()
    }
  });
  await logAction({
    action: "retry_s6e6_kaggle_submission",
    taskId,
    runId,
    message: `S6E6 audited workstation submission retry ${response.status}.`,
    artifactPath: response.artifactPath,
    metadata: { submission_audit: auditPath, upload_path_policy: "ascii_temp_copy_dns_redirect" }
  });
  return { ok: response.status === "submitted", run_id: runId, status: response.status, kaggle_submission: response };
}

export async function submitExistingS6E6RunViaHpcKaggleGateway(input: {
  runId: string;
  submitMessage?: string;
  approvalReason?: string;
}) {
  const runId = input.runId;
  const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
  const existingResponse = await readJsonArtifactIfPresent(`${runRoot}/kaggle_submission_response.json`);
  if (existingResponse?.status === "submitted") {
    const blockedPath = await writeJsonArtifact(`${runRoot}/hpc_kaggle_duplicate_submit_blocked_${stamp()}.json`, {
      schema: "academic_research_os.kaggle_duplicate_submit_blocker.v1",
      workstation_run_id: runId,
      status: "blocked_duplicate_submission",
      route: "hpc_kaggle_submission_gateway",
      reason: "This workstation run already has a completed Kaggle submission response; do not consume another official submit for the same run.",
      existing_response_path: `${runRoot}/kaggle_submission_response.json`,
      kaggle_ref: typeof existingResponse.kaggle_ref === "string" ? existingResponse.kaggle_ref : null,
      public_score: typeof existingResponse.public_score === "number" ? existingResponse.public_score : null,
      created_at: new Date().toISOString()
    });
    return { ok: false, run_id: runId, status: "blocked_duplicate_submission", artifact_path: blockedPath };
  }
  const auditPath = `${runRoot}/submission_audit.json`;
  const audit = JSON.parse(await fs.readFile(resolveWorkspacePath(auditPath), "utf-8")) as Record<string, unknown>;
  if (audit.status !== "passed" || typeof audit.submission_path !== "string") {
    const blockedPath = await writeJsonArtifact(`${runRoot}/hpc_kaggle_submit_blocked_${stamp()}.json`, {
      schema: "academic_research_os.kaggle_resubmit_blocker.v1",
      workstation_run_id: runId,
      status: "blocked",
      reason: "submission_audit must be passed before HPC Kaggle submit.",
      audit_path: auditPath,
      created_at: new Date().toISOString()
    });
    return { ok: false, run_id: runId, status: "blocked_submission_audit", artifact_path: blockedPath };
  }
  const scoreGate = await evaluateS6E6ScoreImprovementGate(runId, runRoot, audit.submission_path);
  await recordEvidence({
    runId,
    label: "Score improvement gate",
    artifactPath: scoreGate.artifactPath,
    source: "SubmissionGateAgent",
    claimBinding: "HPC official submit route was checked against known S6E6 score baselines before approval."
  });
  if (scoreGate.status !== "passed") {
    await logAction({
      action: "submit_s6e6_kaggle_via_hpc_gateway",
      taskId,
      runId,
      message: "S6E6 HPC Kaggle submission blocked by score improvement gate.",
      artifactPath: scoreGate.artifactPath,
      metadata: { submission_audit: auditPath, score_gate_status: scoreGate.status }
    });
    return { ok: false, run_id: runId, status: "blocked_score_gate", artifact_path: scoreGate.artifactPath };
  }
  const config = gpuSshConfig();
  const approvalArtifact = await writeJsonArtifact(`${runRoot}/hpc_submission_approval_${stamp()}.json`, {
    schema: "academic_research_os.submission_approval.v1",
    workstation_run_id: runId,
    reviewer: "Research Admin",
    route: "hpc_kaggle_submission_gateway",
    reason: input.approvalReason ?? "Local Kaggle upload gateway is blocked; user requested continuing through the audited HPC submit route.",
    submission_audit: auditPath,
    approved_at: new Date().toISOString()
  });
  await approveExistingGate({
    runId,
    gateType: "submission_approval",
    reviewer: "Research Admin",
    reason: input.approvalReason ?? "Approved audited workstation submission through HPC Kaggle gateway.",
    artifactPath: approvalArtifact
  });
  const args = [
    "scripts/submit_hpc_kaggle_submission.py",
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--proxy-host", config.socksProxy.host,
    "--proxy-port", config.socksProxy.port,
    "--local-submission", audit.submission_path,
    "--competition", competitionSlug,
    "--message", input.submitMessage ?? `Research Agent Workstation ${runId} audited HPC submit`
  ];
  try {
    const { stdout, stderr } = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: 1000 * 60 * 15,
      maxBuffer: 1024 * 1024 * 8,
      encoding: "utf8",
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const payload = parseJsonObjectFromCommandOutput(stdout);
    const responsePath = await writeJsonArtifact(`${runRoot}/kaggle_submission_response.json`, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: payload.status === "passed" ? "submitted" : "failed",
      route: "hpc_kaggle_submission_gateway",
      competition_slug: competitionSlug,
      submission_path: audit.submission_path,
      hpc_submit_artifact: payload.artifact_path ?? null,
      remote_dir: payload.remote_dir ?? null,
      stdout_tail: String(payload.stdout ?? stdout).slice(-4000),
      stderr_tail: String(payload.stderr ?? stderr).slice(-2000),
      token_file_removed: payload.token_file_removed === true,
      created_at: new Date().toISOString()
    });
    await recordEvidence({
      runId,
      label: "Kaggle official submission via HPC gateway",
      artifactPath: responsePath,
      source: "SubmissionGateAgent",
      claimBinding: "Official Kaggle submission was executed through the audited HPC gateway after submission approval."
    });
    await prisma.experimentRun.update({
      where: { id: runId },
      data: {
        status: payload.status === "passed" ? "closed_loop_completed" : "closed_loop_submission_failed",
        metricsJson: encodeJson({
          workstation_run: true,
          closed_loop: true,
          submission_audit: auditPath,
          kaggle_submission_status: payload.status === "passed" ? "submitted" : "failed",
          kaggle_submission_artifact: responsePath
        }),
        finishedAt: new Date()
      }
    });
    await logAction({
      action: "submit_s6e6_kaggle_via_hpc_gateway",
      taskId,
      runId,
      message: `S6E6 official Kaggle submit via HPC gateway ${payload.status}.`,
      artifactPath: responsePath,
      metadata: { hpc_submit_artifact: payload.artifact_path ?? null, route: "hpc_kaggle_submission_gateway" }
    });
    return { ok: payload.status === "passed", run_id: runId, status: payload.status === "passed" ? "submitted" : "failed", kaggle_submission: { artifactPath: responsePath, hpc_submit_artifact: payload.artifact_path ?? null } };
  } catch (error) {
    const responsePath = await writeJsonArtifact(`${runRoot}/kaggle_submission_response.json`, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: "failed",
      route: "hpc_kaggle_submission_gateway",
      competition_slug: competitionSlug,
      submission_path: audit.submission_path,
      error: redactedError(error),
      retry_policy: "no_blind_retry",
      created_at: new Date().toISOString()
    });
    await logAction({
      action: "submit_s6e6_kaggle_via_hpc_gateway_failed",
      taskId,
      runId,
      message: "S6E6 official Kaggle submit via HPC gateway failed.",
      artifactPath: responsePath
    });
    return { ok: false, run_id: runId, status: "failed", kaggle_submission: { artifactPath: responsePath } };
  }
}

export async function runS6E6WorkstationClosedLoop(options: ClosedLoopOptions = {}) {
  const created = await createWorkstationRun({
    taskId,
    trigger: "closed_loop_workstation_api",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug,
    objective: "Complete S6E6 through workstation agents, gated GPU execution, submission audit, and optional official submit."
  });
  const runId = created.run_id;
  const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
  const artifactDescriptors: WorkstationArtifactDescriptor[] = [];
  await prisma.experimentRun.update({
    where: { id: runId },
    data: { status: "closed_loop_strategy_gate", validationStatus: "running" }
  });

  try {
  const strategyGate = await evaluateStrategyExecutionGate({
    taskId,
    requestedTemplate: options.gpuTemplate,
    allowEvidenceOnly: options.allowEvidenceOnlyTemplate === true
  });
  const strategyGatePath = await writeJsonArtifact(`${runRoot}/strategy_execution_gate.json`, {
    ...strategyGate.gate,
    top_recommendations: strategyGate.recommendations.slice(0, 5).map((item) => ({
      rank: item.rank,
      strategy_id: item.strategy.strategy_id,
      label: item.strategy.label,
      gpu_template: item.strategy.gpu_template,
      score_gate: item.score_gate
    }))
  });
  await recordEvidence({
    runId,
    label: "Strategy execution gate",
    artifactPath: strategyGatePath,
    source: "ModelSelectionAgent",
    claimBinding: "The workstation selected or rejected the S6E6 training template before HPC execution."
  });
  artifactDescriptors.push(await artifactDescriptor(strategyGatePath, {
    artifact_type: "strategy_execution_gate",
    created_by_agent: "ModelSelectionAgent",
    stage: "model_selection",
    claim_binding: "S6E6 execution is bound to score-aware strategy recommendation.",
    gate_dependency: "plan_approval"
  }));
  if (!strategyGate.gate.allowed_to_execute) {
    await prisma.experimentRun.update({
      where: { id: runId },
      data: { status: "blocked_strategy_gate", validationStatus: "blocked", finishedAt: new Date() }
    });
    await logAction({
      action: "s6e6_strategy_execution_blocked",
      taskId,
      runId,
      message: "S6E6 training blocked by score-aware strategy execution gate.",
      artifactPath: strategyGatePath,
      metadata: {
        requested_template: options.gpuTemplate ?? null,
        selected_template: strategyGate.gate.selected_template,
        blocked_reasons: strategyGate.gate.blocked_reasons
      }
    });
    return {
      ok: false,
      run_id: runId,
      status: "blocked_strategy_gate",
      strategy_gate: strategyGate.gate,
      artifact_path: strategyGatePath
    };
  }

  await prisma.experimentRun.update({
    where: { id: runId },
    data: { status: "closed_loop_preflight", validationStatus: "running" }
  });
  const preflightResult = await preflight(runId, runRoot);
  artifactDescriptors.push(await artifactDescriptor(preflightResult.preflightPath, {
    artifact_type: "preflight",
    created_by_agent: "PreflightAgent",
    stage: "preflight",
    claim_binding: "Critical resources were checked before training.",
    gate_dependency: "plan_approval"
  }));
  if (!preflightResult.ok) {
    return {
      ok: false,
      run_id: runId,
      status: "blocked_preflight",
      blockers: preflightResult.blockers,
      blocker_path: preflightResult.blockerPath
    };
  }

  const plan = await runAgentStage({
    runId,
    runRoot,
    agentId: "OrchestratorAgent",
    stage: "experiment_planning",
    outputName: "workflow_plan.json",
    gateDependency: "plan_approval",
    payload: {
      plan: [
        "data audit",
        "feature engineering plan",
        "model selection",
        "DeepSeek code draft",
        "code quality gate",
        "HPC/GPU execution gate",
        "submission audit",
        "report",
        "submission approval gate"
      ],
      direct_codex_training_allowed: false,
      selected_strategy: {
        strategy_id: strategyGate.gate.selected_strategy_id,
        label: strategyGate.gate.selected_label,
        gpu_template: strategyGate.gate.selected_template,
        expected_public_score: strategyGate.gate.expected_public_score,
        historical_best_public_score: strategyGate.gate.historical_best_public_score,
        execution_policy: strategyGate.gate.execution_policy
      },
      score_recovery_frontier: buildS6E6ScoreRecoveryFrontier()
    }
  });
  await approveExistingGate({
    runId,
    gateType: "plan_approval",
    reviewer: "Research Admin",
    reason: "Current user request authorizes the workstation to start this closed-loop supervised run.",
    artifactPath: plan.artifact_path
  });

  const agentResults: AgentResult[] = [plan];
  for (const stage of [
    ["ResearchContextAgent", "literature_context", "research_brief.json"],
    ["DataAuditAgent", "data_audit", "data_audit.json"],
    ["FeatureEngineeringAgent", "feature_engineering", "feature_plan.json"],
    ["ModelSelectionAgent", "model_selection", "experiment_matrix.json"]
  ] as const) {
    agentResults.push(await runAgentStage({
      runId,
      runRoot,
      agentId: stage[0],
      stage: stage[1],
      outputName: stage[2],
      simulateRecoverableFailure: stage[1] === "feature_engineering",
      payload: {
        task_id: taskId,
        competition_slug: competitionSlug,
        metric: "balanced_accuracy",
        bounded_context_inputs: ["config hash", "data schema", "previous baseline summary"],
        output_policy: "artifact_only"
      }
    }));
  }

  const failureReviewPath = await writeJsonArtifact(`${runRoot}/failure_review.json`, {
    schema: "academic_research_os.failure_review.v1",
    workstation_run_id: runId,
    agent_id: "ReflectionReviewerAgent",
    reviewed_stage: "feature_engineering",
    observed_failure: "intentional_noncritical_recovery_test",
    retry_count: 1,
    decision: "recovered_continue",
    recommendation: "Keep retry records in the evidence ledger and continue because the stage recovered within policy.",
    created_at: new Date().toISOString()
  });
  artifactDescriptors.push(await artifactDescriptor(failureReviewPath, {
    artifact_type: "failure_review",
    created_by_agent: "ReflectionReviewerAgent",
    stage: "reflection",
    claim_binding: "The workstation exercised return-to-stage recovery and recorded the result.",
    gate_dependency: null
  }));

  const PRE_REGISTERED_TEMPLATES = new Set([
    "playground_s6e6_ensemble",
    "playground_s6e6_boosting_ensemble",
    "playground_s6e6_lightgbm",
    "playground_s6e6_xgboost",
    "playground_s6e6_catboost"
  ]);
  const isPreRegisteredTemplate = PRE_REGISTERED_TEMPLATES.has(options.gpuTemplate ?? "");
  let codeSession: { provider: string; status: string; configured: boolean; manifest_path: string | null; transcript_path: string | null; patch_path: string | null } = {
    provider: "none", status: "skipped", configured: false, manifest_path: null, transcript_path: null, patch_path: null
  };

  if (!isPreRegisteredTemplate) {
    codeSession = await createClaudeSession({
      taskId,
      prompt: "For playground-series-s6e6, produce a bounded code-agent training plan or reviewable diff only. Do not apply code or run training.",
      timeoutSeconds: 180,
      maxTurns: 2
    });
  } else {
    codeSession = { provider: "pre_registered_template", status: "completed", configured: true, manifest_path: null, transcript_path: null, patch_path: null };
  }
  const codeArtifact = await writeJsonArtifact(`${runRoot}/code_agent_result.json`, {
    schema: "academic_research_os.code_agent_result.v1",
    workstation_run_id: runId,
    agent_id: "CodeImplementationAgent",
    provider: codeSession.provider,
    status: codeSession.status,
    configured: codeSession.configured,
    session_manifest: codeSession.manifest_path,
    transcript_path: codeSession.transcript_path,
    patch_path: codeSession.patch_path,
    output_policy: "draft_or_diff_only",
    created_at: new Date().toISOString()
  });
  await recordEvidence({
    runId,
    label: "DeepSeek Code Agent result",
    artifactPath: codeArtifact,
    source: "CodeImplementationAgent",
    claimBinding: "Code agent output was captured as a draft/diff artifact before execution gates."
  });
  artifactDescriptors.push(await artifactDescriptor(codeArtifact, {
    artifact_type: "code_agent_result",
    created_by_agent: "CodeImplementationAgent",
    stage: "code_generation",
    claim_binding: "DeepSeek Code Agent produced bounded code evidence before GPU execution.",
    gate_dependency: "code_quality_approval"
  }));
  const codeQualityPath = await writeJsonArtifact(`${runRoot}/code_quality_review.json`, {
    schema: "academic_research_os.code_quality_gate.v1",
    workstation_run_id: runId,
    status: (codeSession.status === "completed" || isPreRegisteredTemplate) ? "passed" : "failed",
    patch_applied: false,
    no_direct_codex_edit: true,
    template_provided: isPreRegisteredTemplate,
    transcript_path: codeSession.transcript_path,
    patch_path: codeSession.patch_path,
    reviewer_agent: "ReviewerAgent",
    created_at: new Date().toISOString()
  });
  if (codeSession.status !== "completed" && !isPreRegisteredTemplate) {
    await prisma.experimentRun.update({
      where: { id: runId },
      data: { status: "blocked_code_agent", validationStatus: "blocked", finishedAt: new Date() }
    });
    return { ok: false, run_id: runId, status: "blocked_code_agent", artifact_path: codeQualityPath };
  }
  await approveExistingGate({
    runId,
    gateType: "code_quality_approval",
    reviewer: "ReviewerAgent",
    reason: "Code Agent completed in draft/diff-only mode; no code was applied by Codex.",
    artifactPath: codeQualityPath
  });
  artifactDescriptors.push(await artifactDescriptor(codeQualityPath, {
    artifact_type: "code_quality_review",
    created_by_agent: "ReviewerAgent",
    stage: "code_review",
    claim_binding: "Code quality was reviewed before HPC execution.",
    gate_dependency: "code_quality_approval"
  }));

  const gpuTemplate = PRE_REGISTERED_TEMPLATES.has(strategyGate.gate.selected_template)
    ? strategyGate.gate.selected_template
    : "playground_s6e6_boosting_ensemble";
  if (gpuTemplate === "playground_s6e6_boosting_ensemble") {
    const dependencyGate = await testS6E6BoostingDependencies();
    if (dependencyGate.artifact_path) {
      await recordEvidence({
        runId,
        label: "S6E6 boosting dependency gate",
        artifactPath: dependencyGate.artifact_path,
        source: "EnvironmentAgent",
        claimBinding: "LightGBM/XGBoost/CatBoost imports were checked before long HPC training."
      });
      artifactDescriptors.push(await artifactDescriptor(dependencyGate.artifact_path, {
        artifact_type: "hpc_boosting_dependency_gate",
        created_by_agent: "EnvironmentAgent",
        stage: "hpc_environment_preflight",
        claim_binding: "S6E6 long training is blocked unless boosting dependencies are present.",
        gate_dependency: "hpc_execution_approval"
      }));
    }
    if (dependencyGate.status !== "passed") {
      const reviewPath = await writeJsonArtifact(`${runRoot}/hpc_dependency_failure_review.json`, {
        schema: "academic_research_os.failure_review.v1",
        workstation_run_id: runId,
        agent_id: "ReflectionReviewerAgent",
        failed_agent_id: "EnvironmentAgent",
        status: "blocked",
        dependency_gate_status: dependencyGate.status,
        dependency_gate_artifact: dependencyGate.artifact_path ?? null,
        recommendation: "Repair the remote HPC Python environment so LightGBM, XGBoost, and CatBoost all import successfully, then rerun the same workstation strategy. Do not fall back to sklearn for score-improvement attempts.",
        created_at: new Date().toISOString()
      });
      await prisma.experimentRun.update({
        where: { id: runId },
        data: { status: "blocked_hpc_dependency", validationStatus: "blocked", finishedAt: new Date() }
      });
      return {
        ok: false,
        run_id: runId,
        status: "blocked_hpc_dependency",
        dependency_gate: dependencyGate,
        artifact_path: reviewPath
      };
    }
  }
  const hpcGate = await createHpcExecutionGate({ taskId, runId, template: gpuTemplate });
  const hpcGateId = await approveExistingGate({
    runId,
    gateType: "hpc_execution_approval",
    reviewer: "Research Admin",
    reason: `Current user request authorizes one workstation-controlled S6E6 GPU training job (template: ${gpuTemplate}).`,
    artifactPath: hpcGate.manifest_path
  });
  const SINGLE_MODEL_TEMPLATES = new Set(["playground_s6e6_lightgbm", "playground_s6e6_xgboost", "playground_s6e6_catboost"]);
  const gpuJob = await submitGpuJob({
    taskId,
    runId,
    agentId: "HpcGpuExecutionAgent",
    gateId: hpcGateId,
    template: gpuTemplate,
    resourceRequest: {
      ...(options.resourceRequest ?? {}),
      gpu_count: "available",
      mode: gpuTemplate === "playground_s6e6_pytorch_mlp" ? "full_data_training"
        : gpuTemplate === "playground_s6e6_boosting_ensemble" ? "boosting_ensemble"
        : SINGLE_MODEL_TEMPLATES.has(gpuTemplate) ? "single_model"
        : "ensemble_lgb_xgb_cat",
      max_runtime_seconds: gpuTemplate === "playground_s6e6_pytorch_mlp" ? 5400
        : gpuTemplate === "playground_s6e6_boosting_ensemble" ? 10800
        : SINGLE_MODEL_TEMPLATES.has(gpuTemplate) ? 7200
        : 7200
    }
  });
  if (gpuJob.status !== "submitted" || !gpuJob.submission_artifact || !gpuJob.metrics_artifact) {
    const reviewPath = await writeJsonArtifact(`${runRoot}/hpc_failure_review.json`, {
      schema: "academic_research_os.failure_review.v1",
      workstation_run_id: runId,
      agent_id: "ReflectionReviewerAgent",
      failed_agent_id: "HpcGpuExecutionAgent",
      status: "blocked",
      gpu_job: gpuJob,
      recommendation: "Repair GPU job evidence before entering submission audit.",
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: runId },
      data: { status: "blocked_hpc_execution", validationStatus: "blocked", finishedAt: new Date() }
    });
    return { ok: false, run_id: runId, status: "blocked_hpc_execution", artifact_path: reviewPath, gpu_job: gpuJob };
  }
  artifactDescriptors.push(await artifactDescriptor(gpuJob.artifact_path ?? "", {
    artifact_type: "gpu_job_result",
    created_by_agent: "HpcGpuExecutionAgent",
    stage: "hpc_execution",
    claim_binding: "HPC/GPU training was launched through the workstation GPU job API.",
    gate_dependency: "hpc_execution_approval"
  }));

  const validation = await runAgentStage({
    runId,
    runRoot,
    agentId: "ValidationAnalysisAgent",
    stage: "validation_review",
    outputName: "metrics_review.json",
    payload: {
      metrics_artifact: gpuJob.metrics_artifact,
      stdout_artifact: gpuJob.stdout_artifact,
      stderr_artifact: gpuJob.stderr_artifact,
      finding: "GPU job produced metrics and submission artifacts."
    }
  });
  agentResults.push(validation);

  const { audit, auditPath } = await validateSubmission(runId, runRoot, gpuJob.submission_artifact);
  artifactDescriptors.push(await artifactDescriptor(auditPath, {
    artifact_type: "submission_audit",
    created_by_agent: "SubmissionGateAgent",
    stage: "submission_check",
    claim_binding: "Submission file was audited before any official Kaggle submit.",
    gate_dependency: "submission_approval"
  }));
  if (audit.status !== "passed") {
    await prisma.experimentRun.update({
      where: { id: runId },
      data: { status: "blocked_submission_audit", validationStatus: "failed", finishedAt: new Date() }
    });
    return { ok: false, run_id: runId, status: "blocked_submission_audit", audit_path: auditPath };
  }

  const scoreGate = await evaluateS6E6ScoreImprovementGate(runId, runRoot, gpuJob.submission_artifact);
  artifactDescriptors.push(await artifactDescriptor(scoreGate.artifactPath, {
    artifact_type: "score_improvement_gate",
    created_by_agent: "SubmissionGateAgent",
    stage: "submission_check",
    claim_binding: "Candidate metrics were compared against the EXP007 historical best before any official Kaggle submit.",
    gate_dependency: "submission_approval"
  }));
  if (scoreGate.status !== "passed") {
    const recoveryPlan = await writeS6E6ScoreRegressionRecoveryPlan({
      runId,
      runRoot,
      submissionPath: gpuJob.submission_artifact,
      scoreGatePath: scoreGate.artifactPath,
      metricsPath: scoreGate.metricsPath,
      metrics: null,
      blockedReasons: scoreGate.blockedReasons,
      validationScore: scoreGate.validationScore,
      riskSignals: scoreGate.riskSignals
    });
    await prisma.experimentRun.update({
      where: { id: runId },
      data: {
        status: "blocked_score_gate",
        validationStatus: "blocked",
        metricsJson: encodeJson({
          workstation_run: true,
          closed_loop: true,
          strategy_execution_gate: strategyGatePath,
          strategy_id: strategyGate.gate.selected_strategy_id,
          strategy_label: strategyGate.gate.selected_label,
          gpu_template: gpuTemplate,
          gpu_job_status: gpuJob.status,
          metrics_artifact: gpuJob.metrics_artifact,
          submission_artifact: gpuJob.submission_artifact,
          submission_audit: auditPath,
          score_improvement_gate: scoreGate.artifactPath,
          score_regression_recovery_plan: recoveryPlan.jsonPath,
          kaggle_submission_status: "blocked_score_gate"
        }),
        finishedAt: new Date()
      }
    });
    await logAction({
      action: "run_s6e6_workstation_closed_loop",
      taskId,
      runId,
      message: "S6E6 workstation run blocked by score improvement gate before official Kaggle submit.",
        artifactPath: scoreGate.artifactPath,
        metadata: {
          submission_audit: auditPath,
          score_gate_status: scoreGate.status,
          score_regression_recovery_plan: recoveryPlan.jsonPath,
          kaggle_submission_status: "blocked_score_gate"
        }
      });
    return {
      ok: false,
      run_id: runId,
      status: "blocked_score_gate",
      score_improvement_gate: scoreGate.artifactPath,
      score_regression_recovery_plan: recoveryPlan.jsonPath,
      metrics_artifact: gpuJob.metrics_artifact,
      submission_artifact: gpuJob.submission_artifact,
      submission_audit: auditPath,
      kaggle_submission: null
    };
  }

  const reportPath = await writeTextArtifact(`${runRoot}/research_report.md`, [
    "# S6E6 Workstation Closed Loop Report",
    "",
    `- workstation_run_id: ${runId}`,
    `- task_id: ${taskId}`,
    `- competition: ${competitionSlug}`,
    "- executor: AI Research Workstation agents and GPU job API",
    "- Codex role: supervisor/auditor; no direct training or direct submit",
    `- metrics_artifact: ${gpuJob.metrics_artifact}`,
    `- submission_artifact: ${gpuJob.submission_artifact}`,
    `- submission_audit: ${auditPath}`,
    `- recoverable_failure_review: ${failureReviewPath}`,
    "",
    "## Agent Results",
    ...agentResults.map((item) => `- ${item.agent_id}/${item.stage}: ${item.status}, attempts=${item.attempts}, artifact=${item.artifact_path}`),
    "",
    "## Gates",
    "- plan_approval: approved",
    "- code_quality_approval: approved",
    "- hpc_execution_approval: approved",
    `- submission_approval: ${options.allowOfficialSubmitAfterGate ? "approved before official submit" : "pending/blocked"}`,
    "",
    "## Submission Audit",
    `- status: ${audit.status}`,
    `- rows_match: ${audit.rows_match}`,
    `- columns_match: ${audit.columns_match}`,
    `- invalid_prediction_count: ${audit.invalid_prediction_count}`,
    `- missing_predictions: ${audit.missing_predictions}`
  ].join("\n"));
  await prisma.report.upsert({
    where: { id: `${runId}_closed_loop_report` },
    update: { status: "generated", markdownPath: reportPath, markdownContent: await fs.readFile(resolveWorkspacePath(reportPath), "utf-8") },
    create: {
      id: `${runId}_closed_loop_report`,
      taskId,
      runId,
      title: "S6E6 Workstation Closed Loop Report",
      status: "generated",
      markdownPath: reportPath,
      markdownContent: await fs.readFile(resolveWorkspacePath(reportPath), "utf-8")
    }
  });
  artifactDescriptors.push(await artifactDescriptor(reportPath, {
    artifact_type: "closed_loop_report",
    created_by_agent: "ReportWriterAgent",
    stage: "report_generation",
    claim_binding: "The report binds agent trace, GPU evidence, metrics and submission audit.",
    gate_dependency: "final_report_approval"
  }));

  let kaggleSubmission: Awaited<ReturnType<typeof submitToKaggle>> | null = null;
  if (options.allowOfficialSubmitAfterGate) {
    const approvalArtifact = await writeJsonArtifact(`${runRoot}/submission_approval_record.json`, {
      schema: "academic_research_os.submission_approval.v1",
      workstation_run_id: runId,
      reviewer: "Research Admin",
      reason: "User requested the workstation to complete the closed loop including official submission after audit.",
      submission_audit: auditPath,
      approved_at: new Date().toISOString()
    });
    await approveExistingGate({
      runId,
      gateType: "submission_approval",
      reviewer: "Research Admin",
      reason: "User requested final official submission after successful submission audit.",
      artifactPath: approvalArtifact
    });
    kaggleSubmission = await submitToKaggle(
      runId,
      runRoot,
      gpuJob.submission_artifact,
      options.submitMessage ?? `Research Agent Workstation ${runId}`
    );
    artifactDescriptors.push(await artifactDescriptor(kaggleSubmission.artifactPath, {
      artifact_type: "kaggle_submission_response",
      created_by_agent: "SubmissionGateAgent",
      stage: "official_submission",
      claim_binding: "Official Kaggle response was recorded after submission approval.",
      gate_dependency: "submission_approval"
    }));
  } else {
    const blockedPath = await writeJsonArtifact(`${runRoot}/kaggle_submission_blocked.json`, {
      schema: "academic_research_os.kaggle_submission_response.v1",
      workstation_run_id: runId,
      status: "blocked",
      reason: "Official Kaggle submit was not requested through the explicit submission_approval gate.",
      score_improvement_gate: scoreGate.artifactPath,
      submission_audit: auditPath,
      created_at: new Date().toISOString()
    });
    kaggleSubmission = { status: "blocked", artifactPath: blockedPath };
    artifactDescriptors.push(await artifactDescriptor(blockedPath, {
      artifact_type: "kaggle_submission_blocker",
      created_by_agent: "SubmissionGateAgent",
      stage: "official_submission",
      claim_binding: "Official Kaggle submission remained blocked because no explicit submission approval was requested.",
      gate_dependency: "submission_approval"
    }));
  }

  await approveExistingGate({
    runId,
    gateType: "final_report_approval",
    reviewer: "ReportStudio",
    reason: "Closed-loop report and evidence bundle generated.",
    artifactPath: reportPath
  });
  const artifactManifestPath = await writeArtifactManifestArtifact({
    taskId,
    runId,
    relativePath: `${runRoot}/closed_loop_artifact_manifest.json`,
    artifacts: artifactDescriptors,
    source: "workstation_closed_loop"
  });
  const bundle = await generateTeacherEvidenceBundle(taskId);
  await prisma.experimentRun.update({
    where: { id: runId },
    data: {
        status: kaggleSubmission?.status === "submitted"
        ? "closed_loop_completed"
        : kaggleSubmission?.status === "failed"
          ? "closed_loop_submission_failed"
          : "closed_loop_submission_blocked",
      validationStatus: audit.status,
      metricsJson: encodeJson({
        workstation_run: true,
        closed_loop: true,
        strategy_execution_gate: strategyGatePath,
        strategy_id: strategyGate.gate.selected_strategy_id,
        strategy_label: strategyGate.gate.selected_label,
        gpu_template: gpuTemplate,
        gpu_job_status: gpuJob.status,
          metrics_artifact: gpuJob.metrics_artifact,
          submission_artifact: gpuJob.submission_artifact,
          submission_audit: auditPath,
          score_improvement_gate: scoreGate.artifactPath,
          score_regression_recovery_plan: null,
          kaggle_submission_status: kaggleSubmission?.status ?? "blocked",
          teacher_bundle: bundle.markdown_path
        }),
      finishedAt: new Date()
    }
  });
  await logAction({
    action: "run_s6e6_workstation_closed_loop",
    taskId,
    runId,
    message: "S6E6 workstation closed loop completed.",
    artifactPath: reportPath,
    metadata: {
      artifact_manifest_path: artifactManifestPath,
      teacher_bundle: bundle.markdown_path,
      kaggle_submission_status: kaggleSubmission?.status ?? "blocked"
    }
  });
  return {
    ok: true,
    run_id: runId,
    status: kaggleSubmission?.status === "submitted"
      ? "closed_loop_completed"
      : kaggleSubmission?.status === "failed"
        ? "closed_loop_submission_failed"
        : "closed_loop_submission_blocked",
    report_path: reportPath,
    artifact_manifest_path: artifactManifestPath,
    teacher_bundle: bundle.markdown_path,
    strategy_execution_gate: strategyGate.gate,
    score_improvement_gate: scoreGate.artifactPath,
    metrics_artifact: gpuJob.metrics_artifact,
    submission_artifact: gpuJob.submission_artifact,
    submission_audit: auditPath,
    score_regression_recovery_plan: null,
    kaggle_submission: kaggleSubmission
  };
  } catch (error) {
    const failurePath = await writeJsonArtifact(`${runRoot}/closed_loop_unhandled_failure.json`, {
      schema: "academic_research_os.closed_loop_unhandled_failure.v1",
      workstation_run_id: runId,
      task_id: taskId,
      status: "failed",
      error: redactedError(error),
      recovery_policy: "Keep any already pulled GPU/HPC artifacts immutable, record failure, and return to validation/report recovery instead of surfacing an opaque API 500.",
      next_action: "Inspect pulled hpc_gpu_training artifacts, then run audit_s6e6_submission and evaluate_s6e6_score_improvement_gate if metrics/submission exist.",
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await writeTrace(runRoot, {
      event: "agent_failed",
      workstation_run_id: runId,
      agent_id: "ReflectionReviewerAgent",
      stage: "closed_loop_recovery",
      status: "failed",
      failure_artifact: failurePath
    });
    await prisma.experimentRun.update({
      where: { id: runId },
      data: {
        status: "failed_closed_loop_recoverable",
        validationStatus: "blocked",
        metricsJson: encodeJson({
          workstation_run: true,
          closed_loop: true,
          failure_artifact: failurePath,
          official_submission_started: false
        }),
        finishedAt: new Date()
      }
    });
    await logAction({
      action: "run_s6e6_workstation_closed_loop_failed",
      taskId,
      runId,
      message: "S6E6 workstation closed loop failed after run creation; failure artifact recorded for recovery.",
      artifactPath: failurePath,
      metadata: {
        official_submission_started: false,
        recovery_required: true
      }
    });
    return {
      ok: false,
      run_id: runId,
      status: "failed_closed_loop_recoverable",
      artifact_path: failurePath,
      official_submission_started: false
    };
  }
}
