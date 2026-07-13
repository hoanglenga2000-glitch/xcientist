/**
 * Agent Strategy Registry
 *
 * Auto-recommends training templates based on task type, data scale, metric,
 * and historical experiment performance. Integrates with experiment memory
 * and existing GPU whitelist templates.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
import { resolveWorkspacePath } from "@/lib/server/paths";

// ── Strategy types ──────────────────────────────────────────────────────────

export type StrategyCategory =
  | "single_model_baseline"
  | "tree_ensemble"
  | "boosting_ensemble"
  | "deep_learning"
  | "stacking"
  | "blending"
  | "calibration";

export type StrategyRisk = "low" | "medium" | "high";

export interface StrategyTemplate {
  strategy_id: string;
  category: StrategyCategory;
  label: string;
  description: string;
  gpu_template: string;
  models: string[];
  n_folds: number;
  n_seeds: number;
  cv_strategy: "stratified_kfold" | "kfold" | "group_kfold";
  expected_runtime_seconds: number;
  risk: StrategyRisk;
  min_data_rows: number;
  max_data_rows: number;
  supported_task_types: string[];
  supported_metrics: string[];
  requires_gpu: boolean;
  requires_hpc: boolean;
  ensemble_method?: "weighted_blend" | "logistic_stack" | "rank_decision" | "none";
  known_benchmarks: {
    task_id: string;
    public_score: number;
    validation_score: number;
    rank: number | null;
    date: string;
  }[];
}

export interface StrategyRecommendation {
  strategy: StrategyTemplate;
  rank: number;
  score: number;
  reasoning: string;
  evidence_refs: string[];
  score_gate: {
    expected_public_score: number | null;
    historical_best_public_score: number | null;
    improves_known_best: boolean | null;
    official_submit_policy: "candidate" | "evidence_only" | "needs_validation";
  };
}

export interface StrategyExecutionGate {
  schema: "academic_research_os.strategy_execution_gate.v1";
  task_id: string;
  requested_template: string | null;
  selected_template: string;
  selected_strategy_id: string;
  selected_label: string;
  expected_public_score: number | null;
  historical_best_public_score: number | null;
  improves_known_best: boolean | null;
  official_submit_policy: "candidate" | "evidence_only" | "needs_validation";
  execution_policy: "score_improvement_candidate" | "evidence_only" | "needs_validation";
  allowed_to_execute: boolean;
  allowed_to_official_submit: boolean;
  blocked_reasons: string[];
  recommendation_rank: number | null;
  created_at: string;
}

export interface TaskProfile {
  task_id: string;
  task_type: string;
  target: string | null;
  metric: string;
  n_rows: number;
  n_features: number;
  n_classes: number | null;
  class_distribution: Record<string, number> | null;
  has_categorical: boolean;
  has_missing: boolean;
}

type PublicScoreFeedback = {
  run_id: string;
  public_score: number;
  kaggle_ref: string | null;
  gpu_template: string | null;
  runner: string | null;
  candidate: string | null;
  response_path: string;
  score_gap_path: string | null;
  status: "beats_historical_best" | "below_historical_best" | "unknown";
  delta_vs_historical_best: number | null;
  note: string | null;
};

// ── Strategy catalog ────────────────────────────────────────────────────────

const STRATEGY_CATALOG: StrategyTemplate[] = [
  {
    strategy_id: "sklearn_mlp_baseline",
    category: "deep_learning",
    label: "PyTorch MLP Baseline",
    description: "Single PyTorch MLP with BN+Dropout, class-balanced loss, 18 epochs on GPU",
    gpu_template: "playground_s6e6_pytorch_mlp",
    models: ["pytorch_mlp"],
    n_folds: 1,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 5400,
    risk: "low",
    min_data_rows: 1000,
    max_data_rows: 1000000,
    supported_task_types: ["kaggle_classification", "tabular_classification"],
    supported_metrics: ["balanced_accuracy", "accuracy", "macro_f1"],
    requires_gpu: true,
    requires_hpc: true,
    known_benchmarks: [
      {
        task_id: "playground_series_s6e6",
        public_score: 0.95295,
        validation_score: 0.94889,
        rank: 547,
        date: "2026-06-15",
      },
    ],
  },
  {
    strategy_id: "sklearn_rf_hgb_et_ensemble",
    category: "tree_ensemble",
    label: "Sklearn RF+HGB+ET Ensemble (5fold×3seed)",
    description:
      "RandomForest + HistGradientBoosting + ExtraTrees with logistic stacker and grid-searched blend weights. Pure sklearn, guaranteed HPC-compatible.",
    gpu_template: "playground_s6e6_ensemble",
    models: ["RandomForest", "HistGradientBoosting", "ExtraTrees", "LogisticRegression"],
    n_folds: 5,
    n_seeds: 3,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 7200,
    risk: "medium",
    min_data_rows: 1000,
    max_data_rows: 2000000,
    supported_task_types: ["kaggle_classification", "tabular_classification"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "logistic_stack",
    known_benchmarks: [],
  },
  {
    strategy_id: "lgb_xgb_cat_blend",
    category: "boosting_ensemble",
    label: "LightGBM + XGBoost + CatBoost Blend (EXP007-style)",
    description:
      "Individual LGB/XGB/CAT 5fold CV, then OOF-guided weighted/calibrated blend with grid search. EXP017 workstation submission is the current public best at 0.96731; EXP007 remains rollback baseline at 0.96659.",
    gpu_template: "playground_s6e6_boosting_ensemble",
    models: ["LightGBM", "XGBoost", "CatBoost"],
    n_folds: 5,
    n_seeds: 3,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 10800,
    risk: "medium",
    min_data_rows: 1000,
    max_data_rows: 5000000,
    supported_task_types: ["kaggle_classification", "tabular_classification", "tabular_regression"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss", "rmse", "rmsle"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "weighted_blend",
    known_benchmarks: [
      {
        task_id: "playground_series_s6e6",
        public_score: 0.96731,
        validation_score: 0.9665172829535322,
        rank: 500,
        date: "2026-06-18",
      },
    ],
  },
  {
    strategy_id: "catboost_single",
    category: "boosting_ensemble",
    label: "CatBoost Single Model (5fold)",
    description: "Single CatBoost with 5fold CV. Fast, strong baseline for mixed-type tabular data.",
    gpu_template: "playground_s6e6_catboost",
    models: ["CatBoost"],
    n_folds: 5,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 3600,
    risk: "low",
    min_data_rows: 500,
    max_data_rows: 5000000,
    supported_task_types: ["kaggle_classification", "tabular_classification", "tabular_regression"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss", "rmse", "rmsle"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "none",
    known_benchmarks: [],
  },
  {
    strategy_id: "lightgbm_single",
    category: "boosting_ensemble",
    label: "LightGBM Single Model (5fold)",
    description: "Single LightGBM with 5fold CV. Fast baseline with Optuna hyperparameter search potential.",
    gpu_template: "playground_s6e6_lightgbm",
    models: ["LightGBM"],
    n_folds: 5,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 2400,
    risk: "low",
    min_data_rows: 500,
    max_data_rows: 10000000,
    supported_task_types: ["kaggle_classification", "tabular_classification", "tabular_regression"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss", "rmse", "rmsle"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "none",
    known_benchmarks: [],
  },
  {
    strategy_id: "lgbm_optuna_exp018",
    category: "boosting_ensemble",
    label: "EXP018 LightGBM Optuna Challenger",
    description:
      "Governed full-data LightGBM Optuna search using the EXP016 scaffold. Produces OOF/test probabilities and trial evidence for model-selection review; it is not directly submittable until a later blend/submission gate promotes it.",
    gpu_template: "playground_s6e6_lgbm_optuna",
    models: ["LightGBM", "Optuna"],
    n_folds: 5,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 28800,
    risk: "medium",
    min_data_rows: 1000,
    max_data_rows: 10000000,
    supported_task_types: ["kaggle_classification", "tabular_classification"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "none",
    known_benchmarks: [],
  },
  {
    strategy_id: "xgboost_single",
    category: "boosting_ensemble",
    label: "XGBoost Single Model (5fold)",
    description: "Single XGBoost with 5fold CV. Strong on structured data with careful preprocessing.",
    gpu_template: "playground_s6e6_xgboost",
    models: ["XGBoost"],
    n_folds: 5,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 3000,
    risk: "low",
    min_data_rows: 500,
    max_data_rows: 10000000,
    supported_task_types: ["kaggle_classification", "tabular_classification", "tabular_regression"],
    supported_metrics: ["balanced_accuracy", "accuracy", "log_loss", "rmse", "rmsle"],
    requires_gpu: false,
    requires_hpc: true,
    ensemble_method: "none",
    known_benchmarks: [],
  },
  {
    strategy_id: "tabular_sklearn_baseline",
    category: "single_model_baseline",
    label: "Sklearn Tabular Baseline (local CPU)",
    description:
      "LogisticRegression + RandomForest + ExtraTrees + GradientBoosting with 5fold CV. Runs locally, no HPC required.",
    gpu_template: "all_tasks_baseline",
    models: ["LogisticRegression", "RandomForest", "ExtraTrees", "GradientBoosting"],
    n_folds: 5,
    n_seeds: 1,
    cv_strategy: "stratified_kfold",
    expected_runtime_seconds: 600,
    risk: "low",
    min_data_rows: 100,
    max_data_rows: 200000,
    supported_task_types: ["tabular_classification", "tabular_regression", "research_task"],
    supported_metrics: ["accuracy", "rmse", "rmsle", "mae", "macro_f1"],
    requires_gpu: false,
    requires_hpc: false,
    ensemble_method: "none",
    known_benchmarks: [],
  },
];

function bestKnownPublicScore(taskId: string): number | null {
  const scores = STRATEGY_CATALOG.flatMap((strategy) =>
    strategy.known_benchmarks
      .filter((benchmark) => benchmark.task_id === taskId)
      .map((benchmark) => benchmark.public_score)
  );
  return scores.length ? Math.max(...scores) : null;
}

function knownPublicScore(strategy: StrategyTemplate, taskId: string): number | null {
  const scores = strategy.known_benchmarks
    .filter((benchmark) => benchmark.task_id === taskId)
    .map((benchmark) => benchmark.public_score);
  return scores.length ? Math.max(...scores) : null;
}

function numeric(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

async function readJsonIfPresent(relativePath: string): Promise<Record<string, unknown> | null> {
  try {
    return JSON.parse(await fs.readFile(resolveWorkspacePath(relativePath), "utf-8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function templateFromRunner(runner: string | null) {
  const normalized = (runner ?? "").toLowerCase();
  if (normalized.includes("sklearn") || normalized.includes("rf_hgb_et")) return "playground_s6e6_ensemble";
  if (normalized.includes("pytorch") || normalized.includes("mlp")) return "playground_s6e6_pytorch_mlp";
  if (normalized.includes("boosting") || normalized.includes("lgb_xgb_cat")) return "playground_s6e6_boosting_ensemble";
  if (normalized.includes("lightgbm")) return "playground_s6e6_lightgbm";
  if (normalized.includes("xgboost")) return "playground_s6e6_xgboost";
  if (normalized.includes("catboost")) return "playground_s6e6_catboost";
  return null;
}

function scoreGapRunner(scoreGap: Record<string, unknown> | null) {
  const comparedRuns = scoreGap?.compared_runs;
  if (!comparedRuns || typeof comparedRuns !== "object" || Array.isArray(comparedRuns)) return null;
  const current = (comparedRuns as Record<string, unknown>).current;
  if (!current || typeof current !== "object" || Array.isArray(current)) return null;
  const runner = (current as Record<string, unknown>).runner;
  return typeof runner === "string" ? runner : null;
}

function inferTemplate(input: {
  metrics: Record<string, unknown> | null;
  response: Record<string, unknown>;
  scoreGap: Record<string, unknown> | null;
}) {
  const explicit = typeof input.metrics?.gpu_template === "string" ? input.metrics.gpu_template : null;
  if (explicit) return explicit;
  const runner = typeof input.metrics?.runner === "string" ? input.metrics.runner : scoreGapRunner(input.scoreGap);
  const byRunner = templateFromRunner(runner);
  if (byRunner) return byRunner;
  const note = `${input.response.description ?? ""} ${input.response.note ?? ""}`.toLowerCase();
  if (note.includes("sklearn") || note.includes("rf+hgb+et")) return "playground_s6e6_ensemble";
  if (note.includes("mlp")) return "playground_s6e6_pytorch_mlp";
  if (note.includes("exp007") || note.includes("lgb") || note.includes("xgb") || note.includes("cat")) return "playground_s6e6_boosting_ensemble";
  return null;
}

async function loadPublicScoreFeedback(taskId: string): Promise<PublicScoreFeedback[]> {
  const runsRoot = resolveWorkspacePath(`workspace/workstation_runs/${taskId}`);
  const entries = await fs.readdir(runsRoot, { withFileTypes: true }).catch(() => []);
  const bestPublic = bestKnownPublicScore(taskId);
  const feedback: PublicScoreFeedback[] = [];
  for (const entry of entries.filter((item) => item.isDirectory())) {
    const runId = entry.name;
    const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
    const responsePath = `${runRoot}/kaggle_submission_response.json`;
    const response = await readJsonIfPresent(responsePath);
    const publicScore = numeric(response?.public_score);
    if (!response || publicScore === null) continue;
    const [runMetrics, hpcMetrics, scoreGap] = await Promise.all([
      readJsonIfPresent(`${runRoot}/metrics.json`),
      readJsonIfPresent(`${runRoot}/hpc_gpu_training/metrics.json`),
      readJsonIfPresent(`${runRoot}/score_gap_analysis.json`)
    ]);
    const metrics = runMetrics ?? hpcMetrics;
    const runner = typeof metrics?.runner === "string" ? metrics.runner : scoreGapRunner(scoreGap);
    const gpuTemplate = inferTemplate({ metrics, response, scoreGap });
    const delta = bestPublic === null ? null : publicScore - bestPublic;
    feedback.push({
      run_id: runId,
      public_score: publicScore,
      kaggle_ref: typeof response.kaggle_ref === "string" ? response.kaggle_ref : null,
      gpu_template: gpuTemplate,
      runner,
      candidate: typeof metrics?.source_experiment_id === "string" ? metrics.source_experiment_id : typeof metrics?.candidate === "string" ? metrics.candidate : null,
      response_path: responsePath,
      score_gap_path: scoreGap ? `${runRoot}/score_gap_analysis.json` : null,
      status: delta === null ? "unknown" : delta >= 0 ? "beats_historical_best" : "below_historical_best",
      delta_vs_historical_best: delta,
      note: typeof response.note === "string" ? response.note : null
    });
  }
  return feedback.sort((a, b) => b.public_score - a.public_score);
}

function officialSubmitPolicy(strategy: StrategyTemplate, profile: TaskProfile | null) {
  if (!profile) return "needs_validation" as const;
  const expected = knownPublicScore(strategy, profile.task_id);
  const best = bestKnownPublicScore(profile.task_id);
  if (expected !== null && best !== null) {
    return expected >= best ? "candidate" as const : "evidence_only" as const;
  }
  if (strategy.strategy_id === "lgb_xgb_cat_blend") return "candidate" as const;
  return "needs_validation" as const;
}

// ── Strategy scoring functions ──────────────────────────────────────────────

function scoreStrategyForTask(strategy: StrategyTemplate, profile: TaskProfile): number {
  let score = 0;

  // Task type match
  if (strategy.supported_task_types.includes(profile.task_type)) score += 30;
  else if (
    strategy.supported_task_types.some((t) => profile.task_type.includes(t))
  )
    score += 15;

  // Metric match
  if (strategy.supported_metrics.includes(profile.metric)) score += 25;
  else if (
    strategy.supported_metrics.some(
      (m) => profile.metric.includes(m) || m.includes(profile.metric)
    )
  )
    score += 10;

  // Data scale match
  if (
    profile.n_rows >= strategy.min_data_rows &&
    profile.n_rows <= strategy.max_data_rows
  )
    score += 20;
  else if (profile.n_rows >= strategy.min_data_rows * 0.5) score += 8;

  // Classification-specific: multi-class bonus for ensemble strategies
  if (
    profile.n_classes !== null &&
    profile.n_classes > 2 &&
    strategy.category !== "single_model_baseline"
  )
    score += 10;

  // Boost for strategies with proven benchmarks on this task.
  // Leaderboard evidence must dominate convenience/risk bonuses; otherwise the
  // low-risk MLP baseline can incorrectly outrank the proven EXP007-style blend.
  const taskBenchmark = strategy.known_benchmarks.find(
    (b) => b.task_id === profile.task_id
  );
  if (taskBenchmark) score += taskBenchmark.public_score * 1000;

  const bestPublic = bestKnownPublicScore(profile.task_id);
  if (taskBenchmark && bestPublic !== null) {
    const publicGap = bestPublic - taskBenchmark.public_score;
    if (publicGap > 0) score -= publicGap * 500;
    else score += 20;
  }

  // Risk penalty for high-risk strategies when a low-risk alternative exists
  if (strategy.risk === "high") score -= 5;
  if (strategy.risk === "low") score += 5;

  return score;
}

function generateReasoning(
  strategy: StrategyTemplate,
  profile: TaskProfile,
  score: number
): string {
  const parts: string[] = [];
  const benchmark = strategy.known_benchmarks.find(
    (b) => b.task_id === profile.task_id
  );

  if (benchmark) {
    parts.push(
      `Known benchmark on ${profile.task_id}: public=${benchmark.public_score}, rank=${benchmark.rank}`
    );
  }
  parts.push(
    `${strategy.models.length} model(s): ${strategy.models.join(", ")}`
  );
  parts.push(
    `${strategy.n_folds}fold × ${strategy.n_seeds}seed CV (${strategy.cv_strategy})`
  );
  parts.push(
    `Expected runtime: ~${Math.round(strategy.expected_runtime_seconds / 60)}min`
  );
  if (strategy.ensemble_method && strategy.ensemble_method !== "none") {
    parts.push(`Ensemble: ${strategy.ensemble_method}`);
  }
  parts.push(
    strategy.requires_hpc ? "Runs on HPC/GPU" : "Runs on local CPU"
  );
  parts.push(`Risk: ${strategy.risk}`);
  return parts.join(" | ");
}

// ── Memory integration ──────────────────────────────────────────────────────

async function loadExperimentMemory(
  taskId: string
): Promise<
  { run_id: string; metrics: Record<string, unknown>; status: string; created_at: string }[]
> {
  const runs = await prisma.experimentRun.findMany({
    where: { taskId },
    orderBy: { createdAt: "desc" },
    take: 50,
  });
  return runs.map((run) => ({
    run_id: run.id,
    metrics: run.metricsJson
      ? (JSON.parse(run.metricsJson) as Record<string, unknown>)
      : {},
    status: run.status,
    created_at: run.createdAt.toISOString(),
  }));
}

function boostFromMemory(
  taskId: string,
  strategies: StrategyTemplate[],
  memory: { run_id: string; metrics: Record<string, unknown>; status: string }[],
  publicFeedback: PublicScoreFeedback[]
): Map<string, number> {
  const boosts = new Map<string, number>();
  const bestPublic = bestKnownPublicScore(taskId);
  for (const entry of memory) {
    const gpuTemplate = entry.metrics?.gpu_template as string | undefined;
    if (!gpuTemplate) continue;
    const score = entry.metrics?.public_score as number | undefined;
    if (score !== undefined && bestPublic !== null && score < bestPublic) {
      const current = boosts.get(gpuTemplate) ?? 0;
      boosts.set(gpuTemplate, current - (bestPublic - score) * 1000);
      continue;
    }
    if (score !== undefined && (bestPublic === null ? score > 0.95 : score >= bestPublic)) {
      const current = boosts.get(gpuTemplate) ?? 0;
      boosts.set(gpuTemplate, current + score * 3);
    }
    if (entry.status === "closed_loop_completed" && score === undefined) {
      const current = boosts.get(gpuTemplate) ?? 0;
      boosts.set(gpuTemplate, current + 2);
    }
  }
  for (const item of publicFeedback) {
    if (!item.gpu_template || bestPublic === null) continue;
    const current = boosts.get(item.gpu_template) ?? 0;
    if (item.public_score < bestPublic) {
      boosts.set(item.gpu_template, current - Math.max(25, (bestPublic - item.public_score) * 5000));
    } else {
      boosts.set(item.gpu_template, current + item.public_score * 10);
    }
  }
  return boosts;
}

// ── Main API ────────────────────────────────────────────────────────────────

export async function buildTaskProfile(taskId: string): Promise<TaskProfile | null> {
  const task = await prisma.task.findUnique({ where: { id: taskId } });
  if (!task) return null;

  // Try to read data profile from latest workstation run's data_audit.json
  const runsDir = resolveWorkspacePath(`workspace/workstation_runs/${taskId}`);
  let dataAuditPath = "";
  try {
    const runDirs = await fs.readdir(runsDir, { withFileTypes: true });
    const sorted = runDirs
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort()
      .reverse();
    for (const runDir of sorted) {
      const candidate = `workspace/workstation_runs/${taskId}/${runDir}/data_audit.json`;
      try {
        await fs.stat(resolveWorkspacePath(candidate));
        dataAuditPath = candidate;
        break;
      } catch { /* continue */ }
    }
  } catch { /* no runs dir */ }
  let nRows = 0;
  let nFeatures = 0;
  let nClasses: number | null = null;
  let classDist: Record<string, number> | null = null;
  const hasCat = true;
  const hasMissing = true;

  try {
    if (!dataAuditPath) throw new Error("No data_audit.json found");
    const auditRaw = await fs.readFile(resolveWorkspacePath(dataAuditPath), "utf-8");
    const audit = JSON.parse(auditRaw) as Record<string, unknown>;
    nRows = (audit.train_rows as number) ?? 0;
    nFeatures = (audit.features_after_encoding as number) ?? (audit.n_features as number) ?? 0;
    nClasses = (audit.n_classes as number) ?? null;
    classDist = (audit.class_distribution as Record<string, number>) ?? null;
    if (!nRows || !nFeatures) {
      throw new Error("Data audit artifact does not include row/feature statistics");
    }
  } catch {
    // Fall back to task DB record and try data files
    try {
      const trainPath = resolveWorkspacePath(`tasks/${taskId}/data/train.csv`);
      const head = await fs.readFile(trainPath, "utf-8");
      const lines = head.split("\n").filter(Boolean);
      nRows = lines.length - 1;
      nFeatures = (lines[0]?.split(",").length ?? 2) - 1;
    } catch {
      nRows = 100000;
      nFeatures = 50;
    }
  }

  return {
    task_id: taskId,
    task_type: task.taskType,
    target: task.target,
    metric: task.metric ?? "balanced_accuracy",
    n_rows: nRows,
    n_features: nFeatures,
    n_classes: nClasses,
    class_distribution: classDist,
    has_categorical: hasCat,
    has_missing: hasMissing,
  };
}

export async function recommendStrategies(
  taskId: string,
  topK: number = 3
): Promise<{
  profile: TaskProfile | null;
  recommendations: StrategyRecommendation[];
  historical_context: string[];
  public_score_feedback: PublicScoreFeedback[];
}> {
  const profile = await buildTaskProfile(taskId);

  const memory = await loadExperimentMemory(taskId);
  const publicScoreFeedback = await loadPublicScoreFeedback(taskId);
  const memoryBoosts = boostFromMemory(taskId, STRATEGY_CATALOG, memory as any, publicScoreFeedback);

  const scored: { strategy: StrategyTemplate; score: number }[] =
    STRATEGY_CATALOG.map((strategy) => {
      let score = profile ? scoreStrategyForTask(strategy, profile) : 0;
      const boost = memoryBoosts.get(strategy.gpu_template) ?? 0;
      score += boost;
      return { strategy, score };
    });

  scored.sort((a, b) => b.score - a.score);

  const recommendations: StrategyRecommendation[] = scored
    .slice(0, topK)
    .map((item, index) => ({
      strategy: item.strategy,
      rank: index + 1,
      score: item.score,
      reasoning: profile
        ? generateReasoning(item.strategy, profile, item.score)
        : "No task profile available; using catalog defaults.",
      evidence_refs: item.strategy.known_benchmarks.map(
        (b) => `benchmark:${b.task_id}:${b.date}:${b.public_score}`
      ).concat(
        publicScoreFeedback
          .filter((feedback) => feedback.gpu_template === item.strategy.gpu_template)
          .map((feedback) => `kaggle_public_feedback:${feedback.run_id}:${feedback.public_score}:${feedback.status}`)
      ),
      score_gate: {
        expected_public_score: profile ? knownPublicScore(item.strategy, profile.task_id) : null,
        historical_best_public_score: profile ? bestKnownPublicScore(profile.task_id) : null,
        improves_known_best: profile && knownPublicScore(item.strategy, profile.task_id) !== null && bestKnownPublicScore(profile.task_id) !== null
          ? (knownPublicScore(item.strategy, profile.task_id) as number) > (bestKnownPublicScore(profile.task_id) as number)
          : null,
        official_submit_policy: officialSubmitPolicy(item.strategy, profile),
      },
    }));

  const historicalContext = memory.slice(0, 10).map((m) => {
    const gpuTemplate = m.metrics?.gpu_template as string | undefined;
    const publicScore = m.metrics?.public_score as number | undefined;
    return `[${m.created_at.slice(0, 10)}] ${m.run_id}: status=${m.status}, template=${gpuTemplate ?? "unknown"}, public_score=${publicScore ?? "N/A"}`;
  });

  const publicFeedbackContext = publicScoreFeedback.slice(0, 10).map((feedback) => {
    const delta = feedback.delta_vs_historical_best === null
      ? "N/A"
      : feedback.delta_vs_historical_best.toFixed(5);
    return `[public-feedback] ${feedback.run_id}: template=${feedback.gpu_template ?? "unknown"}, public_score=${feedback.public_score}, delta_vs_best=${delta}, status=${feedback.status}`;
  });

  return {
    profile,
    recommendations,
    historical_context: historicalContext.concat(publicFeedbackContext),
    public_score_feedback: publicScoreFeedback,
  };
}

export async function evaluateStrategyExecutionGate(input: {
  taskId: string;
  requestedTemplate?: string | null;
  allowEvidenceOnly?: boolean;
}): Promise<{
  gate: StrategyExecutionGate;
  recommendation: StrategyRecommendation | null;
  recommendations: StrategyRecommendation[];
}> {
  const strategies = await recommendStrategies(input.taskId, STRATEGY_CATALOG.length);
  const requestedTemplate = input.requestedTemplate?.trim() || null;
  const recommendation = requestedTemplate
    ? strategies.recommendations.find((item) => item.strategy.gpu_template === requestedTemplate) ?? null
    : strategies.recommendations[0] ?? null;
  const fallback = strategies.recommendations[0] ?? null;
  const selected = recommendation ?? fallback;
  if (!selected) {
    return {
      recommendation: null,
      recommendations: strategies.recommendations,
      gate: {
        schema: "academic_research_os.strategy_execution_gate.v1",
        task_id: input.taskId,
        requested_template: requestedTemplate,
        selected_template: requestedTemplate ?? "unknown",
        selected_strategy_id: "unknown",
        selected_label: "No strategy available",
        expected_public_score: null,
        historical_best_public_score: bestKnownPublicScore(input.taskId),
        improves_known_best: null,
        official_submit_policy: "needs_validation",
        execution_policy: "needs_validation",
        allowed_to_execute: false,
        allowed_to_official_submit: false,
        blocked_reasons: ["No strategy recommendation is available for this task."],
        recommendation_rank: null,
        created_at: new Date().toISOString()
      }
    };
  }
  const blockedReasons: string[] = [];
  const officialPolicy = selected.score_gate.official_submit_policy;
  const latestBadPublicFeedback = strategies.public_score_feedback.find((feedback) =>
    feedback.gpu_template === selected.strategy.gpu_template && feedback.status === "below_historical_best"
  );
  const evidenceOnly = officialPolicy === "evidence_only";
  const needsValidation = officialPolicy === "needs_validation";
  if (requestedTemplate && !recommendation) {
    blockedReasons.push(`Requested template ${requestedTemplate} is not registered in the strategy catalog.`);
  }
  if (evidenceOnly && input.allowEvidenceOnly !== true) {
    blockedReasons.push("Requested strategy is evidence-only because it is below the historical best score.");
  }
  if (needsValidation && input.allowEvidenceOnly !== true) {
    blockedReasons.push("Requested strategy has no score-improvement evidence and needs validation before execution.");
  }
  if (latestBadPublicFeedback && input.allowEvidenceOnly !== true) {
    blockedReasons.push(`Requested strategy has negative Kaggle public-score feedback: run ${latestBadPublicFeedback.run_id} scored ${latestBadPublicFeedback.public_score}, below historical best by ${Math.abs(latestBadPublicFeedback.delta_vs_historical_best ?? 0).toFixed(5)}.`);
  }
  const executionPolicy = officialPolicy === "candidate"
    ? "score_improvement_candidate"
    : officialPolicy === "evidence_only"
      ? "evidence_only"
      : "needs_validation";
  return {
    recommendation: selected,
    recommendations: strategies.recommendations,
    gate: {
      schema: "academic_research_os.strategy_execution_gate.v1",
      task_id: input.taskId,
      requested_template: requestedTemplate,
      selected_template: selected.strategy.gpu_template,
      selected_strategy_id: selected.strategy.strategy_id,
      selected_label: selected.strategy.label,
      expected_public_score: selected.score_gate.expected_public_score,
      historical_best_public_score: selected.score_gate.historical_best_public_score,
      improves_known_best: selected.score_gate.improves_known_best,
      official_submit_policy: officialPolicy,
      execution_policy: executionPolicy,
      allowed_to_execute: blockedReasons.length === 0,
      allowed_to_official_submit: officialPolicy === "candidate",
      blocked_reasons: blockedReasons,
      recommendation_rank: selected.rank,
      created_at: new Date().toISOString()
    }
  };
}

export function getStrategyById(strategyId: string): StrategyTemplate | null {
  return STRATEGY_CATALOG.find((s) => s.strategy_id === strategyId) ?? null;
}

export function getStrategiesByCategory(category: StrategyCategory): StrategyTemplate[] {
  return STRATEGY_CATALOG.filter((s) => s.category === category);
}

export function getAllStrategies(): StrategyTemplate[] {
  return STRATEGY_CATALOG;
}

export function getDefaultStrategyForTask(taskId: string): StrategyTemplate {
  if (taskId === "playground_series_s6e6") {
    // For S6E6, default to LGB+XGB+CAT blend; current public best target is 0.96731+.
    return STRATEGY_CATALOG[2]; // lgb_xgb_cat_blend
  }
  return STRATEGY_CATALOG[STRATEGY_CATALOG.length - 1]; // tabular_sklearn_baseline
}
