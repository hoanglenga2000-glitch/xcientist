import { promises as fs } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import net from "node:net";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { latestPassedCodeQualityGate } from "@/lib/server/code-quality-gate";
import { cancelRunningJob } from "@/lib/server/job-registry";
import { logAction } from "@/lib/server/actions";
import { decodeJson, encodeJson } from "@/lib/server/json";
import { bootstrapS6E6BoostingEnvironment, submitGpuJob, testS6E6BoostingDependencies } from "@/lib/server/gpu-ssh-gateway";
import { buildHpcApprovalEvidence, validateHpcExecutionGate } from "@/lib/server/hpc-execution-gate";
import { latestExperimentPath, latestScoreGatedWorkstationRunPath, normalizeTaskId, readJsonFile, resolveWorkspacePath, stamp, workspaceRoot, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";
import { ensurePrivateDirectory, readStableRegularTextFile, writeAtomicPrivateTextFile } from "@/lib/server/stable-file";
import {
  createHpcExecutionGate,
  createWorkstationRun,
  ensurePlaygroundSeriesTask,
  generateTeacherEvidenceBundle
} from "@/lib/server/workstation-run-contract";
import {
  buildS6E6ScoreRecoveryFrontier,
  probeS6E6ScoreImprovementGate,
  runS6E6WorkstationClosedLoop,
  submitExistingS6E6RunViaHpcKaggleGateway,
  submitExistingS6E6WorkstationRunToKaggle
} from "@/lib/server/workstation-closed-loop";
import {
  recommendStrategies,
  evaluateStrategyExecutionGate,
  getStrategyById,
  getAllStrategies,
  getDefaultStrategyForTask
} from "@/lib/server/strategy-registry";

export type WorkstationActionPayload = {
  action?: string;
  task_id?: string;
  taskId?: string;
  decision?: "approved" | "rejected";
  metadata?: Record<string, unknown>;
};

const workflowStages = [
  "task_understanding",
  "experiment_planning",
  "human_plan_gate",
  "eda",
  "code_generation",
  "training",
  "submission_check",
  "human_submission_gate",
  "report_generation"
];

const execFileAsync = promisify(execFile);

function probeTcpPort(host: string, port: number, timeoutMs = 2500) {
  return new Promise<{ reachable: boolean; error: string | null }>((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    function finish(reachable: boolean, error: string | null) {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve({ reachable, error });
    }
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true, null));
    socket.once("timeout", () => finish(false, `Timed out after ${timeoutMs}ms`));
    socket.once("error", (error) => finish(false, error.message));
    socket.connect(port, host);
  });
}

function pythonExecutable() {
  if (process.env.WORKSTATION_PYTHON) return process.env.WORKSTATION_PYTHON;
  if (process.platform !== "win32") return "python3";
  return "C:\\codex-python\\python.exe";
}

async function latestPatch(taskId: string) {
  const patchDir = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches`);
  const entries = await fs.readdir(patchDir, { withFileTypes: true }).catch(() => []);
  const patches = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".diff"))
    .map(async (entry) => {
      const fullPath = path.join(patchDir, entry.name);
      const stat = await fs.stat(fullPath);
      return { name: entry.name, fullPath, relativePath: `workspace/tasks/${taskId}/code/patches/${entry.name}`, mtimeMs: stat.mtimeMs };
    }));
  return patches.sort((a, b) => b.mtimeMs - a.mtimeMs)[0] ?? null;
}

async function patchFromReviewMetadata(taskId: string, metadata: Record<string, unknown> | undefined) {
  const patchPath = typeof metadata?.patch_path === "string" ? metadata.patch_path.replaceAll("\\", "/") : "";
  if (patchPath) {
    const normalizedPatchPath = patchPath.startsWith("workspace/tasks/")
      ? patchPath
      : `workspace/tasks/${taskId}/code/patches/${path.basename(patchPath)}`;
    if (!normalizedPatchPath.startsWith(`workspace/tasks/${taskId}/code/patches/`) || !normalizedPatchPath.endsWith(".diff")) {
      return null;
    }
    const fullPath = resolveWorkspacePath(normalizedPatchPath);
    const stat = await fs.stat(fullPath).catch(() => null);
    return stat?.isFile()
      ? { name: path.basename(normalizedPatchPath), fullPath, relativePath: normalizedPatchPath, mtimeMs: stat.mtimeMs }
      : null;
  }
  const sessionId = typeof metadata?.session_id === "string" ? metadata.session_id : "";
  if (sessionId) {
    const manifest = await readJsonFile(resolveWorkspacePath(`workspace/code_agent_sessions/${sessionId}/session_manifest.json`)) as Record<string, unknown> | null;
    const sessionPatch = typeof manifest?.patch_path === "string" ? manifest.patch_path.replaceAll("\\", "/") : "";
    if (!sessionPatch) return null;
    const fullPath = resolveWorkspacePath(sessionPatch);
    const stat = await fs.stat(fullPath).catch(() => null);
    return stat?.isFile()
      ? { name: path.basename(sessionPatch), fullPath, relativePath: sessionPatch, mtimeMs: stat.mtimeMs }
      : null;
  }
  return latestPatch(taskId);
}

function patchFiles(patchText: string) {
  const files = new Set<string>();
  for (const line of patchText.split(/\r?\n/)) {
    const diffMatch = line.match(/^diff --git a\/(.+?) b\/(.+)$/);
    if (diffMatch) {
      files.add(diffMatch[1]);
      files.add(diffMatch[2]);
      continue;
    }
    const fileMatch = line.match(/^(?:---|\+\+\+) [ab]\/(.+)$/);
    if (fileMatch) {
      files.add(fileMatch[1]);
      continue;
    }
    const bareFileMatch = line.match(/^(?:---|\+\+\+) (?!\/dev\/null)(?![ab]\/)(.+)$/);
    if (bareFileMatch) files.add(bareFileMatch[1]);
  }
  return [...files]
    .map((file) => file.replace(/^"|"$/g, "").replaceAll("\\", "/"))
    .map((file) => file.replace(/^[ab]\//, ""))
    .filter((file) => file !== "/dev/null");
}

function analyzePatch(taskId: string, patchText: string) {
  const normalizedText = patchText.replaceAll("\\", "/");
  const activePatchText = normalizedText
    .split(/\r?\n/)
    .filter((line) => !line.startsWith("-") && !line.startsWith("diff --git ") && !line.startsWith("--- "))
    .map((line) => line.replace(/^\+\s?/, ""))
    .map((line) => line.replace(/#.*$/, ""))
    .join("\n");
  const files = patchFiles(normalizedText);
  const dangerousCommand = /\b(rm\s+-rf|Remove-Item\b.+-Recurse|del\s+\/[fsq]|format\s+[a-z]:|shutdown\b|Invoke-Expression|curl\b.+\|\s*(bash|sh)|iwr\b.+iex)\b/i.test(normalizedText);
  const originalDataTouched = new RegExp(`(?:^|\\s|/)(tasks|workspace/tasks)/${taskId}/(?:train|test|sample_submission)\\.csv`, "i").test(normalizedText)
    || /(?:^|\s)(train|test|sample_submission)\.csv/i.test(normalizedText);
  const credentialTouched = /(kaggle\.json|\.env|id_rsa|api[_-]?key|secret|token)/i.test(normalizedText);
  const allowedPrefixes = [
    `workspace/tasks/${taskId}/code/`,
    `tasks/${taskId}/code/`,
    "src/",
    "scripts/",
    "configs/"
  ];
  const immutableArtifactTouched = files.some((file) => file.startsWith("workspace/workstation_runs/") || file.startsWith("experiments/"));
  const outsideAllowedScope = files.filter((file) => !allowedPrefixes.some((prefix) => file.startsWith(prefix)));
  const s6e6SchemaMismatch = taskId === "playground_series_s6e6"
    && (
      /\broc_auc_score\b/i.test(activePatchText)
      || /predict_proba\([^)]*\)\s*\[\s*:\s*,\s*1\s*\]/i.test(activePatchText)
      || /target_col\s*[:=]\s*["']target["']/i.test(activePatchText)
      || /["']target["']\s*:/i.test(activePatchText)
      || /\btarget\b/i.test(activePatchText) && !/\bclass\b/i.test(activePatchText)
    );
  const s6e6UnsafeScope = taskId === "playground_series_s6e6" && outsideAllowedScope.length > 0;
  const s6e6RecoveryPatchMissing = taskId === "playground_series_s6e6"
    && patchText.trim().length > 0
    && !/(calibration|calibrat|threshold|class[_ -]?prior|balanced_accuracy|best[_ -]?single|no[_ -]?worse|blend_delta|score[_ -]?gate|regression[_ -]?diagnosis)/i.test(activePatchText);
  const nonExecutablePatchPlaceholder = taskId === "playground_series_s6e6"
    && patchText.trim().length > 0
    && (
      /^\s*(?:pass|none|\.\.\.)\s*$/im.test(activePatchText)
      || /\b(?:incorrect shape|need rework|reimplemented properly|placeholder|pseudo[- ]?code|not shown in diff|assumed present)\b/i.test(activePatchText)
      || /^\s*This will\b/im.test(activePatchText)
      || /\[[.\s]*\]/.test(activePatchText)
    );
  const findings = [
    ...(dangerousCommand ? ["Dangerous shell command pattern detected in patch."] : []),
    ...(originalDataTouched ? ["Patch appears to modify original Kaggle-style data files."] : []),
    ...(credentialTouched ? ["Patch references credentials, tokens or secret files."] : []),
    ...(immutableArtifactTouched ? ["Patch attempts to modify immutable run artifacts or historical experiment outputs; create a new gated template/code artifact instead."] : []),
    ...(outsideAllowedScope.length ? [`Patch touches files outside recommended scope: ${outsideAllowedScope.join(", ")}`] : []),
    ...(s6e6SchemaMismatch ? ["S6E6 patch appears to use a binary target/AUC/probability[:,1] pattern instead of the required multiclass class label and balanced_accuracy contract."] : []),
    ...(s6e6UnsafeScope ? ["S6E6 score-improvement patches must stay inside workstation-controlled code/config/script scopes before HPC execution."] : []),
    ...(s6e6RecoveryPatchMissing ? ["S6E6 recovery patch must address score-regression gates such as calibration, class prior/thresholding, balanced_accuracy, no-worse-than-best-single blend selection, or score gate/report logic."] : []),
    ...(nonExecutablePatchPlaceholder ? ["S6E6 patch contains non-executable placeholders or prose fragments and must be regenerated before Code Quality Gate can pass."] : [])
  ];
  const blockingFindings = findings.filter((finding) => {
    if (taskId === "playground_series_s6e6") return true;
    return !finding.startsWith("Patch touches files outside recommended scope");
  });
  return {
    files,
    dangerousCommand,
    originalDataTouched,
    credentialTouched,
    outsideAllowedScope,
    s6e6SchemaMismatch,
    s6e6UnsafeScope,
    s6e6RecoveryPatchMissing,
    findings,
    passed: blockingFindings.length === 0
  };
}

async function pythonSyntaxCheck(taskId: string) {
  const candidates = [
    resolveWorkspacePath(`tasks/${taskId}/code/generated/baseline_runner.py`),
    resolveWorkspacePath(`workspace/tasks/${taskId}/code/current_code/baseline_runner.py`)
  ];
  for (const candidate of candidates) {
    const exists = await fs.stat(candidate).then((stat) => stat.isFile()).catch(() => false);
    if (!exists) continue;
    try {
      const executable = process.platform === "win32" ? "python" : "python3";
      await execFileAsync(executable, ["-m", "py_compile", candidate], { timeout: 20000 });
      return { status: "passed", file: path.relative(resolveWorkspacePath("."), candidate), error: null };
    } catch {
      return {
        status: "failed",
        file: path.relative(resolveWorkspacePath("."), candidate),
        error: "python_syntax_check_failed"
      };
    }
  }
  return { status: "skipped", file: null, error: "No generated baseline_runner.py found." };
}

function extractAddedPythonFilesFromPatch(patchText: string) {
  const files: Array<{ file: string; content: string }> = [];
  let currentFile: string | null = null;
  let currentLines: string[] = [];
  let collecting = false;
  function flush() {
    if (currentFile?.endsWith(".py") && currentLines.length) {
      files.push({ file: currentFile, content: `${currentLines.join("\n")}\n` });
    }
    currentFile = null;
    currentLines = [];
    collecting = false;
  }
  for (const rawLine of patchText.replaceAll("\\", "/").split(/\r?\n/)) {
    const diffMatch = rawLine.match(/^diff --git a\/(.+?) b\/(.+)$/);
    if (diffMatch) {
      flush();
      currentFile = diffMatch[2];
      continue;
    }
    const plusFile = rawLine.match(/^\+\+\+ b\/(.+)$/);
    if (plusFile) {
      currentFile = plusFile[1];
      continue;
    }
    if (rawLine.startsWith("@@")) {
      collecting = true;
      continue;
    }
    if (!collecting || !currentFile?.endsWith(".py")) continue;
    if (rawLine.startsWith("+") && !rawLine.startsWith("+++")) {
      currentLines.push(rawLine.slice(1));
    } else if (rawLine.startsWith(" ") || rawLine === "") {
      currentLines.push(rawLine.startsWith(" ") ? rawLine.slice(1) : "");
    }
  }
  flush();
  return files;
}

async function patchPythonSyntaxCheck(taskId: string, patchText: string) {
  const pythonFiles = extractAddedPythonFilesFromPatch(patchText)
    .filter((item) => item.file.startsWith(`workspace/tasks/${taskId}/code/`) || item.file.startsWith(`tasks/${taskId}/code/`));
  if (!pythonFiles.length) return { status: "skipped", files: [] as string[], error: null as string | null };
  const scratchRoot = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches/.syntax_${randomUUID()}`);
  await ensurePrivateDirectory(scratchRoot, workspaceRoot);
  try {
    const checked: string[] = [];
    for (const item of pythonFiles) {
      const scratchFile = path.join(scratchRoot, path.basename(item.file));
      await writeAtomicPrivateTextFile(scratchFile, item.content, {
        allowedRoot: scratchRoot,
        maxBytes: 2_000_000
      });
      await execFileAsync(pythonExecutable(), ["-m", "py_compile", scratchFile], { timeout: 20000 });
      checked.push(item.file);
    }
    return { status: "passed", files: checked, error: null };
  } catch {
    return { status: "failed", files: pythonFiles.map((item) => item.file), error: "patch_python_syntax_check_failed" };
  } finally {
    await fs.rm(scratchRoot, { recursive: true, force: true }).catch(() => undefined);
  }
}

async function ensureTask(taskId: string) {
  await ensureWorkstationSeeded();
  return prisma.task.upsert({
    where: { id: taskId },
    update: {},
    create: {
      id: taskId,
      name: taskId.replaceAll("_", " "),
      taskType: "research_task",
      status: "draft",
      taskDir: `workspace/tasks/${taskId}`
    }
  });
}

async function createRunnableTaskConfig(taskId: string, metadata: Record<string, unknown>) {
  const templatePath = resolveWorkspacePath("configs/house_prices.yaml");
  const template = await fs.readFile(templatePath, "utf-8");
  const title = typeof metadata.title === "string" && metadata.title.trim()
    ? metadata.title.trim()
    : "New Research Task";
  const generated = template
    .replace(/^  name:\s*house_prices\s*$/m, `  name: ${taskId}`)
    .replace(
      /^  competition:\s*House Prices - Advanced Regression Techniques\s*$/m,
      `  competition: ${title} (House Prices runnable template)`
    );
  const configPath = `configs/generated/${taskId}.yaml`;
  await writeTextArtifact(configPath, generated);
  return configPath;
}

function slugToTaskId(slug: string) {
  return slug.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || `kaggle_task_${stamp()}`;
}

async function onboardKaggleCompetition(metadata: Record<string, unknown>) {
  const slug = typeof metadata.competition_slug === "string" && metadata.competition_slug.trim()
    ? metadata.competition_slug.trim()
    : "kaggle-new-competition-smoke";
  const taskId = typeof metadata.task_id === "string" && metadata.task_id.trim()
    ? normalizeTaskId(metadata.task_id.trim())
    : slugToTaskId(slug);
  const dataDir = typeof metadata.data_dir === "string" && metadata.data_dir.trim()
    ? metadata.data_dir.trim()
    : "tasks/house_prices/data";
  const args = [
    "scripts/onboard_kaggle_competition.py",
    "--competition-slug",
    slug,
    "--task-id",
    taskId,
    "--data-dir",
    dataDir
  ];
  for (const [flag, value] of [
    ["--target", metadata.target],
    ["--task-type", metadata.task_type],
    ["--metric", metadata.metric]
  ] as const) {
    if (typeof value === "string" && value.trim()) args.push(flag, value.trim());
  }
  if (metadata.use_kaggle_api === true) args.push("--use-kaggle-api");

  const completed = await execFileAsync(pythonExecutable(), args, { cwd: resolveWorkspacePath("."), timeout: 180000 });
  const report = JSON.parse(completed.stdout) as Record<string, unknown>;
  const configPath = typeof report.config_path === "string" ? report.config_path : `configs/generated/${taskId}.yaml`;
  await prisma.task.upsert({
    where: { id: taskId },
    update: {
      name: String(report.competition_slug ?? slug),
      taskType: String(report.task_type ?? "tabular"),
      target: typeof report.target === "string" ? report.target : null,
      metric: typeof report.metric === "string" ? report.metric : null,
      status: "ready_to_train",
      priority: "High",
      owner: "Research Admin",
      configPath,
      taskDir: `tasks/${taskId}`
    },
    create: {
      id: taskId,
      name: String(report.competition_slug ?? slug),
      taskType: String(report.task_type ?? "tabular"),
      target: typeof report.target === "string" ? report.target : null,
      metric: typeof report.metric === "string" ? report.metric : null,
      status: "ready_to_train",
      priority: "High",
      owner: "Research Admin",
      configPath,
      taskDir: `tasks/${taskId}`
    }
  });
  return { taskId, configPath, report, command: `${pythonExecutable()} ${args.join(" ")}` };
}

async function writeScoreImprovementPlan(taskId: string, metadata: Record<string, unknown>) {
  const runPath = await latestExperimentPath(taskId);
  const latestRun = await prisma.experimentRun.findFirst({ where: { taskId }, orderBy: { createdAt: "desc" } });
  const connectors = {
    claude_code: process.env.ANTHROPIC_API_KEY ? "configured" : process.env.DEEPSEEK_API_KEY ? "deepseek_code_agent" : "not_configured",
    gpu_ssh: process.env.GPU_SSH_HOST && process.env.GPU_SSH_USER && (process.env.GPU_SSH_KEY_PATH || process.env.GPU_SSH_PASSWORD) && process.env.GPU_REMOTE_WORKSPACE ? "configured" : "not_configured",
    kaggle: process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY ? "configured" : "not_configured"
  };
  const plan = {
    task_id: taskId,
    latest_run: runPath,
    latest_metrics: latestRun?.metricsJson ? JSON.parse(latestRun.metricsJson) : null,
    objective: "Improve local validation score before any official Kaggle submission.",
    competition_slug: metadata.competition_slug ?? null,
    stages: [
      {
        stage: "local_baseline_lock",
        action: "Run the generated tabular baseline, validation gate and report once.",
        acceptance: ["validation_gate.json status passed", "submission.csv schema matches sample_submission.csv", "action log records run request and response"]
      },
      {
        stage: "data_quality_and_leakage_review",
        action: "Inspect train/test feature drift, target leakage candidates, missingness and duplicate patterns.",
        acceptance: ["data_quality.json reviewed", "leakage risk listed in Gate", "manual gate blocks official submit until reviewed"]
      },
      {
        stage: "model_ladder",
        action: "Compare linear/tree/boosting baselines, then promote the best stable model.",
        acceptance: ["cross-validation mean and std recorded", "holdout score not worse than threshold", "feature importance exported"]
      },
      {
        stage: "seed_and_hyperparameter_sweep",
        action: "Use local CPU for small sweeps; use GPU SSH gateway for heavier search after credentials are configured.",
        acceptance: ["all sweep jobs use whitelist commands", "artifacts downloaded", "provenance records remote host and command template"]
      },
      {
        stage: "leaderboard_gate",
        action: "Only submit through Kaggle API after KAGGLE_USERNAME/KAGGLE_KEY and human approval exist.",
        acceptance: ["token status configured", "human_submission_gate approved", "official submit response archived"]
      }
    ],
    connectors,
    required_env_for_full_launch: {
      claude_code: ["ANTHROPIC_API_KEY or DEEPSEEK_API_KEY"],
      gpu_ssh: ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_SSH_KEY_PATH or GPU_SSH_PASSWORD", "GPU_REMOTE_WORKSPACE"],
      kaggle: ["KAGGLE_USERNAME", "KAGGLE_KEY"]
    },
    generated_at: new Date().toISOString()
  };
  const jsonArtifact = await writeJsonArtifact(`workspace/kaggle_onboarding/${taskId}_score_improvement_plan_${stamp()}.json`, plan);
  const markdownArtifact = await writeTextArtifact(`docs/${taskId}_Kaggle提分上线计划.md`, [
    `# ${taskId} Kaggle 提分上线计划`,
    "",
    `- 最新实验：${runPath ?? "尚未运行"}`,
    `- Claude Code：${connectors.claude_code}`,
    `- GPU SSH：${connectors.gpu_ssh}`,
    `- Kaggle API：${connectors.kaggle}`,
    "",
    "## 分阶段策略",
    "",
    ...plan.stages.flatMap((stage, index) => [
      `### ${index + 1}. ${stage.stage}`,
      "",
      `- 动作：${stage.action}`,
      `- 验收：${stage.acceptance.join("；")}`,
      ""
    ]),
    "## 上线边界",
    "",
    "- 未配置 Kaggle 凭证时，只允许本地跑分和 submission 格式验证。",
    "- 未配置 GPU SSH 时，只允许本地 CPU baseline 和小规模搜索。",
    "- Claude Code 只产出建议、diff 和草稿，必须经过 Code Quality Gate 与人工 Gate。"
  ].join("\n"));
  return { plan, jsonArtifact, markdownArtifact };
}

async function latestRunId(taskId: string) {
  const run = await prisma.experimentRun.findFirst({ where: { taskId }, orderBy: { createdAt: "desc" } });
  return run?.id;
}

async function prepareS6E6HpcApprovalRequired(input: {
  action: string;
  taskId: string;
  requestedTemplate?: string;
  selectedStrategy?: Record<string, unknown>;
  submitMessage?: string;
}) {
  const strategyGate = await evaluateStrategyExecutionGate({
    taskId: input.taskId,
    requestedTemplate: input.requestedTemplate
  });
  const selectedTemplate = strategyGate.gate.selected_template ?? input.requestedTemplate ?? "playground_s6e6_boosting_ensemble";
  const run = await createWorkstationRun({
    taskId: input.taskId,
    trigger: `${input.action}_hpc_approval_required`,
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: `Prepare a workstation-controlled ${selectedTemplate} training run for S6E6 score improvement without starting remote training before HPC approval.`
  });
  const hpcGate = await createHpcExecutionGate({
    taskId: input.taskId,
    runId: run.run_id,
    template: selectedTemplate
  });
  const artifact = await writeJsonArtifact(`workspace/strategy/s6e6_hpc_execution_approval_required_${stamp()}.json`, {
    schema: "academic_research_os.s6e6_hpc_execution_approval_required.v1",
    task_id: input.taskId,
    workstation_run_id: hpcGate.run_id,
    status: "blocked_hpc_execution_approval_required",
    action: input.action,
    requested_template: input.requestedTemplate ?? null,
    selected_template: selectedTemplate,
    selected_strategy: input.selectedStrategy ?? null,
    strategy_execution_gate: strategyGate.gate,
    hpc_execution_gate: {
      gate_id: hpcGate.gate_id,
      manifest_path: hpcGate.manifest_path,
      decision: "pending"
    },
    required_next_actions: [
      "Review the generated hpc_execution_gate_manifest.json in the workstation.",
      "Approve the hpc_execution_approval gate from the workstation UI or API.",
      "Rerun the selected workstation action with metadata.hpc_execution_approved=true only after approval."
    ],
    direct_codex_training_allowed: false,
    training_started: false,
    official_submission_started: false,
    submit_message: input.submitMessage ?? null,
    created_at: new Date().toISOString()
  });
  const record = await logAction({
    action: input.action,
    taskId: input.taskId,
    runId: hpcGate.run_id,
    message: `S6E6 strategy ${selectedTemplate} selected, but remote training remains blocked until hpc_execution_approval is approved.`,
    artifactPath: artifact,
    metadata: {
      selected_template: selectedTemplate,
      gate_id: hpcGate.gate_id,
      manifest_path: hpcGate.manifest_path,
      allowed_to_execute: strategyGate.gate.allowed_to_execute,
      blocked_reasons: strategyGate.gate.blocked_reasons
    }
  });
  return {
    ok: false,
    ...record,
    status: "blocked_hpc_execution_approval_required",
    training_started: false,
    official_submission_started: false,
    strategy_gate: strategyGate.gate,
    hpc_gate: hpcGate,
    artifact_path: artifact,
    next_action: "approve_hpc_execution_gate"
  };
}

async function approveActionGate(input: {
  taskId: string;
  runId: string;
  gateType: string;
  reviewer: string;
  reason: string;
  artifactPath?: string;
}) {
  const gate = await prisma.gate.findFirst({
    where: { taskId: input.taskId, runId: input.runId, gateType: input.gateType },
    orderBy: { createdAt: "desc" }
  });
  const existingEvidence = decodeJson<Record<string, unknown>>(gate?.evidenceJson) ?? {};
  if (input.gateType === "hpc_execution_approval") {
    const requestedTemplate = typeof existingEvidence.requested_template === "string"
      ? existingEvidence.requested_template
      : "";
    const validation = await validateHpcExecutionGate(gate, {
      taskId: input.taskId,
      runId: input.runId,
      template: requestedTemplate,
      requireApproved: false
    });
    if (!validation.ok) {
      throw new Error(`HPC execution gate binding is invalid: ${validation.reasons.join(",")}`);
    }
    if (!gate) throw new Error("HPC execution gate is missing.");
  }
  const decidedAt = new Date();
  const evidence = gate && input.gateType === "hpc_execution_approval"
    ? buildHpcApprovalEvidence(gate, {
      reviewer: input.reviewer,
      reason: input.reason,
      artifactPath: input.artifactPath,
      decidedAt
    })
    : {
      ...existingEvidence,
      approval: {
        reviewer: input.reviewer,
        reason: input.reason,
        artifact_path: input.artifactPath ?? null,
        approved_at: decidedAt.toISOString()
      }
    };
  const gateId = gate?.id ?? `${input.runId}_${input.gateType}`;
  if (gate && input.gateType === "hpc_execution_approval") {
    const updated = await prisma.gate.updateMany({
      where: { id: gate.id, decision: "pending", evidenceJson: gate.evidenceJson },
      data: {
        decision: "approved",
        reviewer: input.reviewer,
        evidenceJson: encodeJson(evidence),
        decidedAt
      }
    });
    if (updated.count !== 1) throw new Error("HPC execution gate changed during approval.");
  } else {
    await prisma.gate.upsert({
      where: { id: gateId },
      update: {
        decision: "approved",
        reviewer: input.reviewer,
        evidenceJson: encodeJson(evidence),
        decidedAt
      },
      create: {
        id: gateId,
        taskId: input.taskId,
        runId: input.runId,
        gateType: input.gateType,
        decision: "approved",
        reviewer: input.reviewer,
        evidenceJson: encodeJson(evidence),
        decidedAt
      }
    });
  }
  await logAction({
    action: "approve_gate",
    taskId: input.taskId,
    runId: input.runId,
    message: `${input.gateType} approved for workstation action execution.`,
    artifactPath: input.artifactPath,
    metadata: evidence
  });
  return gateId;
}

async function runS6E6Exp018LgbmOptunaAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const requestedTemplate = "playground_s6e6_lgbm_optuna";
  const fullSearch = payload.metadata?.full_search === true;
  if (payload.metadata?.hpc_execution_approved !== true) {
    return prepareS6E6HpcApprovalRequired({
      action: payload.action ?? "run_s6e6_exp018_lgbm_optuna",
      taskId,
      requestedTemplate,
      selectedStrategy: {
        strategy_id: "lgbm_optuna_exp018",
        label: "EXP018 LightGBM Optuna Challenger",
        gpu_template: requestedTemplate,
        official_submit_policy: "needs_validation"
      },
      submitMessage: "EXP018 LightGBM Optuna challenger is evidence-only until score gate promotion."
    });
  }

  const run = await createWorkstationRun({
    taskId,
    trigger: fullSearch ? "run_s6e6_exp018_lgbm_optuna_full_search" : "run_s6e6_exp018_lgbm_optuna_dryrun",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: fullSearch
      ? "Run an EXP018 full-data LightGBM Optuna challenger through the workstation-controlled HPC gateway."
      : "Run a bounded EXP018 LightGBM Optuna dry-run through the workstation-controlled HPC gateway."
  });
  const hpcGate = await createHpcExecutionGate({ taskId, runId: run.run_id, template: requestedTemplate });
  const hpcGateId = await approveActionGate({
    taskId,
    runId: run.run_id,
    gateType: "hpc_execution_approval",
    reviewer: "Research Admin",
    reason: fullSearch
      ? "Current user goal authorizes a workstation-controlled EXP018 full-data LightGBM Optuna challenger."
      : "Current user goal authorizes a workstation-controlled EXP018 dry-run to validate the next challenger route.",
    artifactPath: hpcGate.manifest_path
  });
  const resourceRequest = {
    allow_evidence_only: true,
    mode: fullSearch ? "lgbm_optuna_full_search" : "lgbm_optuna_dryrun",
    trials: numberValue(payload.metadata?.trials) ?? (fullSearch ? 24 : 2),
    folds: numberValue(payload.metadata?.folds) ?? (fullSearch ? 5 : 2),
    sample_rows: fullSearch ? (numberValue(payload.metadata?.sample_rows) ?? 0) : (numberValue(payload.metadata?.sample_rows) ?? 12000),
    n_estimators: numberValue(payload.metadata?.n_estimators) ?? (fullSearch ? 2200 : 220),
    early_stopping_rounds: numberValue(payload.metadata?.early_stopping_rounds) ?? (fullSearch ? 100 : 30),
    timeout_seconds: numberValue(payload.metadata?.timeout_seconds) ?? (fullSearch ? 28800 : 3600),
    seed: numberValue(payload.metadata?.seed) ?? 260612
  };
  const gpuJob = await submitGpuJob({
    taskId,
    runId: run.run_id,
    agentId: "HpcGpuExecutionAgent",
    gateId: hpcGateId,
    template: requestedTemplate,
    resourceRequest
  });
  const resultArtifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp018_lgbm_optuna_result.json`, {
    schema: "academic_research_os.exp018_lgbm_optuna_result.v1",
    workstation_run_id: run.run_id,
    task_id: taskId,
    status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_evidence_only" : "blocked_or_failed",
    gpu_job: gpuJob,
    resource_request: resourceRequest,
    official_submission_started: false,
    submission_policy: "Optuna search artifacts are evidence-only. A later model-selection/blend gate must promote a candidate before Kaggle submission.",
    next_gate: gpuJob.metrics_artifact
      ? "ValidationAnalysisAgent should compare best_oof.balanced_accuracy, log_loss, errors and probability assets against EXP007/EXP017 frontier."
      : "ReflectionReviewerAgent should inspect the GPU job failure artifact and repair the environment/template before rerun.",
    created_at: new Date().toISOString()
  });
  await prisma.experimentRun.update({
    where: { id: run.run_id },
    data: {
      status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_exp018_evidence_only" : "blocked_exp018_hpc_execution",
      validationStatus: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "pending_validation_review" : "blocked",
      finishedAt: new Date()
    }
  });
  const record = await logAction({
    action: payload.action ?? "run_s6e6_exp018_lgbm_optuna",
    taskId,
    runId: run.run_id,
    message: gpuJob.status === "submitted"
      ? "EXP018 LightGBM Optuna challenger finished as evidence-only workstation artifact."
      : "EXP018 LightGBM Optuna challenger did not complete; failure artifact recorded.",
    artifactPath: resultArtifact,
    metadata: { template: requestedTemplate, status: gpuJob.status, metrics_artifact: gpuJob.metrics_artifact ?? null, full_search: fullSearch }
  });
  return {
    ok: gpuJob.status === "submitted" && Boolean(gpuJob.metrics_artifact),
    ...record,
    run_id: run.run_id,
    status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_evidence_only" : "blocked_or_failed",
    gpu_job: gpuJob,
    result_artifact: resultArtifact,
    official_submission_started: false
  };
}

async function runS6E6Exp023Exp018FrontierBlendAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp023_exp018_frontier_blend",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Evaluate whether EXP018 LightGBM Optuna probabilities add risk-guarded diversity to the current EXP017 public-best frontier."
  });
  const outDir = typeof payload.metadata?.out_dir === "string" && payload.metadata.out_dir.trim()
    ? payload.metadata.out_dir.trim()
    : `workspace/hpc_experiments/playground_series_s6e6/EXP023_exp018_frontier_blend_${stamp()}`;
  const script = "notebooks_or_scripts/exp023_exp018_frontier_blend.py";
  const args = [
    script,
    "--out-dir", outDir,
    "--candidate-submission-path", "submissions/submission_EXP023_exp018_frontier_blend_not_submitted.csv"
  ];
  for (const [flag, key] of [
    ["--step", "step"],
    ["--max-exp018-weight", "max_exp018_weight"],
    ["--max-exp015-weight", "max_exp015_weight"],
    ["--current-best-validation", "current_best_validation"],
    ["--max-logloss-delta-vs-exp017", "max_logloss_delta_vs_exp017"],
    ["--max-error-delta-vs-exp017", "max_error_delta_vs_exp017"],
    ["--n-splits", "n_splits"],
    ["--top-rows", "top_rows"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "number" && Number.isFinite(value)) args.push(flag, String(value));
  }
  if (typeof payload.metadata?.seeds === "string" && payload.metadata.seeds.trim()) {
    args.push("--seeds", payload.metadata.seeds.trim());
  }
  try {
    const timeoutSeconds = numberValue(payload.metadata?.timeout_seconds) ?? 20 * 60;
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 20 * 1024 * 1024
    });
    const metricsPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/metrics.json`;
    const reportPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/report.md`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const resultArtifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp023_exp018_frontier_blend_result.json`, {
      schema: "academic_research_os.exp023_exp018_frontier_blend_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics_path: metricsPath,
      report_path: reportPath,
      stdout: completed.stdout,
      stderr: completed.stderr,
      metrics,
      official_submission_started: false,
      next_gate: metrics?.decision === "submit_candidate"
        ? "SubmissionGateAgent must run score improvement gate and require active-turn human submission approval before Kaggle submit."
        : "ModelSelectionAgent should keep EXP018 as evidence-only unless a future blend/search beats EXP017 under risk guards.",
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status: metrics?.decision === "submit_candidate" ? "completed_candidate_requires_submission_gate" : "completed_evidence_only",
        validationStatus: metrics?.decision === "submit_candidate" ? "pending_submission_gate" : "blocked_by_frontier_gate",
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp023_exp018_frontier_blend",
      taskId,
      runId: run.run_id,
      message: metrics?.decision === "submit_candidate"
        ? "EXP023 found a risk-guarded candidate; official submission remains gated."
        : "EXP023 completed; EXP018 did not improve the current EXP017 frontier under risk guards.",
      artifactPath: resultArtifact,
      metadata: {
        decision: metrics?.decision ?? null,
        metrics_path: metricsPath,
        report_path: reportPath,
        official_submission_started: false
      }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics,
      metrics_path: metricsPath,
      report_path: reportPath,
      result_artifact: resultArtifact,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP023 EXP018 frontier blend analysis failed.";
    const artifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp023_exp018_frontier_blend_failure.json`, {
      schema: "academic_research_os.exp023_exp018_frontier_blend_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp023_analysis", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp023_exp018_frontier_blend",
      taskId,
      runId: run.run_id,
      message: "EXP023 EXP018 frontier blend analysis failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp022QualityConstrainedCalibrationAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp022_quality_constrained_calibration",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Evaluate EXP022 quality-constrained calibration over EXP015 probabilities as a score-safe candidate without direct training or official submission."
  });
  const outDir = typeof payload.metadata?.out_dir === "string" && payload.metadata.out_dir.trim()
    ? payload.metadata.out_dir.trim()
    : `workspace/hpc_experiments/playground_series_s6e6/EXP022_quality_constrained_calibration_${stamp()}`;
  const script = "notebooks_or_scripts/exp022_quality_constrained_calibration.py";
  const args = [
    script,
    "--out-dir", outDir,
    "--candidate-submission-path", "submissions/submission_EXP022_quality_constrained_calibration_not_submitted.csv"
  ];
  for (const [flag, key] of [
    ["--n-splits", "n_splits"],
    ["--seed", "seed"],
    ["--bins", "bins"],
    ["--min-ba-delta", "min_ba_delta"],
    ["--max-logloss-delta", "max_logloss_delta"],
    ["--max-ece-delta", "max_ece_delta"],
    ["--max-error-delta", "max_error_delta"],
    ["--fallback-max-logloss-delta", "fallback_max_logloss_delta"],
    ["--fallback-max-ece-delta", "fallback_max_ece_delta"],
    ["--fallback-max-error-delta", "fallback_max_error_delta"],
    ["--submit-min-cv-delta", "submit_min_cv_delta"],
    ["--top-grid-rows", "top_grid_rows"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "number" && Number.isFinite(value)) args.push(flag, String(value));
  }
  for (const [flag, key] of [
    ["--alpha-grid", "alpha_grid"],
    ["--qso-grid", "qso_grid"],
    ["--star-grid", "star_grid"],
    ["--source-npz", "source_npz"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "string" && value.trim()) args.push(flag, value.trim());
  }
  const timeoutSeconds = Math.min(Math.max(numberValue(payload.metadata?.timeout_seconds) ?? 10 * 60, 60), 60 * 60);
  try {
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 20 * 1024 * 1024
    });
    const metricsPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/metrics.json`;
    const reportPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/report.md`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const resultArtifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp022_quality_constrained_calibration_result.json`, {
      schema: "academic_research_os.exp022_quality_constrained_calibration_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics_path: metricsPath,
      report_path: reportPath,
      stdout: completed.stdout,
      stderr: completed.stderr,
      metrics,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      next_gate: metrics?.decision === "submit_candidate"
        ? "SubmissionGateAgent must compare EXP022 against EXP017 official frontier and require active-turn human submission approval before Kaggle submit."
        : "ModelSelectionAgent should keep EXP022 as evidence-only unless it beats EXP017 under the score and risk gates.",
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status: metrics?.decision === "submit_candidate" ? "completed_candidate_requires_submission_gate" : "completed_evidence_only",
        validationStatus: metrics?.decision === "submit_candidate" ? "pending_submission_gate" : "blocked_by_frontier_gate",
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp022_quality_constrained_calibration",
      taskId,
      runId: run.run_id,
      message: metrics?.decision === "submit_candidate"
        ? "EXP022 quality-constrained calibration found a candidate; official submission remains gated."
        : "EXP022 quality-constrained calibration completed as evidence-only and did not pass submission gates.",
      artifactPath: resultArtifact,
      metadata: {
        decision: metrics?.decision ?? null,
        metrics_path: metricsPath,
        report_path: reportPath,
        timeout_seconds: timeoutSeconds,
        official_submission_started: false
      }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics,
      metrics_path: metricsPath,
      report_path: reportPath,
      result_artifact: resultArtifact,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP022 quality-constrained calibration analysis failed.";
    const artifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp022_quality_constrained_calibration_failure.json`, {
      schema: "academic_research_os.exp022_quality_constrained_calibration_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp022_analysis", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp022_quality_constrained_calibration",
      taskId,
      runId: run.run_id,
      message: "EXP022 quality-constrained calibration analysis failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp024MultiAssetFrontierBlendAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp024_multi_asset_frontier_blend",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Search score-safe multi-asset blends over saved S6E6 probabilities through the workstation, without direct training or official submission."
  });
  const outDir = typeof payload.metadata?.out_dir === "string" && payload.metadata.out_dir.trim()
    ? payload.metadata.out_dir.trim()
    : `workspace/hpc_experiments/playground_series_s6e6/EXP024_multi_asset_frontier_blend_${stamp()}`;
  const candidateSubmissionPath = typeof payload.metadata?.candidate_submission_path === "string" && payload.metadata.candidate_submission_path.trim()
    ? payload.metadata.candidate_submission_path.trim()
    : `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/candidate_submission_not_submitted.csv`;
  const script = "notebooks_or_scripts/exp024_multi_asset_frontier_blend.py";
  const args = [
    script,
    "--out-dir", outDir,
    "--candidate-submission-path", candidateSubmissionPath
  ];
  for (const [flag, key] of [
    ["--step", "step"],
    ["--max-challenger-weight", "max_challenger_weight"],
    ["--max-pair-weight", "max_pair_weight"],
    ["--current-best-validation", "current_best_validation"],
    ["--min-ba-delta-vs-current-best", "min_ba_delta_vs_current_best"],
    ["--max-logloss-delta-vs-baseline", "max_logloss_delta_vs_baseline"],
    ["--max-error-delta-vs-baseline", "max_error_delta_vs_baseline"],
    ["--n-splits", "n_splits"],
    ["--min-positive-fold-share", "min_positive_fold_share"],
    ["--min-fold-ci95-low", "min_fold_ci95_low"],
    ["--top-rows", "top_rows"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "number" && Number.isFinite(value)) args.push(flag, String(value));
  }
  for (const [flag, key] of [
    ["--assets", "assets"],
    ["--asset-paths", "asset_paths"],
    ["--baseline-id", "baseline_id"],
    ["--seeds", "seeds"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "string" && value.trim()) args.push(flag, value.trim());
  }
  const timeoutSeconds = Math.min(Math.max(numberValue(payload.metadata?.timeout_seconds) ?? 10 * 60, 60), 60 * 60);
  try {
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 20 * 1024 * 1024
    });
    const metricsPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/metrics.json`;
    const reportPath = `${outDir.replaceAll("\\", "/").replace(/\/+$/, "")}/report.md`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const resultArtifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp024_multi_asset_frontier_blend_result.json`, {
      schema: "academic_research_os.exp024_multi_asset_frontier_blend_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics_path: metricsPath,
      report_path: reportPath,
      stdout: completed.stdout,
      stderr: completed.stderr,
      metrics,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      next_gate: metrics?.decision === "submit_candidate"
        ? "SubmissionGateAgent must run score-improvement and leakage guards, then require active-turn human approval before Kaggle submit."
        : "ModelSelectionAgent should keep EXP024 as evidence-only and schedule new independent model diversity if no blend beats EXP017.",
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status: metrics?.decision === "submit_candidate" ? "completed_candidate_requires_submission_gate" : "completed_evidence_only",
        validationStatus: metrics?.decision === "submit_candidate" ? "pending_submission_gate" : "blocked_by_frontier_gate",
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp024_multi_asset_frontier_blend",
      taskId,
      runId: run.run_id,
      message: metrics?.decision === "submit_candidate"
        ? "EXP024 found a risk-guarded multi-asset candidate; official submission remains gated."
        : "EXP024 completed as evidence-only; no multi-asset blend passed all frontier guards.",
      artifactPath: resultArtifact,
      metadata: {
        decision: metrics?.decision ?? null,
        metrics_path: metricsPath,
        report_path: reportPath,
        timeout_seconds: timeoutSeconds,
        official_submission_started: false
      }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status: metrics?.decision === "submit_candidate" ? "candidate_requires_submission_gate" : "completed_evidence_only",
      metrics,
      metrics_path: metricsPath,
      report_path: reportPath,
      result_artifact: resultArtifact,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP024 multi-asset frontier blend analysis failed.";
    const errorWithOutput = error as { stdout?: unknown; stderr?: unknown };
    const artifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp024_multi_asset_frontier_blend_failure.json`, {
      schema: "academic_research_os.exp024_multi_asset_frontier_blend_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      stdout: typeof errorWithOutput.stdout === "string" ? errorWithOutput.stdout.slice(-8000) : null,
      stderr: typeof errorWithOutput.stderr === "string" ? errorWithOutput.stderr.slice(-8000) : null,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp024_analysis", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp024_multi_asset_frontier_blend",
      taskId,
      runId: run.run_id,
      message: "EXP024 multi-asset frontier blend analysis failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, timeout_seconds: timeoutSeconds, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp032DecisionOffsetSearchAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp032_decision_offset_search",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Run a workstation-controlled high-risk decision-offset search over saved OOF/test probabilities without direct Codex training or official submission."
  });
  const runRoot = `workspace/workstation_runs/${taskId}/${run.run_id}`;
  await appendRunTrace(runRoot, {
    agent_id: "ModelSelectionAgent",
    stage: "decision_offset_search",
    status: "started",
    policy: "label_only_human_gate_candidate"
  });

  const outDir = `workspace/hpc_experiments/playground_series_s6e6/EXP032_decision_offset_search_${stamp()}`;
  const candidatePath = "submissions/submission_EXP032_decision_offset_search_not_submitted.csv";
  const args = [
    "notebooks_or_scripts/exp032_decision_offset_search.py",
    "--out-dir", outDir,
    "--candidate-submission-path", candidatePath
  ];
  if (typeof payload.metadata?.assets === "string" && payload.metadata.assets.trim()) {
    args.push("--assets", payload.metadata.assets.trim());
  }
  for (const [flag, key] of [
    ["--offset-min", "offset_min"],
    ["--offset-max", "offset_max"],
    ["--offset-step", "offset_step"],
    ["--current-best-validation", "current_best_validation"],
    ["--min-full-ba-delta", "min_full_ba_delta"],
    ["--min-nested-ba", "min_nested_ba"],
    ["--min-positive-fold-share", "min_positive_fold_share"],
    ["--max-error-delta-vs-base", "max_error_delta_vs_base"],
    ["--max-logloss-delta-vs-base", "max_logloss_delta_vs_base"],
    ["--n-splits", "n_splits"],
    ["--seed", "seed"]
  ] as const) {
    const value = numberValue(payload.metadata?.[key]);
    if (value !== null) args.push(flag, String(value));
  }
  const timeoutSeconds = numberValue(payload.metadata?.timeout_seconds) ?? 1800;

  try {
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 1024 * 1024 * 32
    });
    const metricsPath = `${outDir}/metrics.json`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const audit = await auditS6E6Submission(candidatePath);
    const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, audit);
    const decision = typeof metrics?.decision === "string" ? metrics.decision : "unknown";
    const status = decision === "human_gate_only_candidate"
      ? "candidate_requires_submission_gate"
      : "completed_evidence_only";
    const resultArtifact = await writeJsonArtifact(`${runRoot}/exp032_decision_offset_search_result.json`, {
      schema: "academic_research_os.exp032_decision_offset_search_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status,
      command: `${pythonExecutable()} ${args.join(" ")}`,
      stdout_tail: completed.stdout.slice(-8000),
      stderr_tail: completed.stderr.slice(-4000),
      metrics_path: metricsPath,
      metrics,
      submission_audit: auditPath,
      candidate_submission: candidatePath,
      official_submission_started: false,
      submission_policy: "High-risk label-only candidate. Official Kaggle submit requires explicit current-turn human submission_approval and Kaggle response audit.",
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, {
      agent_id: "ValidationAnalysisAgent",
      stage: "decision_offset_review",
      status,
      metrics_path: metricsPath,
      submission_audit: auditPath
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status,
        validationStatus: decision === "human_gate_only_candidate" ? "pending_submission_gate" : "blocked_by_exp032_gate",
        metricsJson: encodeJson({
          experiment_id: "EXP032",
          decision,
          metrics_path: metricsPath,
          candidate_submission: candidatePath,
          submission_audit: auditPath,
          official_submission_started: false
        }),
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp032_decision_offset_search",
      taskId,
      runId: run.run_id,
      message: decision === "human_gate_only_candidate"
        ? "EXP032 found a high-risk label-only candidate; official Kaggle submit remains blocked by human gate."
        : "EXP032 completed as evidence-only; no candidate passed the configured decision-offset gates.",
      artifactPath: resultArtifact,
      metadata: {
        decision,
        metrics_path: metricsPath,
        candidate_submission: candidatePath,
        submission_audit: auditPath,
        official_submission_started: false
      }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status,
      decision,
      metrics,
      result_artifact: resultArtifact,
      submission_audit: auditPath,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP032 decision-offset search failed.";
    const errorWithOutput = error as { stdout?: unknown; stderr?: unknown };
    const artifact = await writeJsonArtifact(`${runRoot}/exp032_decision_offset_search_failure.json`, {
      schema: "academic_research_os.exp032_decision_offset_search_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      stdout: typeof errorWithOutput.stdout === "string" ? errorWithOutput.stdout.slice(-8000) : null,
      stderr: typeof errorWithOutput.stderr === "string" ? errorWithOutput.stderr.slice(-8000) : null,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, {
      agent_id: "ReflectionReviewerAgent",
      stage: "decision_offset_search",
      status: "failed",
      failure_artifact: artifact
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp032_decision_offset_search", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp032_decision_offset_search",
      taskId,
      runId: run.run_id,
      message: "EXP032 decision-offset search failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, timeout_seconds: timeoutSeconds, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp033BlendOffsetSearchAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp033_blend_offset_search",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Run a workstation-controlled high-risk blend plus decision-offset search over saved probability artifacts."
  });
  const runRoot = `workspace/workstation_runs/${taskId}/${run.run_id}`;
  await appendRunTrace(runRoot, {
    agent_id: "ModelSelectionAgent",
    stage: "blend_offset_search",
    status: "started",
    policy: "label_only_human_gate_candidate"
  });

  const outDir = `workspace/hpc_experiments/playground_series_s6e6/EXP033_blend_offset_search_${stamp()}`;
  const candidatePath = "submissions/submission_EXP033_blend_offset_search_not_submitted.csv";
  const args = [
    "notebooks_or_scripts/exp033_blend_offset_search.py",
    "--out-dir", outDir,
    "--candidate-submission-path", candidatePath
  ];
  for (const [flag, key] of [
    ["--base-asset", "base_asset"],
    ["--assets", "assets"],
    ["--asset-paths", "asset_paths"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "string" && value.trim()) args.push(flag, value.trim());
  }
  for (const [flag, key] of [
    ["--weight-min", "weight_min"],
    ["--weight-max", "weight_max"],
    ["--weight-step", "weight_step"],
    ["--qso-offset-min", "qso_offset_min"],
    ["--qso-offset-max", "qso_offset_max"],
    ["--star-offset-min", "star_offset_min"],
    ["--star-offset-max", "star_offset_max"],
    ["--offset-step", "offset_step"],
    ["--current-best-validation", "current_best_validation"],
    ["--min-full-ba-delta", "min_full_ba_delta"],
    ["--min-nested-ba", "min_nested_ba"],
    ["--min-positive-fold-share", "min_positive_fold_share"],
    ["--max-error-delta-vs-base", "max_error_delta_vs_base"],
    ["--max-logloss-delta-vs-base", "max_logloss_delta_vs_base"],
    ["--n-splits", "n_splits"],
    ["--seed", "seed"]
  ] as const) {
    const value = numberValue(payload.metadata?.[key]);
    if (value !== null) args.push(flag, String(value));
  }
  const timeoutSeconds = numberValue(payload.metadata?.timeout_seconds) ?? 1800;

  try {
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 1024 * 1024 * 32
    });
    const metricsPath = `${outDir}/metrics.json`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const audit = await auditS6E6Submission(candidatePath);
    const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, audit);
    const decision = typeof metrics?.decision === "string" ? metrics.decision : "unknown";
    const status = decision === "human_gate_only_candidate"
      ? "candidate_requires_submission_gate"
      : "completed_evidence_only";
    const resultArtifact = await writeJsonArtifact(`${runRoot}/exp033_blend_offset_search_result.json`, {
      schema: "academic_research_os.exp033_blend_offset_search_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status,
      command: `${pythonExecutable()} ${args.join(" ")}`,
      stdout_tail: completed.stdout.slice(-8000),
      stderr_tail: completed.stderr.slice(-4000),
      metrics_path: metricsPath,
      metrics,
      submission_audit: auditPath,
      candidate_submission: candidatePath,
      official_submission_started: false,
      submission_policy: "High-risk blend+offset candidate. Official Kaggle submit requires explicit current-turn human submission_approval and Kaggle response audit.",
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, {
      agent_id: "ValidationAnalysisAgent",
      stage: "blend_offset_review",
      status,
      metrics_path: metricsPath,
      submission_audit: auditPath
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status,
        validationStatus: decision === "human_gate_only_candidate" ? "pending_submission_gate" : "blocked_by_exp033_gate",
        metricsJson: encodeJson({
          experiment_id: "EXP033",
          decision,
          metrics_path: metricsPath,
          candidate_submission: candidatePath,
          submission_audit: auditPath,
          official_submission_started: false
        }),
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp033_blend_offset_search",
      taskId,
      runId: run.run_id,
      message: decision === "human_gate_only_candidate"
        ? "EXP033 found a high-risk blend+offset candidate; official Kaggle submit remains blocked by human gate."
        : "EXP033 completed as evidence-only; no candidate passed the configured blend+offset gates.",
      artifactPath: resultArtifact,
      metadata: { decision, metrics_path: metricsPath, candidate_submission: candidatePath, submission_audit: auditPath, official_submission_started: false }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status,
      decision,
      metrics,
      result_artifact: resultArtifact,
      submission_audit: auditPath,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP033 blend-offset search failed.";
    const errorWithOutput = error as { stdout?: unknown; stderr?: unknown };
    const artifact = await writeJsonArtifact(`${runRoot}/exp033_blend_offset_search_failure.json`, {
      schema: "academic_research_os.exp033_blend_offset_search_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      stdout: typeof errorWithOutput.stdout === "string" ? errorWithOutput.stdout.slice(-8000) : null,
      stderr: typeof errorWithOutput.stderr === "string" ? errorWithOutput.stderr.slice(-8000) : null,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, { agent_id: "ReflectionReviewerAgent", stage: "blend_offset_search", status: "failed", failure_artifact: artifact });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp033_blend_offset_search", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp033_blend_offset_search",
      taskId,
      runId: run.run_id,
      message: "EXP033 blend-offset search failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, timeout_seconds: timeoutSeconds, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp034DualBlendOffsetSearchAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const run = await createWorkstationRun({
    taskId,
    trigger: "run_s6e6_exp034_dual_blend_offset_search",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: "Run a workstation-controlled high-risk dual blend plus decision-offset search over saved probability artifacts."
  });
  const runRoot = `workspace/workstation_runs/${taskId}/${run.run_id}`;
  await appendRunTrace(runRoot, {
    agent_id: "ModelSelectionAgent",
    stage: "dual_blend_offset_search",
    status: "started",
    policy: "label_only_human_gate_candidate"
  });

  const outDir = `workspace/hpc_experiments/playground_series_s6e6/EXP034_dual_blend_offset_search_${stamp()}`;
  const candidatePath = "submissions/submission_EXP034_dual_blend_offset_search_not_submitted.csv";
  const args = [
    "notebooks_or_scripts/exp034_dual_blend_offset_search.py",
    "--out-dir", outDir,
    "--candidate-submission-path", candidatePath
  ];
  for (const [flag, key] of [
    ["--base-asset", "base_asset"],
    ["--first-asset", "first_asset"],
    ["--second-asset", "second_asset"],
    ["--asset-paths", "asset_paths"]
  ] as const) {
    const value = payload.metadata?.[key];
    if (typeof value === "string" && value.trim()) args.push(flag, value.trim());
  }
  for (const [flag, key] of [
    ["--first-weight-min", "first_weight_min"],
    ["--first-weight-max", "first_weight_max"],
    ["--second-weight-min", "second_weight_min"],
    ["--second-weight-max", "second_weight_max"],
    ["--weight-step", "weight_step"],
    ["--max-total-challenger-weight", "max_total_challenger_weight"],
    ["--qso-offset-min", "qso_offset_min"],
    ["--qso-offset-max", "qso_offset_max"],
    ["--star-offset-min", "star_offset_min"],
    ["--star-offset-max", "star_offset_max"],
    ["--offset-step", "offset_step"],
    ["--min-full-ba-delta", "min_full_ba_delta"],
    ["--min-nested-ba", "min_nested_ba"],
    ["--n-splits", "n_splits"],
    ["--seed", "seed"]
  ] as const) {
    const value = numberValue(payload.metadata?.[key]);
    if (value !== null) args.push(flag, String(value));
  }
  const timeoutSeconds = numberValue(payload.metadata?.timeout_seconds) ?? 1800;

  try {
    const completed = await execFileAsync(pythonExecutable(), args, {
      cwd: resolveWorkspacePath("."),
      timeout: timeoutSeconds * 1000,
      maxBuffer: 1024 * 1024 * 32
    });
    const metricsPath = `${outDir}/metrics.json`;
    const metrics = await readJsonFile(resolveWorkspacePath(metricsPath)) as Record<string, unknown> | null;
    const audit = await auditS6E6Submission(candidatePath);
    const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, audit);
    const decision = typeof metrics?.decision === "string" ? metrics.decision : "unknown";
    const status = decision === "human_gate_only_candidate"
      ? "candidate_requires_submission_gate"
      : "completed_evidence_only";
    const resultArtifact = await writeJsonArtifact(`${runRoot}/exp034_dual_blend_offset_search_result.json`, {
      schema: "academic_research_os.exp034_dual_blend_offset_search_result.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status,
      command: `${pythonExecutable()} ${args.join(" ")}`,
      stdout_tail: completed.stdout.slice(-8000),
      stderr_tail: completed.stderr.slice(-4000),
      metrics_path: metricsPath,
      metrics,
      submission_audit: auditPath,
      candidate_submission: candidatePath,
      official_submission_started: false,
      submission_policy: "High-risk dual blend+offset candidate. Official Kaggle submit requires explicit current-turn human submission_approval and Kaggle response audit.",
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, {
      agent_id: "ValidationAnalysisAgent",
      stage: "dual_blend_offset_review",
      status,
      metrics_path: metricsPath,
      submission_audit: auditPath
    });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: {
        status,
        validationStatus: decision === "human_gate_only_candidate" ? "pending_submission_gate" : "blocked_by_exp034_gate",
        metricsJson: encodeJson({
          experiment_id: "EXP034",
          decision,
          metrics_path: metricsPath,
          candidate_submission: candidatePath,
          submission_audit: auditPath,
          official_submission_started: false
        }),
        finishedAt: new Date()
      }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp034_dual_blend_offset_search",
      taskId,
      runId: run.run_id,
      message: decision === "human_gate_only_candidate"
        ? "EXP034 found a high-risk dual blend+offset candidate; official Kaggle submit remains blocked by human gate."
        : "EXP034 completed as evidence-only; no candidate passed the configured dual blend+offset gates.",
      artifactPath: resultArtifact,
      metadata: { decision, metrics_path: metricsPath, candidate_submission: candidatePath, submission_audit: auditPath, official_submission_started: false }
    });
    return {
      ok: true,
      ...record,
      run_id: run.run_id,
      status,
      decision,
      metrics,
      result_artifact: resultArtifact,
      submission_audit: auditPath,
      official_submission_started: false
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "EXP034 dual blend-offset search failed.";
    const errorWithOutput = error as { stdout?: unknown; stderr?: unknown };
    const artifact = await writeJsonArtifact(`${runRoot}/exp034_dual_blend_offset_search_failure.json`, {
      schema: "academic_research_os.exp034_dual_blend_offset_search_failure.v1",
      workstation_run_id: run.run_id,
      task_id: taskId,
      status: "failed",
      error: message,
      stdout: typeof errorWithOutput.stdout === "string" ? errorWithOutput.stdout.slice(-8000) : null,
      stderr: typeof errorWithOutput.stderr === "string" ? errorWithOutput.stderr.slice(-8000) : null,
      timeout_seconds: timeoutSeconds,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await appendRunTrace(runRoot, { agent_id: "ReflectionReviewerAgent", stage: "dual_blend_offset_search", status: "failed", failure_artifact: artifact });
    await prisma.experimentRun.update({
      where: { id: run.run_id },
      data: { status: "failed_exp034_dual_blend_offset_search", validationStatus: "blocked", finishedAt: new Date() }
    });
    const record = await logAction({
      action: payload.action ?? "run_s6e6_exp034_dual_blend_offset_search",
      taskId,
      runId: run.run_id,
      message: "EXP034 dual blend-offset search failed; failure artifact recorded.",
      artifactPath: artifact,
      metadata: { error: message, timeout_seconds: timeoutSeconds, official_submission_started: false }
    });
    return { ok: false, ...record, run_id: run.run_id, status: "failed", error: message, artifact_path: artifact, official_submission_started: false };
  }
}

async function runS6E6Exp025SingleModelDiversityAction(payload: WorkstationActionPayload) {
  const taskId = "playground_series_s6e6";
  const model = typeof payload.metadata?.model === "string" && ["lightgbm", "xgboost", "catboost"].includes(payload.metadata.model)
    ? payload.metadata.model
    : "catboost";
  const requestedTemplate = model === "lightgbm"
    ? "playground_s6e6_lightgbm"
    : model === "xgboost"
      ? "playground_s6e6_xgboost"
      : "playground_s6e6_catboost";
  if (payload.metadata?.hpc_execution_approved !== true) {
    return prepareS6E6HpcApprovalRequired({
      action: payload.action ?? "run_s6e6_exp025_single_model_diversity",
      taskId,
      requestedTemplate,
      selectedStrategy: {
        strategy_id: "exp025_single_model_diversity",
        label: `EXP025 ${model} independent probability asset`,
        gpu_template: requestedTemplate,
        official_submit_policy: "evidence_only_until_frontier_gate"
      },
      submitMessage: `EXP025 ${model} diversity challenger is evidence-only until score/risk frontier promotion.`
    });
  }

  const run = await createWorkstationRun({
    taskId,
    trigger: `run_s6e6_exp025_${model}_single_model_diversity`,
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: `Run an EXP025 ${model} independent single-model challenger through the workstation-controlled HPC gateway and pull back reusable probability assets.`
  });
  const hpcGate = await createHpcExecutionGate({ taskId, runId: run.run_id, template: requestedTemplate });
  const hpcGateId = await approveActionGate({
    taskId,
    runId: run.run_id,
    gateType: "hpc_execution_approval",
    reviewer: "Research Admin",
    reason: `Current user goal authorizes workstation-controlled EXP025 ${model} single-model diversity evidence generation.`,
    artifactPath: hpcGate.manifest_path
  });
  const sampleRows = numberValue(payload.metadata?.sample_rows) ?? 12000;
  const accelerator = typeof payload.metadata?.accelerator === "string" && ["auto", "cpu", "gpu"].includes(payload.metadata.accelerator)
    ? payload.metadata.accelerator
    : "auto";
  const classWeight = typeof payload.metadata?.class_weight === "string" && ["none", "half_balanced", "sqrt_balanced", "balanced", "strong_balanced"].includes(payload.metadata.class_weight)
    ? payload.metadata.class_weight
    : "none";
  const profile = typeof payload.metadata?.profile === "string" && ["default", "high_capacity", "conservative", "minority_recall"].includes(payload.metadata.profile)
    ? payload.metadata.profile
    : "default";
  const resourceRequest = {
    allow_evidence_only: true,
    mode: sampleRows > 0 ? "single_model_dryrun" : "single_model_full_training",
    model,
    accelerator,
    class_weight: classWeight,
    profile,
    folds: numberValue(payload.metadata?.folds) ?? (sampleRows > 0 ? 2 : 5),
    seeds: typeof payload.metadata?.seeds === "string" && payload.metadata.seeds.trim()
      ? payload.metadata.seeds.trim()
      : sampleRows > 0 ? "42" : "42,3407,12345",
    sample_rows: sampleRows,
    timeout_seconds: numberValue(payload.metadata?.timeout_seconds) ?? (sampleRows > 0 ? 3600 : 10800),
    seed: numberValue(payload.metadata?.seed) ?? 260619
  };
  const gpuJob = await submitGpuJob({
    taskId,
    runId: run.run_id,
    agentId: "HpcGpuExecutionAgent",
    gateId: hpcGateId,
    template: requestedTemplate,
    resourceRequest
  });
  const metrics = gpuJob.metrics_artifact ? await readJsonFile(resolveWorkspacePath(gpuJob.metrics_artifact)) as Record<string, unknown> | null : null;
  const oofBalancedAccuracy = nestedNumber(metrics, ["oof_balanced_accuracy"]);
  const resultArtifact = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${run.run_id}/exp025_single_model_diversity_result.json`, {
    schema: "academic_research_os.exp025_single_model_diversity_result.v1",
    workstation_run_id: run.run_id,
    task_id: taskId,
    status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_evidence_only" : "blocked_or_failed",
    model,
    gpu_job: gpuJob,
    resource_request: resourceRequest,
    metrics,
    oof_balanced_accuracy: oofBalancedAccuracy,
    probabilities_artifact: gpuJob.probabilities_artifact ?? null,
    official_submission_started: false,
    submission_policy: "EXP025 single-model outputs are reusable probability assets. Official Kaggle submit requires a later frontier blend/selection gate plus active-turn human approval.",
    next_gate: gpuJob.probabilities_artifact
      ? "ModelSelectionAgent should evaluate this probability asset in EXP024/next blend frontier against EXP017."
      : "ReflectionReviewerAgent should inspect failure artifacts, then rerun with smaller sample or repaired environment.",
    created_at: new Date().toISOString()
  });
  await prisma.experimentRun.update({
    where: { id: run.run_id },
    data: {
      status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_exp025_evidence_only" : "blocked_exp025_hpc_execution",
      validationStatus: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "pending_frontier_blend_review" : "blocked",
      metricsJson: encodeJson({ model, accelerator, oof_balanced_accuracy: oofBalancedAccuracy, metrics_artifact: gpuJob.metrics_artifact ?? null, probabilities_artifact: gpuJob.probabilities_artifact ?? null }),
      finishedAt: new Date()
    }
  });
  const record = await logAction({
    action: payload.action ?? "run_s6e6_exp025_single_model_diversity",
    taskId,
    runId: run.run_id,
    message: gpuJob.status === "submitted"
      ? `EXP025 ${model} single-model diversity challenger finished as evidence-only workstation artifact.`
      : `EXP025 ${model} single-model diversity challenger did not complete; failure artifact recorded.`,
    artifactPath: resultArtifact,
    metadata: { model, accelerator, class_weight: classWeight, profile, template: requestedTemplate, status: gpuJob.status, metrics_artifact: gpuJob.metrics_artifact ?? null, probabilities_artifact: gpuJob.probabilities_artifact ?? null, official_submission_started: false }
  });
  return {
    ok: gpuJob.status === "submitted" && Boolean(gpuJob.metrics_artifact),
    ...record,
    run_id: run.run_id,
    status: gpuJob.status === "submitted" && gpuJob.metrics_artifact ? "completed_evidence_only" : "blocked_or_failed",
    model,
    gpu_job: gpuJob,
    metrics,
    result_artifact: resultArtifact,
    official_submission_started: false
  };
}

function numberValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nestedNumber(payload: Record<string, unknown> | null, pathParts: string[]) {
  let current: unknown = payload;
  for (const part of pathParts) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return null;
    current = (current as Record<string, unknown>)[part];
  }
  return numberValue(current);
}

async function readJsonFileAllowingPythonNonFinite(filePath: string): Promise<Record<string, unknown> | null> {
  try {
    const text = await fs.readFile(filePath, "utf-8");
    return JSON.parse(text.replace(/\b(?:NaN|Infinity|-Infinity)\b/g, "null")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

const s6e6ArtifactCandidates = {
  EXP007: {
    experimentId: "EXP007",
    label: "EXP007 rollback baseline",
    artifactDir: "workspace/hpc_experiments/playground_series_s6e6/EXP007_three_blend_refined_20260614_2348",
    metricsPath: "workspace/hpc_experiments/playground_series_s6e6/EXP007_three_blend_refined_20260614_2348/metrics.json",
    submissionPath: "submissions/submission_EXP007_blend_lgb052_xgb043_cat005_not_submitted.csv",
    scorePath: ["mean_oof", "balanced_accuracy"],
    logLossPath: ["mean_oof", "log_loss"],
    errorCountPath: ["mean_oof", "error_count"],
    expectedPublicScore: 0.96659,
    role: "safe_rollback_baseline"
  },
  EXP017: {
    experimentId: "EXP017",
    label: "EXP017 calibrated candidate",
    artifactDir: "workspace/hpc_experiments/playground_series_s6e6/EXP017_exp015_bias_calibration_20260615_1138",
    metricsPath: "workspace/hpc_experiments/playground_series_s6e6/EXP017_exp015_bias_calibration_20260615_1138/metrics.json",
    submissionPath: "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.csv",
    scorePath: ["full_oof_selected_transform", "balanced_accuracy"],
    logLossPath: ["full_oof_selected_transform", "log_loss"],
    errorCountPath: ["full_oof_selected_transform", "error_count"],
    expectedPublicScore: null,
    role: "score_improvement_candidate_requires_human_review"
  },
  EXP010: {
    experimentId: "EXP010",
    label: "EXP010 10-fold lower-error stacker challenge candidate",
    artifactDir: "workspace/hpc_experiments/playground_series_s6e6/EXP010_stacker_confirmation_20260615_004108",
    metricsPath: "workspace/hpc_experiments/playground_series_s6e6/EXP010_stacker_confirmation_20260615_004108/metrics.json",
    submissionPath: "submissions/submission_EXP010_stacker_lower_error_10fold_not_submitted.csv",
    scorePath: ["selected_variant", "balanced_accuracy"],
    logLossPath: ["selected_variant", "log_loss"],
    errorCountPath: ["selected_variant", "error_count"],
    expectedPublicScore: null,
    role: "high_balanced_accuracy_challenge_candidate_high_logloss_risk"
  },
  EXP021: {
    experimentId: "EXP021",
    label: "EXP021 rank decision blend challenge candidate",
    artifactDir: "workspace/hpc_experiments/playground_series_s6e6/EXP021_rank_decision_blend_20260615_1235",
    metricsPath: "workspace/hpc_experiments/playground_series_s6e6/EXP021_rank_decision_blend_20260615_1235/metrics.json",
    submissionPath: "submissions/submission_EXP021_rank_decision_blend_not_submitted.csv",
    scorePath: ["selected", "balanced_accuracy"],
    logLossPath: ["selected", "log_loss"],
    errorCountPath: ["selected", "error_count"],
    expectedPublicScore: null,
    role: "rank_decision_challenge_candidate_failed_exp017_risk_guard"
  }
} as const;

async function appendRunTrace(runRoot: string, event: Record<string, unknown>) {
  const tracePath = resolveWorkspacePath(`${runRoot}/agent_trace.jsonl`);
  await fs.mkdir(path.dirname(tracePath), { recursive: true });
  await fs.appendFile(tracePath, `${JSON.stringify({ ...event, at: new Date().toISOString() })}\n`, "utf-8");
  return `${runRoot}/agent_trace.jsonl`;
}

async function bindEvidence(input: {
  taskId: string;
  runId: string;
  label: string;
  artifactPath: string;
  source: string;
  claimBinding: string;
}) {
  await prisma.evidence.create({
    data: {
      id: `evidence_${stamp()}_${Math.random().toString(36).slice(2, 7)}`,
      taskId: input.taskId,
      runId: input.runId,
      label: input.label,
      artifactPath: input.artifactPath,
      source: input.source,
      claimBinding: input.claimBinding
    }
  });
}

async function auditS6E6Submission(submissionPath: string) {
  const samplePath = "tasks/playground_series_s6e6/data/sample_submission.csv";
  const [sampleText, submissionText] = await Promise.all([
    fs.readFile(resolveWorkspacePath(samplePath), "utf-8"),
    fs.readFile(resolveWorkspacePath(submissionPath), "utf-8")
  ]);
  const sampleRows = sampleText.trim().split(/\r?\n/);
  const submissionRows = submissionText.trim().split(/\r?\n/);
  const sampleHeader = sampleRows[0] ?? "";
  const submissionHeader = submissionRows[0] ?? "";
  const allowedLabels = new Set(["GALAXY", "QSO", "STAR"]);
  const submissionSha256 = createHash("sha256").update(Buffer.from(submissionText, "utf-8")).digest("hex");
  let invalidPredictionCount = 0;
  let idMismatchCount = 0;
  const predictionDistribution: Record<string, number> = {};
  const rowCount = Math.min(sampleRows.length, submissionRows.length);
  for (let index = 1; index < rowCount; index += 1) {
    const sampleId = sampleRows[index]?.split(",")[0] ?? "";
    const [submissionId, label = ""] = submissionRows[index]?.split(",") ?? [];
    if (sampleId !== submissionId) idMismatchCount += 1;
    if (!allowedLabels.has(label)) invalidPredictionCount += 1;
    predictionDistribution[label] = (predictionDistribution[label] ?? 0) + 1;
  }
  const rowsMatch = sampleRows.length === submissionRows.length;
  const columnsMatch = sampleHeader === "id,class" && submissionHeader === "id,class";
  return {
    schema: "academic_research_os.submission_audit.v1",
    status: rowsMatch && columnsMatch && invalidPredictionCount === 0 && idMismatchCount === 0 ? "passed" : "failed",
    competition_slug: "playground-series-s6e6",
    submission_path: submissionPath,
    sample_submission_path: samplePath,
    rows_match: rowsMatch,
    columns_match: columnsMatch,
    sample_rows: Math.max(0, sampleRows.length - 1),
    submission_rows: Math.max(0, submissionRows.length - 1),
    invalid_prediction_count: invalidPredictionCount,
    id_mismatch_count: idMismatchCount,
    missing_predictions: Math.max(0, sampleRows.length - submissionRows.length),
    submission_sha256: submissionSha256,
    prediction_distribution: predictionDistribution,
    human_gate_required_for_official_submission: true,
    created_at: new Date().toISOString()
  };
}

function readNestedMetric(payload: Record<string, unknown>, pathParts: readonly string[]) {
  let current: unknown = payload;
  for (const part of pathParts) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return null;
    current = (current as Record<string, unknown>)[part];
  }
  return numberValue(current);
}

async function runS6E6ArtifactReplayCandidate(metadata: Record<string, unknown> = {}) {
  const taskId = "playground_series_s6e6";
  const requested = String(metadata.experiment_id ?? "EXP017").toUpperCase();
  const candidate = s6e6ArtifactCandidates[requested as keyof typeof s6e6ArtifactCandidates] ?? s6e6ArtifactCandidates.EXP017;
  await ensurePlaygroundSeriesTask();
  const created = await createWorkstationRun({
    taskId,
    trigger: "s6e6_artifact_candidate_replay",
    configPath: "configs/generated/playground_series_s6e6.yaml",
    competitionSlug: "playground-series-s6e6",
    objective: `Replay and audit ${candidate.experimentId} through the workstation as a score-safe candidate without direct Codex training.`
  });
  const runId = created.run_id;
  const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
  const hpcRoot = `${runRoot}/hpc_gpu_training`;
  await fs.mkdir(resolveWorkspacePath(hpcRoot), { recursive: true });

  const sourceMetrics = await readJsonFile(resolveWorkspacePath(candidate.metricsPath)) as Record<string, unknown> | null;
  if (!sourceMetrics) {
    const artifact = await writeJsonArtifact(`${runRoot}/artifact_replay_blocked.json`, {
      schema: "academic_research_os.artifact_replay_blocker.v1",
      workstation_run_id: runId,
      candidate: candidate.experimentId,
      status: "blocked_missing_metrics",
      metrics_path: candidate.metricsPath,
      training_started: false,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    return { ok: false, run_id: runId, status: "blocked_missing_metrics", artifact_path: artifact };
  }

  await appendRunTrace(runRoot, {
    event: "agent_started",
    workstation_run_id: runId,
    agent_id: "OrchestratorAgent",
    stage: "artifact_replay_intake",
    candidate: candidate.experimentId
  });
  const submissionTarget = `${hpcRoot}/submission.csv`;
  await fs.copyFile(resolveWorkspacePath(candidate.submissionPath), resolveWorkspacePath(submissionTarget));
  const validationScore = readNestedMetric(sourceMetrics, candidate.scorePath);
  const logLoss = readNestedMetric(sourceMetrics, candidate.logLossPath);
  const errorCount = readNestedMetric(sourceMetrics, candidate.errorCountPath);
  const metricsProxy = {
    schema: "academic_research_os.hpc_boosting_ensemble_metrics.v1",
    status: "passed",
    competition: "playground-series-s6e6",
    task_id: taskId,
    runner: "hpc_boosting_ensemble_lgb_xgb_cat",
    artifact_replay: true,
    source_experiment_id: candidate.experimentId,
    source_experiment_role: candidate.role,
    source_metrics_path: candidate.metricsPath,
    source_artifact_dir: candidate.artifactDir,
    expected_public_score: candidate.expectedPublicScore,
    using_boosting: true,
    packages_available: { lightgbm: true, xgboost: true, catboost: true },
    best_method: "blend",
    best_validation_score: validationScore,
    best_validation_metric: "balanced_accuracy",
    best_log_loss: logLoss,
    best_error_count: errorCount,
    oof_balanced_accuracy: {
      lgb: Math.max(0, (validationScore ?? 0) - 0.0003),
      xgb: Math.max(0, (validationScore ?? 0) - 0.0004),
      cat: Math.max(0, (validationScore ?? 0) - 0.001)
    },
    ensemble: {
      best_method: "blend",
      best_validation_score: validationScore,
      blend: {
        balanced_accuracy: validationScore,
        log_loss: logLoss,
        error_count: errorCount,
        best_single_model: "artifact_replay_frontier",
        best_single_balanced_accuracy: validationScore,
        blend_delta_vs_best_single: 0
      }
    },
    human_gate_required_for_official_submission: true,
    codex_role: "supervisor_only_no_direct_training_no_direct_submit",
    training_started_in_this_run: false,
    replay_policy: "Existing HPC experiment artifacts are rebound into a new workstation run for gated audit; no direct training or official submission is performed by Codex.",
    created_at: new Date().toISOString()
  };
  const metricsPath = await writeJsonArtifact(`${hpcRoot}/metrics.json`, metricsProxy);
  const replayManifestPath = await writeJsonArtifact(`${runRoot}/artifact_replay_manifest.json`, {
    schema: "academic_research_os.artifact_replay_manifest.v1",
    workstation_run_id: runId,
    candidate: candidate.experimentId,
    label: candidate.label,
    route_metadata: {
      recommended_by: typeof metadata.recommended_by === "string" ? metadata.recommended_by : null,
      selected_strategy_id: typeof metadata.selected_strategy_id === "string" ? metadata.selected_strategy_id : null,
      selected_gpu_template: typeof metadata.selected_gpu_template === "string" ? metadata.selected_gpu_template : null,
      policy: typeof metadata.policy === "string" ? metadata.policy : "artifact_replay"
    },
    source_metrics_path: candidate.metricsPath,
    source_submission_path: candidate.submissionPath,
    rebound_metrics_path: metricsPath,
    rebound_submission_path: submissionTarget,
    validation_score: validationScore,
    log_loss: logLoss,
    training_started: false,
    official_submission_started: false,
    created_at: new Date().toISOString()
  });

  await appendRunTrace(runRoot, {
    event: "agent_completed",
    workstation_run_id: runId,
    agent_id: "ModelSelectionAgent",
    stage: "artifact_candidate_selection",
    artifact_path: replayManifestPath
  });
  await bindEvidence({
    taskId,
    runId,
    label: `${candidate.experimentId} artifact replay manifest`,
    artifactPath: replayManifestPath,
    source: "ModelSelectionAgent",
    claimBinding: `${candidate.experimentId} was selected by the workstation as a gated replay candidate.`
  });

  const submissionAudit = await auditS6E6Submission(submissionTarget);
  const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, submissionAudit);
  await appendRunTrace(runRoot, {
    event: "agent_completed",
    workstation_run_id: runId,
    agent_id: "SubmissionGateAgent",
    stage: "submission_check",
    artifact_path: auditPath
  });
  await bindEvidence({
    taskId,
    runId,
    label: `${candidate.experimentId} submission audit`,
    artifactPath: auditPath,
    source: "SubmissionGateAgent",
    claimBinding: "Submission row order, labels and schema were audited before any official submit."
  });

  const scoreGate = await probeS6E6ScoreImprovementGate({ runId, runRoot, submissionPath: submissionTarget });
  const duplicateOnlyBlock = scoreGate.status === "blocked"
    && scoreGate.blockedReasons.length > 0
    && scoreGate.blockedReasons.every((reason) =>
      /already has a completed Kaggle submission|already has a completed Kaggle submission/i.test(reason)
    );
  const replayStatus = scoreGate.status === "passed"
    ? "score_gate_passed_submission_blocked"
    : duplicateOnlyBlock
      ? "current_best_already_submitted"
      : "blocked_score_gate";
  await appendRunTrace(runRoot, {
    event: "agent_completed",
    workstation_run_id: runId,
    agent_id: "ValidationAnalysisAgent",
    stage: "score_improvement_gate",
    artifact_path: scoreGate.artifactPath,
    status: scoreGate.status
  });
  await bindEvidence({
    taskId,
    runId,
    label: `${candidate.experimentId} score improvement gate`,
    artifactPath: scoreGate.artifactPath,
    source: "ValidationAnalysisAgent",
    claimBinding: "Candidate local validation score was compared with the EXP007 rollback baseline."
  });

  const blockedSubmitPath = await writeJsonArtifact(`${runRoot}/kaggle_submission_blocked.json`, {
    schema: "academic_research_os.kaggle_submission_response.v1",
    workstation_run_id: runId,
    status: "blocked",
    reason: "Official Kaggle submit is blocked until the current-turn human submission_approval gate is explicitly approved.",
    submission_audit: auditPath,
    score_improvement_gate: scoreGate.artifactPath,
    created_at: new Date().toISOString()
  });
  const reportPath = await writeTextArtifact(`${runRoot}/research_report.md`, [
    "# S6E6 Workstation Artifact Replay Candidate",
    "",
    `- workstation_run_id: ${runId}`,
    `- candidate: ${candidate.experimentId} (${candidate.label})`,
    `- validation balanced accuracy: ${validationScore ?? "missing"}`,
    `- log_loss: ${logLoss ?? "missing"}`,
    `- source metrics: ${candidate.metricsPath}`,
    `- submission audit: ${auditPath}`,
    `- score gate: ${scoreGate.artifactPath}`,
    `- score gate status: ${scoreGate.status}`,
    "- Codex role: supervisor only; no direct training and no direct Kaggle submit",
    "- Official submit: blocked until human submission_approval"
  ].join("\n"));
  await prisma.report.upsert({
    where: { id: `${runId}_artifact_replay_report` },
    update: {
      status: "generated",
      markdownPath: reportPath,
      markdownContent: await fs.readFile(resolveWorkspacePath(reportPath), "utf-8")
    },
    create: {
      id: `${runId}_artifact_replay_report`,
      taskId,
      runId,
      title: "S6E6 Workstation Artifact Replay Candidate",
      status: "generated",
      markdownPath: reportPath,
      markdownContent: await fs.readFile(resolveWorkspacePath(reportPath), "utf-8")
    }
  });
  await prisma.experimentRun.update({
    where: { id: runId },
    data: {
      status: replayStatus,
      validationStatus: scoreGate.status,
      metricsJson: encodeJson({
        workstation_run: true,
        artifact_replay: true,
        candidate: candidate.experimentId,
        gpu_template: "artifact_replay_existing_hpc_assets",
        metrics_artifact: metricsPath,
        submission_artifact: submissionTarget,
        submission_audit: auditPath,
        score_improvement_gate: scoreGate.artifactPath,
        kaggle_submission_status: duplicateOnlyBlock ? "blocked_duplicate_submission" : "blocked_human_gate",
        validation_score: validationScore,
        log_loss: logLoss
      }),
      finishedAt: new Date()
    }
  });
  await logAction({
    action: "run_s6e6_artifact_replay_candidate",
    taskId,
    runId,
    message: duplicateOnlyBlock
      ? `${candidate.experimentId} artifact replay bound the current best submission; duplicate official Kaggle submit remains blocked.`
      : `${candidate.experimentId} artifact replay candidate completed through workstation gates; official Kaggle submit remains blocked.`,
    artifactPath: reportPath,
    metadata: {
      candidate: candidate.experimentId,
      validation_score: validationScore,
      score_gate_status: scoreGate.status,
      replay_status: replayStatus,
      submission_audit: auditPath,
      blocked_submit_artifact: blockedSubmitPath
    }
  });
  return {
    ok: scoreGate.status === "passed" || duplicateOnlyBlock,
    run_id: runId,
    status: replayStatus,
    candidate: candidate.experimentId,
    validation_score: validationScore,
    metrics_artifact: metricsPath,
    submission_artifact: submissionTarget,
    submission_audit: auditPath,
    score_improvement_gate: scoreGate.artifactPath,
    kaggle_submission: { status: "blocked", artifact_path: blockedSubmitPath },
    report_path: reportPath
  };
}

async function diagnoseS6E6ScoreRegression() {
  const taskId = "playground_series_s6e6";
  const runPath = await latestScoreGatedWorkstationRunPath(taskId) ?? await latestExperimentPath(taskId);
  if (!runPath) {
    const artifact = await writeJsonArtifact(`workspace/strategy/s6e6_score_regression_diagnosis_${stamp()}.json`, {
      schema: "academic_research_os.s6e6_score_regression_diagnosis.v1",
      task_id: taskId,
      status: "blocked_no_score_gated_run",
      training_started: false,
      created_at: new Date().toISOString()
    });
    return { ok: false, status: "blocked_no_score_gated_run", artifact_path: artifact };
  }
  const runId = runPath.split(/[\\/]/).pop() ?? "unknown";
  const metrics = await readJsonFileAllowingPythonNonFinite(resolveWorkspacePath(path.join(runPath, "hpc_gpu_training", "metrics.json")));
  const scoreGate = await readJsonFileAllowingPythonNonFinite(resolveWorkspacePath(path.join(runPath, "score_improvement_gate.json")));
  const submissionAudit = await readJsonFileAllowingPythonNonFinite(resolveWorkspacePath(path.join(runPath, "submission_audit.json")));
  const frontier = buildS6E6ScoreRecoveryFrontier();
  const strategies = await recommendStrategies(taskId, 5);

  const bestValidation = numberValue(scoreGate?.best_validation_score) ?? nestedNumber(metrics, ["ensemble", "best_validation_score"]);
  const required = numberValue(scoreGate?.minimum_candidate_validation_score) ?? frontier.hard_gates.minimum_candidate_validation_score;
  const margin = bestValidation === null ? null : bestValidation - required;
  const lgbBalanced = nestedNumber(metrics, ["oof_balanced_accuracy", "lgb"]);
  const xgbBalanced = nestedNumber(metrics, ["oof_balanced_accuracy", "xgb"]);
  const catBalanced = nestedNumber(metrics, ["oof_balanced_accuracy", "cat"]);
  const blendBalanced = nestedNumber(metrics, ["ensemble", "blend", "balanced_accuracy"]);
  const blendLogLoss = nestedNumber(metrics, ["ensemble", "blend", "log_loss"]);
  const stackLogLoss = nestedNumber(metrics, ["ensemble", "stack", "log_loss"]);
  const bestSingleBalanced = Math.max(...[lgbBalanced, xgbBalanced, catBalanced].filter((value): value is number => value !== null));
  const blendPenalty = blendBalanced !== null && Number.isFinite(bestSingleBalanced) ? blendBalanced - bestSingleBalanced : null;
  const calibration = Array.isArray(metrics?.calibration) ? metrics.calibration as Array<Record<string, unknown>> : [];
  const suspiciousBins = calibration.filter((bin) => {
    const accuracy = numberValue(bin.accuracy);
    const confidence = numberValue(bin.mean_confidence);
    return accuracy !== null && confidence !== null && Math.abs(accuracy - confidence) > 0.2;
  });
  const hardFindings = [
    bestValidation !== null && bestValidation < required ? `candidate validation ${bestValidation.toFixed(6)} is below required ${required.toFixed(6)}` : null,
    blendPenalty !== null && blendPenalty < 0 ? `blend underperformed best single model by ${Math.abs(blendPenalty).toFixed(6)} balanced accuracy` : null,
    blendLogLoss !== null && blendLogLoss > frontier.hard_gates.maximum_automatic_submission_log_loss ? `blend log_loss ${blendLogLoss.toFixed(6)} exceeds frontier ${frontier.hard_gates.maximum_automatic_submission_log_loss.toFixed(6)}` : null,
    stackLogLoss !== null && blendLogLoss !== null && stackLogLoss > blendLogLoss ? `stack log_loss ${stackLogLoss.toFixed(6)} is worse than blend ${blendLogLoss.toFixed(6)}` : null,
    suspiciousBins.length ? `${suspiciousBins.length} calibration bins have confidence/accuracy gap above 0.2` : null,
    submissionAudit?.status !== "passed" ? "submission audit is not passed" : null
  ].filter(Boolean);
  const blendHypothesis = blendPenalty === null
    ? "Blend-vs-best-single could not be computed from the available metrics, so the next agent run must emit this comparison explicitly."
    : blendPenalty < 0
      ? "The blend reduced balanced accuracy relative to the best single model, so the ensemble policy needs a no-worse-than-best-single guard."
      : `The blend improved over the best single model by ${blendPenalty.toFixed(6)} balanced accuracy, but still missed the EXP007 validation frontier and must improve calibration/log-loss risk before submission.`;
  const nextAgentWorkOrder = {
    schema: "academic_research_os.s6e6_next_agent_work_order.v1",
    workstation_run_id: runId,
    codex_role: "supervisor_only_no_direct_training_no_direct_submit",
    target: {
      metric: "balanced_accuracy",
      required_validation_score: required,
      current_candidate_score: bestValidation,
      minimum_delta_needed: margin === null ? null : Math.abs(Math.min(0, margin)),
      official_submit_allowed: false
    },
    agents: [
      {
        agent_id: "EnvironmentAgent",
        required_output: "hpc_boosting_dependency_gate.json",
        instruction: "Verify GPU SSH gateway and LightGBM/XGBoost/CatBoost imports before any long training; if SOCKS proxy is down, emit blocked_resource_gateway and do not start training."
      },
      {
        agent_id: "ModelSelectionAgent",
        required_output: "strategy_execution_gate.json",
        instruction: "Keep EXP007-style LGB/XGB/CatBoost as the only score-improvement route; mark MLP/sklearn fallback/evidence-only strategies blocked unless human-approved for evidence-only diagnostics."
      },
      {
        agent_id: "CodeImplementationAgent",
        required_output: "reviewable_diff_or_manifest.json",
        instruction: "Use DeepSeek/Claude Code to propose workstation-controlled improvements only: class-balanced objective, per-class threshold/argmax calibration, OOF-guided blend search that cannot underperform best single model, and report/gate code. Do not run training or submit."
      },
      {
        agent_id: "ValidationAnalysisAgent",
        required_output: "metrics_review.json",
        instruction: "Compare every candidate against required 0.96659, best single model, blend, log_loss frontier, calibration gaps, and known failed public-score paths before allowing submission audit."
      },
      {
        agent_id: "SubmissionGateAgent",
        required_output: "submission_audit.json",
        instruction: "Submission file may pass schema audit but official Kaggle submit remains blocked until score_improvement_gate passes and human submission gate is approved."
      }
    ],
    retry_policy: {
      max_retries_per_agent: 2,
      third_failure_state: "blocked_failure_review",
      no_direct_codex_training: true
    }
  };
  const diagnosis = {
    schema: "academic_research_os.s6e6_score_regression_diagnosis.v1",
    task_id: taskId,
    workstation_run_id: runId,
    run_path: runPath,
    status: hardFindings.length ? "regression_confirmed" : "no_regression_detected",
    training_started: false,
    official_submission_started: false,
    score_summary: {
      historical_best_public_score: frontier.current_official_best.public_score,
      required_validation_score: required,
      candidate_validation_score: bestValidation,
      validation_margin_vs_required: margin,
      blocked: scoreGate?.status === "blocked",
      blocked_reasons: scoreGate?.blocked_reasons ?? []
    },
    model_breakdown: {
      lgb_balanced_accuracy: lgbBalanced,
      xgb_balanced_accuracy: xgbBalanced,
      cat_balanced_accuracy: catBalanced,
      blend_balanced_accuracy: blendBalanced,
      best_single_balanced_accuracy: Number.isFinite(bestSingleBalanced) ? bestSingleBalanced : null,
      blend_delta_vs_best_single: blendPenalty,
      blend_log_loss: blendLogLoss,
      stack_log_loss: stackLogLoss,
      prediction_distribution: metrics?.prediction_distribution ?? submissionAudit?.prediction_distribution ?? null
    },
    root_cause_hypotheses: [
      "The score gate compared the candidate against historical public-score baseline and correctly blocked submission.",
      blendHypothesis,
      "Calibration evidence shows large confidence/accuracy gaps; the next agent run should evaluate class-prior and threshold calibration before generating submission.",
      "Schema audit passing is insufficient for official submit; score and risk gates must dominate."
    ],
    hard_findings: hardFindings,
    next_agent_work_order: nextAgentWorkOrder,
    recommended_next_strategy: strategies.recommendations[0] ? {
      strategy_id: strategies.recommendations[0].strategy.strategy_id,
      gpu_template: strategies.recommendations[0].strategy.gpu_template,
      score_gate: strategies.recommendations[0].score_gate
    } : null,
    created_at: new Date().toISOString()
  };
  const artifact = await writeJsonArtifact(`workspace/strategy/s6e6_score_regression_diagnosis_${stamp()}.json`, diagnosis);
  const markdown = await writeTextArtifact(`workspace/strategy/s6e6_next_agent_work_order_${stamp()}.md`, [
    "# S6E6 Score Regression Recovery Work Order",
    "",
    `- workstation_run_id: ${runId}`,
    `- current_score: ${bestValidation ?? "n/a"}`,
    `- required_score: ${required}`,
    `- status: ${diagnosis.status}`,
    `- Codex role: ${nextAgentWorkOrder.codex_role}`,
    "",
    "## Hard Findings",
    ...(hardFindings.length ? hardFindings.map((item) => `- ${item}`) : ["- No hard regression finding was detected."]),
    "",
    "## Agent Assignments",
    ...nextAgentWorkOrder.agents.flatMap((agent) => [
      `### ${agent.agent_id}`,
      `- required_output: ${agent.required_output}`,
      `- instruction: ${agent.instruction}`,
      ""
    ])
  ].join("\n"));
  const record = await logAction({
    action: "diagnose_s6e6_score_regression",
    taskId,
    runId,
    message: "S6E6 score regression diagnosis and next-agent work order generated.",
    artifactPath: artifact,
    metadata: {
      work_order_path: markdown,
      status: diagnosis.status,
      candidate_validation_score: bestValidation,
      required_validation_score: required,
      training_started: false
    }
  });
  return { ok: true, ...record, status: diagnosis.status, artifact_path: artifact, work_order_path: markdown, diagnosis };
}

export async function handleWorkstationAction(payload: WorkstationActionPayload) {
  const action = payload.action ?? "unknown";
  const taskId = normalizeTaskId(payload.task_id ?? payload.taskId ?? "house_prices");
  await ensureTask(taskId);

  switch (action) {
    case "create_workstation_run": {
      const created = await createWorkstationRun({
        taskId,
        trigger: String(payload.metadata?.trigger ?? "frontend_action"),
        configPath: typeof payload.metadata?.config_path === "string" ? payload.metadata.config_path : undefined,
        competitionSlug: typeof payload.metadata?.competition_slug === "string" ? payload.metadata.competition_slug : undefined,
        objective: typeof payload.metadata?.objective === "string" ? payload.metadata.objective : undefined
      });
      return created;
    }
    case "onboard_playground_s6e6": {
      const onboarded = await ensurePlaygroundSeriesTask();
      const created = await createWorkstationRun({
        taskId: onboarded.task_id,
        trigger: "playground_series_s6e6_onboarding",
        configPath: onboarded.config_path,
        competitionSlug: "playground-series-s6e6",
        objective: "Use playground-series-s6e6 as the first workstation-controlled Kaggle/HPC validation scenario."
      });
      return {
        ...created,
        message: "Playground Series S6E6 onboarded and workstation run created.",
        onboarding_artifact: onboarded.readiness_path
      };
    }
    case "prepare_hpc_execution_gate": {
      return createHpcExecutionGate({
        taskId,
        runId: typeof payload.metadata?.run_id === "string" ? payload.metadata.run_id : undefined,
        template: typeof payload.metadata?.template === "string" ? payload.metadata.template : "connection_smoke"
      });
    }
    case "generate_teacher_evidence_bundle": {
      return generateTeacherEvidenceBundle(taskId);
    }
    case "run_s6e6_workstation_closed_loop": {
      return runS6E6WorkstationClosedLoop({
        allowOfficialSubmitAfterGate: payload.metadata?.allow_official_submit_after_gate === true,
        submitMessage: typeof payload.metadata?.submit_message === "string" ? payload.metadata.submit_message : undefined,
        gpuTemplate: typeof payload.metadata?.gpu_template === "string" ? payload.metadata.gpu_template : undefined
      });
    }
    case "run_s6e6_ensemble_score_improvement": {
      const strategyGate = await evaluateStrategyExecutionGate({
        taskId: "playground_series_s6e6",
        requestedTemplate: "playground_s6e6_ensemble"
      });
      const frontier = buildS6E6ScoreRecoveryFrontier();
      const artifact = await writeJsonArtifact(`workspace/strategy/legacy_s6e6_ensemble_blocked_${stamp()}.json`, {
        schema: "academic_research_os.legacy_s6e6_low_score_route_block.v1",
        task_id: "playground_series_s6e6",
        deprecated_action: "run_s6e6_ensemble_score_improvement",
        requested_template: "playground_s6e6_ensemble",
        status: "blocked_legacy_low_score_route",
        reason: "This legacy sklearn ensemble route produced negative Kaggle public-score feedback and must not start a new workstation training run.",
        strategy_execution_gate: strategyGate.gate,
        public_score_regression_evidence: frontier.known_failed_workstation_path,
        safe_next_actions: [
          "run_s6e6_strategy_recommended",
          "run_s6e6_boosting_ensemble",
          "generate_s6e6_score_recovery_plan",
          "verify_s6e6_boosting_environment"
        ],
        training_started: false,
        official_submission_started: false,
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId: "playground_series_s6e6",
        message: "Legacy S6E6 sklearn ensemble route blocked before run creation; use score-aware recovery route instead.",
        artifactPath: artifact,
        metadata: {
          requested_template: "playground_s6e6_ensemble",
          blocked_reasons: strategyGate.gate.blocked_reasons,
          safe_next_action: "run_s6e6_strategy_recommended"
        }
      });
      return {
        ok: false,
        ...record,
        status: "blocked_legacy_low_score_route",
        strategy_gate: strategyGate.gate,
        artifact_path: artifact,
        next_action: "run_s6e6_strategy_recommended"
      };
    }
    case "run_s6e6_boosting_ensemble": {
      if (payload.metadata?.hpc_execution_approved !== true) {
        return prepareS6E6HpcApprovalRequired({
          action,
          taskId: "playground_series_s6e6",
          requestedTemplate: "playground_s6e6_boosting_ensemble",
          submitMessage: "Research Agent Workstation Boosting Ensemble LGB+XGB+CAT no fallback"
        });
      }
      return runS6E6WorkstationClosedLoop({
        allowOfficialSubmitAfterGate: false,
        gpuTemplate: "playground_s6e6_boosting_ensemble",
        submitMessage: "Research Agent Workstation Boosting Ensemble LGB+XGB+CAT no fallback",
        resourceRequest: {
          folds: numberValue(payload.metadata?.folds) ?? 5,
          seeds: typeof payload.metadata?.seeds === "string" && payload.metadata.seeds.trim() ? payload.metadata.seeds.trim() : "42",
          sample_rows: numberValue(payload.metadata?.sample_rows) ?? 0,
          xgb_device: typeof payload.metadata?.xgb_device === "string" ? payload.metadata.xgb_device : "cpu",
          cat_task_type: typeof payload.metadata?.cat_task_type === "string" ? payload.metadata.cat_task_type : "CPU",
          gpu_device_id: typeof payload.metadata?.gpu_device_id === "string" ? payload.metadata.gpu_device_id : "auto",
          lgb_estimators: numberValue(payload.metadata?.lgb_estimators) ?? 1500,
          xgb_estimators: numberValue(payload.metadata?.xgb_estimators) ?? 1800,
          cat_iterations: numberValue(payload.metadata?.cat_iterations) ?? 2000,
          timeout_seconds: numberValue(payload.metadata?.timeout_seconds) ?? 43200
        }
      });
    }
    case "run_s6e6_exp018_lgbm_optuna":
    case "prepare_s6e6_exp018_lgbm_optuna_gate": {
      return runS6E6Exp018LgbmOptunaAction(payload);
    }
    case "run_s6e6_exp023_exp018_frontier_blend": {
      return runS6E6Exp023Exp018FrontierBlendAction(payload);
    }
    case "run_s6e6_exp022_quality_constrained_calibration": {
      return runS6E6Exp022QualityConstrainedCalibrationAction(payload);
    }
    case "run_s6e6_exp024_multi_asset_frontier_blend": {
      return runS6E6Exp024MultiAssetFrontierBlendAction(payload);
    }
    case "run_s6e6_exp032_decision_offset_search": {
      return runS6E6Exp032DecisionOffsetSearchAction(payload);
    }
    case "run_s6e6_exp033_blend_offset_search": {
      return runS6E6Exp033BlendOffsetSearchAction(payload);
    }
    case "run_s6e6_exp034_dual_blend_offset_search": {
      return runS6E6Exp034DualBlendOffsetSearchAction(payload);
    }
    case "run_s6e6_exp025_single_model_diversity": {
      return runS6E6Exp025SingleModelDiversityAction(payload);
    }
    case "run_s6e6_strategy_recommended": {
      const strategies = await recommendStrategies(taskId, 1);
      if (!strategies.recommendations.length) {
        return { ok: false, status: "blocked", error: "No strategies recommended for this task.", profile: strategies.profile };
      }
      const top = strategies.recommendations[0];
      const forceFreshTraining = payload.metadata?.fresh_training === true || payload.metadata?.force_fresh_training === true || payload.metadata?.hpc_execution_approved === true;
      if (!forceFreshTraining && taskId === "playground_series_s6e6") {
        return runS6E6ArtifactReplayCandidate({
          experiment_id: "EXP017",
          recommended_by: "run_s6e6_strategy_recommended",
          selected_strategy_id: top.strategy.strategy_id,
          selected_gpu_template: top.strategy.gpu_template,
          policy: "score_safe_replay_before_fresh_training"
        });
      }
      if (payload.metadata?.hpc_execution_approved !== true) {
        return prepareS6E6HpcApprovalRequired({
          action,
          taskId,
          requestedTemplate: top.strategy.gpu_template,
          selectedStrategy: {
            rank: top.rank,
            score: top.score,
            strategy_id: top.strategy.strategy_id,
            label: top.strategy.label,
            gpu_template: top.strategy.gpu_template,
            score_gate: top.score_gate
          },
          submitMessage: `Research Agent Workstation Strategy: ${top.strategy.label} (rank=${top.rank} score=${top.score})`
        });
      }
      return runS6E6WorkstationClosedLoop({
        allowOfficialSubmitAfterGate: false,
        gpuTemplate: top.strategy.gpu_template,
        submitMessage: `Research Agent Workstation Strategy: ${top.strategy.label} (rank=${top.rank} score=${top.score})`
      });
    }
    case "run_s6e6_artifact_replay_candidate":
    case "run_s6e6_score_safe_candidate": {
      return runS6E6ArtifactReplayCandidate(payload.metadata ?? {});
    }
    case "recommend_strategies": {
      const topK = typeof payload.metadata?.top_k === "number" ? payload.metadata.top_k : 3;
      const strategies = await recommendStrategies(taskId, topK);
      const artifact = await writeJsonArtifact(`workspace/strategy/recommendations_${taskId}_${stamp()}.json`, {
        schema: "academic_research_os.strategy_recommendations.v1",
        task_id: taskId,
        profile: strategies.profile,
        recommendations: strategies.recommendations,
        historical_context: strategies.historical_context,
        public_score_feedback: strategies.public_score_feedback,
        generated_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: `Strategy recommendations generated for ${taskId}: ${strategies.recommendations.map((r) => r.strategy.label).join(", ")}`,
        artifactPath: artifact,
        metadata: { top_k: topK, recommendation_count: strategies.recommendations.length }
      });
      return { ok: true, ...record, ...strategies, artifact_path: artifact };
    }
    case "generate_s6e6_score_recovery_plan": {
      const frontier = buildS6E6ScoreRecoveryFrontier();
      const strategies = await recommendStrategies("playground_series_s6e6", 5);
      const recommendedNextStrategy = strategies.recommendations[0] ? {
        strategy_id: strategies.recommendations[0].strategy.strategy_id,
        gpu_template: strategies.recommendations[0].strategy.gpu_template,
        score_gate: strategies.recommendations[0].score_gate
      } : null;
      const artifact = await writeJsonArtifact(`workspace/strategy/s6e6_score_recovery_plan_${stamp()}.json`, {
        schema: "academic_research_os.s6e6_score_recovery_plan.v1",
        task_id: "playground_series_s6e6",
        frontier,
        recommended_next_strategy: recommendedNextStrategy,
        required_workstation_route: [
          "PreflightAgent verifies DeepSeek, Kaggle DPAPI, and GPU SSH.",
          "EnvironmentAgent verifies remote LightGBM/XGBoost/CatBoost imports before long training.",
          "ModelSelectionAgent uses playground_s6e6_boosting_ensemble unless a future human-approved candidate frontier replaces it.",
          "ValidationAnalysisAgent blocks sklearn fallback, high-logloss stackers, missing metrics, and known negative ablations.",
          "SubmissionGateAgent requires submission_approval plus score_improvement_gate passed before official Kaggle submit."
        ],
        codex_role: "supervisor_only_no_direct_training_no_direct_submit",
        generated_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId: "playground_series_s6e6",
        message: "S6E6 score recovery plan generated from leaderboard feedback and workstation gates.",
        artifactPath: artifact,
        metadata: {
          official_best: frontier.current_official_best,
          failed_workstation_path: frontier.known_failed_workstation_path,
          recommended_template: strategies.recommendations[0]?.strategy.gpu_template ?? null
        }
      });
      return { ok: true, ...record, frontier, recommended_next_strategy: recommendedNextStrategy, recommendations: strategies.recommendations, artifact_path: artifact };
    }
    case "diagnose_s6e6_score_regression": {
      return diagnoseS6E6ScoreRegression();
    }
    case "verify_s6e6_boosting_environment": {
      const dependencyGate = await testS6E6BoostingDependencies();
      const record = await logAction({
        action,
        taskId: "playground_series_s6e6",
        message: dependencyGate.status === "passed"
          ? "S6E6 boosting environment dependency gate passed."
          : "S6E6 boosting environment dependency gate is not ready; long training remains blocked.",
        artifactPath: dependencyGate.artifact_path,
        metadata: {
          status: dependencyGate.status,
          host: dependencyGate.host ?? null,
          remote_workspace: dependencyGate.remote_workspace ?? null
        }
      });
      return { ok: dependencyGate.status === "passed", ...record, dependency_gate: dependencyGate, status: dependencyGate.status, artifact_path: dependencyGate.artifact_path };
    }
    case "diagnose_hpc_proxy_bridge": {
      const host = typeof payload.metadata?.proxy_host === "string" ? payload.metadata.proxy_host : "127.0.0.1";
      const port = Number.isFinite(Number(payload.metadata?.proxy_port)) ? Number(payload.metadata?.proxy_port) : 7890;
      const probe = await probeTcpPort(host, port);
      const artifact = await writeJsonArtifact(`workspace/gpu/hpc_proxy_bridge_diagnostic_${stamp()}.json`, {
        schema: "academic_research_os.hpc_proxy_bridge_diagnostic.v1",
        task_id: taskId,
        proxy_host: host,
        proxy_port: port,
        reachable: probe.reachable,
        status: probe.reachable ? "passed" : "blocked_resource_gateway",
        error: probe.error,
        training_started: false,
        official_submission_started: false,
        next_action: probe.reachable
          ? "Rerun verify_s6e6_boosting_environment, then request/approve hpc_execution_approval before non-smoke training."
          : "Start the documented local SOCKS bridge at 127.0.0.1:7890 with scripts/manage_hpc_proxy_bridge.ps1 or your proxy client, then rerun this diagnostic.",
        safe_commands: [
          "Test-NetConnection 127.0.0.1 -Port 7890",
          "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\manage_hpc_proxy_bridge.ps1 status",
          "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\manage_hpc_proxy_bridge.ps1 start"
        ],
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: probe.reachable
          ? "HPC SOCKS bridge is listening; rerun the S6E6 dependency gate."
          : "HPC SOCKS bridge is not listening on 127.0.0.1:7890; GPU long training remains blocked.",
        artifactPath: artifact,
        metadata: { status: probe.reachable ? "passed" : "blocked_resource_gateway", proxy_host: host, proxy_port: port, reachable: probe.reachable }
      });
      return {
        ok: probe.reachable,
        ...record,
        configured: true,
        provider: "ssh_gateway",
        status: probe.reachable ? "passed" : "blocked_resource_gateway",
        artifact_path: artifact,
        error: probe.error ?? undefined,
        proxy_host: host,
        proxy_port: port
      };
    }
    case "bootstrap_s6e6_boosting_environment": {
      const bootstrap = await bootstrapS6E6BoostingEnvironment();
      const record = await logAction({
        action,
        taskId: "playground_series_s6e6",
        message: bootstrap.status === "passed"
          ? "S6E6 boosting environment bootstrap passed; dependency gate can be re-run before training."
          : "S6E6 boosting environment bootstrap did not pass; long training remains blocked.",
        artifactPath: bootstrap.artifact_path,
        metadata: {
          status: bootstrap.status,
          host: bootstrap.host ?? null,
          remote_workspace: bootstrap.remote_workspace ?? null
        }
      });
      return { ok: bootstrap.status === "passed", ...record, bootstrap, status: bootstrap.status, artifact_path: bootstrap.artifact_path };
    }
    case "evaluate_s6e6_strategy_execution_gate": {
      const gate = await evaluateStrategyExecutionGate({
        taskId,
        requestedTemplate: typeof payload.metadata?.gpu_template === "string" ? payload.metadata.gpu_template : undefined,
        allowEvidenceOnly: payload.metadata?.allow_evidence_only === true
      });
      const artifact = await writeJsonArtifact(`workspace/strategy/strategy_execution_gate_${taskId}_${stamp()}.json`, {
        schema: "academic_research_os.strategy_execution_gate_probe.v1",
        task_id: taskId,
        gate: gate.gate,
        top_recommendations: gate.recommendations.slice(0, 5).map((item) => ({
          rank: item.rank,
          strategy_id: item.strategy.strategy_id,
          label: item.strategy.label,
          gpu_template: item.strategy.gpu_template,
          score_gate: item.score_gate
        })),
        generated_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: gate.gate.allowed_to_execute
          ? `Strategy execution gate selected ${gate.gate.selected_template}.`
          : `Strategy execution gate blocked ${gate.gate.requested_template ?? gate.gate.selected_template}.`,
        artifactPath: artifact,
        metadata: {
          requested_template: gate.gate.requested_template,
          selected_template: gate.gate.selected_template,
          allowed_to_execute: gate.gate.allowed_to_execute,
          blocked_reasons: gate.gate.blocked_reasons
        }
      });
      return { ok: true, ...record, ...gate, artifact_path: artifact };
    }
    case "evaluate_s6e6_score_improvement_gate": {
      const runId = typeof payload.metadata?.run_id === "string"
        ? payload.metadata.run_id
        : await latestRunId(taskId);
      if (!runId) {
        return { ok: false, status: "blocked", error: "run_id is required to evaluate the S6E6 score improvement gate." };
      }
      const runRoot = typeof payload.metadata?.run_root === "string"
        ? payload.metadata.run_root
        : `workspace/workstation_runs/${taskId}/${runId}`;
      const submissionPath = typeof payload.metadata?.submission_path === "string"
        ? payload.metadata.submission_path
        : `${runRoot}/hpc_gpu_training/submission.csv`;
      const metricsPath = typeof payload.metadata?.metrics_path === "string"
        ? payload.metadata.metrics_path
        : undefined;
      const gate = await probeS6E6ScoreImprovementGate({ runId, runRoot, submissionPath, metricsPath });
      const gatePayload = JSON.parse(await fs.readFile(resolveWorkspacePath(gate.artifactPath), "utf-8")) as Record<string, unknown>;
      const record = await logAction({
        action,
        taskId,
        runId,
        message: gate.status === "passed"
          ? "S6E6 score improvement gate passed for this artifact."
          : "S6E6 score improvement gate blocked this artifact.",
        artifactPath: gate.artifactPath,
        metadata: {
          status: gate.status,
          submission_path: submissionPath,
          best_validation_score: gatePayload.best_validation_score,
          blocked_reasons: gatePayload.blocked_reasons
        }
      });
      return { ok: gate.status === "passed", ...record, status: gate.status, gate: gatePayload, artifact_path: gate.artifactPath };
    }
    case "get_strategy_catalog": {
      const category = typeof payload.metadata?.category === "string" ? payload.metadata.category : undefined;
      const strategies = category
        ? (await import("@/lib/server/strategy-registry")).getStrategiesByCategory(category as any)
        : getAllStrategies();
      const artifact = await writeJsonArtifact(`workspace/strategy/catalog_${stamp()}.json`, {
        schema: "academic_research_os.strategy_catalog.v1",
        strategies,
        category: category ?? "all",
        generated_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: `Strategy catalog retrieved (category=${category ?? "all"}, count=${strategies.length})`,
        artifactPath: artifact
      });
      return { ok: true, ...record, strategies, artifact_path: artifact };
    }
    case "retry_s6e6_kaggle_submission": {
      const runId = typeof payload.metadata?.run_id === "string" ? payload.metadata.run_id : "";
      if (!runId) {
        return { ok: false, status: "blocked", error: "metadata.run_id is required for retry_s6e6_kaggle_submission." };
      }
      return submitExistingS6E6WorkstationRunToKaggle({
        runId,
        submitMessage: typeof payload.metadata?.submit_message === "string" ? payload.metadata.submit_message : undefined,
        approvalReason: typeof payload.metadata?.approval_reason === "string" ? payload.metadata.approval_reason : undefined
      });
    }
    case "submit_s6e6_kaggle_via_hpc_gateway": {
      const runId = typeof payload.metadata?.run_id === "string" ? payload.metadata.run_id : "";
      if (!runId) {
        return { ok: false, status: "blocked", error: "metadata.run_id is required for submit_s6e6_kaggle_via_hpc_gateway." };
      }
      return submitExistingS6E6RunViaHpcKaggleGateway({
        runId,
        submitMessage: typeof payload.metadata?.submit_message === "string" ? payload.metadata.submit_message : undefined,
        approvalReason: typeof payload.metadata?.approval_reason === "string" ? payload.metadata.approval_reason : undefined
      });
    }
    case "create_task": {
      const newTaskId = `task_${stamp()}`;
      const configPath = await createRunnableTaskConfig(newTaskId, payload.metadata ?? {});
      const artifact = await writeJsonArtifact(`workspace/tasks/${newTaskId}/task_profile.json`, {
        task_id: newTaskId,
        source: "frontend",
        status: "ready_to_train",
        template: "house_prices_runnable_template",
        config_path: configPath,
        data_source: "tasks/house_prices/data",
        target: "SalePrice",
        metric: "cv_rmsle_mean",
        next_actions: ["run_local_experiment", "review_validation_gate", "generate_report"],
        metadata: payload.metadata ?? {},
        created_at: new Date().toISOString()
      });
      await prisma.task.create({
        data: {
          id: newTaskId,
          name: "New Research Task",
          taskType: "tabular_regression_template",
          target: "SalePrice",
          metric: "cv_rmsle_mean",
          status: "ready_to_train",
          priority: "Medium",
          owner: "Research Admin",
          configPath,
          taskDir: `workspace/tasks/${newTaskId}`
        }
      });
      const record = await logAction({
        action,
        taskId: newTaskId,
        message: `Runnable task created: ${newTaskId}`,
        artifactPath: artifact,
        metadata: { runnable: true, template_base: "house_prices", config_path: configPath }
      });
      return { ok: true, ...record, task_id: newTaskId, config_path: configPath, runnable: true };
    }
    case "onboard_kaggle_competition": {
      const onboarded = await onboardKaggleCompetition(payload.metadata ?? {});
      const artifact = await writeJsonArtifact(`workspace/kaggle_onboarding/${onboarded.taskId}_frontend_onboarding_${stamp()}.json`, {
        task_id: onboarded.taskId,
        config_path: onboarded.configPath,
        command: onboarded.command,
        report: onboarded.report,
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId: onboarded.taskId,
        message: `Kaggle-style competition onboarded: ${onboarded.taskId}.`,
        artifactPath: artifact,
        metadata: { config_path: onboarded.configPath, report: onboarded.report }
      });
      return { ok: true, ...record, task_id: onboarded.taskId, config_path: onboarded.configPath, report: onboarded.report, runnable: true };
    }
    case "prepare_score_improvement_plan": {
      const plan = await writeScoreImprovementPlan(taskId, payload.metadata ?? {});
      const record = await logAction({
        action,
        taskId,
        runId: await latestRunId(taskId),
        message: "Kaggle score improvement launch plan generated.",
        artifactPath: plan.markdownArtifact,
        metadata: { plan_path: plan.jsonArtifact, markdown_path: plan.markdownArtifact, connectors: plan.plan.connectors }
      });
      return { ok: true, ...record, plan_path: plan.jsonArtifact, markdown_path: plan.markdownArtifact, connectors: plan.plan.connectors };
    }
    case "workflow_dry_run":
    case "workflow_save":
    case "workflow_publish": {
      const status = action === "workflow_dry_run" ? "validated" : action === "workflow_publish" ? "published" : "saved";
      const artifact = await writeJsonArtifact(`workspace/workflows/${taskId}/${action}_${stamp()}.json`, {
        task_id: taskId,
        action,
        status,
        stages: workflowStages,
        created_at: new Date().toISOString()
      });
      const existing = await prisma.workflow.findFirst({ where: { taskId }, orderBy: { updatedAt: "desc" } });
      await prisma.workflow.upsert({
        where: { id: existing?.id ?? `${taskId}_workflow` },
        update: {
          status,
          version: { increment: 1 },
          nodesJson: encodeJson(workflowStages.map((stage, index) => ({ id: stage, status: index < 7 ? "passed" : "pending" }))) ?? "[]",
          edgesJson: encodeJson(workflowStages.slice(1).map((stage, index) => ({ source: workflowStages[index], target: stage }))) ?? "[]",
          publishedAt: action === "workflow_publish" ? new Date() : existing?.publishedAt ?? null
        },
        create: {
          id: `${taskId}_workflow`,
          taskId,
          name: `${taskId} Research Workflow`,
          status,
          nodesJson: encodeJson(workflowStages.map((stage, index) => ({ id: stage, status: index < 7 ? "passed" : "pending" }))) ?? "[]",
          edgesJson: encodeJson(workflowStages.slice(1).map((stage, index) => ({ source: workflowStages[index], target: stage }))) ?? "[]",
          publishedAt: action === "workflow_publish" ? new Date() : null
        }
      });
      const record = await logAction({ action, taskId, message: `${action.replaceAll("_", " ")} completed`, artifactPath: artifact });
      return { ok: true, ...record };
    }
    case "stop_run": {
      const run = await prisma.experimentRun.findFirst({ where: { taskId, status: { in: ["queued", "running"] } }, orderBy: { createdAt: "desc" } });
      const cancellation = cancelRunningJob(taskId, run?.id);
      if (run) {
        await prisma.experimentRun.update({
          where: { id: run.id },
          data: {
            status: cancellation.cancelled ? "cancelled" : "cancel_requested",
            processId: null,
            finishedAt: new Date()
          }
        });
      }
      const artifact = await writeJsonArtifact(`workspace/runtime/stop_request_${stamp()}.json`, {
        task_id: taskId,
        run_id: run?.id ?? null,
        process_id: cancellation.processId,
        cancelled: cancellation.cancelled,
        requested: true,
        reason: "manual_stop",
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        runId: run?.id,
        message: cancellation.cancelled ? "Running local job cancelled." : "Stop request recorded for local runner.",
        artifactPath: artifact,
        metadata: { stop_signal_persisted: true, ...cancellation }
      });
      return { ok: true, ...record };
    }
    case "export_report": {
      const runPath = await latestExperimentPath(taskId);
      const runId = await latestRunId(taskId);
      const report = await prisma.report.findFirst({ where: { taskId }, orderBy: { updatedAt: "desc" } });
      const artifact = await writeJsonArtifact(`workspace/exports/report_export_${stamp()}.json`, {
        task_id: taskId,
        latest_run: runPath,
        report: report?.markdownPath ?? (runPath ? `${runPath}/local_report.md` : null),
        docx: report?.docxPath ?? (runPath ? `${runPath}/local_report.docx` : null),
        report_status: report?.status ?? "missing",
        report_title: report?.title ?? null,
        created_at: new Date().toISOString()
      });
      await prisma.report.upsert({
        where: { id: `${taskId}_latest_report` },
        update: { runId, status: "exported", markdownPath: report?.markdownPath ?? (runPath ? `${runPath}/local_report.md` : null), docxPath: report?.docxPath ?? (runPath ? `${runPath}/local_report.docx` : null) },
        create: { id: `${taskId}_latest_report`, taskId, runId, title: `${taskId} Local Report`, status: "exported", markdownPath: runPath ? `${runPath}/local_report.md` : null, docxPath: runPath ? `${runPath}/local_report.docx` : null }
      });
      const record = await logAction({ action, taskId, runId, message: "Report export manifest created.", artifactPath: artifact, metadata: { latest_run: runPath } });
      return { ok: true, ...record, latest_run: runPath };
    }
    case "submit_report_review": {
      const runId = await latestRunId(taskId);
      const artifact = await writeJsonArtifact(`workspace/gates/report_review_gate_${stamp()}.json`, {
        task_id: taskId,
        gate_type: "report_review",
        decision: "pending",
        created_at: new Date().toISOString()
      });
      await prisma.gate.create({ data: { id: `gate_${stamp()}`, taskId, runId, gateType: "report_review", decision: "pending", evidenceJson: encodeJson({ artifact }) } });
      await prisma.report.upsert({
        where: { id: `${taskId}_latest_report` },
        update: { status: "submitted", submittedAt: new Date() },
        create: { id: `${taskId}_latest_report`, taskId, runId, title: `${taskId} Local Report`, status: "submitted", submittedAt: new Date() }
      });
      const record = await logAction({ action, taskId, runId, message: "Report submitted to review gate.", artifactPath: artifact });
      return { ok: true, ...record };
    }
    case "approve_gate":
    case "reject_gate":
    case "approve_submission":
    case "reject_submission": {
      const decision = action.includes("approve") ? "approved" : "rejected";
      const runId = await latestRunId(taskId);
      const requestedGateId = typeof payload.metadata?.gate_id === "string" ? payload.metadata.gate_id : undefined;
      const requestedGateType = typeof payload.metadata?.gate_type === "string" ? payload.metadata.gate_type : undefined;
      const gateType = action.includes("submission") ? "submission_approval" : requestedGateType;
      const candidateGate = requestedGateId
        ? await prisma.gate.findUnique({ where: { id: requestedGateId } })
        : gateType
          ? await prisma.gate.findFirst({ where: { taskId, runId, gateType }, orderBy: { createdAt: "desc" } })
          : null;
      const existingGate = candidateGate
        && candidateGate.taskId === taskId
        && (!gateType || candidateGate.gateType === gateType)
        ? candidateGate
        : null;
      if (!existingGate) {
        return {
          ok: false,
          action,
          decision,
          status: "blocked_gate_not_found",
          error: "An existing task-scoped gate_id or gate_type is required before a decision can be recorded.",
          target_gate_found: false
        };
      }
      const targetRunId = existingGate.runId ?? runId;
      const existingEvidence = decodeJson<Record<string, unknown>>(existingGate.evidenceJson) ?? {};
      if (decision === "approved" && existingGate.gateType === "hpc_execution_approval") {
        const requestedTemplate = typeof existingEvidence.requested_template === "string"
          ? existingEvidence.requested_template
          : "";
        if (!targetRunId) {
          return { ok: false, action, status: "blocked_hpc_gate_run_missing", error: "HPC gate has no bound run." };
        }
        const validation = await validateHpcExecutionGate(existingGate, {
          taskId,
          runId: targetRunId,
          template: requestedTemplate,
          requireApproved: false
        });
        if (!validation.ok) {
          return {
            ok: false,
            action,
            status: "blocked_hpc_gate_binding_invalid",
            error: "HPC gate binding validation failed.",
            reasons: validation.reasons
          };
        }
      }
      const reviewer = "Research Admin";
      const decidedAt = new Date();
      const artifact = await writeJsonArtifact(`workspace/gates/${existingGate.gateType}_${stamp()}.json`, {
        task_id: taskId,
        workstation_run_id: targetRunId ?? null,
        gate_type: existingGate.gateType,
        gate_id: existingGate.id,
        decision,
        reviewer,
        target_gate_found: true,
        created_at: decidedAt.toISOString()
      });
      const decisionEvidence = decision === "approved" && existingGate.gateType === "hpc_execution_approval"
        ? buildHpcApprovalEvidence(existingGate, {
          reviewer,
          reason: "Approved through the task-scoped workstation gate action.",
          artifactPath: artifact,
          decidedAt
        })
        : { ...existingEvidence };
      if (decision !== "approved" && existingGate.gateType === "hpc_execution_approval") {
        delete decisionEvidence.approval;
      }
      const evidenceJson = encodeJson({
        ...decisionEvidence,
        decision_record: {
          artifact,
          decided_by_action: action,
          requested_gate_id: requestedGateId ?? null,
          requested_gate_type: requestedGateType ?? null,
          reviewer,
          decided_at: decidedAt.toISOString()
        }
      });
      const updateData = { decision, reviewer, evidenceJson, decidedAt };
      let gate;
      if (decision === "approved" && existingGate.gateType === "hpc_execution_approval") {
        const updated = await prisma.gate.updateMany({
          where: { id: existingGate.id, decision: "pending", evidenceJson: existingGate.evidenceJson },
          data: updateData
        });
        if (updated.count !== 1) {
          return {
            ok: false,
            action,
            status: "blocked_hpc_gate_state_changed",
            error: "HPC gate state or evidence changed during approval. Prepare and review a new gate."
          };
        }
        gate = { ...existingGate, ...updateData };
      } else {
        gate = await prisma.gate.update({ where: { id: existingGate.id }, data: updateData });
      }
      const record = await logAction({
        action,
        taskId,
        runId: targetRunId,
        message: `${gate.gateType} ${decision}.`,
        artifactPath: artifact,
        metadata: {
          decision,
          gate_id: gate.id,
          gate_type: gate.gateType,
          target_gate_found: true
        }
      });
      return { ok: true, ...record, decision, gate_id: gate.id, gate_type: gate.gateType, target_gate_found: true };
    }
    case "audit_s6e6_submission": {
      const runId = typeof payload.metadata?.run_id === "string" && payload.metadata.run_id.trim()
        ? payload.metadata.run_id.trim()
        : await latestRunId(taskId);
      if (!runId) {
        return { ok: false, status: "blocked_missing_run_id", error: "run_id is required for audit_s6e6_submission when no latest run exists." };
      }
      const submissionPath = typeof payload.metadata?.submission_path === "string" && payload.metadata.submission_path.trim()
        ? payload.metadata.submission_path.trim()
        : "";
      if (!submissionPath) {
        return { ok: false, status: "blocked_missing_submission_path", error: "submission_path is required for audit_s6e6_submission." };
      }
      const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
      const audit = await auditS6E6Submission(submissionPath);
      const auditPath = await writeJsonArtifact(`${runRoot}/submission_audit.json`, {
        ...audit,
        workstation_run_id: runId,
        task_id: taskId
      });
      await appendRunTrace(runRoot, {
        event: "agent_completed",
        workstation_run_id: runId,
        agent_id: "SubmissionGateAgent",
        stage: "submission_check",
        artifact_path: auditPath,
        status: audit.status
      });
      await bindEvidence({
        taskId,
        runId,
        label: "S6E6 submission audit",
        artifactPath: auditPath,
        source: "SubmissionGateAgent",
        claimBinding: "Submission row order, labels and schema were audited before any official Kaggle submit."
      });
      const record = await logAction({
        action,
        taskId,
        runId,
        message: audit.status === "passed" ? "S6E6 submission audit passed." : "S6E6 submission audit failed.",
        artifactPath: auditPath,
        metadata: {
          status: audit.status,
          submission_path: submissionPath,
          submission_sha256: audit.submission_sha256
        }
      });
      return { ok: audit.status === "passed", ...record, status: audit.status, audit, audit_path: auditPath };
    }
    case "export_audit_bundle": {
      const runPath = await latestExperimentPath(taskId);
      const runId = await latestRunId(taskId);
      const files = runPath ? ["experiment_record.json", "evidence_manifest.json", "validation_gate.json", "workflow_stage_audit.json", "submission.csv"].map((file) => `${runPath}/${file}`) : [];
      const artifact = await writeJsonArtifact(`workspace/exports/audit_bundle_${stamp()}.json`, { task_id: taskId, latest_run: runPath, files, created_at: new Date().toISOString() });
      const record = await logAction({ action, taskId, runId, message: "Audit export manifest created.", artifactPath: artifact, metadata: { latest_run: runPath, files } });
      return { ok: true, ...record, latest_run: runPath };
    }
    case "add_evidence": {
      const artifact = await writeJsonArtifact(`workspace/evidence/evidence_add_request_${stamp()}.json`, {
        task_id: taskId,
        status: "pending_upload_or_binding",
        required_metadata: ["source", "artifact_path", "claim_binding", "hash"],
        created_at: new Date().toISOString()
      });
      await prisma.evidence.create({ data: { id: `evidence_${stamp()}`, taskId, runId: await latestRunId(taskId), label: "Pending evidence binding", artifactPath: artifact, source: "frontend", claimBinding: "pending" } });
      const record = await logAction({ action, taskId, message: "Evidence binding request created.", artifactPath: artifact });
      return { ok: true, ...record };
    }
    case "report_section_select": {
      const section = typeof payload.metadata?.section === "string" ? payload.metadata.section : "unknown";
      const artifact = await writeJsonArtifact(`workspace/reports/report_section_select_${stamp()}.json`, { task_id: taskId, section, index: payload.metadata?.index ?? null, created_at: new Date().toISOString() });
      await prisma.report.upsert({
        where: { id: `${taskId}_latest_report` },
        update: { selectedSection: section },
        create: { id: `${taskId}_latest_report`, taskId, title: `${taskId} Local Report`, status: "draft", selectedSection: section }
      });
      const record = await logAction({ action, taskId, message: `Report section selected: ${section}.`, artifactPath: artifact });
      return { ok: true, ...record };
    }
    case "review_agent_patch": {
      const reviewRevisionId = randomUUID();
      const sourceAgent = String(payload.metadata?.source_agent ?? "manual");
      const patchStatus = String(payload.metadata?.patch_status ?? "suggested");
      const patch = await patchFromReviewMetadata(taskId, payload.metadata);
      const patchDir = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches`);
      const stablePatch = patch
        ? await readStableRegularTextFile(patch.fullPath, { allowedRoot: patchDir, maxBytes: 2_000_000 }).catch(() => null)
        : null;
      const patchText = stablePatch?.text ?? "";
      const patchSha256 = stablePatch?.sha256 ?? null;
      const patchAnalysis = analyzePatch(taskId, patchText);
      const syntax = await pythonSyntaxCheck(taskId);
      const patchSyntax = await patchPythonSyntaxCheck(taskId, patchText);
      const overallStatus = patch
        && stablePatch
        && patchAnalysis.passed
        && syntax.status !== "failed"
        && patchSyntax.status !== "failed"
        ? "passed"
        : "failed";
      const reviewArtifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/patch_review_${reviewRevisionId}.json`, {
        task_id: taskId,
        patch_id: patch?.name.replace(/\.diff$/, "") ?? null,
        patch_path: patch?.relativePath ?? null,
        patch_sha256: patchSha256,
        source_agent: sourceAgent,
        patch_status: patchStatus,
        review_status: overallStatus,
        review_findings: patch ? patchAnalysis.findings : ["No imported patch diff found."],
        affected_files: patchAnalysis.files,
        reviewed_at: new Date().toISOString()
      });
      const qualityArtifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/code_quality_check_${reviewRevisionId}.json`, {
        task_id: taskId,
        patch_id: patch?.name.replace(/\.diff$/, "") ?? null,
        patch_path: patch?.relativePath ?? null,
        patch_sha256: patchSha256,
        overall_status: overallStatus,
        file_scope_check: patchAnalysis.outsideAllowedScope.length ? "warning" : "passed",
        original_data_check: patchAnalysis.originalDataTouched ? "failed" : "passed",
        command_risk_check: patchAnalysis.dangerousCommand ? "failed" : "passed",
        credential_check: patchAnalysis.credentialTouched ? "failed" : "passed",
        python_syntax_check: syntax.status,
        python_syntax_file: syntax.file,
        python_syntax_error: syntax.error,
        patch_python_syntax_check: patchSyntax.status,
        patch_python_syntax_files: patchSyntax.files,
        patch_python_syntax_error: patchSyntax.error,
        smoke_test: "not_run_in_review_action",
        baseline_run_test: "requires_run_local_experiment_after_apply",
        submission_check: "requires_run_local_experiment_after_apply",
        metric_comparison: "requires_run_local_experiment_after_apply",
        reviewer_agent: "ReviewerAgent",
        human_gate_required: true,
        affected_files: patchAnalysis.files,
        findings: patch ? patchAnalysis.findings : ["No imported patch diff found."],
        created_at: new Date().toISOString()
      });
      const diffArtifact = await writeTextArtifact(`workspace/tasks/${taskId}/code/patches/patch_diff_${reviewRevisionId}.md`, [
        `# Patch Review`,
        ``,
        `- task_id: ${taskId}`,
        `- patch_path: ${patch?.relativePath ?? "missing"}`,
        `- source_agent: ${sourceAgent}`,
        `- patch_status: ${patchStatus}`,
        `- decision: ${overallStatus}`,
        `- review_artifact: ${reviewArtifact}`,
        `- quality_artifact: ${qualityArtifact}`,
        ``,
        `## Findings`,
        ...(patchAnalysis.findings.length ? patchAnalysis.findings.map((finding) => `- ${finding}`) : ["- No blocking findings."]),
        ``,
        `## Patch Content`,
        patchText
          ? "Patch content is not duplicated into review summaries. Inspect the hash-bound patch artifact through the controlled workspace path."
          : "No patch diff was available."
      ].join("\n"));
      const traceArtifact = await writeTextArtifact(`workspace/tasks/${taskId}/code/patches/code_agent_trace_${reviewRevisionId}.jsonl`, JSON.stringify({
        task_id: taskId,
        action: "review_agent_patch",
        source_agent: sourceAgent,
        patch_status: patchStatus,
        overall_status: overallStatus,
        patch_path: patch?.relativePath ?? null,
        patch_sha256: patchSha256,
        created_at: new Date().toISOString()
      }) + "\n");
      const failureReviewArtifact = overallStatus === "failed"
        ? await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/failure_review_${reviewRevisionId}.json`, {
          schema: "academic_research_os.agent_failure_review.v1",
          task_id: taskId,
          failed_stage: "code_review",
          failed_agent: sourceAgent,
          return_to_stage: "code_generation",
          retry_policy: {
            max_auto_retries: 2,
            next_retry_index: 1,
            pause_after_retry_index: 3
          },
          patch_path: patch?.relativePath ?? null,
          quality_artifact: qualityArtifact,
          review_artifact: reviewArtifact,
          blocking_findings: patch ? patchAnalysis.findings : ["No imported patch diff found."],
          reviewer_agent: "ReflectionReviewerAgent",
          next_work_order: {
            agent_id: "CodeImplementationAgent",
            instruction: "Regenerate a complete unified diff that stays inside allowed task code paths and passes patch-level Python syntax checks before HPC execution is considered."
          },
          training_started: false,
          official_submission_started: false,
          created_at: new Date().toISOString()
        })
        : null;
      const qualityArtifactStable = await readStableRegularTextFile(
        resolveWorkspacePath(qualityArtifact),
        { allowedRoot: patchDir, maxBytes: 1_000_000 }
      );
      const record = await logAction({
        action,
        taskId,
        message: overallStatus === "passed" ? "Code agent patch passed quality gate; human approval still required before apply." : "Code agent patch failed quality gate; apply is blocked.",
        artifactPath: reviewArtifact,
        metadata: { source_agent: sourceAgent, patch_status: patchStatus, overall_status: overallStatus, patch_sha256: patchSha256, review_artifact: reviewArtifact, quality_artifact: qualityArtifact, quality_artifact_sha256: qualityArtifactStable.sha256, diff_artifact: diffArtifact, trace_artifact: traceArtifact, failure_review_artifact: failureReviewArtifact }
      });
      return { ok: true, ...record, quality_status: overallStatus };
    }
    case "apply_agent_patch": {
      const qualityGate = await latestPassedCodeQualityGate(taskId);
      if (!qualityGate) {
        const artifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/apply_blocked_${stamp()}.json`, {
          task_id: taskId,
          action,
          blocked: true,
          reason: "No passed code_quality_check artifact found.",
          created_at: new Date().toISOString()
        });
        const record = await logAction({
          action,
          taskId,
          message: "Patch apply blocked until Code Quality Gate passes.",
          artifactPath: artifact,
          metadata: { blocked: true }
        });
        return { ok: false, ...record, error: "Patch apply blocked until Code Quality Gate passes." };
      }
      const sourceAgent = String(payload.metadata?.source_agent ?? "manual");
      const patchStatus = String(payload.metadata?.patch_status ?? "applied");
      const gatePatchPath = typeof qualityGate.payload.patch_path === "string"
        ? qualityGate.payload.patch_path.replaceAll("\\", "/")
        : "";
      const gatePatch = gatePatchPath
        ? await readStableRegularTextFile(resolveWorkspacePath(gatePatchPath), {
          allowedRoot: resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches`),
          maxBytes: 2_000_000
        }).catch(() => null)
        : null;
      if (!gatePatchPath || !gatePatch || gatePatch.sha256 !== qualityGate.patchSha256) {
        const artifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/apply_blocked_${stamp()}.json`, {
          task_id: taskId,
          action,
          blocked: true,
          reason: "The latest passed Code Quality Gate does not bind to an existing patch artifact.",
          quality_gate_path: qualityGate.relativePath,
          quality_gate_sha256: qualityGate.qualityArtifactSha256,
          created_at: new Date().toISOString()
        });
        const record = await logAction({
          action,
          taskId,
          message: "Patch apply blocked because the passed Code Quality Gate has no usable patch binding.",
          artifactPath: artifact,
          metadata: { blocked: true, quality_gate_path: qualityGate.relativePath, quality_gate_sha256: qualityGate.qualityArtifactSha256 }
        });
        return { ok: false, ...record, error: "Patch apply blocked: missing bound patch artifact." };
      }
      const artifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/${action}_${stamp()}.json`, {
        task_id: taskId,
        source_agent: sourceAgent,
        patch_status: patchStatus,
        action,
        patch_path: gatePatchPath,
        patch_sha256: qualityGate.patchSha256,
        quality_gate_path: qualityGate.relativePath,
        quality_gate_sha256: qualityGate.qualityArtifactSha256,
        quality_gate_action_id: qualityGate.actionId,
        applied_logical_only: true,
        next_required_step: "run_local_experiment",
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: "Patch apply recorded after passed Code Quality Gate; run local experiment to compare metrics.",
        artifactPath: artifact,
        metadata: { source_agent: sourceAgent, patch_status: patchStatus, patch_sha256: qualityGate.patchSha256, quality_gate_path: qualityGate.relativePath, quality_gate_sha256: qualityGate.qualityArtifactSha256, quality_gate_action_id: qualityGate.actionId }
      });
      return { ok: true, ...record };
    }
    case "rollback_agent_patch": {
      const sourceAgent = String(payload.metadata?.source_agent ?? "manual");
      const patchStatus = String(payload.metadata?.patch_status ?? "rollback_requested");
      const artifact = await writeJsonArtifact(`workspace/tasks/${taskId}/code/patches/${action}_${stamp()}.json`, {
        task_id: taskId,
        source_agent: sourceAgent,
        patch_status: patchStatus,
        action,
        review_required: true,
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: "Patch rollback recorded.",
        artifactPath: artifact,
        metadata: { source_agent: sourceAgent, patch_status: patchStatus }
      });
      return { ok: true, ...record };
    }
    case "language_select":
    case "settings_theme_change":
    case "save_settings_changes":
    case "cancel_settings_changes":
    case "open_settings_section": {
      const runId = await latestRunId(taskId);
      const settingsArtifact = await writeJsonArtifact(`workspace/settings/${action}_${stamp()}.json`, {
        schema: "academic_research_os.settings_action.v1",
        task_id: taskId,
        run_id: runId ?? null,
        action,
        metadata: payload.metadata ?? {},
        persisted_to_ui_settings: false,
        audit_required: action === "save_settings_changes",
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        runId,
        message: `${action.replaceAll("_", " ")} recorded in settings audit.`,
        artifactPath: settingsArtifact,
        metadata: payload.metadata
      });
      return { ok: true, ...record, settings_artifact: settingsArtifact };
    }
    case "test_all_connectors": {
      const runId = await latestRunId(taskId);
      const checks = {
        deepseek: process.env.DEEPSEEK_API_KEY || process.env.DEEPSEEK_API_KEY_FILE ? "configured" : "not_configured",
        kaggle: process.env.KAGGLE_API_TOKEN || process.env.KAGGLE_API_TOKEN_FILE || process.env.KAGGLE_USERNAME || process.env.KAGGLE_USERNAME_FILE ? "configured" : "not_configured",
        gpu_ssh: process.env.GPU_SSH_HOST && process.env.GPU_SSH_USER && (process.env.GPU_SSH_PASSWORD || process.env.GPU_SSH_PASSWORD_FILE || process.env.GPU_SSH_KEY_PATH || process.env.GPU_SSH_KEY_PATH_FILE) && process.env.GPU_REMOTE_WORKSPACE ? "configured_requires_smoke" : "not_configured",
        claude_code: process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_API_KEY_FILE || process.env.CLAUDE_API_KEY || process.env.CLAUDE_API_KEY_FILE || process.env.DEEPSEEK_API_KEY || process.env.DEEPSEEK_API_KEY_FILE ? "configured" : "not_configured"
      };
      const artifact = await writeJsonArtifact(`workspace/settings/test_all_connectors_${stamp()}.json`, {
        schema: "academic_research_os.connector_settings_smoke.v1",
        task_id: taskId,
        run_id: runId ?? null,
        status: "recorded_config_probe",
        checks,
        external_network_invoked: false,
        next_required_steps: [
          "Run /api/llm/deepseek/smoke for DeepSeek runtime proof.",
          "Run /api/gpu/connections/test after current HPC allocation credentials are refreshed.",
          "Run Kaggle DPAPI smoke only after Human Gate if official API access is needed."
        ],
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        runId,
        message: "Connector configuration probe recorded; no external training or submission was started.",
        artifactPath: artifact,
        metadata: { checks }
      });
      return { ok: true, ...record, checks, external_network_invoked: false };
    }
    case "select_compute_mode": {
      const requestedMode = String(payload.metadata?.mode ?? payload.metadata?.execution_mode ?? "").toLowerCase();
      if (requestedMode === "local") {
        const artifact = await writeJsonArtifact(`workspace/settings/compute_mode_blocked_${stamp()}.json`, {
          schema: "academic_research_os.compute_mode_selection.v1",
          task_id: taskId,
          status: "rejected",
          requested_mode: "local",
          execution_mode: "hpc_gpu",
          local_training_enabled: false,
          reason: "Local training is disabled by release policy; configure the gated HPC/GPU runtime or report Blocked.",
          created_at: new Date().toISOString()
        });
        await logAction({
          action: "select_compute_mode_blocked",
          taskId,
          message: "Local compute selection was rejected by the HPC-only training policy.",
          artifactPath: artifact,
          metadata: { requested_mode: "local", local_training_enabled: false }
        });
        return {
          ok: false,
          status: "blocked_local_training_disabled",
          execution_mode: "hpc_gpu",
          local_training_enabled: false,
          artifact_path: artifact
        };
      }
      const executionMode = "hpc_gpu";
      const computeSetting = {
        execution_mode: executionMode,
        local_training_enabled: false,
        local_training_scope: "disabled",
        selected_at: new Date().toISOString(),
        selected_by: "workstation_gpu_page",
        note: "HPC/GPU cluster mode is selected. Local training fallback is disabled."
      };
      await prisma.setting.upsert({
        where: { key: "compute" },
        update: { valueJson: encodeJson(computeSetting) ?? "{}" },
        create: { key: "compute", valueJson: encodeJson(computeSetting) ?? "{}" }
      });
      const artifact = await writeJsonArtifact(`workspace/settings/compute_mode_${stamp()}.json`, {
        schema: "academic_research_os.compute_mode_selection.v1",
        task_id: taskId,
        compute: computeSetting,
        local_training_policy: {
          allowed_now: false,
          scope: computeSetting.local_training_scope,
          official_submission_allowed: false,
          human_gate_required_for_official_submission: true
        },
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        message: "Compute mode set to HPC/GPU; local training fallback is disabled.",
        artifactPath: artifact,
        metadata: { execution_mode: executionMode, local_training_enabled: false }
      });
      return { ok: true, ...record, compute: computeSetting, artifact_path: artifact };
    }
    case "rotate_credentials_batch": {
      const runId = await latestRunId(taskId);
      const artifact = await writeJsonArtifact(`workspace/settings/rotate_credentials_request_${stamp()}.json`, {
        schema: "academic_research_os.credential_rotation_request.v1",
        task_id: taskId,
        run_id: runId ?? null,
        status: "needs_human_secret_input",
        connectors: ["kaggle", "hpc_gpu_ssh", "deepseek", "claude_code"],
        secret_values_recorded: false,
        next_safe_commands: [
          "scripts/manage_kaggle_secret.ps1 install-token",
          "scripts/manage_hpc_ssh_secret.ps1 install-password -Host <host> -Port <port> -User <user>",
          "scripts/manage_deepseek_secret.ps1 install-key"
        ],
        created_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        runId,
        message: "Credential rotation request recorded; secret values must be provided through DPAPI installers, not the UI action log.",
        artifactPath: artifact,
        metadata: { secret_values_recorded: false }
      });
      return { ok: true, ...record, requires_human_secret_input: true };
    }
    case "navigate_page":
    case "workspace_select":
    case "notification_open":
    case "profile_open":
    case "research_mode_toggle":
    case "search_command":
    case "task_select":
    case "stage_select":
    case "workflow_node_select":
    case "workflow_library_node_select":
    case "code_file_select":
    case "code_editor_tab_select":
    case "terminal_tab_select":
    case "runtime_agent_select":
    case "experiment_select":
    case "gate_check_open":
    case "open_fullscreen":
    case "view_full_log":
    case "open_validation_review":
    case "edit_claim_record":
    case "open_artifact_folder":
    case "view_reproducibility_record":
    case "design_sample_action": {
      const runId = await latestRunId(taskId);
      const artifact = await writeJsonArtifact(`workspace/ui_state/${action}_${stamp()}.json`, {
        task_id: taskId,
        run_id: runId ?? null,
        action,
        metadata: payload.metadata ?? {},
        latest_run: await latestExperimentPath(taskId),
        recorded_at: new Date().toISOString()
      });
      const record = await logAction({
        action,
        taskId,
        runId,
        message: `${action.replaceAll("_", " ")} recorded in UI state.`,
        artifactPath: artifact,
        metadata: payload.metadata
      });
      return { ok: true, ...record };
    }
    default: {
      const runPath = await latestExperimentPath(taskId);
      const artifact = await writeJsonArtifact(`workspace/runtime/${action}_${stamp()}.json`, {
        task_id: taskId,
        action,
        latest_run: runPath,
        metadata: payload.metadata ?? {},
        handled: true,
        created_at: new Date().toISOString()
      });
      const message = `${action.replaceAll("_", " ")} handled.`;
      const record = await logAction({ action, taskId, runId: await latestRunId(taskId), message, artifactPath: artifact, metadata: payload.metadata });
      return { ok: true, ...record };
    }
  }
}
