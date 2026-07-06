import { prisma } from "@/lib/db";
import { claudeApiKeyStatus, deepSeekApiKeyStatus, deepSeekConfig, gpuSshConfig, gpuSshStatus } from "@/lib/server/capabilities";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { decodeJson } from "@/lib/server/json";
import { latestExperimentPath, latestScoreGatedWorkstationRunPath, latestWorkstationRunPath, readJsonFile, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

type RuntimeSummary = {
  task_id: string;
  latest_experiment_dir: string | null;
  latest_workstation_run_dir?: string | null;
  latest_score_gated_run_dir?: string | null;
  task_state?: Record<string, unknown> | null;
  agent_trace?: Array<Record<string, unknown>>;
  event_log?: Array<Record<string, unknown>>;
  artifact_manifest?: Record<string, unknown> | null;
  evidence_graph?: Record<string, unknown> | null;
  experiment_graph?: Record<string, unknown> | null;
  reflection?: Record<string, unknown> | null;
  memory?: Record<string, unknown> | null;
  gate_engine?: Record<string, unknown> | null;
  runtime_snapshot?: Record<string, unknown> | null;
  report_markdown?: string;
  generated_code?: string;
  training_log?: string[];
  latest_code_agent_session?: Record<string, unknown> | null;
  latest_code_agent_transcript?: Record<string, unknown> | null;
  latest_code_agent_review?: Record<string, unknown> | null;
  score_improvement_gate?: Record<string, unknown> | null;
  submission_audit?: Record<string, unknown> | null;
  score_regression_diagnosis?: Record<string, unknown> | null;
  score_regression_recovery_plan?: Record<string, unknown> | null;
};

type XsciRunCandidate = {
  runId: string;
  absolutePath: string;
  relativePath: string;
  mtimeMs: number;
  hasSummary: boolean;
  hasEvents: boolean;
};

const runtimeTaskIds = ["playground_series_s6e6", "house_prices", "titanic", "telco_churn"];

const stages = [
  "task_understanding",
  "literature_context",
  "experiment_planning",
  "human_plan_gate",
  "eda",
  "code_generation",
  "code_review",
  "training",
  "validation_review",
  "submission_check",
  "human_submission_gate",
  "report_generation",
  "human_final_gate",
  "reflection"
].map((stage) => ({ stage, status: "reserved" }));

function readJsonl(text: string) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as Record<string, unknown>;
      } catch {
        return { raw: line };
      }
    });
}

function safeSummaryText(value: unknown, fallback = "") {
  const text = typeof value === "string" ? value : value === null || value === undefined ? fallback : String(value);
  if (!text) return fallback;
  if (/(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]/i.test(text)) {
    return "[redacted-sensitive-text]";
  }
  return text.length > 700 ? `${text.slice(0, 697)}...` : text;
}

function basename(value: string) {
  return value.replaceAll("\\", "/").split("/").filter(Boolean).pop() ?? value;
}

function inferTaskFromRunId(runId: string) {
  return runId.replace(/_(gpu|local)_\d{8}_\d{6}$/i, "");
}

async function readTextFile(filePath: string) {
  const fs = await import("node:fs/promises");
  return fs.readFile(filePath, "utf-8").catch(() => "");
}

async function fileStatMtime(path: string) {
  const fs = await import("node:fs/promises");
  return fs.stat(path).then((stat) => stat.mtimeMs).catch(() => 0);
}

async function loadXsciTerminalAgentSummary() {
  const fs = await import("node:fs/promises");
  const root = resolveWorkspacePath("experiments/evolution");
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = (await Promise.all(entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith("_"))
    .map(async (entry): Promise<XsciRunCandidate> => {
      const absolutePath = `${root}/${entry.name}`;
      const relativePath = `experiments/evolution/${entry.name}`;
      const [dirMtime, summaryMtime, eventsMtime] = await Promise.all([
        fileStatMtime(absolutePath),
        fileStatMtime(`${absolutePath}/summary.json`),
        fileStatMtime(`${absolutePath}/events.jsonl`)
      ]);
      return {
        runId: entry.name,
        absolutePath,
        relativePath,
        mtimeMs: Math.max(dirMtime, summaryMtime, eventsMtime),
        hasSummary: summaryMtime > 0,
        hasEvents: eventsMtime > 0
      };
    }))).sort((a, b) => b.mtimeMs - a.mtimeMs);

  const latestAny = candidates[0] ?? null;
  const latest = candidates.find((item) => item.hasEvents || item.hasSummary) ?? latestAny;
  const latestPending = latestAny && latestAny.runId !== latest?.runId && !latestAny.hasEvents && !latestAny.hasSummary
    ? latestAny
    : null;
  const summary = latest?.hasSummary
    ? await readJsonFile(`${latest.absolutePath}/summary.json`) as Record<string, unknown> | null
    : null;
  const eventsText = latest?.hasEvents ? await readTextFile(`${latest.absolutePath}/events.jsonl`) : "";
  const events = readJsonl(eventsText);
  const iterations = Array.isArray(summary?.iterations)
    ? summary.iterations.slice(-12).map((item) => asRecordForSummary(item)).filter(Boolean)
    : [];
  const taskId = safeSummaryText(summary?.task, latest ? inferTaskFromRunId(latest.runId) : "");
  const bestCvScore = typeof summary?.best_cv_score === "number" ? summary.best_cv_score : null;
  const memoryPath = `${root}/retrospective_memory.json`;
  const memoryPayload = await readJsonFile(memoryPath);
  const memoryRecords = Array.isArray(memoryPayload) ? memoryPayload.map((item) => asRecordForSummary(item)).filter(Boolean) : [];
  const recentMemory = memoryRecords.slice(-8).reverse().map((record) => ({
    memory_id: safeSummaryText(record.memory_id),
    task_type: safeSummaryText(record.task_type),
    method: safeSummaryText(record.method),
    what_worked: safeSummaryText(record.what_worked),
    what_failed: safeSummaryText(record.what_failed),
    metric_delta: typeof record.metric_delta === "number" ? record.metric_delta : null,
    reusable_strategy: safeSummaryText(record.reusable_strategy),
    failure_pattern: safeSummaryText(record.failure_pattern),
    linked_exp_ids: Array.isArray(record.linked_exp_ids) ? record.linked_exp_ids.map((id) => safeSummaryText(id)).slice(0, 8) : []
  }));
  const commandsTask = taskId || "<task_id>";
  const commands = [
    {
      label: "Agent",
      command: `$env:PYTHONPATH='src'; python -m xsci agent ${commandsTask} --compute gpu`,
      description: "Start the interactive Kaggle Research Agent; events are written to events.jsonl."
    },
    {
      label: "Run",
      command: `$env:PYTHONPATH='src'; python -m xsci run ${commandsTask} --compute gpu --iterations 3`,
      description: "Run the audited evolution loop through the Search Controller."
    },
    {
      label: "Watch",
      command: `$env:PYTHONPATH='src'; python -m xsci watch ${latest?.runId ?? ""} --lines 40`,
      description: "Replay or follow the latest run event stream."
    },
    {
      label: "Memory",
      command: "$env:PYTHONPATH='src'; python -m xsci memory --limit 8",
      description: "Inspect recent retrospective memory successes and failures."
    },
    {
      label: "Dashboard",
      command: "$env:PYTHONPATH='src'; python -m xsci dashboard status --port 8088",
      description: "Check the current workstation frontend service status."
    }
  ];

  return {
    status: latest ? latest.hasEvents ? "live_events" : summary ? "summary_only" : "pending_run" : "no_runs",
    dashboard_url: "http://127.0.0.1:8088",
    evolution_root: "experiments/evolution",
    run_count: candidates.length,
    completed_run_count: candidates.filter((item) => item.hasSummary).length,
    latest_run_id: latest?.runId ?? null,
    latest_run_dir: latest?.relativePath ?? null,
    latest_run_mtime: latest ? new Date(latest.mtimeMs).toISOString() : null,
    latest_pending_run_id: latestPending?.runId ?? null,
    latest_pending_run_dir: latestPending?.relativePath ?? null,
    task_id: taskId || null,
    metric: safeSummaryText(summary?.metric, "cv_score"),
    metric_direction: safeSummaryText(summary?.metric_direction, "maximize"),
    best_exp_id: safeSummaryText(summary?.best_exp_id, ""),
    best_cv_score: bestCvScore,
    n_iterations: typeof summary?.n_iterations === "number" ? summary.n_iterations : iterations.length,
    n_promotions: typeof summary?.n_promotions === "number" ? summary.n_promotions : iterations.filter((item) => item.promoted === true).length,
    events_path: latest ? `${latest.relativePath}/events.jsonl` : null,
    events_present: latest?.hasEvents ?? false,
    event_count: events.length,
    recent_events: events.slice(-12).map((event) => sanitizeSummaryRecord(event)),
    summary_path: latest?.hasSummary ? `${latest.relativePath}/summary.json` : null,
    summary_present: latest?.hasSummary ?? false,
    iterations: iterations.map((item) => sanitizeSummaryRecord(item)),
    memory_path: "experiments/evolution/retrospective_memory.json",
    memory_count: memoryRecords.length,
    recent_memory: recentMemory,
    commands,
    claim_boundary: "No official Kaggle rank, medal, or MLE-Bench claim is shown unless a Kaggle response artifact exists and passes claim audit."
  };
}

function asRecordForSummary(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function sanitizeSummaryRecord(record: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => {
    if (typeof value === "string") return [key, safeSummaryText(value)];
    if (Array.isArray(value)) return [key, value.slice(0, 12).map((item) => typeof item === "string" ? safeSummaryText(item) : item)];
    return [key, value];
  }));
}

function summarizeExperimentLog(log: Record<string, unknown> | null) {
  if (!log) return [];
  const evaluation = log.evaluation as Record<string, unknown> | undefined;
  const modelResults = evaluation?.model_results as Record<string, Record<string, unknown>> | undefined;
  const lines = Object.entries(modelResults ?? {}).map(([model, metrics]) => (
    `[model] ${model}: ${Object.entries(metrics).map(([key, value]) => `${key}=${value}`).join(", ")}`
  ));
  const submission = log.submission_check as Record<string, unknown> | undefined;
  if (submission) {
    lines.push(`[submission] valid=${submission.valid}; rows_match=${submission.rows_match}; columns_match=${submission.columns_match}; missing=${submission.missing_predictions}`);
  }
  return lines;
}

async function latestCodeAgentSession(taskId: string) {
  const fs = await import("node:fs/promises");
  const root = resolveWorkspacePath("workspace/code_agent_sessions");
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isDirectory())
    .map(async (entry) => {
      const manifestPath = `workspace/code_agent_sessions/${entry.name}/session_manifest.json`;
      const manifest = await readJsonFile(resolveWorkspacePath(manifestPath)) as Record<string, unknown> | null;
      if (!manifest || manifest.task_id !== taskId) return null;
      const stat = await fs.stat(resolveWorkspacePath(manifestPath)).catch(() => null);
      return { manifest, manifestPath, mtimeMs: stat?.mtimeMs ?? 0 };
    }));
  const latest = candidates.filter(Boolean).sort((a, b) => (b?.mtimeMs ?? 0) - (a?.mtimeMs ?? 0))[0];
  if (!latest) return { session: null, transcript: null };
  const transcriptPath = String(latest.manifest.transcript_path ?? "");
  const transcriptText = transcriptPath ? await readTextFile(resolveWorkspacePath(transcriptPath)) : "";
  const firstLine = transcriptText.split(/\r?\n/).find(Boolean);
  let transcript: Record<string, unknown> | null = null;
  if (firstLine) {
    try {
      transcript = JSON.parse(firstLine) as Record<string, unknown>;
    } catch {
      transcript = { raw: firstLine.slice(0, 1000) };
    }
  }
  return {
    session: { ...latest.manifest, manifest_path: latest.manifestPath },
    transcript
  };
}

async function latestStrategyArtifact(prefix: string) {
  const fs = await import("node:fs/promises");
  const root = resolveWorkspacePath("workspace/strategy");
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.startsWith(prefix) && entry.name.endsWith(".json"))
    .map(async (entry) => {
      const absolutePath = resolveWorkspacePath(`workspace/strategy/${entry.name}`);
      const stat = await fs.stat(absolutePath).catch(() => null);
      return { path: `workspace/strategy/${entry.name}`, absolutePath, mtimeMs: stat?.mtimeMs ?? 0 };
    }));
  const latest = candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  if (!latest) return null;
  const payload = await readJsonFile(latest.absolutePath) as Record<string, unknown> | null;
  return payload ? { ...payload, artifact_path: latest.path } : null;
}

async function latestPatchArtifact(taskId: string, prefix: string, suffix: string) {
  const fs = await import("node:fs/promises");
  const patchRoot = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches`);
  const entries = await fs.readdir(patchRoot, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.startsWith(prefix) && entry.name.endsWith(suffix))
    .map(async (entry) => {
      const absolutePath = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches/${entry.name}`);
      const stat = await fs.stat(absolutePath).catch(() => null);
      return {
        name: entry.name,
        path: `workspace/tasks/${taskId}/code/patches/${entry.name}`,
        absolutePath,
        mtimeMs: stat?.mtimeMs ?? 0
      };
    }));
  return candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0] ?? null;
}

async function latestCodeAgentReview(taskId: string) {
  const [reviewRef, qualityRef, diffRef, traceRef] = await Promise.all([
    latestPatchArtifact(taskId, "patch_review_", ".json"),
    latestPatchArtifact(taskId, "code_quality_check_", ".json"),
    latestPatchArtifact(taskId, "patch_diff_", ".md"),
    latestPatchArtifact(taskId, "code_agent_trace_", ".jsonl")
  ]);
  if (!reviewRef && !qualityRef && !diffRef && !traceRef) return null;
  const [review, quality, diffText, traceText] = await Promise.all([
    reviewRef ? readJsonFile(reviewRef.absolutePath) : Promise.resolve(null),
    qualityRef ? readJsonFile(qualityRef.absolutePath) : Promise.resolve(null),
    diffRef ? readTextFile(diffRef.absolutePath) : Promise.resolve(""),
    traceRef ? readTextFile(traceRef.absolutePath) : Promise.resolve("")
  ]);
  return {
    patch_review: review ? { ...(review as Record<string, unknown>), artifact_path: reviewRef?.path } : null,
    code_quality_check: quality ? { ...(quality as Record<string, unknown>), artifact_path: qualityRef?.path } : null,
    patch_diff_path: diffRef?.path ?? null,
    patch_diff_excerpt: diffText.slice(0, 5000),
    trace_path: traceRef?.path ?? null,
    trace: readJsonl(traceText).slice(-5)
  };
}

async function hpcGpuProbeStatus() {
  const fs = await import("node:fs/promises");
  const probePath = resolveWorkspacePath("workspace/hpc/web_terminal_probe.txt");
  try {
    const text = await fs.readFile(probePath, "utf-8");
    const required = ["$ whoami", "$ hostname", "$ pwd", "Python", "NVIDIA-SMI", "$ df -hT", "$ free -h"];
    const missing = required.filter((term) => !text.includes(term));
    const a800Hits = (text.match(/NVIDIA A800-SXM4-80GB|NVIDIAA800|A800-SXM4/g) ?? []).length;
    return {
      present: true,
      path: "workspace/hpc/web_terminal_probe.txt",
      fullyReadyAllowed: missing.length === 0 && a800Hits >= 4,
      missing,
      a800Hits
    };
  } catch {
    return {
      present: false,
      path: "workspace/hpc/web_terminal_probe.txt",
      fullyReadyAllowed: false,
      missing: ["probe_file"],
      a800Hits: 0
    };
  }
}

async function latestGpuSshConnectionStatus() {
  const fs = await import("node:fs/promises");
  const gpuDir = resolveWorkspacePath("workspace/gpu");
  try {
    const entries = await fs.readdir(gpuDir, { withFileTypes: true });
    const candidates = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && /^connection_test_.*\.json$/.test(entry.name))
        .map(async (entry) => {
          const absolutePath = `${gpuDir}/${entry.name}`;
          const stat = await fs.stat(absolutePath);
          return { name: entry.name, absolutePath, mtimeMs: stat.mtimeMs };
        })
    );
    const latest = candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
    if (!latest) {
      return { present: false, passed: false, path: null, gpuCount: 0, gpuSummary: "", torchImport: null as boolean | null };
    }
    const payload = await readJsonFile(latest.absolutePath) as Record<string, unknown> | null;
    const stdout = typeof payload?.stdout === "string" ? payload.stdout : "";
    const gpuLines = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => /^GPU \d+:\s+NVIDIA/i.test(line));
    const gpuNames = gpuLines
      .map((line) => line.match(/^GPU \d+:\s+(.+?)(?:\s+\(UUID|$)/i)?.[1]?.trim())
      .filter(Boolean) as string[];
    const counts = new Map<string, number>();
    for (const name of gpuNames) counts.set(name, (counts.get(name) ?? 0) + 1);
    const gpuSummary = [...counts.entries()].map(([name, count]) => `${count} x ${name}`).join(" + ");
    const runtimeProbe = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .map((line) => {
        try {
          return JSON.parse(line) as Record<string, unknown>;
        } catch {
          return null;
        }
      })
      .find((item) => item && ("torch_import" in item || "python_runtime" in item));
    return {
      present: true,
      passed: payload?.status === "passed" && gpuNames.length > 0,
      path: `workspace/gpu/${latest.name}`,
      gpuCount: gpuNames.length,
      gpuNames,
      gpuSummary,
      pythonRuntime: typeof runtimeProbe?.python_runtime === "string" ? runtimeProbe.python_runtime : null,
      torchImport: typeof runtimeProbe?.torch_import === "boolean" ? runtimeProbe.torch_import : null,
      torchError: typeof runtimeProbe?.torch_error === "string" ? runtimeProbe.torch_error : null,
      proxyPolicy: payload?.proxy_policy ?? null,
      authPolicy: payload?.auth_policy ?? null,
      createdAt: payload?.created_at ?? null
    };
  } catch {
    return { present: false, passed: false, path: null, gpuCount: 0, gpuSummary: "", torchImport: null as boolean | null };
  }
}

async function latestS6E6BoostingDependencyStatus() {
  const fs = await import("node:fs/promises");
  const gpuDir = resolveWorkspacePath("workspace/gpu");
  const entries = await fs.readdir(gpuDir, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.startsWith("s6e6_boosting_dependency_") && entry.name.endsWith(".json"))
    .map(async (entry) => {
      const absolutePath = `${gpuDir}/${entry.name}`;
      const stat = await fs.stat(absolutePath).catch(() => null);
      return { name: entry.name, absolutePath, mtimeMs: stat?.mtimeMs ?? 0 };
    }));
  const latest = candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  if (!latest) return { present: false, status: "missing", path: null, blocker: null, nextAction: null };
  const payload = await readJsonFile(latest.absolutePath) as Record<string, unknown> | null;
  return {
    present: true,
    status: String(payload?.status ?? "unknown"),
    path: `workspace/gpu/${latest.name}`,
    blocker: typeof payload?.blocker === "string" ? payload.blocker : null,
    nextAction: typeof payload?.next_action === "string" ? payload.next_action : null,
    trainingStarted: payload?.training_started === true,
    createdAt: payload?.created_at ?? null
  };
}

async function kaggleDpapiProbeStatus(report: Record<string, unknown> | null) {
  const fs = await import("node:fs/promises");
  const credentialPath = typeof report?.credential_path === "string" ? report.credential_path : "";
  const envConfigured = Boolean(process.env.KAGGLE_API_TOKEN || (process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY));
  const credentialFilePresent = credentialPath ? await fs.access(credentialPath).then(() => true).catch(() => false) : false;
  const toolStatus = (report?.tool_status as Record<string, unknown> | undefined) ?? {};
  const toolchainReady = toolStatus.python_package_installed === true && Boolean(toolStatus.python_package_version || toolStatus.cli_path);
  const configured = envConfigured || Boolean(report?.credential_installed) || credentialFilePresent;

  return {
    present: Boolean(report),
    configured,
    credential_status: configured ? "configured_dpapi" : "not_configured",
    token_type: typeof report?.token_type === "string" ? report.token_type : configured ? "unknown" : "none",
    token_loaded_in_env: envConfigured,
    credential_file_present: credentialFilePresent,
    toolchain_ready: toolchainReady,
    python_package_version: toolStatus.python_package_version ?? null,
    cli_path: toolStatus.cli_path ?? null,
    report_path: "docs/kaggle_dpapi_readiness.json",
    safe_install_command: report?.safe_install_command ?? "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\manage_kaggle_secret.ps1 install-token",
    real_api_smoke_command: report?.real_api_smoke_command ?? "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\manage_kaggle_secret.ps1 smoke -AllowRealExternal",
    human_gate_required_for_submission: true
  };
}

async function loadRuntimeSummary(taskId = "house_prices"): Promise<RuntimeSummary> {
  const latestRaw = await latestExperimentPath(taskId);
  const latestWorkstationRun = await latestWorkstationRunPath(taskId);
  const latestScoreGatedRun = taskId === "playground_series_s6e6" ? await latestScoreGatedWorkstationRunPath(taskId) : null;
  const latest = latestScoreGatedRun ?? latestRaw;
  const outputDir = latest ? resolveWorkspacePath(latest) : null;
  const workspaceSummary = await readJsonFile(resolveWorkspacePath("workspace/workstation_summary.json"));
  if (!outputDir) {
    return {
      task_id: taskId,
      latest_experiment_dir: null,
      latest_workstation_run_dir: latestWorkstationRun,
      latest_score_gated_run_dir: latestScoreGatedRun,
      runtime_snapshot: (workspaceSummary as Record<string, unknown> | null) ?? null
    };
  }
  const [
    taskState,
    artifactManifest,
    evidenceGraph,
    experimentGraph,
    reflection,
    memory,
    gateEngine,
    runtimeSnapshot,
    experimentLog,
    agentTraceText,
    eventLogText,
    reportMarkdown,
    generatedCode,
    metrics,
    scoreImprovementGate,
    submissionAudit,
    scoreRegressionRecoveryPlan,
    scoreRegressionDiagnosis,
    codeAgentResult,
    latestSession,
    latestReview
  ] = await Promise.all([
    readJsonFile(`${outputDir}/task_state_machine.json`),
    readJsonFile(`${outputDir}/artifact_manifest.json`),
    readJsonFile(`${outputDir}/evidence_index.json`),
    readJsonFile(`${outputDir}/experiment_graph.json`),
    readJsonFile(`${outputDir}/reflection.json`),
    readJsonFile(`${outputDir}/memory_records.json`),
    readJsonFile(`${outputDir}/gate_engine.json`),
    readJsonFile(`${outputDir}/runtime_snapshot.json`),
    readJsonFile(`${outputDir}/experiment_log.json`),
    readTextFile(`${outputDir}/agent_trace.jsonl`),
    readTextFile(`${outputDir}/event_log.jsonl`),
    readTextFile(`${outputDir}/research_report.md`).then((text) => text || readTextFile(`${outputDir}/local_report.md`)),
    readTextFile(resolveWorkspacePath(`tasks/${taskId}/code/generated/baseline_runner.py`))
      .then((text) => text || readTextFile(resolveWorkspacePath(`workspace/tasks/${taskId}/code/current_code/agent_draft.py`))),
    readJsonFile(`${outputDir}/hpc_gpu_training/metrics.json`),
    readJsonFile(`${outputDir}/score_improvement_gate.json`),
    readJsonFile(`${outputDir}/submission_audit.json`),
    readJsonFile(`${outputDir}/score_regression_recovery_plan.json`),
    taskId === "playground_series_s6e6" ? latestStrategyArtifact("s6e6_score_regression_diagnosis_") : Promise.resolve(null),
    readJsonFile(`${outputDir}/code_agent_result.json`),
    latestCodeAgentSession(taskId),
    latestCodeAgentReview(taskId)
  ]);
  const metricSnapshot = metrics
    ? {
        latest_metric: {
          best_validation_score: (metrics as Record<string, unknown>).ensemble && typeof (metrics as any).ensemble.best_validation_score === "number"
            ? (metrics as any).ensemble.best_validation_score
            : (metrics as Record<string, unknown>).best_validation_score,
          oof_balanced_accuracy: (metrics as Record<string, unknown>).oof_balanced_accuracy,
          oof_log_loss: (metrics as Record<string, unknown>).oof_log_loss
        },
        score_gate_status: (scoreImprovementGate as Record<string, unknown> | null)?.status,
        score_gate_blocked_reasons: (scoreImprovementGate as Record<string, unknown> | null)?.blocked_reasons,
        code_agent_result: codeAgentResult
      }
    : null;
  return {
    task_id: taskId,
    latest_experiment_dir: latest,
    latest_workstation_run_dir: latestWorkstationRun,
    latest_score_gated_run_dir: latestScoreGatedRun,
    task_state: taskState,
    agent_trace: readJsonl(agentTraceText),
    event_log: readJsonl(eventLogText),
    artifact_manifest: artifactManifest,
    evidence_graph: evidenceGraph,
    experiment_graph: experimentGraph,
    reflection,
    memory,
    gate_engine: gateEngine,
    runtime_snapshot: metricSnapshot ?? runtimeSnapshot ?? workspaceSummary,
    report_markdown: reportMarkdown,
    generated_code: generatedCode,
    training_log: [
      ...summarizeExperimentLog(experimentLog as Record<string, unknown> | null),
      ...(metrics ? [`[hpc] best_validation_score=${(metricSnapshot?.latest_metric as Record<string, unknown> | undefined)?.best_validation_score ?? "n/a"}; rows=${(metrics as Record<string, unknown>).train_rows ?? "n/a"}; runner=${(metrics as Record<string, unknown>).runner ?? "n/a"}`] : []),
      ...(scoreImprovementGate ? [`[score-gate] status=${(scoreImprovementGate as Record<string, unknown>).status}; reasons=${JSON.stringify((scoreImprovementGate as Record<string, unknown>).blocked_reasons ?? [])}`] : [])
    ],
    latest_code_agent_session: latestSession.session,
    latest_code_agent_transcript: latestSession.transcript,
    latest_code_agent_review: latestReview,
    score_improvement_gate: scoreImprovementGate as Record<string, unknown> | null,
    submission_audit: submissionAudit as Record<string, unknown> | null,
    score_regression_recovery_plan: scoreRegressionRecoveryPlan as Record<string, unknown> | null,
    score_regression_diagnosis: scoreRegressionDiagnosis as Record<string, unknown> | null
  };
}

export async function getWorkstationSummary() {
  await ensureWorkstationSeeded();
  const [tasks, runs, connectors, actions, gates, evidence, reports, workflows, runtimes, terminalAgent, finalDeliveryStatus, kaggleNewCompetitionReadiness, kaggleDpapiReadiness, kaggleExperimentInventory, top30NextEvolutionOrders, mlevolveAlignmentMatrix, mlebenchStyleLeaderboard, verifiedLaunchAudit, launchReadiness, learningLoopReadiness, hpcProbe, liveGpu, s6e6DependencyGate] = await Promise.all([
    prisma.task.findMany({ orderBy: { updatedAt: "desc" } }),
    prisma.experimentRun.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.connectorStatus.findMany({ orderBy: { provider: "asc" } }),
    prisma.actionLog.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.gate.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.evidence.findMany({ orderBy: { createdAt: "desc" }, take: 50 }),
    prisma.report.findMany({ orderBy: { updatedAt: "desc" }, take: 20 }),
    prisma.workflow.findMany({ orderBy: { updatedAt: "desc" }, take: 20 }),
    Promise.all(runtimeTaskIds.map((taskId) => loadRuntimeSummary(taskId))),
    loadXsciTerminalAgentSummary(),
    readJsonFile(resolveWorkspacePath("docs/final_delivery_status_20260612.json")),
    readJsonFile(resolveWorkspacePath("docs/kaggle_new_competition_readiness.json")),
    readJsonFile(resolveWorkspacePath("docs/kaggle_dpapi_readiness.json")),
    readJsonFile(resolveWorkspacePath("workspace/kaggle_experiment_inventory_20260624.json")),
    readJsonFile(resolveWorkspacePath("workspace/top30_next_evolution_orders_20260625.json")),
    readJsonFile(resolveWorkspacePath("workspace/mlevolve_alignment_matrix_20260625.json")),
    readJsonFile(resolveWorkspacePath("workspace/mlebench_style_current_leaderboard_20260625.json")),
    readJsonFile(resolveWorkspacePath("docs/verified_workstation_launch_audit.json")),
    readJsonFile(resolveWorkspacePath("workspace/workstation_launch_readiness_20260630.json")),
    readJsonFile(resolveWorkspacePath("workspace/workstation_learning_loop_readiness_20260630.json")),
    hpcGpuProbeStatus(),
    latestGpuSshConnectionStatus(),
    latestS6E6BoostingDependencyStatus()
  ]);
  const runtimeByTask = Object.fromEntries(runtimes.map((item) => [item.task_id, item]));
  const runtime = runtimes.find((item) => item.task_id === "house_prices" && item.latest_experiment_dir) ?? runtimes.find((item) => item.latest_experiment_dir) ?? runtimes[0];
  const runtimeTasks = runtimes
    .filter((item) => item.latest_experiment_dir)
    .map((item) => ({
      id: item.task_id,
      name: item.task_id.replaceAll("_", " "),
      task_type: "tabular_runtime",
      target: null,
      metric: String((item.runtime_snapshot?.latest_metric as Record<string, unknown> | undefined) ? Object.keys(item.runtime_snapshot?.latest_metric as Record<string, unknown>)[0] ?? "" : ""),
      status: String(item.task_state?.state ?? "runtime_ready"),
      priority: "Runtime",
      owner: "Research Agent Runtime",
      config_path: `configs/${item.task_id}.yaml`,
      task_dir: `tasks/${item.task_id}`,
      created_at: null,
      updated_at: null
    }));
  const runtimeTaskIdsSet = new Set(runtimeTasks.map((task) => task.id));

  const runtimeRun = runtimes
    .filter((item) => item.latest_experiment_dir)
    .map((item) => {
      const runtimeMetrics = item.runtime_snapshot?.latest_metric as Record<string, number> | undefined;
      const runtimeRunId = typeof item.latest_experiment_dir === "string" ? item.latest_experiment_dir.split(/[\\/]/).pop() : undefined;
      return {
          id: runtimeRunId,
          task_id: item.task_id,
          output_dir: item.latest_experiment_dir,
          status: "passed",
          best_model: ((item.experiment_graph as any)?.nodes?.[0]?.model as string | undefined) ?? "runtime_baseline",
          best_metrics: runtimeMetrics ?? null,
          accepted: true,
          validation_gate: { status: "passed" },
          started_at: null,
          finished_at: null
        };
    });
  const dbRuns = runs.map((run) => {
    const metrics = decodeJson<Record<string, any>>(run.metricsJson);
    return {
      id: run.id,
      task_id: run.taskId,
      output_dir: run.outputDir,
      status: run.status,
      workstation_run: metrics?.workstation_run === true || run.id.startsWith("wr_"),
      direct_training_allowed: metrics?.direct_training_allowed === true,
      official_submission_allowed: metrics?.official_submission_allowed === true,
      workstation_run_manifest: run.outputDir ? `${run.outputDir}/workstation_run_manifest.json` : null,
      artifact_manifest: run.outputDir ? `${run.outputDir}/artifact_manifest.json` : null,
      best_model: run.bestModel,
      best_metrics: metrics,
      accepted: run.validationStatus === "passed" || run.status === "passed",
      validation_gate: run.validationStatus ? { status: run.validationStatus } : undefined,
      started_at: run.startedAt?.toISOString() ?? null,
      finished_at: run.finishedAt?.toISOString() ?? null
    };
  });
  const runKeys = new Set(runtimeRun.map((run) => run.output_dir));
  const gpuCredentialPresent = gpuSshStatus() === "configured";
  const deepSeekConfigured = deepSeekApiKeyStatus() === "configured";
  const codeAgentConfigured = claudeApiKeyStatus() === "configured" || deepSeekConfigured;
  const deepSeek = deepSeekConfig();
  const gpu = gpuSshConfig();
  const gpuPendingState = "GPU Environment Created / Web Terminal Ready / External SSH Pending";
  const liveGpuSummary = liveGpu.gpuSummary || "nvidia-smi evidence present";
  const gpuVerifiedState = `GPU Verified: ${liveGpuSummary} via SSH Gateway`;
  const gpuSshReadyState = `GPU SSH Gateway Ready: ${liveGpuSummary} / nvidia-smi smoke passed`;
  const gpuLegacyVerifiedState = "GPU Verified: 4 x NVIDIA A800-SXM4-80GB via Login Node / Web Terminal";
  const gpuLegacySshReadyState = "GPU SSH Gateway Ready: 4 x NVIDIA A800-SXM4-80GB / historical CUDA smoke passed";
  const s6e6GatewayBlocked = s6e6DependencyGate.status === "blocked_resource_gateway";
  const gpuFreshSmokeBlocked = liveGpu.present === true && liveGpu.passed === false;
  const liveGpuPassed = liveGpu.present === true && liveGpu.passed === true;
  const latestGpuAllocationBlocker = actions.find((action) => action.action === "gpu_current_allocation_blocker");
  const latestGpuAllocationBlockerMetadata = latestGpuAllocationBlocker
    ? decodeJson<Record<string, unknown>>(latestGpuAllocationBlocker.metadataJson)
    : null;
  const gpuCurrentAllocationBlocked = !liveGpuPassed && latestGpuAllocationBlockerMetadata?.status === "blocked_current_allocation";
  const kaggleDpapi = await kaggleDpapiProbeStatus(kaggleDpapiReadiness as Record<string, unknown> | null);

  return {
    tasks: [
      ...runtimeTasks,
      ...tasks.filter((task) => !runtimeTaskIdsSet.has(task.id.replaceAll("-", "_"))).map((task) => ({
      id: task.id,
      name: task.name,
      task_type: task.taskType,
      target: task.target,
      metric: task.metric,
      status: task.status,
      priority: task.priority,
      owner: task.owner,
      config_path: task.configPath,
      task_dir: task.taskDir,
      created_at: task.createdAt.toISOString(),
      updated_at: task.updatedAt.toISOString()
      }))
    ],
    connector_status: Object.fromEntries(
      [
        ...connectors.filter((connector) => !["code_agent", "gpu", "kaggle"].includes(connector.provider)).map((connector) => [
          connector.provider,
          {
            name: connector.name,
            state: connector.state,
            configured: connector.configured,
            notes: connector.detail
          }
        ] as const),
        [
          "code_agent",
          {
            name: "Code Agent",
            state: claudeApiKeyStatus() === "configured"
              ? "Claude Agent SDK Ready"
              : deepSeekConfigured
                ? `DeepSeek Code Agent Ready (${deepSeek.model})`
                : "Not Configured",
            configured: codeAgentConfigured,
            notes: claudeApiKeyStatus() === "configured"
              ? "ANTHROPIC_API_KEY detected; SDK sessions can run through gated patch flow."
              : deepSeekConfigured
                ? "DEEPSEEK_API_KEY detected; Claude-Code-like patch drafts run through DeepSeek and still require review/manual gates."
                : "Set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY to enable real Code Agent sessions.",
            fallback_provider: deepSeekConfigured && claudeApiKeyStatus() !== "configured" ? "deepseek_code_agent" : undefined
          }
        ] as const,
        [
          "deepseek",
          {
            name: "DeepSeek",
            state: deepSeekConfigured ? `DeepSeek Ready (${deepSeek.model})` : "Not Configured",
            configured: deepSeekConfigured,
            notes: deepSeekConfigured ? "DEEPSEEK_API_KEY detected; general research LLM calls can run through audited smoke route." : "Set DEEPSEEK_API_KEY to enable DeepSeek research LLM calls.",
            model: deepSeek.model,
            base_url: deepSeek.baseUrl
          }
        ] as const,
        [
          "gpu",
          {
            name: "GPU SSH Gateway",
            state: gpuCurrentAllocationBlocked
              ? `GPU Blocked: current allocation ${String(latestGpuAllocationBlockerMetadata?.host ?? "unknown")}:${String(latestGpuAllocationBlockerMetadata?.port ?? "unknown")} closed before SSH handshake`
              : gpuFreshSmokeBlocked
              ? `GPU Blocked: fresh SSH/CUDA smoke failed (${liveGpu.path ?? "no artifact"})`
              : s6e6GatewayBlocked
              ? `GPU Blocked: historical NVIDIA A800 evidence exists, but current resource gate is blocked (${s6e6DependencyGate.blocker ?? "resource gateway unavailable"})`
              : gpuCredentialPresent && liveGpu.passed
              ? gpuSshReadyState
              : liveGpu.passed
                ? gpuVerifiedState
                : gpuCredentialPresent && hpcProbe.fullyReadyAllowed
                  ? gpuLegacySshReadyState
                  : hpcProbe.fullyReadyAllowed
                    ? gpuLegacyVerifiedState
                : gpuPendingState,
            configured: gpuCredentialPresent,
            current_allocation_blocked: gpuCurrentAllocationBlocked || gpuFreshSmokeBlocked || s6e6GatewayBlocked,
            current_gate_ready: gpuCredentialPresent && !gpuCurrentAllocationBlocked && !gpuFreshSmokeBlocked && !s6e6GatewayBlocked && liveGpu.passed === true,
            notes: gpuCurrentAllocationBlocked
              ? `A newer rotating GPU allocation failed fresh SSH validation. Host=${String(latestGpuAllocationBlockerMetadata?.host ?? "unknown")}, port=${String(latestGpuAllocationBlockerMetadata?.port ?? "unknown")}, direct TCP=${String(latestGpuAllocationBlockerMetadata?.tcp_direct ?? "unknown")}, SSH=${String(latestGpuAllocationBlockerMetadata?.ssh_direct ?? "unknown")}. Historical A800 evidence remains archived, but workstation training is blocked until a fresh allocation passes SSH/CUDA smoke.`
              : gpuFreshSmokeBlocked
              ? `DPAPI SSH credentials are loaded, but the latest GPU SSH/CUDA smoke failed. Latest evidence: ${liveGpu.path ?? "missing"}. Historical A800 evidence remains archived; workstation training is blocked until /api/gpu/connections/test passes on the current allocation.`
              : s6e6GatewayBlocked
              ? `Historical NVIDIA A800 GPU evidence exists, but the current S6E6 dependency gate is blocked before training. Blocker: ${s6e6DependencyGate.blocker ?? "resource gateway unavailable"}. Latest gate: ${s6e6DependencyGate.path}. Next action: ${s6e6DependencyGate.nextAction ?? "refresh the rotating GPU allocation and rerun the dependency gate"}.`
              : gpuCredentialPresent && liveGpu.passed
              ? `Windows DPAPI credentials are present and the project SSH helper reached the current GPU allocation through the documented 127.0.0.1:7890 SOCKS5 bridge. Latest evidence: ${liveGpu.path}. Python runtime: ${liveGpu.pythonRuntime ?? "unknown"}; torch import: ${liveGpu.torchImport === null ? "unknown" : String(liveGpu.torchImport)}. GPU jobs remain whitelist-template only; arbitrary shell is not exposed.`
              : liveGpu.passed
                ? `Latest SSH gateway evidence proves ${liveGpuSummary}, but no loaded SSH credential is present for automated jobs.`
                : gpuCredentialPresent && hpcProbe.fullyReadyAllowed
                  ? "Windows DPAPI credentials are present and historical Web Terminal evidence proves 4 x A800. Run a fresh GPU SSH smoke to refresh the current allocation summary."
                  : hpcProbe.fullyReadyAllowed
                    ? "HPC login node + Web Terminal evidence proves 4 x A800, but the workstation does not currently have a loaded SSH credential for automated jobs."
                : "HPC login node is reachable through the PDF ncat path, but the platform GPU environment SSH endpoint is still external-pending; confirm nvidia-smi in Web Terminal before marking fully ready.",
            proxy: gpu.socksProxy.host ? "socks5" : "direct",
            evidence: {
              hpc_probe: hpcProbe,
              latest_ssh_connection: liveGpu,
              latest_s6e6_dependency_gate: s6e6DependencyGate,
              latest_current_allocation_blocker: latestGpuAllocationBlocker
                ? {
                  action_id: latestGpuAllocationBlocker.id,
                  artifact: latestGpuAllocationBlocker.artifactPath,
                  metadata: latestGpuAllocationBlockerMetadata,
                  at: latestGpuAllocationBlocker.createdAt.toISOString()
                }
                : null
            }
          }
        ] as const,
        [
          "kaggle",
          {
            name: "Kaggle",
            state: kaggleDpapi.configured
              ? "Kaggle DPAPI Token Ready"
              : kaggleDpapi.toolchain_ready
                ? "Kaggle Toolchain Installed / Token Not Configured"
                : "Not Configured",
            configured: kaggleDpapi.configured,
            notes: kaggleDpapi.configured
              ? "Kaggle token is available through Windows DPAPI or the current process environment. Official download can be smoke-tested with -AllowRealExternal; leaderboard submission still requires Human Gate."
              : kaggleDpapi.toolchain_ready
                ? `Kaggle Python/CLI is ready (${String(kaggleDpapi.python_package_version ?? "installed")}), but Windows DPAPI token is not installed. Local Kaggle-style runs remain available.`
                : "Install Kaggle Python/CLI and store the official token with Windows DPAPI before API download.",
            credential_status: kaggleDpapi.credential_status,
            token_type: kaggleDpapi.token_type,
            toolchain_ready: kaggleDpapi.toolchain_ready,
            python_package_version: kaggleDpapi.python_package_version,
            cli_path: kaggleDpapi.cli_path,
            report_path: kaggleDpapi.report_path,
            safe_install_command: kaggleDpapi.safe_install_command,
            real_api_smoke_command: kaggleDpapi.real_api_smoke_command,
            human_gate_required_for_submission: kaggleDpapi.human_gate_required_for_submission
          }
        ] as const,
        [
          "env_keys",
          {
            CODE_AGENT_PROVIDER: claudeApiKeyStatus() === "configured" ? "claude_agent_sdk" : deepSeekConfigured ? "deepseek_code_agent" : "not_configured",
            CLAUDE_API_KEY_STATUS: claudeApiKeyStatus(),
            PYTHON_RUNNER: "local",
            GPU_PROVIDER: "ssh_gateway",
            GPU_SSH_STATUS: gpuSshStatus(),
            GPU_SSH_PROXY: gpu.socksProxy.host ? "socks5" : "direct",
            KAGGLE_ENABLED: kaggleDpapi.configured ? "true" : "false",
            KAGGLE_TOKEN_STATUS: kaggleDpapi.credential_status,
            KAGGLE_TOOLCHAIN_STATUS: kaggleDpapi.toolchain_ready ? "ready" : "missing",
            LLM_PROVIDER: "rule_based",
            DEEPSEEK_API_KEY_STATUS: deepSeekApiKeyStatus(),
            DEEPSEEK_MODEL: deepSeek.model,
            DATABASE_PROVIDER: "sqlite"
          }
        ] as const
      ]
    ),
    runs: [...runtimeRun, ...dbRuns.filter((run) => !run.output_dir || !runKeys.has(run.output_dir))],
    actions: actions.map((action) => ({
      id: action.id,
      action: action.action,
      task_id: action.taskId,
      run_id: action.runId,
      message: action.message,
      artifact: action.artifactPath,
      metadata: decodeJson(action.metadataJson),
      at: action.createdAt.toISOString()
    })),
    gates: gates.map((gate) => ({
      id: gate.id,
      task_id: gate.taskId,
      run_id: gate.runId,
      gate_type: gate.gateType,
      decision: gate.decision,
      reviewer: gate.reviewer,
      evidence: decodeJson(gate.evidenceJson),
      created_at: gate.createdAt.toISOString(),
      decided_at: gate.decidedAt?.toISOString() ?? null
    })),
    evidence: evidence.map((item) => ({
      id: item.id,
      task_id: item.taskId,
      run_id: item.runId,
      label: item.label,
      artifact_path: item.artifactPath,
      hash: item.hash,
      source: item.source,
      claim_binding: item.claimBinding,
      created_at: item.createdAt.toISOString()
    })),
    reports: reports.map((report) => ({
      id: report.id,
      task_id: report.taskId,
      run_id: report.runId,
      title: report.title,
      status: report.status,
      markdown_content: report.markdownContent,
      content: decodeJson(report.contentJson),
      markdown_path: report.markdownPath,
      docx_path: report.docxPath,
      selected_section: report.selectedSection,
      submitted_at: report.submittedAt?.toISOString() ?? null
    })),
    workflows: workflows.map((workflow) => ({
      id: workflow.id,
      task_id: workflow.taskId,
      name: workflow.name,
      status: workflow.status,
      version: workflow.version,
      nodes: decodeJson(workflow.nodesJson),
      edges: decodeJson(workflow.edgesJson),
      published_at: workflow.publishedAt?.toISOString() ?? null
    })),
    stages,
    final_delivery_status: finalDeliveryStatus,
    kaggle_new_competition_readiness: kaggleNewCompetitionReadiness,
    kaggle_dpapi_readiness: kaggleDpapi,
    kaggle_experiment_inventory: kaggleExperimentInventory,
    top30_next_evolution_orders: top30NextEvolutionOrders,
    mlevolve_alignment_matrix: mlevolveAlignmentMatrix,
    mlebench_style_leaderboard: mlebenchStyleLeaderboard,
    verified_launch_audit: {
      ...(verifiedLaunchAudit as Record<string, unknown> | null ?? {}),
      latest_readiness: launchReadiness,
      launch_state: (launchReadiness as Record<string, unknown> | null)?.launch_state ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.launch_state ?? null,
      blockers: (launchReadiness as Record<string, unknown> | null)?.blockers ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.blockers ?? [],
      critical_failures: (launchReadiness as Record<string, unknown> | null)?.critical_failures ?? [],
      soft_failures: (launchReadiness as Record<string, unknown> | null)?.soft_failures ?? []
    },
    learning_loop_readiness: learningLoopReadiness,
    runtime,
    runtime_by_task: runtimeByTask,
    terminal_agent: terminalAgent,
    workspace_root: workspaceRoot
  };
}
