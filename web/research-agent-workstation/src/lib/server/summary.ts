import { promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
import type { ActionLog, ConnectorStatus, Evidence, ExperimentRun, Gate, Report, Workflow } from "@prisma/client";
import { listHpcJobLineage } from "@/lib/server/hpc-job-lineage";
import { normalizeConnectorRegistry } from "@/lib/server/connector-health";
import { readCurrentRunLedger } from "@/lib/server/run-ledger";
import { claudeApiKeyStatus, deepSeekApiKeyStatus, deepSeekConfig, gpuSshConfig, gpuSshStatus, openAiApiKeyStatus, openAiConfig } from "@/lib/server/capabilities";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { decodeJson, sanitizeClientJson } from "@/lib/server/json";
import { latestExperimentPath, latestScoreGatedWorkstationRunPath, latestWorkstationRunPath, readJsonFile as readJsonFileRaw, resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";
import { loadLiteratureStateForAllTasks } from "@/lib/server/literature-agent-actions";
import {
  selectLatestValidLiteratureManifest,
  type LiteratureManifestCandidate
} from "@/lib/server/literature-manifest-selection";

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
  current_run?: Record<string, unknown> | null;
  task_graph?: Record<string, unknown> | null;
  handoffs?: Array<Record<string, unknown>>;
  review?: Record<string, unknown> | null;
  hpc_probe?: Record<string, unknown> | null;
  preserved_parent_run_id?: string | null;
  preserved_parent_artifact_manifest?: Record<string, unknown> | null;
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
const requiredRuntimeArtifacts = [
  "task_state_machine.json",
  "agent_trace.jsonl",
  "event_log.jsonl",
  "artifact_manifest.json",
  "evidence_index.json",
  "experiment_graph.json",
  "gate_engine.json",
  "gate_audit_log.jsonl",
  "runtime_snapshot.json",
  "reflection.json",
  "reflection.md",
  "memory_records.json",
  "orchestrator_run.json",
  "validation_gate.json"
];

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

type FileCacheEntry<T> = {
  value: T;
  mtimeMs: number;
  expiresAt: number;
};

const FILE_CACHE_TTL_MS = Math.max(1_000, Number(process.env.WORKSTATION_FILE_CACHE_TTL_MS ?? 15_000));
const FULL_SUMMARY_TTL_MS = Math.max(2_000, Number(process.env.WORKSTATION_SUMMARY_CACHE_TTL_MS ?? 15_000));
const LIGHT_SUMMARY_TTL_MS = Math.max(250, Number(process.env.WORKSTATION_LIGHT_SUMMARY_CACHE_TTL_MS ?? 1_000));
const jsonFileCache = new Map<string, FileCacheEntry<any>>();
const textFileCache = new Map<string, FileCacheEntry<string>>();

async function cachedFileValue<T>(
  filePath: string,
  cache: Map<string, FileCacheEntry<T>>,
  loader: () => Promise<T>,
): Promise<T> {
  const now = Date.now();
  const cached = cache.get(filePath);
  if (cached && cached.expiresAt > now) return cached.value;

  const mtimeMs = await fs.stat(filePath).then((stat) => stat.mtimeMs).catch(() => 0);
  if (cached && cached.mtimeMs === mtimeMs) {
    cached.expiresAt = now + FILE_CACHE_TTL_MS;
    return cached.value;
  }

  const value = await loader();
  cache.set(filePath, { value, mtimeMs, expiresAt: now + FILE_CACHE_TTL_MS });
  return value;
}

async function readJsonFile(filePath: string) {
  return cachedFileValue(filePath, jsonFileCache, () => readJsonFileRaw(filePath));
}

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

function runtimeEventsFromAgentTrace(agentTrace: Array<Record<string, unknown>>) {
  return agentTrace.map((event, index) => ({
    source: "agent_trace_projection",
    index,
    event_type: event.event_type ?? event.type ?? event.stage ?? "agent_step",
    stage: event.stage ?? event.agent ?? event.event_type ?? `step_${index + 1}`,
    status: event.status ?? "recorded",
    message: event.message ?? event.summary ?? event.action ?? event.note ?? "",
    artifact_path: event.artifact_path ?? event.artifact ?? null,
    raw: event
  }));
}

function safeSummaryText(value: unknown, fallback = "") {
  const text = typeof value === "string" ? value : value === null || value === undefined ? fallback : String(value);
  if (!text) return fallback;
  if (/(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]/i.test(text)) {
    return "[redacted-sensitive-text]";
  }
  return text.length > 700 ? `${text.slice(0, 697)}...` : text;
}

function inferTaskFromRunId(runId: string) {
  return runId.replace(/_(gpu|local)_\d{8}_\d{6}$/i, "");
}

async function readTextFile(filePath: string) {
  return cachedFileValue(filePath, textFileCache, () => fs.readFile(filePath, "utf-8").catch(() => ""));
}

async function fileStatMtime(path: string) {
  const fs = await import("node:fs/promises");
  return fs.stat(path).then((stat) => stat.mtimeMs).catch(() => 0);
}

async function loadXsciTerminalAgentSummary() {
  const fs = await import("node:fs/promises");
  const root = resolveWorkspacePath("experiments/evolution");
  // TerminalEventEmitter writes to the workspace root. Keep the historical
  // .xsci sink as a compatibility source for older runs, then merge by event
  // identity so Runtime shows the latest terminal action exactly once.
  const terminalEventSources = [
    { relativePath: "terminal_events.jsonl", absolutePath: resolveWorkspacePath("terminal_events.jsonl") },
    { relativePath: ".xsci/terminal_events.jsonl", absolutePath: resolveWorkspacePath(".xsci/terminal_events.jsonl") }
  ];
  const terminalEventRows = (await Promise.all(terminalEventSources.map(async (source, sourceIndex) => {
    const text = await readTextFile(source.absolutePath);
    return readJsonl(text).map((event, eventIndex) => ({ event, sourceIndex, eventIndex }));
  }))).flat();
  terminalEventRows.sort((left, right) => {
    const leftTime = Date.parse(String(left.event.ts ?? ""));
    const rightTime = Date.parse(String(right.event.ts ?? ""));
    if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) return leftTime - rightTime;
    return left.sourceIndex - right.sourceIndex || left.eventIndex - right.eventIndex;
  });
  const terminalEventKeys = new Set<string>();
  const terminalEvents = terminalEventRows
    .filter(({ event }) => {
      const key = [event.type, event.stage, event.message, event.status, event.artifact, event.ts].map((value) => String(value ?? "")).join("\u0001");
      if (terminalEventKeys.has(key)) return false;
      terminalEventKeys.add(key);
      return true;
    })
    .map(({ event }) => event);
  const terminalEventsPath = terminalEventRows.some(({ sourceIndex }) => sourceIndex === 0)
    ? terminalEventSources[0].relativePath
    : terminalEventRows.some(({ sourceIndex }) => sourceIndex === 1)
      ? terminalEventSources[1].relativePath
      : null;
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
  const displayEvents = terminalEvents.length ? terminalEvents : events;
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
    status: displayEvents.length ? "live_events" : latest ? summary ? "summary_only" : "pending_run" : "no_runs",
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
    event_count: displayEvents.length,
    recent_events: displayEvents.slice(-12).map((event) => sanitizeSummaryRecord(event)),
    terminal_events_path: terminalEvents.length ? terminalEventsPath : null,
    terminal_event_count: terminalEvents.length,
    recent_terminal_events: terminalEvents.slice(-20).map((event) => sanitizeSummaryRecord(event)),
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

async function loadLatestLiteratureSummary() {
  const fs = await import("node:fs/promises");
  const tasksRoot = resolveWorkspacePath("workspace/tasks");
  const taskEntries = await fs.readdir(tasksRoot, { withFileTypes: true }).catch(() => []);
  const candidates: LiteratureManifestCandidate[] = [];
  for (const taskEntry of taskEntries.filter((entry) => entry.isDirectory())) {
    const ragRoot = path.join(tasksRoot, taskEntry.name, "rag");
    const entries = await fs.readdir(ragRoot, { withFileTypes: true }).catch(() => []);
    const taskCandidates: LiteratureManifestCandidate[] = [];
    for (const entry of entries.filter((item) => item.isFile() && /^context_.*\.json$/i.test(item.name))) {
      const absolutePath = path.join(ragRoot, entry.name);
      const [stat, payload] = await Promise.all([
        fs.stat(absolutePath).catch(() => null),
        readJsonFile(absolutePath) as Promise<Record<string, unknown> | null>
      ]);
      if (stat?.isFile() && payload) {
        taskCandidates.push({
          absolutePath,
          expectedTaskId: taskEntry.name,
          mtimeMs: stat.mtimeMs,
          payload
        });
      }
    }
    const selected = selectLatestValidLiteratureManifest(taskCandidates);
    if (selected) candidates.push(selected);
  }
  const latest = candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  if (!latest) return { present: false, papers: [], claim_audit: [], context_path: null, manifest_path: null, source_errors: [], integrity: null };
  const payload = latest.payload as Record<string, unknown>;
  return {
    ...payload,
    present: true,
    manifest_path: toRelativeLiteraturePath(latest.absolutePath),
    context_path: typeof payload.context_path === "string" ? payload.context_path : null,
    papers: Array.isArray(payload.papers) ? payload.papers.slice(0, 40) : [],
    claim_audit: Array.isArray(payload.claim_audit) ? payload.claim_audit.slice(0, 40) : [],
    source_errors: Array.isArray(payload.source_errors) ? payload.source_errors : [],
    integrity: payload.integrity ?? null
  };
}

function toRelativeLiteraturePath(absolutePath: string) {
  return path.relative(workspaceRoot, absolutePath).replaceAll("\\", "/");
}

async function loadScientistAutopilotSummary() {
  const relativePath = ".xsci/scientist_autopilot.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      mode: "not_run",
      selected_task: null,
      summary_lines: [],
      tool_trace: [],
      next_actions: ["Run Scientist Autopilot from the Control page or `evomind autopilot`."],
      blockers: [],
      human_gate: {
        official_kaggle_submit: "blocked_until_explicit_user_approval",
        rank_or_medal_claims: "blocked_without_kaggle_response_artifact"
      }
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistActionQueueSummary() {
  const relativePath = ".xsci/scientist_action_queue.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_action_queue",
      selected_task: null,
      actions: [
        {
          id: "run_autopilot_first",
          title: "Run AI Scientist Autopilot",
          status: "ready",
          command: "evomind autopilot",
          gate: "read_only",
          why: "Create the first action queue from current system, task, data, memory, and gate evidence.",
          risk: "none; read-only diagnosis",
          rollback_condition: "stay in planner mode until action queue exists",
          expected_artifacts: [".xsci/scientist_autopilot.json", ".xsci/scientist_action_queue.json"],
          evidence: ["scientist_autopilot", "scientist_step_trace"],
          autonomy: "read_only"
        }
      ],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistContinuationStatusSummary() {
  const relativePath = ".xsci/scientist_continuation_status.json";
  const continuationPath = ".xsci/scientist_continuation.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      continuation_artifact_path: continuationPath,
      tool: "scientist_continuation_status",
      selected_task: null,
      status: "no_continuation",
      completion_ratio: 0,
      total_required_tools: 0,
      completed_required_tools: 0,
      remaining_count: 0,
      remaining_safe_tools: [],
      executed_or_completed_tools: [],
      progress_history: [],
      next_safe_action_command: "evomind turn",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist continuation status has not been generated yet."
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistLoopSummary() {
  const relativePath = ".xsci/scientist_loop.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_loop",
      mode: "not_run",
      stop_reason: "not_run",
      selected_task: null,
      steps: [],
      final_autopilot: null,
      final_next_action: null,
      lesson: null,
      lessons_path: ".xsci/scientist_loop_lessons.jsonl",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistLoopLessonsSummary() {
  const relativePath = ".xsci/scientist_loop_lessons.jsonl";
  const text = await readTextFile(resolveWorkspacePath(relativePath));
  const entries = readJsonl(text).slice(-20).map((event) => sanitizeSummaryRecord(event));
  return {
    present: entries.length > 0,
    artifact_path: relativePath,
    count: entries.length,
    latest: entries.at(-1) ?? null,
    recent: sanitizeClientJson(entries)
  };
}

async function loadScientistMemoryConsolidationSummary() {
  const relativePath = ".xsci/scientist_memory_consolidation.json";
  const memoryPath = "experiments/evolution/retrospective_memory.json";
  const [payload, memoryPayload] = await Promise.all([
    readJsonFile(resolveWorkspacePath(relativePath)) as Promise<Record<string, unknown> | null>,
    readJsonFile(resolveWorkspacePath(memoryPath))
  ]);
  const memoryRecords = Array.isArray(memoryPayload)
    ? memoryPayload.map((item) => asRecordForSummary(item)).filter(Boolean)
    : [];
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      memory_path: memoryPath,
      tool: "scientist_memory_consolidation",
      records_before: memoryRecords.length,
      candidate_records: 0,
      records_added: 0,
      records_total: memoryRecords.length,
      added_memory_ids: [],
      source_counts: {},
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    memory_path: memoryPath,
    records_total: memoryRecords.length || payload.records_total || 0,
    ...payload
  });
}

async function loadScientistSelfAuditSummary() {
  const relativePath = ".xsci/scientist_self_audit.json";
  const backlogPath = ".xsci/scientist_upgrade_backlog.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      backlog_artifact_path: backlogPath,
      tool: "scientist_self_audit",
      overall_score: 0,
      launch_readiness: "not_run",
      capabilities: [],
      gaps: [],
      upgrade_backlog: [
        {
          id: "run_self_audit_first",
          title: "Run EvoMind self-audit",
          priority: "P0",
          status: "ready",
          safe_next_command: "evomind self-audit",
          gate: "read_only"
        }
      ],
      evidence_sources: {},
      next_safe_commands: ["evomind self-audit"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistReadinessReportSummary() {
  const relativePath = ".xsci/scientist_readiness_report.json";
  const markdownPath = ".xsci/scientist_readiness_report.md";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_readiness_report",
      overall_score: 0,
      capability_readiness: "not_run",
      launch_readiness: "not_run",
      claim_readiness: {
        training_readiness_claim: "not_run",
        rank_or_medal_claim: "blocked_without_kaggle_response_artifact",
        official_submit_claim: "blocked_until_explicit_human_approval"
      },
      readiness_matrix: [],
      recommended_next_commands: ["evomind readiness-report", "evomind self-audit"],
      artifact_evidence: [],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    markdown_artifact_path: markdownPath,
    ...payload
  });
}

async function loadScientistCausalDiagnosisSummary() {
  const relativePath = ".xsci/scientist_causal_diagnosis.json";
  const markdownPath = ".xsci/scientist_causal_diagnosis.md";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_causal_diagnosis",
      posture: "not_run",
      symptoms: [],
      root_causes: [],
      interventions: [],
      causal_graph: { nodes: [], edges: [] },
      next_safe_command: "evomind causal-diagnosis",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    markdown_artifact_path: markdownPath,
    ...payload
  });
}

async function loadScientistStrategyOptimizerSummary() {
  const relativePath = ".xsci/scientist_strategy_optimizer.json";
  const markdownPath = ".xsci/scientist_strategy_optimizer.md";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_strategy_optimizer",
      strategy_posture: "not_run",
      selected_strategy: null,
      intervention_ranking: [],
      decision_matrix: { candidate_count: 0, source_presence: {} },
      next_safe_command: "evomind strategy",
      claim_boundary: {
        rank_or_medal: "blocked_without_kaggle_response_artifact",
        official_submit: "blocked_until_explicit_human_approval"
      },
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    markdown_artifact_path: markdownPath,
    ...payload
  });
}

async function loadScientistContextPacketSummary() {
  const relativePath = ".xsci/scientist_context_packet.json";
  const markdownPath = ".xsci/scientist_context_packet.md";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_context_packet",
      schema: "evomind.ai_scientist.context_packet.v1",
      context_quality: {
        score: 0,
        present_artifacts: 0,
        missing_sources: ["scientist_context_packet.json"],
        interpretation: "not_run"
      },
      readiness: {
        can_execute: false,
        blocking_gates: []
      },
      active_strategy: {
        present: false,
        selected_command: "evomind briefing",
        gate_status: "safe_read_only"
      },
      memory_digest: {
        retrospective_records: 0,
        recent_lessons: []
      },
      next_safe_command: "evomind briefing",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    markdown_artifact_path: markdownPath,
    ...payload
  });
}

async function loadScientistReasoningSynthesisSummary() {
  const relativePath = ".xsci/scientist_reasoning_synthesis.json";
  const markdownPath = ".xsci/scientist_reasoning_synthesis.md";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      markdown_artifact_path: markdownPath,
      tool: "scientist_reasoning_synthesis",
      reasoning_mode: "not_run",
      direct_answer: "",
      hypotheses: [],
      comparison: [],
      reasoning_quality: {
        score: 0,
        status: "not_run",
        hypotheses_requested: 0,
        hypotheses_produced: 0,
        complete_falsifiable_hypotheses: 0
      },
      next_safe_action: {
        command: "evomind ask \"analyze the current task\"",
        gate: "safe_scientist_turn"
      },
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    markdown_artifact_path: markdownPath,
    ...payload
  });
}

async function loadScientistTerminalTurnSummary() {
  const relativePath = ".xsci/scientist_terminal_turn.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_terminal_turn",
      selected_task: null,
      user_goal: "",
      executed_tools: [],
      artifacts: [],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistEngineeringLoopSummary() {
  const relativePath = ".xsci/scientist_engineering_loop.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_engineering_loop",
      status: "not_run",
      changed_files: [],
      acceptance_checks: [],
      main_worktree_modified: false,
      merge_ready: false,
      next_safe_command: "evomind patch-order",
      human_gate: "review_candidate_before_merge",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistInnovationBacklogSummary() {
  const relativePath = ".xsci/scientist_innovation_backlog.json";
  const innovationLogPath = ".xsci/innovation_log.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      innovation_log_path: innovationLogPath,
      tool: "scientist_innovation_backlog",
      selected_task: null,
      memory_summary: {},
      innovation_hypotheses: [],
      next_safe_commands: ["evomind innovate-plan"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistHypothesisReviewSummary() {
  const relativePath = ".xsci/scientist_hypothesis_review.json";
  const backlogPath = ".xsci/scientist_innovation_backlog.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      source_backlog_path: backlogPath,
      tool: "scientist_hypothesis_review",
      selected_task: null,
      hypotheses_reviewed: 0,
      reviews: [],
      selected_hypothesis: null,
      recommendation: "not_run",
      gate_summary: {},
      next_safe_commands: ["evomind review-hypotheses"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistExperimentBlueprintSummary() {
  const relativePath = ".xsci/scientist_experiment_blueprint.json";
  const reviewPath = ".xsci/scientist_hypothesis_review.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      source_review_path: reviewPath,
      tool: "scientist_experiment_blueprint",
      selected_task: null,
      blueprint_status: "not_run",
      selected_hypothesis: null,
      experiment_blueprint: null,
      gate_summary: {},
      next_safe_commands: ["evomind blueprint"],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistSituationModelSummary() {
  const relativePath = ".xsci/scientist_situation_model.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_situation_model",
      selected_task: null,
      situation_status: "not_run",
      situation_model: null,
      readiness_score: 0,
      blockers: [],
      next_safe_commands: ["evomind situation"],
      source_artifacts: [],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist situation model has not been generated yet."
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistTurnPlanSummary() {
  const relativePath = ".xsci/scientist_turn_plan.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      tool: "scientist_turn_plan",
      selected_task: null,
      intent: { kind: "not_run" },
      autonomy_level: "not_run",
      selected_tools: [],
      tool_sequence: [],
      expected_artifacts: [],
      stop_conditions: [],
      next_safe_command: "evomind turn-plan",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      message: "Scientist turn plan has not been generated yet."
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistWorkplanSummary() {
  const relativePath = ".xsci/scientist_workplan.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      mode: "not_run",
      current_focus: null,
      summary: { steps_total: 0, completed: 0, ready: 0, pending: 0, blocked: 0 },
      steps: [],
      resume_commands: ["Run Scientist Autopilot from the Control page or `evomind workplan`."],
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistRepairPlanSummary() {
  const relativePath = ".xsci/scientist_repair_plan.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      mode: "not_run",
      diagnosis: [],
      root_causes: [],
      repair_steps: [],
      safe_next_command: "Run Scientist Autopilot or `evomind repair` to create the first repair plan.",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistExecutionContractSummary() {
  const relativePath = ".xsci/scientist_execution_contract.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      go_no_go: "not_run",
      agent_session_ready: false,
      model_training_ready: false,
      data_contract_status: "unknown",
      required_artifacts: [],
      execution_gate_decision: {
        ok: true,
        blocked: true,
        status: "not_run",
        require_model_ready: true,
        blocked_by: ["execution_contract_not_generated"],
        root_causes: ["missing_execution_contract"],
        setup_blockers: ["Run `evomind contract` or Scientist Autopilot to generate the execution gate evidence."],
        safe_next_commands: ["evomind contract", "evomind ready"],
        message: "No Scientist execution contract has been generated yet.",
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      },
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval",
      claim_boundary: "No execution-contract evidence has been generated yet."
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

async function loadScientistTurnsSummary() {
  const relativePath = ".xsci/scientist_turns.jsonl";
  const text = await readTextFile(resolveWorkspacePath(relativePath));
  const entries = readJsonl(text).slice(-12).map((event) => sanitizeSummaryRecord(event));
  return {
    present: entries.length > 0,
    artifact_path: relativePath,
    count: entries.length,
    latest: entries.at(-1) ?? null,
    recent: sanitizeClientJson(entries)
  };
}

async function loadScientistStepTraceSummary() {
  const relativePath = ".xsci/scientist_step_trace.jsonl";
  const text = await readTextFile(resolveWorkspacePath(relativePath));
  const entries = readJsonl(text).slice(-50).map((event) => sanitizeSummaryRecord(event));
  return {
    present: entries.length > 0,
    artifact_path: relativePath,
    count: entries.length,
    latest: entries.at(-1) ?? null,
    recent: sanitizeClientJson(entries)
  };
}

async function loadScientistAutopilotStatusSummary() {
  const relativePath = ".xsci/scientist_autopilot_status.json";
  const payload = await readJsonFile(resolveWorkspacePath(relativePath)) as Record<string, unknown> | null;
  if (!payload) {
    return {
      present: false,
      artifact_path: relativePath,
      running: false,
      status: "not_started",
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    };
  }
  return sanitizeClientJson({
    present: true,
    artifact_path: relativePath,
    ...payload
  });
}

function asRecordForSummary(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function finiteSummaryNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function siimHpcRuntimeWithEvidence(
  runtime: unknown,
  trainingHistory: unknown,
  telemetryText: string,
): Record<string, unknown> | null {
  const runtimeRecord = asRecordForSummary(runtime);
  if (!Object.keys(runtimeRecord).length) return null;
  const history = asRecordForSummary(trainingHistory);
  const formalSeeds = Array.isArray(history.formal_seeds)
    ? history.formal_seeds.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : Array.isArray(runtimeRecord.formal_seeds)
      ? runtimeRecord.formal_seeds.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
      : [];
  const sourceRuns = Array.isArray(history.source_runs)
    ? history.source_runs.map(asRecordForSummary)
    : [];
  const foldRecords = sourceRuns.flatMap((sourceRun) => {
    const sourceHistory = asRecordForSummary(sourceRun.history);
    return Array.isArray(sourceHistory.folds) ? sourceHistory.folds.map(asRecordForSummary) : [];
  });
  const selectedEpochs = foldRecords
    .map((record) => finiteSummaryNumber(record.selected_epoch))
    .filter((value): value is number => value !== null);
  const allocatedMemory = foldRecords.flatMap((record) => [
    finiteSummaryNumber(record.selection_peak_memory_allocated_mib),
    finiteSummaryNumber(record.peak_memory_allocated_mib),
  ]).filter((value): value is number => value !== null);
  const telemetry = readJsonl(telemetryText);
  const gpuSamples = telemetry.map((event) => asRecordForSummary(event.gpu)).filter((sample) => Object.keys(sample).length > 0);
  const memoryUsed = gpuSamples.map((sample) => finiteSummaryNumber(sample.memory_used_mib)).filter((value): value is number => value !== null);
  const utilizations = gpuSamples.map((sample) => finiteSummaryNumber(sample.utilization_percent)).filter((value): value is number => value !== null);
  const otherMemory = telemetry.map((event) => finiteSummaryNumber(event.other_process_memory_mib)).filter((value): value is number => value !== null);
  const fallbackOuterFolds = finiteSummaryNumber(history.outer_folds) ?? 0;
  const completedOuterFolds = foldRecords.length || fallbackOuterFolds;
  const totalOuterFolds = formalSeeds.length ? formalSeeds.length * 5 : completedOuterFolds;
  const existingProgress = asRecordForSummary(runtimeRecord.training_progress);
  const existingTelemetry = asRecordForSummary(runtimeRecord.telemetry_summary);
  return {
    ...runtimeRecord,
    training_progress: Object.keys(existingProgress).length ? existingProgress : {
      status: runtimeRecord.training_status === "completed" ? "completed" : "pending",
      completed_formal_seeds: formalSeeds.length,
      total_formal_seeds: 3,
      completed_outer_folds: completedOuterFolds,
      total_outer_folds: totalOuterFolds,
      selected_epoch_min: selectedEpochs.length ? Math.min(...selectedEpochs) : null,
      selected_epoch_max: selectedEpochs.length ? Math.max(...selectedEpochs) : null,
      peak_model_memory_allocated_mib: allocatedMemory.length ? Math.max(...allocatedMemory) : null,
    },
    telemetry_summary: Object.keys(existingTelemetry).length ? existingTelemetry : {
      sample_count: telemetry.length,
      peak_gpu_memory_used_mib: memoryUsed.length ? Math.max(...memoryUsed) : null,
      max_gpu_utilization_percent: utilizations.length ? Math.max(...utilizations) : null,
      max_other_process_memory_mib: otherMemory.length ? Math.max(...otherMemory) : 0,
      hold_sample_count: telemetry.filter((event) => Array.isArray(event.hold_reasons) && event.hold_reasons.length > 0).length,
    },
  };
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

async function latestCompleteRuntimePath(taskId: string) {
  const root = resolveWorkspacePath(path.join("experiments", taskId));
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort()
    .reverse();

  for (const name of candidates) {
    const runRoot = path.join(root, name);
    const checks = await Promise.all(
      requiredRuntimeArtifacts.map((artifact) => fs.access(path.join(runRoot, artifact)).then(() => true).catch(() => false))
    );
    if (checks.every(Boolean)) return path.join("experiments", taskId, name);
  }
  return null;
}

async function loadMultiAgentRuntimeFromPointer(pointer: Record<string, unknown> | null): Promise<RuntimeSummary | null> {
  if (!pointer || !["evomind.current_run.v1", "evomind.run_ledger_entry.v1"].includes(String(pointer.schema))) return null;
  const runDir = typeof pointer.run_dir === "string" ? pointer.run_dir : "";
  const taskId = typeof pointer.task_id === "string" ? pointer.task_id : "";
  const runId = typeof pointer.run_id === "string" ? pointer.run_id : "";
  if (!/^[A-Za-z0-9_-]{1,160}$/.test(taskId) || !/^[A-Za-z0-9_-]{8,160}$/.test(runId)) return null;
  const expectedRunDir = path.join("workspace", "evomind_runs", runId);
  if (path.normalize(runDir) !== path.normalize(expectedRunDir)) return null;
  const absoluteRunDir = path.resolve(workspaceRoot, runDir);
  const relativeBoundary = path.relative(workspaceRoot, absoluteRunDir);
  if (!relativeBoundary || relativeBoundary.startsWith("..") || path.isAbsolute(relativeBoundary)) return null;
  const [
    run, taskGraph, eventsText, handoffsText, artifactManifest, review, rootMetrics, reportMarkdown,
    currentHpcProbe, llmHpcProbe, requestPayload, qloraConfig, datasetManifest, llmMetrics,
    llmEvaluation, llmEnvironment, llmTelemetryText, adapterReload, claimAudit, refinementPayload,
    datasetProfile, experimentComparison, hpcRuntime, historicalThresholds, deliverables,
    privateGrader, privateGraderLedger, candidateFreeze, trainingHistory, hpcTelemetryText,
  ] = await Promise.all([
    readJsonFile(`${absoluteRunDir}/run.json`),
    readJsonFile(`${absoluteRunDir}/task_graph.json`),
    readTextFile(`${absoluteRunDir}/events.jsonl`),
    readTextFile(`${absoluteRunDir}/handoffs.jsonl`),
    readJsonFile(`${absoluteRunDir}/artifact_manifest.json`),
    readJsonFile(`${absoluteRunDir}/review.json`),
    readJsonFile(`${absoluteRunDir}/metrics.json`),
    readTextFile(`${absoluteRunDir}/research_report.md`),
    readJsonFile(`${absoluteRunDir}/hpc_probe.json`),
    readJsonFile(`${absoluteRunDir}/hpc_llm_probe.json`),
    readJsonFile(`${absoluteRunDir}/request.json`),
    readJsonFile(`${absoluteRunDir}/qlora_config.json`),
    readJsonFile(`${absoluteRunDir}/data/dataset_manifest.json`),
    readJsonFile(`${absoluteRunDir}/llm_output/metrics.json`),
    readJsonFile(`${absoluteRunDir}/llm_output/evaluation.json`),
    readJsonFile(`${absoluteRunDir}/llm_output/environment.json`),
    readTextFile(`${absoluteRunDir}/llm_output/telemetry.jsonl`),
    readJsonFile(`${absoluteRunDir}/llm_output/adapter_reload.json`),
    readJsonFile(`${absoluteRunDir}/claim_audit.json`),
    readJsonFile(`${absoluteRunDir}/refinement.json`),
    readJsonFile(`${absoluteRunDir}/dataset_profile.json`),
    readJsonFile(`${absoluteRunDir}/experiment_comparison.json`),
    readJsonFile(`${absoluteRunDir}/hpc_runtime.json`),
    readJsonFile(`${absoluteRunDir}/historical_thresholds.json`),
    readJsonFile(`${absoluteRunDir}/deliverables.json`),
    readJsonFile(`${absoluteRunDir}/private_grader.json`),
    readJsonFile(`${absoluteRunDir}/private_grader_ledger.json`),
    readJsonFile(`${absoluteRunDir}/candidate_freeze.json`),
    readJsonFile(`${absoluteRunDir}/training_history.json`),
    readTextFile(`${absoluteRunDir}/hpc_telemetry.jsonl`)
  ]);
  if (!run) {
    const status = typeof pointer.status === "string" ? pointer.status : "unknown";
    return {
      task_id: taskId,
      latest_experiment_dir: runDir,
      latest_workstation_run_dir: typeof pointer.output_dir === "string" ? pointer.output_dir : null,
      task_state: { run_id: runId, status, seq: pointer.last_seq ?? 0 },
      runtime_snapshot: { schema: "evomind.current_runtime_snapshot.v1", task_id: taskId, run_id: runId, status },
      current_run: pointer,
      agent_trace: [],
      event_log: [],
    };
  }
  if ((run as Record<string, unknown>).run_id !== runId) return null;
  const canonicalStatus = typeof pointer.status === "string" ? pointer.status : (run as Record<string, unknown>).status;
  const canonicalRun = { ...(run as Record<string, unknown>), status: canonicalStatus };
  const events = readJsonl(eventsText).filter((event) => event.run_id === runId);
  const handoffs = readJsonl(handoffsText).filter((handoff) => handoff.run_id === runId);
  const tasks = ((run as Record<string, unknown>).tasks ?? {}) as Record<string, Record<string, unknown>>;
  const activeAgents = Object.values(tasks)
    .filter((task) => task.status === "running")
    .map((task) => ({ task_id: task.task_id, role: task.role, status: task.status }));
  const trainingLogs: string[] = [];
  for (const task of Object.values(tasks)) {
    if (task.role !== "TunerAgent" || typeof task.solution_id !== "string") continue;
    const logText = await readTextFile(`${absoluteRunDir}/solutions/${task.solution_id}/output/training.log`);
    if (logText) trainingLogs.push(...logText.split(/\r?\n/).filter(Boolean).slice(-12).map((line) => `[${task.solution_id}] ${line}`));
  }
  const llmTrainingLog = await readTextFile(`${absoluteRunDir}/llm_output/training.log`);
  if (llmTrainingLog) trainingLogs.push(...llmTrainingLog.split(/\r?\n/).filter(Boolean).slice(-20).map((line) => `[hpc_train] ${line}`));
  const requestRecord = requestPayload as Record<string, unknown> | null;
  const refinementRecord = refinementPayload as Record<string, unknown> | null;
  const parentRunId = typeof refinementRecord?.parent_run_id === "string"
    && /^[A-Za-z0-9_-]{8,160}$/.test(refinementRecord.parent_run_id)
    && refinementRecord.parent_run_id !== runId
    ? refinementRecord.parent_run_id
    : null;
  const preservedParentArtifactManifest = parentRunId
    ? await readJsonFile(path.join(workspaceRoot, "workspace", "evomind_runs", parentRunId, "artifact_manifest.json")) as Record<string, unknown> | null
    : null;
  const preservedParentArtifacts = Array.isArray(preservedParentArtifactManifest?.artifacts)
    ? preservedParentArtifactManifest.artifacts as Array<Record<string, unknown>>
    : [];
  const isLlmRun = requestRecord?.task_type === "llm_finetune";
  const metrics = (isLlmRun ? llmMetrics : rootMetrics) as Record<string, unknown> | null;
  const currentHpcRuntime = taskId === "siim-isic-melanoma-classification"
    ? siimHpcRuntimeWithEvidence(hpcRuntime, trainingHistory, hpcTelemetryText)
    : hpcRuntime;
  const llmTraining = metrics?.training as Record<string, unknown> | undefined;
  const llmFacts = isLlmRun ? {
    task_type: "llm_finetune",
    base_model: metrics?.base_model ?? (qloraConfig as Record<string, unknown> | null)?.base_model ?? null,
    dataset: datasetManifest,
    qlora_config: qloraConfig,
    training: llmTraining ? {
      step: llmTraining.steps ?? null,
      loss: llmTraining.train_loss ?? null,
      gpu_memory_mb: llmTraining.max_cuda_memory_mb ?? null,
    } : null,
    before_after_eval: metrics ? {
      before: metrics.before ?? null,
      after: metrics.after ?? null,
      improvement_pp: metrics.improvement_pp ?? null,
    } : null,
    evaluation: llmEvaluation,
    environment: llmEnvironment,
    telemetry: readJsonl(llmTelemetryText),
    adapter_reload: adapterReload,
    claim_audit: claimAudit,
  } : null;
  const snapshot = {
    schema: "evomind.current_runtime_snapshot.v1",
    task_id: taskId,
    run_id: runId,
    status: canonicalStatus,
    seq: (run as Record<string, unknown>).seq,
    active_agents: activeAgents,
    open_requirements: (run as Record<string, unknown>).open_requirements ?? [],
    next_action: (run as Record<string, unknown>).next_action ?? null,
    metrics,
    llm: llmFacts,
    reviewer: review,
    claim_audit: claimAudit,
    gates: (run as Record<string, unknown>).gates ?? {},
    request: requestRecord,
    dataset_profile: datasetProfile ?? (run as Record<string, unknown>).dataset_profile ?? null,
    experiment_comparison: experimentComparison ?? (run as Record<string, unknown>).experiment_comparison ?? null,
    hpc_runtime: currentHpcRuntime ?? (run as Record<string, unknown>).hpc_runtime ?? null,
    historical_thresholds: historicalThresholds ?? (run as Record<string, unknown>).historical_thresholds ?? null,
    deliverables: deliverables ?? (run as Record<string, unknown>).deliverables ?? null,
    private_grader: privateGrader,
    private_grader_ledger: privateGraderLedger,
    candidate_freeze: candidateFreeze,
    preserved_parent: parentRunId ? {
      run_id: parentRunId,
      version: refinementRecord?.parent_version ?? "parent",
      artifact_count: preservedParentArtifacts.length,
      review_status: preservedParentArtifactManifest?.review_status ?? null,
      preserved: true,
    } : null,
    pointer
  };
  const reviewRecord = review as Record<string, unknown> | null;
  const metricsRecord = metrics;
  const manifestRecord = artifactManifest as Record<string, unknown> | null;
  const acceptedCandidates = Array.isArray(reviewRecord?.accepted_candidates)
    ? reviewRecord.accepted_candidates as Array<Record<string, unknown>>
    : [];
  const reviewChecks = Array.isArray(reviewRecord?.checks)
    ? reviewRecord.checks as Array<Record<string, unknown>>
    : [];
  const manifestArtifacts = Array.isArray(manifestRecord?.artifacts)
    ? manifestRecord.artifacts as Array<Record<string, unknown>>
    : [];
  const selectedSolution = typeof metricsRecord?.selected_solution === "string" ? metricsRecord.selected_solution : null;
  const gpuCandidate = acceptedCandidates.find((candidate) => candidate.uses_cuda === true);
  const gpuSolutionId = typeof gpuCandidate?.solution_id === "string" ? gpuCandidate.solution_id : null;
  const gpuReview = reviewChecks.find((check) => check.solution_id === gpuSolutionId);
  const evidenceHashes = (refs: string[]) => manifestArtifacts
    .filter((artifact) => typeof artifact.path === "string" && refs.includes(artifact.path))
    .map((artifact) => artifact.sha256)
    .filter((value): value is string => typeof value === "string" && value.length > 0);
  const selectedRefs = isLlmRun ? ["llm_output/metrics.json", "review.json"] : ["metrics.json", "review.json"];
  const gpuRefs = gpuSolutionId
    ? [
        `solutions/${gpuSolutionId}/hpc_job.json`,
        `solutions/${gpuSolutionId}/output/environment.json`,
        `solutions/${gpuSolutionId}/output/metrics.json`,
        "review.json"
      ]
    : ["review.json"];
  const submissionRefs = ["artifact_manifest.json", "research_report.md"];
  const evidenceGraph = isLlmRun ? {
    schema: "evomind.claim_evidence_graph.v1",
    run_id: runId,
    claims: [
      {
        claim_id: "remote_gpu_domain_qlora_observed",
        statement: "A domain QLoRA run is supported only when the verified NVIDIA HPC environment, metrics, telemetry, and adapter reload artifacts agree.",
        status: reviewRecord?.status === "passed" && (claimAudit as Record<string, unknown> | null)?.status === "passed" ? "supported" : "blocked",
        evidence_refs: ["llm_output/environment.json", "llm_output/metrics.json", "llm_output/telemetry.jsonl", "llm_output/adapter_reload.json", "review.json", "claim_audit.json"],
        artifact_hashes: evidenceHashes(["llm_output/environment.json", "llm_output/metrics.json", "llm_output/telemetry.jsonl", "llm_output/adapter_reload.json", "review.json", "claim_audit.json"])
      },
      {
        claim_id: "model_publication_blocked",
        statement: "The reviewed domain adapter remains unpublished behind the Human Gate.",
        status: manifestRecord?.model_publication === "blocked" ? "supported" : "blocked",
        evidence_refs: ["artifact_manifest.json", "research_report.md"],
        artifact_hashes: evidenceHashes(["research_report.md"])
      }
    ]
  } : {
    schema: "evomind.claim_evidence_graph.v1",
    run_id: runId,
    claims: [
      {
        claim_id: "selected_candidate_reviewed",
        statement: selectedSolution ? `${selectedSolution} was selected from independently reviewed candidates.` : "No reviewed candidate selected.",
        status: reviewRecord?.status === "passed" && selectedSolution ? "supported" : "blocked",
        evidence_refs: selectedRefs,
        artifact_hashes: evidenceHashes(selectedRefs)
      },
      {
        claim_id: "hpc_gpu_training_observed",
        statement: gpuSolutionId ? `${gpuSolutionId} produced reviewer-verified CUDA telemetry on the HPC runtime.` : "No GPU candidate evidence found.",
        status: gpuCandidate && gpuReview?.passed === true && gpuReview.gpu_telemetry === true && gpuReview.hpc_job_binding === true
          ? "supported"
          : "blocked",
        evidence_refs: gpuRefs,
        artifact_hashes: evidenceHashes(gpuRefs)
      },
      {
        claim_id: "official_submission_blocked",
        statement: "The run generated a candidate submission but did not perform an official Kaggle submission.",
        status: manifestRecord?.official_submission === "blocked" ? "supported" : "blocked",
        evidence_refs: submissionRefs,
        artifact_hashes: evidenceHashes(submissionRefs)
      }
    ]
  };
  return {
    task_id: taskId,
    latest_experiment_dir: runDir,
    latest_workstation_run_dir: runDir,
    task_state: canonicalRun,
    agent_trace: events,
    event_log: events,
    artifact_manifest: artifactManifest as Record<string, unknown> | null,
    evidence_graph: evidenceGraph,
    experiment_graph: taskGraph as Record<string, unknown> | null,
    gate_engine: { gates: (run as Record<string, unknown>).gates ?? {}, review, claim_audit: claimAudit },
    runtime_snapshot: snapshot,
    report_markdown: reportMarkdown,
    training_log: trainingLogs,
    current_run: {
      ...pointer,
      status: canonicalStatus,
      last_seq: (run as Record<string, unknown>).seq,
      updated_at: (run as Record<string, unknown>).updated_at ?? pointer.updated_at,
    },
    task_graph: taskGraph as Record<string, unknown> | null,
    handoffs,
    review: review as Record<string, unknown> | null,
    hpc_probe: (llmHpcProbe ?? currentHpcProbe) as Record<string, unknown> | null,
    preserved_parent_run_id: parentRunId,
    preserved_parent_artifact_manifest: preservedParentArtifactManifest,
  };
}

export async function loadMultiAgentRuntimeByRunId(taskId: string, runId: string): Promise<RuntimeSummary | null> {
  if (!/^[A-Za-z0-9_-]{1,160}$/.test(taskId) || !/^[A-Za-z0-9_-]{8,160}$/.test(runId)) return null;
  return loadMultiAgentRuntimeFromPointer({
    schema: "evomind.current_run.v1",
    task_id: taskId,
    run_id: runId,
    run_dir: path.join("workspace", "evomind_runs", runId),
  });
}

async function loadCurrentMultiAgentRuntime(): Promise<RuntimeSummary | null> {
  const ledger = await readJsonFile(resolveWorkspacePath("workspace/run_ledger/current.json")) as Record<string, unknown> | null;
  const pointer = ledger?.schema === "evomind.run_ledger_entry.v1"
    ? ledger
    : await readJsonFile(resolveWorkspacePath("workspace/current_run.json")) as Record<string, unknown> | null;
  return loadMultiAgentRuntimeFromPointer(pointer);
}

async function loadRuntimeSummary(taskId = "house_prices"): Promise<RuntimeSummary> {
  const latestRaw = await latestExperimentPath(taskId);
  const latestWorkstationRun = await latestWorkstationRunPath(taskId);
  const latestScoreGatedRun = taskId === "playground_series_s6e6" ? await latestScoreGatedWorkstationRunPath(taskId) : null;
  const latestCompleteRuntime = await latestCompleteRuntimePath(taskId);
  const latest = latestScoreGatedRun ?? latestCompleteRuntime ?? latestRaw;
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
  const agentTrace = readJsonl(agentTraceText);
  let eventLog = readJsonl(eventLogText);
  if (eventLog.length < 5 && agentTrace.length >= 5) {
    eventLog = runtimeEventsFromAgentTrace(agentTrace);
  }

  return {
    task_id: taskId,
    latest_experiment_dir: latest,
    latest_workstation_run_dir: latestWorkstationRun,
    latest_score_gated_run_dir: latestScoreGatedRun,
    task_state: taskState,
    agent_trace: agentTrace,
    event_log: eventLog,
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

async function buildFullWorkstationSummary() {
  await ensureWorkstationSeeded();
  const [tasks, runs, requiredPassedRuns, connectors, actions, gates, evidence, reports, workflows, runtimes, terminalAgent, scientistAutopilot, scientistActionQueue, scientistContinuationStatus, scientistLoop, scientistLoopLessons, scientistMemoryConsolidation, scientistSelfAudit, scientistReadinessReport, scientistCausalDiagnosis, scientistStrategyOptimizer, scientistContextPacket, scientistReasoningSynthesis, scientistTerminalTurn, scientistEngineeringLoop, scientistInnovationBacklog, scientistHypothesisReview, scientistExperimentBlueprint, scientistSituationModel, scientistTurnPlan, scientistWorkplan, scientistRepairPlan, scientistExecutionContract, scientistTurns, scientistStepTrace, scientistAutopilotStatus, finalDeliveryStatus, kaggleNewCompetitionReadiness, kaggleDpapiReadiness, kaggleExperimentInventory, top30NextEvolutionOrders, mlevolveAlignmentMatrix, mlebenchStyleLeaderboard, verifiedLaunchAudit, launchReadiness, learningLoopReadiness, hpcProbe, liveGpu, s6e6DependencyGate, literatureContext, literatureByTask] = await Promise.all([
    prisma.task.findMany({ orderBy: { updatedAt: "desc" } }),
    prisma.experimentRun.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.experimentRun.findMany({
      where: { taskId: { in: runtimeTaskIds }, validationStatus: "passed" },
      orderBy: { createdAt: "desc" }
    }),
    prisma.connectorStatus.findMany({ orderBy: { provider: "asc" } }),
    prisma.actionLog.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.gate.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.evidence.findMany({ orderBy: { createdAt: "desc" }, take: 50 }),
    prisma.report.findMany({ orderBy: { updatedAt: "desc" }, take: 20 }),
    prisma.workflow.findMany({ orderBy: { updatedAt: "desc" }, take: 20 }),
    Promise.all(runtimeTaskIds.map((taskId) => loadRuntimeSummary(taskId))),
    loadXsciTerminalAgentSummary(),
    loadScientistAutopilotSummary(),
    loadScientistActionQueueSummary(),
    loadScientistContinuationStatusSummary(),
    loadScientistLoopSummary(),
    loadScientistLoopLessonsSummary(),
    loadScientistMemoryConsolidationSummary(),
    loadScientistSelfAuditSummary(),
    loadScientistReadinessReportSummary(),
    loadScientistCausalDiagnosisSummary(),
    loadScientistStrategyOptimizerSummary(),
    loadScientistContextPacketSummary(),
    loadScientistReasoningSynthesisSummary(),
    loadScientistTerminalTurnSummary(),
    loadScientistEngineeringLoopSummary(),
    loadScientistInnovationBacklogSummary(),
    loadScientistHypothesisReviewSummary(),
    loadScientistExperimentBlueprintSummary(),
    loadScientistSituationModelSummary(),
    loadScientistTurnPlanSummary(),
    loadScientistWorkplanSummary(),
    loadScientistRepairPlanSummary(),
    loadScientistExecutionContractSummary(),
    loadScientistTurnsSummary(),
    loadScientistStepTraceSummary(),
    loadScientistAutopilotStatusSummary(),
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
    latestS6E6BoostingDependencyStatus(),
    loadLatestLiteratureSummary(),
    loadLiteratureStateForAllTasks()
  ]);
  const strictResourceReadiness = await readJsonFile(
    resolveWorkspacePath("docs/launch_resource_readiness.json")
  ) as Record<string, any> | null;
  const hpcJobLineage = await listHpcJobLineage();
  const currentMultiAgentRuntime = await loadCurrentMultiAgentRuntime();
  const runtimeByTask = Object.fromEntries(runtimes.map((item) => [item.task_id, item]));
  if (currentMultiAgentRuntime) runtimeByTask[currentMultiAgentRuntime.task_id] = currentMultiAgentRuntime;
  const runtime = currentMultiAgentRuntime ?? runtimes.find((item) => item.latest_experiment_dir) ?? runtimes[0];
  const runtimeCandidates = currentMultiAgentRuntime
    ? [currentMultiAgentRuntime, ...runtimes.filter((item) => item.task_id !== currentMultiAgentRuntime.task_id)]
    : runtimes;
  const runtimeMetricsFor = (item: RuntimeSummary): Record<string, unknown> | undefined => {
    const latestMetric = item.runtime_snapshot?.latest_metric;
    if (latestMetric && typeof latestMetric === "object") return latestMetric as Record<string, unknown>;
    const aggregate = item.runtime_snapshot?.metrics;
    if (!aggregate || typeof aggregate !== "object") return undefined;
    const metricRecord = aggregate as Record<string, unknown>;
    const metric = typeof metricRecord.metric === "string" ? metricRecord.metric.toLowerCase() : "";
    const score = metricRecord.cv_score;
    if (typeof score !== "number") return undefined;
    return {
      metric,
      best_score: score,
      ...(metric === "accuracy" ? { cv_accuracy_mean: score } : {}),
      ...(metric === "rmsle" ? { cv_rmsle_mean: score } : {})
    };
  };
  const runtimeTasks = runtimeCandidates
    .filter((item) => item.latest_experiment_dir)
    .map((item) => ({
      id: item.task_id,
      name: item.task_id.replaceAll("_", " "),
      task_type: "tabular_runtime",
      target: null,
      metric: String(runtimeMetricsFor(item)?.metric ?? Object.keys(runtimeMetricsFor(item) ?? {})[0] ?? ""),
      status: String(item.task_state?.status ?? item.task_state?.state ?? "runtime_ready"),
      priority: "Runtime",
      owner: "Research Agent Runtime",
      config_path: `configs/${item.task_id}.yaml`,
      task_dir: `tasks/${item.task_id}`,
      created_at: null,
      updated_at: null
    }));
  const runtimeTaskIdsSet = new Set(runtimeTasks.map((task) => task.id));

  const runtimeRun = runtimeCandidates
    .filter((item) => item.latest_experiment_dir)
    .map((item) => {
      const runtimeMetrics = runtimeMetricsFor(item);
      const runtimeRunId = typeof item.latest_experiment_dir === "string" ? item.latest_experiment_dir.split(/[\\/]/).pop() : undefined;
      return {
          id: runtimeRunId,
          task_id: item.task_id,
          output_dir: item.latest_experiment_dir,
          status: String(item.task_state?.status ?? item.task_state?.state ?? "passed"),
          best_model: ((item.experiment_graph as any)?.nodes?.[0]?.model as string | undefined) ?? "runtime_baseline",
          best_metrics: runtimeMetrics ?? null,
          accepted: true,
          validation_gate: { status: "passed" },
          started_at: null,
          finished_at: null
        };
    });
  const runtimeManifest = currentMultiAgentRuntime?.artifact_manifest as Record<string, unknown> | null | undefined;
  const currentRuntimeArtifacts = Array.isArray(runtimeManifest?.artifacts)
    ? runtimeManifest.artifacts as Array<Record<string, unknown>>
    : [];
  const preservedParentManifest = currentMultiAgentRuntime?.preserved_parent_artifact_manifest as Record<string, unknown> | null | undefined;
  const preservedParentArtifacts = Array.isArray(preservedParentManifest?.artifacts)
    ? preservedParentManifest.artifacts as Array<Record<string, unknown>>
    : [];
  const runtimeEvidenceSource = currentRuntimeArtifacts.length > 0
    ? {
        artifacts: currentRuntimeArtifacts,
        manifest: runtimeManifest,
        runId: String(currentMultiAgentRuntime?.current_run?.run_id ?? "current"),
        source: "multi_agent_artifact_manifest",
        claimBinding: "current_reviewed_run",
      }
    : {
        artifacts: preservedParentArtifacts,
        manifest: preservedParentManifest,
        runId: currentMultiAgentRuntime?.preserved_parent_run_id ?? "preserved_parent",
        source: "preserved_parent_artifact_manifest",
        claimBinding: "preserved_reviewed_parent",
      };
  const runtimeEvidence = currentMultiAgentRuntime
    ? runtimeEvidenceSource.artifacts.map((artifact, index) => ({
        id: `runtime:${runtimeEvidenceSource.runId}:${index + 1}`,
        task_id: currentMultiAgentRuntime.task_id,
        run_id: runtimeEvidenceSource.runId,
        label: String(artifact.kind ?? artifact.path ?? `Artifact ${index + 1}`),
        name: String(artifact.kind ?? artifact.path ?? `Artifact ${index + 1}`),
        artifact_type: String(artifact.kind ?? "verified_artifact"),
        verification_status: "verified",
        artifact_path: typeof artifact.path === "string" ? artifact.path : null,
        hash: typeof artifact.sha256 === "string" ? artifact.sha256 : null,
        bytes: typeof artifact.bytes === "number" ? artifact.bytes : null,
        source: runtimeEvidenceSource.source,
        claim_binding: runtimeEvidenceSource.claimBinding,
        created_at: typeof runtimeEvidenceSource.manifest?.generated_at === "string"
          ? runtimeEvidenceSource.manifest.generated_at
          : currentMultiAgentRuntime.current_run?.updated_at ?? null,
      }))
    : [];
  const recentRunIds = new Set(runs.map((run: any) => run.id));
  const mergedRuns = [...runs, ...requiredPassedRuns.filter((run: any) => !recentRunIds.has(run.id))];
  const canonicalCurrentRunId = currentMultiAgentRuntime?.current_run?.run_id;
  const canonicalCurrentStatus = currentMultiAgentRuntime?.current_run?.status;
  const dbRuns = mergedRuns.map((run: any) => {
    const metrics = decodeJson<Record<string, any>>(run.metricsJson);
    const status = run.id === canonicalCurrentRunId && canonicalCurrentStatus ? canonicalCurrentStatus : run.status;
    return {
      id: run.id,
      task_id: run.taskId,
      output_dir: run.outputDir,
      status,
      workstation_run: metrics?.workstation_run === true || run.id.startsWith("wr_"),
      direct_training_allowed: metrics?.direct_training_allowed === true,
      official_submission_allowed: metrics?.official_submission_allowed === true,
      workstation_run_manifest: run.outputDir ? `${run.outputDir}/workstation_run_manifest.json` : null,
      artifact_manifest: run.outputDir ? `${run.outputDir}/artifact_manifest.json` : null,
      best_model: run.bestModel,
      best_metrics: metrics,
      accepted: run.validationStatus === "passed" || status === "passed" || status === "COMPLETED",
      validation_gate: run.validationStatus ? { status: run.validationStatus } : undefined,
      started_at: run.startedAt?.toISOString() ?? null,
      finished_at: run.finishedAt?.toISOString() ?? null
    };
  });
  const runKeys = new Set(runtimeRun.map((run) => run.output_dir));
  const gpuCredentialPresent = gpuSshStatus() === "configured";
  const deepSeekConfigured = deepSeekApiKeyStatus() === "configured";
  const openAiConfigured = openAiApiKeyStatus() === "configured";
  const workstationPythonConfigured = Boolean(process.env.WORKSTATION_PYTHON?.trim());
  const codeAgentConfigured = claudeApiKeyStatus() === "configured" || deepSeekConfigured;
  const deepSeek = deepSeekConfig();
  const openAi = openAiConfig();
  const gpu = gpuSshConfig();
  const gpuRoute = gpu.socksProxy.host
    ? `${gpu.socksProxy.host}:${gpu.socksProxy.port} SOCKS5 bridge`
    : "direct SSH route";
  const currentRunHpcProbe = currentMultiAgentRuntime?.hpc_probe;
  const currentRunGpuInventory = Array.isArray(currentRunHpcProbe?.gpu_inventory)
    ? currentRunHpcProbe.gpu_inventory as Array<Record<string, unknown>>
    : [];
  const currentRunTorch = currentRunHpcProbe?.torch && typeof currentRunHpcProbe.torch === "object"
    ? currentRunHpcProbe.torch as Record<string, unknown>
    : null;
  const currentRunReviewChecks = Array.isArray(currentMultiAgentRuntime?.review?.checks)
    ? currentMultiAgentRuntime.review.checks as Array<Record<string, unknown>>
    : [];
  const currentRunGpuReviewPassed = currentRunReviewChecks.some((check) =>
    check.passed === true && check.cuda_contract === true && check.gpu_telemetry === true && check.hpc_job_binding === true
  );
  const currentRunHpcReady = currentMultiAgentRuntime?.task_state?.status === "completed"
    && currentRunHpcProbe?.status === "passed"
    && currentRunTorch?.cuda_available === true
    && currentRunGpuInventory.length > 0
    && currentRunGpuReviewPassed;
  const currentRunGpuSummary = currentRunGpuInventory
    .map((entry) => typeof entry.name === "string" ? entry.name : "")
    .filter(Boolean)
    .join(", ") || "current-run GPU";
  const gpuPendingState = "GPU Environment Created / Web Terminal Ready / External SSH Pending";
  const liveGpuSummary = liveGpu.gpuSummary || "nvidia-smi evidence present";
  const gpuVerifiedState = `GPU Verified: ${liveGpuSummary} via SSH Gateway`;
  const gpuSshReadyState = `GPU SSH Gateway Ready: ${liveGpuSummary} / nvidia-smi smoke passed`;
  const gpuLegacyVerifiedState = "GPU Verified: 4 x NVIDIA A800-SXM4-80GB via Login Node / Web Terminal";
  const gpuLegacySshReadyState = "GPU SSH Gateway Ready: 4 x NVIDIA A800-SXM4-80GB / historical CUDA smoke passed";
  const s6e6GatewayBlocked = s6e6DependencyGate.status === "blocked_resource_gateway";
  const gpuFreshSmokeBlocked = !currentRunHpcReady && liveGpu.present === true && liveGpu.passed === false;
  const liveGpuPassed = liveGpu.present === true && liveGpu.passed === true;
  const latestGpuAllocationBlocker = actions.find((action: any) => action.action === "gpu_current_allocation_blocker");
  const latestGpuAllocationBlockerMetadata = latestGpuAllocationBlocker
    ? decodeJson<Record<string, unknown>>(latestGpuAllocationBlocker.metadataJson)
    : null;
  const gpuCurrentAllocationBlocked = !currentRunHpcReady && !liveGpuPassed && latestGpuAllocationBlockerMetadata?.status === "blocked_current_allocation";
  const kaggleDpapi = await kaggleDpapiProbeStatus(kaggleDpapiReadiness as Record<string, unknown> | null);
  const strictHpcRuntime = strictResourceReadiness?.external_resources?.hpc_gpu_strict_runtime as Record<string, any> | undefined;
  const strictHpcProfile = strictHpcRuntime?.profile as Record<string, any> | undefined;
  const strictHpcProbe = strictHpcRuntime?.live_probe as Record<string, any> | undefined;
  const strictHpcSmoke = strictHpcRuntime?.bounded_smoke as Record<string, any> | undefined;
  const latestHpcProfileIdentity = hpcJobLineage.find((identity) => identity.identity_kind === "allocation_profile");
  const currentHpcProfileName = typeof latestHpcProfileIdentity?.profile === "string" ? latestHpcProfileIdentity.profile : null;
  const strictHpcProfileName = typeof strictHpcProfile?.profile === "string" ? strictHpcProfile.profile : null;
  const hpcIdentityBlocked = Boolean(currentHpcProfileName && currentHpcProfileName !== strictHpcProfileName);
  const strictHpcReady = strictResourceReadiness?.strict_hpc_runtime_status === "ready"
    && strictHpcRuntime?.configured === true
    && strictHpcRuntime?.state === "strict_hpc_runtime_verified"
    && strictHpcProbe?.job_container_verified === true
    && strictHpcProbe?.samples_requested === 5
    && strictHpcProbe?.samples_passed === 5
    && strictHpcSmoke?.status === "passed"
    && !hpcIdentityBlocked;
  const strictGpuName = String(strictHpcProbe?.samples?.[0]?.gpu_name ?? "verified GPU");
  const strictProfileName = String(strictHpcProfile?.profile ?? "named profile");

  return {
    hpc_job_lineage: hpcJobLineage,
    tasks: [
      ...runtimeTasks,
      ...tasks.filter((task: any) => !runtimeTaskIdsSet.has(task.id.replaceAll("-", "_"))).map((task: any) => ({
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
    connector_status: normalizeConnectorRegistry(Object.fromEntries(
      [
        ...connectors.filter((connector: any) => !["code_agent", "gpu", "kaggle"].includes(connector.provider)).map((connector: any) => [
          connector.provider,
          {
            name: connector.name,
            state: connector.state,
            configured: connector.configured,
            notes: connector.detail
          }
        ] as const),
        [
          "llm",
          {
            name: "Local reasoning fallback",
            state: "rule_based",
            configured: true,
            notes: "Deterministic local reasoning remains available when an external LLM is not configured."
          }
        ] as const,
        [
          "storage",
          {
            name: "Workspace storage",
            state: "local_workspace",
            configured: true,
            notes: "Artifacts are stored inside the configured local workstation root."
          }
        ] as const,
        [
          "python_runner",
          {
            name: "Workstation Python",
            state: workstationPythonConfigured ? "local" : "not_configured",
            configured: workstationPythonConfigured,
            notes: workstationPythonConfigured ? "WORKSTATION_PYTHON is configured." : "WORKSTATION_PYTHON is not configured."
          }
        ] as const,
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
          "openai",
          {
            name: "OpenAI-compatible Agent LLM",
            state: openAiConfigured ? `OpenAI Agent Ready (${openAi.model})` : "Not Configured",
            configured: openAiConfigured,
            notes: openAiConfigured ? "OPENAI_API_KEY detected; assistant streaming and tool calling use the configured OpenAI-compatible gateway." : "Set OPENAI_API_KEY to enable the OpenAI-compatible assistant runtime.",
            model: openAi.model,
            base_url: openAi.baseUrl,
            streaming: openAiConfigured,
            tool_calling: openAiConfigured
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
            state: hpcIdentityBlocked
              ? `GPU Degraded: current profile ${currentHpcProfileName} has no bound execution job identity`
              : strictHpcReady
              ? `GPU SSH Gateway Ready: ${strictProfileName} / ${strictGpuName} / 5 of 5 identity samples and bounded smoke passed`
              : currentRunHpcReady
              ? `GPU SSH Gateway Ready: ${currentRunGpuSummary} / current-run nvidia-smi and CUDA training passed`
              : gpuCurrentAllocationBlocked
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
            configured: strictHpcReady || gpuCredentialPresent || Boolean(currentHpcProfileName),
            current_allocation_blocked: hpcIdentityBlocked ? false : strictHpcReady || currentRunHpcReady ? false : gpuCurrentAllocationBlocked || gpuFreshSmokeBlocked || s6e6GatewayBlocked,
            current_gate_ready: hpcIdentityBlocked ? false : strictHpcReady || (gpuCredentialPresent && (currentRunHpcReady || (!gpuCurrentAllocationBlocked && !gpuFreshSmokeBlocked && !s6e6GatewayBlocked && liveGpu.passed === true))),
            notes: hpcIdentityBlocked
              ? `The newest profile identity ${currentHpcProfileName} is profile-only and execution_eligible=false. Historical ${strictHpcProfileName ?? "HPC"} evidence remains archived but cannot make the current connector READY.`
              : strictHpcReady
              ? `Strict HPC runtime evidence verifies ${strictProfileName}, ${strictGpuName}, 5 of 5 job-container identity samples, and a bounded GPU smoke. Remote work remains restricted to the profile-bound allowed root.`
              : currentRunHpcReady
              ? `The current Multi-Agent run completed on ${currentRunGpuSummary}; hpc_probe.json, per-fold CUDA telemetry, downloaded artifact hashes, and Independent Reviewer checks all passed. Historical connection records remain audit-only.`
              : gpuCurrentAllocationBlocked
              ? `A newer rotating GPU allocation failed fresh SSH validation. Host=${String(latestGpuAllocationBlockerMetadata?.host ?? "unknown")}, port=${String(latestGpuAllocationBlockerMetadata?.port ?? "unknown")}, direct TCP=${String(latestGpuAllocationBlockerMetadata?.tcp_direct ?? "unknown")}, SSH=${String(latestGpuAllocationBlockerMetadata?.ssh_direct ?? "unknown")}. Historical A800 evidence remains archived, but workstation training is blocked until a fresh allocation passes SSH/CUDA smoke.`
              : gpuFreshSmokeBlocked
              ? `DPAPI SSH credentials are loaded, but the latest GPU SSH/CUDA smoke failed. Latest evidence: ${liveGpu.path ?? "missing"}. Historical A800 evidence remains archived; workstation training is blocked until /api/gpu/connections/test passes on the current allocation.`
              : s6e6GatewayBlocked
              ? `Historical NVIDIA A800 GPU evidence exists, but the current S6E6 dependency gate is blocked before training. Blocker: ${s6e6DependencyGate.blocker ?? "resource gateway unavailable"}. Latest gate: ${s6e6DependencyGate.path}. Next action: ${s6e6DependencyGate.nextAction ?? "refresh the rotating GPU allocation and rerun the dependency gate"}.`
              : gpuCredentialPresent && liveGpu.passed
              ? `Windows DPAPI credentials are present and the project SSH helper reached the current GPU allocation through the configured ${gpuRoute}. Latest evidence: ${liveGpu.path}. Python runtime: ${liveGpu.pythonRuntime ?? "unknown"}; torch import: ${liveGpu.torchImport === null ? "unknown" : String(liveGpu.torchImport)}. GPU jobs remain whitelist-template only; arbitrary shell is not exposed.`
              : liveGpu.passed
                ? `Latest SSH gateway evidence proves ${liveGpuSummary}, but no loaded SSH credential is present for automated jobs.`
                : gpuCredentialPresent && hpcProbe.fullyReadyAllowed
                  ? "Windows DPAPI credentials are present and historical Web Terminal evidence proves 4 x A800. Run a fresh GPU SSH smoke to refresh the current allocation summary."
                  : hpcProbe.fullyReadyAllowed
                    ? "HPC login node + Web Terminal evidence proves 4 x A800, but the workstation does not currently have a loaded SSH credential for automated jobs."
                : "HPC login node is reachable through the PDF ncat path, but the platform GPU environment SSH endpoint is still external-pending; confirm nvidia-smi in Web Terminal before marking fully ready.",
            proxy: strictHpcReady || gpu.socksProxy.host ? "socks5" : "direct",
            evidence: {
              strict_hpc_runtime: strictHpcReady ? {
                report_path: "docs/launch_resource_readiness.json",
                profile: strictProfileName,
                job_id: strictHpcProfile?.job_id ?? null,
                gpu_name: strictGpuName,
                samples_passed: strictHpcProbe?.samples_passed ?? null,
                samples_requested: strictHpcProbe?.samples_requested ?? null,
                bounded_smoke_status: strictHpcSmoke?.status ?? null,
                signals_sent: strictHpcProbe?.signals_sent ?? null,
                other_processes_modified: strictHpcProbe?.other_processes_modified ?? null
              } : null,
              hpc_probe: hpcProbe,
              current_multi_agent_hpc_probe: currentRunHpcProbe ?? null,
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
            LLM_PROVIDER: process.env.EVOLUTION_PRIMARY_PROVIDER || (openAiConfigured ? "openai" : "rule_based"),
            OPENAI_API_KEY_STATUS: openAiApiKeyStatus(),
            OPENAI_MODEL: openAi.model,
            OPENAI_BASE_URL: openAi.baseUrl,
            DEEPSEEK_API_KEY_STATUS: deepSeekApiKeyStatus(),
            DEEPSEEK_MODEL: deepSeek.model,
            DATABASE_PROVIDER: "sqlite"
          }
        ] as const
      ]
    )),
    runs: [...runtimeRun, ...dbRuns.filter((run: any) => !run.output_dir || !runKeys.has(run.output_dir))],
    actions: actions.map((action: any) => ({
      id: action.id,
      action: action.action,
      task_id: action.taskId,
      run_id: action.runId,
      message: action.message,
      artifact: action.artifactPath,
      metadata: decodeJson(action.metadataJson),
      at: action.createdAt.toISOString()
    })),
    gates: gates.map((gate: any) => ({
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
    evidence: [...runtimeEvidence, ...evidence.map((item: any) => ({
      id: item.id,
      task_id: item.taskId,
      run_id: item.runId,
      label: item.label,
      artifact_path: item.artifactPath,
      hash: item.hash,
      source: item.source,
      claim_binding: item.claimBinding,
      created_at: item.createdAt.toISOString()
    }))],
    reports: reports.map((report: any) => ({
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
    workflows: workflows.map((workflow: any) => ({
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
      ...(verifiedLaunchAudit as Record<string, unknown> | null),
      latest_readiness: launchReadiness,
      launch_state: (launchReadiness as Record<string, unknown> | null)?.launch_state ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.launch_state ?? null,
      blockers: (launchReadiness as Record<string, unknown> | null)?.blockers ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.blockers ?? [],
      critical_failures: (launchReadiness as Record<string, unknown> | null)?.critical_failures ?? [],
      soft_failures: (launchReadiness as Record<string, unknown> | null)?.soft_failures ?? []
    },
    learning_loop_readiness: learningLoopReadiness,
    runtime,
    runtime_by_task: runtimeByTask,
    terminal_agent: currentMultiAgentRuntime
      ? {
          ...terminalAgent,
          task_id: currentMultiAgentRuntime.task_id,
          latest_run_id: currentMultiAgentRuntime.current_run?.run_id ?? null,
          latest_run_dir: currentMultiAgentRuntime.latest_experiment_dir,
          status: currentMultiAgentRuntime.task_state?.status ?? "unknown",
          last_seq: currentMultiAgentRuntime.task_state?.seq ?? 0,
          events: currentMultiAgentRuntime.event_log ?? [],
          source: "workspace/run_ledger/current.json"
        }
      : terminalAgent,
    literature_context: literatureContext,
    literature_by_task: literatureByTask,
    scientist_autopilot: scientistAutopilot,
    scientist_action_queue: scientistActionQueue,
    scientist_continuation_status: scientistContinuationStatus,
    scientist_loop: scientistLoop,
    scientist_loop_lessons: scientistLoopLessons,
    scientist_memory_consolidation: scientistMemoryConsolidation,
    scientist_self_audit: scientistSelfAudit,
    scientist_readiness_report: scientistReadinessReport,
    scientist_causal_diagnosis: scientistCausalDiagnosis,
    scientist_strategy_optimizer: scientistStrategyOptimizer,
    scientist_context_packet: scientistContextPacket,
    scientist_reasoning_synthesis: scientistReasoningSynthesis,
    scientist_terminal_turn: scientistTerminalTurn,
    scientist_engineering_loop: scientistEngineeringLoop,
    scientist_innovation_backlog: scientistInnovationBacklog,
    scientist_hypothesis_review: scientistHypothesisReview,
    scientist_experiment_blueprint: scientistExperimentBlueprint,
    scientist_situation_model: scientistSituationModel,
    scientist_turn_plan: scientistTurnPlan,
    scientist_workplan: scientistWorkplan,
    scientist_repair_plan: scientistRepairPlan,
    scientist_execution_contract: scientistExecutionContract,
    scientist_turns: scientistTurns,
    scientist_step_trace: scientistStepTrace,
    scientist_autopilot_status: scientistAutopilotStatus,
    workspace_root: workspaceRoot
  };
}

type SummaryRecord = Record<string, any>;

const detailFields: Record<string, string[]> = {
  assistant: ["runtime", "runtime_by_task", "terminal_agent", "scientist_context_packet", "scientist_terminal_turn"],
  overview: ["runtime", "runtime_by_task", "terminal_agent", "scientist_autopilot", "scientist_autopilot_status", "scientist_next_action", "scientist_context_packet", "scientist_terminal_turn", "verified_launch_audit", "learning_loop_readiness"],
  tasks: ["runtime", "runtime_by_task", "terminal_agent", "scientist_context_packet", "scientist_terminal_turn"],
  data: ["kaggle_dpapi_readiness", "kaggle_experiment_inventory", "kaggle_new_competition_readiness", "learning_loop_readiness"],
  gpu: ["runtime", "runtime_by_task", "kaggle_dpapi_readiness", "verified_launch_audit"],
  evidence: ["runtime", "runtime_by_task", "terminal_agent", "evidence", "gates", "runs"],
  literature: ["literature_context", "literature_by_task", "evidence", "runs"],
  workflow: ["runtime", "runtime_by_task", "terminal_agent", "scientist_context_packet", "scientist_terminal_turn"],
  code: ["runtime", "runtime_by_task", "scientist_engineering_loop", "scientist_context_packet", "scientist_terminal_turn"],
  runtime: ["runtime", "runtime_by_task", "terminal_agent", "scientist_context_packet", "scientist_terminal_turn"],
  experiments: ["runtime", "runtime_by_task", "kaggle_experiment_inventory", "learning_loop_readiness"],
  evolution: [
    "runtime", "runtime_by_task", "terminal_agent", "scientist_autopilot", "scientist_action_queue",
    "scientist_continuation_status", "scientist_loop", "scientist_loop_lessons", "scientist_memory_consolidation",
    "scientist_self_audit", "scientist_readiness_report", "scientist_causal_diagnosis", "scientist_strategy_optimizer",
    "scientist_context_packet", "scientist_reasoning_synthesis", "scientist_terminal_turn", "scientist_engineering_loop",
    "scientist_innovation_backlog", "scientist_hypothesis_review", "scientist_experiment_blueprint",
    "scientist_situation_model", "scientist_turn_plan", "scientist_workplan", "scientist_repair_plan",
    "scientist_execution_contract", "scientist_turns", "scientist_step_trace", "scientist_autopilot_status"
  ],
  report: ["runtime", "runtime_by_task", "terminal_agent", "literature_context", "literature_by_task", "evidence", "gates", "runs", "reports"],
  gates: ["runtime", "runtime_by_task", "terminal_agent", "evidence", "gates", "runs"],
  settings: ["kaggle_dpapi_readiness", "verified_launch_audit"],
  control: [
    "runtime", "runtime_by_task", "terminal_agent", "final_delivery_status", "kaggle_new_competition_readiness",
    "kaggle_dpapi_readiness", "kaggle_experiment_inventory", "top30_next_evolution_orders", "mlevolve_alignment_matrix",
    "mlebench_style_leaderboard", "verified_launch_audit", "learning_loop_readiness", "literature_context",
    "literature_by_task", "scientist_autopilot", "scientist_action_queue", "scientist_continuation_status",
    "scientist_loop", "scientist_loop_lessons", "scientist_memory_consolidation", "scientist_self_audit",
    "scientist_readiness_report", "scientist_causal_diagnosis", "scientist_strategy_optimizer", "scientist_context_packet",
    "scientist_reasoning_synthesis", "scientist_terminal_turn", "scientist_engineering_loop", "scientist_innovation_backlog",
    "scientist_hypothesis_review", "scientist_experiment_blueprint", "scientist_situation_model", "scientist_turn_plan",
    "scientist_workplan", "scientist_repair_plan", "scientist_execution_contract", "scientist_turns",
    "scientist_step_trace", "scientist_autopilot_status"
  ]
};

function serializeTaskLite(task: any) {
  return {
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
  };
}

async function buildLightweightSummary(): Promise<SummaryRecord> {
  await ensureWorkstationSeeded();
  const [tasks, runs, connectors, actions, gates, evidence, reports, workflows, finalDeliveryStatus, verifiedLaunchAudit, launchReadiness, learningLoopReadiness, kaggleNewCompetitionReadiness, kaggleDpapiReadiness] = await Promise.all([
    prisma.task.findMany({ orderBy: { updatedAt: "desc" } }),
    prisma.experimentRun.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.connectorStatus.findMany({ orderBy: { provider: "asc" } }),
    prisma.actionLog.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.gate.findMany({ orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.evidence.findMany({ orderBy: { createdAt: "desc" }, take: 30 }),
    prisma.report.findMany({ orderBy: { updatedAt: "desc" }, take: 10 }),
    prisma.workflow.findMany({ orderBy: { updatedAt: "desc" }, take: 20 }),
    readJsonFile(resolveWorkspacePath("docs/final_delivery_status_20260612.json")),
    readJsonFile(resolveWorkspacePath("docs/verified_workstation_launch_audit.json")),
    readJsonFile(resolveWorkspacePath("workspace/workstation_launch_readiness_20260630.json")),
    readJsonFile(resolveWorkspacePath("workspace/workstation_learning_loop_readiness_20260630.json")),
    readJsonFile(resolveWorkspacePath("docs/kaggle_new_competition_readiness.json")),
    readJsonFile(resolveWorkspacePath("docs/kaggle_dpapi_readiness.json"))
  ]);

  const strictResourceReadiness = await readJsonFile(
    resolveWorkspacePath("docs/launch_resource_readiness.json")
  ) as Record<string, any> | null;
  const hpcJobLineage = await listHpcJobLineage();
  const currentRunLedger = await readCurrentRunLedger();
  const strictHpcRuntime = strictResourceReadiness?.external_resources?.hpc_gpu_strict_runtime as Record<string, any> | undefined;
  const strictHpcProfile = strictHpcRuntime?.profile as Record<string, any> | undefined;
  const strictHpcProbe = strictHpcRuntime?.live_probe as Record<string, any> | undefined;
  const strictHpcSmoke = strictHpcRuntime?.bounded_smoke as Record<string, any> | undefined;
  const latestHpcProfileIdentity = hpcJobLineage.find((identity) => identity.identity_kind === "allocation_profile");
  const currentHpcProfileName = typeof latestHpcProfileIdentity?.profile === "string" ? latestHpcProfileIdentity.profile : null;
  const strictHpcProfileName = typeof strictHpcProfile?.profile === "string" ? strictHpcProfile.profile : null;
  const hpcIdentityBlocked = Boolean(currentHpcProfileName && currentHpcProfileName !== strictHpcProfileName);
  const strictHpcReady = strictResourceReadiness?.strict_hpc_runtime_status === "ready"
    && strictHpcRuntime?.configured === true
    && strictHpcRuntime?.state === "strict_hpc_runtime_verified"
    && strictHpcProbe?.job_container_verified === true
    && strictHpcProbe?.samples_requested === 5
    && strictHpcProbe?.samples_passed === 5
    && strictHpcSmoke?.status === "passed"
    && !hpcIdentityBlocked;
  const kaggleRealApiReady = (kaggleDpapiReadiness as Record<string, any> | null)?.status === "passed"
    && (kaggleDpapiReadiness as Record<string, any> | null)?.verification_method === "dpapi_real_api"
    && (kaggleDpapiReadiness as Record<string, any> | null)?.credential_status === "authenticated_real_api";

  const connectorStatus: Record<string, Record<string, unknown>> = Object.fromEntries(connectors.map((connector: ConnectorStatus) => [connector.provider, {
    name: connector.name,
    state: connector.state,
    configured: connector.configured,
    notes: connector.detail
  }]));
  if (hpcIdentityBlocked) {
    connectorStatus.gpu = {
      name: "GPU SSH Gateway",
      state: `GPU Degraded: current profile ${currentHpcProfileName} has no bound execution job identity`,
      configured: true,
      current_allocation_blocked: false,
      current_gate_ready: false,
      notes: `Historical ${strictHpcProfileName ?? "HPC"} evidence is archive-only until the current profile is bound to a verified execution run.`
    };
  } else if (strictHpcReady) {
    const profile = String(strictHpcProfile?.profile ?? "named profile");
    const gpuName = String(strictHpcProbe?.samples?.[0]?.gpu_name ?? "verified GPU");
    connectorStatus.gpu = {
      name: "GPU SSH Gateway",
      state: `GPU SSH Gateway Ready: ${profile} / ${gpuName} / 5 of 5 identity samples and bounded smoke passed`,
      configured: true,
      current_allocation_blocked: false,
      current_gate_ready: true,
      proxy: "socks5",
      notes: "Strict named-profile container identity and bounded GPU smoke evidence passed.",
      evidence: {
        report_path: "docs/launch_resource_readiness.json",
        profile,
        job_id: strictHpcProfile?.job_id ?? null,
        gpu_name: gpuName,
        samples_passed: 5,
        bounded_smoke_status: "passed",
        signals_sent: strictHpcProbe?.signals_sent ?? null,
        other_processes_modified: strictHpcProbe?.other_processes_modified ?? null
      }
    };
  }
  if (kaggleRealApiReady) {
    connectorStatus.kaggle = {
      name: "Kaggle",
      state: "Kaggle DPAPI Real API Ready",
      configured: true,
      credential_status: "authenticated_real_api",
      verification_method: "dpapi_real_api",
      human_gate_required_for_submission: true,
      notes: "Read-only Kaggle API smoke passed through the Windows DPAPI profile; official submission remains Human Gate controlled."
    };
  }
  const openAi = openAiConfig();
  const deepSeek = deepSeekConfig();
  const openAiConfigured = openAiApiKeyStatus() === "configured";
  const deepSeekConfigured = deepSeekApiKeyStatus() === "configured";
  connectorStatus.code_agent = {
    name: "Code Agent",
    state: claudeApiKeyStatus() === "configured" ? "Claude Agent SDK Ready" : deepSeekConfigured ? `DeepSeek Code Agent Ready (${deepSeek.model})` : "Not Configured",
    configured: claudeApiKeyStatus() === "configured" || deepSeekConfigured
  };
  connectorStatus.openai = {
    name: "OpenAI-compatible Agent LLM",
    state: openAiConfigured ? `OpenAI Agent Ready (${openAi.model})` : "Not Configured",
    configured: openAiConfigured,
    model: openAi.model,
    base_url: openAi.baseUrl
  };
  connectorStatus.deepseek = {
    name: "DeepSeek",
    state: deepSeekConfigured ? `DeepSeek Ready (${deepSeek.model})` : "Not Configured",
    configured: deepSeekConfigured,
    model: deepSeek.model,
    base_url: deepSeek.baseUrl
  };
  const latestRun = currentRunLedger
    ? runs.find((run: ExperimentRun) => run.id === currentRunLedger.run_id) ?? runs[0]
    : runs[0];

  return {
    hpc_job_lineage: hpcJobLineage,
    tasks: tasks.map(serializeTaskLite),
    connector_status: normalizeConnectorRegistry(connectorStatus),
    runs: runs.map((run: ExperimentRun) => ({
      id: run.id, task_id: run.taskId, output_dir: run.outputDir,
      status: run.id === currentRunLedger?.run_id ? currentRunLedger.status : run.status,
      best_model: run.bestModel, metrics: decodeJson(run.metricsJson), validation_status: run.validationStatus,
      process_id: run.processId, started_at: run.startedAt?.toISOString() ?? null,
      finished_at: run.finishedAt?.toISOString() ?? null, created_at: run.createdAt.toISOString()
    })),
    actions: actions.map((action: ActionLog) => ({
      id: action.id, action: action.action, task_id: action.taskId, run_id: action.runId,
      message: action.message, artifact: action.artifactPath, metadata: decodeJson(action.metadataJson),
      at: action.createdAt.toISOString()
    })),
    gates: gates.map((gate: Gate) => ({
      id: gate.id, task_id: gate.taskId, run_id: gate.runId, gate_type: gate.gateType,
      decision: gate.decision, reviewer: gate.reviewer, evidence: decodeJson(gate.evidenceJson),
      created_at: gate.createdAt.toISOString(), decided_at: gate.decidedAt?.toISOString() ?? null
    })),
    evidence: evidence.map((item: Evidence) => ({
      id: item.id, task_id: item.taskId, run_id: item.runId, label: item.label,
      artifact_path: item.artifactPath, hash: item.hash, source: item.source,
      claim_binding: item.claimBinding, created_at: item.createdAt.toISOString()
    })),
    reports: reports.map((report: Report) => ({
      id: report.id, task_id: report.taskId, run_id: report.runId, title: report.title,
      status: report.status, markdown_content: report.markdownContent, content: decodeJson(report.contentJson),
      markdown_path: report.markdownPath, docx_path: report.docxPath,
      selected_section: report.selectedSection, submitted_at: report.submittedAt?.toISOString() ?? null
    })),
    workflows: workflows.map((workflow: Workflow) => ({
      id: workflow.id, task_id: workflow.taskId, name: workflow.name, status: workflow.status,
      version: workflow.version, nodes: decodeJson(workflow.nodesJson), edges: decodeJson(workflow.edgesJson),
      published_at: workflow.publishedAt?.toISOString() ?? null
    })),
    stages,
    final_delivery_status: finalDeliveryStatus,
    verified_launch_audit: {
      ...(verifiedLaunchAudit as Record<string, unknown> | null),
      latest_readiness: launchReadiness,
      launch_state: (launchReadiness as Record<string, unknown> | null)?.launch_state ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.launch_state ?? null,
      blockers: (launchReadiness as Record<string, unknown> | null)?.blockers ?? (verifiedLaunchAudit as Record<string, unknown> | null)?.blockers ?? [],
      critical_failures: (launchReadiness as Record<string, unknown> | null)?.critical_failures ?? [],
      soft_failures: (launchReadiness as Record<string, unknown> | null)?.soft_failures ?? []
    },
    learning_loop_readiness: learningLoopReadiness,
    kaggle_new_competition_readiness: kaggleNewCompetitionReadiness,
    kaggle_dpapi_readiness: kaggleDpapiReadiness,
    runtime: latestRun ? {
      task_id: currentRunLedger?.task_id ?? latestRun.taskId,
      latest_experiment_dir: currentRunLedger?.output_dir ?? latestRun.outputDir,
      task_state: {
        status: currentRunLedger?.status ?? latestRun.status,
        updated_at: currentRunLedger?.updated_at ?? latestRun.updatedAt.toISOString()
      },
      current_run: currentRunLedger,
      agent_trace: [],
      event_log: [],
      training_log: []
    } : undefined,
    workspace_root: workspaceRoot,
    _meta: { mode: "lite", generated_at: new Date().toISOString() }
  };
}

let lightSummaryCache: FileCacheEntry<SummaryRecord> | null = null;
let lightSummaryPromise: Promise<SummaryRecord> | null = null;
let fullSummaryCache: FileCacheEntry<SummaryRecord> | null = null;
let fullSummaryPromise: Promise<SummaryRecord> | null = null;

async function getLightSummary(force = false) {
  const now = Date.now();
  if (!force && lightSummaryCache && lightSummaryCache.expiresAt > now) return lightSummaryCache.value;
  if (!force && lightSummaryPromise) return lightSummaryPromise;
  const request = buildLightweightSummary().then((value) => {
    lightSummaryCache = { value, mtimeMs: 0, expiresAt: Date.now() + LIGHT_SUMMARY_TTL_MS };
    return value;
  }).finally(() => {
    lightSummaryPromise = null;
  });
  lightSummaryPromise = request;
  return request;
}

async function getFullSummary(force = false) {
  const now = Date.now();
  if (!force && fullSummaryCache && fullSummaryCache.expiresAt > now) return fullSummaryCache.value;
  if (!force && fullSummaryPromise) return fullSummaryPromise;
  const request: Promise<SummaryRecord> = buildFullWorkstationSummary().then((value) => {
    const record = value as SummaryRecord;
    fullSummaryCache = { value: record, mtimeMs: 0, expiresAt: Date.now() + FULL_SUMMARY_TTL_MS };
    return record;
  }).finally(() => {
    fullSummaryPromise = null;
  });
  fullSummaryPromise = request;
  return request;
}

/**
 * The default endpoint is intentionally lightweight. Heavy runtime, literature,
 * scientist and inventory projections are loaded only for the active page and
 * share a short server-side cache to prevent duplicate filesystem scans.
 */
export async function getWorkstationSummary(options: { detail?: string; force?: boolean; full?: boolean } = {}) {
  const base = await getLightSummary(Boolean(options.force));
  if (!options.detail && !options.full) return base;

  const full = await getFullSummary(Boolean(options.force));
  if (options.full) return full;

  const fields = detailFields[options.detail ?? ""] ?? [];
  const detail = Object.fromEntries(fields.filter((field) => field in full).map((field) => [field, full[field]]));
  return {
    ...base,
    ...detail,
    _meta: {
      mode: "detail",
      section: options.detail,
      generated_at: new Date().toISOString(),
      cache_ttl_ms: FULL_SUMMARY_TTL_MS
    }
  };
}
