import { promises as fs } from "node:fs";
import path from "node:path";
import { logAction } from "@/lib/server/actions";
import { runManagedCommand } from "@/lib/server/job-registry";
import { normalizeTaskId, stamp, workspaceRoot } from "@/lib/server/paths";
import { runMCGSExperiment, runEvolutionEngineExperiment } from "@/lib/server/runs";
import type { EvolutionConfigSummary } from "@/lib/api/types";

/**
 * Safe adapter to the Python evolution engine brain.
 *
 * Boundaries honored here:
 *  - No shell string interpolation: we spawn python with an argv array and pass
 *    all caller data through a JSON file (never on the command line).
 *  - No secrets read or printed.
 *  - The CLI itself never bypasses training: `step` is dry_run-first and a real
 *    step returns `blocked_use_workstation`.
 *  - Every call is recorded via logAction() and artifacts land under the
 *    workstation workspace (`workspace/evolution/<task_id>/...`).
 */

export type EvolutionMode = "state" | "plan" | "step" | "graph" | "memory" | "ingest_result" | "ingest_summary";

const EVOLUTION_TASK_ID_RE = /^[A-Za-z0-9._-]+$/;

function pythonExecutable() {
  if (process.env.WORKSTATION_PYTHON) return process.env.WORKSTATION_PYTHON;
  if (process.platform !== "win32") return "python3";
  return "C:\\codex-python\\python.exe";
}

function assertSafeTaskId(taskId: string) {
  if (!taskId || !EVOLUTION_TASK_ID_RE.test(taskId) || taskId.includes("..")) {
    throw new Error(`Unsafe task_id for evolution engine: ${taskId}`);
  }
  return taskId;
}

async function tmpInputFile(taskId: string, mode: EvolutionMode, payload: Record<string, unknown>) {
  const dir = path.join(workspaceRoot, "workspace", "evolution", "_io");
  await fs.mkdir(dir, { recursive: true });
  const file = path.join(dir, `${taskId}_${mode}_${stamp()}.json`);
  await fs.writeFile(file, JSON.stringify(payload ?? {}), "utf-8");
  return file;
}

/** Run one evolution-engine mode and parse its JSON stdout. */
export async function runEvolutionCli(mode: EvolutionMode, payloadInput: Record<string, unknown>) {
  const rawTaskId = typeof payloadInput.task_id === "string" ? payloadInput.task_id : "";
  const taskId = assertSafeTaskId(normalizeTaskId(rawTaskId));
  const payload = { ...payloadInput, task_id: taskId };
  const inputFile = await tmpInputFile(taskId, mode, payload);
  const runId = `evolution_${mode}_${stamp()}`;

  try {
    const { stdout } = await runManagedCommand({
      command: pythonExecutable(),
      args: ["scripts/evolution_engine_cli.py", "--mode", mode, "--input", inputFile],
      cwd: workspaceRoot,
      timeout: 120000,
      taskId,
      runId
    });
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(stdout) as Record<string, unknown>;
    } catch {
      throw new Error(`Evolution CLI returned non-JSON output for mode=${mode}.`);
    }
    if (parsed.ok === false) {
      throw new Error(typeof parsed.error === "string" ? parsed.error : `Evolution CLI failed for mode=${mode}.`);
    }
    return { taskId, payload: parsed };
  } finally {
    await fs.rm(inputFile, { force: true }).catch(() => undefined);
  }
}

const EVOLUTION_CONFIG_DIR = path.join(workspaceRoot, "configs", "evolution");

function numOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Discover the real evolution task configs (configs/evolution/*.json) so the UI
 * task picker stays in sync with the engine's actual task set instead of being
 * hardcoded. Read-only: never writes, never runs training, never reads secrets.
 * The file stem is a valid task_id for evolution_run_cli.py (exact-match branch).
 */
export async function listEvolutionConfigs(): Promise<EvolutionConfigSummary[]> {
  const entries = await fs.readdir(EVOLUTION_CONFIG_DIR, { withFileTypes: true }).catch(() => []);
  const summaries: EvolutionConfigSummary[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const stem = entry.name.slice(0, -".json".length);
    if (!EVOLUTION_TASK_ID_RE.test(stem)) continue; // stay consistent with task-id safety
    const raw = await fs.readFile(path.join(EVOLUTION_CONFIG_DIR, entry.name), "utf-8").catch(() => "");
    if (!raw) continue;
    let cfg: Record<string, unknown>;
    try {
      cfg = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      continue; // skip malformed config rather than fail the whole listing
    }
    summaries.push({
      task_id: stem,
      task_name: typeof cfg.task_name === "string" ? cfg.task_name : stem,
      modality: typeof cfg.modality === "string" ? cfg.modality : undefined,
      task_type: typeof cfg.task_type === "string" ? cfg.task_type : undefined,
      metric: typeof cfg.metric === "string" ? cfg.metric : undefined,
      metric_direction: typeof cfg.metric_direction === "string" ? cfg.metric_direction : undefined,
      n_train: numOrNull(cfg.n_train),
      n_features: numOrNull(cfg.n_features),
      has_gpu_data_dir: typeof cfg.gpu_data_dir === "string" && cfg.gpu_data_dir.length > 0,
      has_local_data_dir: typeof cfg.local_data_dir === "string" && cfg.local_data_dir.length > 0
    });
  }
  summaries.sort((a, b) => a.task_id.localeCompare(b.task_id));
  return summaries;
}

export async function getEvolutionState(taskId: string, extra: Record<string, unknown> = {}) {
  const { payload } = await runEvolutionCli("state", { task_id: taskId, ...extra });
  return payload;
}

export async function getEvolutionGraph(taskId: string, extra: Record<string, unknown> = {}) {
  const { payload } = await runEvolutionCli("graph", { task_id: taskId, ...extra });
  return payload;
}

export async function getEvolutionMemory(taskId: string, extra: Record<string, unknown> = {}) {
  const { payload } = await runEvolutionCli("memory", { task_id: taskId, ...extra });
  return payload;
}

export async function planEvolution(input: Record<string, unknown>) {
  const { taskId, payload } = await runEvolutionCli("plan", input);
  await logAction({
    action: "evolution_plan",
    taskId,
    message: `Evolution plan: ${String(payload.search_controller_decision ?? "planned")} on ${String(payload.selected_branch ?? "EXP000")} (${String(payload.code_generation_mode ?? "Base")}/${String(payload.expansion_type ?? "primary")}).`,
    artifactPath: typeof payload.plan_path === "string" ? payload.plan_path : null,
    metadata: {
      decision: payload.search_controller_decision,
      selected_branch: payload.selected_branch,
      code_generation_mode: payload.code_generation_mode,
      expansion_type: payload.expansion_type,
      official_submit_allowed: false,
      claim_boundary: payload.claim_boundary
    }
  });
  return payload;
}

export async function runEvolutionStep(input: Record<string, unknown>) {
  const dryRun = input.dry_run !== false; // dry_run is the default
  const { taskId, payload } = await runEvolutionCli("step", { ...input, dry_run: dryRun });
  const artifacts = Array.isArray(payload.artifacts) ? (payload.artifacts as string[]) : [];
  await logAction({
    action: dryRun ? "evolution_step_dry_run" : "evolution_step_blocked",
    taskId,
    message: dryRun
      ? `Evolution dry-run step recorded node ${String(payload.exp_id ?? "?")} (${String(payload.code_generation_mode ?? "Base")}/${String(payload.expansion_type ?? "primary")}); no training executed.`
      : `Evolution real step blocked: ${String(payload.reason ?? "use the workstation orchestrator")}`,
    artifactPath: artifacts[0] ?? null,
    metadata: {
      dry_run: dryRun,
      decision: payload.decision,
      gate_status: payload.gate_status,
      artifacts,
      official_submit_allowed: false,
      claim_boundary: payload.claim_boundary
    }
  });
  return payload;
}

/** Backfill bridge: ingest a REAL training result into the graph + memory and
 *  apply the promotion gate. Scores/artifacts are never fabricated by the brain;
 *  it only attaches files that exist on disk under experiments/<task>/<run_id>/. */
export async function ingestEvolutionResult(input: Record<string, unknown>) {
  const { taskId, payload } = await runEvolutionCli("ingest_result", input);
  const promoted = payload.decision === "promoted";
  const best = (payload.best_so_far ?? {}) as Record<string, unknown>;
  await logAction({
    action: promoted ? "evolution_ingest_promoted" : "evolution_ingest_held",
    taskId,
    message: promoted
      ? `Evolution ingested REAL result for ${String(payload.exp_id ?? "?")}: promoted, best_so_far ${String(best.metric ?? "cv")}=${String(best.cv_score ?? "?")}.`
      : `Evolution ingested REAL result for ${String(payload.exp_id ?? "?")}: held (${String((payload.promotion as Record<string, unknown> | undefined)?.reason ?? "not promoted")}).`,
    artifactPath: Array.isArray(payload.artifacts) ? (payload.artifacts as string[])[0] ?? null : null,
    metadata: {
      dry_run: false,
      run_id: payload.run_id,
      decision: payload.decision,
      gate_status: payload.gate_status,
      cv_score: payload.cv_score,
      run_success: payload.run_success,
      artifacts_found: payload.artifacts_found,
      best_so_far: payload.best_so_far,
      official_submit_allowed: false,
      claim_boundary: payload.claim_boundary
    }
  });
  return payload;
}

/** Engine-A backfill bridge: ingest an ALREADY-GATED research_os run by reading
 *  its summary.json + best-EXP validation_contract from exp_dir. Re-applies the
 *  workstation promotion gate on top. Never fabricates: keys on engine A's own
 *  recorded on-disk governance. */
export async function ingestEvolutionSummary(input: Record<string, unknown>) {
  const { taskId, payload } = await runEvolutionCli("ingest_summary", input);
  const promoted = payload.decision === "promoted";
  const best = (payload.best_so_far ?? {}) as Record<string, unknown>;
  await logAction({
    action: promoted ? "evolution_ingest_summary_promoted" : "evolution_ingest_summary_held",
    taskId,
    message: promoted
      ? `Evolution ingested REAL engine-A result ${String(payload.engine_a_exp_id ?? "?")} as ${String(payload.exp_id ?? "?")}: promoted, best_so_far ${String(best.metric ?? "cv")}=${String(best.cv_score ?? "?")}.`
      : `Evolution ingested engine-A result ${String(payload.engine_a_exp_id ?? "?")}: held (${String((payload.promotion as Record<string, unknown> | undefined)?.reason ?? "not promoted")}).`,
    artifactPath: Array.isArray(payload.artifacts) ? (payload.artifacts as string[])[0] ?? null : null,
    metadata: {
      dry_run: false,
      engine: "research_os",
      engine_a_exp_id: payload.engine_a_exp_id,
      decision: payload.decision,
      gate_status: payload.gate_status,
      cv_score: payload.cv_score,
      run_success: payload.run_success,
      artifacts_found: payload.artifacts_found,
      best_so_far: payload.best_so_far,
      official_submit_allowed: false,
      claim_boundary: payload.claim_boundary
    }
  });
  return payload;
}

/**
 * Full closed loop: plan -> (human approval gate) -> REAL training via the
 * workstation orchestrator -> ingest the real score back into the graph + memory.
 *
 * Safety: real training only launches when the caller passes `approve: true`.
 * Without it, we return the plan and stop (no training, no side effects beyond
 * the plan artifacts). `official_submit_allowed` is forced false throughout: this
 * loop never submits to Kaggle and never claims an official rank.
 */
export async function runEvolutionCycle(input: Record<string, unknown>) {
  const rawTaskId = typeof input.task_id === "string" ? input.task_id : "";
  const taskId = assertSafeTaskId(normalizeTaskId(rawTaskId));
  const approved = input.approve === true;

  // 1) Always plan first (safe, no training).
  const plan = await planEvolution({ ...input, task_id: taskId });

  // 2) Human approval gate. Real training is expensive + touches shared state, so
  //    it stays behind an explicit opt-in even though the rest of the loop is auto.
  if (!approved) {
    await logAction({
      action: "evolution_cycle_awaiting_approval",
      taskId,
      message: `Evolution cycle planned ${String(plan.selected_branch ?? "EXP000")} (${String(plan.code_generation_mode ?? "Base")}/${String(plan.expansion_type ?? "primary")}); awaiting approval to launch real training.`,
      artifactPath: typeof plan.plan_path === "string" ? plan.plan_path : null,
      metadata: { stage: "awaiting_approval", official_submit_allowed: false }
    });
    return {
      ok: true,
      task_id: taskId,
      stage: "awaiting_approval",
      approved: false,
      plan,
      next_action: "Re-POST with { approve: true } to launch real training and ingest the score.",
      official_submit_allowed: false,
      claim_boundary: plan.claim_boundary
    };
  }

  // Engine switch: "research_os" drives the corrected engine A (EvolutionLoop);
  // "legacy" (default) keeps the original mlevolve_search path as a fallback.
  const engine = input.engine === "research_os" ? "research_os" : "legacy";
  const runner: "gpu" | "local" = input.runner === "local" ? "local" : "gpu";

  // 3) Approved: launch REAL training through the workstation orchestrator.
  await logAction({
    action: "evolution_cycle_training_launch",
    taskId,
    message: `Evolution cycle approved: launching real ${engine === "research_os" ? `research_os (${runner})` : "MCGS"} training for ${taskId}.`,
    metadata: { stage: "training", engine, runner, official_submit_allowed: false }
  });

  let training: Awaited<ReturnType<typeof runMCGSExperiment>> | Awaited<ReturnType<typeof runEvolutionEngineExperiment>>;
  try {
    training = engine === "research_os"
      ? await runEvolutionEngineExperiment(taskId, {
          runner,
          iterations: typeof input.iterations === "number" ? input.iterations : undefined,
          mcgs: input.mcgs !== false
        })
      : await runMCGSExperiment(taskId, {
          budgetNodes: typeof input.budget_nodes === "number" ? input.budget_nodes : undefined,
          fast: input.fast === true
        });
  } catch (error) {
    const message = error instanceof Error ? error.message : "training failed";
    // Ingest the FAILURE so the graph/memory learn from it (run_success=false ->
    // gate holds, never promotes). This keeps negative results in the loop.
    const ingest = await ingestEvolutionResult({
      ...input, task_id: taskId, run_id: "", cv_score: null, run_success: false,
      method: plan.code_generation_mode, expansion_type: plan.expansion_type
    }).catch(() => null);
    return {
      ok: false, task_id: taskId, stage: "training_failed", approved: true,
      error: message, plan, ingest, official_submit_allowed: false,
      claim_boundary: plan.claim_boundary
    };
  }

  // Training may be BLOCKED by policy (external-resource / local-training gate).
  // In that case there is no score to ingest; report it honestly, do not fabricate.
  if (!("search_result" in training)) {
    const blocked = training as { status?: string; reason?: string; next_action?: string };
    await logAction({
      action: "evolution_cycle_training_blocked",
      taskId,
      message: `Evolution cycle: real training blocked by policy (${String(blocked.reason ?? "local training disabled")}).`,
      metadata: { stage: "training_blocked", official_submit_allowed: false }
    });
    return {
      ok: false, task_id: taskId, stage: "training_blocked", approved: true,
      reason: blocked.reason ?? "local training disabled by policy",
      next_action: blocked.next_action, plan, official_submit_allowed: false,
      claim_boundary: plan.claim_boundary
    };
  }

  // 4) Ingest the REAL result back into the graph + memory.
  const searchResult = (training.search_result ?? {}) as Record<string, unknown>;
  const cvScore = typeof training.best_score === "number" ? training.best_score : null;
  let ingest: Record<string, unknown>;
  if (engine === "research_os") {
    // Engine A already gated its run; ingest_summary reads its summary.json from
    // exp_dir and re-applies the workstation gate. No fabricated local artifacts.
    const expDir = typeof searchResult.exp_dir === "string" ? searchResult.exp_dir : "";
    ingest = await ingestEvolutionSummary({
      ...input,
      task_id: taskId,
      exp_dir: expDir,
      method: plan.code_generation_mode,
      expansion_type: plan.expansion_type
    });
  } else {
    const runId = typeof searchResult.best_run_id === "string" ? searchResult.best_run_id : "";
    ingest = await ingestEvolutionResult({
      ...input,
      task_id: taskId,
      run_id: runId,
      cv_score: cvScore,
      run_success: cvScore !== null,
      method: plan.code_generation_mode,
      expansion_type: plan.expansion_type
    });
  }

  return {
    ok: true,
    task_id: taskId,
    stage: "completed",
    approved: true,
    plan,
    training: { run_id: training.run_id, best_score: training.best_score, nodes_evaluated: training.nodes_evaluated },
    ingest,
    best_so_far: ingest.best_so_far,
    official_submit_allowed: false,
    claim_boundary: ingest.claim_boundary
  };
}
