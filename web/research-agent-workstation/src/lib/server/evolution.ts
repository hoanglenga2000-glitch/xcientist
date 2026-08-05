import { promises as fs } from "node:fs";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { runManagedCommand } from "@/lib/server/job-registry";
import { assertWorkspacePath, normalizeTaskId, stamp, workspaceRoot } from "@/lib/server/paths";
import { runMCGSExperiment, runEvolutionEngineExperiment } from "@/lib/server/runs";
import { CANONICAL_HASH_SCHEMA, sha256Canonical, verifyExperienceBoard } from "@/lib/server/evolution-integrity";
import { loadExperienceReviewAudit } from "@/lib/server/experience-review-audit";
import { consumeEvolutionApprovalPlan, createEvolutionApprovalPlan } from "@/lib/server/evolution-approval";
import type {
  EvolutionBudgetLedger,
  EvolutionCandidateFreezeSummary,
  EvolutionConfigSummary,
  EvolutionExperienceCard,
  EvolutionExperienceResponse,
  EvolutionPairedABSummary,
  EvolutionRetrievalBundle,
  EvolutionSearchMode,
  EvolutionSearchOperator,
  EvolutionSelectionTrace
} from "@/lib/api/types";

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

async function writeJsonAtomicExclusive(file: string, payload: Record<string, unknown>) {
  // Reject values outside the versioned canonical JSON contract before they
  // become release evidence. A hard-link commit is atomic and create-only on
  // the same filesystem, so an existing immutable receipt is never replaced.
  sha256Canonical(payload);
  const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await fs.link(temporary, file).catch(async (error) => {
    await fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  });
  // The immutable target is already committed. A transient cleanup failure
  // must not turn a successful create into a non-retryable false failure.
  await fs.rm(temporary, { force: true }).catch(() => undefined);
}

async function fileSha256(file: string) {
  return createHash("sha256").update(await fs.readFile(file)).digest("hex");
}

function samePath(left: string, right: string) {
  return path.relative(left, right) === "" && path.relative(right, left) === "";
}

function pathInside(root: string, target: string) {
  const relative = path.relative(root, target);
  return Boolean(relative) && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function exactDemoRunRoot(expDir: string, taskId: string, runId: string) {
  if (!runId || !/^[A-Za-z0-9._-]{1,220}$/.test(runId) || runId.includes("..")) {
    throw new Error("Demo database run identity is invalid.");
  }
  if (!expDir) throw new Error("Demo run root is missing.");
  const target = path.resolve(assertWorkspacePath(path.isAbsolute(expDir) ? expDir : path.join(workspaceRoot, expDir)));
  const allowed = path.resolve(workspaceRoot, "experiments", "evolution");
  if (!pathInside(allowed, target)) {
    throw new Error("Demo run root is outside experiments/evolution.");
  }
  if (!path.basename(target).startsWith(`${taskId}_`)) throw new Error("Demo run root identity mismatch.");
  const stat = await fs.lstat(target);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error("Demo run root is not a regular directory.");
  const [realTarget, realAllowed, databaseRun, latestTaskRun] = await Promise.all([
    fs.realpath(target),
    fs.realpath(allowed),
    prisma.experimentRun.findUnique({
      where: { id: runId },
      select: { taskId: true, outputDir: true, status: true, validationStatus: true },
    }),
    prisma.experimentRun.findFirst({
      where: { taskId },
      orderBy: { createdAt: "desc" },
      select: { id: true },
    }),
  ]);
  if (!pathInside(realAllowed, realTarget)) throw new Error("Demo run realpath escaped experiments/evolution.");
  if (!path.basename(realTarget).startsWith(`${taskId}_`)) throw new Error("Demo run realpath identity mismatch.");
  if (
    !databaseRun
    || databaseRun.taskId !== taskId
    || databaseRun.status !== "passed"
    || databaseRun.validationStatus !== "passed"
    || !databaseRun.outputDir
    || latestTaskRun?.id !== runId
  ) {
    throw new Error("Demo database run binding is not the newest passed task run.");
  }
  const databaseTarget = path.resolve(assertWorkspacePath(
    path.isAbsolute(databaseRun.outputDir) ? databaseRun.outputDir : path.join(workspaceRoot, databaseRun.outputDir),
  ));
  const realDatabaseTarget = await fs.realpath(databaseTarget);
  if (!samePath(realTarget, realDatabaseTarget)) throw new Error("Demo database outputDir does not bind the exact run root.");
  return realTarget;
}

async function exactDemoArtifact(runRoot: string, name: string) {
  const candidate = path.join(runRoot, name);
  const stat = await fs.lstat(candidate);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Demo cycle evidence is missing: ${name}`);
  const real = await fs.realpath(candidate);
  if (!samePath(path.dirname(real), runRoot) || path.basename(real) !== name) {
    throw new Error(`Demo cycle evidence escaped the exact run root: ${name}`);
  }
  return real;
}

async function readEvidenceRecord(file: string, label: string) {
  const stat = await fs.stat(file);
  if (!stat.isFile() || stat.size <= 0 || stat.size > 1024 * 1024) {
    throw new Error(`${label} is not a bounded JSON artifact.`);
  }
  const parsed = JSON.parse(await fs.readFile(file, "utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must contain a JSON object.`);
  }
  sha256Canonical(parsed);
  return parsed as Record<string, unknown>;
}

function assertCanonicalField(
  left: Record<string, unknown>,
  right: Record<string, unknown>,
  key: string,
  label: string,
) {
  if (!Object.hasOwn(left, key) || !Object.hasOwn(right, key) || sha256Canonical(left[key]) !== sha256Canonical(right[key])) {
    throw new Error(`${label} drifted at ${key}.`);
  }
}

async function verifyDemoInputCliBinding(
  runRoot: string,
  taskId: string,
  approval: Awaited<ReturnType<typeof consumeEvolutionApprovalPlan>>,
  inputFile: string,
  cliFile: string,
) {
  const [inputContract, cliReceipt] = await Promise.all([
    readEvidenceRecord(inputFile, "Demo input contract"),
    readEvidenceRecord(cliFile, "Demo CLI receipt"),
  ]);
  if (
    inputContract.schema !== "evomind.demo.input_contract.v1"
    || inputContract.task_id !== taskId
    || inputContract.demo_campaign !== true
    || inputContract.official_submission !== "disabled"
  ) {
    throw new Error("Demo input contract identity verification failed.");
  }
  const requestContract = approval.request_contract;
  for (const key of [
    "task_id", "runner", "iterations", "mcgs", "search_mode",
    "max_nodes", "max_tokens", "max_wall_seconds", "max_cost",
  ]) {
    assertCanonicalField(inputContract, requestContract, key, "Demo approved request contract");
  }

  if (
    cliReceipt.schema !== "evomind.evolution_cli_receipt.v1"
    || cliReceipt.ok !== true
    || cliReceipt.task_id !== taskId
    || cliReceipt.demo_campaign !== true
  ) {
    throw new Error("Demo CLI receipt identity verification failed.");
  }
  for (const key of ["runner", "search_mode"]) {
    assertCanonicalField(cliReceipt, inputContract, key, "Demo CLI receipt");
  }
  if (sha256Canonical(cliReceipt.n_iterations) !== sha256Canonical(inputContract.iterations)) {
    throw new Error("Demo CLI receipt iteration count drifted from its input contract.");
  }
  const budgets = cliReceipt.budgets;
  if (!budgets || typeof budgets !== "object" || Array.isArray(budgets)) {
    throw new Error("Demo CLI receipt budgets are missing.");
  }
  for (const key of ["max_nodes", "max_tokens", "max_wall_seconds", "max_cost"]) {
    assertCanonicalField(budgets as Record<string, unknown>, inputContract, key, "Demo CLI budget");
  }

  const inputSha256 = await fileSha256(inputFile);
  if (cliReceipt.input_contract_sha256 !== inputSha256) {
    throw new Error("Demo CLI receipt input contract SHA-256 mismatch.");
  }
  for (const [field, expected] of [["input_contract_path", inputFile], ["exp_dir", runRoot]] as const) {
    const declared = cliReceipt[field];
    if (typeof declared !== "string" || !declared) throw new Error(`Demo CLI receipt ${field} is missing.`);
    const configured = path.resolve(assertWorkspacePath(path.isAbsolute(declared) ? declared : path.join(workspaceRoot, declared)));
    const real = await fs.realpath(configured);
    if (!samePath(real, expected)) throw new Error(`Demo CLI receipt ${field} does not bind the exact artifact.`);
  }
  return { inputContract, cliReceipt, inputSha256 };
}

async function writeDemoCycleReceipts(
  taskId: string,
  runId: string,
  expDir: string,
  approval: Awaited<ReturnType<typeof consumeEvolutionApprovalPlan>>,
) {
  const runRoot = await exactDemoRunRoot(expDir, taskId, runId);
  const requiredNames = [
    "input-contract.json",
    "cli-receipt.json",
    "candidate-freeze.json",
    "independent-review.json",
    "claim-audit.json",
  ] as const;
  const artifactPaths = Object.fromEntries(await Promise.all(
    requiredNames.map(async (name) => [name, await exactDemoArtifact(runRoot, name)]),
  )) as Record<typeof requiredNames[number], string>;
  await verifyDemoInputCliBinding(
    runRoot,
    taskId,
    approval,
    artifactPaths["input-contract.json"],
    artifactPaths["cli-receipt.json"],
  );

  const approvalFile = path.join(runRoot, "approval-receipt.json");
  await writeJsonAtomicExclusive(approvalFile, approval as unknown as Record<string, unknown>);
  const artifactHashes = Object.fromEntries(await Promise.all(
    [...requiredNames, "approval-receipt.json"].map(async (name) => [name, await fileSha256(path.join(runRoot, name))]),
  )) as Record<string, string>;
  const cycleReceipt = {
    schema: "evomind.demo.cycle_receipt.v1",
    hash_canonicalization: CANONICAL_HASH_SCHEMA,
    task_id: taskId,
    run_id: runId,
    output_dir_name: path.basename(runRoot),
    status: "completed",
    plan_id: approval.plan_id,
    plan_sha256: approval.plan_sha256,
    request_fingerprint: approval.request_fingerprint,
    request_contract_sha256: sha256Canonical(approval.request_contract),
    // This field is the canonical approval content hash used by the independent
    // verifier. The exact on-disk byte hash remains separately and unambiguously
    // bound both here and in artifact_hashes.
    approval_receipt_sha256: approval.receipt_sha256,
    approval_receipt_file_sha256: artifactHashes["approval-receipt.json"],
    input_contract_sha256: artifactHashes["input-contract.json"],
    cli_receipt_sha256: artifactHashes["cli-receipt.json"],
    artifact_hashes: artifactHashes,
    official_submit_allowed: false,
    completed_at: new Date().toISOString(),
  };
  const cycleFile = path.join(runRoot, "cycle-receipt.json");
  await writeJsonAtomicExclusive(cycleFile, cycleReceipt);
  return {
    approval_receipt: path.relative(workspaceRoot, approvalFile),
    cycle_receipt: path.relative(workspaceRoot, cycleFile),
    cycle_receipt_sha256: await fileSha256(cycleFile),
  };
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
      has_local_data_dir: typeof cfg.local_data_dir === "string" && cfg.local_data_dir.length > 0,
      compute_backend: typeof cfg.compute_backend === "string" ? cfg.compute_backend : undefined,
      demo_campaign: cfg.demo_campaign === true,
      required_runner: ["local", "local_gpu", "gpu"].includes(String(cfg.required_runner))
        ? cfg.required_runner as "local" | "local_gpu" | "gpu"
        : undefined,
      default_search_mode: ["legacy_uct", "experience_mcgs_v1"].includes(String(cfg.default_search_mode))
        ? cfg.default_search_mode as EvolutionSearchMode
        : undefined,
      required_iterations: numOrNull(cfg.required_iterations),
      required_max_nodes: numOrNull(cfg.required_max_nodes),
      required_max_tokens: numOrNull(cfg.required_max_tokens),
      required_max_wall_seconds: numOrNull(cfg.required_max_wall_seconds),
      required_max_cost: numOrNull(cfg.required_max_cost),
      required_mcgs: typeof cfg.required_mcgs === "boolean" ? cfg.required_mcgs : undefined
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

const EXPERIENCE_ARTIFACT_LIMIT = 8 * 1024 * 1024;
const EXPERIENCE_OPERATORS: EvolutionSearchOperator[] = ["Draft", "Improve", "Debug", "Crossover"];

function recordOf(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function finiteInteger(value: unknown): number {
  const number = finiteNumber(value);
  return number === null ? 0 : Math.max(0, Math.trunc(number));
}

function optionalFiniteInteger(value: unknown): number | null {
  const number = finiteNumber(value);
  return number === null || !Number.isInteger(number) || number < 0 ? null : number;
}

function stringValue(value: unknown, maxLength = 500): string {
  return typeof value === "string" ? value.slice(0, maxLength) : "";
}

function stringArray(value: unknown, limit = 32): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string").slice(0, limit).map((item) => item.slice(0, 200))
    : [];
}

function searchOperator(value: unknown): EvolutionSearchOperator {
  if (!EXPERIENCE_OPERATORS.includes(value as EvolutionSearchOperator)) {
    throw new Error(`Unknown Experience-MCGS operator: ${String(value).slice(0, 40)}`);
  }
  return value as EvolutionSearchOperator;
}

async function readBoundedJson(filePath: string): Promise<Record<string, unknown> | null> {
  const stat = await fs.stat(filePath).catch(() => null);
  if (!stat?.isFile() || stat.size > EXPERIENCE_ARTIFACT_LIMIT) return null;
  const raw = await fs.readFile(filePath, "utf-8").catch(() => "");
  if (!raw) return null;
  try {
    return recordOf(JSON.parse(raw.replace(/^\uFEFF/, "")));
  } catch {
    return null;
  }
}

async function readBoundedJsonl(filePath: string, limit = 512): Promise<Record<string, unknown>[]> {
  const stat = await fs.stat(filePath).catch(() => null);
  if (!stat?.isFile() || stat.size > EXPERIENCE_ARTIFACT_LIMIT) return [];
  const raw = await fs.readFile(filePath, "utf-8").catch(() => "");
  const records: Record<string, unknown>[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim() || records.length >= limit) continue;
    try {
      records.push(recordOf(JSON.parse(line)));
    } catch {
      // A partial final line can occur while a live run is flushing. Ignore it.
    }
  }
  return records;
}

async function canonicalRunArtifact(runRoot: string, name: string): Promise<string | null> {
  const configured = path.resolve(runRoot, name);
  const real = await fs.realpath(configured).catch(() => null);
  if (!real) return null;
  const root = path.resolve(runRoot);
  const relative = path.relative(root, real);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative) ? real : null;
}

async function readRunJson(runRoot: string, name: string) {
  const file = await canonicalRunArtifact(runRoot, name);
  return file ? readBoundedJson(file) : null;
}

async function readRunJsonl(runRoot: string, name: string) {
  const file = await canonicalRunArtifact(runRoot, name);
  return file ? readBoundedJsonl(file) : [];
}

function projectExperienceCard(value: unknown): EvolutionExperienceCard | null {
  const card = recordOf(value);
  const cardId = stringValue(card.card_id, 100);
  const nodeId = stringValue(card.node_id, 100);
  if (!cardId || !nodeId) return null;
  const cost = recordOf(card.cost);
  const content = { ...card };
  delete content.card_id;
  return {
    card_id: cardId,
    node_id: nodeId,
    parent_ids: stringArray(card.parent_ids, 8),
    operator: searchOperator(card.operator),
    method_family: stringValue(card.method_family, 120) || "unknown",
    status: stringValue(card.status, 40) || "unknown",
    public_validation_score: finiteNumber(card.public_validation_score),
    metric_direction: stringValue(card.metric_direction, 40) || "maximize",
    error_signature: stringValue(card.error_signature, 80),
    code_hash: stringValue(card.code_hash, 64),
    prompt_hash: stringValue(card.prompt_hash, 64),
    execution_id: stringValue(card.execution_id, 100),
    content_hash: sha256Canonical(content),
    cost: {
      prompt_tokens: finiteInteger(cost.prompt_tokens),
      completion_tokens: finiteInteger(cost.completion_tokens),
      total_tokens: finiteInteger(cost.total_tokens),
      wall_seconds: Math.max(0, finiteNumber(cost.wall_seconds) ?? 0),
      gpu_seconds: Math.max(0, finiteNumber(cost.gpu_seconds) ?? 0),
      estimated_cost_usd: Math.max(0, finiteNumber(cost.estimated_cost_usd) ?? 0)
    },
    // The raw provenance object is intentionally not exposed. Summary is capped
    // and comes from the public-only card boundary.
    summary: stringValue(card.summary, 500)
  };
}

function projectSelectionTrace(value: unknown): EvolutionSelectionTrace | null {
  const trace = recordOf(value);
  const selectedNodeId = stringValue(trace.selected_node_id, 100);
  const selectedCardId = stringValue(trace.selected_card_id, 100);
  if (!selectedNodeId || !selectedCardId) return null;
  const candidates = (Array.isArray(trace.candidates) ? trace.candidates : []).slice(0, 128).flatMap((candidate) => {
    const item = recordOf(candidate);
    const nodeId = stringValue(item.node_id, 100);
    const cardId = stringValue(item.card_id, 100);
    if (!nodeId || !cardId) return [];
    return [{
      node_id: nodeId,
      card_id: cardId,
      quality: finiteNumber(item.quality) ?? 0,
      progress: finiteNumber(item.progress) ?? 0,
      novelty: finiteNumber(item.novelty) ?? 0,
      exploration: finiteNumber(item.exploration) ?? 0,
      utility: finiteNumber(item.utility) ?? 0,
      visits: finiteInteger(item.visits),
      parent_visits: finiteInteger(item.parent_visits)
    }];
  });
  return {
    selected_node_id: selectedNodeId,
    selected_card_id: selectedCardId,
    exploration_c: finiteNumber(trace.exploration_c) ?? 1,
    tie_break: stringValue(trace.tie_break, 160),
    candidates,
    operator: EXPERIENCE_OPERATORS.includes(trace.operator as EvolutionSearchOperator)
      ? trace.operator as EvolutionSearchOperator
      : null,
    selection_reason: stringValue(trace.selection_reason, 240) || null,
    selected_parent_ids: stringArray(trace.selected_parent_ids, 2)
  };
}

function projectRetrieval(value: unknown): EvolutionRetrievalBundle | null {
  const retrieval = recordOf(value);
  const operator = searchOperator(retrieval.operator);
  const cardIds = stringArray(retrieval.card_ids, 8);
  if (!cardIds.length && !stringValue(retrieval.selected_node_id, 100)) return null;
  return {
    seq: finiteNumber(retrieval.seq),
    ts: stringValue(retrieval.ts, 64) || null,
    operator,
    selected_node_id: stringValue(retrieval.selected_node_id, 100) || null,
    card_ids: cardIds,
    estimated_tokens: finiteInteger(retrieval.estimated_tokens),
    max_cards: finiteInteger(retrieval.max_cards),
    max_tokens: finiteInteger(retrieval.max_tokens),
    truncated_by: stringValue(retrieval.truncated_by, 80),
    cache_key: stringValue(retrieval.cache_key, 64),
    cache_hit: typeof retrieval.cache_hit === "boolean" ? retrieval.cache_hit : null,
    cache_status: stringValue(retrieval.cache_status, 40) || "not_recorded",
    card_cache_hits: optionalFiniteInteger(retrieval.card_cache_hits),
    card_cache_misses: optionalFiniteInteger(retrieval.card_cache_misses)
  };
}

function projectBudget(value: unknown): EvolutionBudgetLedger | null {
  const budget = recordOf(value);
  if (!Object.keys(budget).length) return null;
  return {
    max_nodes: finiteInteger(budget.max_nodes),
    max_total_tokens: finiteInteger(budget.max_total_tokens),
    max_wall_seconds: Math.max(0, finiteNumber(budget.max_wall_seconds) ?? 0),
    max_cost_usd: finiteNumber(budget.max_cost_usd),
    nodes: finiteInteger(budget.nodes),
    prompt_tokens: finiteInteger(budget.prompt_tokens),
    completion_tokens: finiteInteger(budget.completion_tokens),
    total_tokens: finiteInteger(budget.total_tokens),
    wall_seconds: Math.max(0, finiteNumber(budget.wall_seconds) ?? 0),
    gpu_seconds: Math.max(0, finiteNumber(budget.gpu_seconds) ?? 0),
    estimated_cost_usd: Math.max(0, finiteNumber(budget.estimated_cost_usd) ?? 0),
    terminal_reason: stringValue(budget.terminal_reason, 100),
    can_continue: budget.can_continue === true
  };
}

function evidenceRecords(payload: Record<string, unknown>): Record<string, unknown>[] {
  return [
    payload,
    recordOf(payload.summary),
    recordOf(payload.metrics),
    recordOf(payload.aggregate),
    recordOf(payload.result),
    recordOf(payload.paired_bootstrap),
    recordOf(payload.confidence_interval)
  ];
}

function firstString(records: Record<string, unknown>[], keys: string[]): string | null {
  for (const record of records) for (const key of keys) {
    const value = stringValue(record[key], 200);
    if (value) return value;
  }
  return null;
}

function firstNumber(records: Record<string, unknown>[], keys: string[]): number | null {
  for (const record of records) for (const key of keys) {
    const value = finiteNumber(record[key]);
    if (value !== null) return value;
  }
  return null;
}

async function readFirstArtifact(runRoot: string, names: string[]) {
  for (const name of names) {
    const file = await canonicalRunArtifact(runRoot, name);
    const payload = file ? await readBoundedJson(file) : null;
    if (payload && file) {
      const sha256 = createHash("sha256").update(await fs.readFile(file)).digest("hex");
      return { name, payload, sha256 };
    }
  }
  return null;
}

function projectCandidateFreeze(artifact: Awaited<ReturnType<typeof readFirstArtifact>>): EvolutionCandidateFreezeSummary {
  if (!artifact) return {
    present: false, status: null, candidate_id: null, frozen_at: null, sha256: null,
    artifact_count: null, source: "local_candidate_freeze", artifact: null
  };
  const records = evidenceRecords(artifact.payload);
  const frozen = Array.isArray(artifact.payload.artifacts)
    ? artifact.payload.artifacts
    : Array.isArray(artifact.payload.files)
      ? artifact.payload.files
    : Array.isArray(artifact.payload.frozen_artifacts)
      ? artifact.payload.frozen_artifacts
      : Array.isArray(artifact.payload.records)
        ? artifact.payload.records
        : null;
  return {
    present: true,
    status: firstString(records, ["status", "freeze_status", "decision"]),
    candidate_id: firstString(records, ["candidate_id", "best_exp_id", "node_id"]),
    frozen_at: firstString(records, ["frozen_at", "created_at", "generated_at"]),
    // Always expose the hash of this exact file's bytes. Payload-declared hashes
    // are links to other evidence and must never replace the artifact digest.
    sha256: artifact.sha256,
    artifact_count: frozen ? frozen.length : firstNumber(records, ["artifact_count", "frozen_artifact_count"]),
    source: "local_candidate_freeze",
    artifact: artifact.name
  };
}

function projectPairedAb(artifact: Awaited<ReturnType<typeof readFirstArtifact>>): EvolutionPairedABSummary {
  if (!artifact) return {
    present: false, status: null, phase: null, task_count: null, seed_count: null, paired_wins: null,
    baseline_valid_rate: null, treatment_valid_rate: null, median_paired_delta: null, ci_low: null, ci_high: null,
    prompt_p99_delta_percent: null, new_best_per_million_delta_percent: null, source: "local_mle_grader"
  };
  const records = evidenceRecords(artifact.payload);
  const baseline = recordOf(artifact.payload.baseline);
  const treatment = recordOf(artifact.payload.treatment);
  return {
    present: true,
    status: firstString(records, ["status", "decision", "gate_status"]),
    phase: firstString(records, ["phase", "campaign_phase", "evaluation_phase"]),
    task_count: firstNumber(records, ["task_count", "n_tasks", "tasks_completed"]),
    seed_count: firstNumber(records, ["seed_count", "n_seeds"]),
    paired_wins: firstNumber(records, ["paired_wins", "task_wins", "wins"]),
    baseline_valid_rate: firstNumber([baseline, ...records], ["valid_rate", "baseline_valid_rate"]),
    treatment_valid_rate: firstNumber([treatment, ...records], ["valid_rate", "treatment_valid_rate"]),
    median_paired_delta: firstNumber(records, ["median_paired_delta", "paired_delta_median", "median_delta"]),
    ci_low: firstNumber(records, ["ci_low", "lower", "lower_bound", "ci_95_low"]),
    ci_high: firstNumber(records, ["ci_high", "upper", "upper_bound", "ci_95_high"]),
    prompt_p99_delta_percent: firstNumber(records, ["prompt_p99_delta_percent", "prompt_p99_reduction_percent"]),
    new_best_per_million_delta_percent: firstNumber(records, ["new_best_per_million_delta_percent", "new_best_per_million_improvement_percent"]),
    source: "local_mle_grader"
  };
}

async function evolutionRunRoot(taskId: string, runId?: string) {
  const recent = runId
    ? await prisma.experimentRun.findMany({ where: { id: runId, taskId }, take: 1 })
    : await prisma.experimentRun.findMany({ where: { taskId }, orderBy: { createdAt: "desc" }, take: 30 });
  const configuredEvidenceRoot = path.resolve(workspaceRoot, "experiments", "evolution");
  const evidenceRoot = await fs.realpath(configuredEvidenceRoot).catch(() => configuredEvidenceRoot);
  for (const run of recent) {
    if (!run.outputDir) continue;
    const configuredCandidate = path.isAbsolute(run.outputDir)
      ? path.resolve(run.outputDir)
      : path.resolve(workspaceRoot, run.outputDir);
    const candidate = await fs.realpath(configuredCandidate).catch(() => configuredCandidate);
    assertWorkspacePath(candidate);
    const relative = path.relative(evidenceRoot, candidate);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) continue;
    const stat = await fs.stat(candidate).catch(() => null);
    if (stat?.isDirectory()) return { run, root: candidate };
  }
  return null;
}

/** Read a bounded, public-validation-only projection of Experience-MCGS artifacts. */
export async function getEvolutionExperience(taskIdInput: string, runIdInput = ""): Promise<EvolutionExperienceResponse> {
  const taskId = assertSafeTaskId(normalizeTaskId(taskIdInput));
  const runId = runIdInput.trim();
  if (runId && (!/^[A-Za-z0-9._-]{1,220}$/.test(runId) || runId.includes(".."))) {
    throw new Error("Unsafe run_id for evolution experience.");
  }
  const located = await evolutionRunRoot(taskId, runId || undefined);
  const emptyFreeze = projectCandidateFreeze(null);
  const emptyPaired = projectPairedAb(null);
  const emptyIntegrity = verifyExperienceBoard(null);
  const emptyIndependentReview: EvolutionExperienceResponse["independent_review"] = {
    present: false, status: null, reviewer: null, reviewed_at: null, rows: null, metric: null,
    score: null, sha256: null, candidate_freeze_sha256: null, review_source: null, artifact: null
  };
  const emptyClaimAudit: EvolutionExperienceResponse["claim_audit"] = {
    present: false, status: null, sha256: null, supported_claim_count: null,
    unsupported_claim_count: null, claim_boundary: null, artifact: null
  };
  const emptyReviewChain: EvolutionExperienceResponse["review_chain"] = {
    status: "not_present",
    stages: { candidate_freeze: "not_present", independent_review: "not_present", claim_audit: "not_present" },
    artifact_hashes_valid: null, run_binding_valid: null,
    review_candidate_freeze_hash_valid: null, claim_candidate_freeze_hash_valid: null,
    claim_independent_review_hash_valid: null, errors: []
  };
  const emptyLineage: EvolutionExperienceResponse["lineage"] = { nodes: [], edges: [] };
  if (!located) return {
    ok: true, task_id: taskId, run_id: runId || null, present: false, search_mode: "legacy_uct", board_hash: null,
    integrity: emptyIntegrity, lineage: emptyLineage,
    cards: [], selection_traces: [], retrievals: [], budget: null,
    operator_counts: { Draft: 0, Improve: 0, Debug: 0, Crossover: 0 },
    cache: { hits: 0, misses: 0, unknown: 0, source: "none", stats_consistent: null },
    artifacts: [], candidate_freeze: emptyFreeze,
    independent_review: emptyIndependentReview, claim_audit: emptyClaimAudit, review_chain: emptyReviewChain,
    paired_ab: emptyPaired,
    claim_boundary: "No Experience-MCGS artifact is bound to this task/run."
  };

  const contract = await readRunJson(located.root, "search-run-contract.json");
  const board = await readRunJson(located.root, "experience-board.json");
  const integrity = verifyExperienceBoard(board);
  const trustedBoard = integrity.status === "verified" ? board : null;
  const traceRecords = await readRunJsonl(located.root, "selection-traces.jsonl");
  const retrievalRecords = await readRunJsonl(located.root, "retrieval-bundles.jsonl");
  const summaryCacheStats = await readRunJson(located.root, "summary-cache-stats.json");
  const budgetPayload = await readRunJson(located.root, "budget-ledger.json");
  const reviewAudit = await loadExperienceReviewAudit(located.root, taskId);
  const candidateArtifact = reviewAudit.candidate_freeze.present
    ? null
    : await readFirstArtifact(located.root, ["candidate_freeze.json", "freeze-manifest.json"]);
  const pairedArtifact = await readFirstArtifact(located.root, ["paired-ab-report.json", "paired_ab_report.json", "ab-report.json", "ab_report.json", "screening-ab-report.json", "formal-ab-report.json"]);
  const candidateFreeze = reviewAudit.candidate_freeze.present
    ? reviewAudit.candidate_freeze
    : projectCandidateFreeze(candidateArtifact);

  const cards = (Array.isArray(trustedBoard?.cards) ? trustedBoard.cards : []).slice(0, 512).flatMap((item) => {
    const card = projectExperienceCard(item);
    return card ? [card] : [];
  });
  const selectionTraces = traceRecords.flatMap((item) => {
    const trace = projectSelectionTrace(item);
    return trace ? [trace] : [];
  });
  const retrievals = retrievalRecords.flatMap((item) => {
    const retrieval = projectRetrieval(item);
    return retrieval ? [retrieval] : [];
  });
  const retrievalsWithCardStats = retrievals.filter(
    (item) => item.card_cache_hits !== null && item.card_cache_misses !== null
  );
  const retrievalCardHits = retrievalsWithCardStats.reduce((total, item) => total + (item.card_cache_hits ?? 0), 0);
  const retrievalCardMisses = retrievalsWithCardStats.reduce((total, item) => total + (item.card_cache_misses ?? 0), 0);
  const statsHits = optionalFiniteInteger(summaryCacheStats?.hits);
  const statsMisses = optionalFiniteInteger(summaryCacheStats?.misses);
  const summaryStatsValid = summaryCacheStats?.schema === "evomind.experience_mcgs.summary_cache_stats.v1"
    && statsHits !== null
    && statsMisses !== null;
  const cardStatsComplete = retrievals.length > 0 && retrievalsWithCardStats.length === retrievals.length;
  const cacheStatsConsistent = summaryStatsValid && cardStatsComplete
    ? statsHits === retrievalCardHits && statsMisses === retrievalCardMisses
    : null;
  const cacheSummary: EvolutionExperienceResponse["cache"] = summaryStatsValid
    ? {
        hits: statsHits,
        misses: statsMisses,
        unknown: retrievals.length - retrievalsWithCardStats.length,
        source: "summary_cache_stats",
        stats_consistent: cacheStatsConsistent
      }
    : retrievalsWithCardStats.length > 0
      ? {
          hits: retrievalCardHits,
          misses: retrievalCardMisses,
          unknown: retrievals.length - retrievalsWithCardStats.length,
          source: "retrieval_bundles",
          stats_consistent: null
        }
      : {
          hits: retrievals.filter((item) => item.cache_hit === true).length,
          misses: retrievals.filter((item) => item.cache_hit === false).length,
          unknown: retrievals.filter((item) => item.cache_hit === null).length,
          source: retrievals.length ? "bundle_status" : "none",
          stats_consistent: null
        };
  const cacheEvidenceFailed = cacheSummary.stats_consistent === false;
  const operatorCounts: Record<EvolutionSearchOperator, number> = { Draft: 0, Improve: 0, Debug: 0, Crossover: 0 };
  for (const card of cards) operatorCounts[card.operator] += 1;
  const artifacts = [...new Set([
    contract && "search-run-contract.json",
    board && "experience-board.json",
    traceRecords.length && "selection-traces.jsonl",
    retrievalRecords.length && "retrieval-bundles.jsonl",
    summaryCacheStats && "summary-cache-stats.json",
    budgetPayload && "budget-ledger.json",
    candidateArtifact?.name,
    ...reviewAudit.artifact_names,
    pairedArtifact?.name
  ].filter((item): item is string => typeof item === "string" && Boolean(item)))];
  const contractMode = stringValue(contract?.search_mode, 40);
  const searchMode: EvolutionSearchMode = contractMode === "experience_mcgs_v1" ? "experience_mcgs_v1" : "legacy_uct";
  const lineage: EvolutionExperienceResponse["lineage"] = {
    nodes: cards.map((card) => ({
      node_id: card.node_id,
      operator: card.operator,
      method_family: card.method_family,
      score: card.public_validation_score
    })),
    edges: cards.flatMap((card) => card.parent_ids.map((parentNodeId) => ({
      parent_node_id: parentNodeId,
      child_node_id: card.node_id
    })))
  };
  return {
    ok: integrity.status !== "failed" && reviewAudit.review_chain.status !== "failed" && !cacheEvidenceFailed,
    task_id: taskId,
    run_id: located.run.id,
    present: Boolean(board || budgetPayload || contract || reviewAudit.artifact_names.length || candidateArtifact),
    search_mode: searchMode,
    board_hash: integrity.status === "verified" ? stringValue(board?.board_hash, 64) || null : null,
    integrity,
    lineage,
    cards,
    selection_traces: selectionTraces,
    retrievals,
    budget: projectBudget(budgetPayload),
    operator_counts: operatorCounts,
    cache: cacheSummary,
    artifacts,
    candidate_freeze: candidateFreeze,
    independent_review: reviewAudit.independent_review,
    claim_audit: reviewAudit.claim_audit,
    review_chain: reviewAudit.review_chain,
    paired_ab: projectPairedAb(pairedArtifact),
    claim_boundary: "Cards, selection utilities and budgets are public-validation-only. Paired A/B, when present, is labeled local MLE grading; no official competition result is inferred.",
    ...(integrity.status === "failed" || reviewAudit.review_chain.status === "failed" || cacheEvidenceFailed
      ? {
          error: cacheEvidenceFailed
            ? "Experience summary cache evidence mismatch failed closed."
            : "Experience evidence integrity verification failed closed."
        }
      : {})
  };
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
  const effectiveInput = taskId === "evomind_demo_customer_churn"
    ? {
        ...input,
        engine: "research_os",
        runner: "local",
        iterations: 8,
        mcgs: true,
        search_mode: "experience_mcgs_v1",
        max_nodes: 8,
        max_tokens: 100_000,
        max_wall_seconds: 600,
        max_cost: 0.01
      }
    : input;
  const boundInput = { ...effectiveInput, task_id: taskId, official_submit_allowed: false };

  // 1) Planning and approval are two distinct, hash-bound requests. A plan is
  //    generated once, persisted as a one-time receipt, and the exact receipt
  //    must be consumed before any training process can launch.
  if (!approved) {
    const plan = await planEvolution(boundInput);
    const approvalPlan = await createEvolutionApprovalPlan(taskId, boundInput, plan);
    await logAction({
      action: "evolution_cycle_awaiting_approval",
      taskId,
      message: `Evolution cycle planned ${String(plan.selected_branch ?? "EXP000")} (${String(plan.code_generation_mode ?? "Base")}/${String(plan.expansion_type ?? "primary")}); awaiting approval to launch real training.`,
      artifactPath: typeof plan.plan_path === "string" ? plan.plan_path : null,
      metadata: {
        stage: "awaiting_approval",
        plan_id: approvalPlan.plan_id,
        plan_sha256: approvalPlan.plan_sha256,
        request_fingerprint: approvalPlan.request_fingerprint,
        expires_at: approvalPlan.expires_at,
        official_submit_allowed: false
      }
    });
    return {
      ok: true,
      task_id: taskId,
      stage: "awaiting_approval",
      approved: false,
      plan,
      plan_id: approvalPlan.plan_id,
      plan_sha256: approvalPlan.plan_sha256,
      request_fingerprint: approvalPlan.request_fingerprint,
      approval_expires_at: approvalPlan.expires_at,
      next_action: "Approve this exact hash-bound plan to launch real training and ingest the score.",
      official_submit_allowed: false,
      claim_boundary: plan.claim_boundary
    };
  }

  const approvalPlan = await consumeEvolutionApprovalPlan(taskId, boundInput);
  const plan = approvalPlan.plan;
  await logAction({
    action: "evolution_cycle_plan_approved",
    taskId,
    message: `Evolution plan ${approvalPlan.plan_id} approved with verified request and plan hashes.`,
    artifactPath: null,
    metadata: {
      stage: "approved",
      plan_id: approvalPlan.plan_id,
      plan_sha256: approvalPlan.plan_sha256,
      request_fingerprint: approvalPlan.request_fingerprint,
      approved_at: approvalPlan.approved_at,
      official_submit_allowed: false
    }
  });

  // Engine switch: "research_os" drives the corrected engine A (EvolutionLoop);
  // "legacy" (default) keeps the original mlevolve_search path as a fallback.
  const engine = effectiveInput.engine === "research_os" ? "research_os" : "legacy";
  const runner: "gpu" | "local_gpu" | "local" = effectiveInput.runner === "local_gpu"
    ? "local_gpu"
    : effectiveInput.runner === "local"
      ? "local"
      : "gpu";
  const searchMode: EvolutionSearchMode = effectiveInput.search_mode === "experience_mcgs_v1"
    ? "experience_mcgs_v1"
    : "legacy_uct";
  const boundedNumber = (value: unknown, fallback: number, minimum: number, maximum: number, label: string, integer = true) => {
    if (value == null || value === "") return fallback;
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
    const normalized = integer ? Math.trunc(value) : value;
    if (normalized < minimum || normalized > maximum) throw new Error(`${label} must be between ${minimum} and ${maximum}.`);
    return normalized;
  };
  const iterations = boundedNumber(effectiveInput.iterations, 8, 1, 64, "iterations");
  const maxNodes = boundedNumber(effectiveInput.max_nodes, iterations, 1, 64, "max_nodes");
  const maxTokens = boundedNumber(effectiveInput.max_tokens, 2_000_000, 1, 10_000_000, "max_tokens");
  const maxWallSeconds = boundedNumber(effectiveInput.max_wall_seconds, 5_400, 1, 43_200, "max_wall_seconds", false);
  const maxCost = effectiveInput.max_cost == null
    ? null
    : boundedNumber(effectiveInput.max_cost, 0, 0, 100_000, "max_cost", false);
  if (searchMode === "experience_mcgs_v1" && effectiveInput.mcgs === false) {
    throw new Error("experience_mcgs_v1 requires mcgs=true.");
  }

  // 3) Approved: launch REAL training through the workstation orchestrator.
  await logAction({
    action: "evolution_cycle_training_launch",
    taskId,
    message: `Evolution cycle approved: launching real ${engine === "research_os" ? `research_os (${runner}/${searchMode})` : "MCGS"} training for ${taskId}.`,
    metadata: {
      stage: "training", engine, runner, search_mode: searchMode,
      max_nodes: maxNodes, max_tokens: maxTokens, max_wall_seconds: maxWallSeconds, max_cost: maxCost,
      official_submit_allowed: false
    }
  });

  let training: Awaited<ReturnType<typeof runMCGSExperiment>> | Awaited<ReturnType<typeof runEvolutionEngineExperiment>>;
  try {
    training = engine === "research_os"
      ? await runEvolutionEngineExperiment(taskId, {
          runner,
          iterations,
          mcgs: effectiveInput.mcgs !== false,
          searchMode,
          maxNodes,
          maxTokens,
          maxWallSeconds,
          maxCost
        })
      : await runMCGSExperiment(taskId, {
          budgetNodes: typeof effectiveInput.budget_nodes === "number" ? effectiveInput.budget_nodes : undefined,
          fast: effectiveInput.fast === true
        });
  } catch (error) {
    const message = error instanceof Error ? error.message : "training failed";
    // Ingest the FAILURE so the graph/memory learn from it (run_success=false ->
    // gate holds, never promotes). This keeps negative results in the loop.
    const ingest = await ingestEvolutionResult({
      ...effectiveInput, task_id: taskId, run_id: "", cv_score: null, run_success: false,
      method: plan.code_generation_mode, expansion_type: plan.expansion_type
    }).catch(() => null);
    return {
      ok: false, task_id: taskId, stage: "training_failed", approved: true,
      error: message, plan, ingest,
      plan_id: approvalPlan.plan_id, plan_sha256: approvalPlan.plan_sha256,
      request_fingerprint: approvalPlan.request_fingerprint,
      official_submit_allowed: false,
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
      plan_id: approvalPlan.plan_id, plan_sha256: approvalPlan.plan_sha256,
      request_fingerprint: approvalPlan.request_fingerprint,
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
      ...effectiveInput,
      task_id: taskId,
      exp_dir: expDir,
      method: plan.code_generation_mode,
      expansion_type: plan.expansion_type
    });
  } else {
    const runId = typeof searchResult.best_run_id === "string" ? searchResult.best_run_id : "";
    ingest = await ingestEvolutionResult({
      ...effectiveInput,
      task_id: taskId,
      run_id: runId,
      cv_score: cvScore,
      run_success: cvScore !== null,
      method: plan.code_generation_mode,
      expansion_type: plan.expansion_type
    });
  }

  const expDir = typeof searchResult.exp_dir === "string" ? searchResult.exp_dir : "";
  let cycleEvidence: Awaited<ReturnType<typeof writeDemoCycleReceipts>> | null = null;
  if (taskId === "evomind_demo_customer_churn" && engine === "research_os") {
    try {
      cycleEvidence = await writeDemoCycleReceipts(taskId, training.run_id, expDir, approvalPlan);
    } catch (error) {
      try {
        const failedRun = await prisma.experimentRun.update({
          where: { id: training.run_id },
          data: { validationStatus: "failed" },
          select: { id: true, validationStatus: true },
        });
        if (failedRun.id !== training.run_id || failedRun.validationStatus !== "failed") {
          throw new Error("Demo run validation fail-closed marker did not persist.");
        }
      } catch (markerError) {
        throw new AggregateError(
          [error, markerError],
          "Demo cycle evidence failed and its database validation marker could not be persisted.",
        );
      }
      throw error;
    }
  }

  return {
    ok: true,
    task_id: taskId,
    stage: "completed",
    approved: true,
    plan,
    plan_id: approvalPlan.plan_id,
    plan_sha256: approvalPlan.plan_sha256,
    request_fingerprint: approvalPlan.request_fingerprint,
    training: {
      run_id: training.run_id,
      best_score: training.best_score,
      nodes_evaluated: training.nodes_evaluated,
      search_mode: searchMode,
      observability: Object.keys(recordOf(searchResult.observability)).length
        ? recordOf(searchResult.observability) as Record<string, string | null>
        : null,
      resource_gate: searchResult.resource_gate ?? null,
      independent_review: searchResult.independent_review ?? null
    },
    ingest,
    cycle_evidence: cycleEvidence,
    best_so_far: ingest.best_so_far,
    official_submit_allowed: false,
    claim_boundary: ingest.claim_boundary
  };
}
