import path from "node:path";
import { prisma } from "@/lib/db";
import { decodeJson, encodeJson } from "@/lib/server/json";
import { readJsonFile, workspaceRoot } from "@/lib/server/paths";

const summaryPath = path.join(workspaceRoot, "workspace", "workstation_summary.json");

const baselineTasks = [
  {
    id: "house_prices",
    name: "House Prices Regression",
    taskType: "tabular_regression",
    target: "SalePrice",
    metric: "RMSLE",
    status: "running",
    priority: "High",
    owner: "Developer Agent",
    configPath: "configs/house_prices.yaml",
    taskDir: "workspace/tasks/house_prices"
  },
  {
    id: "titanic",
    name: "Titanic Survival",
    taskType: "tabular_classification",
    target: "Survived",
    metric: "Accuracy",
    status: "review",
    priority: "Medium",
    owner: "Analyst Agent",
    configPath: "configs/titanic.yaml",
    taskDir: "workspace/tasks/titanic"
  }
];

let seedPromise: Promise<void> | null = null;
let seedCompleted = false;

async function configureSqliteRuntime() {
  const busyTimeoutMs = Math.min(60_000, Math.max(1_000, Number(process.env.WORKSTATION_SQLITE_BUSY_TIMEOUT_MS ?? 5_000)));
  await prisma.$queryRawUnsafe(`PRAGMA busy_timeout = ${busyTimeoutMs}`);
  if (process.env.WORKSTATION_SQLITE_WAL !== "0") {
    await prisma.$queryRawUnsafe("PRAGMA journal_mode = WAL");
    await prisma.$queryRawUnsafe("PRAGMA synchronous = NORMAL");
  }
}

async function seedWorkstationOnce() {
  await configureSqliteRuntime();
  const summary = await readJsonFile(summaryPath);

  for (const task of baselineTasks) {
    const existing = await prisma.task.findUnique({ where: { id: task.id }, select: { id: true } });
    if (!existing) await prisma.task.create({ data: task });
  }

  const connectorStatus = summary?.connector_status ?? {};
  for (const [provider, value] of Object.entries(connectorStatus)) {
    if (provider === "env_keys" || typeof value !== "object" || value === null) continue;
    const item = value as { name?: string; state?: string; configured?: boolean; notes?: string };
    const next = {
      name: item.name ?? provider,
      state: item.state ?? "Unknown",
      configured: Boolean(item.configured),
      detail: item.notes ?? null
    };
    const existing = await prisma.connectorStatus.findUnique({ where: { provider } });
    if (!existing) {
      await prisma.connectorStatus.create({ data: { provider, ...next } });
    } else if (
      existing.name !== next.name
      || existing.state !== next.state
      || existing.configured !== next.configured
      || existing.detail !== next.detail
    ) {
      await prisma.connectorStatus.update({ where: { provider }, data: next });
    }
  }

  const defaultSettings = [
    ["general", { workspace_name: "AI Data Scientist Lab", default_language: "zh-CN", theme: "dark", default_mission: "house_prices" }],
    ["language", { ui_language: "zh-CN", report_language: "zh-CN", agent_output_language: "zh-CN" }],
    ["database", { provider: "sqlite", path: "web/research-agent-workstation/prisma/workstation.db", migration_status: "synced" }],
    ["code_agent", { provider: "claude_agent_sdk", default_agent: "claude_code", model: "sonnet", max_turns: 2, timeout_seconds: 120, workspace_scope: "read_only_context_plus_gated_patch", api_key_status: "not_configured", export_context_path: "workspace/tasks/{task_id}/code_agent_context", import_patch_path: "workspace/tasks/{task_id}/code/patches", enable_patch_review_gate: true }],
    ["runner", { provider: "local_python", working_directory: "workspace", timeout_seconds: 1800, max_runtime_minutes: 60 }],
    ["compute", { execution_mode: "hpc_gpu", local_training_enabled: false, local_training_scope: "small_tasks_only", selected_at: null, selected_by: "system_default", note: "HPC/GPU remains default. Local compute must be explicitly selected from the GPU page before local training APIs run." }],
    ["gpu", { provider: "ssh_gateway", host: "", port: "22", username: "", auth_method: "private_key_env_path", remote_workspace: "", status: "not_configured", long_training_requires_approval: true }],
    ["kaggle", { token_status: "not_configured", enable_download: false, enable_submit: false, submission_requires_human_gate: true }],
    ["llm", { provider: "rule_based", api_key_status: "hidden", test_provider: false }],
    ["storage", { provider: "local_workspace", artifact_path: "workspace", report_export_path: "workspace/tasks/{task_id}/reports/draft" }],
    ["audit", { enable_audit_log: true, enable_code_quality_gate: true, enable_evidence_binding_requirement: true, enable_final_report_approval: true, block_unsafe_commands: true }]
  ] as const;
  for (const [key, value] of defaultSettings) {
    const existing = await prisma.setting.findUnique({ where: { key } });
    const existingValue = decodeJson<Record<string, unknown>>(existing?.valueJson);
    const mergedValue = {
      ...(value as Record<string, unknown>),
      ...existingValue
    };
    if (key === "code_agent" && (mergedValue.provider === "local_template" || !mergedValue.provider)) {
      mergedValue.provider = "claude_agent_sdk";
      mergedValue.default_agent = "claude_code";
      mergedValue.api_key_status = "not_configured";
    }
    if (key === "gpu" && (mergedValue.provider === "mock" || !mergedValue.provider)) {
      mergedValue.provider = "ssh_gateway";
      mergedValue.status = "not_configured";
    }
    const valueJson = encodeJson(mergedValue) ?? "{}";
    if (!existing) {
      await prisma.setting.create({ data: { key, valueJson } });
    } else if (existing.valueJson !== valueJson) {
      await prisma.setting.update({ where: { key }, data: { valueJson } });
    }
  }

  for (const run of summary?.runs ?? []) {
    const taskId = run.task_id === "house-prices" ? "house_prices" : run.task_id;
    if (!taskId || !baselineTasks.some((task) => task.id === taskId)) continue;
    if (run.output_dir) {
      const existingByOutput = await prisma.experimentRun.findFirst({ where: { outputDir: run.output_dir } });
      if (existingByOutput) continue;
    }
    const id = `${taskId}_${String(run.output_dir ?? "seed").replace(/[^a-zA-Z0-9]+/g, "_")}`;
    const existingRun = await prisma.experimentRun.findUnique({ where: { id }, select: { id: true } });
    if (!existingRun) {
      await prisma.experimentRun.create({
        data: {
        id,
        taskId,
        outputDir: run.output_dir ?? null,
        status: run.validation_gate?.status === "passed" || run.accepted ? "passed" : "unknown",
        bestModel: run.best_model ?? null,
        metricsJson: encodeJson(run.best_metrics ?? null),
        validationStatus: run.validation_gate?.status ?? null,
        startedAt: null,
        finishedAt: new Date()
        }
      });
    }

    if (run.validation_gate) {
      const gateId = `${id}_validation_gate`;
      const existingGate = await prisma.gate.findUnique({ where: { id: gateId }, select: { id: true } });
      if (!existingGate) {
        await prisma.gate.create({
          data: {
          id: gateId,
          taskId,
          runId: id,
          gateType: "validation_gate",
          decision: run.validation_gate.status ?? "unknown",
          reviewer: "Local Validator",
          evidenceJson: encodeJson(run.validation_gate),
          decidedAt: new Date()
          }
        });
      }
    }
  }
}

/**
 * Initialize immutable workstation defaults once per server process.
 *
 * Route handlers may call this defensively, but after the first successful
 * initialization they perform no file reads and no database writes.  The seed
 * implementation itself is value-aware so separate Next.js route bundles also
 * avoid UPDATE statements when the stored values already match.
 */
export async function ensureWorkstationSeeded() {
  if (seedCompleted) return;
  if (!seedPromise) {
    seedPromise = seedWorkstationOnce()
      .then(() => {
        seedCompleted = true;
      })
      .catch((error) => {
        seedPromise = null;
        throw error;
      });
  }
  await seedPromise;
}
