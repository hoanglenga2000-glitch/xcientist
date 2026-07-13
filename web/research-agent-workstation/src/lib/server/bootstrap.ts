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

export async function ensureWorkstationSeeded() {
  const taskCount = await prisma.task.count();
  const summary = await readJsonFile(summaryPath);

  for (const task of baselineTasks) {
    await prisma.task.upsert({
      where: { id: task.id },
      update: taskCount === 0 ? task : {},
      create: task
    });
  }

  const connectorStatus = summary?.connector_status ?? {};
  for (const [provider, value] of Object.entries(connectorStatus)) {
    if (provider === "env_keys" || typeof value !== "object" || value === null) continue;
    const item = value as { name?: string; state?: string; configured?: boolean; notes?: string };
    await prisma.connectorStatus.upsert({
      where: { provider },
      update: {
        name: item.name ?? provider,
        state: item.state ?? "Unknown",
        configured: Boolean(item.configured),
        detail: item.notes ?? null
      },
      create: {
        provider,
        name: item.name ?? provider,
        state: item.state ?? "Unknown",
        configured: Boolean(item.configured),
        detail: item.notes ?? null
      }
    });
  }

  const defaultSettings = [
    ["general", { workspace_name: "AI Data Scientist Lab", default_language: "zh-CN", theme: "light", default_mission: "house_prices" }],
    ["language", { ui_language: "zh-CN", report_language: "zh-CN", agent_output_language: "zh-CN" }],
    ["database", { provider: "sqlite", path: "web/research-agent-workstation/prisma/workstation.db", migration_status: "synced" }],
    ["code_agent", { provider: "claude_agent_sdk", default_agent: "claude_code", model: "sonnet", max_turns: 2, timeout_seconds: 120, workspace_scope: "read_only_context_plus_gated_patch", api_key_status: "not_configured", export_context_path: "workspace/tasks/{task_id}/code_agent_context", import_patch_path: "workspace/tasks/{task_id}/code/patches", enable_patch_review_gate: true }],
    ["runner", { provider: "local_python", working_directory: "workspace", timeout_seconds: 1800, max_runtime_minutes: 60 }],
    ["compute", { execution_mode: "hpc_gpu", local_training_enabled: false, local_training_scope: "disabled", selected_at: null, selected_by: "system_default", note: "HPC/GPU is required. Local training APIs fail closed when the remote runtime is unavailable." }],
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
      ...(existingValue ?? {})
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
    await prisma.setting.upsert({
      where: { key },
      update: existing ? { valueJson } : taskCount === 0 ? { valueJson } : {},
      create: { key, valueJson }
    });
  }

  for (const run of summary?.runs ?? []) {
    const taskId = run.task_id === "house-prices" ? "house_prices" : run.task_id;
    if (!taskId || !baselineTasks.some((task) => task.id === taskId)) continue;
    if (run.output_dir) {
      const existingByOutput = await prisma.experimentRun.findFirst({ where: { outputDir: run.output_dir } });
      if (existingByOutput) continue;
    }
    const id = `${taskId}_${String(run.output_dir ?? "seed").replace(/[^a-zA-Z0-9]+/g, "_")}`;
    await prisma.experimentRun.upsert({
      where: { id },
      update: {},
      create: {
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

    if (run.validation_gate) {
      await prisma.gate.upsert({
        where: { id: `${id}_validation_gate` },
        update: {},
        create: {
          id: `${id}_validation_gate`,
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
