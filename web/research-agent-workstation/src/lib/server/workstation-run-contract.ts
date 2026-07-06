import { createHash } from "node:crypto";
import { createReadStream, promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { encodeJson } from "@/lib/server/json";
import { normalizeTaskId, resolveWorkspacePath, stamp, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";

export type WorkstationArtifactDescriptor = {
  artifact_type: string;
  created_by_agent: string;
  stage: string;
  path: string;
  sha256: string | null;
  claim_binding: string | null;
  gate_dependency: string | null;
};

type CreateRunInput = {
  taskId: string;
  trigger?: string;
  configPath?: string;
  competitionSlug?: string;
  objective?: string;
};

const agentProtocol = [
  ["orchestrator", "总控 Agent", "task_decomposition", "workflow_plan.json"],
  ["research_context", "研究背景 Agent", "literature_context", "research_brief.md"],
  ["data_audit", "数据审计 Agent", "data_audit", "data_audit.json"],
  ["feature_engineering", "特征工程 Agent", "feature_engineering", "feature_plan.json"],
  ["model_selection", "模型选择 Agent", "model_selection", "experiment_matrix.json"],
  ["code_implementation", "代码实现 Agent", "code_generation", "code_patch.diff"],
  ["hpc_execution", "HPC/GPU 执行 Agent", "hpc_execution", "remote_job_manifest.json"],
  ["validation_analysis", "验证分析 Agent", "validation_review", "metrics_review.json"],
  ["submission_gate", "提交门禁 Agent", "submission_check", "submission_audit.json"],
  ["report_writer", "报告总结 Agent", "report_generation", "research_report.md"],
  ["reflection_reviewer", "反思审查 Agent", "reflection", "reflection_review.md"]
] as const;

const gateTypes = [
  {
    gate_type: "plan_approval",
    stage: "human_plan_gate",
    reason: "工作站任务规划必须先由人工确认，禁止 Codex 旁路直接训练。"
  },
  {
    gate_type: "code_quality_approval",
    stage: "code_review",
    reason: "Code Agent 只能生成草稿/diff，应用前必须通过代码质量与人工 Gate。"
  },
  {
    gate_type: "hpc_execution_approval",
    stage: "hpc_execution",
    reason: "HPC/GPU 重训练或远程作业必须由工作站生成 job manifest 并经人工批准。"
  },
  {
    gate_type: "submission_approval",
    stage: "human_submission_gate",
    reason: "Kaggle 官方提交必须由人工明确批准，默认阻断。"
  },
  {
    gate_type: "final_report_approval",
    stage: "human_final_gate",
    reason: "教师汇报与最终科研结论必须绑定证据后人工确认。"
  }
] as const;

async function sha256File(relativePath: string) {
  const absolutePath = resolveWorkspacePath(relativePath);
  try {
    await fs.access(absolutePath);
  } catch {
    return null;
  }
  return new Promise<string>((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(absolutePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

export async function artifactDescriptor(
  relativePath: string,
  metadata: Omit<WorkstationArtifactDescriptor, "path" | "sha256">
): Promise<WorkstationArtifactDescriptor> {
  return {
    ...metadata,
    path: relativePath,
    sha256: await sha256File(relativePath)
  };
}

export async function writeArtifactManifestArtifact(input: {
  taskId: string;
  runId: string;
  relativePath: string;
  artifacts: WorkstationArtifactDescriptor[];
  source: string;
  extra?: Record<string, unknown>;
}) {
  return writeJsonArtifact(input.relativePath, {
    schema: "academic_research_os.artifact_manifest.v1",
    task_id: input.taskId,
    workstation_run_id: input.runId,
    source: input.source,
    artifacts: input.artifacts,
    generated_at: new Date().toISOString(),
    ...input.extra
  });
}

function workflowNodes(runId: string) {
  return agentProtocol.map(([id, label, stage, expectedArtifact], index) => ({
    id,
    label,
    stage,
    expected_artifact: expectedArtifact,
    status: index === 0 ? "ready" : "waiting",
    workstation_run_id: runId
  }));
}

function workflowEdges() {
  return agentProtocol.slice(1).map(([id], index) => ({
    source: agentProtocol[index][0],
    target: id,
    handoff_contract: "bounded_artifact_context"
  }));
}

export async function createWorkstationRun(input: CreateRunInput) {
  const taskId = normalizeTaskId(input.taskId);
  const runId = `wr_${stamp()}_${Math.random().toString(36).slice(2, 7)}`;
  const runRoot = `workspace/workstation_runs/${taskId}/${runId}`;
  const configPath = input.configPath ?? (taskId === "playground_series_s6e6" ? "configs/generated/playground_series_s6e6.yaml" : `configs/${taskId}.yaml`);
  const objective = input.objective ?? "Use the AI research workstation as the orchestrator for an evidence-bound Kaggle/HPC workflow.";
  const createdAt = new Date();

  await prisma.task.upsert({
    where: { id: taskId },
    update: {
      status: "workstation_ready",
      configPath,
      owner: "Orchestrator Agent"
    },
    create: {
      id: taskId,
      name: taskId === "playground_series_s6e6" ? "Playground Series S6E6 Stellar Class" : taskId.replaceAll("_", " "),
      taskType: taskId === "playground_series_s6e6" ? "kaggle_classification" : "research_task",
      target: taskId === "playground_series_s6e6" ? "class" : null,
      metric: taskId === "playground_series_s6e6" ? "balanced_accuracy" : null,
      status: "workstation_ready",
      priority: "High",
      owner: "Orchestrator Agent",
      configPath,
      taskDir: `tasks/${taskId}`
    }
  });

  await prisma.experimentRun.create({
    data: {
      id: runId,
      taskId,
      outputDir: runRoot,
      status: "workstation_planned",
      bestModel: null,
      metricsJson: encodeJson({
        workstation_run: true,
        direct_training_allowed: false,
        official_submission_allowed: false,
        objective
      }),
      validationStatus: "pending",
      startedAt: createdAt
    }
  });

  const nodes = workflowNodes(runId);
  const edges = workflowEdges();
  await prisma.workflow.upsert({
    where: { id: `${runId}_workflow` },
    update: {
      status: "draft",
      nodesJson: encodeJson(nodes) ?? "[]",
      edgesJson: encodeJson(edges) ?? "[]"
    },
    create: {
      id: `${runId}_workflow`,
      taskId,
      name: `${taskId} Workstation Agent Workflow`,
      status: "draft",
      nodesJson: encodeJson(nodes) ?? "[]",
      edgesJson: encodeJson(edges) ?? "[]"
    }
  });

  for (const gate of gateTypes) {
    await prisma.gate.create({
      data: {
        id: `${runId}_${gate.gate_type}`,
        taskId,
        runId,
        gateType: gate.gate_type,
        decision: "pending",
        reviewer: "Research Admin",
        evidenceJson: encodeJson({
          workstation_run_id: runId,
          stage: gate.stage,
          reason: gate.reason,
          required_artifacts: ["workstation_run_manifest.json", "artifact_manifest.json"]
        })
      }
    });
  }

  const contractPath = `${runRoot}/agent_protocol.json`;
  await writeJsonArtifact(contractPath, {
    schema: "academic_research_os.agent_protocol.v1",
    task_id: taskId,
    workstation_run_id: runId,
    agents: agentProtocol.map(([id, label, stage, expectedArtifact]) => ({
      id,
      label,
      stage,
      input_policy: "bounded_context_only",
      output_artifact: expectedArtifact,
      failure_return: `return_to_${stage}`
    })),
    gates: gateTypes,
    direct_codex_training_allowed: false,
    official_kaggle_submission_default: "blocked"
  });

  const manifestPath = `${runRoot}/workstation_run_manifest.json`;
  await writeJsonArtifact(manifestPath, {
    schema: "academic_research_os.workstation_run.v1",
    task_id: taskId,
    workstation_run_id: runId,
    trigger: input.trigger ?? "frontend_or_api",
    objective,
    config_path: configPath,
    competition_slug: input.competitionSlug ?? null,
    state: "planned",
    launched_by: "workstation_orchestrator",
    direct_codex_training_allowed: false,
    official_submission_requires_human_gate: true,
    workflow: { nodes, edges },
    gates: gateTypes.map((gate) => ({ ...gate, decision: "pending" })),
    created_at: createdAt.toISOString()
  });

  const artifacts = [
    await artifactDescriptor(contractPath, {
      artifact_type: "agent_protocol",
      created_by_agent: "OrchestratorAgent",
      stage: "experiment_planning",
      claim_binding: "Multi-agent boundaries are explicit and auditable.",
      gate_dependency: "plan_approval"
    }),
    await artifactDescriptor(manifestPath, {
      artifact_type: "workstation_run_manifest",
      created_by_agent: "OrchestratorAgent",
      stage: "task_understanding",
      claim_binding: "The run is initiated by the workstation rather than a side-channel training command.",
      gate_dependency: "plan_approval"
    })
  ];
  const artifactManifestPath = await writeArtifactManifestArtifact({
    taskId,
    runId,
    relativePath: `${runRoot}/artifact_manifest.json`,
    artifacts,
    source: "workstation_run_contract"
  });

  await prisma.evidence.createMany({
    data: [
      {
        id: `${runId}_manifest_evidence`,
        taskId,
        runId,
        label: "Workstation run manifest",
        artifactPath: manifestPath,
        hash: artifacts[1].sha256,
        source: "OrchestratorAgent",
        claimBinding: "工作站是本次任务的执行主体。"
      },
      {
        id: `${runId}_artifact_manifest_evidence`,
        taskId,
        runId,
        label: "Artifact manifest",
        artifactPath: artifactManifestPath,
        source: "ArtifactRegistry",
        claimBinding: "每个阶段产物有统一证据字段。"
      }
    ]
  });

  const action = await logAction({
    action: "create_workstation_run",
    taskId,
    runId,
    message: `Workstation run created: ${runId}. Training and official submission remain gated.`,
    artifactPath: manifestPath,
    metadata: { workstation_run_id: runId, artifact_manifest: artifactManifestPath, config_path: configPath }
  });

  return { ok: true, ...action, run_id: runId, workflow_id: `${runId}_workflow`, manifest_path: manifestPath, artifact_manifest_path: artifactManifestPath };
}

export async function ensurePlaygroundSeriesTask() {
  const taskId = "playground_series_s6e6";
  const configPath = "configs/generated/playground_series_s6e6.yaml";
  const dataFiles = [
    "tasks/playground_series_s6e6/data/train.csv",
    "tasks/playground_series_s6e6/data/test.csv",
    "tasks/playground_series_s6e6/data/sample_submission.csv"
  ];
  const dataArtifacts = await Promise.all(dataFiles.map(async (file) => ({
    path: file,
    present: await fs.access(resolveWorkspacePath(file)).then(() => true).catch(() => false),
    sha256: await sha256File(file)
  })));

  await prisma.task.upsert({
    where: { id: taskId },
    update: {
      name: "Playground Series S6E6 Stellar Class",
      taskType: "kaggle_classification",
      target: "class",
      metric: "balanced_accuracy",
      status: "workstation_ready",
      priority: "High",
      owner: "Orchestrator Agent",
      configPath,
      taskDir: `tasks/${taskId}`
    },
    create: {
      id: taskId,
      name: "Playground Series S6E6 Stellar Class",
      taskType: "kaggle_classification",
      target: "class",
      metric: "balanced_accuracy",
      status: "workstation_ready",
      priority: "High",
      owner: "Orchestrator Agent",
      configPath,
      taskDir: `tasks/${taskId}`
    }
  });

  const readinessPath = await writeJsonArtifact("workspace/kaggle_onboarding/playground_series_s6e6_workstation_readiness.json", {
    schema: "academic_research_os.kaggle_task_readiness.v1",
    task_id: taskId,
    competition_slug: "playground-series-s6e6",
    config_path: configPath,
    target: "class",
    metric: "balanced_accuracy",
    dataset_artifacts: dataArtifacts,
    leaderboard_boundary: "Official Kaggle submission is blocked until submission_approval gate is approved.",
    direct_codex_training_allowed: false,
    generated_at: new Date().toISOString()
  });

  await prisma.evidence.upsert({
    where: { id: `${taskId}_readiness_evidence` },
    update: { artifactPath: readinessPath, source: "DataAuditAgent", claimBinding: "playground-series-s6e6 is onboarded as a workstation task." },
    create: {
      id: `${taskId}_readiness_evidence`,
      taskId,
      label: "Playground Series S6E6 workstation readiness",
      artifactPath: readinessPath,
      source: "DataAuditAgent",
      claimBinding: "playground-series-s6e6 已作为工作站任务接入。"
    }
  });

  const action = await logAction({
    action: "onboard_playground_s6e6",
    taskId,
    message: "Playground Series S6E6 is onboarded as a workstation-controlled validation task.",
    artifactPath: readinessPath,
    metadata: { data_files_present: dataArtifacts.every((item) => item.present), config_path: configPath }
  });
  return { ok: true, ...action, task_id: taskId, config_path: configPath, readiness_path: readinessPath };
}

async function latestPassedCodeQualityGate(taskId: string) {
  const patchDir = resolveWorkspacePath(`workspace/tasks/${taskId}/code/patches`);
  const entries = await fs.readdir(patchDir, { withFileTypes: true }).catch(() => []);
  const checks = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.startsWith("code_quality_check_") && entry.name.endsWith(".json"))
    .map(async (entry) => {
      const relativePath = `workspace/tasks/${taskId}/code/patches/${entry.name}`;
      const absolutePath = resolveWorkspacePath(relativePath);
      const stat = await fs.stat(absolutePath).catch(() => null);
      const payload = JSON.parse(await fs.readFile(absolutePath, "utf-8").catch(() => "{}")) as Record<string, unknown>;
      return { relativePath, payload, mtimeMs: stat?.mtimeMs ?? 0 };
    }));
  const latest = checks
    .filter((check) => check.payload.overall_status === "passed")
    .sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  if (!latest) return null;
  return {
    quality_gate_path: latest.relativePath,
    patch_path: typeof latest.payload.patch_path === "string" ? latest.payload.patch_path : null,
    affected_files: Array.isArray(latest.payload.affected_files) ? latest.payload.affected_files : [],
    patch_python_syntax_check: latest.payload.patch_python_syntax_check ?? null
  };
}

export async function createHpcExecutionGate(input: { taskId: string; runId?: string; template?: string }) {
  const taskId = normalizeTaskId(input.taskId);
  const run = input.runId
    ? await prisma.experimentRun.findUnique({ where: { id: input.runId } })
    : await prisma.experimentRun.findFirst({ where: { taskId }, orderBy: { createdAt: "desc" } });
  const runId = run?.id ?? (await createWorkstationRun({ taskId, trigger: "hpc_gate_preparation" })).run_id;
  const gateId = `${runId}_hpc_execution_approval`;
  const latestCodeGate = await latestPassedCodeQualityGate(taskId);
  const manifestPath = await writeJsonArtifact(`workspace/workstation_runs/${taskId}/${runId}/hpc_execution_gate_manifest.json`, {
    schema: "academic_research_os.hpc_execution_gate.v1",
    task_id: taskId,
    workstation_run_id: runId,
    requested_template: input.template ?? "connection_smoke",
    status: "pending_approval",
    resource_policy: "whitelist_templates_only",
    remote_training_allowed: false,
    code_agent_dependency: latestCodeGate ? {
      status: "code_quality_passed",
      ...latestCodeGate,
      required_before_training: true
    } : {
      status: "missing_passed_code_quality_gate",
      required_before_training: true
    },
    approval_required_before: "POST /api/gpu/jobs",
    generated_at: new Date().toISOString()
  });

  await prisma.gate.upsert({
    where: { id: gateId },
    update: {
      decision: "pending",
      evidenceJson: encodeJson({ manifest_path: manifestPath, requested_template: input.template ?? "connection_smoke", code_agent_dependency: latestCodeGate })
    },
    create: {
      id: gateId,
      taskId,
      runId,
      gateType: "hpc_execution_approval",
      decision: "pending",
      reviewer: "Research Admin",
      evidenceJson: encodeJson({ manifest_path: manifestPath, requested_template: input.template ?? "connection_smoke", code_agent_dependency: latestCodeGate })
    }
  });

  const action = await logAction({
    action: "prepare_hpc_execution_gate",
    taskId,
    runId,
    message: "HPC execution gate prepared. Remote job remains blocked until approval.",
    artifactPath: manifestPath,
    metadata: { gate_id: gateId, template: input.template ?? "connection_smoke", code_agent_dependency: latestCodeGate }
  });
  return { ok: true, ...action, run_id: runId, gate_id: gateId, manifest_path: manifestPath };
}

export async function generateTeacherEvidenceBundle(taskIdInput: string) {
  const taskId = normalizeTaskId(taskIdInput);
  const [task, runs, gates, evidence, workflows, actions] = await Promise.all([
    prisma.task.findUnique({ where: { id: taskId } }),
    prisma.experimentRun.findMany({ where: { taskId }, orderBy: { createdAt: "desc" }, take: 10 }),
    prisma.gate.findMany({ where: { taskId }, orderBy: { createdAt: "desc" }, take: 20 }),
    prisma.evidence.findMany({ where: { taskId }, orderBy: { createdAt: "desc" }, take: 30 }),
    prisma.workflow.findMany({ where: { taskId }, orderBy: { updatedAt: "desc" }, take: 5 }),
    prisma.actionLog.findMany({ where: { taskId }, orderBy: { createdAt: "desc" }, take: 30 })
  ]);
  const bundleRoot = `workspace/teacher_evidence/${taskId}_${stamp()}`;
  const jsonPath = await writeJsonArtifact(`${bundleRoot}/teacher_evidence_bundle.json`, {
    schema: "academic_research_os.teacher_evidence_bundle.v1",
    task,
    workstation_runs: runs,
    gates,
    evidence,
    workflows,
    actions,
    boundaries: {
      direct_codex_training_allowed: false,
      official_kaggle_submission_default: "blocked",
      credentials_policy: "DPAPI/env/secret-file only; no plaintext secrets in repository artifacts."
    },
    generated_at: new Date().toISOString()
  });
  const markdown = [
    `# ${taskId} 教师汇报证据包`,
    "",
    "## 任务定位",
    `- 工作站任务：${task?.name ?? taskId}`,
    `- 配置文件：${task?.configPath ?? "未记录"}`,
    "- 执行主体：AI 科研工作站多 Agent 编排，不是 Codex 旁路直接训练。",
    "",
    "## 工作站 Run",
    ...runs.map((run) => `- ${run.id}: ${run.status}; output=${run.outputDir ?? "pending"}; validation=${run.validationStatus ?? "pending"}`),
    "",
    "## Gate 状态",
    ...gates.map((gate) => `- ${gate.gateType}: ${gate.decision}; reviewer=${gate.reviewer ?? "pending"}`),
    "",
    "## 证据链",
    ...evidence.slice(0, 12).map((item) => `- ${item.label}: ${item.artifactPath ?? "pending"}; claim=${item.claimBinding ?? "pending"}`),
    "",
    "## 安全边界",
    "- Kaggle 官方提交默认阻断，必须通过 submission_approval。",
    "- HPC/GPU 作业默认只允许白名单模板或工作站生成的 job manifest。",
    "- DeepSeek/Code Agent 只能生成草稿、diff、transcript、manifest，应用前必须过 Gate。",
    "- 凭据只能通过 DPAPI/env/secret-file 加载，不写入报告或仓库。",
    "",
    `JSON 证据：${jsonPath}`
  ].join("\n");
  const markdownPath = await writeTextArtifact(`${bundleRoot}/teacher_evidence_bundle.md`, markdown);
  const action = await logAction({
    action: "generate_teacher_evidence_bundle",
    taskId,
    message: "Teacher-facing evidence bundle generated.",
    artifactPath: markdownPath,
    metadata: { json_path: jsonPath }
  });
  return { ok: true, ...action, markdown_path: markdownPath, json_path: jsonPath };
}

// ── GPU/HPC Job Manifest Validation ─────────────────────────────────────────

const REQUIRED_JOB_MANIFEST_FIELDS = [
  "task_id",
  "workstation_run_id",
  "agent_id",
  "gate_id",
  "resource_request",
  "remote_workspace",
  "command_template",
  "log_path",
  "pullback_policy",
  "timeout_seconds",
] as const;

export interface GpuJobManifest {
  schema: string;
  job_id: string;
  task_id: string;
  workstation_run_id: string | null;
  agent_id: string;
  gate_id: string | null;
  command_template: string;
  resource_request: Record<string, unknown>;
  remote_workspace: string;
  local_artifact_root?: string;
  log_path: string;
  pullback_policy: string;
  timeout_seconds: number;
  cancel_record_path?: string;
  status: string;
  created_at: string;
}

export function validateGpuJobManifest(manifest: Record<string, unknown>): {
  valid: boolean;
  missing: string[];
  warnings: string[];
} {
  const missing: string[] = [];
  const warnings: string[] = [];

  for (const field of REQUIRED_JOB_MANIFEST_FIELDS) {
    if (manifest[field] === undefined || manifest[field] === null) {
      missing.push(field);
    }
  }

  if (manifest.schema !== "academic_research_os.gpu_job_manifest.v1") {
    warnings.push(`Expected schema academic_research_os.gpu_job_manifest.v1, got ${String(manifest.schema)}`);
  }

  const timeout = Number(manifest.timeout_seconds);
  if (timeout > 0 && timeout > 43200) {
    warnings.push(`timeout_seconds ${timeout} exceeds 12-hour maximum; consider splitting the job.`);
  }

  if (typeof manifest.resource_request === "object" && manifest.resource_request !== null) {
    const rr = manifest.resource_request as Record<string, unknown>;
    if (!rr.gpu && !rr.mode) {
      warnings.push("resource_request should specify gpu or mode.");
    }
  }

  return { valid: missing.length === 0, missing, warnings };
}

export function validateArtifactManifest(artifact: Record<string, unknown>): {
  valid: boolean;
  missing: string[];
} {
  const required = ["artifact_type", "created_by_agent", "stage", "path", "claim_binding"];
  const missing = required.filter((f) => artifact[f] === undefined || artifact[f] === null);
  if (artifact.sha256 === undefined || artifact.sha256 === null) {
    missing.push("sha256");
  }
  return { valid: missing.length === 0, missing };
}
