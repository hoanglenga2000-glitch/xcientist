import path from "node:path";
import { promises as fs } from "node:fs";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { runManagedCommand } from "@/lib/server/job-registry";
import { normalizeTaskId, toRelativePath, workspaceRoot } from "@/lib/server/paths";
import { getWorkstationSummary } from "@/lib/server/summary";

type TaskRunConfig = { config: string; validator: string; validatorArgs: string[] };
type EnsembleRunOptions = { fast?: boolean; sampleRows?: number; nFolds?: number; seeds?: string; timeoutMs?: number };

const taskConfig: Record<string, TaskRunConfig> = {
  house_prices: {
    config: "configs/house_prices.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/house_prices.yaml"]
  },
  house_prices_advanced_regression_techniques: {
    config: "configs/house_prices.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/house_prices.yaml"]
  },
  titanic: {
    config: "configs/titanic.yaml",
    validator: "scripts/validate_titanic_experiment.py",
    validatorArgs: ["--config", "configs/titanic.yaml"]
  },
  telco_churn: {
    config: "configs/telco_churn.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/telco_churn.yaml"]
  },
  digit_recognizer: {
    config: "configs/digit_recognizer.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/digit_recognizer.yaml"]
  },
  playground_series_s6e6: {
    config: "configs/generated/playground_series_s6e6.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/generated/playground_series_s6e6.yaml"]
  },
  spaceship_titanic: {
    config: "configs/spaceship_titanic.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/spaceship_titanic.yaml"]
  },
  bike_sharing_demand: {
    config: "configs/bike_sharing_demand.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/bike_sharing_demand.yaml"]
  },
  porto_seguro_safe_driver_prediction: {
    config: "configs/porto_seguro_safe_driver_prediction.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/porto_seguro_safe_driver_prediction.yaml"]
  },
  store_sales_time_series_forecasting: {
    config: "configs/store_sales_time_series_forecasting.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/store_sales_time_series_forecasting.yaml"]
  },
  tabular_playground_series_aug_2022: {
    config: "configs/tabular_playground_series_aug_2022.yaml",
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", "configs/tabular_playground_series_aug_2022.yaml"]
  }
};

function pythonExecutable() {
  if (process.env.WORKSTATION_PYTHON) return process.env.WORKSTATION_PYTHON;
  if (process.platform !== "win32") return "python3";
  return "C:\\codex-python\\python.exe";
}

async function localTrainingFallbackDisabled() {
  return true;
}

async function blockedLocalTrainingResponse(action: string, taskId: string) {
  await logAction({
    action,
    taskId,
    message: "Local training fallback is disabled by workstation resource policy; use HPC/GPU gated execution.",
    metadata: {
      status: "blocked_local_training_disabled",
      required_route: "prepare_hpc_execution_gate_or_gpu_job_manifest"
    }
  });
  return {
    ok: false,
    task_id: taskId,
    status: "blocked_local_training_disabled",
    reason: "Local training fallback is disabled because the workstation must use gated HPC/GPU execution when available.",
    next_action: "Use a workstation GPU/HPC action such as run_s6e6_strategy_recommended or prepare_hpc_execution_gate."
  };
}

function safeGeneratedConfig(configPath: string | null | undefined) {
  if (!configPath) return null;
  const normalized = configPath.replaceAll("\\", "/");
  if (!normalized.startsWith("configs/generated/")) return null;
  if (!normalized.endsWith(".yaml") && !normalized.endsWith(".yml")) return null;
  if (normalized.includes("..")) return null;
  return normalized;
}

async function resolveTaskRunConfig(taskId: string): Promise<TaskRunConfig> {
  const staticConfig = taskConfig[taskId];
  if (staticConfig) return staticConfig;

  const task = await prisma.task.findUnique({ where: { id: taskId } });
  const generatedConfig = safeGeneratedConfig(task?.configPath);
  if (!generatedConfig) {
    throw new Error(`Unsupported task: ${taskId}. Create a runnable task or use house_prices/titanic.`);
  }

  const fullConfigPath = path.join(workspaceRoot, generatedConfig);
  const exists = await fs.stat(fullConfigPath).then((stat) => stat.isFile()).catch(() => false);
  if (!exists) throw new Error(`Generated task config is missing: ${generatedConfig}`);

  return {
    config: generatedConfig,
    validator: "scripts/validate_tabular_experiment.py",
    validatorArgs: ["--config", generatedConfig]
  };
}

async function latestExperimentDir(taskId: string) {
  const root = path.join(workspaceRoot, "experiments", taskId);
  const entries = await fs.readdir(root, { withFileTypes: true });
  const dirs = await Promise.all(entries
    .filter((entry) => entry.isDirectory())
    .map(async (entry) => {
      const fullPath = path.join(root, entry.name);
      const stat = await fs.stat(fullPath);
      return { fullPath, mtimeMs: stat.mtimeMs };
    }));
  if (!dirs.length) throw new Error(`No experiment directory found for ${taskId}`);
  return dirs.sort((a, b) => b.mtimeMs - a.mtimeMs)[0].fullPath;
}

export async function runLocalExperiment(taskIdInput: string) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(taskIdInput);
  if (await localTrainingFallbackDisabled()) {
    return blockedLocalTrainingResponse("run_local_experiment_blocked", taskId);
  }
  const config = await resolveTaskRunConfig(taskId);

  await prisma.task.update({ where: { id: taskId }, data: { status: "running" } });
  const run = await prisma.experimentRun.create({
    data: {
      id: `run_${taskId}_${Date.now()}`,
      taskId,
      status: "running",
      startedAt: new Date()
    }
  });
  await logAction({ action: "run_local_experiment", taskId, runId: run.id, message: `Local experiment started for ${taskId}.` });

  try {
    await runManagedCommand({
      command: pythonExecutable(),
      args: ["scripts/run_workstation_orchestrator.py", "--config", config.config, "--output-base", "experiments", "--random-state", "42"],
      cwd: workspaceRoot,
      timeout: 600000,
      taskId,
      runId: run.id,
      onStart: (pid) => prisma.experimentRun.update({ where: { id: run.id }, data: { processId: pid } }).then(() => undefined)
    });

    const experimentDir = await latestExperimentDir(taskId);
    const relativeExperimentDir = toRelativePath(experimentDir) ?? experimentDir;
    const validation = await runManagedCommand({
      command: pythonExecutable(),
      args: [config.validator, "--experiment-dir", relativeExperimentDir, ...config.validatorArgs],
      cwd: workspaceRoot,
      timeout: 300000,
      taskId,
      runId: run.id
    });
    const validationPayload = JSON.parse(validation.stdout);

    await runManagedCommand({
      command: pythonExecutable(),
      args: ["scripts/write_workstation_summary.py", "--tasks", ...Array.from(new Set(["house_prices", "titanic", "telco_churn", taskId]))],
      cwd: workspaceRoot,
      timeout: 180000,
      taskId,
      runId: run.id
    });

    const metrics = {
      cv_rmsle_mean: validationPayload.cv_rmsle_mean,
      holdout_rmsle: validationPayload.holdout_rmsle,
      submission_rows: validationPayload.submission_rows
    };
    const updatedRun = await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        outputDir: relativeExperimentDir,
        status: validationPayload.status === "passed" ? "passed" : "failed",
        bestModel: validationPayload.best_model ?? null,
        metricsJson: encodeJson(metrics),
        validationStatus: validationPayload.status ?? null,
        processId: null,
        finishedAt: new Date()
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: validationPayload.status === "passed" ? "review" : "failed" } });
    await prisma.gate.create({
      data: {
        id: `gate_${taskId}_${Date.now()}`,
        taskId,
        runId: run.id,
        gateType: "validation_gate",
        decision: validationPayload.status ?? "unknown",
        reviewer: "Local Validator",
        evidenceJson: encodeJson(validationPayload),
        decidedAt: new Date()
      }
    });
    for (const [label, artifactPath] of Object.entries({
      submission: `${relativeExperimentDir}/submission.csv`,
      validation_gate: `${relativeExperimentDir}/validation_gate.json`,
      experiment_record: `${relativeExperimentDir}/experiment_record.json`,
      evidence_manifest: `${relativeExperimentDir}/evidence_manifest.json`
    })) {
      await prisma.evidence.create({
        data: {
          id: `evidence_${label}_${Date.now()}_${Math.round(Math.random() * 1000)}`,
          taskId,
          runId: run.id,
          label,
          artifactPath,
          source: "local_runner",
          claimBinding: label === "validation_gate" ? "validation_status" : "run_artifact"
        }
      });
    }
    await logAction({ action: "run_local_experiment_passed", taskId, runId: run.id, message: `Experiment passed: ${relativeExperimentDir}`, artifactPath: relativeExperimentDir, metadata: { validation: validationPayload } });

    return {
      ok: true,
      task_id: taskId,
      run_id: updatedRun.id,
      experiment_dir: relativeExperimentDir,
      validation: {
        ...validationPayload,
        submission_path: `${relativeExperimentDir}/submission.csv`
      },
      summary: await getWorkstationSummary()
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown run error";
    const currentRun = await prisma.experimentRun.findUnique({ where: { id: run.id } });
    const cancelled = currentRun?.status === "cancelled" || currentRun?.status === "cancel_requested";
    await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        status: cancelled ? "cancelled" : "failed",
        finishedAt: currentRun?.finishedAt ?? new Date(),
        validationStatus: cancelled ? "cancelled" : "failed",
        processId: null
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: cancelled ? "cancelled" : "failed" } });
    await logAction({ action: cancelled ? "run_local_experiment_cancelled" : "run_local_experiment_failed", taskId, runId: run.id, message, metadata: { error: message } });
    throw error;
  }
}

export async function runEnsembleExperiment(taskIdInput: string, options: EnsembleRunOptions = {}) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(taskIdInput);
  if (await localTrainingFallbackDisabled()) {
    return blockedLocalTrainingResponse("run_ensemble_experiment_blocked", taskId);
  }
  const config = await resolveTaskRunConfig(taskId);

  await prisma.task.update({ where: { id: taskId }, data: { status: "running" } });
  const run = await prisma.experimentRun.create({
    data: {
      id: `run_${taskId}_ensemble_${Date.now()}`,
      taskId,
      status: "running",
      startedAt: new Date()
    }
  });
  await logAction({ action: "run_ensemble_experiment", taskId, runId: run.id, message: `Workstation ensemble experiment started for ${taskId}.`, metadata: { fast: Boolean(options.fast), sampleRows: options.sampleRows ?? null } });

  try {
    await runManagedCommand({
      command: pythonExecutable(),
      args: [
        "scripts/run_workstation_ensemble.py",
        "--config",
        config.config,
        "--template",
        "sklearn_rf_hgb_et_ensemble",
        "--output-base",
        "experiments",
        "--random-state",
        "42",
        ...(options.fast ? ["--fast"] : []),
        ...(options.sampleRows ? ["--sample-rows", String(options.sampleRows)] : []),
        ...(options.nFolds ? ["--n-folds", String(options.nFolds)] : []),
        ...(options.seeds ? ["--seeds", options.seeds] : []),
        ...(options.timeoutMs ? ["--timeout-seconds", String(Math.max(1, Math.floor(options.timeoutMs / 1000)))] : [])
      ],
      cwd: workspaceRoot,
      timeout: options.timeoutMs ?? (options.fast ? 600000 : 1800000),
      taskId,
      runId: run.id,
      onStart: (pid) => prisma.experimentRun.update({ where: { id: run.id }, data: { processId: pid } }).then(() => undefined)
    });

    const experimentDir = await latestExperimentDir(taskId);
    const relativeExperimentDir = toRelativePath(experimentDir) ?? experimentDir;
    const metricsPath = path.join(experimentDir, "metrics.json");
    const metricsPayload = JSON.parse(await fs.readFile(metricsPath, "utf-8"));
    const bestScore = metricsPayload?.ensemble?.best_validation_score ?? null;

    await runManagedCommand({
      command: pythonExecutable(),
      args: ["scripts/write_workstation_summary.py", "--tasks", ...Array.from(new Set(["house_prices", "titanic", "telco_churn", taskId]))],
      cwd: workspaceRoot,
      timeout: 180000,
      taskId,
      runId: run.id
    });

    const updatedRun = await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        outputDir: relativeExperimentDir,
        status: metricsPayload.status === "passed" ? "passed" : "failed",
        bestModel: `ensemble_${metricsPayload?.ensemble?.best_method ?? "unknown"}`,
        metricsJson: encodeJson(metricsPayload),
        validationStatus: metricsPayload.status ?? null,
        processId: null,
        finishedAt: new Date()
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: metricsPayload.status === "passed" ? "review" : "failed" } });
    await prisma.gate.create({
      data: {
        id: `gate_${taskId}_ensemble_${Date.now()}`,
        taskId,
        runId: run.id,
        gateType: "ensemble_validation_gate",
        decision: metricsPayload.status ?? "unknown",
        reviewer: "Workstation Search Controller",
        evidenceJson: encodeJson(metricsPayload),
        decidedAt: new Date()
      }
    });
    for (const [label, artifactPath] of Object.entries({
      submission: `${relativeExperimentDir}/submission.csv`,
      metrics: `${relativeExperimentDir}/metrics.json`,
      oof_predictions: `${relativeExperimentDir}/oof_predictions.csv`,
      artifact_manifest: `${relativeExperimentDir}/artifact_manifest.json`
    })) {
      await prisma.evidence.create({
        data: {
          id: `evidence_${label}_${Date.now()}_${Math.round(Math.random() * 1000)}`,
          taskId,
          runId: run.id,
          label,
          artifactPath,
          source: "workstation_ensemble_runner",
          claimBinding: label === "metrics" ? "search_controller_metric" : "run_artifact"
        }
      });
    }
    await logAction({
      action: "run_ensemble_experiment_passed",
      taskId,
      runId: run.id,
      message: `Ensemble experiment completed: ${relativeExperimentDir}`,
      artifactPath: relativeExperimentDir,
      metadata: { bestScore, metrics: metricsPayload }
    });

    return {
      ok: true,
      task_id: taskId,
      run_id: updatedRun.id,
      experiment_dir: relativeExperimentDir,
      metrics: metricsPayload,
      summary: await getWorkstationSummary()
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown run error";
    const currentRun = await prisma.experimentRun.findUnique({ where: { id: run.id } });
    const cancelled = currentRun?.status === "cancelled" || currentRun?.status === "cancel_requested";
    await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        status: cancelled ? "cancelled" : "failed",
        finishedAt: currentRun?.finishedAt ?? new Date(),
        validationStatus: cancelled ? "cancelled" : "failed",
        processId: null
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: cancelled ? "cancelled" : "failed" } });
    await logAction({ action: cancelled ? "run_ensemble_experiment_cancelled" : "run_ensemble_experiment_failed", taskId, runId: run.id, message, metadata: { error: message } });
    throw error;
  }
}

export async function runMCGSExperiment(taskIdInput: string, options: { budgetNodes?: number; fast?: boolean } = {}) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(taskIdInput);
  if (await localTrainingFallbackDisabled()) {
    return blockedLocalTrainingResponse("run_mcgs_experiment_blocked", taskId);
  }
  const config = await resolveTaskRunConfig(taskId);

  await prisma.task.update({ where: { id: taskId }, data: { status: "running" } });
  const run = await prisma.experimentRun.create({
    data: {
      id: `run_${taskId}_mcgs_${Date.now()}`,
      taskId,
      status: "running",
      startedAt: new Date()
    }
  });
  await logAction({ action: "run_mcgs_experiment", taskId, runId: run.id, message: `MCGS self-evolving search started for ${taskId}.`, metadata: { budgetNodes: options.budgetNodes ?? 8 } });

  try {
    await runManagedCommand({
      command: pythonExecutable(),
      args: [
        "scripts/run_workstation_mcgs.py",
        "--config", config.config,
        "--output-base", "experiments",
        "--task-id", taskId,
        "--budget-nodes", String(options.budgetNodes ?? 8),
        ...(options.fast ? ["--fast"] : []),
      ],
      cwd: workspaceRoot,
      timeout: 3600000,  // 60 min for MCGS search
      taskId,
      runId: run.id,
      onStart: (pid) => prisma.experimentRun.update({ where: { id: run.id }, data: { processId: pid } }).then(() => undefined)
    });

    const mcgsDirs = (await fs.readdir(path.join(workspaceRoot, "experiments", taskId), { withFileTypes: true }))
      .filter(e => e.isDirectory() && e.name.startsWith("mcgs_"))
      .sort();
    const latestDir = mcgsDirs[mcgsDirs.length - 1];
    const relativeDir = `experiments/${taskId}/${latestDir?.name ?? ""}`;
    const resultPath = latestDir ? path.join(workspaceRoot, "experiments", taskId, latestDir.name, "mcgs_search_result.json") : null;
    const payload = resultPath ? JSON.parse(await fs.readFile(resultPath, "utf-8")) : {};

    await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        outputDir: relativeDir,
        status: "passed",
        bestModel: `mcgs_${payload.best_node_id ?? "unknown"}`,
        metricsJson: encodeJson(payload),
        validationStatus: "passed",
        processId: null,
        finishedAt: new Date()
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: "review" } });
    await prisma.gate.create({
      data: {
        id: `gate_${taskId}_mcgs_${Date.now()}`,
        taskId,
        runId: run.id,
        gateType: "mcgs_search_gate",
        decision: "promote",
        reviewer: "MCGS Search Controller",
        evidenceJson: encodeJson(payload),
        decidedAt: new Date()
      }
    });
    await logAction({
      action: "run_mcgs_experiment_passed",
      taskId,
      runId: run.id,
      message: `MCGS search completed: ${payload.best_score}`,
      metadata: payload
    });

    return {
      ok: true,
      task_id: taskId,
      run_id: run.id,
      best_score: payload.best_score,
      nodes_evaluated: payload.nodes_evaluated,
      search_result: payload,
      summary: await getWorkstationSummary()
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown MCGS error";
    await prisma.experimentRun.update({
      where: { id: run.id },
      data: { status: "failed", finishedAt: new Date(), validationStatus: "failed", processId: null }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: "failed" } });
    await logAction({ action: "run_mcgs_experiment_failed", taskId, runId: run.id, message, metadata: { error: message } });
    throw error;
  }
}

/**
 * Engine A (research_os EvolutionLoop) training path — the corrected search brain.
 *
 * Mirrors runMCGSExperiment's DB/gate/log shape so the Experiments / Gates UI
 * renders it identically, but drives scripts/evolution_run_cli.py (opus-4-8 +
 * GPURunner or LocalSubprocessRunner) instead of the legacy mlevolve_search.
 * The wrapper emits one JSON line; we parse the LAST JSON line from stdout.
 * Returns a shape compatible with runEvolutionCycle's ingest step (search_result
 * carries exp_dir so ingest_summary can read engine A's summary.json).
 */
export async function runEvolutionEngineExperiment(
  taskIdInput: string,
  options: { runner?: "gpu" | "local"; iterations?: number; mcgs?: boolean; dataDir?: string } = {}
) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(taskIdInput);
  // Engine A's GPU runner is real remote training; the local runner is the only
  // "local compute" path and stays behind the same policy gate as the legacy engine.
  const runner = options.runner ?? "gpu";
  if (runner === "local" && (await localTrainingFallbackDisabled())) {
    return blockedLocalTrainingResponse("run_evolution_engine_blocked", taskId);
  }

  await prisma.task.update({ where: { id: taskId }, data: { status: "running" } });
  const run = await prisma.experimentRun.create({
    data: {
      id: `run_${taskId}_research_os_${Date.now()}`,
      taskId,
      status: "running",
      startedAt: new Date()
    }
  });
  await logAction({ action: "run_evolution_engine", taskId, runId: run.id, message: `research_os evolution engine (${runner}) started for ${taskId}.`, metadata: { runner, iterations: options.iterations ?? 8, mcgs: options.mcgs ?? true } });

  const inputPayload = {
    task_id: taskId,
    runner,
    iterations: options.iterations ?? 8,
    mcgs: options.mcgs ?? true,
    ...(options.dataDir ? { data_dir: options.dataDir } : {})
  };
  const inputDir = path.join(workspaceRoot, "workspace", "evolution", "_io");
  await fs.mkdir(inputDir, { recursive: true });
  const inputFile = path.join(inputDir, `${taskId}_run_${Date.now()}.json`);
  await fs.writeFile(inputFile, JSON.stringify(inputPayload), "utf-8");

  try {
    const { stdout } = await runManagedCommand({
      command: pythonExecutable(),
      args: ["scripts/evolution_run_cli.py", "--input", inputFile],
      cwd: workspaceRoot,
      timeout: 5400000,  // 90 min: GPU evolution loop across multiple iterations
      taskId,
      runId: run.id,
      onStart: (pid) => prisma.experimentRun.update({ where: { id: run.id }, data: { processId: pid } }).then(() => undefined)
    });

    // The CLI emits one JSON object; take the last JSON line to ignore any noise.
    const jsonLine = stdout.split(/\r?\n/).map(l => l.trim()).reverse().find(l => l.startsWith("{") && l.endsWith("}"));
    const payload = jsonLine ? (JSON.parse(jsonLine) as Record<string, unknown>) : {};
    if (payload.ok === false) {
      throw new Error(typeof payload.error === "string" ? payload.error : "research_os evolution engine failed.");
    }

    const expDir = typeof payload.exp_dir === "string" ? payload.exp_dir : "";
    const bestScore = typeof payload.best_cv_score === "number" ? payload.best_cv_score : null;
    await prisma.experimentRun.update({
      where: { id: run.id },
      data: {
        outputDir: expDir,
        status: "passed",
        bestModel: `research_os_${payload.best_exp_id ?? "unknown"}`,
        metricsJson: encodeJson(payload),
        validationStatus: "passed",
        processId: null,
        finishedAt: new Date()
      }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: "review" } });
    await prisma.gate.create({
      data: {
        id: `gate_${taskId}_research_os_${Date.now()}`,
        taskId,
        runId: run.id,
        gateType: "evolution_engine_gate",
        decision: "promote",
        reviewer: "research_os EvolutionLoop",
        evidenceJson: encodeJson(payload),
        decidedAt: new Date()
      }
    });
    await logAction({
      action: "run_evolution_engine_passed",
      taskId,
      runId: run.id,
      message: `research_os evolution completed: ${payload.best_exp_id ?? "?"} ${payload.metric ?? "cv"}=${bestScore}`,
      metadata: payload
    });

    return {
      ok: true,
      task_id: taskId,
      run_id: run.id,
      best_score: bestScore,
      best_exp_id: payload.best_exp_id,
      nodes_evaluated: payload.n_iterations,
      // search_result carries exp_dir so the cycle's ingest step reads engine A's summary.
      search_result: { ...payload, best_run_id: run.id, exp_dir: expDir },
      summary: await getWorkstationSummary()
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown research_os error";
    await prisma.experimentRun.update({
      where: { id: run.id },
      data: { status: "failed", finishedAt: new Date(), validationStatus: "failed", processId: null }
    });
    await prisma.task.update({ where: { id: taskId }, data: { status: "failed" } });
    await logAction({ action: "run_evolution_engine_failed", taskId, runId: run.id, message, metadata: { error: message } });
    throw error;
  } finally {
    await fs.rm(inputFile, { force: true }).catch(() => undefined);
  }
}

export async function generatePaperEvidenceBundle() {
  await ensureWorkstationSeeded();
  const runId = `paper_evidence_bundle_${Date.now()}`;
  await logAction({ action: "generate_paper_evidence_bundle", runId, message: "Generating three-layer paper evidence bundle." });

  const result = await runManagedCommand({
    command: pythonExecutable(),
    args: ["scripts/build_paper_evidence_bundle.py"],
    cwd: workspaceRoot,
    timeout: 300000,
    taskId: "paper_evidence_bundle",
    runId
  });
  const payload = JSON.parse(result.stdout);
  await logAction({
    action: "generate_paper_evidence_bundle_passed",
    runId,
    message: "Three-layer paper evidence bundle generated.",
    artifactPath: payload.paper_report,
    metadata: payload
  });
  return {
    ok: true,
    run_id: runId,
    ...payload,
    summary: await getWorkstationSummary()
  };
}
