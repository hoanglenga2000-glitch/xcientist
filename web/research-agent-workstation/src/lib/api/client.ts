import type { ClaudeAgentSessionResponse, DeepSeekSmokeResponse, EvolutionConfigsResponse, EvolutionCycleRequest, EvolutionCycleResponse, EvolutionGraphResponse, EvolutionMemoryResponse, EvolutionPlanRequest, EvolutionPlanResponse, EvolutionStateResponse, EvolutionStepResponse, GpuGatewayResponse, LiteratureSearchResponse, PaperEvidenceBundleResponse, RunLocalExperimentResponse, ScientistActionQueueSummary, ScientistAutopilotStatusSummary, ScientistAutopilotSummary, ScientistCausalDiagnosisSummary, ScientistContextPacketSummary, ScientistContinuationResumeSummary, ScientistContinuationStatusSummary, ScientistEngineeringLoopResponse, ScientistExecutionContractSummary, ScientistExperimentBlueprintSummary, ScientistHypothesisReviewSummary, ScientistInnovationBacklogSummary, ScientistLoopLessonsSummary, ScientistLoopSummary, ScientistMemoryConsolidationSummary, ScientistNextActionSummary, ScientistPatchActionQueueSummary, ScientistPatchWorkOrderSummary, ScientistReadinessReportSummary, ScientistRecoverySummary, ScientistRepairPlanSummary, ScientistSelfAuditSummary, ScientistSelfUpgradeLoopSummary, ScientistSituationModelSummary, ScientistStepTraceSummary, ScientistStrategyOptimizerSummary, ScientistStreamSummary, ScientistTerminalTurnResponse, ScientistTerminalTurnSummary, ScientistTurnPlanSummary, ScientistTurnsSummary, ScientistUpgradeCampaignRequest, ScientistUpgradeCampaignSummary, ScientistUpgradePlanSummary, ScientistWorkplanSummary, WorkstationActionRequest, WorkstationActionResponse, WorkstationSummary } from "@/lib/api/types";

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string; ok?: boolean };
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload;
}

export async function getWorkstationSummary() {
  return readJson<WorkstationSummary>(await fetch("/api/workstation-summary"));
}

export async function getScientistAutopilot() {
  return readJson<{ ok: boolean; scientist_autopilot: ScientistAutopilotSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_workplan?: ScientistWorkplanSummary; scientist_repair_plan?: ScientistRepairPlanSummary; scientist_execution_contract?: ScientistExecutionContractSummary; scientist_turns?: ScientistTurnsSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_autopilot_status?: ScientistAutopilotStatusSummary }>(await fetch("/api/scientist/autopilot"));
}

export async function runScientistAutopilot() {
  return readJson<{ ok: boolean; action: string; scientist_autopilot: ScientistAutopilotSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_workplan?: ScientistWorkplanSummary; scientist_repair_plan?: ScientistRepairPlanSummary; scientist_execution_contract?: ScientistExecutionContractSummary; scientist_turns?: ScientistTurnsSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_autopilot_status?: ScientistAutopilotStatusSummary }>(
    await fetch("/api/scientist/autopilot", { method: "POST" })
  );
}

export async function startScientistAutopilot() {
  return readJson<{ ok: boolean; action: string; run_id: string; pid?: number | null; status_artifact: string; scientist_autopilot_status?: ScientistAutopilotStatusSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_repair_plan?: ScientistRepairPlanSummary; scientist_execution_contract?: ScientistExecutionContractSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/autopilot/start", { method: "POST" })
  );
}

export async function runScientistNextAction() {
  return readJson<{ ok: boolean; action: string; scientist_next_action: ScientistNextActionSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/next-action", { method: "POST" })
  );
}

export async function getScientistContinuationStatus() {
  return readJson<{ ok: boolean; action: string; scientist_continuation_status: ScientistContinuationStatusSummary; scientist_continuation?: Record<string, unknown>; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/continuation-status")
  );
}

export async function refreshScientistContinuationStatus() {
  return readJson<{ ok: boolean; action: string; scientist_continuation_status: ScientistContinuationStatusSummary; scientist_continuation?: Record<string, unknown>; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/continuation-status", { method: "POST" })
  );
}

export async function getScientistContinuationResume() {
  return readJson<{ ok: boolean; action: string; scientist_continuation_resume: ScientistContinuationResumeSummary; scientist_continuation_status?: ScientistContinuationStatusSummary; scientist_continuation?: Record<string, unknown>; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/continuation-resume")
  );
}

export async function runScientistContinuationResume() {
  return readJson<{ ok: boolean; action: string; scientist_continuation_resume: ScientistContinuationResumeSummary; scientist_continuation_status?: ScientistContinuationStatusSummary; scientist_continuation?: Record<string, unknown>; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/continuation-resume", { method: "POST" })
  );
}

export async function getScientistRecovery() {
  return readJson<{ ok: boolean; action: string; scientist_recovery: ScientistRecoverySummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/recovery")
  );
}

export async function runScientistRecovery() {
  return readJson<{ ok: boolean; action: string; scientist_recovery: ScientistRecoverySummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/recovery", { method: "POST" })
  );
}

export async function getScientistLoop() {
  return readJson<{ ok: boolean; action: string; scientist_loop: ScientistLoopSummary; scientist_loop_lessons?: ScientistLoopLessonsSummary; scientist_memory_consolidation?: ScientistMemoryConsolidationSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_recovery?: ScientistRecoverySummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/loop")
  );
}

export async function runScientistLoop() {
  return readJson<{ ok: boolean; action: string; scientist_loop: ScientistLoopSummary; scientist_loop_lessons?: ScientistLoopLessonsSummary; scientist_memory_consolidation?: ScientistMemoryConsolidationSummary; scientist_action_queue?: ScientistActionQueueSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_recovery?: ScientistRecoverySummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/loop", { method: "POST" })
  );
}

export async function getScientistSelfAudit() {
  return readJson<{ ok: boolean; action: string; scientist_self_audit: ScientistSelfAuditSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/self-audit")
  );
}

export async function runScientistSelfAudit() {
  return readJson<{ ok: boolean; action: string; scientist_self_audit: ScientistSelfAuditSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/self-audit", { method: "POST" })
  );
}

export async function getScientistReadinessReport() {
  return readJson<{ ok: boolean; action: string; scientist_readiness_report: ScientistReadinessReportSummary; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/readiness-report")
  );
}

export async function runScientistReadinessReport() {
  return readJson<{ ok: boolean; action: string; scientist_readiness_report: ScientistReadinessReportSummary; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/readiness-report", { method: "POST" })
  );
}

export async function getScientistCausalDiagnosis() {
  return readJson<{ ok: boolean; action: string; scientist_causal_diagnosis: ScientistCausalDiagnosisSummary; scientist_readiness_report?: ScientistReadinessReportSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/causal-diagnosis")
  );
}

export async function runScientistCausalDiagnosis() {
  return readJson<{ ok: boolean; action: string; scientist_causal_diagnosis: ScientistCausalDiagnosisSummary; scientist_readiness_report?: ScientistReadinessReportSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/causal-diagnosis", { method: "POST" })
  );
}

export async function getScientistStrategyOptimizer() {
  return readJson<{ ok: boolean; action: string; scientist_strategy_optimizer: ScientistStrategyOptimizerSummary; scientist_readiness_report?: ScientistReadinessReportSummary | null; scientist_causal_diagnosis?: ScientistCausalDiagnosisSummary | null; scientist_action_queue?: ScientistActionQueueSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/strategy-optimizer")
  );
}

export async function runScientistStrategyOptimizer() {
  return readJson<{ ok: boolean; action: string; scientist_strategy_optimizer: ScientistStrategyOptimizerSummary; scientist_readiness_report?: ScientistReadinessReportSummary | null; scientist_causal_diagnosis?: ScientistCausalDiagnosisSummary | null; scientist_action_queue?: ScientistActionQueueSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/strategy-optimizer", { method: "POST" })
  );
}

export async function getScientistContextPacket() {
  return readJson<{ ok: boolean; action: string; scientist_context_packet: ScientistContextPacketSummary; scientist_strategy_optimizer?: ScientistStrategyOptimizerSummary | null; scientist_readiness_report?: ScientistReadinessReportSummary | null; scientist_action_queue?: ScientistActionQueueSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/context-packet")
  );
}

export async function runScientistContextPacket() {
  return readJson<{ ok: boolean; action: string; scientist_context_packet: ScientistContextPacketSummary; scientist_strategy_optimizer?: ScientistStrategyOptimizerSummary | null; scientist_readiness_report?: ScientistReadinessReportSummary | null; scientist_action_queue?: ScientistActionQueueSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/context-packet", { method: "POST" })
  );
}

export async function getScientistUpgradePlan() {
  return readJson<{ ok: boolean; action: string; scientist_upgrade_plan: ScientistUpgradePlanSummary; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/upgrade-plan")
  );
}

export async function runScientistUpgradePlan() {
  return readJson<{ ok: boolean; action: string; scientist_upgrade_plan: ScientistUpgradePlanSummary; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/upgrade-plan", { method: "POST" })
  );
}

export async function getScientistSelfUpgradeLoop() {
  return readJson<{ ok: boolean; action: string; scientist_self_upgrade_loop: ScientistSelfUpgradeLoopSummary; scientist_upgrade_plan?: ScientistUpgradePlanSummary | null; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/self-upgrade")
  );
}

export async function runScientistSelfUpgradeLoop() {
  return readJson<{ ok: boolean; action: string; scientist_self_upgrade_loop: ScientistSelfUpgradeLoopSummary; scientist_upgrade_plan?: ScientistUpgradePlanSummary | null; scientist_self_audit?: ScientistSelfAuditSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/self-upgrade", { method: "POST" })
  );
}

export async function getScientistUpgradeCampaign() {
  return readJson<{ ok: boolean; action: string; scientist_upgrade_campaign: ScientistUpgradeCampaignSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/upgrade-campaign")
  );
}

export async function runScientistUpgradeCampaign(request: ScientistUpgradeCampaignRequest) {
  return readJson<{ ok: boolean; action: string; scientist_upgrade_campaign: ScientistUpgradeCampaignSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/upgrade-campaign", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request)
    })
  );
}

export async function getScientistPatchWorkOrder() {
  return readJson<{ ok: boolean; action: string; scientist_patch_work_order: ScientistPatchWorkOrderSummary; scientist_action_queue?: ScientistPatchActionQueueSummary | null; scientist_terminal_turn?: ScientistTerminalTurnSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/patch-work-order")
  );
}

export async function runScientistPatchWorkOrder() {
  return readJson<{ ok: boolean; action: string; scientist_patch_work_order: ScientistPatchWorkOrderSummary; scientist_action_queue?: ScientistPatchActionQueueSummary | null; scientist_terminal_turn?: ScientistTerminalTurnSummary | null; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/patch-work-order", { method: "POST" })
  );
}

export async function getScientistEngineeringLoop() {
  return readJson<ScientistEngineeringLoopResponse>(
    await fetch("/api/scientist/engineering-loop")
  );
}

export async function runScientistEngineeringLoop(options?: {
  generatePatch?: boolean;
  patchPath?: string;
  workOrderPath?: string;
  timeoutSeconds?: number;
}) {
  return readJson<ScientistEngineeringLoopResponse>(
    await fetch("/api/scientist/engineering-loop", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        generate_patch: options?.generatePatch ?? false,
        patch_path: options?.patchPath,
        work_order_path: options?.workOrderPath,
        timeout_seconds: options?.timeoutSeconds ?? 240
      })
    })
  );
}

export async function getScientistInnovationBacklog() {
  return readJson<{ ok: boolean; action: string; scientist_innovation_backlog: ScientistInnovationBacklogSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/innovation-backlog")
  );
}

export async function runScientistInnovationBacklog() {
  return readJson<{ ok: boolean; action: string; scientist_innovation_backlog: ScientistInnovationBacklogSummary; scientist_self_audit?: ScientistSelfAuditSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/innovation-backlog", { method: "POST" })
  );
}

export async function getScientistHypothesisReview() {
  return readJson<{ ok: boolean; action: string; scientist_hypothesis_review: ScientistHypothesisReviewSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/hypothesis-review")
  );
}

export async function runScientistHypothesisReview() {
  return readJson<{ ok: boolean; action: string; scientist_hypothesis_review: ScientistHypothesisReviewSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/hypothesis-review", { method: "POST" })
  );
}

export async function getScientistExperimentBlueprint() {
  return readJson<{ ok: boolean; action: string; scientist_experiment_blueprint: ScientistExperimentBlueprintSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/experiment-blueprint")
  );
}

export async function runScientistExperimentBlueprint() {
  return readJson<{ ok: boolean; action: string; scientist_experiment_blueprint: ScientistExperimentBlueprintSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/experiment-blueprint", { method: "POST" })
  );
}

export async function getScientistSituationModel() {
  return readJson<{ ok: boolean; action: string; scientist_situation_model: ScientistSituationModelSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/situation-model")
  );
}

export async function runScientistSituationModel() {
  return readJson<{ ok: boolean; action: string; scientist_situation_model: ScientistSituationModelSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/situation-model", { method: "POST" })
  );
}

export async function getScientistTurnPlan() {
  return readJson<{ ok: boolean; action: string; scientist_turn_plan: ScientistTurnPlanSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/turn-plan")
  );
}

export async function runScientistTurnPlan() {
  return readJson<{ ok: boolean; action: string; scientist_turn_plan: ScientistTurnPlanSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/turn-plan", { method: "POST" })
  );
}

export async function getScientistTurn() {
  return readJson<ScientistTerminalTurnResponse>(await fetch("/api/scientist/turn"));
}

export async function runScientistTurn(prompt: string, maxTools = 4) {
  return readJson<ScientistTerminalTurnResponse>(
    await fetch("/api/scientist/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_tools: maxTools })
    })
  );
}

export async function getScientistStream() {
  return readJson<{ ok: boolean; action: string; scientist_stream: ScientistStreamSummary; scientist_step_trace?: ScientistStepTraceSummary; scientist_turns?: ScientistTurnsSummary; scientist_terminal_turn?: ScientistTerminalTurnSummary; scientist_autopilot_status?: ScientistAutopilotStatusSummary; no_training_started?: boolean; official_submit?: string }>(
    await fetch("/api/scientist/stream")
  );
}

export async function runLocalExperiment(taskId: string) {
  return readJson<RunLocalExperimentResponse>(await fetch(`/api/tasks/${taskId}/run-local-experiment`, { method: "POST" }));
}

export async function getPaperEvidenceBundle() {
  return readJson<PaperEvidenceBundleResponse>(await fetch("/api/paper-evidence-bundle"));
}

// ── Evolution engine client ──────────────────────────────────────────────────
function evolutionQuery(taskId: string, extra: Record<string, string> = {}) {
  const params = new URLSearchParams({ task_id: taskId, ...extra });
  return params.toString();
}

export async function getEvolutionState(taskId: string, opts: { taskType?: string; metricDirection?: string } = {}) {
  const query = evolutionQuery(taskId, {
    ...(opts.taskType ? { task_type: opts.taskType } : {}),
    ...(opts.metricDirection ? { metric_direction: opts.metricDirection } : {})
  });
  return readJson<EvolutionStateResponse>(await fetch(`/api/evolution/state?${query}`));
}

export async function getEvolutionGraph(taskId: string, opts: { metricDirection?: string } = {}) {
  const query = evolutionQuery(taskId, opts.metricDirection ? { metric_direction: opts.metricDirection } : {});
  return readJson<EvolutionGraphResponse>(await fetch(`/api/evolution/graph?${query}`));
}

export async function getEvolutionMemory(taskId: string, opts: { taskType?: string } = {}) {
  const query = evolutionQuery(taskId, opts.taskType ? { task_type: opts.taskType } : {});
  return readJson<EvolutionMemoryResponse>(await fetch(`/api/evolution/memory?${query}`));
}

export async function planEvolution(payload: EvolutionPlanRequest) {
  return readJson<EvolutionPlanResponse>(
    await fetch("/api/evolution/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, official_submit_allowed: false })
    })
  );
}

export async function runEvolutionStep(payload: EvolutionPlanRequest & { dry_run?: boolean }) {
  return readJson<EvolutionStepResponse>(
    await fetch("/api/evolution/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, dry_run: payload.dry_run !== false, official_submit_allowed: false })
    })
  );
}

export async function listEvolutionConfigs() {
  return readJson<EvolutionConfigsResponse>(await fetch("/api/evolution/configs"));
}

/**
 * Drive the full evolution closed loop. Without `approve:true` the server returns
 * the plan and stops (no training). Real training only launches on approval and is
 * still routed through the workstation gates. Kaggle submit stays hard-disabled.
 */
export async function runEvolutionCycle(payload: EvolutionCycleRequest) {
  return readJson<EvolutionCycleResponse>(
    await fetch("/api/evolution/cycle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, official_submit_allowed: false })
    })
  );
}

export async function generatePaperEvidenceBundle() {
  return readJson<PaperEvidenceBundleResponse>(await fetch("/api/paper-evidence-bundle", { method: "POST" }));
}

export async function runWorkstationAction(action: string, taskId = "house_prices", metadata?: Record<string, unknown>) {
  return readJson<WorkstationActionResponse>(
    await fetch("/api/workstation-actions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, task_id: taskId, metadata } satisfies WorkstationActionRequest)
    })
  );
}

export async function exportCodeAgentContext(taskId: string, targetAgent = "claude_code") {
  return readJson<{ ok: boolean; task_id: string; context_dir: string; target_agent?: string }>(
    await fetch(`/api/tasks/${taskId}/export-code-agent-context`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_agent: targetAgent })
    })
  );
}

export async function importAgentPatch(taskId: string, payload: { source_agent: string; patch_diff: string }) {
  return readJson<{ ok: boolean; task_id: string; patch_id: string; patch_path: string }>(
    await fetch(`/api/tasks/${taskId}/import-agent-patch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function generateCodeAgentDraft(taskId: string, payload: {
  source_agent: string;
  model?: string;
  prompt?: string;
  max_turns?: number;
  timeout_seconds?: number;
  cache_only?: boolean;
  cache_probe?: boolean;
}) {
  return readJson<{
    ok: boolean;
    task_id: string;
    source_agent: string;
    draft_path: string;
    patch_path: string | null;
    manifest_path: string;
    generated_code: string;
    patch_diff: string;
    cli_status: string;
    configured?: boolean;
    session_id?: string;
    transcript_path?: string;
    usage?: Record<string, unknown>;
    prompt_cache_hit_tokens?: number | null;
    deepseek_cache?: Record<string, unknown>;
    prompt_fingerprint?: string;
    cache_key?: string;
    local_response_cache_hit?: boolean;
    cache_entry_path?: string | null;
    external_model_calls_allowed?: boolean;
    missing_env?: string[];
    error?: string;
  }>(
    await fetch(`/api/tasks/${taskId}/code-agent-draft`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function createClaudeAgentSession(taskId: string, payload: { prompt?: string; model?: string; max_turns?: number; timeout_seconds?: number }) {
  return readJson<ClaudeAgentSessionResponse>(
    await fetch("/api/code-agents/claude/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, ...payload })
    })
  );
}

export async function getClaudeAgentSession(sessionId: string) {
  return readJson<ClaudeAgentSessionResponse>(await fetch(`/api/code-agents/claude/sessions/${sessionId}`));
}

export async function cancelClaudeAgentSession(sessionId: string) {
  return readJson<ClaudeAgentSessionResponse>(await fetch(`/api/code-agents/claude/sessions/${sessionId}/cancel`, { method: "POST" }));
}

export async function testGpuConnection() {
  return readJson<GpuGatewayResponse>(await fetch("/api/gpu/connections/test", { method: "POST" }));
}

export async function submitGpuJob(
  taskId: string,
  template?: string,
  metadata?: { run_id?: string; agent_id?: string; gate_id?: string; resource_request?: Record<string, unknown> }
) {
  return readJson<GpuGatewayResponse>(
    await fetch("/api/gpu/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, template, ...metadata })
    })
  );
}

export async function cancelGpuJob(jobId: string) {
  return readJson<GpuGatewayResponse>(await fetch(`/api/gpu/jobs/${jobId}/cancel`, { method: "POST" }));
}

export async function testDeepSeek(prompt?: string) {
  return readJson<DeepSeekSmokeResponse>(
    await fetch("/api/llm/deepseek/smoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    })
  );
}

export async function listTasks() {
  return readJson<{ ok: boolean; tasks: unknown[] }>(await fetch("/api/tasks"));
}

export async function listRuns(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; runs: unknown[] }>(await fetch(`/api/tasks/${taskId}/runs`));
}

export async function listGates(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; gates: unknown[] }>(await fetch(`/api/tasks/${taskId}/gates`));
}

export async function listEvidence(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; evidence: unknown[] }>(await fetch(`/api/tasks/${taskId}/evidence`));
}

export async function getReport(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; report: unknown }>(await fetch(`/api/tasks/${taskId}/report`));
}

export async function saveReport(taskId: string, payload: { title: string; markdown_content: string; selected_section?: string | null; status?: string }) {
  return readJson<{ ok: boolean; task_id: string; report: unknown; markdown_path?: string | null; html_path?: string | null }>(
    await fetch(`/api/tasks/${taskId}/report`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function generateReportDraft(taskId: string, payload: { language?: string; style?: string }) {
  return readJson<{
    ok: boolean;
    task_id: string;
    report: unknown;
    markdown_content: string;
    markdown_path?: string | null;
    html_path?: string | null;
  }>(
    await fetch(`/api/tasks/${taskId}/generate-report-draft`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function searchLiterature(payload: { task_id: string; query: string; max_results?: number; include_arxiv?: boolean }) {
  return readJson<LiteratureSearchResponse>(
    await fetch("/api/literature/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function getWorkflow(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; workflow: unknown }>(await fetch(`/api/tasks/${taskId}/workflow`));
}

export async function getSettings() {
  return readJson<{ ok: boolean; settings: Record<string, Record<string, unknown>> }>(await fetch("/api/settings"));
}

export async function saveSettings(settings: Record<string, Record<string, unknown>>) {
  return readJson<{ ok: boolean; settings: Record<string, Record<string, unknown>> }>(
    await fetch("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings })
    })
  );
}

export async function generateFigures(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; figures: Array<{ name: string; path: string; type: string }>; manifest_path: string }>(
    await fetch(`/api/tasks/${taskId}/generate-figures`, { method: "POST" })
  );
}

export async function listFigures(taskId: string) {
  return readJson<{ ok: boolean; task_id: string; figures: Array<{ name: string; path: string; type: string }> }>(
    await fetch(`/api/tasks/${taskId}/figures`)
  );
}

export async function insertReportFigure(reportId: string, figurePath: string, caption: string) {
  return readJson<{ ok: boolean; report: unknown }>(
    await fetch(`/api/reports/${reportId}/insert-figure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ figure_path: figurePath, caption })
    })
  );
}
