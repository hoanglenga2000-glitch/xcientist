import type { ClaudeAgentSessionResponse, DeepSeekSmokeResponse, EvolutionConfigsResponse, EvolutionCycleRequest, EvolutionCycleResponse, EvolutionGraphResponse, EvolutionMemoryResponse, EvolutionPlanRequest, EvolutionPlanResponse, EvolutionStateResponse, EvolutionStepResponse, GpuGatewayResponse, LiteratureSearchResponse, PaperEvidenceBundleResponse, RunLocalExperimentResponse, WorkstationActionRequest, WorkstationActionResponse, WorkstationSummary } from "@/lib/api/types";

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
