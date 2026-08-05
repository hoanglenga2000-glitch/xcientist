"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { PageHeader } from "./primitives/Layout";
import { t as tV2 } from "./localization";
import type {
  EvolutionConfigSummary,
  EvolutionCycleResponse,
  EvolutionEngine,
  EvolutionExperienceResponse,
  EvolutionGraphResponse,
  LocalGpuTelemetryResponse,
  EvolutionMemoryResponse,
  EvolutionRunner,
  EvolutionSearchMode,
  EvolutionStateResponse,
  EvolutionStepResponse
} from "@/lib/api/types";

const mono = "font-mono text-[12px] tracking-normal";
const cnEvolution = cn;

type Props = {
  selectedTask: string;
  refreshSummary?: () => Promise<unknown>;
};

function fmtScore(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(5);
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fmtCompact(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(value);
}

function cycleTone(stage: EvolutionCycleResponse["stage"]): "green" | "amber" | "slate" | "red" {
  switch (stage) {
    case "completed":
      return "green";
    case "training":
      return "amber";
    case "training_blocked":
    case "training_failed":
      return "red";
    default:
      return "slate";
  }
}

function evidenceTone(status: "verified" | "failed" | "not_present"): "green" | "slate" | "red" {
  if (status === "verified") return "green";
  if (status === "failed") return "red";
  return "slate";
}

export function EvolutionConsole({ selectedTask, refreshSummary }: Props) {
  const [state, setState] = useState<EvolutionStateResponse | null>(null);
  const [graph, setGraph] = useState<EvolutionGraphResponse | null>(null);
  const [memory, setMemory] = useState<EvolutionMemoryResponse | null>(null);
  const [experience, setExperience] = useState<EvolutionExperienceResponse | null>(null);
  const [lastStep, setLastStep] = useState<EvolutionStepResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<"" | "plan" | "step" | "cycle">("");
  const [error, setError] = useState<string>("");
  const [message, setMessage] = useState<string>("");

  // Real closed-loop controls. `engineTask` is chosen from configs/evolution/*.json
  // so the picker matches the engine's actual task set. Approval is a two-click
  // human gate: the first "Approve & run" click arms it, the second launches.
  const [configs, setConfigs] = useState<EvolutionConfigSummary[]>([]);
  const [engineTask, setEngineTask] = useState<string>(selectedTask);
  const [engine, setEngine] = useState<EvolutionEngine>("research_os");
  const [runner, setRunner] = useState<EvolutionRunner>("gpu");
  const [iterations, setIterations] = useState<number>(3);
  const [useMcgs, setUseMcgs] = useState<boolean>(true);
  const [searchMode, setSearchMode] = useState<EvolutionSearchMode>("legacy_uct");
  const [maxNodes, setMaxNodes] = useState<number>(3);
  const [maxTokens, setMaxTokens] = useState<number>(700_000);
  const [maxWallSeconds, setMaxWallSeconds] = useState<number>(14_400);
  const [maxCost, setMaxCost] = useState<string>("");
  const [armApproval, setArmApproval] = useState<boolean>(false);
  const [approvalReceipt, setApprovalReceipt] = useState<{
    plan_id: string;
    plan_sha256: string;
    request_fingerprint: string;
    approval_expires_at?: string;
  } | null>(null);
  const [cycle, setCycle] = useState<EvolutionCycleResponse | null>(null);
  const [telemetry, setTelemetry] = useState<LocalGpuTelemetryResponse | null>(null);

  const activeConfig = useMemo(
    () => configs.find((c) => c.task_id === engineTask) ?? null,
    [configs, engineTask]
  );
  const demoLocked = activeConfig?.demo_campaign === true;

  const independentReview = recordValue(cycle?.training?.independent_review);
  const reviewMetrics = recordValue(independentReview.metrics);
  const claimAudit = recordValue(independentReview.claim_audit);
  const resourceGate = recordValue(cycle?.training?.resource_gate);
  const resourceGpu = recordValue(resourceGate.gpu);

  const load = useCallback(async () => {
    if (!selectedTask) return;
    setLoading(true);
    setError("");
    try {
      const [s, g, m, x] = await Promise.all([
        api.getEvolutionState(selectedTask),
        api.getEvolutionGraph(selectedTask),
        api.getEvolutionMemory(selectedTask),
        api.getEvolutionExperience(selectedTask)
      ]);
      setState(s);
      setGraph(g);
      setMemory(m);
      setExperience(x);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load evolution state");
    } finally {
      setLoading(false);
    }
  }, [selectedTask]);

  const loadExperience = useCallback(async (taskId: string, runId?: string | null) => {
    if (!taskId) return;
    try {
      setExperience(await api.getEvolutionExperience(taskId, runId));
    } catch {
      setExperience(null);
    }
  }, []);

  useEffect(() => {
    if (activeConfig?.demo_campaign) {
      setEngine("research_os");
      setRunner(activeConfig.required_runner ?? "local");
      setIterations(activeConfig.required_iterations ?? 8);
      setUseMcgs(activeConfig.required_mcgs ?? true);
      setSearchMode(activeConfig.default_search_mode ?? "experience_mcgs_v1");
      setMaxNodes(activeConfig.required_max_nodes ?? 8);
      setMaxTokens(activeConfig.required_max_tokens ?? 100_000);
      setMaxWallSeconds(activeConfig.required_max_wall_seconds ?? 600);
      setMaxCost(String(activeConfig.required_max_cost ?? 0.01));
    } else if (activeConfig?.compute_backend === "local_gpu") {
      setRunner("local_gpu");
    }
  }, [activeConfig]);

  useEffect(() => {
    if (engineTask && engineTask !== selectedTask) void loadExperience(engineTask);
  }, [engineTask, selectedTask, loadExperience]);

  useEffect(() => {
    if (runner !== "local_gpu") {
      setTelemetry(null);
      return;
    }
    let cancelled = false;
    const refresh = async () => {
      try {
        const next = await api.getLocalGpuTelemetry();
        if (!cancelled) setTelemetry(next);
      } catch {
        if (!cancelled) setTelemetry(null);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [runner]);

  useEffect(() => {
    void load();
  }, [load]);

  // Discover real task configs once; keep engineTask on the parent selection when
  // it is a known config, otherwise fall back to the first available config.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await api.listEvolutionConfigs();
        if (cancelled) return;
        setConfigs(res.configs);
        setEngineTask((prev) => {
          if (prev && res.configs.some((c) => c.task_id === prev)) return prev;
          if (selectedTask && res.configs.some((c) => c.task_id === selectedTask)) return selectedTask;
          return res.configs[0]?.task_id ?? prev;
        });
      } catch {
        // Non-fatal: the picker just stays on the free-typed task id.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedTask]);

  // Reset the approval arm whenever the target task or run shape changes so a
  // primed "run" never carries over to a different configuration.
  useEffect(() => {
    setArmApproval(false);
    setApprovalReceipt(null);
  }, [engineTask, engine, runner, iterations, useMcgs, searchMode, maxNodes, maxTokens, maxWallSeconds, maxCost]);

  const onPlan = useCallback(async () => {
    if (!selectedTask) return;
    setBusy("plan");
    setError("");
    setMessage("");
    try {
      const plan = await api.planEvolution({ task_id: selectedTask, objective: `Evolve ${selectedTask}` });
      setMessage(
        `Planned: ${plan.search_controller_decision} on ${plan.selected_branch} (${plan.code_generation_mode}/${plan.expansion_type}). Strategies: ${plan.recommended_strategies.join(", ") || "—"}.`
      );
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Plan failed");
    } finally {
      setBusy("");
    }
  }, [selectedTask, load]);

  const onStep = useCallback(async () => {
    if (!selectedTask) return;
    setBusy("step");
    setError("");
    setMessage("");
    try {
      const step = await api.runEvolutionStep({ task_id: selectedTask, dry_run: true });
      setLastStep(step);
      setMessage(
        step.dry_run
          ? `Dry-run step recorded node ${step.exp_id ?? "?"} (${step.code_generation_mode}/${step.expansion_type}). No training executed.`
          : `Step decision: ${step.decision}.`
      );
      await load();
      await refreshSummary?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Step failed");
    } finally {
      setBusy("");
    }
  }, [selectedTask, load, refreshSummary]);

  // Full closed loop. approve=false returns the plan and stops (no training).
  // approve=true only runs because the human clicked the armed "Approve & run"
  // button; even then Kaggle submit stays hard-disabled server-side.
  const onCycle = useCallback(
    async (approve: boolean) => {
      const task = engineTask || selectedTask;
      if (!task) return;
      if (approve && !approvalReceipt) {
        setError("Plan this exact configuration before approving training.");
        setArmApproval(false);
        return;
      }
      setBusy("cycle");
      setError("");
      setMessage("");
      try {
        const result = await api.runEvolutionCycle({
          task_id: task,
          engine,
          runner,
          iterations,
          mcgs: useMcgs,
          search_mode: searchMode,
          max_nodes: maxNodes,
          max_tokens: maxTokens,
          max_wall_seconds: maxWallSeconds,
          max_cost: maxCost.trim() ? Number(maxCost) : null,
          approve,
          ...(approve && approvalReceipt ? {
            plan_id: approvalReceipt.plan_id,
            plan_sha256: approvalReceipt.plan_sha256,
            request_fingerprint: approvalReceipt.request_fingerprint
          } : {})
        });
        setCycle(result);
        if (!approve && result.plan_id && result.plan_sha256 && result.request_fingerprint) {
          setApprovalReceipt({
            plan_id: result.plan_id,
            plan_sha256: result.plan_sha256,
            request_fingerprint: result.request_fingerprint,
            approval_expires_at: result.approval_expires_at
          });
        } else if (approve) {
          setApprovalReceipt(null);
          setArmApproval(false);
        }
        setMessage(
          approve
            ? `Cycle ${result.stage}: ${result.next_action ?? result.reason ?? "see result below"}.`
            : `Plan ready (${result.stage}). Review below, then Approve & run to launch real training.`
        );
        await load();
        if (approve && result.training?.run_id) await loadExperience(task, result.training.run_id);
        else await loadExperience(task);
        await refreshSummary?.();
      } catch (err) {
        if (approve) {
          setApprovalReceipt(null);
          setArmApproval(false);
        }
        setError(err instanceof Error ? err.message : "Cycle failed");
      } finally {
        setBusy("");
      }
    },
    [engineTask, selectedTask, engine, runner, iterations, useMcgs, searchMode, maxNodes, maxTokens, maxWallSeconds, maxCost, approvalReceipt, load, loadExperience, refreshSummary]
  );

  const nodesByScore = useMemo(() => {
    if (!graph?.nodes) return [];
    const lower = (graph.metric_direction ?? "maximize").toLowerCase().startsWith("min");
    return [...graph.nodes].sort((a, b) => {
      const av = a.cv_score ?? (lower ? Infinity : -Infinity);
      const bv = b.cv_score ?? (lower ? Infinity : -Infinity);
      return lower ? av - bv : bv - av;
    });
  }, [graph]);

  const latestTrace = experience?.selection_traces.at(-1) ?? null;
  const latestRetrieval = experience?.retrievals.at(-1) ?? null;
  const selectionByCard = useMemo(
    () => new Map((latestTrace?.candidates ?? []).map((item) => [item.card_id, item])),
    [latestTrace]
  );
  const experienceCards = useMemo(
    () => [...(experience?.cards ?? [])].sort((a, b) => a.execution_id.localeCompare(b.execution_id) || a.card_id.localeCompare(b.card_id)),
    [experience]
  );
  const retrievedCardIds = useMemo(() => new Set(latestRetrieval?.card_ids ?? []), [latestRetrieval]);

  const hasRun = Boolean(state?.has_run ?? graph?.has_run);
  return (
    <div className="space-y-4">
      <PageHeader
        title={tV2("zh-CN", "Evolution Engine", "自进化引擎")}
        subtitle={tV2("zh-CN", "Search graph, retrospective memory, and branch expansion planning", "搜索图、检索记忆与分支扩展规划")}
        breadcrumb={tV2("zh-CN", "Research Loop > Evolution Engine", "研究循环 > 自进化引擎")}
      />
      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
          <div className="min-w-0">
            <CardTitle>Evolution Engine</CardTitle>
            <CardDescription>
              Self-evolving search brain for <span className={mono}>{selectedTask || "—"}</span>. Plans expansions,
              records a real search graph, and reuses retrospective memory. Training runs through the workstation
              orchestrator — this console never bypasses the gates.
            </CardDescription>
          </div>
          <div className="flex w-full flex-wrap gap-2 sm:w-auto sm:shrink-0">
            <Button variant="secondary" data-ui-action="evolution_refresh" data-ui-skip-action="true" onClick={() => void load()} disabled={loading || !selectedTask}>
              {loading ? "Loading…" : "Refresh"}
            </Button>
            <Button variant="secondary" data-ui-action="evolution_plan_next" data-ui-skip-action="true" onClick={() => void onPlan()} disabled={busy !== "" || !selectedTask}>
              {busy === "plan" ? "Planning…" : "Plan next"}
            </Button>
            <Button variant="primary" data-ui-action="evolution_dry_run_step" data-ui-skip-action="true" onClick={() => void onStep()} disabled={busy !== "" || !selectedTask}>
              {busy === "step" ? "Stepping…" : "Dry-run step"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {error && (
            <p className={cnEvolution("rounded-md border border-danger/40 bg-danger-light px-3 py-2 text-danger-text", mono)}>
              {error}
            </p>
          )}
          {message && (
            <p className={cnEvolution("rounded-md border border-success/40 bg-success-light px-3 py-2 text-success-text", mono)}>
              {message}
            </p>
          )}
          <div className="space-y-3 rounded-md border border-edge bg-surface-sunken p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-[12px] font-semibold text-ink">Closed loop (plan → approve → train → ingest)</p>
              <StatusBadge tone="slate">Kaggle submit disabled</StatusBadge>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Task config
                <select
                  className={cnEvolution("rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink disabled:cursor-not-allowed disabled:opacity-60", mono)}
                  value={engineTask}
                  disabled={busy !== ""}
                  onChange={(e) => setEngineTask(e.target.value)}
                >
                  {configs.length === 0 && <option value={engineTask}>{engineTask || "—"}</option>}
                  {configs.map((c) => (
                    <option key={c.task_id} value={c.task_id}>
                      {c.task_id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Engine
                <select
                  className={cnEvolution("rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink disabled:cursor-not-allowed disabled:opacity-60", mono)}
                  value={engine}
                  disabled={busy !== "" || demoLocked}
                  onChange={(e) => setEngine(e.target.value as EvolutionEngine)}
                >
                  <option value="research_os">research_os</option>
                  <option value="legacy">legacy</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Runner
                <select
                  className={cnEvolution("rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink disabled:cursor-not-allowed disabled:opacity-60", mono)}
                  value={runner}
                  disabled={busy !== "" || demoLocked}
                  onChange={(e) => setRunner(e.target.value as EvolutionRunner)}
                >
                  <option value="gpu">gpu</option>
                  <option value="local_gpu">local_gpu · RTX 4060</option>
                  <option value="local">local</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Iterations
                <input
                  type="number"
                  min={1}
                  max={50}
                  className={cnEvolution("rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink disabled:cursor-not-allowed disabled:opacity-60", mono)}
                  value={iterations}
                  disabled={busy !== "" || demoLocked}
                  onChange={(e) => setIterations(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
                />
              </label>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Search mode
                <select
                  aria-label="Evolution search mode"
                  className={cnEvolution("min-h-11 rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink disabled:cursor-not-allowed disabled:opacity-60", mono)}
                  value={searchMode}
                  disabled={busy !== "" || engine !== "research_os" || demoLocked}
                  onChange={(event) => {
                    const mode = event.target.value as EvolutionSearchMode;
                    setSearchMode(mode);
                    if (mode === "experience_mcgs_v1") setUseMcgs(true);
                  }}
                >
                  <option value="legacy_uct">legacy_uct</option>
                  <option value="experience_mcgs_v1">experience_mcgs_v1</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Max nodes
                <input
                  aria-label="Maximum evolution nodes"
                  type="number"
                  min={1}
                  max={64}
                  className={cnEvolution("min-h-11 rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink", mono)}
                  value={maxNodes}
                  disabled={busy !== "" || demoLocked}
                  onChange={(event) => setMaxNodes(Math.max(1, Math.min(64, Number(event.target.value) || 1)))}
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Max tokens
                <input
                  aria-label="Maximum total LLM tokens"
                  type="number"
                  min={1}
                  max={10_000_000}
                  step={10_000}
                  className={cnEvolution("min-h-11 rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink", mono)}
                  value={maxTokens}
                  disabled={busy !== "" || demoLocked}
                  onChange={(event) => setMaxTokens(Math.max(1, Math.min(10_000_000, Number(event.target.value) || 1)))}
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Max wall (s)
                <input
                  aria-label="Maximum wall clock seconds"
                  type="number"
                  min={1}
                  max={43_200}
                  step={300}
                  className={cnEvolution("min-h-11 rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink", mono)}
                  value={maxWallSeconds}
                  disabled={busy !== "" || demoLocked}
                  onChange={(event) => setMaxWallSeconds(Math.max(1, Math.min(43_200, Number(event.target.value) || 1)))}
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-secondary">
                Max cost (USD)
                <input
                  aria-label="Maximum estimated cost in USD"
                  type="number"
                  min={0}
                  max={100_000}
                  step={0.01}
                  placeholder="unbounded"
                  className={cnEvolution("min-h-11 rounded-md border border-edge bg-surface-raised px-2 py-1 text-ink", mono)}
                  value={maxCost}
                  disabled={busy !== "" || demoLocked}
                  onChange={(event) => setMaxCost(event.target.value)}
                />
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
                <input
                  type="checkbox"
                  checked={useMcgs}
                  disabled={busy !== "" || searchMode === "experience_mcgs_v1" || demoLocked}
                  onChange={(e) => setUseMcgs(e.target.checked)}
                />
                MCGS search
              </label>
              {activeConfig && (
                <p className="text-[11px] text-ink-secondary">
                  {activeConfig.modality ?? "?"}/{activeConfig.task_type ?? "?"} · {activeConfig.metric ?? "?"} (
                  {activeConfig.metric_direction ?? "?"}) · n_train={activeConfig.n_train ?? "?"} ·{" "}
                  {activeConfig.compute_backend === "local_gpu" ? "RTX 4060 · " : ""}
                  {activeConfig.has_local_data_dir ? "local data ✓" : activeConfig.has_gpu_data_dir ? "GPU data ✓" : "data missing"}
                  {activeConfig.compute_backend === "local_gpu" ? " · Human Gate" : ""}
                  {activeConfig.demo_campaign ? " · Demo contract locked: local / Experience-MCGS / 8 nodes" : ""}
                </p>
              )}
            </div>
            {runner === "gpu" && activeConfig && !activeConfig.has_gpu_data_dir && (
              <p className={cnEvolution("rounded-md border border-warning/40 bg-warning-light px-3 py-2 text-warning-text", mono)}>
                This config has no gpu_data_dir — a GPU run will likely fail. Pick a GPU-ready config or switch runner to local.
              </p>
            )}
            {runner === "local_gpu" && activeConfig && !activeConfig.has_local_data_dir && (
              <p className={cnEvolution("rounded-md border border-danger/40 bg-danger-light px-3 py-2 text-danger-text", mono)}>
                Local RTX 4060 data is missing. Resource Gate will block before training.
              </p>
            )}
            {runner === "local_gpu" && (
              <div className="grid grid-cols-2 gap-2 rounded-md border border-edge bg-surface-raised p-2 md:grid-cols-5">
                <StatCard label="Local GPU" value={telemetry?.gpu?.name ?? "probing…"} />
                <StatCard label="Utilization" value={telemetry?.gpu?.utilization_percent == null ? "—" : `${telemetry.gpu.utilization_percent}%`} />
                <StatCard label="VRAM free" value={telemetry?.gpu?.memory_free_mib == null ? "—" : `${telemetry.gpu.memory_free_mib} MiB`} />
                <StatCard label="Temperature" value={telemetry?.gpu?.temperature_c == null ? "—" : `${telemetry.gpu.temperature_c} °C`} />
                <StatCard label="System RAM free" value={telemetry?.system_memory?.available_gib == null ? "—" : `${telemetry.system_memory.available_gib.toFixed(2)} GiB`} />
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <Button
                variant="secondary"
                data-ui-action="evolution_plan_cycle"
                data-ui-skip-action="true"
                onClick={() => void onCycle(false)}
                disabled={busy !== "" || !engineTask}
              >
                {busy === "cycle" && !armApproval ? "Planning…" : "Plan cycle (no training)"}
              </Button>
              {!armApproval ? (
                <Button variant="primary" data-ui-action="evolution_arm_approval" data-ui-skip-action="true" onClick={() => setArmApproval(true)} disabled={busy !== "" || !engineTask || !approvalReceipt}>
                  {approvalReceipt ? "Approve exact plan & run…" : "Plan first to enable approval"}
                </Button>
              ) : (
                <>
                  <Button variant="primary" data-ui-action="evolution_confirm_run" data-ui-skip-action="true" onClick={() => void onCycle(true)} disabled={busy !== ""}>
                    {busy === "cycle" ? "Launching…" : `Confirm: run real training (${runner})`}
                  </Button>
                  <Button variant="secondary" data-ui-action="evolution_cancel_approval" data-ui-skip-action="true" onClick={() => setArmApproval(false)} disabled={busy !== ""}>
                    Cancel
                  </Button>
                </>
              )}
            </div>
            {armApproval && (
              <p className="text-[11px] leading-relaxed text-warning-text">
                Confirming launches real training on the {runner} runner for{" "}
                <span className={mono}>{engineTask}</span> ({iterations} iteration{iterations === 1 ? "" : "s"},{" "}
                {engine}/{searchMode}; caps {maxNodes} nodes, {maxTokens.toLocaleString()} tokens, {maxWallSeconds.toLocaleString()}s
                {maxCost.trim() ? `, $${maxCost}` : ""}). Plan SHA: {approvalReceipt?.plan_sha256.slice(0, 12) ?? "—"}…. Kaggle submission stays disabled regardless.
              </p>
            )}
            {cycle && (
              <div className="space-y-1 rounded-md border border-edge bg-surface-raised p-2">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={cycleTone(cycle.stage)}>{cycle.stage}</StatusBadge>
                  <StatusBadge tone={cycle.approved ? "amber" : "slate"}>
                    {cycle.approved ? "approved" : "plan only"}
                  </StatusBadge>
                  {cycle.training?.search_mode && <StatusBadge tone="blue">{cycle.training.search_mode}</StatusBadge>}
                  {cycle.training?.run_id && <span className={mono}>run {cycle.training.run_id}</span>}
                  {cycle.training?.best_score != null && (
                    <span className={mono}>best {fmtScore(cycle.training.best_score)}</span>
                  )}
                </div>
                {(cycle.next_action || cycle.reason) && (
                  <p className="text-[11px] text-ink-secondary">{cycle.next_action ?? cycle.reason}</p>
                )}
                {cycle.claim_boundary && (
                  <p className="text-[11px] leading-relaxed text-ink-secondary">{cycle.claim_boundary}</p>
                )}
                {cycle.error && <p className="text-[11px] text-danger-text">{cycle.error}</p>}
                {(Object.keys(resourceGate).length > 0 || Object.keys(independentReview).length > 0) && (
                  <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
                    <StatCard label="Resource Gate" value={String(resourceGate.status ?? "—")} />
                    <StatCard label="GPU" value={String(resourceGpu.name ?? "—")} />
                    <StatCard label="Reviewer" value={String(independentReview.status ?? "—")} />
                    <StatCard label="Claim Audit" value={String(claimAudit.status ?? "—")} />
                    <StatCard label="PR-AUC" value={fmtScore(numberValue(reviewMetrics.pr_auc))} />
                    <StatCard label="F1" value={fmtScore(numberValue(reviewMetrics.f1))} />
                    <StatCard label="Recall" value={fmtScore(numberValue(reviewMetrics.recall))} />
                    <StatCard label="Brier / calibration" value={fmtScore(numberValue(reviewMetrics.brier_score))} />
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatCard label="Stage" value={state?.current_stage ?? "—"} />
            <StatCard
              label={`Best ${state?.best_so_far?.metric ?? graph?.metric_name ?? "score"}`}
              value={fmtScore(state?.best_so_far?.cv_score)}
            />
            <StatCard label="Graph nodes" value={String(state?.search_graph_summary?.node_count ?? graph?.node_count ?? 0)} />
            <StatCard label="Memory hits" value={String(state?.memory_hits ?? memory?.record_count ?? 0)} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone={hasRun ? "green" : "slate"} >{hasRun ? "Has run" : "No run yet"}</StatusBadge>
            <StatusBadge
              tone={state?.official_submit_allowed ? "amber" : "slate"}
              >{state?.official_submit_allowed ? "Official submit ON" : "Official submit disabled"}</StatusBadge>
            {state?.search_graph_summary?.global_stagnation && <StatusBadge tone="amber">Global stagnation</StatusBadge>}
            {(state?.risk_flags ?? []).map((flag) => (
              <StatusBadge key={flag} tone="amber" >{flag}</StatusBadge>
            ))}
          </div>
          {state?.claim_boundary && (
            <p className="text-[11px] leading-relaxed text-ink-secondary">{state.claim_boundary}</p>
          )}
        </CardContent>
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Search graph</CardTitle>
            <CardDescription>
              {graph && graph.node_count > 0
                ? `${graph.node_count} experiment nodes, ${graph.edges?.length ?? 0} edges. Top: ${(graph.top_candidates ?? []).slice(0, 3).join(", ") || "—"}`
                : "No evolution nodes yet. Run a dry-run step to record the first planned node."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {nodesByScore.length === 0 ? (
              <EmptyHint text="The search graph is empty. Nothing is fabricated — plan and step to populate it." />
            ) : (
              <div className="max-h-[320px] space-y-2 overflow-y-auto pr-1">
                {nodesByScore.map((node) => (
                  <div
                    key={node.exp_id}
                    className={cnEvolution(
                      "rounded-md border px-3 py-2",
                      node.exp_id === graph?.best_exp_id
                        ? "border-success/50 bg-success-light"
                        : "border-edge bg-surface-sunken"
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className={cnEvolution(mono, "text-ink")}>
                        {node.exp_id}
                        {node.parent_id ? ` ← ${node.parent_id}` : " (root)"}
                      </span>
                      <span className={cnEvolution(mono, "text-ink-secondary")}>{fmtScore(node.cv_score)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <StatusBadge tone="slate" >{node.branch_type || "Base"}</StatusBadge>
                      {node.promoted && <StatusBadge tone="green">promoted</StatusBadge>}
                      {(node.risk_flags ?? []).map((flag) => (
                        <StatusBadge key={`${node.exp_id}-${flag}`} tone="amber" >{flag}</StatusBadge>
                      ))}
                    </div>
                    {node.implementation_summary && (
                      <p className="mt-1 text-[11px] leading-relaxed text-ink-secondary">{node.implementation_summary}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Retrospective memory</CardTitle>
            <CardDescription>
              Shared cross-task store ({memory?.record_count ?? 0} records) at{" "}
              <span className={mono}>{memory?.memory_store ?? "experiments/evolution/retrospective_memory.json"}</span>.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {(memory?.memory?.length ?? 0) === 0 ? (
              <EmptyHint text="No retrospective memory records for this task type yet." />
            ) : (
              <div className="max-h-[320px] space-y-2 overflow-y-auto pr-1">
                {memory?.memory.slice(0, 12).map((rec) => (
                  <div key={rec.memory_id} className="rounded-md border border-edge bg-surface-sunken px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className={cnEvolution(mono, "text-ink")}>{rec.method || rec.memory_id}</span>
                      <span className={cnEvolution(mono, "text-ink-secondary")}>
                        Δ {rec.metric_delta == null ? "—" : rec.metric_delta.toFixed(4)}
                      </span>
                    </div>
                    {rec.reusable_strategy && (
                      <p className="mt-1 text-[11px] leading-relaxed text-success-text">✓ {rec.reusable_strategy}</p>
                    )}
                    {rec.failure_pattern && (
                      <p className="mt-1 text-[11px] leading-relaxed text-warning-text">✗ {rec.failure_pattern}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>Experience-Guided MCGS · 多轮进化证据</CardTitle>
            <CardDescription>
              只读展示真实落盘的 Experience Board、父子 lineage、选择效用、检索与预算。分数来源限定为 public validation；
              A/B 存在时明确标为本地 MLE 复核。
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusBadge tone={experience?.present ? "green" : "slate"}>{experience?.present ? "evidence bound" : "no bound run"}</StatusBadge>
            <StatusBadge tone={experience?.search_mode === "experience_mcgs_v1" ? "blue" : "slate"}>{experience?.search_mode ?? "legacy_uct"}</StatusBadge>
            <StatusBadge tone={experience?.integrity.status === "verified" ? "green" : experience?.integrity.status === "failed" ? "red" : "slate"}>
              integrity {experience?.integrity.status ?? "not_present"}
            </StatusBadge>
            {experience?.run_id && <span className={cnEvolution(mono, "max-w-[280px] truncate text-ink-secondary")} title={experience.run_id}>run {experience.run_id}</span>}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
            <StatCard label="Cards" value={String(experienceCards.length)} />
            {(["Draft", "Improve", "Debug", "Crossover"] as const).map((operator) => (
              <StatCard key={operator} label={operator} value={String(experience?.operator_counts[operator] ?? 0)} />
            ))}
            <StatCard label="Prompt tokens" value={fmtCompact(experience?.budget?.prompt_tokens, 0)} />
            <StatCard label="Runtime" value={experience?.budget ? `${fmtCompact(experience.budget.wall_seconds)}s` : "—"} />
            <StatCard label="Cost" value={experience?.budget ? `$${fmtCompact(experience.budget.estimated_cost_usd, 4)}` : "—"} />
          </div>

          <section aria-labelledby="evolution-operator-timeline">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 id="evolution-operator-timeline" className="text-[12px] font-semibold text-ink">四操作时间线</h3>
              {experience?.board_hash && <span className={cnEvolution(mono, "text-[10px] text-ink-muted")} title={experience.board_hash}>board {experience.board_hash.slice(0, 12)}…</span>}
            </div>
            {experienceCards.length === 0 ? (
              <div className="mt-2"><EmptyHint text="当前 Run 尚无 Experience Board；legacy_uct 不生成经验卡。" /></div>
            ) : (
              <ol className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                {experienceCards.slice(0, 16).map((card, index) => (
                  <li key={card.card_id} className="relative rounded-md border border-edge bg-surface-sunken p-3">
                    <div className="flex items-center justify-between gap-2">
                      <StatusBadge tone={card.operator === "Debug" ? "amber" : card.operator === "Crossover" ? "blue" : card.operator === "Improve" ? "green" : "slate"}>{card.operator}</StatusBadge>
                      <span className={cnEvolution(mono, "text-[10px] text-ink-muted")}>#{index + 1}</span>
                    </div>
                    <p className={cnEvolution(mono, "mt-2 truncate text-ink")} title={card.node_id}>{card.node_id}</p>
                    <p className="mt-1 truncate text-[10px] text-ink-secondary" title={card.parent_ids.join(", ") || "root"}>
                      lineage: {card.parent_ids.length ? card.parent_ids.join(" + ") : "root"}
                    </p>
                    <p className="mt-1 text-[10px] text-ink-secondary">public validation {fmtScore(card.public_validation_score)}</p>
                  </li>
                ))}
              </ol>
            )}
          </section>

          {experience?.integrity.status === "failed" && (
            <div role="alert" className="rounded-md border border-danger/40 bg-danger-light px-3 py-2 text-[11px] text-danger-text">
              Experience Board 校验失败，Card、lineage 与宣传 Claim 已 fail-closed。{experience.integrity.errors.join("；")}
            </div>
          )}

          {experience?.review_chain.status === "failed" && (
            <div role="alert" className="rounded-md border border-danger/40 bg-danger-light px-3 py-2 text-[11px] text-danger-text">
              Freeze → Independent Review → Claim Audit 证据链校验失败，审查详情与 Claim 边界已 fail-closed。
              {experience.review_chain.errors.slice(0, 4).join("；")}
              {experience.review_chain.errors.length > 4 ? `；另有 ${experience.review_chain.errors.length - 4} 项` : ""}
            </div>
          )}

          <section aria-labelledby="experience-lineage-graph">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 id="experience-lineage-graph" className="text-[12px] font-semibold text-ink">多轮进化 lineage 图</h3>
              <span className="text-[10px] text-ink-muted">实线/双亲边均来自已校验 Card parent_ids</span>
            </div>
            <div className="mt-2">
              <LineageGraph lineage={experience?.lineage ?? { nodes: [], edges: [] }} />
            </div>
          </section>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(320px,1fr)]">
            <section aria-labelledby="experience-board-cards">
              <h3 id="experience-board-cards" className="text-[12px] font-semibold text-ink">Experience Board / Card lineage</h3>
              <div className="mt-2 max-h-[460px] space-y-2 overflow-y-auto pr-1">
                {experienceCards.length === 0 ? (
                  <EmptyHint text="Experience cards will appear only after a real experience_mcgs_v1 run writes verified public-validation outcomes." />
                ) : experienceCards.map((card) => {
                  const score = selectionByCard.get(card.card_id);
                  const retrieved = retrievedCardIds.has(card.card_id);
                  return (
                    <details key={card.card_id} className="group rounded-md border border-edge bg-surface-raised p-3" open={card.card_id === latestTrace?.selected_card_id}>
                      <summary className="flex min-h-11 cursor-pointer list-none flex-wrap items-center gap-2 rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
                        <StatusBadge tone={card.status === "success" ? "green" : "amber"}>{card.status}</StatusBadge>
                        <StatusBadge tone={card.operator === "Debug" ? "amber" : card.operator === "Crossover" ? "blue" : "slate"}>{card.operator}</StatusBadge>
                        <span className={cnEvolution(mono, "font-semibold text-ink")}>{card.node_id}</span>
                        <span className="ml-auto text-[11px] text-ink-secondary">{fmtScore(card.public_validation_score)}</span>
                      </summary>
                      <div className="mt-2 space-y-2 border-t border-edge pt-2">
                        <p className="text-[11px] text-ink-secondary">parent: <span className={mono}>{card.parent_ids.join(" + ") || "root"}</span> · family: {card.method_family}</p>
                        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                          <StatCard label="q quality" value={score ? fmtCompact(score.quality, 4) : "—"} />
                          <StatCard label="g progress" value={score ? fmtCompact(score.progress, 4) : "—"} />
                          <StatCard label="n novelty" value={score ? fmtCompact(score.novelty, 4) : "—"} />
                          <StatCard label="exploration" value={score ? fmtCompact(score.exploration, 4) : "—"} />
                          <StatCard label="utility" value={score ? fmtCompact(score.utility, 4) : "—"} />
                        </div>
                        <div className="flex flex-wrap gap-2 text-[10px] text-ink-muted">
                          {retrieved && <StatusBadge tone="blue">retrieved in latest prompt</StatusBadge>}
                          <span className={mono}>tokens {card.cost.total_tokens}</span>
                          <span className={mono}>wall {fmtCompact(card.cost.wall_seconds)}s</span>
                          <span className={mono}>code {card.code_hash.slice(0, 12)}…</span>
                          {card.error_signature && <span className={mono}>error {card.error_signature}</span>}
                        </div>
                        {card.summary && <p className="text-[11px] leading-relaxed text-ink-secondary">{card.summary}</p>}
                      </div>
                    </details>
                  );
                })}
              </div>
            </section>

            <aside className="space-y-3" aria-label="Experience MCGS budget and review evidence">
              <div className="rounded-md border border-edge bg-surface-sunken p-3">
                <h3 className="text-[12px] font-semibold text-ink">Prompt memory / retrieval</h3>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <StatCard label="Cards used" value={latestRetrieval ? `${latestRetrieval.card_ids.length}/${latestRetrieval.max_cards}` : "—"} />
                  <StatCard label="Est. tokens" value={latestRetrieval ? `${latestRetrieval.estimated_tokens}/${latestRetrieval.max_tokens}` : "—"} />
                  <StatCard label="Truncated by" value={latestRetrieval?.truncated_by || "none"} />
                  <StatCard label="Cache" value={experience ? `${experience.cache.hits} hit · ${experience.cache.misses} miss · ${experience.cache.unknown} unknown` : "—"} />
                </div>
                {latestRetrieval && <p className="mt-2 text-[10px] text-ink-muted">operator {latestRetrieval.operator} · latest bundle {latestRetrieval.card_cache_hits ?? 0} hit / {latestRetrieval.card_cache_misses ?? 0} miss</p>}
                {experience && (
                  <p className={`mt-1 text-[10px] ${experience.cache.stats_consistent === false ? "text-danger-text" : "text-ink-muted"}`}>
                    source {experience.cache.source} · stats {experience.cache.stats_consistent === false ? "mismatch" : experience.cache.stats_consistent === true ? "verified" : "not cross-checked"}
                  </p>
                )}
              </div>

              <div className="rounded-md border border-edge bg-surface-sunken p-3">
                <h3 className="text-[12px] font-semibold text-ink">Budget ledger</h3>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <StatCard label="Nodes" value={experience?.budget ? `${experience.budget.nodes}/${experience.budget.max_nodes}` : "—"} />
                  <StatCard label="Total tokens" value={experience?.budget ? `${experience.budget.total_tokens.toLocaleString()}/${experience.budget.max_total_tokens.toLocaleString()}` : "—"} />
                  <StatCard label="Wall seconds" value={experience?.budget ? `${fmtCompact(experience.budget.wall_seconds)}/${fmtCompact(experience.budget.max_wall_seconds)}` : "—"} />
                  <StatCard label="GPU seconds" value={experience?.budget ? fmtCompact(experience.budget.gpu_seconds) : "—"} />
                </div>
                {experience?.budget?.terminal_reason && <p className="mt-2 text-[10px] font-medium text-warning-text">terminal: {experience.budget.terminal_reason}</p>}
              </div>

              <div className="rounded-md border border-edge bg-surface-sunken p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-[12px] font-semibold text-ink">审计证据链</h3>
                  <StatusBadge tone={evidenceTone(experience?.review_chain.status ?? "not_present")}>
                    chain {experience?.review_chain.status ?? "not_present"}
                  </StatusBadge>
                </div>
                <p className="mt-1 text-[10px] text-ink-muted">严格按同一 Run 根目录的 freeze → review → claim 顺序投影</p>
                <ol className="mt-3 space-y-2">
                  <li className="rounded-md border border-edge bg-surface-raised p-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[11px] font-semibold text-ink">1 · Candidate Freeze</span>
                      <StatusBadge tone={evidenceTone(experience?.review_chain.stages.candidate_freeze ?? "not_present")}>
                        {experience?.candidate_freeze.status ?? "not_present"}
                      </StatusBadge>
                    </div>
                    <p className="mt-1 text-[10px] text-ink-muted">来源：<span className={mono}>{experience?.candidate_freeze.artifact ?? "candidate-freeze.json (not present)"}</span></p>
                    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-ink-secondary">
                      <dt>Candidate</dt><dd className={mono}>{experience?.candidate_freeze.candidate_id ?? "—"}</dd>
                      <dt>Artifacts</dt><dd>{fmtCompact(experience?.candidate_freeze.artifact_count, 0)}</dd>
                      <dt>SHA-256</dt><dd className={mono} title={experience?.candidate_freeze.sha256 ?? ""}>{experience?.candidate_freeze.sha256?.slice(0, 12) ?? "—"}</dd>
                    </dl>
                  </li>

                  <li className="rounded-md border border-edge bg-surface-raised p-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[11px] font-semibold text-ink">2 · Independent Review</span>
                      <StatusBadge tone={evidenceTone(experience?.review_chain.stages.independent_review ?? "not_present")}>
                        {experience?.independent_review.status ?? "not_present"}
                      </StatusBadge>
                    </div>
                    <p className="mt-1 text-[10px] text-ink-muted">
                      来源：<span className={mono}>{experience?.independent_review.artifact ?? "independent-review.json (not present)"}</span>
                      {experience?.independent_review.review_source ? ` · ${experience.independent_review.review_source}` : ""}
                    </p>
                    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-ink-secondary">
                      <dt>Reviewer</dt><dd>{experience?.independent_review.reviewer ?? "—"}</dd>
                      <dt>Rows</dt><dd>{fmtCompact(experience?.independent_review.rows, 0)}</dd>
                      <dt>Metric</dt><dd>{experience?.independent_review.metric ?? "—"}</dd>
                      <dt>Score</dt><dd>{fmtScore(experience?.independent_review.score)}</dd>
                      <dt>SHA-256</dt><dd className={mono} title={experience?.independent_review.sha256 ?? ""}>{experience?.independent_review.sha256?.slice(0, 12) ?? "—"}</dd>
                    </dl>
                  </li>

                  <li className="rounded-md border border-edge bg-surface-raised p-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[11px] font-semibold text-ink">3 · Claim Audit</span>
                      <StatusBadge
                        tone={(experience?.claim_audit.unsupported_claim_count ?? 0) > 0
                          ? "amber"
                          : evidenceTone(experience?.review_chain.stages.claim_audit ?? "not_present")}
                      >
                        {experience?.claim_audit.status ?? "not_present"}
                      </StatusBadge>
                    </div>
                    <p className="mt-1 text-[10px] text-ink-muted">来源：<span className={mono}>{experience?.claim_audit.artifact ?? "claim-audit.json (not present)"}</span></p>
                    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-ink-secondary">
                      <dt>Supported</dt><dd>{fmtCompact(experience?.claim_audit.supported_claim_count, 0)}</dd>
                      <dt>Unsupported</dt><dd>{fmtCompact(experience?.claim_audit.unsupported_claim_count, 0)}</dd>
                      <dt>SHA-256</dt><dd className={mono} title={experience?.claim_audit.sha256 ?? ""}>{experience?.claim_audit.sha256?.slice(0, 12) ?? "—"}</dd>
                    </dl>
                    {experience?.claim_audit.claim_boundary && (
                      <p className="mt-2 border-t border-edge pt-2 text-[10px] leading-relaxed text-ink-secondary">
                        Claim boundary：{experience.claim_audit.claim_boundary}
                      </p>
                    )}
                  </li>
                </ol>
              </div>

              <div className="rounded-md border border-edge bg-surface-sunken p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-[12px] font-semibold text-ink">Paired A/B evidence</h3>
                  <StatusBadge tone={experience?.paired_ab.present ? "blue" : "slate"}>A/B {experience?.paired_ab.status ?? "not_present"}</StatusBadge>
                </div>
                <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-ink-secondary">
                  <dt>Paired wins</dt><dd>{fmtCompact(experience?.paired_ab.paired_wins, 0)} / {fmtCompact(experience?.paired_ab.task_count, 0)}</dd>
                  <dt>Median Δ</dt><dd>{fmtCompact(experience?.paired_ab.median_paired_delta, 5)}</dd>
                  <dt>95% CI</dt><dd>[{fmtCompact(experience?.paired_ab.ci_low, 5)}, {fmtCompact(experience?.paired_ab.ci_high, 5)}]</dd>
                  <dt>Source</dt><dd>{experience?.paired_ab.present ? "本地 MLE grader" : "—"}</dd>
                </dl>
              </div>
            </aside>
          </div>
          {experience?.claim_boundary && <p className="text-[11px] leading-relaxed text-ink-muted">{experience.claim_boundary}</p>}
        </CardContent>
      </Card>

      {lastStep && (
        <Card>
          <CardHeader>
            <CardTitle>Last step result</CardTitle>
            <CardDescription>{lastStep.reason ?? lastStep.decision}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge tone={lastStep.dry_run ? "slate" : "amber"} >{lastStep.dry_run ? "dry_run" : "real"}</StatusBadge>
              <StatusBadge tone="slate" >{`decision: ${lastStep.decision}`}</StatusBadge>
              <StatusBadge tone="slate" >{`gate: ${lastStep.gate_status}`}</StatusBadge>
            </div>
            {lastStep.next_action && <p className="text-[11px] text-ink-secondary">Next: {lastStep.next_action}</p>}
            {(lastStep.artifacts ?? []).length > 0 && (
              <ul className="space-y-1">
                {lastStep.artifacts?.map((path) => (
                  <li key={path} className={cnEvolution(mono, "text-[11px] text-ink-secondary")}>
                    {path}
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-edge bg-surface-sunken px-3 py-2">
      <p className="text-[11px] uppercase text-ink-secondary">{label}</p>
      <p className={cnEvolution(mono, "mt-1 text-ink")}>{value}</p>
    </div>
  );
}

function EmptyHint({ text }: { text: string }) {
  return <p className="rounded-md border border-dashed border-edge bg-surface-sunken px-3 py-6 text-center text-[12px] text-ink-secondary">{text}</p>;
}

function LineageGraph({ lineage }: { lineage: EvolutionExperienceResponse["lineage"] }) {
  if (!lineage.nodes.length) return <EmptyHint text="尚无通过完整性校验的 lineage 节点。" />;
  const nodes = lineage.nodes.slice(0, 24);
  const nodeIndex = new Map(nodes.map((node, index) => [node.node_id, index]));
  const lanes: Record<string, number> = { Draft: 34, Improve: 88, Debug: 142, Crossover: 196 };
  const width = Math.max(760, nodes.length * 126 + 80);
  const point = (nodeId: string) => {
    const index = nodeIndex.get(nodeId) ?? 0;
    const node = nodes[index];
    return { x: 72 + index * 126, y: lanes[node.operator] ?? 88 };
  };
  return (
    <div className="overflow-x-auto rounded-md border border-edge bg-surface-sunken" tabIndex={0} aria-label="Experience MCGS 多轮进化 lineage 图">
      <svg width={width} height={236} role="img" aria-label={`${nodes.length} 个进化节点与 ${lineage.edges.length} 条父子边`}>
        {(["Draft", "Improve", "Debug", "Crossover"] as const).map((operator) => (
          <g key={operator}>
            <text x={8} y={lanes[operator] + 4} className="fill-ink-muted text-[10px]">{operator}</text>
            <line x1={62} x2={width - 16} y1={lanes[operator]} y2={lanes[operator]} className="stroke-edge" strokeDasharray="3 5" />
          </g>
        ))}
        {lineage.edges.filter((edge) => nodeIndex.has(edge.parent_node_id) && nodeIndex.has(edge.child_node_id)).map((edge, index) => {
          const source = point(edge.parent_node_id);
          const target = point(edge.child_node_id);
          return <path key={`${edge.parent_node_id}-${edge.child_node_id}-${index}`} d={`M ${source.x + 24} ${source.y} C ${source.x + 58} ${source.y}, ${target.x - 58} ${target.y}, ${target.x - 24} ${target.y}`} fill="none" className="stroke-accent/70" strokeWidth={1.5} />;
        })}
        {nodes.map((node) => {
          const { x, y } = point(node.node_id);
          const tone = node.operator === "Debug" ? "fill-warning" : node.operator === "Crossover" ? "fill-accent" : node.operator === "Improve" ? "fill-success" : "fill-ink-muted";
          return (
            <g key={node.node_id} transform={`translate(${x},${y})`}>
              <circle r={22} className={`${tone} stroke-surface-raised`} strokeWidth={3} />
              <text textAnchor="middle" y={3} className="fill-surface text-[9px] font-semibold">{node.node_id.replace(/^EXP/, "")}</text>
              <text textAnchor="middle" y={35} className="fill-ink-secondary text-[9px]">{node.method_family.slice(0, 16)}</text>
              <title>{`${node.node_id} · ${node.operator} · public validation ${fmtScore(node.score)}`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
