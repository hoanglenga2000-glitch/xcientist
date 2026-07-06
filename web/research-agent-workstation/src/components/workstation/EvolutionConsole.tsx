"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import { cn } from "@/lib/utils";
import type {
  EvolutionConfigSummary,
  EvolutionCycleResponse,
  EvolutionEngine,
  EvolutionGraphResponse,
  EvolutionMemoryResponse,
  EvolutionRunner,
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

export function EvolutionConsole({ selectedTask, refreshSummary }: Props) {
  const [state, setState] = useState<EvolutionStateResponse | null>(null);
  const [graph, setGraph] = useState<EvolutionGraphResponse | null>(null);
  const [memory, setMemory] = useState<EvolutionMemoryResponse | null>(null);
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
  const [armApproval, setArmApproval] = useState<boolean>(false);
  const [cycle, setCycle] = useState<EvolutionCycleResponse | null>(null);

  const activeConfig = useMemo(
    () => configs.find((c) => c.task_id === engineTask) ?? null,
    [configs, engineTask]
  );

  const load = useCallback(async () => {
    if (!selectedTask) return;
    setLoading(true);
    setError("");
    try {
      const [s, g, m] = await Promise.all([
        api.getEvolutionState(selectedTask),
        api.getEvolutionGraph(selectedTask),
        api.getEvolutionMemory(selectedTask)
      ]);
      setState(s);
      setGraph(g);
      setMemory(m);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load evolution state");
    } finally {
      setLoading(false);
    }
  }, [selectedTask]);

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
  }, [engineTask, engine, runner, iterations, useMcgs]);

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
          approve
        });
        setCycle(result);
        setArmApproval(false);
        setMessage(
          approve
            ? `Cycle ${result.stage}: ${result.next_action ?? result.reason ?? "see result below"}.`
            : `Plan ready (${result.stage}). Review below, then Approve & run to launch real training.`
        );
        await load();
        await refreshSummary?.();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Cycle failed");
      } finally {
        setBusy("");
      }
    },
    [engineTask, selectedTask, engine, runner, iterations, useMcgs, load, refreshSummary]
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

  const hasRun = Boolean(state?.has_run ?? graph?.has_run);
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-4">
          <div>
            <CardTitle>Evolution Engine</CardTitle>
            <CardDescription>
              Self-evolving search brain for <span className={mono}>{selectedTask || "—"}</span>. Plans expansions,
              records a real search graph, and reuses retrospective memory. Training runs through the workstation
              orchestrator — this console never bypasses the gates.
            </CardDescription>
          </div>
          <div className="flex shrink-0 gap-2">
            <Button variant="secondary" onClick={() => void load()} disabled={loading || !selectedTask}>
              {loading ? "Loading…" : "Refresh"}
            </Button>
            <Button variant="secondary" onClick={() => void onPlan()} disabled={busy !== "" || !selectedTask}>
              {busy === "plan" ? "Planning…" : "Plan next"}
            </Button>
            <Button variant="primary" onClick={() => void onStep()} disabled={busy !== "" || !selectedTask}>
              {busy === "step" ? "Stepping…" : "Dry-run step"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {error && (
            <p className={cnEvolution("rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-red-300", mono)}>
              {error}
            </p>
          )}
          {message && (
            <p className={cnEvolution("rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-emerald-300", mono)}>
              {message}
            </p>
          )}
          <div className="rounded-lg border border-slate-700/70 bg-slate-900/40 p-3 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-[12px] font-medium text-slate-200">Closed loop (plan → approve → train → ingest)</p>
              <StatusBadge tone="slate">Kaggle submit disabled</StatusBadge>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <label className="flex flex-col gap-1 text-[11px] text-slate-400">
                Task config
                <select
                  className={cnEvolution("rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-slate-100", mono)}
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
              <label className="flex flex-col gap-1 text-[11px] text-slate-400">
                Engine
                <select
                  className={cnEvolution("rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-slate-100", mono)}
                  value={engine}
                  disabled={busy !== ""}
                  onChange={(e) => setEngine(e.target.value as EvolutionEngine)}
                >
                  <option value="research_os">research_os</option>
                  <option value="legacy">legacy</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] text-slate-400">
                Runner
                <select
                  className={cnEvolution("rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-slate-100", mono)}
                  value={runner}
                  disabled={busy !== ""}
                  onChange={(e) => setRunner(e.target.value as EvolutionRunner)}
                >
                  <option value="gpu">gpu</option>
                  <option value="local">local</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-[11px] text-slate-400">
                Iterations
                <input
                  type="number"
                  min={1}
                  max={50}
                  className={cnEvolution("rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-slate-100", mono)}
                  value={iterations}
                  disabled={busy !== ""}
                  onChange={(e) => setIterations(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
                />
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-[12px] text-slate-300">
                <input
                  type="checkbox"
                  checked={useMcgs}
                  disabled={busy !== ""}
                  onChange={(e) => setUseMcgs(e.target.checked)}
                />
                MCGS search
              </label>
              {activeConfig && (
                <p className="text-[11px] text-slate-400">
                  {activeConfig.modality ?? "?"}/{activeConfig.task_type ?? "?"} · {activeConfig.metric ?? "?"} (
                  {activeConfig.metric_direction ?? "?"}) · n_train={activeConfig.n_train ?? "?"} ·{" "}
                  {activeConfig.has_gpu_data_dir ? "GPU data ✓" : "no GPU data"}
                </p>
              )}
            </div>
            {runner === "gpu" && activeConfig && !activeConfig.has_gpu_data_dir && (
              <p className={cnEvolution("rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-amber-300", mono)}>
                This config has no gpu_data_dir — a GPU run will likely fail. Pick a GPU-ready config or switch runner to local.
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <Button
                variant="secondary"
                onClick={() => void onCycle(false)}
                disabled={busy !== "" || !engineTask}
              >
                {busy === "cycle" && !armApproval ? "Planning…" : "Plan cycle (no training)"}
              </Button>
              {!armApproval ? (
                <Button variant="primary" onClick={() => setArmApproval(true)} disabled={busy !== "" || !engineTask}>
                  Approve & run…
                </Button>
              ) : (
                <>
                  <Button variant="primary" onClick={() => void onCycle(true)} disabled={busy !== ""}>
                    {busy === "cycle" ? "Launching…" : `Confirm: run real training (${runner})`}
                  </Button>
                  <Button variant="secondary" onClick={() => setArmApproval(false)} disabled={busy !== ""}>
                    Cancel
                  </Button>
                </>
              )}
            </div>
            {armApproval && (
              <p className="text-[11px] leading-relaxed text-amber-300/90">
                Confirming launches real training on the {runner} runner for{" "}
                <span className={mono}>{engineTask}</span> ({iterations} iteration{iterations === 1 ? "" : "s"},{" "}
                {engine}). Kaggle submission stays disabled regardless.
              </p>
            )}
            {cycle && (
              <div className="rounded-md border border-slate-700/70 bg-slate-950/50 p-2 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={cycleTone(cycle.stage)}>{cycle.stage}</StatusBadge>
                  <StatusBadge tone={cycle.approved ? "amber" : "slate"}>
                    {cycle.approved ? "approved" : "plan only"}
                  </StatusBadge>
                  {cycle.training?.run_id && <span className={mono}>run {cycle.training.run_id}</span>}
                  {cycle.training?.best_score != null && (
                    <span className={mono}>best {fmtScore(cycle.training.best_score)}</span>
                  )}
                </div>
                {(cycle.next_action || cycle.reason) && (
                  <p className="text-[11px] text-slate-300">{cycle.next_action ?? cycle.reason}</p>
                )}
                {cycle.claim_boundary && (
                  <p className="text-[11px] leading-relaxed text-slate-400">{cycle.claim_boundary}</p>
                )}
                {cycle.error && <p className="text-[11px] text-red-300">{cycle.error}</p>}
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
            <p className="text-[11px] leading-relaxed text-slate-400">{state.claim_boundary}</p>
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
                        ? "border-emerald-500/50 bg-emerald-500/10"
                        : "border-slate-700/60 bg-slate-900/40"
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className={cnEvolution(mono, "text-slate-200")}>
                        {node.exp_id}
                        {node.parent_id ? ` ← ${node.parent_id}` : " (root)"}
                      </span>
                      <span className={cnEvolution(mono, "text-slate-300")}>{fmtScore(node.cv_score)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <StatusBadge tone="slate" >{node.branch_type || "Base"}</StatusBadge>
                      {node.promoted && <StatusBadge tone="green">promoted</StatusBadge>}
                      {(node.risk_flags ?? []).map((flag) => (
                        <StatusBadge key={`${node.exp_id}-${flag}`} tone="amber" >{flag}</StatusBadge>
                      ))}
                    </div>
                    {node.implementation_summary && (
                      <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{node.implementation_summary}</p>
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
                  <div key={rec.memory_id} className="rounded-md border border-slate-700/60 bg-slate-900/40 px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className={cnEvolution(mono, "text-slate-200")}>{rec.method || rec.memory_id}</span>
                      <span className={cnEvolution(mono, "text-slate-300")}>
                        Δ {rec.metric_delta == null ? "—" : rec.metric_delta.toFixed(4)}
                      </span>
                    </div>
                    {rec.reusable_strategy && (
                      <p className="mt-1 text-[11px] leading-relaxed text-emerald-300/90">✓ {rec.reusable_strategy}</p>
                    )}
                    {rec.failure_pattern && (
                      <p className="mt-1 text-[11px] leading-relaxed text-amber-300/90">✗ {rec.failure_pattern}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

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
            {lastStep.next_action && <p className="text-[11px] text-slate-400">Next: {lastStep.next_action}</p>}
            {(lastStep.artifacts ?? []).length > 0 && (
              <ul className="space-y-1">
                {lastStep.artifacts?.map((path) => (
                  <li key={path} className={cnEvolution(mono, "text-[11px] text-slate-400")}>
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
    <div className="rounded-md border border-slate-700/60 bg-slate-900/40 px-3 py-2">
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className={cnEvolution(mono, "mt-1 text-slate-100")}>{value}</p>
    </div>
  );
}

function EmptyHint({ text }: { text: string }) {
  return <p className="rounded-md border border-dashed border-slate-700/60 px-3 py-6 text-center text-[12px] text-slate-500">{text}</p>;
}
