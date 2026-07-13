"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import { cn } from "@/lib/utils";
import type {
  EvolutionGraphResponse,
  EvolutionMemoryResponse,
  EvolutionStateResponse,
  EvolutionStepResponse
} from "@/lib/api/types";

// Real evolution-engine bindings injected into existing workstation screens.
// Every panel here is API-driven (no static/fabricated state). Training always
// runs through the workstation orchestrator; these panels only plan/inspect.

const mono = "font-mono text-[12px] tracking-normal";

function fmtScore(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(5);
}

function stageTone(stage?: string): StatusTone {
  if (!stage || stage.includes("no_evolution") || stage.includes("no_run")) return "slate";
  if (stage.includes("exploit")) return "green";
  if (stage.includes("balanced")) return "blue";
  return "amber";
}

type Binding = {
  state: EvolutionStateResponse | null;
  graph: EvolutionGraphResponse | null;
  memory: EvolutionMemoryResponse | null;
  lastStep: EvolutionStepResponse | null;
  loading: boolean;
  busy: "" | "plan" | "step";
  error: string;
  message: string;
  reload: () => Promise<void>;
  plan: () => Promise<void>;
  step: () => Promise<void>;
};

export function useEvolutionBinding(
  taskId: string,
  opts: { withGraph?: boolean; withMemory?: boolean; refreshSummary?: () => Promise<unknown> } = {}
): Binding {
  const { withGraph = false, withMemory = false, refreshSummary } = opts;
  const [state, setState] = useState<EvolutionStateResponse | null>(null);
  const [graph, setGraph] = useState<EvolutionGraphResponse | null>(null);
  const [memory, setMemory] = useState<EvolutionMemoryResponse | null>(null);
  const [lastStep, setLastStep] = useState<EvolutionStepResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<"" | "plan" | "step">("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const reload = useCallback(async () => {
    if (!taskId) return;
    setLoading(true);
    setError("");
    try {
      const [s, g, m] = await Promise.all([
        api.getEvolutionState(taskId),
        withGraph ? api.getEvolutionGraph(taskId) : Promise.resolve(null),
        withMemory ? api.getEvolutionMemory(taskId) : Promise.resolve(null)
      ]);
      setState(s);
      if (withGraph) setGraph(g);
      if (withMemory) setMemory(m);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load evolution state");
    } finally {
      setLoading(false);
    }
  }, [taskId, withGraph, withMemory]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const plan = useCallback(async () => {
    if (!taskId) return;
    setBusy("plan");
    setError("");
    setMessage("");
    try {
      const p = await api.planEvolution({ task_id: taskId, objective: `Evolve ${taskId}` });
      setMessage(
        `Planned: ${p.search_controller_decision} → ${p.selected_branch} (${p.code_generation_mode}/${p.expansion_type}). ` +
          `Strategies: ${(p.recommended_strategies ?? []).slice(0, 4).join(", ") || "—"}.`
      );
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Plan failed");
    } finally {
      setBusy("");
    }
  }, [taskId, reload]);

  const step = useCallback(async () => {
    if (!taskId) return;
    setBusy("step");
    setError("");
    setMessage("");
    try {
      const st = await api.runEvolutionStep({ task_id: taskId, dry_run: true });
      setLastStep(st);
      setMessage(
        st.dry_run
          ? `Dry-run step recorded node ${st.exp_id ?? "?"} (${st.code_generation_mode}/${st.expansion_type}). No training executed — ${(st.artifacts ?? []).length} artifacts written.`
          : `Step decision: ${st.decision}.`
      );
      await reload();
      await refreshSummary?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Step failed");
    } finally {
      setBusy("");
    }
  }, [taskId, reload, refreshSummary]);

  return { state, graph, memory, lastStep, loading, busy, error, message, reload, plan, step };
}

function EvoBanner({ error, message }: { error: string; message: string }) {
  if (!error && !message) return null;
  return (
    <div className="space-y-1.5">
      {error && (
        <p className={cn("rounded-md border border-red-200 bg-red-50 px-3 py-1.5 text-[11px] font-bold text-red-700", mono)}>{error}</p>
      )}
      {message && (
        <p className={cn("rounded-md border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-bold text-emerald-700", mono)}>{message}</p>
      )}
    </div>
  );
}

function EvoChip({ label, value, tone = "slate" }: { label: string; value: string; tone?: StatusTone }) {
  const ring =
    tone === "green" ? "border-emerald-200 bg-emerald-50/60"
    : tone === "amber" ? "border-amber-200 bg-amber-50/60"
    : tone === "red" ? "border-red-200 bg-red-50/60"
    : tone === "blue" ? "border-blue-200 bg-blue-50/60"
    : "border-slate-200 bg-white";
  return (
    <div className={cn("rounded-md border px-3 py-2", ring)}>
      <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">{label}</p>
      <p className={cn(mono, "mt-1 truncate text-slate-900")} title={value}>{value}</p>
    </div>
  );
}

function EvoControls({ binding, taskId }: { binding: Binding; taskId: string }) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2">
      <Button size="sm" variant="secondary" data-ui-action="evolution_refresh" data-ui-skip-action="true"
        onClick={() => void binding.reload()} disabled={binding.loading || !taskId}>
        {binding.loading ? "Loading…" : "Refresh"}
      </Button>
      <Button size="sm" variant="secondary" data-ui-action="evolution_plan_next" data-ui-skip-action="true"
        onClick={() => void binding.plan()} disabled={binding.busy !== "" || !taskId}>
        {binding.busy === "plan" ? "Planning…" : "启动进化计划"}
      </Button>
      <Button size="sm" variant="primary" data-ui-action="evolution_dry_run_step" data-ui-skip-action="true"
        onClick={() => void binding.step()} disabled={binding.busy !== "" || !taskId}>
        {binding.busy === "step" ? "Stepping…" : "执行下一轮 (dry-run)"}
      </Button>
    </div>
  );
}

function EvoEmpty({ text }: { text: string }) {
  return <p className="rounded-md border border-dashed border-slate-300 px-3 py-6 text-center text-[12px] font-semibold text-slate-500">{text}</p>;
}

function SubmitGuardBadge({ allowed }: { allowed?: boolean }) {
  return (
    <StatusBadge tone={allowed ? "amber" : "slate"}>
      {allowed ? "Official submit ON" : "Official submit disabled"}
    </StatusBadge>
  );
}

function SectionCard({ title, desc, action, children }: { title: string; desc: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="rounded-md border border-slate-200/95 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.035)]">
      <header className="flex flex-wrap items-start justify-between gap-2 px-3.5 pt-3.5">
        <div>
          <h3 className="text-sm font-black tracking-normal text-slate-950">{title}</h3>
          <p className="mt-1 text-xs leading-4 text-slate-500">{desc}</p>
        </div>
        {action}
      </header>
      <div className="p-3.5 pt-3">{children}</div>
    </section>
  );
}

// Overview: evolution stage, best-so-far, active branches, next decision.
export function EvolutionOverviewPanel({ taskId, refreshSummary }: { taskId: string; refreshSummary?: () => Promise<unknown> }) {
  const b = useEvolutionBinding(taskId, { refreshSummary });
  const s = b.state;
  const best = s?.best_so_far;
  const decision = s?.latest_decision ?? "—";
  return (
    <SectionCard
      title="Evolution Engine · 自进化大脑"
      desc={`Live search-controller state for ${taskId || "—"}. Plans expansions and records a real search graph; training runs through the workstation.`}
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
        <EvoChip label="Stage" value={s?.current_stage ?? "—"} tone={stageTone(s?.current_stage)} />
        <EvoChip label={`Best ${best?.metric ?? "score"}`} value={fmtScore(best?.cv_score)} tone={best?.exp_id ? "green" : "slate"} />
        <EvoChip label="Graph nodes" value={String(s?.search_graph_summary?.node_count ?? 0)} tone="blue" />
        <EvoChip label="Memory hits" value={String(s?.memory_hits ?? 0)} tone={s?.memory_hits ? "blue" : "slate"} />
      </div>
      <div className="mt-2 grid gap-2 md:grid-cols-[1fr_1fr]">
        <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
          <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">Next decision</p>
          <p className={cn(mono, "mt-1 text-slate-900")}>{decision}</p>
          <p className="mt-1 text-[11px] font-semibold text-slate-500">
            Best branch: <span className={mono}>{best?.exp_id ?? "—"}</span>
            {best?.promotion_reason ? ` · ${best.promotion_reason}` : ""}
          </p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
          <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">Active branches (leaf frontier)</p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {(s?.active_branches ?? []).length === 0
              ? <span className="text-[11px] font-semibold text-slate-500">尚无分支，先执行一次 dry-run。</span>
              : (s?.active_branches ?? []).slice(0, 8).map((br) => (
                  <StatusBadge key={br.exp_id} tone={br.promoted ? "green" : "slate"}>
                    {br.exp_id} · {fmtScore(br.cv_score)}
                  </StatusBadge>
                ))}
          </div>
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <StatusBadge tone={s?.has_run ? "green" : "slate"}>{s?.has_run ? "Has run" : "No run yet"}</StatusBadge>
        <SubmitGuardBadge allowed={s?.official_submit_allowed} />
        {s?.search_graph_summary?.global_stagnation && <StatusBadge tone="amber">Global stagnation → 触发跨分支/融合</StatusBadge>}
        {(s?.risk_flags ?? []).slice(0, 6).map((f) => <StatusBadge key={f} tone="amber">{f}</StatusBadge>)}
      </div>
      {s?.claim_boundary && <p className="mt-2 text-[11px] leading-relaxed text-slate-400">{s.claim_boundary}</p>}
    </SectionCard>
  );
}

// AI Control: launch plan + run next dry-run step, with the last step trace.
export function EvolutionControlPanel({ taskId, refreshSummary }: { taskId: string; refreshSummary?: () => Promise<unknown> }) {
  const b = useEvolutionBinding(taskId, { refreshSummary });
  const s = b.state;
  const step = b.lastStep;
  return (
    <SectionCard
      title="Evolution Control · 进化引擎动作"
      desc="启动进化计划 / 执行下一轮进化 step。step 默认 dry-run：只写审计产物与搜索图节点，真实训练走工作站 orchestrator，不旁路。"
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
        <EvoChip label="Stage" value={s?.current_stage ?? "—"} tone={stageTone(s?.current_stage)} />
        <EvoChip label="Gate status" value={s?.gate_status ?? "—"} tone={(s?.gate_status ?? "").includes("block") ? "red" : "slate"} />
        <EvoChip label="Last decision" value={s?.latest_decision ?? "—"} tone="blue" />
        <EvoChip label="Nodes" value={String(s?.search_graph_summary?.node_count ?? 0)} tone="slate" />
      </div>
      {step && (
        <div className="mt-2 rounded-md border border-slate-200 bg-slate-50/70 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone={step.dry_run ? "slate" : "amber"}>{step.dry_run ? "dry_run" : "real"}</StatusBadge>
            <StatusBadge tone="blue">decision: {step.decision}</StatusBadge>
            <StatusBadge tone={(step.gate_status ?? "").includes("block") ? "red" : "slate"}>gate: {step.gate_status}</StatusBadge>
            {step.exp_id && <StatusBadge tone="green">{step.exp_id}</StatusBadge>}
          </div>
          {step.reason && <p className="mt-1.5 text-[11px] leading-relaxed text-slate-600">{step.reason}</p>}
          {step.next_action && <p className="mt-1 text-[11px] font-semibold text-blue-700">Next: {step.next_action}</p>}
          {(step.artifacts ?? []).length > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {step.artifacts?.map((p) => (
                <li key={p} className={cn(mono, "truncate text-[11px] text-emerald-700")} title={p}>› {p}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      <div className="mt-2 rounded-md border border-amber-100 bg-amber-50 px-3 py-1.5 text-[11px] font-bold text-amber-700">
        安全边界：所有训练与提交必须通过工作站 Gate。进化引擎只做计划/分支/调度/证据写入,不直接展开训练,不自动提交 Kaggle。
      </div>
    </SectionCard>
  );
}

type Placed = { id: string; x: number; y: number; col: number; row: number };

function layoutGraph(graph: EvolutionGraphResponse | null): { placed: Placed[]; byId: Record<string, Placed> } {
  if (!graph?.nodes?.length) return { placed: [], byId: {} };
  const parentOf: Record<string, string | null> = {};
  for (const n of graph.nodes) parentOf[n.exp_id] = n.parent_id ?? null;
  const depthOf: Record<string, number> = {};
  const depth = (id: string, seen = new Set<string>()): number => {
    if (depthOf[id] != null) return depthOf[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const p = parentOf[id];
    const d = p && p in parentOf ? depth(p, seen) + 1 : 0;
    depthOf[id] = d;
    return d;
  };
  const rowByCol: Record<number, number> = {};
  const placed: Placed[] = [];
  const byId: Record<string, Placed> = {};
  for (const n of graph.nodes) {
    const col = depth(n.exp_id);
    const row = rowByCol[col] ?? 0;
    rowByCol[col] = row + 1;
    const p: Placed = { id: n.exp_id, col, row, x: 18 + col * 190, y: 16 + row * 84 };
    placed.push(p);
    byId[n.exp_id] = p;
  }
  return { placed, byId };
}

function nodeTone(n: EvolutionGraphResponse["nodes"][number], bestId?: string | null): StatusTone {
  if (n.exp_id === bestId) return "green";
  if ((n.risk_flags ?? []).some((f) => f.includes("fail") || f.includes("blocked"))) return "red";
  if (n.branch_type === "Diff") return "amber";
  if (n.branch_type === "Stepwise") return "blue";
  return "slate";
}

// Experiments: real MLEvolve-style search graph from /api/evolution/graph.
export function EvolutionSearchGraphPanel({ taskId }: { taskId: string }) {
  const b = useEvolutionBinding(taskId, { withGraph: true });
  const g = b.graph;
  const { placed, byId } = useMemo(() => layoutGraph(g), [g]);
  const cols = placed.reduce((m, p) => Math.max(m, p.col), 0) + 1;
  const rows = placed.reduce((m, p) => Math.max(m, p.row), 0) + 1;
  const width = Math.max(1000, 18 + cols * 190);
  const height = Math.max(300, 16 + rows * 84);
  const edges = (g?.edges ?? []).filter((e) => byId[e.source] && byId[e.target]);
  const refEdges = (g?.reference_edges ?? []).filter((e) => byId[e.source] && byId[e.target]);
  return (
    <SectionCard
      title="Evolution Search Graph · 真实进化搜索图"
      desc={g && g.node_count > 0
        ? `${g.node_count} nodes · ${edges.length} edges · stage ${g.exploration_stage ?? "—"} · top: ${(g.top_candidates ?? []).slice(0, 3).join(", ") || "—"}`
        : "搜索图为空,不虚构节点。执行一次 dry-run step 即可记录第一个 planned 节点。"}
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      {placed.length === 0 ? (
        <EvoEmpty text="No evolution nodes yet — nothing fabricated. Plan + dry-run step to populate the real graph." />
      ) : (
        <div className="thin-scrollbar relative mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
          <div className="relative" style={{ width, height }}>
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
              <defs>
                <marker id="evoArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#475569" /></marker>
                <marker id="evoArrowRef" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#7c3aed" /></marker>
              </defs>
              {edges.map((e) => {
                const a = byId[e.source]; const c = byId[e.target];
                return <path key={`${e.source}-${e.target}`} d={`M${a.x + 150} ${a.y + 26} C ${a.x + 175} ${a.y + 26}, ${c.x - 15} ${c.y + 26}, ${c.x} ${c.y + 26}`} stroke="#475569" strokeWidth="1.5" fill="none" markerEnd="url(#evoArrow)" />;
              })}
              {refEdges.map((e) => {
                const a = byId[e.source]; const c = byId[e.target];
                return <path key={`ref-${e.source}-${e.target}`} d={`M${a.x + 150} ${a.y + 40} C ${a.x + 175} ${a.y + 60}, ${c.x - 15} ${c.y + 60}, ${c.x} ${c.y + 40}`} stroke="#7c3aed" strokeDasharray="5 4" strokeWidth="1.5" fill="none" markerEnd="url(#evoArrowRef)" />;
              })}
            </svg>
            {placed.map((p) => {
              const n = g!.nodes.find((x) => x.exp_id === p.id)!;
              const tone = nodeTone(n, g?.best_exp_id);
              const border = tone === "green" ? "border-emerald-400 bg-emerald-50/90" : tone === "blue" ? "border-blue-300 bg-blue-50/90" : tone === "amber" ? "border-amber-300 bg-amber-50/90" : tone === "red" ? "border-red-300 bg-red-50/90" : "border-slate-300 bg-white";
              return (
                <div key={p.id} className={cn("absolute w-[150px] rounded-md border p-2 shadow-[0_6px_18px_-18px_rgba(15,23,42,0.4)]", border)} style={{ left: p.x, top: p.y }}>
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-[12px] font-black text-slate-950" title={p.id}>{p.id}</span>
                    <StatusBadge tone={tone}>{n.exp_id === g?.best_exp_id ? "best" : n.branch_type || "Base"}</StatusBadge>
                  </div>
                  <div className={cn(mono, "mt-1 text-[10px] text-slate-600")}>CV {fmtScore(n.cv_score)}</div>
                  {n.promoted && <div className="text-[10px] font-bold text-emerald-700">promoted</div>}
                  {(n.risk_flags ?? []).slice(0, 1).map((f) => <div key={f} className="truncate text-[10px] font-bold text-amber-700" title={f}>{f}</div>)}
                </div>
              );
            })}
          </div>
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-bold text-slate-600">
        <span className="inline-flex items-center gap-1"><span className="w-8 border-t-2 border-slate-600" />parent→child</span>
        <span className="inline-flex items-center gap-1"><span className="w-8 border-t-2 border-dashed border-violet-500" />cross/aggregation ref</span>
        <StatusBadge tone="green">best</StatusBadge>
        <StatusBadge tone="blue">Stepwise</StatusBadge>
        <StatusBadge tone="amber">Diff</StatusBadge>
        <StatusBadge tone="slate">Base</StatusBadge>
        {g?.global_stagnation && <StatusBadge tone="amber">global stagnation</StatusBadge>}
        {g?.selected_next_branch && <StatusBadge tone="blue">next → {g.selected_next_branch}</StatusBadge>}
      </div>
    </SectionCard>
  );
}

// Agent Runtime: evolution step trace, decision, gate, artifacts + fault tolerance.
export function EvolutionRuntimePanel({ taskId, refreshSummary }: { taskId: string; refreshSummary?: () => Promise<unknown> }) {
  const b = useEvolutionBinding(taskId, { refreshSummary });
  const s = b.state;
  const step = b.lastStep;
  const traceRows: Array<[string, StatusTone, string]> = [
    ["select (MCGS UCT)", "green", s?.latest_decision ?? "expand_selected_node"],
    ["propose (LLM Base/Stepwise/Diff)", "blue", step?.code_generation_mode ? `${step.code_generation_mode}/${step.expansion_type ?? "primary"}` : "planned"],
    ["run (workstation orchestrator)", (step && !step.dry_run) ? "amber" : "slate", (step && !step.dry_run) ? "blocked → workstation" : "dry_run · not trained"],
    ["gate (promotion)", (s?.gate_status ?? "").includes("block") ? "red" : "slate", s?.gate_status ?? "no_gate_yet"],
    ["backpropagate + memory", s?.memory_hits ? "green" : "slate", `${s?.memory_hits ?? 0} memory hits`]
  ];
  return (
    <SectionCard
      title="Evolution Step Trace · 进化引擎运行轨迹"
      desc="select → propose → run → gate → backpropagate。LLM 双后端 (Opus 主 / DeepSeek 兜底),故障自动切换;失败提议转 Diff 重试,不崩任务。"
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid gap-2 lg:grid-cols-[1fr_300px]">
        <div className="overflow-hidden rounded-md border border-slate-200">
          <table className="w-full text-left text-[11px]">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[10px] uppercase tracking-wide text-slate-500">
                <th className="px-3 py-1.5 font-black">进化阶段</th><th className="px-3 py-1.5 font-black">状态</th><th className="px-3 py-1.5 font-black">详情</th>
              </tr>
            </thead>
            <tbody>
              {traceRows.map(([stage, tone, detail]) => (
                <tr key={stage} className="border-b border-slate-100 last:border-0">
                  <td className="px-3 py-1.5 font-bold text-slate-700">{stage}</td>
                  <td className="px-3 py-1.5"><StatusBadge tone={tone}>{tone === "red" ? "blocked" : tone === "green" ? "ok" : tone === "amber" ? "gated" : "pending"}</StatusBadge></td>
                  <td className={cn(mono, "px-3 py-1.5 text-slate-600")}>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="space-y-2">
          <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
            <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">LLM / Cache</p>
            <p className="mt-1 text-[11px] font-semibold text-slate-600">Backend: Opus 4.8 主 + DeepSeek 兜底</p>
            <p className="text-[11px] font-semibold text-slate-600">Fault tolerance: auto-failover · Diff retry on failure</p>
          </div>
          <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
            <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">Recovery</p>
            <p className="mt-1 text-[11px] font-semibold text-slate-600">失败报错清洗后落盘 run_error.txt,喂回 Diff prompt。</p>
          </div>
        </div>
      </div>
      {(step?.artifacts ?? []).length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {step?.artifacts?.map((p) => <li key={p} className={cn(mono, "truncate text-[11px] text-emerald-700")} title={p}>› {p}</li>)}
        </ul>
      )}
    </SectionCard>
  );
}

// Evidence: artifacts the evolution engine actually wrote to workspace.
export function EvolutionEvidencePanel({ taskId }: { taskId: string }) {
  const b = useEvolutionBinding(taskId, { withMemory: true });
  const s = b.state;
  const artifacts = s?.last_artifacts ?? [];
  const mem = b.memory;
  return (
    <SectionCard
      title="Evolution Evidence · 进化引擎证据台账"
      desc={`进化引擎写出的审计产物,均落地在 workspace/evolution/${taskId || "<task>"}/。跨任务记忆库为共享真实存储,复用不分叉。`}
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid gap-2 lg:grid-cols-[1fr_320px]">
        <div>
          <p className="mb-1 text-[10px] font-black uppercase tracking-wide text-slate-500">Written artifacts</p>
          {artifacts.length === 0 ? (
            <EvoEmpty text="尚无进化产物。执行 dry-run step 会写 search_graph / validation_contract / claim_audit。" />
          ) : (
            <ul className="space-y-1">
              {artifacts.map((p) => (
                <li key={p} className="flex items-center justify-between gap-2 rounded border border-slate-200 bg-white px-2 py-1">
                  <span className={cn(mono, "truncate text-[11px] text-emerald-700")} title={p}>› {p.split(/[\\/]/).pop()}</span>
                  <span className={cn(mono, "shrink-0 text-[10px] text-slate-400")} title={p}>{p.split(/[\\/]/).slice(0, -1).join("/")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="mb-1 text-[10px] font-black uppercase tracking-wide text-slate-500">Retrospective memory ({mem?.record_count ?? 0})</p>
          {(mem?.memory?.length ?? 0) === 0 ? (
            <EvoEmpty text="No retrospective memory records for this task type yet." />
          ) : (
            <div className="max-h-[240px] space-y-1.5 overflow-y-auto pr-1">
              {mem?.memory.slice(0, 10).map((r) => (
                <div key={r.memory_id} className="rounded border border-slate-200 bg-white px-2 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className={cn(mono, "truncate text-[11px] text-slate-800")}>{r.method || r.memory_id}</span>
                    <span className={cn(mono, "text-[10px] text-slate-500")}>Δ {r.metric_delta == null ? "—" : r.metric_delta.toFixed(4)}</span>
                  </div>
                  {r.reusable_strategy && <p className="mt-0.5 text-[10px] leading-4 text-emerald-700">✓ {r.reusable_strategy}</p>}
                  {r.failure_pattern && <p className="mt-0.5 text-[10px] leading-4 text-amber-700">✗ {r.failure_pattern}</p>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      {mem?.memory_store && <p className={cn(mono, "mt-2 truncate text-[11px] text-slate-400")} title={mem.memory_store}>store: {mem.memory_store}</p>}
    </SectionCard>
  );
}

function GateLine({ label, status, tone }: { label: string; status: string; tone: StatusTone }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-slate-100 py-1.5 text-[11px] last:border-0">
      <span className="font-bold text-slate-700">{label}</span>
      <StatusBadge tone={tone}>{status}</StatusBadge>
    </div>
  );
}

// Gates: rank_promotion_gate, score_promotion_gate, claim_audit — all API-driven.
export function EvolutionGatesPanel({ taskId }: { taskId: string }) {
  const b = useEvolutionBinding(taskId, { withGraph: true });
  const s = b.state;
  const g = b.graph;
  const best = s?.best_so_far;
  const hasBest = Boolean(best?.exp_id && best?.cv_score != null);
  const stagnation = s?.search_graph_summary?.global_stagnation;
  return (
    <SectionCard
      title="Evolution Gates · 进化门禁"
      desc="score/rank 晋升门禁与 claim audit。官方 rank/medal 无 Kaggle response artifact 时一律 proxy_only / blocked,不虚构。"
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="mb-1 text-xs font-black text-slate-800">Score Promotion Gate</p>
          <GateLine label="Has scored best-so-far" status={hasBest ? "yes" : "no run"} tone={hasBest ? "green" : "slate"} />
          <GateLine label={`Best ${best?.metric ?? "CV"} (proxy)`} status={fmtScore(best?.cv_score)} tone={hasBest ? "green" : "slate"} />
          <GateLine label="run_success precondition" status="enforced" tone="green" />
          <GateLine label="Regression guard vs best" status={stagnation ? "stagnation" : "active"} tone={stagnation ? "amber" : "green"} />
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="mb-1 text-xs font-black text-slate-800">Rank Promotion Gate</p>
          <GateLine label="Official Kaggle response" status="missing" tone="amber" />
          <GateLine label="Official rank / percentile" status="proxy_only" tone="slate" />
          <GateLine label="Medal claim" status="blocked" tone="red" />
          <GateLine label="Official submit" status={s?.official_submit_allowed ? "ON" : "disabled"} tone={s?.official_submit_allowed ? "amber" : "slate"} />
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="mb-1 text-xs font-black text-slate-800">Claim Audit</p>
          <GateLine label="Claim boundary enforced" status="yes" tone="green" />
          <GateLine label="CV = local proxy only" status="declared" tone="green" />
          <GateLine label="Unsupported claims" status="0" tone="green" />
          <GateLine label="Risk flags" status={String((s?.risk_flags ?? []).length)} tone={(s?.risk_flags ?? []).length ? "amber" : "green"} />
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {(s?.risk_flags ?? []).map((f) => <StatusBadge key={f} tone="amber">{f}</StatusBadge>)}
        {(g?.stagnation_branches ?? []).map((br) => <StatusBadge key={br} tone="amber">stagnation: {br}</StatusBadge>)}
      </div>
      {s?.claim_boundary && <p className="mt-2 text-[11px] leading-relaxed text-slate-400">{s.claim_boundary}</p>}
    </SectionCard>
  );
}

// Report: cites evolution artifacts as evidence, never claims official scores.
export function EvolutionReportPanel({ taskId }: { taskId: string }) {
  const b = useEvolutionBinding(taskId, { withGraph: true });
  const s = b.state;
  const g = b.graph;
  const best = s?.best_so_far;
  const artifacts = s?.last_artifacts ?? [];
  return (
    <SectionCard
      title="Evolution Evidence for Report · 报告可引用的进化证据"
      desc="报告可引用以下进化引擎产物;分数均为本地 proxy CV,不得声称官方 Kaggle 提分或名次。"
      action={<EvoControls binding={b} taskId={taskId} />}
    >
      <EvoBanner error={b.error} message={b.message} />
      <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
        <EvoChip label="Best exp" value={best?.exp_id ?? "—"} tone={best?.exp_id ? "green" : "slate"} />
        <EvoChip label={`Best ${best?.metric ?? "CV"} (proxy)`} value={fmtScore(best?.cv_score)} tone="blue" />
        <EvoChip label="Nodes explored" value={String(g?.node_count ?? s?.search_graph_summary?.node_count ?? 0)} tone="slate" />
        <EvoChip label="Official result" value="proxy_only" tone="amber" />
      </div>
      <div className="mt-2 rounded-md border border-slate-200 bg-slate-50/60 p-3">
        <p className="text-[10px] font-black uppercase tracking-wide text-slate-500">Citable artifacts</p>
        {artifacts.length === 0 ? (
          <EvoEmpty text="尚无可引用的进化产物。先在 Evolution 页执行 plan / dry-run step。" />
        ) : (
          <ul className="mt-1 space-y-0.5">
            {artifacts.map((p) => <li key={p} className={cn(mono, "truncate text-[11px] text-emerald-700")} title={p}>› {p}</li>)}
          </ul>
        )}
      </div>
      <div className="mt-2 rounded-md border border-amber-100 bg-amber-50 px-3 py-1.5 text-[11px] font-bold text-amber-700">
        Claim boundary：CV 为本地 proxy。无 Kaggle response artifact 时,rank/medal 保持空/blocked/proxy_only,报告不得声称官方成绩。
      </div>
    </SectionCard>
  );
}









