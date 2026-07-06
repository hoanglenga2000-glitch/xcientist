"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bell,
  BookOpen,
  Bot,
  Box,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Code2,
  Cpu,
  Database,
  Download,
  FileCode2,
  FileText,
  Filter,
  GitBranch,
  Globe2,
  KeyRound,
  Languages,
  Layers3,
  LineChart,
  Lock,
  Moon,
  Network,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Sun,
  TerminalSquare,
  Upload,
  UserCheck,
  Users
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import {
  EvolutionControlPanel,
  EvolutionEvidencePanel,
  EvolutionGatesPanel,
  EvolutionOverviewPanel,
  EvolutionReportPanel,
  EvolutionRuntimePanel,
  EvolutionSearchGraphPanel
} from "@/components/workstation/EvolutionPanels";

type Locale = "zh-CN" | "en-US";

type ScreenProps = {
  selectedTask: string;
  setSelectedTask: (id: string) => void;
  selectedStage: string;
  setSelectedStage: (id: string) => void;
  selectedExperiment: string;
  setSelectedExperiment: (id: string) => void;
  gateStatus: "Pending" | "Approved" | "Rejected";
  setGateStatus: (status: "Pending" | "Approved" | "Rejected") => void;
  patchApplied: boolean;
  setPatchApplied: (value: boolean) => void;
  reportSubmitted: boolean;
  setReportSubmitted: (value: boolean) => void;
  summary?: WorkstationSummary | null;
  refreshSummary?: () => Promise<WorkstationSummary>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
  runState?: { status: "idle" | "running" | "passed" | "failed"; message: string; experimentDir?: string };
  exportCodeAgentContext?: (taskId?: string, targetAgent?: string) => Promise<void>;
  importDemoPatch?: (taskId?: string) => Promise<void>;
  agentActionMessage?: string;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  systemActionMessage?: string;
  lastActionTrace?: {
    action: string;
    taskId?: string;
    request?: Record<string, unknown>;
    response?: Record<string, unknown>;
    message: string;
    artifact?: string | null;
    at: string;
  } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

const mono = "font-mono text-[12px] tracking-normal";

function stopWheelPropagation(event: React.WheelEvent<HTMLElement>) {
  const element = event.currentTarget;
  const canScrollY = element.scrollHeight > element.clientHeight;
  const canScrollX = element.scrollWidth > element.clientWidth;
  if (canScrollY || canScrollX) {
    event.stopPropagation();
  }
}

function downloadTextFile(filename: string, content: string, mime = "text/plain;charset=utf-8") {
  if (typeof window === "undefined") return;
  const blob = new Blob([content], { type: mime });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}

function toCsv(rows: readonly (readonly React.ReactNode[])[]) {
  return rows
    .map((row) =>
      row
        .map((cell) => {
          const text = typeof cell === "string" || typeof cell === "number" ? String(cell) : "";
          return `"${text.replaceAll('"', '""')}"`;
        })
        .join(",")
    )
    .join("\n");
}

const agents = [
  ["Research Agent", "Research plan and hypothesis framing", "completed", "plan.json", 100],
  ["Data Audit Agent", "Dataset schema, leakage, and metric audit", "completed", "data_audit.json", 100],
  ["Feature Eng. Agent", "Feature branch search and reusable transforms", "running", "features.parquet", 78],
  ["Model Selection Agent", "Model family routing and search graph update", "running", "model_select.json", 62],
  ["Code Impl. Agent", "Controlled training and submission code draft", "running", "solution.py", 65],
  ["GPU / HPC Agent", "Manifest-gated remote execution", "running", "gpu_job_manifest.yaml", 41],
  ["Validation Agent", "OOF, CV, schema and stability checks", "running", "metrics.json", 58],
  ["Submission Gate Agent", "Official submission policy and audit", "waiting", "submission.json", 0],
  ["Claim Audit Agent", "Claim drift and evidence boundary review", "waiting", "claim_audit.json", 0],
  ["Report Agent", "Teacher report and evidence bundle", "pending", "report_draft.md", 0],
  ["Reflection Agent", "Failure review and recovery plan", "failed", "recovery_plan.md", 15]
] as const;

const artifacts = [
  ["metrics_20260625_1001", "metrics", "task_001", "run_14", "exp_038", "agent_xcientist_v2", "evaluation", "/runs/.../metrics.json", "91fa...c2b1", "claim_003", "gate_regression_v1", "verified"],
  ["oof_20260625_1001", "oof_prediction", "task_001", "run_14", "exp_038", "agent_xcientist_v2", "evaluation", "/runs/.../oof_pred.parquet", "3b7a...f6a9", "claim_003", "gate_oof_v1", "pending"],
  ["model_code_20260625_1001", "code", "task_001", "run_14", "exp_038", "code-agent", "train", "/artifacts/code/train.py", "a1b2...4e56", "claim_003", "gate_code_v1", "verified"],
  ["submission_20260625_1001", "submission", "task_001", "run_14", "exp_038", "report-agent", "submit", "/submissions/.../submission.csv", "7e2b...9664", "claim_003", "gate_kaggle_v1", "verified"],
  ["gpu_job_manifest_1001", "gpu_job_manifest", "task_001", "run_14", "exp_038", "hpc-agent", "train", "/hpc/job_1001.yaml", "6d8a...e0b3", "-", "gate_hpc_v1", "verified"],
  ["stderr_log_1001", "stderr_log", "task_001", "run_14", "exp_038", "hpc-agent", "train", "/logs/stderr.log", "9a11...3bb7", "-", "gate_hpc_v1", "pending"],
  ["claim_audit_1001", "claim_audit", "task_001", "run_14", "exp_038", "audit-agent", "audit", "/audit/claim_audit.json", "1d23...666", "claim_003", "gate_claim_v1", "blocked"]
] as const;

const jobs = [
  ["jb_20260625_0012", "tk_0842", "run_7f3c1d", "exp_0912", "agent_lightgbm", "lgbm_cv_v2", "8xA800 80GB", "gate_hpc_03", "running", "11:31:02", "00:07:18"],
  ["jb_20260625_0013", "tk_0843", "run_8a9f2e", "exp_0913", "agent_xgb", "xgb_cv_v1", "4xA800 80GB", "gate_hpc_03", "queued", "-", "-"],
  ["jb_20260625_0011", "tk_0841", "run_6d2b7a", "exp_0911", "agent_tabnet", "tabnet_cv_v1", "2xA800 80GB", "gate_hpc_03", "blocked_by_gate", "-", "-"],
  ["jb_20260625_0010", "tk_0840", "run_5c1a8f", "exp_0910", "agent_catboost", "catboost_cv_v1", "4xA800 80GB", "gate_hpc_02", "failed", "10:18:40", "00:06:41"]
] as const;

const codeSample = `import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# 1. load audited data
train = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")
target = train["target"]

# 2. feature engineering bound to Data Agent audit result
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_ab"] = df["a"] / (df["b"] + 1e-6)
    df["log_c"] = np.log1p(df["c"])
    df["inter_a_c"] = df["a"] * df["c"]
    return df

train = add_features(train)
test = add_features(test)

# 3. OOF validation before any official submission
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
    model = lgb.LGBMClassifier(n_estimators=1800, learning_rate=0.025)
    model.fit(train.iloc[trn_idx], target.iloc[trn_idx])
    oof[val_idx] = model.predict_proba(train.iloc[val_idx])[:, 1]
    test_pred += model.predict_proba(test)[:, 1] / 5

metric = log_loss(target, oof)
json.dump({"oof_logloss": metric}, open("metrics.json", "w"))`;

function toneFor(value: string | undefined): StatusTone {
  const v = String(value ?? "").toLowerCase();
  if (["completed", "done", "passed", "verified", "online", "ready", "approved", "generated", "healthy"].some((x) => v.includes(x))) return "green";
  if (["running", "reviewing", "queued", "draft", "indexed"].some((x) => v.includes(x))) return "blue";
  if (["waiting", "pending", "warning", "review", "controlled", "stale"].some((x) => v.includes(x))) return "amber";
  if (["failed", "blocked", "missing", "regression", "rejected"].some((x) => v.includes(x))) return "red";
  return "slate";
}

function Page({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <main className="min-h-[835px] space-y-1.5 xl:space-y-1">
      <div className="lg:hidden">
        <h2 className="text-[24px] font-black leading-tight tracking-normal text-slate-950">{title}</h2>
        <p className="mt-1 text-sm font-semibold text-slate-500">{subtitle}</p>
      </div>
      {children}
    </main>
  );
}

function StatCard({ icon: Icon, label, value, sub, tone = "blue" }: { icon: React.ElementType; label: string; value: string; sub?: string; tone?: StatusTone }) {
  const toneClass = {
    blue: "border-blue-100 bg-blue-50 text-blue-700",
    green: "border-emerald-100 bg-emerald-50 text-emerald-700",
    amber: "border-amber-100 bg-amber-50 text-amber-700",
    red: "border-red-100 bg-red-50 text-red-700",
    slate: "border-slate-200 bg-slate-50 text-slate-700",
    purple: "border-violet-100 bg-violet-50 text-violet-700"
  }[tone];
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-3">
          <span className={cn("flex h-9 w-9 items-center justify-center rounded-md border", toneClass)}>
            <Icon className="h-4 w-4" />
          </span>
          <StatusBadge tone={tone}>{sub ?? tone}</StatusBadge>
        </div>
        <div className="mt-3 text-[11px] font-black uppercase tracking-[0.04em] text-slate-500">{label}</div>
        <div className="mt-1 truncate text-[24px] font-black leading-7 text-slate-950">{value}</div>
      </CardContent>
    </Card>
  );
}

function Panel({ title, description, action, children, className, style }: { title: string; description?: string; action?: React.ReactNode; children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <Card className={cn("min-w-0", className)} style={style}>
      <CardHeader className="flex flex-row items-start justify-between gap-2 px-2.5 pt-2">
        <div className="min-w-0">
          <CardTitle>{title}</CardTitle>
          {description ? <CardDescription>{description}</CardDescription> : null}
        </div>
        {action}
      </CardHeader>
      <CardContent className="p-2">{children}</CardContent>
    </Card>
  );
}

function Row({ label, value, monoValue = false }: { label: string; value: React.ReactNode; monoValue?: boolean }) {
  return (
    <div className="grid min-w-0 grid-cols-[108px_1fr] gap-1.5 border-b border-slate-100 py-0.5 text-[10px] last:border-b-0">
      <span className="font-bold text-slate-500">{label}</span>
      <span className={cn("min-w-0 break-words font-bold text-slate-900", monoValue && mono)}>{value}</span>
    </div>
  );
}

function Progress({ value, tone = "blue" }: { value: number; tone?: StatusTone }) {
  const color = tone === "green" ? "bg-emerald-500" : tone === "amber" ? "bg-amber-500" : tone === "red" ? "bg-red-500" : "bg-blue-600";
  return (
    <div className="h-2 overflow-hidden rounded-full bg-slate-100">
      <div className={cn("h-full rounded-full", color)} style={{ width: `${Math.max(0, Math.min(value, 100))}%` }} />
    </div>
  );
}

function MiniLine({ tone = "blue" }: { tone?: StatusTone }) {
  const stroke = tone === "green" ? "#10b981" : tone === "amber" ? "#f59e0b" : tone === "red" ? "#ef4444" : "#2563eb";
  return (
    <svg viewBox="0 0 120 32" className="h-8 w-full">
      <polyline points="0,22 12,18 24,20 36,11 48,15 60,9 72,17 84,8 96,12 108,5 120,10" fill="none" stroke={stroke} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function GatePipeline() {
  const steps = [
    ["Task Spec", "Done", "green"],
    ["Data Audit", "Done", "green"],
    ["Research Plan", "Done", "green"],
    ["Code Draft", "Running", "blue"],
    ["GPU/HPC Job", "Running", "blue"],
    ["Metrics / OOF", "Waiting", "amber"],
    ["Submission Gate", "Needs Approval", "amber"],
    ["Kaggle Response", "Pending", "slate"],
    ["Report", "Pending", "slate"]
  ] as const;
  return (
    <Card>
      <CardContent className="flex min-w-0 items-center gap-2 overflow-x-auto p-3">
        {steps.map(([label, status, tone], index) => (
          <div key={label} className="flex shrink-0 items-center gap-2">
            <div className="flex min-w-[118px] flex-col items-center rounded-md border border-slate-200 bg-white px-3 py-2 text-center">
              <StatusBadge tone={tone as StatusTone}>{status}</StatusBadge>
              <span className="mt-1 text-xs font-black text-slate-700">{label}</span>
            </div>
            {index < steps.length - 1 ? <ArrowRight className="h-4 w-4 text-slate-300" /> : null}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function AgentGraph() {
  return (
    <Panel title="多智能体执行图 / Multi-Agent Execution Graph" description="任务流转、产物流转和失败回退状态">
      <div className="workflow-bg rounded-lg border border-slate-200 bg-slate-50/60 p-4">
        <div className="mx-auto mb-4 w-fit rounded-lg border border-blue-200 bg-white px-5 py-3 text-center shadow-sm">
          <StatusBadge tone="blue">running · routing</StatusBadge>
          <div className="mt-1 text-sm font-black text-slate-950">Orchestrator Agent</div>
          <div className={`${mono} mt-1 text-blue-700`}>orchestrator_state.json</div>
        </div>
        <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
          {agents.slice(0, 6).map(([name, task, status, artifact, progress]) => (
            <div key={name} className={cn("rounded-lg border bg-white p-3", status === "failed" ? "border-red-200" : status === "waiting" ? "border-amber-200" : "border-slate-200")}>
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-black text-slate-900">{name}</div>
                <StatusBadge tone={toneFor(status)}>{status}</StatusBadge>
              </div>
              <div className="mt-2 min-h-8 text-[11px] leading-4 text-slate-500">{task}</div>
              <div className={`${mono} mt-2 truncate rounded bg-slate-50 px-2 py-1 text-blue-700`}>{artifact}</div>
              <div className="mt-2 flex items-center gap-2">
                <div className="flex-1"><Progress value={progress} tone={toneFor(status)} /></div>
                <span className="w-8 text-right text-[11px] font-black text-slate-500">{progress}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function ActionFooter() {
  return (
    <div className="sticky bottom-0 z-10 mt-4 rounded-lg border border-slate-200 bg-white/95 p-3 shadow-[0_-8px_26px_-26px_rgba(15,23,42,0.55)] backdrop-blur">
      <div className="grid gap-3 md:grid-cols-4">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-blue-600" />
          <div><div className="text-xs font-black text-slate-500">Active Phase</div><div className="text-sm font-black text-slate-900">OOF / CV</div></div>
        </div>
        <div><div className="text-xs font-black text-slate-500">Current Best Protected</div><div className={`${mono} text-slate-900`}>exp_909 · RMSE 0.76321</div></div>
        <div><div className="text-xs font-black text-slate-500">Next Allowed Action</div><div className="text-sm font-black text-amber-700">等待 Submission Gate 审批</div></div>
        <div><div className="text-xs font-black text-slate-500">Human Approval Pending</div><div className="text-sm font-black text-red-700">3 items</div></div>
      </div>
    </div>
  );
}

export function MissionControl(props: ScreenProps) {
  const s = props.summary;
  const tasks = s?.tasks ?? [];
  const activeTasks = tasks.filter(t => t.status === "active" || t.status === "running").length;
  const totalTasks = tasks.length;
  const launchAudit = s?.verified_launch_audit;
  const blockers = Array.isArray(launchAudit?.blockers) ? launchAudit.blockers : [];
  const launchState = launchAudit?.launch_state ?? "unknown";
  const learning = s?.learning_loop_readiness;
  const learningStatus = learning?.status ?? "unknown";
  const learningReady = learningStatus === "passed";
  const memoryRecords = learning?.memory?.record_count ?? 0;
  const searchOrderRecords = learning?.search_orders?.record_count ?? 0;
  const observedRuns = learning?.training_progress?.observed_runs ?? s?.kaggle_experiment_inventory?.total_runs_observed ?? 0;
  const promotedRuns = learning?.training_progress?.promoted_runs ?? s?.kaggle_experiment_inventory?.total_promoted_runs ?? 0;
  const officialTop30 = learning?.training_progress?.official_top30_tasks
    ?? Number(s?.mlebench_style_leaderboard?.summary?.official_top30_count ?? 0);
  const medalCount = learning?.training_progress?.medal_count
    ?? Number(s?.mlebench_style_leaderboard?.summary?.medal_count ?? 0);
  const benchmarkClaim = learning?.training_progress?.benchmark_claim_status
    ?? String(s?.mlebench_style_leaderboard?.summary?.benchmark_claim_status ?? "not_comparable_not_reached");
  const gpuConnector = (s?.connector_status?.gpu ?? {}) as Record<string, unknown>;
  const kaggleConnector = (s?.connector_status?.kaggle ?? {}) as Record<string, unknown>;
  const gpuReady = gpuConnector.current_gate_ready === true;
  const kaggleReady = kaggleConnector.configured === true;
  const cacheBlocked = blockers.includes("deepseek_cache_below_80_for_batch_generation");
  const nextRunReady = learning?.next_run_queue?.ready_to_start_now === true && blockers.length === 0;

  const workflow = [
    ["Task Spec", totalTasks > 0 ? "verified" : "pending", `${totalTasks} tasks`, FileText],
    ["Data Audit", "verified", `${observedRuns} observed runs`, Database],
    ["Search Controller", learningReady ? "verified" : "pending", `${searchOrderRecords} orders`, Search],
    ["Code Agent", cacheBlocked ? "blocked" : "ready", cacheBlocked ? "cache gate <80%" : "cache gate ok", Code2],
    ["GPU/HPC Job", gpuReady ? "ready" : "blocked", gpuReady ? "current gate ready" : "resource smoke blocked", Server],
    ["Metrics/OOF", observedRuns > 0 ? "verified" : "pending", `${promotedRuns} promoted`, LineChart],
    ["Submission Gate", medalCount > 0 ? "verified" : "needs review", `${officialTop30} official top30`, ShieldAlert],
    ["Kaggle Response", kaggleReady ? "ready" : "blocked", kaggleReady ? "DPAPI ready" : "token/tool blocked", Globe2],
    ["Report", learningReady ? "verified" : "pending", `${memoryRecords} memories`, FileText]
  ];
  return (
    <Page title="Academic Research OS" subtitle="Workstation-started runs with agent dispatch, evidence return, gates, and report handoff.">
      <LiveRunEvidencePanel {...props} />
      <EvolutionOverviewPanel taskId={props.selectedTask} refreshSummary={props.refreshSummary} />
      <div className="grid gap-2 xl:grid-cols-[210px_minmax(360px,1fr)_auto]">
        <button className="flex h-10 items-center justify-between rounded-md border border-slate-200 bg-white px-3 text-sm font-black text-slate-800 shadow-[0_1px_2px_rgba(15,23,42,0.03)]" data-ui-action="select_research_workspace">
          <span className="flex items-center gap-2"><BrainCircuit className="h-4 w-4 text-blue-600" />DeepResearch Lab</span>
          <ChevronRight className="h-4 w-4 rotate-90 text-slate-400" />
        </button>
        <div className="flex h-10 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-500 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <Search className="h-4 w-4 text-slate-400" />
          <span className="flex-1 truncate">Search runs, artifacts, datasets, agents...</span>
          <span className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] font-black text-slate-500">⌘ K</span>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <StatusBadge tone={activeTasks > 0 ? "green" : "slate"}>{activeTasks} Tasks Active</StatusBadge>
          <StatusBadge tone={gpuReady ? "green" : "red"}>{gpuReady ? "GPU Ready" : "GPU Blocked"}</StatusBadge>
          <StatusBadge tone={blockers.length ? "red" : "green"}>{blockers.length ? `${blockers.length} Blockers` : "Launch Ready"}</StatusBadge>
          <StatusBadge tone={learningReady ? "green" : "amber"}>Learning {learningStatus}</StatusBadge>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" size="sm" data-ui-action="mission_create_workstation_run"><Play className="h-4 w-4" />Create Run</Button>
        <Button variant="secondary" size="sm" data-ui-action="mission_open_evidence_ledger"><FileText className="h-4 w-4" />Open Evidence Ledger</Button>
        <Button variant="secondary" size="sm" data-ui-action="mission_prepare_hpc_job"><Server className="h-4 w-4" />Launch HPC Job</Button>
      </div>
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-7">
        <MissionStat label="Active Tasks" value={`${totalTasks}`} delta={`${activeTasks} running`} status="Running" tone="blue" />
        <MissionStat label="Official Medals" value={`${medalCount}`} delta={`top30 ${officialTop30}; ${benchmarkClaim}`} status={medalCount > 0 ? "Verified" : "Not claimed"} tone={medalCount > 0 ? "green" : "amber"} />
        <MissionStat label="Workstation API" value="Connected" delta="port 8088 online" status="Verified" tone="green" />
        <MissionStat label="GPU Cluster" value={gpuReady ? "Ready" : "Blocked"} delta={String(gpuConnector.state ?? launchState)} status={gpuReady ? "Healthy" : "Blocked"} tone={gpuReady ? "green" : "red"} />
        <MissionStat label="MCGS Engine" value={learningReady ? "Ready" : "Blocked"} delta={`${memoryRecords} memories / ${searchOrderRecords} orders`} status={learningStatus} tone={learningReady ? "green" : "amber"} />
        <MissionStat label="Kaggle API" value={kaggleReady ? "Connected" : "Blocked"} delta={String(kaggleConnector.state ?? "unknown")} status={kaggleReady ? "Connected" : "Blocked"} tone={kaggleReady ? "green" : "red"} />
        <MissionStat label="System Status" value={blockers.length ? "Blocked" : "Healthy"} delta={blockers.length ? blockers.join(", ") : "all launch gates passed"} status={blockers.length ? "Blocked" : "Healthy"} tone={blockers.length ? "red" : "green"} />
      </div>
      <Panel title="Launch & Learning Readiness" description="真实上线状态、资源门禁和自动化学习闭环">
        <div className="grid gap-2 xl:grid-cols-[1fr_1fr_1fr]">
          <div className="rounded-md border border-slate-200 bg-white p-2">
            <Row label="Launch State" value={<StatusBadge tone={blockers.length ? "red" : "green"}>{launchState}</StatusBadge>} />
            <Row label="Blockers" value={blockers.length ? blockers.join(", ") : "none"} />
            <Row label="Next Run" value={<StatusBadge tone={nextRunReady ? "green" : "amber"}>{nextRunReady ? "ready" : "blocked by gate"}</StatusBadge>} />
          </div>
          <div className="rounded-md border border-slate-200 bg-white p-2">
            <Row label="Learning Loop" value={<StatusBadge tone={learningReady ? "green" : "amber"}>{learningStatus}</StatusBadge>} />
            <Row label="Memory" value={`${memoryRecords} records`} />
            <Row label="Search Orders" value={`${searchOrderRecords} records`} />
          </div>
          <div className="rounded-md border border-slate-200 bg-white p-2">
            <Row label="Observed Runs" value={`${observedRuns}`} />
            <Row label="Promoted / Held" value={`${promotedRuns} / ${learning?.training_progress?.held_runs ?? s?.kaggle_experiment_inventory?.total_held_runs ?? 0}`} />
            <Row label="Medal Claim" value={benchmarkClaim} />
          </div>
        </div>
      </Panel>
      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_390px]">
        <div className="space-y-2">
          <Panel title="Research Workflow Console" description="Run: run-2025-05-18-0942" action={<Button size="sm" variant="secondary" data-ui-action="mission_view_workflow_details">View Details</Button>}>
            <div className="thin-scrollbar grid gap-2 overflow-x-auto xl:grid-cols-9">
              {workflow.map(([name, state, desc, Icon], index) => (
                <div key={name as string} className="relative min-h-[172px] min-w-[128px] rounded-md border border-slate-200 bg-white p-2.5 shadow-[0_1px_1px_rgba(15,23,42,0.03)]">
                  <div className="flex items-center justify-between"><Icon className="h-4 w-4 text-slate-700" /><StatusBadge tone={toneFor(state as string)}>{state as string}</StatusBadge></div>
                  <div className="mt-3 text-xs font-black text-slate-950">{name as string}</div>
                  <div className="mt-2 min-h-10 text-[11px] leading-4 text-slate-500">{desc as string}</div>
                  <div className="mt-2 space-y-1 text-[10px] font-bold text-slate-500">
                    <RowCompact label={index % 2 ? "Agent" : "Run"} value={index % 2 ? "deepseek-coder" : "run-0942"} />
                    <RowCompact label={index % 3 ? "Evidence" : "Queue"} value={index % 3 ? "7/12" : "Default"} />
                  </div>
                  <div className="mt-2 h-6 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[10px] font-bold text-slate-500">{state as string}</div>
                  {index < workflow.length - 1 ? <ArrowRight className="absolute -right-4 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-slate-300 xl:block" /> : null}
                </div>
              ))}
            </div>
          </Panel>
          <div className="grid gap-3 xl:grid-cols-[0.9fr_0.9fr_1fr]">
            <Panel title="Agent Runtime Timeline" description="Live"><DenseTable headers={["time", "agent", "status"]} rows={[["11:24", "Search Controller", "completed"], ["11:22", "Code Agent", "completed"], ["11:20", "Handoff", "completed"], ["11:18", "GPU/HPC Job", "queued"], ["11:16", "Data Audit", "completed"]]} /></Panel>
            <Panel title="Benchmark / Medal Board" description="Kaggle + MLE-Bench"><DenseTable headers={["Run", "Official", "Rank", "Top30"]} rows={[["latest", "no response", "-", "blocked"], ["best-so-far", "no response", "-", "blocked"], ["candidate", "proxy only", "-", "blocked"], ["human gate", "required", "-", "blocked"]]} /></Panel>
            <Panel title="GPU/HPC Resource Monitor" description="运行资源"><DenseTable headers={["Job", "GPU", "Util", "State"]} rows={[["job-8f72c1e", "4 x A100", "78%", "queued"], ["job-6a1f9b2", "2 x A100", "--", "queued"], ["job-3c5e7f0", "4 x A100", "95%", "running"], ["job-1b9a2d3", "2 x A100", "82%", "running"]]} /></Panel>
          </div>
        </div>
        <aside className="space-y-2">
          <Panel title="Evidence Ledger" description="最新证据" action={<button className="text-xs font-black text-blue-700" data-ui-action="mission_view_all_evidence">View All</button>}>
            <DenseTable headers={["Artifact ID", "Type", "Run", "Status"]} rows={[["art-9f2a1c7d", "Model", "run-0942", "verified"], ["art-0b8e4a9c", "Dataset", "run-0941", "verified"], ["art-4c9d7f1b", "Config", "run-0942", "pending"], ["art-7e1a2b3c", "Log", "run-0942", "verified"], ["art-a35f6a7b", "OOF Pred", "run-0942", "pending"]]} />
          </Panel>
          <Panel title="Claim Audit" description="结论审核" action={<button className="text-xs font-black text-blue-700" data-ui-action="mission_view_all_claim_audits">View All</button>}>
            <div className="grid grid-cols-4 gap-2"><StatMini label="Total" value="24" tone="blue" /><StatMini label="Verified" value="13" tone="green" /><StatMini label="Pending" value="8" tone="amber" /><StatMini label="Blocked" value="3" tone="red" /></div>
            <DenseTable headers={["Claim", "Status"]} rows={[["Best model OOF improvement", "pending"], ["No target leakage", "verified"], ["Inference deterministic", "blocked"]]} />
          </Panel>
          <Panel title="Submission Gate" description="Needs Human Review">
            <div className="grid grid-cols-3 gap-2">
              <StatMini label="Provided" value="7" tone="green" />
              <StatMini label="Missing" value="5" tone="amber" />
              <StatMini label="Policy" value="v1.4" tone="blue" />
            </div>
            <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-center text-xs font-black text-amber-800">Needs Human Review</div>
          </Panel>
          <Panel title="Latest Kaggle Response" description="官方提交受 Human Gate 控制，暂无官方 response artifact。">
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div><div className="font-black text-slate-500">Status</div><StatusBadge tone="amber">No Official Response</StatusBadge></div>
              <div><div className="font-black text-slate-500">Official Submit</div><div className="font-black">Human-Gated</div></div>
              <div><div className="font-black text-slate-500">Source</div><div className="font-black">proxy / CV only</div></div>
            </div>
          </Panel>
          <Panel title="Report Studio / Package" description="可复现交付包"><Row label="Artifacts" value="7 / 12" /><Progress value={58} tone="green" /><DenseTable headers={["Step", "Status"]} rows={[["Environment Lock", "completed"], ["Code Snapshot", "completed"], ["Run Reproduction", "in progress"], ["Report Generation", "pending"]]} /><div className="mt-2 grid grid-cols-2 gap-2"><Button size="sm" variant="secondary" data-ui-action="mission_export_evidence">Export Evidence</Button><Button size="sm" variant="primary" data-ui-action="mission_build_report_package">Build Package</Button></div></Panel>
        </aside>
      </div>
      <BottomSummary phase="Search Controller" best="Top30 target: rank <= 30% / current candidate run-0942" next="View Details" pending={`${props.gateStatus} Gate`} />
    </Page>
  );
}



function StatMini({ label, value, tone }: { label: string; value: string; tone: StatusTone }) {
  return <div className="rounded-md border border-slate-200 p-2"><div className="text-[11px] font-bold text-slate-500">{label}</div><div className="mt-1 text-lg font-black text-slate-950">{value}</div><StatusBadge tone={tone}>{tone}</StatusBadge></div>;
}

function RowCompact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-slate-100 py-0.5 last:border-0">
      <span className="truncate text-slate-500">{label}</span>
      <span className={cn(mono, "truncate text-slate-700")}>{value}</span>
    </div>
  );
}

function MissionStat({ label, value, delta, status, tone }: { label: string; value: string; delta: string; status: string; tone: StatusTone }) {
  return (
    <div className="min-h-[118px] rounded-md border border-slate-200 bg-white p-3 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
      <div className="text-[11px] font-black text-slate-500">{label}</div>
      <div className="mt-3 flex items-end justify-between gap-2">
        <div className="text-2xl font-black leading-none text-slate-950">{value}</div>
        <MiniLine />
      </div>
      <div className="mt-3 flex items-center justify-between gap-2">
        <span className="truncate text-[11px] font-semibold text-slate-500">{delta}</span>
        <StatusBadge tone={tone}>{status}</StatusBadge>
      </div>
    </div>
  );
}

function HeroStatusCard({
  icon: Icon,
  label,
  title,
  detail,
  tone,
  meta
}: {
  icon: React.ElementType;
  label: string;
  title: string;
  detail: string;
  tone: StatusTone;
  meta?: string;
}) {
  const styles = {
    green: "border-emerald-200 bg-gradient-to-br from-emerald-50 to-white text-emerald-700",
    blue: "border-blue-200 bg-gradient-to-br from-blue-50 to-white text-blue-700",
    amber: "border-amber-200 bg-gradient-to-br from-amber-50 to-white text-amber-700",
    red: "border-red-200 bg-gradient-to-br from-red-50 to-white text-red-700",
    slate: "border-slate-200 bg-white text-slate-700",
    purple: "border-violet-200 bg-gradient-to-br from-violet-50 to-white text-violet-700"
  }[tone];
  return (
    <div className={cn("flex min-h-[58px] min-w-0 gap-2 rounded-md border p-2 shadow-[0_1px_2px_rgba(15,23,42,0.035)]", styles)}>
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-white bg-white shadow-[0_1px_1px_rgba(15,23,42,0.04)]">
        <Icon className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[10px] font-black uppercase tracking-[0.04em] text-slate-500">{label}</div>
        <div className="truncate text-[14px] font-black leading-4 text-slate-950">{title}</div>
        <div className="mt-0.5 line-clamp-1 text-[10px] font-semibold leading-[13px] text-slate-600">{detail}</div>
        {meta ? <div className="mt-0.5 text-[10px] font-black text-current">{meta}</div> : null}
      </div>
    </div>
  );
}

function MicroStatusRail({ items }: { items: readonly (readonly [string, string, StatusTone])[] }) {
  return (
    <div className="grid gap-1.5 md:grid-cols-2 xl:grid-cols-4">
      {items.map(([label, value, tone]) => (
        <div key={label} className="flex min-h-6 items-center justify-between gap-2 rounded-md border border-slate-200 bg-white px-2 py-0.5 text-[10px] shadow-[0_1px_1px_rgba(15,23,42,0.02)]">
          <span className="truncate font-black text-slate-500">{label}</span>
          <span className={cn("truncate font-black", tone === "green" ? "text-emerald-700" : tone === "amber" ? "text-amber-700" : tone === "red" ? "text-red-700" : "text-blue-700")}>{value}</span>
        </div>
      ))}
    </div>
  );
}

function FilterBar({ items }: { items: string[] }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-slate-200 bg-white p-2 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
      {items.map((item) => (
        <button key={item} data-ui-action={`filter_${item.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="flex h-8 min-w-[112px] items-center justify-between gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-700 hover:bg-white">
          <span className="truncate">{item}</span>
          <ChevronRight className="h-3.5 w-3.5 rotate-90 text-slate-400" />
        </button>
      ))}
      <Button size="sm" variant="secondary" data-ui-action="apply_filters"><Filter className="h-4 w-4" />筛选</Button>
      <Button size="sm" variant="secondary" data-ui-action="refresh_filtered_view"><RefreshCw className="h-4 w-4" />刷新</Button>
    </div>
  );
}

function BottomSummary({
  phase,
  best,
  next,
  pending
}: {
  phase: string;
  best: string;
  next: string;
  pending: string;
}) {
  const summaryItems = [
    ["当前阶段", phase, "text-slate-950"],
    ["Best Protected", best, "text-slate-900"],
    ["下一步动作", next, "text-amber-700"],
    ["人工审批", pending, "text-red-700"]
  ] as const;

  return (
    <div className="sticky bottom-0 z-10 mt-1.5 max-w-full overflow-hidden rounded-md border border-slate-200 bg-white/95 p-2 shadow-[0_-10px_28px_-28px_rgba(15,23,42,0.65)] backdrop-blur">
      <div className="grid min-w-0 gap-2 lg:grid-cols-[280px_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <div className="flex min-w-0 items-center gap-3 rounded-md border border-slate-100 bg-slate-50 px-2.5 py-2">
          <Activity className="h-5 w-5 text-blue-600" />
          <div className="min-w-0">
            <div className="text-[11px] font-black text-slate-500">{summaryItems[0][0]}</div>
            <div className="truncate text-sm font-black text-slate-950" title={summaryItems[0][1]}>{summaryItems[0][1]}</div>
          </div>
        </div>
        {summaryItems.slice(1).map(([label, value, tone]) => (
          <div key={label} className="min-w-0 rounded-md border border-slate-100 bg-white px-2.5 py-2">
            <div className="text-[11px] font-black text-slate-500">{label}</div>
            <div className={cn(mono, "mt-0.5 max-h-[38px] overflow-hidden break-all leading-[17px]", tone)} title={value}>
              {value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MiniTerminal({ lines }: { lines: string[] }) {
  return (
    <pre className={cn(mono, "thin-scrollbar max-h-[154px] overflow-auto rounded-md bg-[#07111f] p-2.5 leading-[18px] text-emerald-200")}>
      {lines.join("\n")}
    </pre>
  );
}

function AgentLog() {
  const rows = ["Code Agent generated train_v2.py", "GPU Agent job 4821 started", "Validation Agent wrote oof_preds.parquet", "Data Agent audit_v2.json verified", "Orchestrator dispatched tasks"];
  return (
    <Panel title="Realtime Agent Log" description="Live">
      <div className="space-y-2">
        {rows.map((row, i) => <div key={row} className="grid grid-cols-[68px_100px_1fr] text-xs"><span className="text-slate-500">10:4{i}:18</span><span className="font-black text-blue-700">{i % 2 ? "GPU Agent" : "Code Agent"}</span><span className="truncate text-slate-700">{row}</span></div>)}
      </div>
    </Panel>
  );
}

function ArtifactCenter() {
  return (
    <Panel title="Artifact Center" description="产物、版本与校验">
      <DenseTable
        headers={["Artifact", "Type", "Agent", "Status"]}
        rows={[
          ["plan_v3.md", "Plan", "Research", "Verified"],
          ["feature_audit.json", "Audit", "Data", "Verified"],
          ["train_v2.py", "Code", "Code", "Generated"],
          ["oof_preds.parquet", "Data", "Validation", "Waiting"],
          ["submission_v3.csv", "Submission", "Report", "Ready"]
        ]}
      />
    </Panel>
  );
}

function TimelinePanel() {
  return (
    <Panel title="Experiment Timeline" description="Agent Gantt">
      <div className="space-y-2">
        {agents.slice(0, 7).map(([name], i) => (
          <div key={name} className="grid grid-cols-[120px_1fr] items-center gap-3 text-xs">
            <span className="truncate font-bold text-slate-600">{name}</span>
            <div className="h-2 rounded-full bg-slate-100"><div className={cn("h-2 rounded-full", i % 3 === 0 ? "bg-blue-500" : i % 3 === 1 ? "bg-emerald-500" : "bg-amber-500")} style={{ width: `${48 + i * 6}%` }} /></div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function AgentAssignment() {
  return <Panel title="Agent Assignment" description="分工表" className="xl:col-span-1"><DenseTable headers={["Agent", "Status"]} rows={agents.slice(0, 7).map(([a, , s]) => [a, s])} /></Panel>;
}

function LatestResult() {
  return <Panel title="Latest Result" description="CV 5-Fold"><DenseTable headers={["Fold", "OOF", "Status"]} rows={[["Fold1", "0.78123", "Done"], ["Fold2", "0.78541", "Done"], ["Fold3", "0.78702", "Done"], ["Mean", "0.78456", "Done"]]} /></Panel>;
}

function FailureRollback() {
  return <Panel title="Failure Rollback" description="失败与回退"><DenseTable headers={["Time", "Reason", "Action"]} rows={[["10:38", "OOM", "Reverted"], ["10:34", "GPU OOM", "Requeued"], ["10:21", "schema", "Fixed"]]} /></Panel>;
}

function NextOptimization() {
  return <Panel title="Next Optimization" description="下一轮建议"><ul className="space-y-2 text-xs font-semibold text-slate-700"><li>• 尝试 LightGBM + CatBoost 堆叠</li><li>• 增加 Graph Centrality / Embedding</li><li>• 调参空间：LR、Depth、L2</li></ul></Panel>;
}

function GpuMonitor() {
  return <Panel title="GPU/HPC Monitor" description="资源使用"><div className="space-y-2"><Row label="GPU" value="75%" /><Progress value={75} tone="blue" /><Row label="Memory" value="63%" /><Progress value={63} tone="green" /><Row label="Jobs" value="1 running / 1 queued" /></div></Panel>;
}

function DenseTable({ headers, rows }: { headers: string[]; rows: React.ReactNode[][] | readonly (readonly React.ReactNode[])[] }) {
  return (
    <div className="thin-scrollbar overscroll-contain overflow-x-auto" onWheel={stopWheelPropagation}>
      <table className="w-full min-w-full table-fixed text-left text-[9.5px]">
        <thead>
          <tr className="border-b border-slate-200 text-[9px] uppercase tracking-[0.02em] text-slate-500">
            {headers.map((h) => <th key={h} className="px-1.5 py-1 font-black">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className="cursor-pointer border-b border-slate-100 transition last:border-0 hover:bg-blue-50/55"
              data-ui-component="dense-table-row"
              data-ui-action="open_table_row"
            >
              {row.map((cell, j) => (
                <td key={j} className="min-w-0 break-words px-1.5 py-[2px] align-middle font-semibold leading-[13px] text-slate-700">
                  {typeof cell === "string" && j === row.length - 1 ? <StatusBadge tone={toneFor(cell)}>{cell}</StatusBadge> : cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function textValue(value: unknown, fallback = "n/a") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") return Number.isFinite(value) ? value.toString() : fallback;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") {
    const record = asRecord(value);
    if (!record) return fallback;
    return textValue(record.artifact ?? record.artifact_path ?? record.path ?? record.id ?? record.label ?? record.status, JSON.stringify(record).slice(0, 96));
  }
  return String(value);
}

function shortPath(value: unknown, fallback = "尚未生成") {
  const text = textValue(value, fallback).replaceAll("\\", "/");
  if (text.length <= 64) return text;
  return `.../${text.split("/").slice(-4).join("/")}`;
}

function formatTime(value: unknown) {
  const raw = typeof value === "string" ? value : "";
  if (!raw) return "--";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return raw.slice(0, 19);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function selectedRuntime(props: ScreenProps) {
  return props.summary?.runtime_by_task?.[props.selectedTask] ?? props.summary?.runtime ?? null;
}

function latestRunForTask(props: ScreenProps) {
  const runs = props.summary?.runs ?? [];
  return runs.find((run) => run.task_id === props.selectedTask)
    ?? runs.find((run) => run.task_id?.replaceAll("-", "_") === props.selectedTask)
    ?? runs[0]
    ?? null;
}

function latestActionForTask(props: ScreenProps) {
  if (props.lastActionTrace && (!props.lastActionTrace.taskId || props.lastActionTrace.taskId === props.selectedTask)) {
    return {
      action: props.lastActionTrace.action,
      message: props.lastActionTrace.message,
      artifact: props.lastActionTrace.artifact,
      at: props.lastActionTrace.at,
      task_id: props.lastActionTrace.taskId,
      run_id: textValue(asRecord(props.lastActionTrace.response)?.run_id, "")
    };
  }
  const actions = props.summary?.actions ?? [];
  return actions.find((action) => action.task_id === props.selectedTask)
    ?? actions.find((action) => !action.task_id)
    ?? actions[0]
    ?? null;
}

function filteredEvidence(props: ScreenProps) {
  const run = latestRunForTask(props);
  const evidence = props.summary?.evidence ?? [];
  return evidence.filter((item) => item.task_id === props.selectedTask || item.run_id === run?.id);
}

function filteredGates(props: ScreenProps) {
  const run = latestRunForTask(props);
  const gates = props.summary?.gates ?? [];
  return gates.filter((item) => item.task_id === props.selectedTask || item.run_id === run?.id);
}

function runtimeTraceRows(props: ScreenProps, limit = 6): React.ReactNode[][] {
  const runtime = selectedRuntime(props);
  const trace = runtime?.agent_trace ?? [];
  if (trace.length) {
    return trace.slice(-limit).reverse().map((item, index) => {
      const row = asRecord(item) ?? {};
      return [
        formatTime(row.at ?? row.time ?? row.timestamp ?? row.created_at ?? `trace-${index}`),
        textValue(row.agent ?? row.agent_id ?? row.stage ?? row.role, "Agent"),
        textValue(row.action ?? row.event ?? row.message ?? row.status, "trace"),
        textValue(row.status ?? row.decision ?? "recorded"),
        shortPath(row.artifact ?? row.artifact_path ?? row.path ?? row.output, "artifact pending")
      ];
    });
  }
  const actions = props.summary?.actions ?? [];
  return actions
    .filter((action) => action.task_id === props.selectedTask || !action.task_id)
    .slice(0, limit)
    .map((action) => [
      formatTime(action.at),
      "Workstation",
      action.action,
      "recorded",
      shortPath(action.artifact ?? action.message)
    ]);
}

function metricSummary(run: ReturnType<typeof latestRunForTask>) {
  const metrics = run?.best_metrics ?? null;
  if (!metrics) return "暂无 metric";
  const entries = Object.entries(metrics)
    .filter(([, value]) => typeof value === "number")
    .slice(0, 3)
    .map(([key, value]) => `${key}=${Number(value).toFixed(5)}`);
  return entries.length ? entries.join(" / ") : "metric 已记录";
}

function officialScoreSummary(run: ReturnType<typeof latestRunForTask>) {
  const metrics = run?.best_metrics ?? {};
  const officialScore = metrics?.official_public_score ?? metrics?.public_score ?? metrics?.kaggle_public_score;
  const rank = metrics?.official_rank ?? metrics?.rank;
  if (typeof officialScore === "number" || typeof officialScore === "string") {
    return rank ? `official ${officialScore} · rank ${rank}` : `official ${officialScore}`;
  }
  return "无官方 Kaggle response，不显示排名/奖牌";
}

function terminalAgentStatusLabel(status: string | undefined) {
  if (status === "live_events") return "Live event stream";
  if (status === "summary_only") return "Summary-only replay";
  if (status === "pending_run") return "Pending run artifacts";
  if (status === "no_runs") return "No xsci run";
  return status ?? "unknown";
}

function formatScore(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(6) : "n/a";
}

function terminalAgentEventRows(agent: WorkstationSummary["terminal_agent"] | undefined, limit = 8): React.ReactNode[][] {
  if (!agent) return [["--", "xsci", "等待 summary API", "no_runs", "experiments/evolution"]];
  const events = agent.recent_events ?? [];
  if (events.length) {
    return events.slice(-limit).reverse().map((event, index) => [
      formatTime(event.ts ?? event.at ?? event.timestamp ?? `event-${index}`),
      textValue(event.type ?? event.agent ?? "event"),
      textValue(event.exp_id ?? event.tool ?? event.mode ?? event.iteration ?? "-"),
      textValue(event.status ?? event.decision ?? event.reason ?? (event.promoted === true ? "promoted" : "recorded")),
      shortPath(event.artifact ?? event.path ?? event.summary ?? event.hypothesis ?? agent.events_path ?? "events.jsonl")
    ]);
  }
  const iterations = agent.iterations ?? [];
  if (iterations.length) {
    return iterations.slice(-limit).reverse().map((item) => [
      "summary",
      textValue(item.exp_id, "EXP"),
      textValue(item.mode, "mode"),
      item.promoted === true ? "promoted" : item.success === false ? "failed" : "held",
      `cv=${formatScore(item.cv_score)} ${textValue(item.note, "")}`
    ]);
  }
  return [[formatTime(agent.latest_run_mtime), "xsci", "no event stream yet", terminalAgentStatusLabel(agent.status), shortPath(agent.latest_run_dir ?? agent.evolution_root)]];
}

function terminalMemoryRows(agent: WorkstationSummary["terminal_agent"] | undefined): React.ReactNode[][] {
  const memory = agent?.recent_memory ?? [];
  if (!memory.length) return [["memory", "none", "等待 retrospective_memory.json", "pending"]];
  return memory.slice(0, 6).map((item) => [
    shortPath(item.memory_id ?? "memory"),
    textValue(item.task_type, "task"),
    textValue(item.method, "method"),
    textValue(item.failure_pattern || item.reusable_strategy || item.what_worked || item.what_failed, "recorded")
  ]);
}

function terminalCommandRows(agent: WorkstationSummary["terminal_agent"] | undefined): React.ReactNode[][] {
  const commands = agent?.commands ?? [];
  if (!commands.length) return [["xsci", "$env:PYTHONPATH='src'; python -m xsci --help", "discover", "ready"]];
  return commands.map((item) => [
    item.label,
    <button
      key={item.label}
      className={cn(mono, "max-w-full truncate text-left text-blue-700 underline-offset-2 hover:underline")}
      data-ui-action={`copy_xsci_command_${item.label.toLowerCase()}`}
      data-ui-skip-action="true"
      title={item.command}
      onClick={() => void navigator.clipboard?.writeText(item.command)}
    >
      {item.command}
    </button>,
    item.description,
    "copy"
  ]);
}

function TerminalKaggleAgentPanel(props: ScreenProps) {
  const agent = props.summary?.terminal_agent;
  const status = agent?.status ?? "no_runs";
  const taskId = agent?.task_id ?? props.selectedTask;
  const scoreLine = `${agent?.best_exp_id || "best pending"} / ${formatScore(agent?.best_cv_score)}`;
  const eventMode = agent?.events_present ? "events.jsonl active" : agent?.summary_present ? "summary.json only" : "waiting for artifacts";
  const terminalLines = [
    "EvoMind / XCIENTIST Research Agent",
    `Provider : audited xsci gateway / secrets masked`,
    `Workspace: ${shortPath(props.summary?.workspace_root ?? "Research OS Workspace")}`,
    `Task     : ${taskId}`,
    `Metric   : ${agent?.metric ?? "cv_score"} (${agent?.metric_direction ?? "maximize"})`,
    `Run      : ${agent?.latest_run_id ?? "waiting"}`,
    `Score    : ${scoreLine}`,
    `Events   : ${eventMode}`,
    `Memory   : ${agent?.memory_count ?? 0} retrospective records`,
    "",
    "> give EvoMind a data science task, then watch plan -> code -> train -> gate -> report"
  ];

  return (
    <Panel
      title="EvoMind Research Agent"
      description="8088 工作站和 EvoMind 终端共用 experiments/evolution 证据源；没有 events.jsonl 时只回放 summary，不伪装实时运行"
      action={
        <div className="flex flex-wrap items-center gap-1.5">
          <StatusBadge tone={toneFor(status)}>{terminalAgentStatusLabel(status)}</StatusBadge>
          <Button size="sm" variant="secondary" data-ui-skip-action="true" data-ui-action="terminal_agent_refresh" onClick={() => void props.refreshSummary?.()}>
            <RefreshCw className="h-3.5 w-3.5" />刷新
          </Button>
        </div>
      }
    >
      <div className="grid gap-2 xl:grid-cols-[minmax(0,1.1fr)_minmax(360px,0.9fr)]">
        <div className="overflow-hidden rounded-md border border-slate-800 bg-[#07090b] shadow-[0_12px_32px_-28px_rgba(2,6,23,0.9)]">
          <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber-300" />
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <span className={cn(mono, "ml-2 text-[11px] font-black text-slate-300")}>xsci://evomind-agent</span>
            </div>
            <span className={cn(mono, "text-[11px] text-slate-500")}>8088 linked</span>
          </div>
          <pre
            className={cn(mono, "thin-scrollbar max-h-[256px] overflow-auto whitespace-pre-wrap px-3 py-3 text-[12px] leading-[19px] text-slate-300")}
            onWheel={stopWheelPropagation}
          >
            <span className="text-[#f5f7a3]">{terminalLines[0]}</span>
            {"\n"}
            <span className="text-slate-100">{terminalLines[1]}</span>
            {"\n"}
            <span className="text-slate-500">{terminalLines.slice(2, 9).join("\n")}</span>
            {"\n\n"}
            <span className="text-slate-400">{terminalLines[10]}</span>
            {"\n"}
            <span className="text-slate-100">&gt; </span>
            <span className="inline-block h-4 w-2 translate-y-0.5 bg-slate-300" />
          </pre>
        </div>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-1">
          <div className="rounded-md border border-slate-200 bg-white p-2">
            <Row label="最新任务" value={taskId} monoValue />
            <Row label="最新 run" value={agent?.latest_run_id ?? "等待 xsci run"} monoValue />
            <Row label="待完成 run" value={agent?.latest_pending_run_id ? shortPath(agent.latest_pending_run_id) : "无 pending 目录"} monoValue />
            <Row label="证据模式" value={<StatusBadge tone={agent?.events_present ? "green" : agent?.summary_present ? "amber" : "slate"}>{eventMode}</StatusBadge>} />
            <Row label="Best CV" value={scoreLine} monoValue />
            <Row label="运行目录" value={shortPath(agent?.latest_run_dir ?? agent?.evolution_root ?? "experiments/evolution")} monoValue />
            <Row label="Claim 边界" value={agent?.claim_boundary ?? "等待 claim audit"} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <StatMini label="Runs" value={`${agent?.completed_run_count ?? 0}/${agent?.run_count ?? 0}`} tone="blue" />
            <StatMini label="Events" value={String(agent?.event_count ?? 0)} tone={agent?.events_present ? "green" : "amber"} />
            <StatMini label="Memory" value={String(agent?.memory_count ?? 0)} tone={(agent?.memory_count ?? 0) ? "green" : "slate"} />
            <StatMini label="Promote" value={`${agent?.n_promotions ?? 0}/${agent?.n_iterations ?? 0}`} tone={(agent?.n_promotions ?? 0) ? "green" : "amber"} />
          </div>
        </div>
      </div>
      <div className="mt-2 grid gap-2 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="min-w-0 rounded-md border border-slate-200 bg-white p-2">
          <div className="mb-1.5 flex items-start justify-between gap-2">
            <div>
              <div className="text-xs font-black text-slate-950">真实事件 / 摘要回放</div>
              <div className="text-[11px] font-semibold text-slate-500">{agent?.events_present ? "events.jsonl live/replay" : "当前 run 没有事件流，使用 summary.json 迭代摘要"}</div>
            </div>
            <StatusBadge tone={agent?.events_present ? "green" : "amber"}>{agent?.events_present ? "live" : "summary"}</StatusBadge>
          </div>
          <DenseTable headers={["time", "type/exp", "mode/tool", "status", "evidence"]} rows={terminalAgentEventRows(agent)} />
        </div>
        <div className="min-w-0 rounded-md border border-slate-200 bg-white p-2">
          <div className="mb-1.5 flex items-start justify-between gap-2">
            <div>
              <div className="text-xs font-black text-slate-950">Retrospective Memory</div>
              <div className="text-[11px] font-semibold text-slate-500">失败归因与成功策略会进入下一轮 Search Controller</div>
            </div>
            <StatusBadge tone={(agent?.memory_count ?? 0) ? "green" : "slate"}>{agent?.memory_count ?? 0}</StatusBadge>
          </div>
          <DenseTable headers={["memory", "task", "method", "lesson"]} rows={terminalMemoryRows(agent)} />
        </div>
      </div>
      <div className="mt-2 min-w-0 rounded-md border border-slate-200 bg-white p-2">
        <div className="mb-1.5">
          <div className="text-xs font-black text-slate-950">CLI Command Bridge</div>
          <div className="text-[11px] font-semibold text-slate-500">这些命令只作为操作入口显示；训练仍需你在终端/工作站 Gate 中明确启动</div>
        </div>
        <DenseTable headers={["入口", "命令", "用途", "操作"]} rows={terminalCommandRows(agent)} />
      </div>
    </Panel>
  );
}

function LiveRunEvidencePanel(props: ScreenProps) {
  const run = latestRunForTask(props);
  const action = latestActionForTask(props);
  const runtime = selectedRuntime(props);
  const evidence = filteredEvidence(props);
  const gates = filteredGates(props);
  const runStatus = props.runState?.status === "running" ? "running" : run?.status ?? runtime?.task_state?.state ?? "ready";
  const runId = run?.id ?? runtime?.latest_experiment_dir?.split(/[\\/]/).pop() ?? "等待首个 run";
  const artifactPath = action?.artifact ?? run?.artifact_manifest ?? run?.workstation_run_manifest ?? runtime?.latest_experiment_dir ?? runtime?.latest_workstation_run_dir;
  const latestGate = gates[0];
  const gateDecision = textValue(latestGate?.decision ?? run?.validation_gate?.status ?? "pending");

  return (
    <Panel
      title="实时运行证据 / Live Run Evidence"
      description="前端只展示工作站真实 action、run、artifact、gate 与 runtime trace；无 Kaggle response 时不显示排名或奖牌"
      action={
        <div className="flex flex-wrap gap-1.5">
          <StatusBadge tone={toneFor(String(runStatus))}>{String(runStatus)}</StatusBadge>
          <Button size="sm" variant="secondary" data-ui-action="live_evidence_refresh" data-ui-skip-action="true" onClick={() => void props.refreshSummary?.()}>
            <RefreshCw className="h-3.5 w-3.5" />刷新
          </Button>
          <Button
            size="sm"
            variant="primary"
            data-ui-action="live_evidence_create_workstation_run"
            data-ui-skip-action="true"
            onClick={() => void props.runWorkstationAction?.("create_workstation_run", { task_id: props.selectedTask, source: "live_evidence_panel" })}
          >
            <Play className="h-3.5 w-3.5" />创建 Run
          </Button>
        </div>
      }
    >
      <div className="grid gap-2 xl:grid-cols-[1.1fr_1.1fr_0.9fr]">
        <div className="rounded-md border border-slate-200 bg-white p-2">
          <Row label="选中任务" value={props.selectedTask} monoValue />
          <Row label="最新 run" value={runId} monoValue />
          <Row label="run 状态" value={<StatusBadge tone={toneFor(String(runStatus))}>{String(runStatus)}</StatusBadge>} />
          <Row label="指标证据" value={metricSummary(run)} monoValue />
          <Row label="官方结果" value={<span className="text-amber-700">{officialScoreSummary(run)}</span>} />
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-2">
          <Row label="最新动作" value={textValue(action?.action, "等待用户动作")} monoValue />
          <Row label="动作反馈" value={action?.message ?? props.systemActionMessage ?? "系统动作已就绪。"} />
          <Row label="动作时间" value={formatTime(action?.at)} monoValue />
          <Row label="产物路径" value={shortPath(artifactPath)} monoValue />
          <Row label="运行器反馈" value={props.runState?.message ?? "等待工作站运行。"} />
        </div>
        <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-1">
          <StatMini label="Evidence" value={String(evidence.length)} tone={evidence.length ? "green" : "amber"} />
          <StatMini label="Gates" value={String(gates.length)} tone={gates.length ? toneFor(gateDecision) : "amber"} />
          <StatMini label="Trace Rows" value={String(runtime?.agent_trace?.length ?? props.summary?.actions?.length ?? 0)} tone="blue" />
        </div>
      </div>
      <div className="mt-2 grid gap-2 xl:grid-cols-2">
        <DenseTable
          headers={["time", "agent/source", "action", "status", "artifact"]}
          rows={runtimeTraceRows(props, 4)}
        />
        <DenseTable
          headers={["gate", "decision", "run", "evidence"]}
          rows={(gates.length ? gates.slice(0, 4) : [{ gate_type: "submission_gate", decision: gateDecision, run_id: run?.id, evidence: "waiting" }]).map((gate) => [
            textValue(gate.gate_type ?? gate.id, "gate"),
            textValue(gate.decision ?? gate.status, "pending"),
            textValue(gate.run_id ?? run?.id, "run pending"),
            shortPath(asRecord(gate.evidence)?.artifact ?? gate.evidence ?? artifactPath)
          ])}
        />
      </div>
    </Panel>
  );
}

export function OverviewBoardEnhanced(props: ScreenProps) {
  return <MissionControl {...props} />;
}

export function AiControlConsole(props: ScreenProps) {
  const run = latestRunForTask(props);
  const action = latestActionForTask(props);
  const runtime = selectedRuntime(props);
  const liveRunId = run?.id ?? runtime?.latest_experiment_dir?.split(/[\\/]/).pop() ?? "等待工作站 run";
  return (
    <Page title="AI Research Agent Console" subtitle="Natural-language control layer for auditable research workflows.">
      <LiveRunEvidencePanel {...props} />
      <EvolutionControlPanel taskId={props.selectedTask} refreshSummary={props.refreshSummary} />
      <div className="grid gap-3 xl:grid-cols-[1fr_470px]">
        <div className="space-y-2">
          <div className="grid gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 md:grid-cols-3">
            <Row label="Workspace" value={props.summary?.workspace_root ?? "Research OS Workspace"} />
            <Row label="Task" value={props.selectedTask} monoValue />
            <Row label="Run ID" value={liveRunId} monoValue />
          </div>
          <Panel title="Agent Command Composer" description="All actions are routed through workstation gates.">
            <div className="rounded-md border border-blue-300 bg-white p-3 shadow-[0_0_0_3px_rgba(37,99,235,0.06)]">
              <textarea
                className="min-h-[96px] w-full resize-none rounded-md border-0 bg-white p-2 text-sm font-semibold text-slate-800 outline-none"
                defaultValue="为 spaceship_titanic 创建第二轮优化任务，保持 best-so-far，生成代码草稿并准备 GPU Gate。"
              />
              <div className="mt-2 flex flex-wrap gap-2">
                {["Attachments (2)", "best_so_far.json", "feature_analysis.ipynb"].map((x) => (
                  <button key={x} data-ui-action={`control_open_attachment_${x.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-bold text-slate-700 hover:bg-white">{x}</button>
                ))}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {["Research Plan", "Search Controller", "Code Agent", "GPU Job", "Validation", "Submission Gate", "Evidence Audit", "Report Draft"].map((x, i) => (
                <StatusBadge key={x} tone={i < 2 ? "green" : i < 5 ? "blue" : "amber"}>{x}</StatusBadge>
              ))}
              <Button className="ml-auto" variant="primary" size="sm" data-ui-action="control_run_through_workstation" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("ai_control_execute")}>
                Run through Workstation
              </Button>
            </div>
            <div className="mt-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-600">
              安全边界：所有训练与提交必须通过工作站 Gate，不直接展开训练，不绕过 agent trace。
            </div>
          </Panel>
          <Panel title="Research Workflow" description="Task Spec -> Report, every stage is bound to artifacts.">
            <div className="grid gap-2 md:grid-cols-5 xl:grid-cols-9">
              {[
                ["Task Spec", "done", CheckCircle2],
                ["Data Audit", "done", Database],
                ["Research Plan", "running", BrainCircuit],
                ["Code Draft", "running", Code2],
                ["GPU/HPC Job", "queued", Server],
                ["Metrics / OOF", "pending", LineChart],
                ["Submission Gate", "needs gate", ShieldAlert],
                ["Kaggle Response", "pending", Globe2],
                ["Report", "pending", FileText]
              ].map(([name, state, Icon], index) => (
                <div key={name as string} className="relative rounded-md border border-slate-200 bg-white p-2 text-center">
                  <span className="mx-auto flex h-7 w-7 items-center justify-center rounded-full border border-blue-100 bg-blue-50 text-blue-700"><Icon className="h-3.5 w-3.5" /></span>
                  <div className="mt-1.5 text-[11px] font-black text-slate-900">{name as string}</div>
                  <StatusBadge tone={toneFor(state as string)}>{state as string}</StatusBadge>
                  {index < 8 ? <ArrowRight className="absolute -right-3 top-1/2 hidden h-3.5 w-3.5 -translate-y-1/2 text-slate-300 xl:block" /> : null}
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Agent Execution Board" description="Assigned agents, artifacts, progress and recovery state.">
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
              {agents.slice(0, 8).map(([name, task, status, artifact, progress]) => (
                <div key={name} className={cn("rounded-md border bg-white p-2.5", status === "failed" ? "border-red-200" : status === "waiting" ? "border-amber-200" : status === "running" ? "border-blue-200" : "border-slate-200")}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="truncate text-xs font-black text-slate-950">{name}</div>
                    <StatusBadge tone={toneFor(status)}>{status}</StatusBadge>
                  </div>
                  <div className="mt-1.5 min-h-7 text-[11px] leading-4 text-slate-500">{task}</div>
                  <div className={cn(mono, "mt-1.5 truncate rounded bg-slate-50 px-2 py-1 text-blue-700")}>{artifact}</div>
                  <div className="mt-1.5 flex items-center gap-2"><Progress value={progress} tone={toneFor(status)} /><span className="w-8 text-right text-[11px] font-black text-slate-500">{progress}%</span></div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
        <aside className="space-y-2">
          <Panel title="Intent & Risk Preview" description="Live">
            <Row label="Intent" value={action?.action ?? "等待工作站动作"} />
            <Row label="Target Task" value={props.selectedTask} monoValue />
            <Row label="Required Agents" value="Search / Code / Validation / GPU" />
            <Row label="Required Gates" value="Code Quality, HPC Execution, Submission Approval" />
            <Row label="Risk Level" value={<StatusBadge tone="amber">Medium - Human Gate Required</StatusBadge>} />
            <Row label="Mode" value="Audit-first · Gate-controlled · Reproducible" />
          </Panel>
          <Panel title="Evidence Summary" description="Artifact readiness">
            <div className="grid grid-cols-3 gap-2">
              <StatMini label="Artifacts" value="12" tone="green" />
              <StatMini label="Complete" value="6 / 9" tone="blue" />
              <StatMini label="Sync" value="1 min" tone="green" />
            </div>
            <Row label="Path" value="workspace/evidence/wr_2026-06-25T10-38-14..." monoValue />
          </Panel>
          <Panel title="Latest Action Trace" description="Recent workstation actions">
            <DenseTable headers={["Time", "Agent", "Action", "Status"]} rows={[["10:42:31", "Orchestrator", "parse_intent", "passed"], ["10:42:35", "Search Ctrl", "generate_space", "passed"], ["10:43:02", "Data Auditor", "audit_data", "passed"], ["10:43:18", "Code Agent", "gen_code", "waiting"], ["10:43:25", "GPU Agent", "request_gpu", "waiting"]]} />
          </Panel>
          <Panel title="Score & Leaderboard" description="Latest known">
            <Row label="Metric Evidence" value={metricSummary(run)} />
            <Row label="Official Result" value={officialScoreSummary(run)} />
            <Row label="Can Submit Now" value={<StatusBadge tone="red">No - waiting for Submission Gate</StatusBadge>} />
          </Panel>
          <Panel title="Gate Stack" description="Submission remains blocked until approvals pass.">
            <DenseTable headers={["Gate", "Owner", "Status"]} rows={[["Code Quality", "Auto", "passed"], ["HPC Execution", "System", "waiting"], ["Submission Approval", "Human", "pending"]]} />
          </Panel>
        </aside>
      </div>
      <div className="grid gap-3 xl:grid-cols-[1fr_1fr_360px]">
        <Panel title="Conversation & Action Log" description="Orchestrator and agents keep a visible audit stream.">
          <div className="space-y-2">
            {[
              ["10:42:31", "Orchestrator Agent", "任务解析完成，已生成执行计划并分配给 Agent。", "plan_overview.md", "success"],
              ["10:42:35", "Search Controller", "生成 128 个候选路径组合，已提交模型空间。", "search_space.json", "success"],
              ["10:43:02", "Data Audit Agent", "发现特征缺失比例低，未发现泄漏风险。", "data_audit_report.json", "success"],
              ["10:43:18", "Code Agent", "代码草稿生成中，包含训练、验证与推理脚本。", "train_v2.py", "running"],
              ["10:43:25", "System Gate", "GPU Gate 尚未通过，训练任务等待审批。", "gpu_job_spec.yaml", "waiting"]
            ].map(([time, agent, msg, artifact, state]) => (
              <div key={`${time}-${agent}`} className="grid grid-cols-[64px_132px_1fr_118px_60px] items-center gap-2 rounded-md border border-slate-100 bg-white px-2 py-1.5 text-[11px]">
                <span className={cn(mono, "text-slate-500")}>{time}</span>
                <span className="truncate font-black text-blue-800">{agent}</span>
                <span className="truncate text-slate-600">{msg}</span>
                <span className={cn(mono, "truncate rounded bg-slate-50 px-1.5 py-1 text-blue-700")}>{artifact}</span>
                <StatusBadge tone={toneFor(state)}>{state}</StatusBadge>
              </div>
            ))}
          </div>
          <div className="mt-3 flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-xs text-slate-400">
            <span className="flex-1">Write a message to agents...</span>
            <ArrowRight className="h-4 w-4 text-blue-600" />
          </div>
        </Panel>
        <Panel title="Agent Runtime Timeline" description="Gantt-style execution trace.">
          <div className="relative overflow-hidden rounded-md border border-slate-100 bg-white p-2">
            <div className="absolute left-[132px] right-3 top-8 grid grid-cols-6 text-center text-[10px] font-bold text-slate-400">
              {["10:40","10:41","10:42","10:43","10:44","10:45"].map((x) => <span key={x}>{x}</span>)}
            </div>
            <div className="mt-8 space-y-2">
              {agents.slice(0, 8).map(([name], i) => (
                <div key={name} className="grid grid-cols-[128px_1fr] items-center gap-3 text-xs">
                  <span className="truncate font-bold text-slate-600">{name}</span>
                  <div className="relative h-5 rounded bg-slate-50">
                    <div className={cn("absolute top-1 h-3 rounded-full", i < 2 ? "bg-emerald-500" : i < 5 ? "bg-blue-500" : i === 5 ? "bg-amber-400" : "bg-slate-300")} style={{ left: `${Math.max(2, i * 7)}%`, width: `${34 + (i % 3) * 11}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Panel>
        <div className="space-y-2">
          <Panel title="Gate Stack" description="必须通过 Gate 才能提交到 Kaggle.">
            <DenseTable headers={["Gate", "Owner", "Status"]} rows={[["Code Quality Gate", "Auto Check", "passed"], ["HPC Execution Gate", "System", "waiting"], ["Submission Approval Gate", "Human", "pending"]]} />
          </Panel>
          <GpuMonitor />
        </div>
      </div>
    </Page>
  );
}

function AiControlConsoleLegacy(props: ScreenProps) {
  return (
    <Page title="EvoMind 工作站入口" subtitle="用自然语言调度科研工作站，所有动作写入审计日志">
      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <Panel title="Command Console" description="科研任务、训练、报告、证据与 Gate 的统一入口">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <textarea className="min-h-[160px] w-full resize-none rounded-md border border-slate-200 bg-white p-3 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" placeholder="例如：让工作站为 spaceship_titanic 创建二轮自进化计划，但不要提交 Kaggle，先生成 evidence bundle..." />
            <div className="mt-3 flex flex-wrap gap-2">
              {["Create Run", "Dispatch Agents", "Generate Report", "Open Gate"].map((x) => <Button key={x} variant="secondary" size="sm" data-ui-action={`legacy_control_${x.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`}>{x}</Button>)}
              <Button variant="primary" size="sm" data-ui-action="legacy_control_execute_guarded_action" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("ai_control_execute")}>执行受控动作</Button>
            </div>
          </div>
        </Panel>
        <Panel title="Action Trace" description="最近一次工作站动作">
          <Row label="Action" value={props.lastActionTrace?.action ?? "navigate_page"} monoValue />
          <Row label="Artifact" value={props.lastActionTrace?.artifact ?? "workspace/ui_state/latest.json"} monoValue />
          <Row label="Message" value={props.systemActionMessage ?? "ready"} />
        </Panel>
      </div>
      <AgentGraph />
    </Page>
  );
}

export function DataKagglePipeline(props: ScreenProps) {
  const readiness = [
    ["数据准备", "ready", "train.parquet, test.parquet"],
    ["数据下载", "completed", "下载完成，sha256 校验通过"],
    ["Schema 校验", "passed", "字段、类型与 sample 对齐"],
    ["特征审计", "passed", "未发现明显泄漏"],
    ["OOF / CV", "completed", "5-fold CV 完成"],
    ["Submission Candidate", "ready", "submission_v12.csv"],
    ["人工 Gate 状态", "待审批", "等待人工审核"],
    ["风险 Flags", "1 个", "CV-public gap 偏大"]
  ];
  const flow = [
    ["数据下载", "completed", "10:21", Database],
    ["Schema 校验", "passed", "10:22", CheckCircle2],
    ["特征审计", "passed", "10:23", GitBranch],
    ["OOF / CV", "completed", "10:36", LineChart],
    ["Submission Candidate", "ready", "10:38", FileText],
    ["人工 Gate", "待审批", "-", UserCheck],
    ["官方提交", "pending", "-", Upload]
  ] as const;
  return (
    <Page title="数据与 Kaggle" subtitle="数据接入、schema 审计、Kaggle 下载与 submission 门禁">
      <LiveRunEvidencePanel {...props} />
      <MicroStatusRail
        items={[
          ["Dataset Hash", "sha256 verified", "green"],
          ["Kaggle API", "connected", "green"],
          ["Submission Budget", "1 / 2 left", "amber"],
          ["Official Rank", "blocked until submit", "red"]
        ]}
      />
      <div className="grid gap-2 xl:grid-cols-4">
        <HeroStatusCard icon={ShieldCheck} label="分数门禁" title="保护历史 best" detail="当前 best：0.965910，候选提交必须先通过 promotion gate。" tone="green" meta="Score Guard" />
        <HeroStatusCard icon={Database} label="证据索引" title="产物可审计" detail="Artifacts: 12 / 12，数据、OOF、submission 与 claim 均可追踪。" tone="blue" meta="Evidence Index" />
        <HeroStatusCard icon={UserCheck} label="人工 Gate" title="提交需审批" detail="官方提交必须等待人工 Gate，不把本地 CV 伪装成 Kaggle 排名。" tone="amber" meta="Pending 1" />
        <HeroStatusCard icon={Lock} label="上线边界" title="算力与提交受控" detail="Kaggle API、HPC/GPU 与提交路径均在工作站门禁下执行。" tone="slate" meta="Gate Controlled" />
      </div>

      <div className="grid gap-2 xl:grid-cols-[0.92fr_1.08fr]">
        <Panel title="Kaggle 数据流概览" description="当前 run、任务、指标与人工 Gate" action={<StatusBadge tone="green">运行中 / running</StatusBadge>} className="h-[254px] overflow-hidden">
          <div className="grid gap-x-5 xl:grid-cols-[1fr_1fr]">
            <Row label="Run ID" value="wr_2026-06-25T10-38-14-633Z_ll7ln" monoValue />
            <Row label="Task ID" value="task_spaceship_titanic_2nd_round" monoValue />
            <Row label="Competition" value={<span>Spaceship Titanic · <span className="text-blue-700">kaggle.com/c/spaceship-titanic</span></span>} />
            <Row label="Metric" value="rmse / official leaderboard" />
            <Row label="Train Rows" value="8,703,498" />
            <Row label="Test Rows" value="4,271,574" />
            <Row label="Target" value="transported" />
            <Row label="Human Gate" value={<span className="inline-flex items-center gap-2"><StatusBadge tone="amber">待审批</StatusBadge><span className="text-slate-500">Submission approval required</span></span>} />
          </div>
        </Panel>
        <Panel title="Kaggle Readiness / Gate Summary" description="数据准备、证据审计与风险边界" className="h-[254px] overflow-hidden">
          <DenseTable headers={["检查项", "状态", "说明"]} rows={readiness.map(([label, state, detail]) => [label, state, detail])} />
        </Panel>
      </div>

      <div className="grid gap-2 xl:grid-cols-[1.2fr_0.72fr_0.66fr]">
        <Panel title="数据接入与提交流水线" description="每一步都有 agent、时间戳和可回放 artifact" className="xl:row-span-1">
          <div className="grid gap-2 md:grid-cols-7">
            {flow.map(([name, state, time, Icon], index) => (
              <div key={name} className="relative rounded-md border border-slate-200 bg-white p-2 text-center">
                <span className="mx-auto flex h-8 w-8 items-center justify-center rounded-full border border-blue-100 bg-blue-50 text-blue-700"><Icon className="h-4 w-4" /></span>
                <div className="mt-1.5 min-h-8 text-xs font-black leading-4 text-slate-950">{name}</div>
                <StatusBadge tone={toneFor(state)}>{state}</StatusBadge>
                <div className={cn(mono, "mt-1 text-slate-500")}>{time}</div>
                {index < flow.length - 1 ? <ArrowRight className="absolute -right-4 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-slate-300 xl:block" /> : null}
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="实验证据 / Artifacts" description="产物、来源 Agent 与状态">
          <DenseTable headers={["Artifact", "Type", "Agent", "Status", "Time"]} rows={[["feature_audit.json", "audit", "Data", "passed", "10:23:51"], ["metrics.json", "metrics", "Validation", "completed", "10:36:12"], ["oof_pred.parquet", "prediction", "Validation", "completed", "10:36:12"], ["submission_v12.csv", "submission", "Report", "ready", "10:38:02"], ["claim_audit.json", "audit", "Claim", "passed", "10:38:03"], ["lineage_graph.json", "lineage", "Orchestrator", "completed", "10:38:05"]]} />
        </Panel>
        <Panel title="风险与门禁 / Risk & Gate" description="提交前审计">
          <DenseTable headers={["项目", "等级", "说明"]} rows={[["目标泄漏风险", "safe", "未检测到高风险泄漏"], ["CV-Official Gap", "blocked", "无官方 response，暂不计算"], ["Submission Schema", "safe", "字段与 sample 完全匹配"], ["Claim Drift", "safe", "证据边界内"], ["数据漂移", "safe", "分布稳定"], ["依赖 / 环境一致性", "safe", "版本锁定一致"]]} />
        </Panel>
      </div>

      <div className="grid gap-2 xl:grid-cols-[1fr_1fr_0.8fr]">
        <Panel title="最近实验结果 / Recent Experiments" description="候选只在 Gate 通过后进入提交列表"><DenseTable headers={["实验 ID", "模型路线", "CV", "OOF", "Official", "状态", "更新"]} rows={[["exp_20260625_1036", "lgbm_v5_stack", "0.9659", "0.9661", "no response", "running", "10:38:12"], ["exp_20260625_1012", "lgbm_v5", "0.9662", "0.9664", "no response", "completed", "10:24:05"], ["exp_20260625_0945", "xgb_v3", "0.9678", "0.9680", "no response", "completed", "09:56:33"], ["exp_20260625_0901", "lgbm_v4", "0.9669", "0.9671", "no response", "completed", "09:12:21"], ["exp_20260625_0840", "catboost_v2", "0.9685", "0.9687", "no response", "completed", "08:51:03"], ["exp_20260625_0815", "leakage_check", "-", "-", "-", "blocked", "08:22:11"]]} /></Panel>
        <Panel title="失败 / 回退记录 / Failure & Rollback" description="失败任务同样进入证据台账"><DenseTable headers={["时间", "原因", "实验 ID", "影响", "处理状态"]} rows={[["10:12:44", "cv_timeout", "exp_20260625_1001", "耗时超限", "resolved"], ["09:41:22", "schema_mismatch", "exp_20260625_0932", "字段不匹配", "resolved"], ["09:15:03", "metric_regression", "exp_20260625_0910", "分数回退", "reviewed"], ["08:43:55", "dependency_fail", "exp_20260625_0830", "依赖缺失", "resolved"], ["08:22:11", "leakage_detected", "exp_20260625_0815", "泄漏风险", "blocked"]]} /></Panel>
        <Panel title="下一轮优化建议 / Next Optimization" description="Search Controller 输出"><DenseTable headers={["建议方向", "优先级", "理由", "建议 Agent"]} rows={[["特征工程优化", "High", "先提升本地 CV/OOF 可信候选", "Data Agent"], ["模型集成 / 堆叠", "High", "泛化仍有提升空间", "Code Agent"], ["阈值搜索 / 后处理", "Medium", "需先绑定 validation contract", "Validation Agent"], ["Submission Gate Review", "High", "当前等待提交审批", "Gate Agent"], ["推迟报告生成", "Low", "生成可解释报告", "Report Agent"]]} /></Panel>
      </div>

      <BottomSummary phase="OOF / CV · 已完成 4 / 7 步骤" best="task_spaceship_titanic_2nd_round / wr_2026-06-25T10-38-14-633Z_ll7ln" next="等待人工 Gate 审批" pending="预计 5-15 分钟内完成" />
    </Page>
  );
}
export function CodeRunner(props: ScreenProps) {
  const files = [
    ["experiments/exp_20260625_1036", "folder", ""],
    ["train.py", "code", "generated"],
    ["features.py", "code", "edited"],
    ["model_lgbm.py", "code", "generated"],
    ["model_xgb.py", "code", "generated"],
    ["blend.py", "code", "edited"],
    ["submission.py", "code", "pending"],
    ["run_manifest.json", "config", "reviewed"],
    ["metrics.json", "metrics", "generated"],
    ["audits/quality_gate.json", "audit", "reviewed"],
    ["contracts/validation_contract.json", "contract", "reviewed"]
  ];
  const [selectedFile, setSelectedFile] = useState("train.py");
  const trace = [["10:38:15", "DeepSeek V3", "implementation_contract.json"], ["10:39:02", "DeepSeek V3", "train.py"], ["10:41:23", "Claude Code", "features.py"], ["10:43:10", "DeepSeek V3", "model_lgbm.py"], ["10:45:32", "Claude Code", "submission.py"]];
  const snippets: Record<string, string> = {
    "train.py": codeSample,
    "features.py": `import numpy as np
import pandas as pd

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature branch generated by Code Agent and bound to Data Audit."""
    out = df.copy()
    out["ratio_ab"] = out["a"] / (out["b"] + 1e-6)
    out["log_c"] = np.log1p(out["c"].clip(lower=0))
    out["inter_a_c"] = out["a"] * out["c"]
    out["missing_count"] = out.isna().sum(axis=1)
    return out

def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    train_x = add_interaction_features(train)
    test_x = add_interaction_features(test)
    return train_x, test_x`,
    "model_lgbm.py": `import lightgbm as lgb

def build_lgbm(seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=63,
        subsample=0.86,
        colsample_bytree=0.82,
        random_state=seed,
    )`,
    "model_xgb.py": `import xgboost as xgb

def build_xgb(seed: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1400,
        learning_rate=0.028,
        max_depth=6,
        subsample=0.88,
        colsample_bytree=0.84,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
    )`,
    "blend.py": `import numpy as np

def blend_probabilities(lgb_oof, xgb_oof, cb_oof, weights=(0.46, 0.34, 0.20)):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return weights[0] * lgb_oof + weights[1] * xgb_oof + weights[2] * cb_oof

def clip_for_submission(pred, eps=1e-5):
    return np.clip(pred, eps, 1 - eps)`,
    "submission.py": `import pandas as pd

def build_submission(sample_path: str, predictions, output_path: str = "submission.csv"):
    sample = pd.read_csv(sample_path)
    target_col = sample.columns[-1]
    sample[target_col] = predictions
    sample.to_csv(output_path, index=False)
    return output_path`,
    "run_manifest.json": `{
  "task_id": "TASK-2026-1036",
  "run_id": "wr_2026-06-25T10-38",
  "exp_id": "exp_20260625_1036",
  "agent_id": "CodeImplementationAgent",
  "entrypoint": "python train.py",
  "gate_dependency": ["code_quality_gate", "hpc_execution_gate"],
  "output_artifacts": ["metrics.json", "oof_pred.parquet", "submission.csv"]
}`,
    "metrics.json": `{
  "cv_logloss": 0.4562,
  "oof_auc": 0.8617,
  "fold_std": 0.0021,
  "schema_audit": "pending",
  "promotion_gate": "hold_until_human_review"
}`,
    "code_agent_trace.json": `{
  "trace_id": "trace_code_agent_20260625_1036",
  "files": ["train.py", "features.py", "model_lgbm.py", "model_xgb.py", "blend.py"],
  "source_agents": ["DeepSeek V3", "Claude Code"],
  "human_review": "required_before_hpc"
}`,
    "implementation_contract.json": `{
  "hypothesis": "HYP-024",
  "implementation": "LGBM + XGB + CatBoost probability blend",
  "expected": "logloss lift > 0.003",
  "rollback": "CV regression or submission schema risk"
}`,
    "diff_review.json": `{
  "added": 6,
  "modified": 4,
  "deleted": 1,
  "risk": "medium",
  "required_gate": "code_quality_gate"
}`,
    "quality_gate.json": `{
  "syntax": "passed",
  "unit_smoke": "passed",
  "secret_scan": "passed",
  "dependency": "pending",
  "hpc_handoff": "blocked_until_gate"
}`,
    "dependency_scan.json": `{
  "python": "3.10",
  "lightgbm": "available",
  "xgboost": "available",
  "catboost": "pending",
  "status": "needs_hpc_image_confirmation"
}`,
    "leakage_check.json": `{
  "target_leakage": "not_detected",
  "cv_public_gap_risk": "watch",
  "submission_schema": "pending",
  "claim_boundary": "proxy validation only before official response"
}`,
  };
  const selectedMeta = files.find(([file]) => file === selectedFile);
  const selectedState = selectedMeta?.[2] || "reviewed";
  const selectedKind = selectedMeta?.[1] || (selectedFile.endsWith(".json") ? "audit" : "code");
  const codeLines = (snippets[selectedFile] ?? `# ${selectedFile}\n\n# This artifact is tracked by the Code Agent workspace.\n# Select another file from Workspace Explorer or the editor tabs.`).split("\n").slice(0, 52);
  const openFile = (file: string) => {
    setSelectedFile(file);
    void props.runWorkstationAction?.("code_file_select", { file, source: "code_agent_ide" });
  };
  const fileAction = (file: string) => `open_code_file_${file.replaceAll("/", "_").replaceAll(".", "_")}`;
  const tabFiles = ["train.py", "features.py", "blend.py", "submission.py", "metrics.json"];
  return (
    <Page title="Code Agent IDE" subtitle="Algorithm code generation, diff review, quality gate, and execution sandbox in one traceable workspace.">
      <LiveRunEvidencePanel {...props} />
      <div className="flex flex-wrap items-center justify-end gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-[11px] font-black shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
        {[
          ["Draft", "slate"],
          ["Reviewing", "blue"],
          ["Gate Passed", "green"],
          ["Ready for HPC", "slate"],
          ["Blocked", "red"]
        ].map(([label, tone], index) => (
          <button
            key={label}
            className={cn(
              "flex h-6 items-center gap-1.5 rounded-full border px-2 transition hover:-translate-y-px",
              tone === "blue" ? "border-blue-200 bg-blue-50 text-blue-700" :
              tone === "green" ? "border-emerald-200 bg-emerald-50 text-emerald-700" :
              tone === "red" ? "border-red-200 bg-red-50 text-red-700" :
              "border-slate-200 bg-slate-50 text-slate-600"
            )}
            data-ui-action={`code_stage_${label.toLowerCase().replaceAll(" ", "_")}`}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", tone === "blue" ? "bg-blue-500" : tone === "green" ? "bg-emerald-500" : tone === "red" ? "bg-red-500" : "bg-slate-400")} />
            {label}
            {index < 4 ? <ChevronRight className="h-3 w-3 text-slate-400" /> : null}
          </button>
        ))}
      </div>
      <div className="grid gap-2 xl:grid-cols-4">
        <HeroStatusCard icon={FileCode2} label="1 代码来源可追溯" title="12 / 12" detail="已绑定 run / exp / agent，所有文件都有 trace。" tone="green" meta="良好" />
        <HeroStatusCard icon={ShieldCheck} label="2 质量审计状态" title="4 passed" detail="核心检查通过 4 项，待处理 2 项，阻断 0。" tone="amber" meta="部分通过" />
        <HeroStatusCard icon={Server} label="3 执行沙箱受控" title="HPC ready" detail="Smoke 通过，Manifest 就绪，GPU Gate 等待。" tone="blue" meta="就绪" />
        <HeroStatusCard icon={Lock} label="4 低分代码不覆盖 best" title="Score Gate" detail="Score Gate 与人工 Gate 双重保护 best-so-far。" tone="blue" meta="策略生效" />
      </div>
      <div className="grid items-start gap-2 xl:grid-cols-[minmax(278px,0.78fr)_minmax(620px,1.72fr)_minmax(318px,0.98fr)]">
        <div className="space-y-2">
          <Panel title="Workspace Explorer" description="Task / run / exp / branch bound file tree." action={<button className="text-xs font-black text-slate-400" data-ui-action="code_explorer_menu">⋮</button>} className="overflow-hidden" style={{ height: 420 }}>
            <div className="mb-1.5 grid grid-cols-5 gap-1">
              {['Task','Run','Exp','Agent','Branch'].map((x) => <button key={x} data-ui-action={`code_filter_${x.toLowerCase()}`} className="h-7 rounded-md border border-slate-200 bg-slate-50 text-[10px] font-black text-slate-700 hover:border-blue-200 hover:bg-white">{x}</button>)}
            </div>
            <div className="thin-scrollbar h-[342px] space-y-0.5 overflow-auto">
              <button className="flex h-6 w-full items-center gap-1.5 rounded px-1.5 text-left text-[11px] font-black text-slate-700 hover:bg-slate-50" data-ui-action="open_code_folder_experiments">
                <ChevronRight className="h-3.5 w-3.5 rotate-90 text-slate-400" /><Database className="h-3.5 w-3.5 text-slate-500" />experiments/exp_20260625_1036
              </button>
              {files.slice(1, 9).map(([file,,state]) => <button key={file} onClick={() => openFile(file)} data-ui-action={fileAction(file)} data-ui-skip-action="true" data-selected-file={selectedFile === file ? "true" : "false"} className={cn("grid h-7 w-full grid-cols-[18px_1fr_70px] items-center gap-1 rounded px-1.5 py-1 text-left text-[11px] transition hover:bg-blue-50", selectedFile === file && "bg-blue-50 text-blue-700 ring-1 ring-blue-200")}><FileText className="h-3.5 w-3.5 text-slate-400" /><span className={cn(mono, "truncate text-[10px] pl-1.5")}>{file}</span>{state ? <StatusBadge tone={toneFor(state)}>{state}</StatusBadge> : null}</button>)}
              <button className="mt-1 flex h-6 w-full items-center gap-1.5 rounded px-1.5 text-left text-[11px] font-black text-slate-700 hover:bg-slate-50" data-ui-action="open_code_folder_agents">
                <ChevronRight className="h-3.5 w-3.5 rotate-90 text-slate-400" /><Bot className="h-3.5 w-3.5 text-slate-500" />agents/
              </button>
              {["code_agent_trace.json", "implementation_contract.json", "diff_review.json"].map((file) => <button key={file} onClick={() => openFile(file)} data-ui-action={fileAction(file)} data-ui-skip-action="true" data-selected-file={selectedFile === file ? "true" : "false"} className={cn("grid h-7 w-full grid-cols-[18px_1fr_70px] items-center gap-1 rounded px-1.5 py-1 text-left text-[11px] transition hover:bg-blue-50", selectedFile === file && "bg-blue-50 text-blue-700 ring-1 ring-blue-200")}><FileText className="h-3.5 w-3.5 text-slate-400" /><span className={cn(mono, "truncate text-[10px] pl-1.5")}>{file}</span><StatusBadge tone="blue">reviewed</StatusBadge></button>)}
              <button className="mt-1 flex h-6 w-full items-center gap-1.5 rounded px-1.5 text-left text-[11px] font-black text-slate-700 hover:bg-slate-50" data-ui-action="open_code_folder_audits">
                <ChevronRight className="h-3.5 w-3.5 rotate-90 text-slate-400" /><ShieldCheck className="h-3.5 w-3.5 text-slate-500" />audits/
              </button>
              {["quality_gate.json", "dependency_scan.json", "leakage_check.json"].map((file) => <button key={file} onClick={() => openFile(file)} data-ui-action={fileAction(file)} data-ui-skip-action="true" data-selected-file={selectedFile === file ? "true" : "false"} className={cn("grid h-7 w-full grid-cols-[18px_1fr_70px] items-center gap-1 rounded px-1.5 py-1 text-left text-[11px] transition hover:bg-blue-50", selectedFile === file && "bg-blue-50 text-blue-700 ring-1 ring-blue-200")}><FileText className="h-3.5 w-3.5 text-slate-400" /><span className={cn(mono, "truncate text-[10px] pl-1.5")}>{file}</span><StatusBadge tone="blue">reviewed</StatusBadge></button>)}
            </div>
          </Panel>
          <Panel title="当前代码绑定信息" description="The selected file inherits bounded context." className="overflow-hidden" style={{ height: 164 }}>
            <Row label="task_id" value="TASK-2026-1036" monoValue />
            <Row label="run_id" value="wr_2026-06-25T10-38" monoValue />
            <Row label="exp_id" value="exp_20260625_1036" monoValue />
            <Row label="branch_type" value="experiment" />
            <Row label="selected_file" value={selectedFile} monoValue />
            <Row label="artifact_type" value={selectedKind} monoValue />
            <Row label="code_generation_mode" value={<span className="inline-flex gap-1"><StatusBadge tone="blue">Base</StatusBadge><StatusBadge tone="slate">Stepwise</StatusBadge><StatusBadge tone="slate">Diff</StatusBadge></span>} />
          </Panel>
        </div>
        <Panel title={selectedFile} description={`${selectedKind} artifact · ${selectedState || "tracked"} · reviewed by human and quality gate.`} action={<div className="flex flex-wrap gap-1.5"><Button size="sm" variant="secondary" data-ui-action="ask_code_agent">Ask Code Agent</Button><Button size="sm" variant="secondary" data-ui-action="review_code_diff">Review Diff</Button><Button size="sm" variant="secondary" data-ui-action="run_code_smoke_test">Run Smoke Test</Button><Button size="sm" variant="secondary" data-ui-action="request_code_quality_gate">Request Gate</Button><Button size="sm" variant="secondary" aria-disabled="true" data-ui-action="blocked_send_to_hpc"><Lock className="h-3.5 w-3.5" />Send to HPC</Button></div>}>
          <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
            <div className="flex items-center gap-1 border-b border-slate-200 bg-slate-50 px-2 py-1.5 text-xs font-bold text-slate-600">
              {tabFiles.map((file) => <button key={file} onClick={() => openFile(file)} data-ui-action={fileAction(file)} data-ui-skip-action="true" data-selected-file={selectedFile === file ? "true" : "false"} className={cn('rounded px-3 py-1 transition hover:bg-white hover:text-blue-700', selectedFile === file ? 'bg-white text-blue-700 shadow-sm ring-1 ring-blue-100' : '')}>{file}</button>)}
              <button data-ui-action="code_add_new_file" className="rounded px-3 py-1 transition hover:bg-white hover:text-blue-700">+</button>
            </div>
            <div className="border-b border-slate-100 bg-white px-3 py-1.5 text-[11px] font-semibold text-slate-500">
              Selected <span className={cn(mono, "font-black text-blue-700")}>{selectedFile}</span> · Code Agent trace retained · bound to <span className="font-black text-blue-700">HYP-024</span>
            </div>
            <div className="grid" style={{ height: 462, gridTemplateColumns: "1fr 66px" }}>
              <div className={cn(mono, "thin-scrollbar relative overflow-auto bg-white p-2.5 text-[11px] leading-[17px] text-slate-800")}>
                {codeLines.map((line, i) => (
                  <div key={i} className={cn("grid grid-cols-[34px_1fr] rounded px-1", i >= 17 && i <= 21 && "bg-emerald-50")}>
                    <span className="select-none text-right text-slate-400">{i + 1}</span>
                    <span className="whitespace-pre-wrap pl-3">{line || " "}</span>
                  </div>
                ))}
                <CodeCommentAnchor top="72px" label="绑定 hypothesis HYP-024" />
                <CodeCommentAnchor top="138px" label="读取 Data Agent 审计结果" />
                <CodeCommentAnchor top="214px" label="此处通过 OOF 验证" />
                <CodeCommentAnchor top="286px" label="官方提交前需 Submission Gate" />
              </div>
              <div className="border-l border-slate-200 bg-slate-50 p-2">
                <div className="h-full rounded bg-white p-1">
                  {Array.from({ length: 42 }).map((_, i) => <div key={i} className={cn("mb-1 h-1 rounded", i > 15 && i < 22 ? "bg-emerald-300" : i % 7 === 0 ? "bg-amber-300" : i % 5 === 0 ? "bg-blue-300" : "bg-slate-200")} />)}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-1.5 grid gap-1.5 lg:grid-cols-[1.1fr_0.72fr_0.72fr_0.72fr]">
            <Panel title="Terminal" description="Smoke output" className="overflow-hidden" style={{ height: 116 }}><pre className={cn(mono, "rounded-md bg-slate-950 p-2 text-[10px] leading-[15px] text-emerald-200")}>10:52:13 [info] open {selectedFile} ... OK{"\n"}10:52:18 [info] trace binding loaded{"\n"}10:53:04 [info] artifact preview ready{"\n"}10:53:05 [warn] HPC handoff still gate controlled</pre></Panel>
            <Panel title="Metrics Preview" description="OOF" className="overflow-hidden" style={{ height: 116 }}><DenseTable headers={["Metric","CV","Delta"]} rows={[["logloss","0.4562","+0.0023"],["auc","0.8617","-0.0018"],["rmse","0.8124","+0.0017"]]} /></Panel>
            <Panel title="Artifacts" description="Generated" className="overflow-hidden" style={{ height: 116 }}><DenseTable headers={["file","status"]} rows={[[selectedFile, selectedState || "tracked"],["metrics.json","generated"],["submission.csv","pending"],["diff_review.json","reviewed"]]} /></Panel>
            <Panel title="Gate Decisions" description="Policy" className="overflow-hidden" style={{ height: 116 }}><DenseTable headers={["Gate","status"]} rows={[["Code","passed"],["Secret","passed"],["Schema","pending"],["HPC","blocked"]]} /></Panel>
          </div>
        </Panel>
        <div className="max-h-[820px] space-y-1.5 overflow-hidden">
          <Panel title="Code Agent Trace" description="Every generated file keeps source agent and timestamp." action={<button className="text-xs font-black text-blue-700" data-ui-action="view_all_code_agent_trace">查看全部</button>} className="overflow-hidden" style={{ height: 170 }}><DenseTable headers={["time","agent","artifact"]} rows={trace} /></Panel>
          <Panel title="Implementation Contract" description="XCIENTIST-style guardrail." className="overflow-hidden" style={{ height: 132 }}><Row label="Hypothesis" value="HYP-024" monoValue /><Row label="Implementation" value="LGBM + XGB + CB blend" /><Row label="Expected" value="logloss lift > 0.003" /><Row label="Rollback" value="CV regression or schema risk" /></Panel>
          <Panel title="Diff Review" description="Patch quality summary." className="overflow-hidden" style={{ height: 116 }}><DenseTable headers={["change","count"]} rows={[["added","6"],["modified","4"],["deleted","1"],["risk","medium"]]} /></Panel>
          <Panel title="Quality Gate" description="No HPC handoff before passing." className="overflow-hidden" style={{ height: 150 }}><DenseTable headers={["check","status"]} rows={[["syntax","passed"],["unit smoke","passed"],["secret scan","passed"],["dependency","pending"],["leakage","waiting"]]} /></Panel>
          <Panel title="执行就绪度" description="受 Gate 控制" className="overflow-hidden" style={{ height: 116 }}><DenseTable headers={["item","status"]} rows={[["本地 Smoke","passed"],["HPC Manifest","ready"],["GPU 资源","available"],["超时预算","120 min"],["数据版本","v20260625-1"]]} /></Panel>
          <Panel title="Failure / Recovery" description="Failure recovery" className="overflow-hidden" style={{ height: 88 }}><Row label="Failure" value={<StatusBadge tone="red">controlled</StatusBadge>} /><Row label="Rollback" value="Return to previous Code Agent prompt" /></Panel>
        </div>
      </div>
    </Page>
  );
}

function CodeCommentAnchor({ top, label }: { top: string; label: string }) {
  return (
    <span className="pointer-events-none absolute right-[94px] rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-[11px] font-black text-blue-700 shadow-sm" style={{ top }}>
      ↳ {label}
    </span>
  );
}





export function GpuHpcConsole(props: ScreenProps) {
  const s = props.summary;
  const [computeMode, setComputeMode] = useState<"hpc_gpu" | "local">("hpc_gpu");
  const [computeFeedback, setComputeFeedback] = useState("当前默认使用 HPC/GPU 集群；本地算力需显式选择后才会放行本地训练。");

  useEffect(() => {
    let cancelled = false;
    fetch("/api/settings")
      .then((response) => response.json())
      .then((payload) => {
        const mode = payload?.settings?.compute?.execution_mode;
        if (!cancelled && (mode === "local" || mode === "hpc_gpu")) {
          setComputeMode(mode);
          setComputeFeedback(mode === "local" ? "本地算力模式已启用，可运行小数据任务测试。" : "HPC/GPU 集群模式已启用，本地训练 fallback 关闭。");
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  async function selectComputeMode(mode: "hpc_gpu" | "local") {
    setComputeMode(mode);
    setComputeFeedback(mode === "local" ? "正在切换到本地算力模式..." : "正在切换到 HPC/GPU 集群模式...");
    try {
      await props.runWorkstationAction?.("select_compute_mode", { mode, small_task_only: true });
      setComputeFeedback(mode === "local" ? "本地算力模式已启用：仅用于小数据量受控测试，官方提交仍需 Gate。" : "已切回 HPC/GPU 集群模式：本地训练 fallback 已关闭。");
    } catch (error) {
      setComputeFeedback(error instanceof Error ? error.message : "算力模式切换失败。");
    }
  }

  async function runLocalSmallTask() {
    setComputeMode("local");
    setComputeFeedback("准备使用本地算力运行 Titanic 小数据任务...");
    try {
      await props.runWorkstationAction?.("select_compute_mode", { mode: "local", small_task_only: true, task_id: "titanic" });
      props.setSelectedTask("titanic");
      await props.runLocalExperiment?.("titanic");
      setComputeFeedback("Titanic 本地小数据训练请求已完成，结果已写入工作站 run/evidence/gate。");
    } catch (error) {
      setComputeFeedback(error instanceof Error ? error.message : "Titanic 本地小数据训练失败。");
    }
  }

  const gpuConnector = (s?.connector_status?.gpu ?? {}) as Record<string, unknown>;
  const gpuReady = gpuConnector.current_gate_ready === true;
  const gpuStateText = String(gpuConnector.state ?? "GPU fresh smoke blocked");
  return (
    <Page title="GPU / HPC 控制台" subtitle="算力资源状态、作业门禁、远程训练与产物回传">
      <LiveRunEvidencePanel {...props} />
      <Panel
        title="算力模式选择 / Compute Mode"
        description="显式选择本地算力或 HPC/GPU 集群；选择会写入设置与 action log"
        action={<StatusBadge tone={computeMode === "local" ? "green" : gpuReady ? "green" : "red"}>{computeMode === "local" ? "Local Compute" : gpuReady ? "HPC Ready" : "HPC Blocked"}</StatusBadge>}
      >
        <div className="grid gap-2 xl:grid-cols-[1fr_1fr_1.4fr]">
          <button
            data-ui-action="compute_select_hpc_gpu"
            data-ui-skip-action="true"
            data-active-compute-mode={computeMode === "hpc_gpu" ? "true" : "false"}
            onClick={() => selectComputeMode("hpc_gpu")}
            className={cn("rounded-md border p-3 text-left transition hover:border-blue-300 hover:bg-blue-50/40", computeMode === "hpc_gpu" ? "border-blue-300 bg-blue-50 ring-1 ring-blue-100" : "border-slate-200 bg-white")}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-2 text-sm font-black text-slate-950"><Server className="h-4 w-4 text-blue-600" />HPC/GPU 集群</span>
              <StatusBadge tone={gpuReady ? "green" : "red"}>{gpuReady ? "smoke passed" : "smoke blocked"}</StatusBadge>
            </div>
            <div className="mt-2 text-xs font-semibold leading-5 text-slate-600">正式 Kaggle/MLE-Bench 长任务默认使用远程白名单模板、HPC Gate 和 artifact pullback。</div>
          </button>
          <button
            data-ui-action="compute_select_local"
            data-ui-skip-action="true"
            data-active-compute-mode={computeMode === "local" ? "true" : "false"}
            onClick={() => selectComputeMode("local")}
            className={cn("rounded-md border p-3 text-left transition hover:border-emerald-300 hover:bg-emerald-50/40", computeMode === "local" ? "border-emerald-300 bg-emerald-50 ring-1 ring-emerald-100" : "border-slate-200 bg-white")}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-2 text-sm font-black text-slate-950"><Cpu className="h-4 w-4 text-emerald-600" />本地算力</span>
              <StatusBadge tone={computeMode === "local" ? "green" : "amber"}>{computeMode === "local" ? "enabled" : "manual select"}</StatusBadge>
            </div>
            <div className="mt-2 text-xs font-semibold leading-5 text-slate-600">仅用于小数据量受控验证；不会启动官方 Kaggle 提交，不宣称 GPU/HPC 或奖牌结果。</div>
          </button>
          <div className="rounded-md border border-slate-200 bg-white p-3">
            <div className="grid gap-2 md:grid-cols-[1fr_auto]">
              <div>
                <div className="text-sm font-black text-slate-950">本地小任务闭环测试</div>
                <div className="mt-1 text-xs font-semibold leading-5 text-slate-600">{computeFeedback}</div>
              </div>
              <Button data-ui-action="compute_run_local_titanic_smoke" data-ui-skip-action="true" variant="primary" onClick={runLocalSmallTask}>
                <Play className="h-4 w-4" />运行 Titanic
              </Button>
            </div>
            <div className="mt-2 grid gap-2 md:grid-cols-3">
              <RowCompact label="Selected mode" value={computeMode === "local" ? "local" : "hpc_gpu"} />
              <RowCompact label="Local scope" value="small_tasks_only" />
              <RowCompact label="Official submit" value="blocked_by_gate" />
            </div>
          </div>
        </div>
      </Panel>
      <MicroStatusRail
        items={[
          ["SSH Gateway", gpuReady ? "ready" : "smoke blocked", gpuReady ? "green" : "red"],
          ["GPU Available", gpuReady ? "8 / 8" : "blocked", gpuReady ? "green" : "red"],
          ["Queue Running", gpuReady ? "2 / 14" : "0 / 0", gpuReady ? "blue" : "slate"],
          ["Gate Mode", "controlled", "amber"]
        ]}
      />
      <div className="grid gap-2 xl:grid-cols-4">
        <HeroStatusCard icon={Server} label="GPU Gateway" title="ssh_gateway" detail={gpuReady ? "ssh_gateway connected / proxy ready，长任务只走工作站 manifest。" : `${gpuStateText}；fresh GPU smoke 未通过前不放行远程训练。`} tone={gpuReady ? "green" : "red"} meta={gpuReady ? "smoke passed" : "smoke blocked"} />
        <HeroStatusCard icon={Cpu} label="Resource Pool" title="A800 Cluster (8 GPUs)" detail={gpuReady ? "Queue depth 2 / 14，CPU 42%，RAM 61%。" : "集群规格来自配置元数据；实时占用需 fresh smoke 通过后才可回传。"} tone={gpuReady ? "blue" : "amber"} meta={gpuReady ? "Healthy" : "Unverified"} />
        <HeroStatusCard icon={UserCheck} label="Job Gate" title="HPC execution approval required" detail="所有训练作业受 Gate 控制，不允许旁路长训练。" tone="amber" meta="Pending 3" />
        <HeroStatusCard icon={Download} label="Artifact Pullback" title={gpuReady ? "Artifacts returning" : "No active pullback"} detail={gpuReady ? "metrics、OOF、submission、logs 正在回传并入账。" : "GPU 训练被阻断，当前无远程 artifact 回传。"} tone={gpuReady ? "blue" : "slate"} meta={gpuReady ? "72%" : "—"} />
      </div>
      <div className="grid gap-2 xl:grid-cols-[1.12fr_1fr]">
        <Panel title="当前 GPU/HPC 资源 / Resource Profile" description="只显示脱敏账号与可验证运行时信息" className="h-[160px] overflow-hidden">
          <div className="grid gap-x-6 md:grid-cols-2">
            {["Cluster Name:a800-cluster-prod", "SSH Gateway:ssh_gateway", "User Account:hpc_user_***", "GPU Model:NVIDIA A800 80GB", "CUDA Version:12.2", "GPU Memory:80 GB", "CPU:64 vCPU", "RAM:256 GB", "Disk Workspace:/workspace", "Python:conda research-3.10", "Dependencies:LightGBM / XGBoost / CatBoost / PyTorch", "nvidia-smi:All 8 GPUs OK"].map((x) => {
              const [k, v] = x.split(":");
              return <Row key={x} label={k} value={<span>{v} <StatusBadge tone={gpuReady ? "green" : "slate"}>{gpuReady ? "ready" : "spec"}</StatusBadge></span>} />;
            })}
          </div>
        </Panel>
        <Panel title="资源健康监控 / Resource Health Monitor" description="GPU、CPU、内存、队列与心跳" className="h-[160px] overflow-hidden">
          <div className="grid gap-2 md:grid-cols-[1.05fr_0.95fr_0.72fr]">
            <div className="rounded-md border border-slate-100 bg-white p-2"><div className="flex justify-between text-xs font-black text-slate-500"><span>GPU Utilization (8 GPUs)</span><span>37%</span></div><MiniLine /><Progress value={37} /></div>
            <div className="space-y-2 rounded-md border border-slate-100 bg-white p-2"><RowCompact label="GPU Memory" value="237 / 384 GB" /><Progress value={62} tone="green" /><RowCompact label="CPU Usage" value="27 / 64 cores" /><Progress value={42} tone="green" /></div>
            <div className="rounded-md border border-slate-100 bg-white p-2">
              <RowCompact label="Running Jobs" value="2" />
              <RowCompact label="Queued Jobs" value="2" />
              <RowCompact label="Avg Wait Time" value="00:07:24" />
              <RowCompact label="Last Heartbeat" value="11:38:21" />
            </div>
          </div>
        </Panel>
      </div>
      <div className="grid gap-2 xl:grid-cols-[1.5fr_0.82fr]">
        <Panel title="作业队列 / Job Queue" description="Manifest-gated remote training" action={<div className="flex gap-1"><StatusBadge tone="slate">All Status</StatusBadge><StatusBadge tone="slate">Last 24h</StatusBadge></div>} className="h-[198px] overflow-hidden">
          <DenseTable headers={["job_id", "task_id", "run_id", "exp_id", "agent", "template", "status", "duration", "workspace", "action"]} rows={jobs.map((j) => [j[0], j[1], j[2], j[3], j[4], j[5], j[8], j[10], `/workspace/${j[2]}`, "View"])} />
        </Panel>
        <Panel title="GPU Job Manifest" description="受控命令模板" action={<Button size="sm" variant="secondary" data-ui-action="gpu_view_job_manifest_yaml">View YAML</Button>} className="h-[198px] overflow-hidden">
          <div className="grid gap-1.5 md:grid-cols-[1fr_0.8fr]">
            <div className="rounded-md border border-slate-200 bg-white p-2">
              <RowCompact label="template" value="python train.py --config {config}" />
              <RowCompact label="timeout" value="03:00:00" />
              <RowCompact label="conda_env" value="research-3.10" />
              <RowCompact label="python" value="3.10.14" />
              <RowCompact label="gate_dependency" value="gate_hpc_03" />
              <RowCompact label="rollback" value="metric_regression" />
            </div>
            <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
              <div className="mb-1 text-[10px] font-black uppercase tracking-[0.05em] text-slate-500">Output Artifacts</div>
              {["metrics.json", "oof_pred.parquet", "submission.csv", "stdout.log", "stderr.log"].map((item) => (
                <div key={item} className={cn(mono, "mb-1 flex items-center justify-between rounded bg-white px-2 py-1 text-[10px] text-slate-700")}>
                  <span>{item}</span>
                  <StatusBadge tone="green">hash</StatusBadge>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>
      <div className="grid gap-2 xl:grid-cols-[1fr_300px_1fr]">
        <Panel title="远程训练日志 / Remote Training Logs" description="stdout / stderr / nvidia-smi / metrics" className="h-[168px] overflow-hidden">
          <MiniTerminal lines={["2026-06-25 11:31:02 [info] CUDA available: True", "2026-06-25 11:31:04 [info] loading train/test parquet", "2026-06-25 11:31:07 [fold 1/5] training LightGBM", "2026-06-25 11:31:48 [info] saving metrics.json", "2026-06-25 11:32:05 [info] pulling submission.csv", "2026-06-25 11:32:17 [warn] waiting for Submission Gate approval"]} />
        </Panel>
        <Panel title="Gate 与安全控制" description="Start 受 Gate 控制" className="h-[168px] overflow-hidden">
          <DenseTable headers={["Gate", "Status"]} rows={[["HPC Execution", "passed"], ["Code Quality", "passed"], ["Resource Preflight", "passed"], ["Secret Scan", "passed"], ["Submission Gate", "waiting"], ["Human Approval", "waiting"]]} />
          <Button className="mt-2 w-full" variant="secondary" aria-disabled="true" data-ui-action="blocked_start_training"><Lock className="h-4 w-4" /> Start Training</Button>
        </Panel>
        <Panel title="Artifact Pullback" description="产物回传" className="h-[168px] overflow-hidden">
          <DenseTable headers={["artifact", "size", "gate", "status"]} rows={[["metrics.json", "18.4 KB", "gate_submission", "returned"], ["oof_pred.parquet", "153.2 MB", "gate_submission", "returned"], ["submission.csv", "12.7 KB", "gate_submission", "pending"], ["stdout.log", "1.2 MB", "gate_hpc_03", "returned"], ["stderr.log", "238 KB", "gate_hpc_03", "returned"], ["gpu_job_manifest.yaml", "6.1 KB", "gate_hpc_03", "verified"]]} />
        </Panel>
      </div>
      <Panel title="失败回退 / Failure & Rollback" description="失败作业保留 evidence，不覆盖 best-so-far" className="h-[122px] overflow-hidden">
        <DenseTable headers={["job_id", "failure_type", "responsible_agent", "failure_artifact", "retry_count", "fallback_action", "best_so_far_protected", "promoted"]} rows={[["jb_20260625_0010", "CUDA OOM", "agent_catboost", "stderr.log", "1", "reduce_batch_size", "Yes", "blocked"], ["jb_20260625_0009", "timeout", "agent_lgbm", "stderr.log", "2", "increase_timeout", "Yes", "No"], ["jb_20260625_0007", "SSH disconnected", "agent_hpc", "failure.log", "1", "reconnect_gateway", "Yes", "No"], ["jb_20260625_0006", "dependency missing", "agent_labs", "stderr.log", "0", "install_dependencies", "Yes", "No"], ["jb_20260625_0005", "schema mismatch", "agent_rag", "audit.log", "1", "schema_align", "Yes", "blocked"], ["jb_20260625_0004", "metric regression", "agent_catboost", "metrics.json", "1", "rollback_to_best", "Yes", "blocked"]]} />
      </Panel>
      <BottomSummary phase="5-Fold CV Training" best="exp_0909 / RMSE 0.76321 / run_4b0f9c" next="等待 Submission Gate 审批" pending="3 items" />
    </Page>
  );
}

export function EvidenceLedger(props: ScreenProps) {
  const s = props.summary;
  const [evidenceFeedback, setEvidenceFeedback] = useState("证据页就绪：导出、预览、下载、复制均在前端可用。");
  const artifactLedgerRows = [
    ["art_9f2a7c1e", "code", "task_001", "run_14", "exp_038", "code-agent", "train", "/artifacts/code/train.py", "a1b2...4e56", "3", "gate_code_v1", "verified", "06-25 10:32", "View / Open / Trace"],
    ["art_c3d8b5aa", "metrics", "task_001", "run_14", "exp_038", "eval-agent", "eval", "/artifacts/metric/metrics.json", "b2c3...f5a7", "2", "gate_metrics_v2", "verified", "06-25 10:33", "View / Trace"],
    ["art_7a1d4910", "oof_prediction", "task_001", "run_14", "exp_038", "infer-agent", "predict", "/artifacts/pred/oof.parquet", "c3d4...f6a8", "2", "gate_oof_v1", "verified", "06-25 10:34", "View / Trace"],
    ["art_4b5e2d11", "submission", "task_001", "run_14", "exp_038", "submit-agent", "submit", "/artifacts/submission/sub.csv", "d4e5...7bc9", "1", "gate_submit_v1", "pending", "06-25 10:35", "View / Trace"],
    ["art_aa9f5c3d", "gpu_job_manifest", "task_001", "run_13", "exp_037", "hpc-agent", "train", "/hpc/job_8173.yaml", "e5f6...9d0", "1", "gate_hpc_v1", "verified", "06-25 09:21", "View"],
    ["art_f1e2d3c4", "stdout_log", "task_001", "run_14", "exp_038", "hpc-agent", "train", "/logs/stdout.log", "f6a7...0e1", "1", "gate_hpc_v1", "verified", "06-25 10:31", "View"],
    ["art_f1e2d3c6", "stderr_log", "task_001", "run_14", "exp_038", "hpc-agent", "train", "/logs/stderr.log", "07b8...2f3", "1", "gate_hpc_v1", "verified", "06-25 10:31", "View"],
    ["art_b9c7d6e5", "validation_contract", "task_001", "run_14", "exp_038", "qa-agent", "eval", "/contracts/exp_038.json", "18c9...1a2", "2", "gate_contract_v1", "verified", "06-25 10:28", "View / Trace"],
    ["art_9d8e7f6a", "claim_audit", "task_001", "run_14", "exp_038", "audit-agent", "audit", "/audit/claim_exp_038.json", "29d0...2b4", "3", "gate_audit_v1", "pending", "06-25 10:36", "View"],
    ["art_3c2b1a0f", "report", "task_001", "run_14", "exp_038", "report-agent", "report", "/reports/exp_038/report.md", "3a0e...ad1", "3", "gate_report_v1", "verified", "06-25 11:02", "Open"],
    ["art_1a2b3c4d", "kaggle_response", "task_001", "run_14", "exp_038", "kaggle-agent", "review", "/kaggle/exp_038/response.json", "4bd0...f2", "1", "gate_kaggle_v1", "stale", "06-24 17:40", "View / Trace"],
    ["art_0f1e2d3c", "failure_review", "task_001", "run_13", "exp_036", "qa-agent", "review", "/review/exp_036/failure.md", "5c60...a7", "0", "gate_review_v1", "blocked", "06-24 16:12", "Trace"]
  ] as const;
  const artifactHeaders = ["artifact_id", "artifact_type", "task_id", "run_id", "exp_id", "created_by_agent", "stage", "path", "sha256", "claim_binding", "gate_dependency", "status", "created_at", "操作"];
  const draftBundle = {
    exported_at: new Date().toISOString(),
    task_id: props.selectedTask,
    scope: "draft_evidence_bundle",
    gate_status: "draft_only_final_export_requires_human_gate",
    artifacts: artifactLedgerRows.map((row) => Object.fromEntries(artifactHeaders.map((header, index) => [header, row[index] ?? ""]))),
    note: "前端导出的审查草稿包；最终对外交付仍需人工 Gate。"
  };
  const selectedArtifact = {
    artifact_id: "art_9f2a7c1e",
    name: "train.py",
    path: "/artifacts/code/train.py",
    sha256: "a1b2c3d4e5f67890...",
    status: "verified",
    preview: "import torch\nfrom model import Model\n\n# audit-bound training entrypoint\n"
  };

  function recordEvidenceAction(action: string, metadata: Record<string, unknown> = {}) {
    void props.runWorkstationAction?.(action, { task_id: props.selectedTask, source: "evidence_ledger", ...metadata });
  }

  function exportCsv() {
    downloadTextFile("artifact_ledger.csv", `${artifactHeaders.join(",")}\n${toCsv(artifactLedgerRows)}`, "text/csv;charset=utf-8");
    setEvidenceFeedback("已生成 artifact_ledger.csv，可用于老师汇报或后端对接验证。");
    recordEvidenceAction("export_audit_bundle", { format: "csv", filename: "artifact_ledger.csv" });
  }

  function exportDraftBundle() {
    downloadTextFile("evidence_draft_bundle.json", JSON.stringify(draftBundle, null, 2), "application/json;charset=utf-8");
    setEvidenceFeedback("已生成 evidence_draft_bundle.json；这是草稿包，最终导出仍需人工 Gate。");
    recordEvidenceAction("export_audit_bundle", { format: "draft_bundle", filename: "evidence_draft_bundle.json" });
  }

  function downloadSelectedArtifact() {
    downloadTextFile("art_9f2a7c1e_train.py.txt", selectedArtifact.preview, "text/plain;charset=utf-8");
    setEvidenceFeedback("已下载当前选中 artifact 的预览文件。");
    recordEvidenceAction("export_audit_bundle", { intent: "download_artifact", artifact_id: selectedArtifact.artifact_id });
  }

  function previewSelectedArtifact() {
    setEvidenceFeedback(`预览已定位：${selectedArtifact.name} · ${selectedArtifact.path}`);
    recordEvidenceAction("open_artifact_folder", { intent: "preview_artifact", artifact_id: selectedArtifact.artifact_id });
  }

  function exportArtifactEvidence(payload: Record<string, unknown>) {
    const artifactId = String(payload.artifact_id ?? selectedArtifact.artifact_id);
    const evidencePayload = {
      exported_at: new Date().toISOString(),
      task_id: props.selectedTask,
      source: "evidence_ledger_drawer",
      gate_status: "draft_only_final_export_requires_human_gate",
      ...payload
    };
    downloadTextFile(`${artifactId}_evidence.json`, JSON.stringify(evidencePayload, null, 2), "application/json;charset=utf-8");
    setEvidenceFeedback(`已导出 ${artifactId}_evidence.json；该文件可交给后端 artifact/export API 对接。`);
    recordEvidenceAction("export_audit_bundle", { format: "artifact_evidence", artifact_id: artifactId });
  }

  async function copyText(text: string, label: string) {
    try {
      await navigator.clipboard?.writeText(text);
      setEvidenceFeedback(`${label} 已复制到剪贴板。`);
    } catch {
      setEvidenceFeedback(`${label} 可复制内容：${text}`);
    }
  }

  return (
    <Page title="证据台账" subtitle="统一归档 artifact、日志、指标、报告和审计证据">
      <LiveRunEvidencePanel {...props} />
      <EvolutionEvidencePanel taskId={props.selectedTask} />
      <div className="rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-xs font-bold text-blue-800" data-testid="evidence-action-feedback">
        {evidenceFeedback}
      </div>
      <div className="grid grid-cols-[repeat(7,minmax(0,1fr))_128px_84px_84px] gap-2 rounded-md border border-slate-200 bg-white p-2 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
        {["task_id 全部", "run_id 全部", "exp_id 全部", "agent 全部", "artifact_type 全部", "gate_status 全部", "claim_binding 全部"].map((item) => (
          <button key={item} data-ui-action={`evidence_filter_${item.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="h-8 truncate rounded-md border border-slate-200 bg-slate-50 px-2 text-xs font-black text-slate-700 hover:border-blue-200 hover:bg-white">
            {item}
          </button>
        ))}
        <button className="h-8 truncate rounded-md border border-slate-200 bg-slate-50 px-2 text-xs font-black text-slate-700 hover:border-blue-200 hover:bg-white" data-ui-action="evidence_date_range">
          2025-06-25
        </button>
        <Button size="sm" variant="secondary" data-ui-action="reset_evidence_filters">重置</Button>
        <Button size="sm" variant="primary" data-ui-action="apply_evidence_filters"><Filter className="h-4 w-4" />筛选</Button>
      </div>
      <div className="grid gap-2 xl:grid-cols-4">
        <EvidenceMetric icon={ShieldCheck} label="Evidence Coverage 覆盖率" value="92%" detail="415 / 450 claims backed by linked artifacts" tone="green" progress={92} />
        <EvidenceMetric icon={Database} label="Artifact Integrity 完整性" value="1,248 / 1,312" detail="Hash 验证通过 · Schema 存在 · Path 完整" tone="blue" progress={96} />
        <EvidenceMetric icon={GitBranch} label="Claim Binding 绑定情况" value="415" detail="需复核 23，未绑定 12" tone="blue" progress={77} />
        <EvidenceMetric icon={AlertTriangle} label="Missing Evidence 缺失风险" value="38" detail="影响 9 个结论，需尽快补齐证据" tone="red" progress={38} />
      </div>

      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-2">
          <Panel title="Evidence Lineage Graph 证据谱系图" description="Task -> Run -> Experiment -> Agent -> Artifact -> Claim -> Report" action={<Button size="sm" variant="secondary" data-ui-action="open_evidence_lineage_graph">展开全图</Button>} className="h-[122px] overflow-hidden">
            <div className="grid gap-2 md:grid-cols-7">
              {["Task", "Run", "Experiment", "Agent", "Artifact", "Claim", "Report"].map((x, i) => (
                <button
                  key={x}
                  className={cn("relative h-[64px] rounded-md border p-2 text-center transition hover:border-blue-300 hover:bg-blue-50", i === 4 ? "border-blue-300 bg-blue-50 ring-2 ring-blue-100" : "border-slate-200 bg-white")}
                  data-ui-action={`open_evidence_lineage_${x.toLowerCase()}`}
                >
                  <StatusBadge tone={i === 4 ? "blue" : i < 4 ? "green" : "slate"}>{i === 4 ? "selected" : "linked"}</StatusBadge>
                  <div className="mt-1 text-[13px] font-black text-slate-900">{x}</div>
                  <div className={cn(mono, "mt-0.5 truncate text-[10px] text-slate-500")}>{i === 4 ? "art_9f2a7c1e" : `${x.toLowerCase()}_id`}</div>
                  {i < 6 ? <ArrowRight className="absolute -right-4 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-slate-300 md:block" /> : null}
                </button>
              ))}
            </div>
          </Panel>
          <Panel title="Artifact Ledger 证据总账" description="共 1,312 条" action={<div className="flex gap-1.5"><Button size="sm" variant="secondary" data-ui-action="configure_evidence_columns">列设置</Button><Button size="sm" variant="secondary" data-ui-action="export_evidence_csv" data-ui-skip-action="true" onClick={exportCsv}>导出 CSV</Button></div>} className="h-[360px] overflow-hidden">
            <div className="thin-scrollbar h-[304px] overscroll-contain overflow-auto" onWheel={stopWheelPropagation} tabIndex={0}>
              <DenseTable
                headers={artifactHeaders}
                rows={artifactLedgerRows}
              />
            </div>
          </Panel>
          <div className="grid gap-2 xl:grid-cols-[1.05fr_0.72fr_1fr]">
            <Panel title="Claim Binding / Claim Audit" description="结论绑定" className="h-[154px] overflow-hidden"><DenseTable headers={["Claim", "证据项", "支持指标", "结论边界"]} rows={[["Confirmed", "415", "CV / OOF / private", "通过"], ["Weak Evidence", "23", "CV only", "需复核"], ["Unsupported", "12", "Official only", "不支持"], ["Needs Review", "17", "manual", "待复核"]]} /></Panel>
            <Panel title="Artifact Integrity" description="完整性检查" className="h-[154px] overflow-hidden"><DenseTable headers={["check", "status"]} rows={[["Hash 已验证", "通过"], ["Schema 合规", "通过"], ["Path 存在", "通过"], ["生成 Agent 已批准", "通过"], ["Gate 依赖满足", "警告"], ["报告已包含", "通过"]]} /></Panel>
            <Panel title="Evidence Timeline / Audit Log" description="审计日志" className="h-[154px] overflow-hidden"><DenseTable headers={["time", "agent", "action", "status"]} rows={[["06-25 11:05", "audit-agent", "Artifact 验证通过", "通过"], ["06-25 11:02", "report-agent", "Report included", "通过"], ["06-25 10:36", "submit-agent", "提交文件生成", "待审"], ["06-25 10:31", "hpc-agent", "GPU Job 拉回", "通过"], ["06-24 16:12", "qa-agent", "Failure Review", "阻断"]]} /></Panel>
          </div>
          <Panel title="Delivery / Evidence Bundle 交付包" description="教师报告、数据表、证据包与审计 JSON" className="h-[154px] overflow-hidden">
            <div className="thin-scrollbar h-[100px] overscroll-contain overflow-auto" onWheel={stopWheelPropagation} tabIndex={0}>
              <DenseTable headers={["文件名", "类型", "大小", "更新于", "完整性", "Gate", "操作"]} rows={[["teacher_report.pdf", "报告", "2.41 MB", "2025-06-25 11:02", "通过", "待审批", "预览 / 下载"], ["experiment_summary.xlsx", "数据表", "1.32 MB", "2025-06-25 10:58", "通过", "待审批", "下载 / 打开路径"], ["evidence_bundle.zip", "证据包", "512.38 MB", "2025-06-25 11:05", "通过", "待审批", "下载 / 打开路径"], ["reproducibility_report.md", "文档", "412 KB", "2025-06-25 10:55", "通过", "待审批", "预览"]]} />
            </div>
          </Panel>
        </div>
        <div className="space-y-2">
          <EvidenceDetailDrawer
            onPreview={previewSelectedArtifact}
            onDownload={downloadSelectedArtifact}
            onCopyPreview={(preview) => void copyText(preview, "Artifact Preview")}
            onCopyPath={() => void copyText(selectedArtifact.path, "Artifact Path")}
            onExportEvidence={exportArtifactEvidence}
          />
          <Panel title="人工 Gate / 最终审批" description="最终导出需要人工审批" className="h-[108px] overflow-hidden">
            <Row label="状态" value={<StatusBadge tone="amber">待审批</StatusBadge>} />
            <Row label="审批人" value="未指派" />
            <Row label="条件" value="需完成全部 Gate 才可最终导出" />
            <div className="mt-2 grid grid-cols-2 gap-2">
              <Button variant="primary" size="sm" data-ui-action="export_draft_evidence_bundle" data-ui-skip-action="true" onClick={exportDraftBundle}>导出草稿包</Button>
              <Button variant="secondary" size="sm" aria-disabled="true" data-ui-action="blocked_final_evidence_approval"><Lock className="h-4 w-4" />最终审批</Button>
            </div>
          </Panel>
        </div>
      </div>
      <BottomSummary phase="Evidence Export Review" best="415 / 450 claims backed by linked artifacts" next="导出草稿包" pending="最终导出需人工 Gate" />
    </Page>
  );
}

export function EvidenceDetail(_props: ScreenProps) {
  const [activeArtifactTab, setActiveArtifactTab] = useState("概览");
  const artifactTabPanels: Record<string, { title: string; rows: [string, React.ReactNode, boolean?][]; preview: string }> = {
    "概览": {
      title: "Artifact Overview / 证据概览",
      rows: [
        ["artifact_type", "metrics", true],
        ["文件路径", "/runs/run_20260625_10/exp_20260625_1001/metrics.json", true],
        ["sha256", "91fa5c8b7d4e2f0c9b1a6f0e4f2c2b1d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7", true],
        ["agent", "agent_xcientist_v2", true],
        ["linked_exp_id", "exp_20260625_1001", true],
        ["linked_run_id", "run_20260625_10", true],
        ["linked_claim", "claim_003 · 预测误差 < 5.0%"],
        ["linked_gate", "gate_regression_v1", true],
        ["验证状态", <StatusBadge key="passed" tone="green">通过</StatusBadge>],
        ["最后验证", "2025-06-25 10:33:21"],
        ["schema_version", "v1.2.0"],
        ["大小", "24.8 KB"],
        ["MIME Type", "application/json"]
      ],
      preview: `{
  "exp_id": "exp_20260625_1001",
  "dataset": "house_prices",
  "metrics": {
    "rmse": 0.3125,
    "mae": 0.2211,
    "r2": 0.8742,
    "mape": 0.0507
  },
  "cv_folds": 5,
  "timestamp": "2025-06-25T10:30:12Z"
}`
    },
    "验证详情": {
      title: "Validation Detail / 验证详情",
      rows: [
        ["schema_check", <StatusBadge key="schema" tone="green">passed</StatusBadge>],
        ["hash_check", <StatusBadge key="hash" tone="green">passed</StatusBadge>],
        ["claim_binding", "claim_003 supported by metrics + OOF"],
        ["required_artifacts", "metrics.json, oof_pred.parquet, submission_audit.json"],
        ["acceptance_criteria", "OOF/CV must beat protected baseline"],
        ["risk_check", "CV-public gap watch, no leakage detected"],
        ["validation_agent", "agent_xcientist_v2", true],
        ["contract_id", "validation_contract_1001", true]
      ],
      preview: `{
  "validation_result": "passed",
  "required_artifacts_present": true,
  "claim_binding": "claim_003",
  "risk_flags": ["cv_public_gap_watch"],
  "acceptance": "meets local evidence gate"
}`
    },
    "依赖关系 (5)": {
      title: "Dependency Graph / 依赖关系",
      rows: [
        ["upstream", "data_audit.json -> feature_manifest.json -> train.py"],
        ["downstream", "claim_audit.json -> teacher_report.pdf"],
        ["gate_dependency", "gate_regression_v1, gate_claim_v1", true],
        ["blocked_by", "final_report_approval pending"],
        ["superseded_by", "metrics_20260625_1127"],
        ["lineage_depth", "5 hops"]
      ],
      preview: `Task -> Run -> Experiment -> Agent -> Artifact -> Claim -> Report
metrics_20260625_1001
  depends_on: data_audit.json, train.py, oof_pred.parquet
  required_by: claim_003, teacher_report.pdf
  gate_dependency: gate_regression_v1`
    },
    "审计日志": {
      title: "Audit Log / 审计日志",
      rows: [
        ["10:33:21", "ValidationAgent verified schema"],
        ["10:33:08", "EvidenceLedger attached sha256"],
        ["10:32:44", "ClaimAuditAgent linked claim_003"],
        ["10:31:10", "ReportAgent requested artifact preview"],
        ["10:30:12", "CodeAgent emitted metrics.json"],
        ["10:29:47", "Gate regression check started"]
      ],
      preview: `[10:33:21] schema_check passed
[10:33:08] hash_check passed
[10:32:44] claim_003 linked
[10:31:10] report preview requested
[10:30:12] metrics.json created`
    }
  };
  const activeArtifactPanel = artifactTabPanels[activeArtifactTab] ?? artifactTabPanels["概览"];
  return (
    <Page title="证据台账" subtitle="统一归档 artifact、日志、指标、报告和审计证据">
      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_342px]">
        <div className="space-y-2">
          <div className="grid grid-cols-[repeat(7,minmax(0,1fr))_auto_auto] gap-2 rounded-md border border-slate-200 bg-white p-2 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
            {["task_id 全部", "run_id 全部", "exp_id 全部", "agent 全部", "artifact_type 全部", "gate_status 全部", "claim_binding 全部"].map((item) => (
              <button key={item} data-ui-action={`evidence_detail_filter_${item.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="h-8 truncate rounded-md border border-slate-200 bg-slate-50 px-2 text-xs font-black text-slate-700 hover:border-blue-200 hover:bg-white">
                {item}
              </button>
            ))}
            <Button size="sm" variant="secondary" data-ui-action="filter_evidence_detail"><Filter className="h-4 w-4" />更多筛选</Button>
            <Button size="sm" variant="secondary" data-ui-action="save_evidence_view">保存视图</Button>
          </div>

          <div className="grid gap-2 xl:grid-cols-4">
            <EvidenceMetric icon={ShieldCheck} label="Evidence Coverage" value="92%" detail="已验证结论证据链" tone="green" progress={92} />
            <EvidenceMetric icon={Database} label="Artifact Integrity" value="1,284 / 1,342" detail="95.7% 完整性通过率" tone="blue" progress={96} />
            <EvidenceMetric icon={GitBranch} label="Claim Binding" value="24 / 31" detail="77.4% 绑定率" tone="blue" progress={77} />
            <EvidenceMetric icon={AlertTriangle} label="Missing Evidence" value="7" detail="影响 3 个关键结论" tone="red" progress={34} />
          </div>

          <Panel
            title="证据谱系 (Lineage)"
            description="Task -> Run -> Experiment -> Agent -> Artifact -> Claim -> Report"
            action={<Button size="sm" variant="secondary" data-ui-action="view_full_lineage">查看全图</Button>}
            className="h-[94px] overflow-hidden"
          >
            <div className="grid gap-2 md:grid-cols-7">
              {["Task", "Run", "Experiment", "Agent", "Artifact", "Claim", "Report"].map((x, i) => (
                <button
                  key={x}
                  className={cn("relative h-[48px] rounded-md border p-1.5 text-left transition hover:border-blue-300 hover:bg-blue-50", i === 4 ? "border-blue-300 bg-blue-50 ring-2 ring-blue-100" : "border-slate-200 bg-white")}
                  data-ui-action={`open_lineage_${x.toLowerCase()}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-black text-slate-800">{x}</span>
                    <StatusBadge tone={i === 4 ? "blue" : i < 4 ? "green" : "slate"}>{i === 4 ? "已选中" : "linked"}</StatusBadge>
                  </div>
                  <div className={cn(mono, "mt-0.5 truncate text-[10px] text-slate-500")}>{i === 4 ? "metrics_20260625_1001" : `${x.toLowerCase()}_id`}</div>
                  {i < 6 ? <ArrowRight className="absolute -right-4 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-slate-300 md:block" /> : null}
                </button>
              ))}
            </div>
          </Panel>

          <Panel
            title="Artifact Ledger (证据台账)"
            description="共 1,342 条记录"
            action={<div className="flex gap-1.5"><Button size="sm" variant="secondary" data-ui-action="add_evidence_annotation">列设置</Button><Button size="sm" variant="secondary" data-ui-action="export_evidence_csv">导出 CSV</Button></div>}
            className="h-[304px] overflow-hidden"
          >
            <div className="thin-scrollbar h-[250px] overflow-auto">
              <DenseTable
                headers={["artifact_id", "artifact_type", "task_id", "run_id", "exp_id", "created_by_agent", "stage", "path", "sha256", "claim_binding", "gate_dependency", "status", "created_at", "操作"]}
                rows={[
                  ["metrics_20260625_1001", "metrics", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "evaluation", "/runs/.../metrics.json", "91fa...c2b1", "claim_003", "gate_regression_v1", "verified", "2025-06-25 10:31", "View / Trace"],
                  ["oof_20260625_1001", "oof_prediction", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "evaluation", "/runs/.../oof_pred.parquet", "3b7a...f6a9", "claim_003", "gate_regression_v1", "pending", "2025-06-25 10:29", "View / Trace"],
                  ["model_code_20260625_1001", "code", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "train", "/model.py", "a1e2...d4e5", "claim_003", "gate_code_v1", "verified", "2025-06-25 10:25", "View / Trace"],
                  ["submission_20260625_1001", "submission", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "submit", "/submissions/.../submission.csv", "7e2b...9664", "claim_003", "gate_kaggle_v1", "verified", "2025-06-25 10:33", "View / Trace"],
                  ["gpu_job_manifest_1001", "gpu_job_manifest", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "train", "/hpc/job_1001.yaml", "6d8a...e0b3", "-", "gate_hpc_v1", "verified", "2025-06-25 10:12", "View"],
                  ["stdout_log_1001", "stdout_log", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "train", "/logs/stdout.log", "c8b7...c112", "-", "gate_hpc_v1", "stale", "2025-06-25 10:12", "View"],
                  ["stderr_log_1001", "stderr_log", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "train", "/logs/stderr.log", "9a11...3bb7", "-", "gate_hpc_v1", "pending", "2025-06-25 10:12", "View"],
                  ["validation_contract_1001", "validation_contract", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "audit", "/contracts/val_contract.json", "bb44...1a9c", "claim_003", "gate_validation_v1", "verified", "2025-06-25 10:20", "View / Trace"],
                  ["claim_audit_1001", "claim_audit", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "audit", "/audit/claim_audit.json", "1d23...666", "claim_003", "gate_claim_v1", "blocked", "2025-06-25 10:34", "View"],
                  ["report_teacher_1001", "report", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_reporter", "report", "/reports/teacher_report.pdf", "77aa...b2cc", "-", "gate_report_v1", "pending", "2025-06-25 10:35", "Open"],
                  ["kaggle_response_1001", "kaggle_response", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "submit", "/kaggle/response.json", "6f9b...0d41", "claim_003", "gate_kaggle_v1", "verified", "2025-06-25 10:36", "View / Trace"],
                  ["failure_review_1001", "failure_review", "task_20260625_01", "run_20260625_10", "exp_20260625_1001", "agent_xcientist_v2", "review", "/review/failure_review.md", "2c99...77ae", "claim_002", "gate_review_v1", "missing", "2025-06-25 10:38", "Trace"]
                ]}
              />
            </div>
          </Panel>

          <div className="grid gap-2 xl:grid-cols-[1.05fr_0.92fr_1.05fr_0.95fr]">
            <Panel title="Claim Binding / Claim Audit" description="结论审计" className="h-[142px] overflow-hidden">
              <DenseTable headers={["claim_id", "状态", "关联证据"]} rows={[["claim_003", "已确认", "metrics, oof"], ["claim_011", "证据较弱", "metrics"], ["claim_002", "不支持", "public only"], ["claim_018", "待修订", "baseline"]]} />
            </Panel>
            <Panel title="Artifact Integrity" description="完整性检查" className="h-[142px] overflow-hidden">
              <DenseTable headers={["检查", "状态"]} rows={[["Hash 验证", "通过"], ["Schema 结构校验", "通过"], ["路径存在且可访问", "通过"], ["由批准 Agent 生成", "通过"], ["Gate 依赖已满足", "警告"]]} />
            </Panel>
            <Panel title="Evidence Timeline / Audit Log" description="审计日志" className="h-[142px] overflow-hidden">
              <DenseTable headers={["时间", "Agent", "事件", "状态"]} rows={[["10:36", "agent_xcientist_v2", "Kaggle 响应已归档", "verified"], ["10:35", "agent_reporter", "报告已生成", "pending"], ["10:34", "gate_claim_v1", "Claim 审计已执行", "blocked"], ["10:31", "agent_xcientist_v2", "Artifact 已创建", "verified"]]} />
            </Panel>
            <Panel title="Delivery / Evidence Bundle" description="交付包" className="h-[142px] overflow-hidden">
              <DenseTable headers={["文件名", "状态", "完整性"]} rows={[["teacher_report.pdf", "pending", "通过"], ["experiment_summary.xlsx", "verified", "通过"], ["evidence_bundle.zip", "verified", "通过"], ["claim_audit.json", "blocked", "失败"], ["submission_audit.json", "pending", "警告"]]} />
            </Panel>
          </div>

          <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
            <div className="font-bold text-amber-900">人工 Gate 审批中：export_release_v1。请等待人工 Gate 审批通过后，才可执行最终导出。</div>
            <Button size="sm" variant="secondary" data-ui-action="export_draft_evidence_bundle"><Download className="h-4 w-4" />导出审查材料</Button>
            <Button size="sm" variant="secondary" aria-disabled="true" data-ui-action="blocked_final_export"><Lock className="h-4 w-4" />最终导出</Button>
          </div>
        </div>

        <aside className="thin-scrollbar h-[calc(100vh-76px)] min-h-[820px] overflow-auto rounded-md border border-slate-200 bg-white shadow-[0_16px_60px_-40px_rgba(15,23,42,0.72)]">
          <div className="sticky top-0 z-10 border-b border-slate-200 bg-white px-3 py-2">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-black text-slate-500">artifact_id</div>
              <div className="flex gap-1.5">
                <button className="rounded-md p-1 text-slate-400 hover:bg-slate-100" aria-label="expand artifact detail" data-ui-action="expand_artifact_detail">↗</button>
                <button className="rounded-md p-1 text-slate-400 hover:bg-slate-100" aria-label="close artifact detail" data-ui-action="close_artifact_detail">×</button>
              </div>
            </div>
            <div className={cn(mono, "mt-1 break-all text-sm font-black text-slate-950")}>metrics_20260625_1001 <StatusBadge tone="green">verified</StatusBadge></div>
            <div className="mt-3 grid grid-cols-4 gap-1.5">
              <Button size="sm" variant="secondary" data-ui-action="preview_evidence_artifact">Preview</Button>
              <Button size="sm" variant="secondary" data-ui-action="download_evidence_artifact">Download</Button>
              <Button size="sm" variant="secondary" data-ui-action="open_evidence_lineage">Open Lineage</Button>
              <Button size="sm" variant="secondary" data-ui-action="open_evidence_menu">⋮</Button>
            </div>
          </div>

          <div className="border-b border-slate-200 px-3 py-2">
            <div className="grid grid-cols-4 gap-1 text-xs font-black">
              {["概览", "验证详情", "依赖关系 (5)", "审计日志"].map((tab, i) => (
                <button key={tab} data-ui-action={`artifact_detail_tab_${i}`} data-ui-skip-action="true" data-active-artifact-tab={activeArtifactTab === tab ? "true" : "false"} onClick={() => setActiveArtifactTab(tab)} className={cn("h-7 rounded-md border text-[11px]", activeArtifactTab === tab ? "border-blue-200 bg-blue-50 text-blue-700" : "border-transparent text-slate-500 hover:bg-slate-50")}>
                  {tab}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3 p-3">
            <div className="space-y-0.5" data-active-artifact-panel={activeArtifactTab}>
              <div className="mb-2 rounded-md border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs font-black text-blue-800">{activeArtifactPanel.title}</div>
              {activeArtifactPanel.rows.map(([label, value, monoValue]) => (
                <Row key={label} label={label} value={value} monoValue={Boolean(monoValue)} />
              ))}
            </div>

            <div className="rounded-md border border-amber-200 bg-amber-50 p-2 text-xs font-bold leading-5 text-amber-800">
              注意事项：Artifact 较旧 (stale)，建议重新生成以确保最新。
            </div>

            <Panel title={`${activeArtifactTab} 内容预览`} description="当前选中证据面板" className="overflow-hidden">
              <pre className={cn(mono, "thin-scrollbar h-[154px] overflow-auto rounded-md bg-slate-950 p-3 text-[10px] leading-[16px] text-slate-100")}>{activeArtifactPanel.preview}</pre>
            </Panel>

            <Panel title="后端接口预留" description="/api/workstation-actions" className="overflow-hidden">
              <Row label="点击动作" value="ui_component_click" monoValue />
              <Row label="主操作" value="open_evidence_lineage" monoValue />
              <Row label="元数据" value="page, component_type, action_id, label" monoValue />
            </Panel>
          </div>
        </aside>
      </div>
    </Page>
  );
}

function EvidenceMetric({ icon: Icon, label, value, detail, tone, progress }: { icon: React.ElementType; label: string; value: string; detail: string; tone: StatusTone; progress: number }) {
  const color = tone === "green" ? "#10b981" : tone === "red" ? "#ef4444" : tone === "amber" ? "#f59e0b" : "#2563eb";
  const muted = tone === "red" ? "#fee2e2" : tone === "green" ? "#d1fae5" : "#dbeafe";
  const dash = Math.max(4, Math.min(100, progress));
  return (
    <Card className="min-w-0 overflow-hidden">
      <CardContent className="relative p-2.5">
        <div className="flex items-start gap-2 pr-16">
          <span className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-md border", tone === "green" ? "border-emerald-100 bg-emerald-50 text-emerald-700" : tone === "red" ? "border-red-100 bg-red-50 text-red-700" : "border-blue-100 bg-blue-50 text-blue-700")}><Icon className="h-4 w-4" /></span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-xs font-black text-slate-600">{label}</div>
            <div className={cn("mt-0.5 text-[22px] font-black leading-6", tone === "red" ? "text-red-700" : "text-slate-950")}>{value}</div>
            <div className="mt-0.5 line-clamp-1 text-[11px] font-semibold leading-[14px] text-slate-500">{detail}</div>
            <div className="mt-1"><Progress value={progress} tone={tone} /></div>
          </div>
        </div>
        <div className="absolute right-3 top-1/2 hidden h-14 w-14 -translate-y-1/2 items-center justify-center md:flex">
          <svg viewBox="0 0 40 40" className="h-14 w-14 rotate-[-90deg]" aria-hidden="true">
            <circle cx="20" cy="20" r="15.5" fill="none" stroke={muted} strokeWidth="6" />
            <circle cx="20" cy="20" r="15.5" fill="none" stroke={color} strokeLinecap="round" strokeWidth="6" strokeDasharray={`${dash} 100`} pathLength="100" />
          </svg>
          <span className="absolute text-[10px] font-black text-slate-700">{progress}%</span>
        </div>
      </CardContent>
    </Card>
  );
}

type EvidenceDetailDrawerProps = {
  onPreview?: () => void;
  onDownload?: () => void;
  onCopyPreview?: (preview: string) => void;
  onCopyPath?: () => void;
  onExportEvidence?: (payload: Record<string, unknown>) => void;
};

function EvidenceDetailDrawer({ onPreview, onDownload, onCopyPreview, onCopyPath, onExportEvidence }: EvidenceDetailDrawerProps) {
  const [activeDrawerTab, setActiveDrawerTab] = useState("Preview");
  const drawerTabs: Record<string, { label: string; title: string; rows: string[][]; preview: string }> = {
    Preview: {
      label: "Preview",
      title: "Artifact Preview / 产物预览",
      rows: [["文件", "train.py"], ["语言", "Python / PyTorch"], ["用途", "训练入口"], ["状态", "verified"]],
      preview: `import torch
from model import Model
from dataset import load_data

def train(cfg):
    model = Model(cfg).to(cfg.device)
    train_loader = load_data(cfg.train_path)
    for epoch in range(cfg.epochs):
        loss = run_epoch(model, train_loader)
        write_metric(epoch, loss)

if __name__ == "__main__":
    train(load_config())`
    },
    Validation: {
      label: "Validation Detail",
      title: "Validation Detail / 验证详情",
      rows: [["schema_check", "passed"], ["required_artifacts", "metrics.json, oof_pred.parquet"], ["contract", "validation_contract_1001"], ["risk_check", "CV-public gap watch"]],
      preview: `{
  "validation_result": "passed",
  "required_artifacts_present": true,
  "validation_contract": "validation_contract_1001",
  "claim_binding": "CLM-102",
  "risk_flags": ["cv_public_gap_watch"]
}`
    },
    Dependencies: {
      label: "Dependencies",
      title: "Dependency Graph / 依赖关系",
      rows: [["upstream", "data_audit.json, feature_manifest.json"], ["downstream", "claim_audit.json, teacher_report.pdf"], ["gate_dependency", "gate_regression_v1"], ["lineage_depth", "5 hops"]],
      preview: `Task -> Run -> Experiment -> Agent -> Artifact -> Claim -> Report
art_9f2a7c1e
  depends_on: data_audit.json, train.py, oof_pred.parquet
  required_by: claim_003, teacher_report.pdf
  gate_dependency: gate_regression_v1`
    },
    Audit: {
      label: "Audit Log",
      title: "Audit Log / 审计日志",
      rows: [["10:33:21", "ValidationAgent verified schema"], ["10:33:08", "EvidenceLedger attached sha256"], ["10:32:44", "ClaimAuditAgent linked claim_003"], ["10:30:12", "CodeAgent emitted train.py"]],
      preview: `[10:33:21] schema_check passed
[10:33:08] hash_check passed
[10:32:44] claim_003 linked
[10:30:12] train.py created`
    }
  };
  const activeDrawerPanel = drawerTabs[activeDrawerTab] ?? drawerTabs.Preview;
  const selectedArtifactEvidence = {
    artifact_id: "art_9f2a7c1e",
    name: "train.py",
    path: "/artifacts/code/train.py",
    sha256: "a1b2c3d4e5f67890...",
    status: "verified",
    active_tab: activeDrawerTab,
    preview: activeDrawerPanel.preview
  };
  return (
    <aside className="thin-scrollbar h-[610px] overscroll-contain overflow-auto rounded-md border border-blue-200 bg-white shadow-[0_18px_56px_-38px_rgba(15,23,42,0.78)] xl:sticky xl:top-[68px]" onWheel={stopWheelPropagation} tabIndex={0}>
      <div className="border-b border-blue-100 bg-blue-50 px-2.5 py-2">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] font-black uppercase tracking-[0.05em] text-blue-700">Selected Artifact</div>
            <div className={cn(mono, "mt-1 text-slate-950")}>art_9f2a7c1e</div>
          </div>
          <div className="flex gap-1.5"><StatusBadge tone="green">verified</StatusBadge><StatusBadge tone="blue">code</StatusBadge></div>
        </div>
      </div>
      <div className="space-y-1 p-2.5">
        <Row label="名称" value="train.py" monoValue />
        <Row label="路径" value="/artifacts/code/train.py" monoValue />
        <Row label="SHA256" value="a1b2c3d4e5f67890..." monoValue />
        <Row label="大小" value="18.7 KB" />
        <Row label="创建时间" value="2025-06-25 10:32:11" />
        <Row label="Created by Agent" value="code-agent (v2.6.1)" monoValue />
        <Row label="关联 Experiment" value="exp_038 (run_14)" monoValue />
        <Row label="关联 Claim" value="CLM-102, CLM-118, CLM-127" monoValue />
        <Row label="依赖 Gate" value="gate_code_v1 (approved)" monoValue />
        <div className="grid grid-cols-3 gap-2">
          <Button variant="secondary" size="sm" data-ui-action="preview_selected_artifact" data-ui-skip-action="true" onClick={onPreview}>预览</Button>
          <Button variant="secondary" size="sm" data-ui-action="download_selected_artifact" data-ui-skip-action="true" onClick={onDownload}>下载</Button>
          <Button variant="secondary" size="sm" data-ui-action="open_selected_artifact_lineage">查看谱系</Button>
        </div>
        <div className="rounded-md border border-amber-200 bg-amber-50 p-2 text-[10px] font-bold leading-4 text-amber-800">
          风险提示：依赖 gate_metrics_v2 仍处于 pending 状态；关联 Claim CLM-118 证据强度偏弱；该 artifact 已有新版本 superseded。
        </div>
        <div className="grid grid-cols-4 gap-1 text-xs font-black">
          {Object.entries(drawerTabs).map(([key, tab], index) => (
            <button
              key={key}
              data-ui-action={`artifact_detail_tab_${index}`}
              data-ui-skip-action="true"
              data-active-artifact-tab={activeDrawerTab === key ? "true" : "false"}
              onClick={() => setActiveDrawerTab(key)}
              className={cn("h-7 rounded-md border text-[11px]", activeDrawerTab === key ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50")}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="thin-scrollbar max-h-[132px] overscroll-contain overflow-auto rounded-md border border-blue-100 bg-blue-50 p-2" data-active-artifact-panel={activeDrawerTab} onWheel={stopWheelPropagation} tabIndex={0}>
          <div className="mb-1 text-xs font-black text-blue-800">{activeDrawerPanel.title}</div>
          <DenseTable headers={["字段", "值"]} rows={activeDrawerPanel.rows} />
        </div>
        <div className="thin-scrollbar max-h-[220px] overscroll-contain overflow-auto" onWheel={stopWheelPropagation} tabIndex={0}>
          <div className="mb-1 flex items-center justify-between text-xs font-black text-slate-500"><span>{activeDrawerPanel.title}</span><button className="text-blue-700" data-ui-action="copy_artifact_preview" data-ui-skip-action="true" onClick={() => onCopyPreview?.(activeDrawerPanel.preview)}>复制</button></div>
          <pre className={cn(mono, "thin-scrollbar max-h-[178px] overflow-auto rounded-md bg-slate-950 p-2 text-[10px] leading-[14px] text-slate-100")}>{activeDrawerPanel.preview}</pre>
        </div>
        <Row label="元数据" value="Python / PyTorch / CUDA 12.2" />
        <Row label="提交" value="git abc1234 · branch main" />
        <Row label="当前状态" value={<StatusBadge tone="green">可追溯</StatusBadge>} />
        <div className="rounded-md border border-slate-200 p-2">
          <div className="mb-1 text-xs font-black text-slate-500">审计结果</div>
          <DenseTable headers={["项目", "状态"]} rows={[["Hash 校验", "通过"], ["路径可访问", "通过"], ["Claim 绑定", "弱证据"], ["Gate 依赖", "待补齐"]]} />
        </div>
        <div className="rounded-md border border-slate-200 p-2">
          <div className="mb-1 text-xs font-black text-slate-500">关联操作</div>
          <DenseTable headers={["动作", "状态"]} rows={[["Preview", "ready"], ["Download", "ready"], ["Open Lineage", "ready"], ["Final Export", "blocked"]]} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="secondary" size="sm" data-ui-action="open_artifact">Open Artifact</Button>
          <Button variant="secondary" size="sm" data-ui-action="copy_artifact_path" data-ui-skip-action="true" onClick={onCopyPath}>Copy Path</Button>
          <Button variant="secondary" size="sm" data-ui-action="trace_artifact_claim">Trace Claim</Button>
          <Button variant="secondary" size="sm" data-ui-action="export_artifact_evidence" data-ui-skip-action="true" onClick={() => onExportEvidence?.(selectedArtifactEvidence)}>Export Evidence</Button>
        </div>
      </div>
    </aside>
  );
}

type ReportViewRecord = {
  id: string;
  taskId: string;
  runId?: string | null;
  title: string;
  status: string;
  markdown: string;
  markdownPath?: string | null;
  docxPath?: string | null;
  source: string;
};

function normalizeReportTaskId(taskId: string | null | undefined) {
  return String(taskId ?? "playground_series_s6e6").replaceAll("-", "_").trim() || "playground_series_s6e6";
}

function stringField(record: Record<string, unknown> | null | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "string" ? value : "";
}

function numericField(record: Record<string, unknown> | null | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "number" ? value : null;
}

function repairMojibake(value: string) {
  if (!value) return value;
  const chineseCount = (value.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const suspectCount = (value.match(/[\u00c3\u00c2\u00e2\u00e5\u00e6\u00e7\u00e8\u00e9\u00e4\u00ef\u00bc\u00e3\u0080]/g) ?? []).length;
  if (chineseCount >= 4 || suspectCount < 3 || typeof TextDecoder === "undefined") return value;
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 255);
    const decoded = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    const decodedChinese = (decoded.match(/[\u4e00-\u9fff]/g) ?? []).length;
    return decodedChinese > chineseCount ? decoded : value;
  } catch {
    return value;
  }
}

function isMeaningfulReportMarkdown(markdown: string) {
  const text = repairMojibake(markdown).trim();
  if (text.length < 120) return false;
  if (/^#\s*Smoke report\s*$/i.test(text)) return false;
  return /(^|\n)#{1,3}\s+/.test(text);
}

function normalizeReportRecord(raw: unknown, source = "summary"): ReportViewRecord | null {
  if (!raw || typeof raw !== "object") return null;
  const record = raw as Record<string, unknown>;
  const taskId = normalizeReportTaskId(stringField(record, "task_id") || stringField(record, "taskId"));
  const id = stringField(record, "id") || `${taskId}_${source}_report`;
  const markdown = repairMojibake(stringField(record, "markdown_content") || stringField(record, "markdownContent"));
  const title = repairMojibake(stringField(record, "title")) || `${taskId.replaceAll("_", " ")} 研究报告`;
  return {
    id,
    taskId,
    runId: stringField(record, "run_id") || stringField(record, "runId") || null,
    title,
    status: stringField(record, "status") || "draft",
    markdown,
    markdownPath: stringField(record, "markdown_path") || stringField(record, "markdownPath") || null,
    docxPath: stringField(record, "docx_path") || stringField(record, "docxPath") || null,
    source
  };
}

function collectSummaryReports(summary: WorkstationSummary | null | undefined) {
  const reports = (summary?.reports ?? [])
    .map((item) => normalizeReportRecord(item, "summary"))
    .filter((item): item is ReportViewRecord => Boolean(item));
  return reports.sort((a, b) => {
    const meaningfulDelta = Number(isMeaningfulReportMarkdown(b.markdown)) - Number(isMeaningfulReportMarkdown(a.markdown));
    if (meaningfulDelta) return meaningfulDelta;
    return b.markdown.length - a.markdown.length;
  });
}

function reportRuntimeForTask(summary: WorkstationSummary | null | undefined, taskId: string) {
  return summary?.runtime_by_task?.[taskId] ?? (summary?.runtime?.task_id === taskId ? summary.runtime : null) ?? null;
}

function currentTaskRecord(summary: WorkstationSummary | null | undefined, taskId: string) {
  return (summary?.tasks ?? []).find((task) => normalizeReportTaskId(task.id) === taskId) ?? null;
}

function taskRuns(summary: WorkstationSummary | null | undefined, taskId: string) {
  return (summary?.runs ?? []).filter((run) => normalizeReportTaskId(run.task_id) === taskId);
}

function taskRecords(summary: WorkstationSummary | null | undefined, key: "gates" | "evidence", taskId: string) {
  const rows = summary?.[key] ?? [];
  return rows.filter((row) => normalizeReportTaskId(stringField(row, "task_id") || stringField(row, "taskId")) === taskId);
}

function formatReportMetric(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(6) : String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return repairMojibake(String(value));
}

function markdownTable(headers: string[], rows: string[][]) {
  return [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...(rows.length ? rows : [headers.map(() => "-")]).map((row) => `| ${row.map((cell) => String(cell).replaceAll("\n", " ")).join(" | ")} |`)
  ].join("\n");
}

function buildReportAppendix(summary: WorkstationSummary | null | undefined, taskId: string) {
  const task = currentTaskRecord(summary, taskId);
  const runtime = reportRuntimeForTask(summary, taskId);
  const runs = taskRuns(summary, taskId).slice(0, 8);
  const gates = taskRecords(summary, "gates", taskId).slice(0, 10);
  const evidence = taskRecords(summary, "evidence", taskId).slice(0, 12);
  const runtimeArtifacts = Array.isArray(runtime?.artifact_manifest?.artifacts)
    ? runtime?.artifact_manifest?.artifacts as Array<Record<string, unknown>>
    : [];
  const runRows = runs.map((run) => [
    run.id ?? "-",
    run.status ?? "-",
    run.best_model ?? "-",
    run.best_metrics ? Object.entries(run.best_metrics).map(([key, value]) => `${key}=${formatReportMetric(value)}`).join("; ") : "-",
    run.output_dir ?? "-"
  ]);
  const gateRows = gates.map((gate) => [
    stringField(gate, "id") || stringField(gate, "gate_id") || "-",
    stringField(gate, "type") || stringField(gate, "name") || stringField(gate, "action") || "-",
    stringField(gate, "status") || stringField(gate, "decision") || "-"
  ]);
  const evidenceRows = evidence.map((item) => [
    stringField(item, "id") || stringField(item, "artifact_id") || stringField(item, "name") || "-",
    stringField(item, "type") || stringField(item, "kind") || "-",
    stringField(item, "path") || stringField(item, "artifact") || stringField(item, "artifact_path") || "-"
  ]);
  const artifactRows = runtimeArtifacts.slice(0, 8).map((item) => [
    stringField(item, "name") || stringField(item, "artifact_id") || "-",
    stringField(item, "type") || stringField(item, "kind") || "-",
    stringField(item, "path") || stringField(item, "artifact_path") || "-"
  ]);

  return [
    "## 工作站审计附录",
    "",
    "### 任务与边界",
    `- task_id: ${taskId}`,
    `- task_name: ${task?.name ?? taskId}`,
    `- task_type: ${task?.task_type ?? "-"}`,
    `- metric: ${task?.metric ?? "-"}`,
    "- 官方成绩边界: 只有存在 Kaggle response artifact 时，才能写官方分数、排名、奖牌。",
    "- 当前导出会保留 claim audit / gate 边界，不把本地 CV 或 proxy 分数写成官方成绩。",
    "",
    "### 最近工作站 Run",
    markdownTable(["run_id", "status", "best_model", "metrics", "output_dir"], runRows),
    "",
    "### Gate / Claim Audit",
    markdownTable(["gate_id", "type", "status"], gateRows),
    "",
    "### Evidence Ledger",
    markdownTable(["artifact", "type", "path"], evidenceRows.length ? evidenceRows : artifactRows),
    "",
    "### 下一步建议",
    "1. 若报告仍是 proxy/CV 证据，先补齐 OOF、submission_audit、claim_audit 与人工审批。",
    "2. 需要官方质量评估时，通过工作站提交门禁产生 Kaggle response 后再更新报告。",
    "3. 对短报告或 smoke 报告，先点击“重新生成完整草稿”，再导出 Markdown/PDF。"
  ].join("\n");
}

function buildCompleteReportMarkdown(summary: WorkstationSummary | null | undefined, taskId: string, report: ReportViewRecord | null) {
  const runtime = reportRuntimeForTask(summary, taskId);
  const runtimeMarkdown = repairMojibake(String(runtime?.report_markdown ?? "")).trim();
  const reportMarkdown = repairMojibake(report?.markdown ?? "").trim();
  const mainMarkdown = isMeaningfulReportMarkdown(reportMarkdown)
    ? reportMarkdown
    : isMeaningfulReportMarkdown(runtimeMarkdown)
      ? runtimeMarkdown
      : [
          `# ${taskId.replaceAll("_", " ")} 科研工作站报告`,
          "",
          "## 摘要",
          "当前没有可直接展示的完整报告正文，工作站已基于 summary、runs、gates 和 evidence 生成审计版报告骨架。请点击“重新生成完整草稿”让 Report Agent 写入正式 Markdown。",
          "",
          "## 当前结论边界",
          "- 本报告只展示工作站可见证据。",
          "- 未绑定 Kaggle response 时，不声明官方排名、奖牌或 MLE-Bench 对齐结果。",
          "- 所有提升结论必须绑定 exp_id、metrics、OOF/submission artifact 与 claim audit。"
        ].join("\n");
  const appendix = buildReportAppendix(summary, taskId);
  return `${mainMarkdown.trim()}\n\n---\n\n${appendix}\n`;
}

function extractMarkdownHeadings(markdown: string) {
  return markdown
    .split(/\r?\n/)
    .map((line, index) => {
      const match = line.match(/^(#{1,3})\s+(.+)$/);
      if (!match) return null;
      return {
        id: `report-heading-${index}`,
        level: match[1].length,
        title: repairMojibake(match[2].trim())
      };
    })
    .filter((item): item is { id: string; level: number; title: string } => Boolean(item));
}

function artifactImageSrc(rawPath: string) {
  const normalized = rawPath.trim().replace(/^<|>$/g, "").replaceAll("\\", "/");
  if (/^(https?:|data:|blob:|\/api\/)/i.test(normalized)) return normalized;
  return `/api/artifacts?path=${encodeURIComponent(normalized)}`;
}

function isMarkdownTableLine(line: string) {
  return /^\s*\|.+\|\s*$/.test(line);
}

function isMarkdownTableSeparator(line: string) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function parseMarkdownTableRow(line: string) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => repairMojibake(cell.trim()));
}

function renderMarkdownTable(lines: string[], key: string) {
  const rows = lines
    .filter((line) => !isMarkdownTableSeparator(line))
    .map(parseMarkdownTableRow)
    .filter((row) => row.some(Boolean));
  const [headers, ...bodyRows] = rows;
  if (!headers?.length) return null;
  return (
    <div key={key} className="my-3 overflow-x-auto rounded-md border border-slate-200">
      <table className="w-full min-w-[520px] table-auto text-left text-[12px]">
        <thead className="bg-slate-50 text-[11px] uppercase tracking-[0.02em] text-slate-500">
          <tr>{headers.map((header, index) => <th key={`${key}-h-${index}`} className="border-b border-slate-200 px-3 py-2 font-black">{header || "-"}</th>)}</tr>
        </thead>
        <tbody>
          {(bodyRows.length ? bodyRows : [headers.map(() => "-")]).map((row, rowIndex) => (
            <tr key={`${key}-r-${rowIndex}`} className="border-t border-slate-100">
              {headers.map((_, cellIndex) => (
                <td key={`${key}-c-${rowIndex}-${cellIndex}`} className="max-w-[260px] break-words px-3 py-2 align-top font-semibold text-slate-700">
                  {row[cellIndex] || "-"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderMarkdownPreview(markdown: string) {
  let headingIndex = -1;
  const lines = markdown.split(/\r?\n/);
  const nodes: React.ReactNode[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
    const line = repairMojibake(rawLine);
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      headingIndex += 1;
      const id = `report-heading-${index}`;
      const level = heading[1].length;
      const text = heading[2];
      if (level === 1) {
        nodes.push(<h1 id={id} key={index} className="mt-2 text-center text-2xl font-black text-slate-950 first:mt-0">{text}</h1>);
      } else if (level === 2) {
        nodes.push(<h2 id={id} key={index} className="mt-6 border-b border-slate-200 pb-2 text-base font-black text-slate-950">{text}</h2>);
      } else {
        nodes.push(<h3 id={id} key={index} className="mt-4 text-sm font-black text-slate-900">{text}</h3>);
      }
      continue;
    }

    const image = line.match(/^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (image) {
      const alt = repairMojibake(image[1].trim() || "报告图表");
      const rawTarget = image[2].trim().replace(/\s+["'][^"']+["']\s*$/, "");
      const src = artifactImageSrc(rawTarget);
      nodes.push(
        <figure key={index} className="my-4 rounded-md border border-slate-200 bg-slate-50 p-3">
          <img
            src={src}
            alt={alt}
            loading="lazy"
            className="mx-auto block max-h-[360px] w-full max-w-[820px] rounded border border-slate-100 bg-white object-contain"
          />
          <figcaption className="mt-2 break-all text-center text-[11px] font-bold text-slate-500">{alt}</figcaption>
        </figure>
      );
      continue;
    }

    if (isMarkdownTableLine(line)) {
      const tableLines = [line];
      while (index + 1 < lines.length && isMarkdownTableLine(repairMojibake(lines[index + 1]))) {
        index += 1;
        tableLines.push(repairMojibake(lines[index]));
      }
      nodes.push(renderMarkdownTable(tableLines, `table-${index}`));
      continue;
    }
    if (!line.trim()) {
      nodes.push(<div key={index} className="h-3" />);
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      nodes.push(<p key={index} className="pl-4 text-sm leading-7 text-slate-700">• {line.replace(/^[-*]\s+/, "")}</p>);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      nodes.push(<p key={index} className="pl-4 text-sm leading-7 text-slate-700">{line}</p>);
      continue;
    }
    if (/^```/.test(line)) {
      nodes.push(<pre key={index} className="rounded-md bg-slate-950 px-3 py-2 font-mono text-[11px] text-slate-100">{line}</pre>);
      continue;
    }
    nodes.push(<p key={`${index}-${headingIndex}`} className="text-sm leading-7 text-slate-700">{line}</p>);
  }

  return nodes;
}

export function ReportStudio(props: ScreenProps) {
  const s = props.summary;
  const selectedTaskId = normalizeReportTaskId(props.selectedTask);
  const summaryReports = useMemo(() => collectSummaryReports(s), [s]);
  const taskOptions = useMemo(() => {
    const fromTasks = (s?.tasks ?? []).map((task) => normalizeReportTaskId(task.id));
    const fromReports = summaryReports.map((report) => report.taskId);
    const fromRuntime = s?.runtime_by_task ? Object.keys(s.runtime_by_task).map(normalizeReportTaskId) : [];
    return Array.from(new Set([selectedTaskId, ...fromTasks, ...fromReports, ...fromRuntime])).filter(Boolean);
  }, [s, selectedTaskId, summaryReports]);
  const [activeReport, setActiveReport] = useState<ReportViewRecord | null>(null);
  const [localStatus, setLocalStatus] = useState("正在加载报告...");
  const [isGenerating, setIsGenerating] = useState(false);
  const reportsForTask = summaryReports.filter((report) => report.taskId === selectedTaskId);
  const preferredSummaryReport = reportsForTask.find((report) => isMeaningfulReportMarkdown(report.markdown)) ?? reportsForTask[0] ?? null;
  const displayReport = activeReport?.taskId === selectedTaskId ? activeReport : preferredSummaryReport;
  const displayMarkdown = useMemo(
    () => buildCompleteReportMarkdown(s, selectedTaskId, displayReport),
    [s, selectedTaskId, displayReport]
  );
  const outline = useMemo(() => extractMarkdownHeadings(displayMarkdown), [displayMarkdown]);
  const runsForTask = taskRuns(s, selectedTaskId);
  const gatesForTask = taskRecords(s, "gates", selectedTaskId);
  const evidenceForTask = taskRecords(s, "evidence", selectedTaskId);
  const reportQuality = Math.min(100, Math.max(15, Math.round((outline.length / 8) * 40 + (evidenceForTask.length ? 35 : 0) + (gatesForTask.length ? 25 : 0))));
  const meaningful = isMeaningfulReportMarkdown(displayReport?.markdown ?? "");
  const reportTitle = repairMojibake(displayReport?.title ?? `${selectedTaskId.replaceAll("_", " ")} 科研报告`);
  const markdownLines = displayMarkdown.split(/\r?\n/).length;

  useEffect(() => {
    let cancelled = false;
    setLocalStatus("正在加载任务报告...");
    api.getReport(selectedTaskId)
      .then((payload) => {
        if (cancelled) return;
        const apiReport = normalizeReportRecord(payload.report, "api");
        const candidate = apiReport && (isMeaningfulReportMarkdown(apiReport.markdown) || !preferredSummaryReport)
          ? apiReport
          : preferredSummaryReport ?? apiReport;
        setActiveReport(candidate);
        setLocalStatus(candidate
          ? isMeaningfulReportMarkdown(candidate.markdown)
            ? `已加载报告：${candidate.title}`
            : "当前最新报告过短，已启用工作站审计附录补全。"
          : "未找到完整报告，已生成审计骨架。");
      })
      .catch((error) => {
        if (cancelled) return;
        setActiveReport(preferredSummaryReport);
        setLocalStatus(error instanceof Error ? error.message : "报告加载失败，已使用 summary 兜底。");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTaskId, preferredSummaryReport]);

  function selectTaskReport(taskId: string, report?: ReportViewRecord | null) {
    const normalized = normalizeReportTaskId(taskId);
    props.setSelectedTask(normalized);
    setActiveReport(report ?? null);
    setLocalStatus(`已切换到 ${normalized} 的报告视图。`);
    void props.runWorkstationAction?.("report_task_select", { task_id: normalized, report_id: report?.id ?? null });
  }

  function selectOutlineItem(id: string, title: string, index: number) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    void props.runWorkstationAction?.("report_section_select", { task_id: selectedTaskId, section: title, index });
  }

  function exportMarkdown() {
    const exportedAt = new Date().toISOString();
    const content = [
      displayMarkdown.trim(),
      "",
      "---",
      "",
      "## 导出元数据",
      `- exported_at: ${exportedAt}`,
      `- task_id: ${selectedTaskId}`,
      `- report_id: ${displayReport?.id ?? "generated_frontend_complete_report"}`,
      `- source: ${displayReport?.source ?? "summary_runtime_fallback"}`,
      `- markdown_path: ${displayReport?.markdownPath ?? "-"}`,
      `- run_count: ${runsForTask.length}`,
      `- evidence_count: ${evidenceForTask.length}`,
      `- gate_count: ${gatesForTask.length}`
    ].join("\n");
    downloadTextFile(
      `${selectedTaskId}_complete_research_report.md`,
      content,
      "text/markdown;charset=utf-8"
    );
    setLocalStatus(`已导出完整 Markdown：${selectedTaskId}_complete_research_report.md`);
    void props.runWorkstationAction?.("export_report", {
      task_id: selectedTaskId,
      report_id: displayReport?.id ?? null,
      markdown_path: displayReport?.markdownPath ?? null,
      export_format: "markdown",
      source: "report_studio_complete_export"
    });
  }

  function exportAuditJson() {
    const payload = {
      exported_at: new Date().toISOString(),
      task_id: selectedTaskId,
      report: displayReport,
      outline,
      runs: runsForTask,
      gates: gatesForTask,
      evidence: evidenceForTask,
      claim_boundary: "未绑定官方 Kaggle response 时，不允许声明官方排名、奖牌或 MLE-Bench 75 任务达标。"
    };
    downloadTextFile(
      `${selectedTaskId}_report_audit_package.json`,
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
    setLocalStatus(`已导出审计 JSON：${selectedTaskId}_report_audit_package.json`);
  }

  function selectNextReportSection() {
    const first = outline[0];
    if (first) {
      selectOutlineItem(first.id, first.title, 0);
      setLocalStatus(`已定位到章节：${first.title}。`);
      return;
    }
    setLocalStatus("当前报告暂无可定位章节，请先重新生成完整草稿。");
    void props.runWorkstationAction?.("report_section_select", { task_id: selectedTaskId, section: null, index: -1 });
  }

  function exportDraftPdfGate() {
    const exportedAt = new Date().toISOString();
    const content = [
      "# AI 科研工作站报告草稿 PDF 门禁",
      "",
      `- exported_at: ${exportedAt}`,
      `- task_id: ${selectedTaskId}`,
      `- report_id: ${displayReport?.id ?? "generated_frontend_complete_report"}`,
      `- source: ${displayReport?.source ?? "summary_runtime_fallback"}`,
      `- markdown_path: ${displayReport?.markdownPath ?? "-"}`,
      "",
      "## Gate Boundary",
      "- 当前文件是草稿 PDF 导出链路的可审计占位，不代表最终 PDF 已发布。",
      "- 最终 PDF 仍必须通过 artifact manifest、claim audit 与人工 Gate。",
      "- 未绑定官方 Kaggle response 时，不写入官方排名、奖牌或 MLE-Bench 75 任务达标结论。"
    ].join("\n");
    downloadTextFile(
      `${selectedTaskId}_draft_pdf_gate.md`,
      content,
      "text/markdown;charset=utf-8"
    );
    setLocalStatus(`已生成草稿 PDF 门禁包：${selectedTaskId}_draft_pdf_gate.md`);
    void props.runWorkstationAction?.("export_report", {
      task_id: selectedTaskId,
      report_id: displayReport?.id ?? null,
      markdown_path: displayReport?.markdownPath ?? null,
      export_format: "pdf_draft",
      source: "report_studio_draft_pdf_gate"
    });
  }

  async function regenerateDraft() {
    setIsGenerating(true);
    setLocalStatus("Report Agent 正在重新生成完整草稿...");
    try {
      const payload = await api.generateReportDraft(selectedTaskId, { language: props.locale ?? "zh-CN", style: "teacher_evidence_bundle" });
      const report = normalizeReportRecord(payload.report, "generated");
      setActiveReport(report ?? {
        id: `${selectedTaskId}_generated_report`,
        taskId: selectedTaskId,
        title: `${selectedTaskId.replaceAll("_", " ")} 科研报告`,
        status: "draft",
        markdown: repairMojibake(payload.markdown_content),
        markdownPath: payload.markdown_path,
        source: "generated"
      });
      setLocalStatus(`完整草稿已生成：${payload.markdown_path ?? "workspace draft"}`);
      await props.refreshSummary?.();
    } catch (error) {
      setLocalStatus(error instanceof Error ? error.message : "重新生成报告失败。");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <Page title="报告工作室" subtitle="AI 自动汇总实验结果、证据链、风险审计与最终交付材料">
      <LiveRunEvidencePanel {...props} />
      <EvolutionReportPanel taskId={props.selectedTask} />
      <div className="flex flex-wrap items-center justify-end gap-2">
        <span className="text-xs font-black text-slate-500">报告状态</span>
        <StatusBadge tone={meaningful ? "green" : "amber"}>{meaningful ? "完整报告" : "已启用补全"}</StatusBadge>
        <StatusBadge tone="blue">{summaryReports.length} reports</StatusBadge>
        <StatusBadge tone={evidenceForTask.length ? "green" : "amber"}>{evidenceForTask.length} evidence</StatusBadge>
        <StatusBadge tone={gatesForTask.length ? "green" : "amber"}>{gatesForTask.length} gates</StatusBadge>
      </div>
      <div className="grid gap-2 xl:grid-cols-4">
        <HeroStatusCard icon={ShieldCheck} label="报告选择" title={selectedTaskId} detail={reportTitle} tone="blue" meta={displayReport?.status ?? "draft"} />
        <HeroStatusCard icon={Database} label="证据索引" title={`${evidenceForTask.length} 条证据`} detail="导出时自动附加 evidence、gate、run 审计附录。" tone={evidenceForTask.length ? "green" : "amber"} meta="绑定中" />
        <HeroStatusCard icon={UserCheck} label="人工 Gate" title={props.reportSubmitted ? "已提交评审" : "等待评审"} detail="最终版导出仍需要人工审批；草稿 Markdown 可随时导出。" tone={props.reportSubmitted ? "green" : "amber"} meta="受控" />
        <HeroStatusCard icon={FileText} label="内容完整性" title={`${markdownLines} 行`} detail={localStatus} tone={meaningful ? "green" : "amber"} meta={`${reportQuality}%`} />
      </div>
      <div className="grid gap-2 xl:grid-cols-[292px_minmax(720px,1fr)_390px]">
        <Panel title="任务报告 / Reports" description="点击比赛或 run 报告后，中间正文立即切换">
          <Row label="当前任务" value={selectedTaskId} monoValue />
          <Row label="报告质量" value={`${reportQuality}% · ${outline.length} sections`} />
          <Progress value={reportQuality} tone={reportQuality >= 80 ? "green" : "amber"} />
          <div className="thin-scrollbar mt-3 max-h-[188px] space-y-1 overflow-auto overscroll-contain pr-1" onWheel={stopWheelPropagation} tabIndex={0}>
            {taskOptions.map((taskId) => {
              const taskReport = summaryReports.find((report) => report.taskId === taskId && isMeaningfulReportMarkdown(report.markdown)) ?? summaryReports.find((report) => report.taskId === taskId) ?? null;
              const active = taskId === selectedTaskId;
              return (
                <button
                  key={taskId}
                  type="button"
                  data-ui-action={`report_task_select_${taskId}`}
                  onClick={() => selectTaskReport(taskId, taskReport)}
                  className={cn("w-full rounded-md border px-2 py-2 text-left text-xs transition", active ? "border-blue-200 bg-blue-50 text-blue-800" : "border-slate-200 bg-white hover:border-blue-200 hover:bg-slate-50")}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 truncate font-black">{taskId}</span>
                    <StatusBadge tone={taskReport && isMeaningfulReportMarkdown(taskReport.markdown) ? "green" : "amber"}>{taskReport ? taskReport.status : "needs draft"}</StatusBadge>
                  </div>
                  <div className="mt-1 truncate text-[11px] font-semibold text-slate-500">{taskReport?.title ?? "暂无完整报告，点击后可生成草稿"}</div>
                </button>
              );
            })}
          </div>
          <div className="mt-3 border-t border-slate-100 pt-3">
            <div className="mb-2 text-xs font-black text-slate-500">章节大纲</div>
            <div className="thin-scrollbar max-h-[250px] space-y-1 overflow-auto overscroll-contain pr-1" onWheel={stopWheelPropagation} tabIndex={0}>
              {outline.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  data-ui-action={`report_section_select_${index}`}
                  onClick={() => selectOutlineItem(item.id, item.title, index)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs font-bold text-slate-700 hover:bg-slate-50"
                >
                  <span className="w-5 text-[10px] text-slate-400">{String(index + 1).padStart(2, "0")}</span>
                  <span className={cn("min-w-0 truncate", item.level === 3 && "pl-3 text-slate-500")}>{item.title}</span>
                </button>
              ))}
            </div>
          </div>
          <Button className="mt-3 w-full" size="sm" variant="secondary" data-ui-skip-action="true" onClick={regenerateDraft} disabled={isGenerating}><FileText className="h-4 w-4" />{isGenerating ? "生成中..." : "重新生成完整草稿"}</Button>
          <Panel title="报告质量审计" description="Report Quality Audit" className="mt-3">
            <Row label="Evidence Coverage" value={`${evidenceForTask.length} artifacts`} />
            <Progress value={reportQuality} tone={reportQuality >= 80 ? "green" : "amber"} />
            <Row label="Claim Boundary" value={meaningful ? "可审计草稿" : "需补全文本"} />
          </Panel>
        </Panel>
        <Panel title="AI 生成报告 / Report Draft" description="人工可编辑、证据可绑定">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            {[`报告 ID：${displayReport?.id ?? "generated"}`, `任务：${selectedTaskId}`, `状态：${displayReport?.status ?? "draft"}`, `来源：${displayReport?.source ?? "fallback"}`, `路径：${displayReport?.markdownPath ?? "not saved"}`].map((x) => <span key={x} className="max-w-full break-all rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-bold text-slate-600">{x}</span>)}
            <span className="ml-auto flex items-center gap-2 text-xs font-black text-slate-500">
              <span>{outline.length} 章节</span><span>{runsForTask.length} runs</span><span>{evidenceForTask.length} evidence</span>
            </span>
          </div>
          <article className="report-page thin-scrollbar max-h-[690px] min-h-[560px] overflow-auto overscroll-contain rounded-md border border-slate-200 bg-white px-10 py-8" onWheel={stopWheelPropagation} tabIndex={0}>
            {renderMarkdownPreview(displayMarkdown)}
          </article>
          <div className="mt-2 grid grid-cols-5 gap-2 text-xs font-bold text-slate-500">
            {[`行数 ${markdownLines}`, `章节 ${outline.length}`, `证据 ${evidenceForTask.length}`, `Gate ${gatesForTask.length}`, meaningful ? "正文完整" : "需要补全"].map((x) => <div key={x} className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5">{x}</div>)}
          </div>
        </Panel>
        <Panel title="审核与证据 / Review & Evidence" description="质量 Gate、Claim Audit 与导出">
          <div className="mb-2 text-xs font-black text-slate-500">Report Quality Gate</div>
          <DenseTable headers={["check", "status"]} rows={[["正文长度/章节", meaningful ? "passed" : "needs draft"], ["Evidence Ledger 绑定", evidenceForTask.length ? "passed" : "missing"], ["Gate / Claim Audit", gatesForTask.length ? "passed" : "warning"], ["官方成绩边界", "passed"], ["人工审批状态", props.reportSubmitted ? "submitted" : "pending"]]} />
          <Row label="Evidence Coverage" value={`${reportQuality}%`} />
          <Progress value={reportQuality} tone={reportQuality >= 80 ? "green" : "amber"} />
          <div className="mt-3 grid grid-cols-3 gap-2">
            {[["Reports", String(reportsForTask.length), "blue"], ["Runs", String(runsForTask.length), "green"], ["Gates", String(gatesForTask.length), gatesForTask.length ? "green" : "amber"]].map(([label, value, tone]) => (
              <div key={label} className="rounded-md border border-slate-200 p-2 text-center">
                <div className="text-lg font-black text-slate-950">{value}</div>
                <StatusBadge tone={tone as StatusTone}>{label}</StatusBadge>
              </div>
            ))}
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
            <div className="rounded-md border border-slate-200 p-2.5">
              <div className="mb-1 text-xs font-black text-slate-500">Missing Evidence</div>
              <DenseTable headers={["#", "风险"]} rows={[
                ["1", meaningful ? "无" : "报告正文过短或 smoke report"],
                ["2", evidenceForTask.length ? "无" : "缺少 evidence ledger 绑定"],
                ["3", gatesForTask.length ? "无" : "缺少 gate / claim audit 记录"]
              ]} />
            </div>
            <div className="rounded-md border border-slate-200 p-2.5">
              <div className="mb-1 text-xs font-black text-slate-500">Version History</div>
              <DenseTable headers={["report", "status"]} rows={reportsForTask.slice(0, 5).map((report) => [report.id, report.status])} />
            </div>
          </div>
          <div className="mt-3 rounded-md border border-slate-200 p-3">
            <div className="text-xs font-black text-slate-500">Report Package</div>
            <DenseTable headers={["file", "status"]} rows={[
              [displayReport?.markdownPath ?? `${selectedTaskId}_complete_research_report.md`, meaningful ? "ready" : "generated fallback"],
              [displayReport?.docxPath ?? "docx not generated", displayReport?.docxPath ? "ready" : "optional"],
              ["report_audit_package.json", "ready"],
              ["claim_audit.json", gatesForTask.length ? "linked" : "needs evidence"]
            ]} />
          </div>
          <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-3 text-xs font-bold leading-5 text-blue-900">
            {localStatus} 草稿导出不会写入官方排名或奖牌；最终版仍需人工 Gate。
          </div>
          <Button className="mt-4 w-full" variant="primary" data-ui-skip-action="true" onClick={exportMarkdown}><Download className="h-4 w-4" /> 导出完整 Markdown</Button>
          <Button className="mt-2 w-full" variant="secondary" data-ui-skip-action="true" onClick={exportAuditJson}><Database className="h-4 w-4" /> 导出审计 JSON</Button>
          <Button className="mt-2 w-full" variant="secondary" data-ui-action="report_add_section" data-ui-skip-action="true" onClick={selectNextReportSection}><FileText className="h-4 w-4" /> 定位/新增报告章节</Button>
          <Button className="mt-2 w-full" variant="secondary" data-ui-action="report_export_draft_pdf" data-ui-skip-action="true" onClick={exportDraftPdfGate}><Download className="h-4 w-4" /> 导出草稿 PDF 门禁包</Button>
          <Button className="mt-2 w-full" variant="secondary" aria-disabled="true" data-ui-action="blocked_final_report_export"><Lock className="h-4 w-4" /> 最终 PDF 需人工审批</Button>
        </Panel>
      </div>
      <div className="grid gap-3 xl:grid-cols-[1fr_0.7fr]">
        <Panel title="实验结果总览 / Experiment Summary" description="报告中的所有分数均绑定 evidence">
          <DenseTable headers={["任务", "最新 run", "CV / Metric", "Official", "决策", "报告"]} rows={taskOptions.slice(0, 10).map((taskId) => {
            const latestRun = taskRuns(s, taskId)[0];
            const metrics = latestRun?.best_metrics ? Object.entries(latestRun.best_metrics).slice(0, 1).map(([key, value]) => `${key}=${formatReportMetric(value)}`).join("") : "-";
            const taskReport = summaryReports.find((report) => report.taskId === taskId && isMeaningfulReportMarkdown(report.markdown)) ?? summaryReports.find((report) => report.taskId === taskId) ?? null;
            return [
              taskId,
              latestRun?.id ?? "-",
              metrics,
              "no response",
              taskReport && isMeaningfulReportMarkdown(taskReport.markdown) ? "report_ready" : "needs_draft",
              <Button key={taskId} size="sm" variant={taskId === selectedTaskId ? "primary" : "secondary"} data-ui-skip-action="true" onClick={() => selectTaskReport(taskId, taskReport)}>查看报告</Button>
            ];
          })} />
        </Panel>
        <Panel title="交付与导出 / Delivery & Export" description="草稿包、人工评审、最终导出">
          <DenseTable headers={["阶段", "状态"]} rows={[["完整 Markdown", "ready"], ["审计 JSON", "ready"], ["人工评审", props.reportSubmitted ? "submitted" : "pending"], ["最终 PDF", "blocked by human gate"]]} />
        </Panel>
      </div>
      <BottomSummary phase="报告审计与导出" best={displayReport?.markdownPath ?? `${selectedTaskId}_complete_research_report.md`} next={meaningful ? "人工审阅 / 证据复核" : "重新生成完整草稿"} pending={props.reportSubmitted ? "review submitted" : "human gate pending"} />
    </Page>
  );
}

export function LiteratureKnowledge(props: ScreenProps) {
  const currentTask = normalizeReportTaskId(props.selectedTask);
  const taskMeta = currentTaskRecord(props.summary, currentTask);
  const [query, setQuery] = useState(`${taskMeta?.name ?? currentTask} ${taskMeta?.task_type ?? ""} ${taskMeta?.metric ?? ""} Kaggle validation ensemble`);
  const [rag, setRag] = useState<import("@/lib/api/types").LiteratureSearchResponse | null>(null);
  const [ragStatus, setRagStatus] = useState("等待检索");
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [activeFilter, setActiveFilter] = useState<"all" | "local" | "arxiv" | "seed" | "risk" | "accepted">("all");

  async function runSearch(nextQuery = query, reason = "manual") {
    const trimmed = nextQuery.trim() || `${currentTask} Kaggle modeling validation ensemble`;
    setIsSearching(true);
    setRagStatus("正在检索本地知识库与 arXiv...");
    try {
      const payload = await api.searchLiterature({
        task_id: currentTask,
        query: trimmed,
        max_results: 18,
        include_arxiv: true
      });
      setRag(payload);
      setSelectedPaperId(payload.papers[0]?.id ?? null);
      setRagStatus(`检索完成：${payload.metrics.paper_count} 篇文献，${payload.metrics.chunk_count} 个 chunk，context 已写入 ${payload.context_path}`);
      void props.runWorkstationAction?.("literature_search", {
        task_id: currentTask,
        query: trimmed,
        reason,
        context_path: payload.context_path,
        manifest_path: payload.manifest_path,
        citation_confidence: payload.metrics.citation_confidence
      });
    } catch (error) {
      setRagStatus(error instanceof Error ? error.message : "文献检索失败。");
    } finally {
      setIsSearching(false);
    }
  }

  useEffect(() => {
    void runSearch(query, "initial_load");
  }, [currentTask]);

  const papers = rag?.papers ?? [];
  const filteredPapers = papers.filter((paper) => {
    if (activeFilter === "local") return paper.source === "local";
    if (activeFilter === "arxiv") return paper.source === "arxiv";
    if (activeFilter === "seed") return paper.source === "seed";
    if (activeFilter === "risk") return Boolean(paper.risks?.length);
    if (activeFilter === "accepted") return paper.score >= 0.5;
    return true;
  });
  const filteredPaperIds = new Set(filteredPapers.map((paper) => paper.id));
  const retrieval = (rag?.retrieval ?? []).filter((chunk) => activeFilter === "all" || filteredPaperIds.has(chunk.paper_id));
  const selectedPaper = filteredPapers.find((paper) => paper.id === selectedPaperId) ?? filteredPapers[0] ?? papers[0] ?? null;
  const metrics = rag?.metrics ?? {
    paper_count: 0,
    chunk_count: 0,
    citation_confidence: 0,
    context_tokens: 0,
    max_tokens: 8192,
    local_documents_indexed: 0,
    arxiv_results: 0
  };
  const knowledgeCoverage = Math.min(100, Math.round((metrics.paper_count / 18) * 70 + (metrics.local_documents_indexed ? 20 : 0) + (metrics.arxiv_results ? 10 : 0)));
  const paperRows = filteredPapers.map((paper) => [
    paper.title,
    paper.type,
    paper.year || "-",
    paper.venue || paper.source,
    paper.score.toFixed(2),
    paper.task,
    paper.exp,
    paper.status
  ]);
  const retrievalRows = retrieval.map((chunk) => [
    String(chunk.rank),
    chunk.chunk,
    chunk.score.toFixed(2),
    chunk.source,
    chunk.page,
    chunk.artifact,
    chunk.used
  ]);
  const strategyRows = (rag?.strategies ?? []).map((strategy) => [strategy.strategy, strategy.paper_id, strategy.family, strategy.exp, strategy.benefit, strategy.risk]);
  const claimRows = (rag?.claim_audit ?? []).map((claim) => [claim.claim, claim.paper, claim.exp, claim.artifact, claim.status]);
  const filterOptions = [
    ["all", "全部"],
    ["local", "本地知识库"],
    ["arxiv", "arXiv"],
    ["seed", "种子论文"],
    ["risk", "风险标签"],
    ["accepted", "高匹配"]
  ] as const;

  function buildAgentContext(intent: string) {
    if (!rag) {
      void runSearch(query, intent);
      return;
    }
    setRagStatus(`${intent} 已绑定 context：${rag.context_path}`);
    void props.runWorkstationAction?.(intent, {
      task_id: currentTask,
      query: rag.query,
      context_path: rag.context_path,
      manifest_path: rag.manifest_path,
      selected_paper_id: selectedPaper?.id ?? null,
      selected_methods: selectedPaper?.methods ?? [],
      claim_boundary: "literature_context_only_not_leaderboard_claim"
    });
  }

  function exportRagMarkdown() {
    if (!rag) {
      void runSearch(query, "export_rag_markdown_without_context");
      return;
    }
    downloadTextFile(`${currentTask}_rag_context.md`, rag.context_markdown, "text/markdown;charset=utf-8");
    setRagStatus(`已导出 RAG Context Markdown：${rag.context_path}`);
    void props.runWorkstationAction?.("rag_export_context_markdown", {
      task_id: currentTask,
      context_path: rag.context_path,
      manifest_path: rag.manifest_path,
      query: rag.query
    });
  }

  function exportRagManifest() {
    if (!rag) {
      void runSearch(query, "export_rag_manifest_without_context");
      return;
    }
    const payload = {
      exported_at: new Date().toISOString(),
      task_id: currentTask,
      query: rag.query,
      source_counts: rag.source_counts,
      metrics: rag.metrics,
      context_path: rag.context_path,
      manifest_path: rag.manifest_path,
      active_filter: activeFilter,
      selected_paper: selectedPaper,
      papers: filteredPapers,
      retrieval,
      strategies: rag.strategies,
      claim_audit: rag.claim_audit,
      claim_boundary: "文献检索只能作为研究上下文与策略候选，不代表官方 Kaggle 提分、排名或奖牌。"
    };
    downloadTextFile(`${currentTask}_rag_manifest.json`, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
    setRagStatus(`已导出 RAG Manifest：${rag.manifest_path}`);
    void props.runWorkstationAction?.("rag_export_manifest_json", {
      task_id: currentTask,
      context_path: rag.context_path,
      manifest_path: rag.manifest_path,
      active_filter: activeFilter
    });
  }

  return (
    <Page title="文献与知识库" subtitle="文献检索、RAG 资料库与研究上下文">
      <LiveRunEvidencePanel {...props} />
      <div className="grid gap-2 xl:grid-cols-4">
        <div className="rounded-md border border-emerald-100 bg-white p-2.5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-emerald-50 text-emerald-600"><BookOpen className="h-5 w-5" /></span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-3"><span className="text-xs font-black text-slate-600">知识覆盖度 / Knowledge Coverage</span><span className="text-xs font-black text-emerald-600">82% ↑</span></div>
              <div className="mt-1 text-2xl font-black tracking-normal text-slate-950">{metrics.paper_count}</div>
              <div className="mt-1 text-xs font-semibold text-slate-600">当前任务：{currentTask} <span className="font-black text-emerald-700">({taskMeta?.metric ?? "research"})</span></div>
              <div className="mt-2"><Progress value={knowledgeCoverage} tone="green" /></div>
            </div>
          </div>
        </div>
        <div className="rounded-md border border-blue-100 bg-white p-2.5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-blue-50 text-blue-600"><Database className="h-5 w-5" /></span>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-black text-slate-600">RAG 索引 / RAG Index</div>
              <div className="mt-1 grid grid-cols-2 gap-3"><div><div className={cn("text-lg font-black", isSearching ? "text-blue-700" : "text-emerald-700")}>● {isSearching ? "Searching" : "Healthy"}</div><div className="text-[11px] font-semibold text-slate-500">{rag?.generated_at ? new Date(rag.generated_at).toLocaleTimeString() : "等待首次检索"}</div></div><div><div className="text-lg font-black text-slate-950">{metrics.chunk_count}</div><div className="text-[11px] font-semibold text-slate-500">命中 Chunk 数</div></div></div>
            </div>
          </div>
        </div>
        <div className="rounded-md border border-amber-100 bg-white p-2.5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-amber-50 text-amber-600"><GitBranch className="h-5 w-5" /></span>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-black text-slate-600">引用绑定 / Citation Binding</div>
              <div className="mt-1 flex items-end justify-between"><div className="text-2xl font-black text-slate-950">{retrieval.filter((item) => item.used === "accepted").length} / {Math.max(1, metrics.chunk_count)}</div><div className="text-lg font-black text-orange-600">{metrics.citation_confidence}%</div></div>
              <div className="mt-2"><Progress value={metrics.citation_confidence} tone="amber" /></div>
              <div className="mt-1 text-[11px] font-semibold text-slate-500">Claim 必须绑定 source artifact</div>
            </div>
          </div>
        </div>
        <div className="rounded-md border border-red-100 bg-white p-2.5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-red-50 text-red-600"><ShieldAlert className="h-5 w-5" /></span>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-black text-slate-600">风险边界 / Risk Boundary</div>
              <div className="mt-2 grid grid-cols-3 divide-x divide-slate-100 text-center"><div><div className="text-xl font-black text-red-700">{papers.filter((paper) => paper.risks?.length).length}</div><div className="text-[11px] font-semibold text-slate-500">风险标签</div></div><div><div className="text-xl font-black text-red-700">{rag?.source_counts.arxiv ?? 0}</div><div className="text-[11px] font-semibold text-slate-500">外部结果</div></div><div><div className="text-xl font-black text-red-700">{rag?.used_fallback ? 1 : 0}</div><div className="text-[11px] font-semibold text-slate-500">回退</div></div></div>
            </div>
          </div>
        </div>
      </div>
      <div className="flex min-h-8 items-center gap-2 rounded-md border border-blue-100 bg-blue-50 px-3 text-xs font-bold text-blue-800"><CheckCircle2 className="h-4 w-4" />Research briefs and citations must bind to real source artifacts before report approval.</div>
      <Card className="min-w-0 overflow-hidden">
        <CardContent className="grid gap-2 p-3 xl:grid-cols-[minmax(320px,1fr)_auto_auto]">
          <div className="flex min-w-0 items-center gap-2 rounded-md border border-slate-200 bg-white px-3">
            <Search className="h-4 w-4 text-slate-400" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void runSearch(query, "enter_query");
              }}
              className="h-9 min-w-0 flex-1 bg-transparent text-sm font-semibold text-slate-800 outline-none"
              placeholder="输入任务、方法、数据集、metric，例如 LightGBM OOF leakage tabular..."
            />
          </div>
          <Button variant="primary" data-ui-action="literature_search_manual" onClick={() => void runSearch(query, "manual_search")} disabled={isSearching}><Search className="h-4 w-4" />{isSearching ? "检索中" : "真实检索"}</Button>
          <Button variant="secondary" data-ui-action="rag_build_agent_context" onClick={() => buildAgentContext("rag_build_agent_context")}><BrainCircuit className="h-4 w-4" />构建 Agent Context</Button>
        </CardContent>
      </Card>
      <div className="grid items-start gap-2 xl:grid-cols-[1.08fr_1.22fr_360px]">
        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="px-3 py-2.5">
            <div className="flex items-start justify-between gap-2">
              <div><CardTitle>文献库 / Literature Library</CardTitle><CardDescription>论文、报告、讨论与内部笔记</CardDescription></div>
              <Button size="icon" variant="secondary" title="刷新" data-ui-action="literature_refresh_library" data-ui-skip-action="true" onClick={() => void runSearch(query, "refresh_library")}><RefreshCw className={cn("h-4 w-4", isSearching && "animate-spin")} /></Button>
            </div>
          </CardHeader>
          <CardContent className="px-3 pb-3">
            <div className="mb-2 grid grid-cols-3 gap-2 xl:grid-cols-6">
              {filterOptions.map(([value, label]) => (
                <button
                  key={value}
                  data-ui-action={`literature_filter_${value}`}
                  onClick={() => {
                    setActiveFilter(value);
                    const nextCount = value === "all"
                      ? papers.length
                      : papers.filter((paper) => {
                        if (value === "local") return paper.source === "local";
                        if (value === "arxiv") return paper.source === "arxiv";
                        if (value === "seed") return paper.source === "seed";
                        if (value === "risk") return Boolean(paper.risks?.length);
                        if (value === "accepted") return paper.score >= 0.5;
                        return true;
                      }).length;
                    setRagStatus(`已应用筛选：${label}，当前显示 ${nextCount} 条。`);
                    void props.runWorkstationAction?.("literature_apply_filter", { task_id: currentTask, filter: value, label, result_count: nextCount });
                  }}
                  className={cn("h-8 truncate rounded-md border px-2 text-xs font-bold", activeFilter === value ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-slate-50 text-slate-600 hover:border-blue-200 hover:bg-white")}
                >
                  {label}
                </button>
              ))}
              <Button size="sm" variant="secondary" data-ui-action="literature_filter_recompute" onClick={() => void runSearch(query, `filter_refresh_${activeFilter}`)}><Filter className="h-4 w-4" />重算</Button>
            </div>
            <div className="thin-scrollbar max-h-[360px] overflow-auto overscroll-contain rounded-md border border-slate-100" onWheel={stopWheelPropagation} tabIndex={0}>
              <table className="w-full table-fixed text-left text-[11px]">
                <thead className="bg-slate-50 text-[10px] uppercase tracking-[0.02em] text-slate-500">
                  <tr>{["Title", "Type", "Year", "Venue", "Score", "Task", "Exp", "Status"].map((h, i) => <th key={h} className={cn("px-2 py-1.5 font-black", i === 0 ? "w-[34%]" : i === 7 ? "w-[68px]" : "w-[9%]")}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {paperRows.map((row, i) => (
                    <tr key={filteredPapers[i]?.id ?? i} onClick={() => setSelectedPaperId(filteredPapers[i]?.id ?? null)} className={cn("cursor-pointer border-t border-slate-100 hover:bg-blue-50/40", (filteredPapers[i]?.id ?? null) === selectedPaper?.id && "bg-blue-50/70")}>
                      {row.map((cell, j) => (
                        <td key={j} className="h-[25px] min-w-0 px-2 py-0.5 align-middle font-semibold text-slate-700">
                          {j === 0 ? <span className="block truncate font-black text-blue-700">{cell}</span> : j === row.length - 1 ? <StatusBadge tone={toneFor(cell)}>{cell}</StatusBadge> : <span className="block truncate">{cell}</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-2 flex items-center justify-between gap-3 text-[11px] font-bold text-slate-500"><span>本次 {metrics.paper_count} 条 / 当前 {filteredPapers.length} 条</span><span className="min-w-0 flex-1 truncate text-center text-blue-700">{ragStatus}</span><span>{metrics.local_documents_indexed} 本地文档</span></div>
          </CardContent>
        </Card>
        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="px-3 py-2.5">
            <div className="flex items-start justify-between gap-3">
              <div><CardTitle>阅读与方法抽取 / Reading & Method Extraction</CardTitle><CardDescription>PDF 预览、关键方法和引用绑定</CardDescription></div>
              <div className="flex shrink-0 gap-1"><StatusBadge tone="blue">PDF 预览</StatusBadge><StatusBadge tone="green">已绑定</StatusBadge></div>
            </div>
          </CardHeader>
          <CardContent className="px-3 pb-3">
            <h3 className="truncate text-sm font-black text-slate-950">{selectedPaper?.title ?? "等待检索结果"}</h3>
            <div className="mt-2 flex flex-wrap gap-1.5 text-[11px] font-bold"><span className="rounded bg-blue-50 px-2 py-1 text-blue-700">{selectedPaper?.type ?? "paper"}</span><span className="rounded bg-slate-50 px-2 py-1 text-slate-600">{selectedPaper?.venue ?? "source"} {selectedPaper?.year ?? ""}</span><span className="rounded bg-slate-50 px-2 py-1 text-slate-600">{selectedPaper?.source ?? "local"} · score {selectedPaper?.score.toFixed(2) ?? "-"}</span></div>
            <div className="mt-2 grid gap-2 xl:grid-cols-[1fr_248px]">
              <div className="grid gap-1.5 md:grid-cols-2">
                {[
                  ["Abstract", selectedPaper?.abstract ?? "暂无摘要。"],
                  ["Key Method", selectedPaper?.methods?.join(" + ") || "等待方法抽取"],
                  ["Dataset / Benchmark", currentTask],
                  ["Implementation Hints", selectedPaper?.methods?.length ? `可作为 ${selectedPaper.methods.join(", ")} 候选策略；必须通过工作站实验 gate 验证。` : "需要补充方法抽取。"],
                  ["Risk / Strategy", selectedPaper?.risks?.join(", ") || "无显式风险标签，仍需 claim audit。"]
                ].map(([k, v]) => <div key={k} className="min-h-[42px] rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] leading-4"><b>{k}</b><br /><span className="line-clamp-2">{v}</span></div>)}
              </div>
              <div className="space-y-2">
                <div className="rounded-md border border-blue-200 bg-blue-50 p-2.5 text-[11px] leading-4"><b>源文摘录（原文）</b><br />{retrieval.find((chunk) => chunk.paper_id === selectedPaper?.id)?.chunk ?? "暂无 chunk"} <StatusBadge tone="blue">引用</StatusBadge><div className={cn(mono, "mt-1 text-[10px] text-blue-700")}>paper_id: {selectedPaper?.id ?? "-"}</div></div>
                <div className="rounded-md border border-emerald-200 bg-emerald-50 p-2.5 text-[11px] leading-4"><b>Agent 解读</b><br />{selectedPaper?.methods?.length ? `可沉淀为 ${selectedPaper.methods.join(", ")} 策略候选。` : "需要更多证据后再交给 Agent。"}</div>
                <div className="rounded-md border border-amber-200 bg-amber-50 p-2.5 text-[11px] leading-4"><b>报告 Claim 风险</b><br />文献命中只支持“研究上下文”，不得直接声明官方提分或奖牌。<StatusBadge tone="amber">需审计</StatusBadge></div>
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5"><StatusBadge tone="blue">Used by Agent</StatusBadge><StatusBadge tone="green">Bound to KIP-034</StatusBadge><StatusBadge tone="amber">Claim CLM-018</StatusBadge></div>
          </CardContent>
        </Card>
        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="px-3 py-2.5"><CardTitle>Agent Context Builder / RAG Context</CardTitle><CardDescription>RAG 上下文与 Agent 绑定</CardDescription></CardHeader>
          <CardContent className="space-y-2 px-3 pb-3">
            <Row label="当前任务" value={currentTask} />
            <Row label="选中文献" value={selectedPaper?.id ?? `${filteredPapers.length} 篇`} monoValue />
            <Row label="检索 Chunk" value={`${metrics.chunk_count} 个`} />
            <Row label="Context Artifact" value={rag?.context_path ?? "检索后生成"} monoValue />
            <Row label="上下文预算" value={`${metrics.context_tokens} / ${metrics.max_tokens} tokens`} />
            <Progress value={(metrics.context_tokens / Math.max(1, metrics.max_tokens)) * 100} tone="green" />
            <div className="grid grid-cols-[78px_1fr] items-center gap-3 rounded-md border border-slate-200 bg-slate-50 p-2">
              <div className="flex h-16 w-16 items-center justify-center rounded-full border-[7px] border-emerald-400 bg-white text-lg font-black text-slate-950">{metrics.citation_confidence}%</div>
              <div><div className="text-xs font-black text-slate-900">引用置信度</div><MiniLine tone="blue" /><div className="text-[11px] font-bold text-slate-500">低风险，高重复度</div></div>
            </div>
            <Button className="w-full" variant="primary" data-ui-action="rag_build_agent_context" onClick={() => buildAgentContext("rag_build_agent_context")}>构建 Agent Context</Button>
            <div className="grid grid-cols-2 gap-2"><Button size="sm" variant="secondary" data-ui-action="rag_send_research_agent" onClick={() => buildAgentContext("rag_send_research_agent")}>发送 Research Agent</Button><Button size="sm" variant="secondary" data-ui-action="rag_send_code_agent" onClick={() => buildAgentContext("rag_send_code_agent")}>发送 Code Agent</Button><Button size="sm" variant="secondary" data-ui-action="rag_bind_report_claim" onClick={() => buildAgentContext("rag_bind_report_claim")}>绑定 Report Claim</Button><Button size="sm" variant="secondary" data-ui-action="rag_request_citation_audit" onClick={() => buildAgentContext("rag_request_citation_audit")}>请求引用审计</Button></div>
            <div className="grid grid-cols-2 gap-2"><Button size="sm" variant="secondary" data-ui-action="rag_export_context_markdown" onClick={exportRagMarkdown}><Download className="h-4 w-4" />导出 Context</Button><Button size="sm" variant="secondary" data-ui-action="rag_export_manifest_json" onClick={exportRagManifest}><Database className="h-4 w-4" />导出 Manifest</Button></div>
            <Button className="w-full" size="sm" variant="secondary" data-ui-action="rag_refresh_index" onClick={() => void runSearch(query, "refresh_index")}><RefreshCw className="h-4 w-4" />刷新 RAG 索引</Button>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-2 xl:grid-cols-[1.05fr_1fr_1fr]">
        <Card className="min-w-0 overflow-hidden"><CardHeader className="px-3 py-2.5"><CardTitle>检索结果 / RAG Retrieval Results</CardTitle><CardDescription>真实检索结果</CardDescription></CardHeader><CardContent className="px-3 pb-3"><DenseTable headers={["#", "Chunk", "Score", "Source", "Page", "Artifact", "Used"]} rows={retrievalRows} /></CardContent></Card>
        <Card className="min-w-0 overflow-hidden"><CardHeader className="px-3 py-2.5"><CardTitle>可复用策略记忆 / Reusable Strategy Memory</CardTitle><CardDescription>由命中文献抽取</CardDescription></CardHeader><CardContent className="px-3 pb-3"><DenseTable headers={["Strategy", "paper_id", "Family", "Exp", "Benefit", "Risk"]} rows={strategyRows} /></CardContent></Card>
        <Card className="min-w-0 overflow-hidden"><CardHeader className="px-3 py-2.5"><CardTitle>引用 / Claim 审计</CardTitle><CardDescription>Citation & Claim Audit</CardDescription></CardHeader><CardContent className="px-3 pb-3"><DenseTable headers={["Claim", "Paper", "Exp", "Artifact", "Status"]} rows={claimRows} /></CardContent></Card>
      </div>
      <Card className="min-w-0 overflow-hidden">
        <CardHeader className="px-3 py-2.5"><div className="flex items-start justify-between gap-3"><div><CardTitle>知识同步日志 / RAG Index Log</CardTitle><CardDescription>解析、chunk、embedding、dedup、风险审计全部留痕</CardDescription></div><Button size="sm" variant="secondary" data-ui-action="rag_open_all_logs">全部日志</Button></div></CardHeader>
        <CardContent className="px-3 pb-3">
          <DenseTable headers={["时间 (UTC+8)", "Agent", "事件类型", "源 / Source", "详情", "状态"]} rows={[
            [rag?.generated_at ? new Date(rag.generated_at).toLocaleString() : "-", "LiteratureSearchAgent", "动态检索", "local + arXiv", `${metrics.paper_count} papers / ${metrics.chunk_count} chunks`, rag ? "成功" : "等待"],
            [rag?.generated_at ? new Date(rag.generated_at).toLocaleString() : "-", "ContextBuilder", "Agent Context", rag?.context_path ?? "-", `${metrics.context_tokens} tokens`, rag ? "已写入" : "等待"],
            [rag?.generated_at ? new Date(rag.generated_at).toLocaleString() : "-", "ClaimAuditAgent", "边界检查", "RAG manifest", "禁止将文献命中写成官方提分", "通过"]
          ]} />
        </CardContent>
      </Card>
      <BottomSummary phase="RAG Index & Citation Audit" best={`${metrics.paper_count} papers / ${metrics.chunk_count} chunks / citation confidence ${metrics.citation_confidence}%`} next={rag?.context_path ?? "构建 Agent Context"} pending={ragStatus} />
    </Page>
  );
}

export function AgentRuntime(props: ScreenProps) {
  const [activeTraceTab, setActiveTraceTab] = useState<"tool_calls" | "llm_cache">("tool_calls");
  const run = latestRunForTask(props);
  const action = latestActionForTask(props);
  const runtime = selectedRuntime(props);
  const terminalAgent = props.summary?.terminal_agent;
  const runtimeBlockers = Array.isArray(props.summary?.verified_launch_audit?.blockers) ? props.summary.verified_launch_audit.blockers : [];
  const cacheBlocked = runtimeBlockers.includes("deepseek_cache_below_80_for_batch_generation");
  const liveRunId = run?.id ?? runtime?.latest_experiment_dir?.split(/[\\/]/).pop() ?? "waiting";
  const runningAgents = runtime?.agent_trace?.length ? Math.min(runtime.agent_trace.length, 11) : terminalAgent?.event_count ? Math.min(terminalAgent.event_count, 11) : agents.filter(([, , status]) => status === "running").length;
  const failedActions = (props.summary?.actions ?? []).filter((item) => /fail|blocked|error/i.test(item.message ?? item.action)).length
    + (terminalAgent?.recent_events ?? []).filter((item) => /fail|blocked|error|repair/i.test(textValue(item.type ?? item.status ?? item.reason, ""))).length;
  const graph = ["Research", "Data Audit", "Feature Eng.", "Model Select", "Code Impl.", "GPU / HPC", "Validation", "Submission", "Claim Audit", "Report", "Recovery"];
  const traceRows = terminalAgentEventRows(terminalAgent, 8);
  const timeline = [
    ["10:20", "调度开始", "Orchestrator", "completed"],
    ["10:21", "研究上下文", "Research Agent", "completed"],
    ["10:26", "代码草稿生成", "Code Agent", "running"],
    ["10:28", "代码质量 Gate", "CodeQuality Gate", "waiting"],
    ["10:30", "HPC 作业启动", "GPU Agent", "running"],
    ["10:32", "失败回退 #1", "Reflection Agent", "failed"],
    ["10:34", "提交 Gate 阻断", "Submission Gate", "waiting"],
    ["10:38", "报告草稿等待", "Report Agent", "pending"]
  ];
  const traceTabRows = activeTraceTab === "tool_calls"
    ? [["read_artifact", "success", "0.32s"], ["call_deepseek", "success", "1.62s"], ["generate_code", "success", "3.41s"], ["run_smoke_test", "success", "1.11s"], ["create_hpc_manifest", "success", "0.74s"], ["submit_gpu_job", "running", "5.23s"], ["pull_artifacts", "waiting", "--"], ["run_claim_audit", "waiting", "--"]]
    : [["prompt_cache_lookup", "hit", "0.08s"], ["bounded_context_hash", "matched", "ctx_91f2"], ["deepseek_batch_cache", "blocked", "cost_guard"], ["retrospective_memory", "hit", "3 records"], ["token_budget", "ok", "18.4k / 32k"], ["cost_guard", "active", "$0.84"], ["cache_target", "watch", ">= 80%"], ["cache_miss_reason", "none", "--"]];
  const traceTabHeaders = activeTraceTab === "tool_calls" ? ["工具调用", "状态", "耗时"] : ["缓存/模型项", "状态", "详情"];
  return (
    <Page title="AI Research Workstation - Agent Runtime" subtitle="Realtime trace, tool calls, failure recovery, runtime timeline, and cache telemetry.">
      <LiveRunEvidencePanel {...props} />
      <EvolutionRuntimePanel taskId={props.selectedTask} refreshSummary={props.refreshSummary} />
      <TerminalKaggleAgentPanel {...props} />
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_390px]">
        <div className="space-y-2">
          <Card className="overflow-hidden">
            <CardContent className="flex min-h-[54px] flex-wrap items-center gap-3 p-2.5 text-xs">
              <div className="flex min-w-[330px] items-center gap-2">
                <span className="font-black text-slate-500">run_id</span>
                <span className={cn(mono, "rounded border border-slate-200 bg-white px-2 py-1 text-slate-900")}>{liveRunId}</span>
              </div>
              <div className="flex flex-1 flex-wrap items-center justify-center gap-3">
                <StatusBadge tone="green">运行证据 {runningAgents}</StatusBadge>
                <StatusBadge tone="amber">Gate {filteredGates(props).length}</StatusBadge>
                <StatusBadge tone={failedActions ? "red" : "green"}>异常 {failedActions}</StatusBadge>
                <StatusBadge tone="blue">{action?.action ?? "ready"}</StatusBadge>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-black text-slate-500">Last Heartbeat</span>
                <span className={cn(mono, "text-slate-900")}>{formatTime(action?.at)}</span>
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              </div>
              <Button size="sm" variant="secondary" data-ui-action="runtime_refresh_5s">刷新 5s</Button>
            </CardContent>
          </Card>

          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
            <RuntimeMetric icon={Users} label="Agent Evidence" value={String(runningAgents)} sub={terminalAgent?.events_present ? "live" : "summary"} tone={terminalAgent?.events_present ? "green" : "blue"} spark="blue" />
            <RuntimeMetric icon={TerminalSquare} label="XSCI Events" value={String(terminalAgent?.event_count ?? 0)} sub={terminalAgentStatusLabel(terminalAgent?.status)} tone={terminalAgent?.events_present ? "green" : "amber"} spark="green" />
            <RuntimeMetric icon={Box} label="Evolution Runs" value={String(terminalAgent?.completed_run_count ?? 0)} sub={`${terminalAgent?.run_count ?? 0} dirs`} tone="blue" spark="blue" />
            <RuntimeMetric icon={Activity} label="Memory Records" value={String(terminalAgent?.memory_count ?? 0)} sub={`${terminalAgent?.n_promotions ?? 0}/${terminalAgent?.n_iterations ?? 0} promoted`} tone={(terminalAgent?.memory_count ?? 0) ? "green" : "slate"} spark="green" />
          </div>

          <Panel title="Multi-Agent Execution Graph" description="Orchestrator fan-out with artifact edges and gate dependencies." action={<div className="flex gap-2 text-[11px] font-bold text-slate-500"><span>实线: 任务流</span><span>虚线: 产物链</span></div>}>
            <div className="relative overflow-hidden rounded-md border border-slate-200 bg-slate-50/80 p-3">
              <svg aria-hidden className="pointer-events-none absolute inset-x-6 top-[54px] hidden h-[72px] text-slate-300 xl:block" viewBox="0 0 1000 90" preserveAspectRatio="none">
                {Array.from({ length: 11 }).map((_, i) => (
                  <path key={i} d={`M500 0 C ${380 + i * 24} 24, ${120 + i * 78} 26, ${40 + i * 86} 76`} fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray={i > 6 ? "5 5" : "0"} />
                ))}
              </svg>
              <div className="relative mx-auto mb-10 w-fit rounded-md border border-blue-300 bg-blue-50 px-5 py-2 text-center shadow-[0_0_0_3px_rgba(37,99,235,0.08)]">
                <StatusBadge tone="blue">routing</StatusBadge>
                <div className="mt-1 text-xs font-black text-blue-800">Orchestrator Agent</div>
              </div>
              <div className="relative grid gap-2 md:grid-cols-4 xl:grid-cols-11">
                {graph.map((name, i) => (
                  <div key={name} className={cn("min-h-[82px] rounded-md border bg-white p-2 text-center shadow-[0_1px_1px_rgba(15,23,42,0.03)]", i < 2 ? "border-emerald-200" : i < 7 ? "border-blue-200" : i < 9 ? "border-amber-200" : i === 10 ? "border-red-200" : "border-slate-200")}>
                    <StatusBadge tone={i < 2 ? "green" : i < 7 ? "blue" : i < 9 ? "amber" : i === 10 ? "red" : "slate"}>{i < 2 ? "已完成" : i < 7 ? "运行中" : i < 9 ? "Gate" : i === 10 ? "重试" : "等待中"}</StatusBadge>
                    <div className="mt-2 text-[11px] font-black text-slate-800">{name}</div>
                    <div className={cn(mono, "mt-1 truncate rounded bg-slate-50 px-1.5 py-1 text-[10px] text-slate-500")}>{i === 4 ? "train.py" : i === 5 ? "job.yaml" : i === 10 ? "recovery.md" : `${name.toLowerCase().replaceAll(" ", "_")}.json`}</div>
                  </div>
                ))}
              </div>
            </div>
          </Panel>

          <Panel title="Agent 状态矩阵 (11/11)" description="任务、进度、产物、失败回退与人工 Gate 状态同屏可见。">
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5">
              {agents.map(([name, task, status, artifact, progress], index) => (
                <div key={name} className={cn("rounded-md border bg-white p-2.5 shadow-[0_1px_1px_rgba(15,23,42,0.035)]", status === "failed" ? "border-red-200 bg-red-50/30" : status === "waiting" ? "border-amber-200 bg-amber-50/25" : status === "running" ? "border-blue-200 bg-blue-50/20" : "border-emerald-200")}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className={cn("flex h-5 w-5 shrink-0 items-center justify-center rounded border text-[10px] font-black", index < 2 ? "border-emerald-200 bg-emerald-50 text-emerald-700" : status === "failed" ? "border-red-200 bg-red-50 text-red-700" : "border-blue-200 bg-blue-50 text-blue-700")}>{index + 1}</span>
                      <div className="truncate text-xs font-black text-slate-950">{name}</div>
                    </div>
                    <StatusBadge tone={toneFor(status)}>{status}</StatusBadge>
                  </div>
                  <div className="mt-1.5 min-h-8 text-[11px] leading-4 text-slate-500">{task}</div>
                  <Progress value={progress} tone={toneFor(status)} />
                  <div className={cn(mono, "mt-1.5 truncate rounded border border-slate-100 bg-white px-2 py-1 text-[11px] text-blue-700")}>{artifact}</div>
                  <div className="mt-1.5 flex justify-between text-[10px] font-bold text-slate-500">
                    <span>{status === "failed" ? "retry 1" : status === "waiting" ? "Gate g_sub_01" : "tool ok"}</span>
                    <span>{progress}%</span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>

          <div className="grid gap-2 xl:grid-cols-[1fr_1fr_360px]">
            <Panel title="Agent 运行时间线" description="Gate-aware event line.">
              <div className="thin-scrollbar overflow-x-auto py-1">
                <div className="flex min-w-[760px] items-start gap-1.5">
                  {timeline.map(([time, stage, agent, state], i) => (
                    <div key={`${time}-${stage}`} className="relative w-[92px] text-center">
                      {i > 0 ? <div className="absolute -left-5 top-6 h-px w-8 border-t border-dashed border-slate-300" /> : null}
                      <div className={cn("mx-auto flex h-7 w-7 items-center justify-center rounded-full border bg-white text-[10px] font-black", toneFor(state) === "green" ? "border-emerald-300 text-emerald-700" : toneFor(state) === "red" ? "border-red-300 text-red-700" : toneFor(state) === "amber" ? "border-amber-300 text-amber-700" : "border-blue-300 text-blue-700")}>{i + 1}</div>
                      <div className={cn(mono, "mt-1 text-[10px] text-slate-500")}>{time}</div>
                      <div className="mt-1 text-[11px] font-black leading-4 text-slate-800">{stage}</div>
                      <div className="text-[10px] font-semibold text-slate-500">{agent}</div>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
            <Panel title="Failure & Recovery" description="失败生成 artifact，且不覆盖 best-so-far。">
              <DenseTable headers={["agent","failure","retry","protected"]} rows={[["Reflection Agent","tool_timeout","1","yes"],["GPU Agent","hpc_job_failed","1","yes"],["Code Agent","code_generation_error","0","yes"], ["Submission Gate", "schema_error", "0", "yes"]]} />
            </Panel>
            <Panel title="Agent Performance" description="Runtime telemetry.">
              <div className="grid grid-cols-2 gap-2">
                <StatMini label="Cache Hit" value={cacheBlocked ? "<80% gated" : "cache ok"} tone={cacheBlocked ? "amber" : "green"} />
                <StatMini label="Tokens" value="1.82M" tone="blue" />
                <StatMini label="Latency" value="1.28s" tone="green" />
                <StatMini label="Fail Rate" value="6.1%" tone="amber" />
              </div>
            </Panel>
          </div>
          <Panel title="Action Log / Event Stream" description="All agent starts, completions, failures and recoveries are audit logged."><DenseTable headers={["time","agent","action","status","artifact"]} rows={traceRows} /></Panel>
        </div>
        <aside className="space-y-2 xl:sticky xl:top-[68px] xl:self-start">
          <Panel title="Trace Detail" description="Selected agent runtime context." action={<button className="text-xs font-black text-slate-400" data-ui-action="runtime_close_trace_detail">×</button>}>
            <div className="rounded-md border border-blue-200 bg-blue-50 p-3">
              <div className="flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-md bg-blue-600 text-white"><Code2 className="h-4 w-4" /></span>
                <div>
                  <div className="text-sm font-black text-blue-950">Code Implementation Agent</div>
                  <div className={cn(mono, "text-[11px] text-blue-700")}>ag_code_impl_03</div>
                </div>
              </div>
            </div>
            <Row label="角色" value="代码实现与训练脚本生成" />
            <Row label="分配任务" value="task_code_impl_91" monoValue />
            <Row label="Bounded Context" value="研究问题 + 特征 + 模型路线" />
            <Row label="状态" value={<StatusBadge tone="blue">运行中</StatusBadge>} />
            <Row label="开始时间" value="10:21:50" />
            <Row label="运行时长" value="00:16:10" />
            <Row label="输入产物" value="model_select.json, features.parquet" monoValue />
            <Row label="输出产物" value="train.py, config.yaml, requirements.txt" monoValue />
            <Row label="Gate 依赖" value="g_code_quality_01" monoValue />
            <div className="mt-3 grid grid-cols-2 gap-2">
              <button className={cn("rounded-md border px-2 py-1.5 text-xs font-black", activeTraceTab === "tool_calls" ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50")} data-ui-action="runtime_trace_tab_tool_calls" data-ui-skip-action="true" data-active-trace-tab={activeTraceTab === "tool_calls" ? "true" : "false"} onClick={() => setActiveTraceTab("tool_calls")}>Tool Calls</button>
              <button className={cn("rounded-md border px-2 py-1.5 text-xs font-black", activeTraceTab === "llm_cache" ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50")} data-ui-action="runtime_trace_tab_llm_cache" data-ui-skip-action="true" data-active-trace-tab={activeTraceTab === "llm_cache" ? "true" : "false"} onClick={() => setActiveTraceTab("llm_cache")}>LLM / Cache</button>
            </div>
            <div data-active-trace-panel={activeTraceTab}>
              <DenseTable headers={traceTabHeaders} rows={traceTabRows} />
            </div>
          </Panel>
          <Panel title="LLM / Cache 信息" description="模型、缓存与成本预算。">
            <Row label="模型" value="DeepSeek-V3.1 (32k)" />
            <Row label="缓存命中" value={<StatusBadge tone={cacheBlocked ? "amber" : "green"}>{cacheBlocked ? "<80% gated" : "HIT"}</StatusBadge>} />
            <Row label="上下文预算" value="18.4k / 32k" />
            <Row label="推理次数" value="12" />
            <Row label="累计延迟" value="24.6s" />
            <Row label="本轮成本" value="$0.84" />
          </Panel>
          <Panel title="选中 Agent 下一步" description="只显示允许动作，不绕过 Gate。">
            <div className="grid gap-2">
              <Button variant="secondary" size="sm" data-ui-action="runtime_view_agent_context">查看上下文</Button>
              <Button variant="secondary" size="sm" data-ui-action="runtime_open_agent_artifact">打开产物</Button>
              <Button variant="secondary" size="sm" data-ui-action="runtime_request_code_gate">请求 Code Gate</Button>
              <Button variant="secondary" size="sm" aria-disabled="true" data-ui-action="blocked_submit_gpu_job"><Lock className="h-4 w-4" />提交 GPU Job</Button>
            </div>
          </Panel>
        </aside>
      </div>
    </Page>
  );
}

function RuntimeMetric({ icon: Icon, label, value, sub, tone, spark }: { icon: React.ElementType; label: string; value: string; sub: string; tone: StatusTone; spark: "blue" | "green" }) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[11px] font-black text-slate-700">{label}</div>
            <div className="mt-1 flex items-end gap-2">
              <span className="text-[23px] font-black leading-6 text-slate-950">{value}</span>
              <StatusBadge tone={tone}>{sub}</StatusBadge>
            </div>
          </div>
          <span className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-md border", tone === "green" ? "border-emerald-100 bg-emerald-50 text-emerald-700" : tone === "amber" ? "border-amber-100 bg-amber-50 text-amber-700" : "border-blue-100 bg-blue-50 text-blue-700")}>
            <Icon className="h-3.5 w-3.5" />
          </span>
        </div>
        <div className="mt-2 flex h-5 items-end gap-1 border-t border-slate-100 pt-1.5">
          {[18, 24, 16, 29, 22, 36, 25, 42, 32, 47, 35, 41].map((h, i) => (
            <span key={i} className={cn("w-full rounded-t-sm", spark === "green" ? "bg-emerald-300" : "bg-blue-300")} style={{ height: `${h}%` }} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}





export function IntegrityGates(props: ScreenProps) {
  const s = props.summary;
  return (
    <Page title="完整性 Gate" subtitle="人工审批、提交阻断、安全边界与完整性校验">
      <LiveRunEvidencePanel {...props} />
      <EvolutionGatesPanel taskId={props.selectedTask} />
      <div className="grid gap-3 md:grid-cols-4">
        <StatCard icon={ShieldAlert} label="Overall Decision" value="blocked" sub="regression" tone="red" />
        <StatCard icon={LineChart} label="Score Protection" value="0.966590" sub="best" tone="green" />
        <StatCard icon={Upload} label="Submission Permission" value="false" sub="blocked" tone="red" />
        <StatCard icon={UserCheck} label="Human Approval" value="pending" sub="2 reviewers" tone="amber" />
      </div>
      <GatePipeline />
      <ScoreRecoveryBlock />
      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <Panel title="Agent Work Order" description="Gate 自动分派回退任务">
          {["EnvironmentAgent: Verify GPU SSH gateway before long training", "ModelSelectionAgent: Preserve EXP007 safe baseline", "CodeImplementationAgent: Generate only workstation-controlled diff", "ValidationAgent: Recompute OOF and schema audit"].map((x) => <div key={x} className="mb-2 rounded-md border border-blue-100 bg-blue-50 p-3 text-sm font-semibold text-blue-900">{x}</div>)}
        </Panel>
        <Panel title="Human Approval Console" description="人工审批面板">
          <textarea className="min-h-[120px] w-full rounded-md border border-slate-200 p-3 text-sm" placeholder="审核意见..." />
          <div className="mt-3 grid grid-cols-2 gap-2"><Button variant="success" data-ui-action="approve_integrity_gate">Approve</Button><Button variant="danger" data-ui-action="reject_integrity_gate">Reject</Button><Button variant="secondary" data-ui-action="request_gate_revision">Request Revision</Button><Button variant="secondary" aria-disabled="true" data-ui-action="blocked_allow_official_submit">Allow Official Submit</Button></div>
        </Panel>
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="Gate Evidence Checklist" description="证据检查"><DenseTable headers={["artifact", "status"]} rows={[["metrics.json", "passed"], ["oof_pred.parquet", "passed"], ["submission.csv", "waiting"], ["claim_audit.json", "passed"], ["kaggle_response.json", "missing"]]} /></Panel>
        <Panel title="Decision History" description="决策历史"><DenseTable headers={["gate", "decision", "time"]} rows={[["gate_code_v1", "approved", "10:21"], ["gate_hpc_v1", "approved", "10:28"], ["gate_submission", "waiting", "10:38"]]} /></Panel>
        <Panel title="Risk Boundary" description="科研风险边界"><DenseTable headers={["risk", "level"]} rows={[["CV-public gap", "watch"], ["data leakage", "safe"], ["overfitting", "medium"], ["unsupported claim", "blocked"]]} /></Panel>
      </div>
    </Page>
  );
}

function ScoreRecoveryBlock() {
  return (
    <Panel title="Score Regression Recovery" description="低分候选被阻断并回退给责任 Agent" action={<StatusBadge tone="red">regression confirmed</StatusBadge>}>
      <div className="grid gap-3 md:grid-cols-4">
        <StatMini label="CURRENT SCORE" value="0.965910" tone="red" />
        <StatMini label="PROTECTED BEST" value="0.966590" tone="green" />
        <StatMini label="VALIDATION FLOOR" value="0.965743" tone="green" />
        <StatMini label="SUBMIT ALLOWED" value="false" tone="red" />
      </div>
      <div className="mt-4 rounded-md border border-red-100 bg-red-50 p-3 text-sm font-semibold text-red-800">Score gate must prove the candidate beats the protected baseline before official submit.</div>
    </Panel>
  );
}

function ExperimentKpi({ icon: Icon, label, value, badge, delta, tone }: { icon: React.ElementType; label: string; value: string; badge: string; delta: string; tone: StatusTone }) {
  return (
    <button data-ui-action={`experiments_kpi_${label.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="rounded-md border border-slate-200 bg-white p-3 text-left shadow-[0_1px_2px_rgba(15,23,42,0.03)] transition hover:border-blue-200 hover:bg-blue-50/35">
      <div className="flex items-start justify-between gap-2">
        <span className={cn("flex h-8 w-8 items-center justify-center rounded-md border", tone === "green" ? "border-emerald-100 bg-emerald-50 text-emerald-700" : tone === "amber" ? "border-amber-100 bg-amber-50 text-amber-700" : tone === "red" ? "border-red-100 bg-red-50 text-red-700" : "border-blue-100 bg-blue-50 text-blue-700")}>
          <Icon className="h-4 w-4" />
        </span>
        <StatusBadge tone={tone}>{badge}</StatusBadge>
      </div>
      <div className="mt-3 text-sm font-black text-slate-800">{label}</div>
      <div className="mt-1 flex items-end justify-between gap-2">
        <span className="text-[26px] font-black leading-7 text-slate-950">{value}</span>
        <MiniLine tone={tone} />
      </div>
      <div className="mt-1 text-[11px] font-bold text-slate-500"><span className="text-emerald-600">↑</span> {delta}</div>
    </button>
  );
}

function ExperimentNode({ title, tag, tone, rows, className }: { title: string; tag: string; tone: StatusTone; rows: string[]; className?: string }) {
  const border = tone === "green" ? "border-emerald-400 bg-emerald-50/90" : tone === "blue" ? "border-blue-300 bg-blue-50/90" : tone === "amber" ? "border-amber-300 bg-amber-50/90" : tone === "red" ? "border-red-300 bg-red-50/90" : "border-slate-300 bg-white";
  return (
    <button data-ui-action={`experiments_select_node_${title.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className={cn("w-[150px] rounded-md border p-2 text-left shadow-[0_6px_18px_-18px_rgba(15,23,42,0.4)] transition hover:-translate-y-0.5 hover:shadow-[0_10px_24px_-20px_rgba(37,99,235,0.8)]", border, className)}>
      <div className="flex items-start justify-between gap-1">
        <span className="truncate text-[12px] font-black text-slate-950">{title}</span>
        <StatusBadge tone={tone}>{tag}</StatusBadge>
      </div>
      <div className="mt-2 space-y-1">
        {rows.map((row) => (
          <div key={row} className={cn("truncate text-[10px] font-bold", row.includes("+") ? "text-emerald-700" : row.includes("-") ? "text-red-600" : "text-slate-600")}>{row}</div>
        ))}
      </div>
    </button>
  );
}

function LegendLine({ label, tone, dashed = false }: { label: string; tone: StatusTone; dashed?: boolean }) {
  const color = tone === "red" ? "border-red-500" : tone === "blue" ? "border-blue-500" : "border-slate-600";
  return (
    <span className="inline-flex items-center gap-2">
      <span className={cn("w-8 border-t-2", color, dashed && "border-dashed")} />
      {label}
    </span>
  );
}

function GateRow({ label, status, tone }: { label: string; status: string; tone: StatusTone }) {
  return (
    <button data-ui-action={`gate_row_${label.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="flex w-full items-center justify-between gap-2 border-b border-slate-100 py-1 text-left text-[11px] last:border-b-0 hover:bg-slate-50">
      <span className="font-bold text-slate-700">{label}</span>
      <StatusBadge tone={tone}>{status}</StatusBadge>
    </button>
  );
}

function FormRow({ label, value, action }: { label: string; value: string; action: string }) {
  return (
    <label className="grid grid-cols-[112px_1fr_auto] items-center gap-2">
      <span className="font-black text-slate-700">{label}</span>
      <input className="h-8 min-w-0 rounded-md border border-slate-200 bg-white px-3 font-semibold text-slate-800 outline-none transition focus:border-blue-300 focus:ring-2 focus:ring-blue-100" value={value} readOnly />
      <Button size="icon" variant="secondary" data-ui-action={action}><ChevronRight className="h-3.5 w-3.5" /></Button>
    </label>
  );
}

function SelectLike({ label, value }: { label: string; value: string }) {
  return (
    <button data-ui-action={`tasks_select_${label.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className="grid gap-1 text-left">
      <span className="font-black text-slate-700">{label}</span>
      <span className="flex h-8 items-center justify-between rounded-md border border-slate-200 bg-white px-3 font-semibold text-slate-800 hover:border-blue-200 hover:bg-blue-50/40">
        {value}
        <ChevronRight className="h-3.5 w-3.5 rotate-90 text-slate-400" />
      </span>
    </button>
  );
}

function AgentAssignmentCard({ name, status, input, output, progress }: { name: string; status: string; input: string; output: string; progress: number }) {
  return (
    <button data-ui-action={`tasks_agent_${name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} className={cn("min-h-[118px] rounded-md border bg-white p-2 text-left transition hover:border-blue-200 hover:bg-blue-50/40", status === "blocked" ? "border-red-200" : status === "waiting" ? "border-amber-200" : status === "verified" ? "border-emerald-200" : "border-blue-200")}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-black text-slate-950">{name}</span>
        <StatusBadge tone={toneFor(status)}>{status}</StatusBadge>
      </div>
      <div className="mt-2 space-y-1 text-[10px] font-bold text-slate-600">
        <RowCompact label="输入" value={input} />
        <RowCompact label="输出" value={output} />
      </div>
      <div className="mt-2"><Progress value={progress} tone={toneFor(status)} /></div>
    </button>
  );
}

function StatusDot({ tone, label }: { tone: StatusTone; label: string }) {
  const color = tone === "green" ? "bg-emerald-500" : tone === "amber" ? "bg-amber-500" : tone === "red" ? "bg-red-500" : "bg-blue-500";
  return <span className="inline-flex items-center gap-1.5"><span className={cn("h-2 w-2 rounded-full", color)} />{label}</span>;
}

export function Experiments(props: ScreenProps) {
  const ledgerRows = [
    ["house_prices", "exp_9981", "xgb_stack", "0.8412", "no response", "+0.0031", "preserve_best", "supported", "12", "ModelSelectionAgent", "1h ago"],
    ["titanic", "exp_8812", "round4", "0.8623", "no response", "+0.0048", "promote_round4", "supported", "16", "ValidationAnalysisAgent", "2h ago"],
    ["telco_churn", "exp_7715", "round4", "0.8078", "no response", "-0.0012", "needs_revision", "weak", "9", "BlendSearchAgent", "3h ago"],
    ["credit_risk", "exp_6610", "baseline", "0.7431", "no response", "-0.0153", "hold", "blocked", "7", "RiskAuditAgent", "5h ago"]
  ];
  return (
    <Page title="实验中心" subtitle="实验台账、分支搜索与分数提升门禁">
      <LiveRunEvidencePanel {...props} />
      <EvolutionSearchGraphPanel taskId={props.selectedTask} />
      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        <ExperimentKpi icon={Database} label="Tasks" value="9" badge="active" delta="1 本周新增" tone="blue" />
        <ExperimentKpi icon={Play} label="Runs" value="96" badge="recorded" delta="14 今日新增" tone="green" />
        <ExperimentKpi icon={LineChart} label="Promoted" value="7" badge="promoted" delta="2 本周" tone="green" />
        <ExperimentKpi icon={Clock3} label="Held" value="14" badge="held" delta="3 本周" tone="amber" />
        <ExperimentKpi icon={AlertTriangle} label="Rollback / Failed" value="5" badge="rollback" delta="1 本周" tone="red" />
        <ExperimentKpi icon={ShieldCheck} label="Best Protected" value="11" badge="best protected" delta="保护中 11 项" tone="blue" />
      </div>

      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Panel
          title="Experiment Search Graph"
          description="MLEvolve-style 分支搜索图"
          action={
            <div className="flex items-center gap-1">
              <Button size="sm" variant="secondary" data-ui-action="experiments_layout_graph"><Layers3 className="h-3.5 w-3.5" />布局</Button>
              <Button size="sm" variant="secondary" data-ui-action="experiments_filter_graph"><Filter className="h-3.5 w-3.5" />筛选</Button>
              <Button size="icon" variant="secondary" data-ui-action="experiments_zoom_out">-</Button>
              <Button size="sm" variant="secondary" data-ui-action="experiments_zoom_reset">100%</Button>
              <Button size="icon" variant="secondary" data-ui-action="experiments_zoom_in">+</Button>
            </div>
          }
        >
          <div className="thin-scrollbar relative min-h-[300px] overflow-x-auto rounded-md border border-slate-200 bg-white">
            <div className="workflow-bg relative h-[300px] min-w-[1000px] p-5">
              <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1000 360" preserveAspectRatio="none">
                <path d="M135 178 C210 178 235 176 300 176" stroke="#475569" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
                <path d="M440 154 C485 118 515 112 560 112" stroke="#475569" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
                <path d="M440 202 C485 250 515 254 560 254" stroke="#475569" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
                <path d="M690 112 C735 112 760 150 808 150" stroke="#64748b" strokeDasharray="5 4" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
                <path d="M690 174 C735 174 760 150 808 150" stroke="#64748b" strokeDasharray="5 4" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
                <path d="M448 190 C500 226 535 250 560 270" stroke="#ef4444" strokeDasharray="5 4" strokeWidth="1.5" fill="none" markerEnd="url(#arrowRed)" />
                <defs>
                  <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#475569" /></marker>
                  <marker id="arrowRed" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#ef4444" /></marker>
                </defs>
              </svg>
              <ExperimentNode className="absolute left-[18px] top-[116px]" title="EXP000 Baseline" tag="baseline" tone="slate" rows={["CV 0.7312  Official no response", "Agent: BaselineAgent", "2024-05-10"]} />
              <ExperimentNode className="absolute left-[185px] top-[116px]" title="EXP003 LGBM" tag="baseline" tone="slate" rows={["CV 0.7945  Official no response", "Agent: FeatureAgent", "2024-05-11"]} />
              <ExperimentNode className="absolute left-[355px] top-[72px] w-[185px]" title="EXP017 Blend" tag="best" tone="green" rows={["CV 0.8412  Official no response", "+0.0467  best-so-far", "Agent: ModelSelectionAgent", "2024-05-13"]} />
              <ExperimentNode className="absolute left-[615px] top-[45px]" title="EXP024 Frontier" tag="candidate" tone="blue" rows={["CV 0.8380  Official no response", "-0.0032", "Agent: FrontierAgent", "2024-05-14"]} />
              <ExperimentNode className="absolute left-[615px] top-[140px]" title="EXP033 Offset" tag="candidate" tone="blue" rows={["CV 0.8361  Official no response", "-0.0051", "Agent: OffsetAgent", "2024-05-14"]} />
              <ExperimentNode className="absolute left-[815px] top-[102px]" title="EXP034 Dual Blend" tag="candidate" tone="blue" rows={["CV 0.8350  Official no response", "-0.0062", "Agent: BlendAgent", "2024-05-14"]} />
              <ExperimentNode className="absolute left-[355px] top-[205px]" title="EXP012 Lightweight" tag="held" tone="amber" rows={["CV 0.7720  Official no response", "-0.0225", "Agent: LightweightAgent", "2024-05-12"]} />
              <ExperimentNode className="absolute left-[615px] top-[225px]" title="EXP009 Overfit Attempt" tag="rollback" tone="red" rows={["CV 0.7011  Official no response", "-0.1401", "Agent: RiskAuditAgent", "2024-05-12"]} />
            </div>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-bold text-slate-600">
            <LegendLine label="主分支" tone="slate" />
            <LegendLine label="探索分支" tone="blue" dashed />
            <LegendLine label="回滚分支" tone="red" dashed />
            <StatusBadge tone="green">best</StatusBadge>
            <StatusBadge tone="blue">candidate</StatusBadge>
            <StatusBadge tone="amber">held</StatusBadge>
            <StatusBadge tone="red">rollback</StatusBadge>
            <StatusBadge tone="slate">baseline</StatusBadge>
          </div>
        </Panel>

        <aside className="space-y-2">
          <Panel title="Selected Experiment Detail" action={<StatusBadge tone="green">best protected</StatusBadge>}>
            <Row label="Task" value="house_prices" monoValue />
            <Row label="Branch" value="blend_search_v2" monoValue />
            <Row label="Parent" value="EXP003_LGBM" monoValue />
            <Row label="Owner Agent" value="ModelSelectionAgent" />
            <Row label="Best-so-far" value={<StatusBadge tone="green">true</StatusBadge>} />
            <Row label="Latest Score" value="CV 0.8412 / official no response (+0.0467 proxy)" />
            <Row label="Tags" value={<span className="flex flex-wrap gap-1"><StatusBadge tone="slate">blend</StatusBadge><StatusBadge tone="slate">stack</StatusBadge><StatusBadge tone="slate">v2</StatusBadge></span>} />
          </Panel>
          <Panel title="Score Promotion Gate">
            <GateRow label="CV uplift threshold >= +0.005" status="passed" tone="green" />
            <GateRow label="Official drift check" status="waiting" tone="amber" />
            <GateRow label="Regression guard vs prev best" status="passed" tone="green" />
            <GateRow label="Human gate required?" status="pending" tone="amber" />
            <div className="mt-2 rounded-md border border-amber-100 bg-amber-50 px-2 py-1 text-center text-[11px] font-black text-amber-700">Gate Status: pending human review</div>
          </Panel>
          <Panel title="Claim Audit">
            <GateRow label="Leaderboard claim supported" status="supported" tone="green" />
            <GateRow label="Official score missing" status="blocked" tone="amber" />
            <GateRow label="Evidence coverage" status="92%" tone="green" />
            <GateRow label="Unsupported claim count" status="0" tone="green" />
          </Panel>
          <Panel title="Artifact Manifest" action={<StatusBadge tone="slate">5 files</StatusBadge>}>
            {["metrics.json  12.4 KB", "oof_pred.parquet  54.1 MB", "blend_config.yaml  1.2 KB", "submission_candidate.csv  3.3 MB", "claim_audit.json  8.7 KB"].map((x) => (
              <button key={x} data-ui-action={`experiments_open_artifact_${x.split(" ")[0].replace(/[^a-z0-9_]/gi, "_")}`} className="flex w-full items-center justify-between gap-2 rounded px-1 py-0.5 text-left text-[11px] font-bold text-slate-700 hover:bg-blue-50">
                <span className="truncate text-emerald-700">› {x.split("  ")[0]}</span>
                <span className="shrink-0 text-slate-500">{x.split("  ")[1]}</span>
              </button>
            ))}
          </Panel>
          <Panel title="Next Action Recommendation">
            <p className="text-xs font-semibold leading-5 text-slate-700">Promote EXP017 to next frontier branch; hold EXP024 pending CV-public stability; block official submission until submission gate review.</p>
            <div className="mt-3 grid grid-cols-3 gap-2">
              <Button size="sm" variant="primary" data-ui-action="experiments_review_gate" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("review_experiment_gate", { task_id: props.selectedTask, exp_id: "EXP017" })}>Review Gate</Button>
              <Button size="sm" variant="secondary" data-ui-action="experiments_open_artifacts"><FileText className="h-3.5 w-3.5" />Artifacts</Button>
              <Button size="sm" variant="secondary" data-ui-action="experiments_launch_branch"><Play className="h-3.5 w-3.5" />Branch</Button>
            </div>
          </Panel>
        </aside>
      </div>

      <Panel
        title="Experiment Ledger"
        description="最新实验台账"
        action={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="secondary" data-ui-action="experiments_filter_ledger"><Filter className="h-3.5 w-3.5" />筛选</Button>
            <Button size="sm" variant="secondary" data-ui-action="experiments_export_ledger"><Download className="h-3.5 w-3.5" />导出</Button>
          </div>
        }
      >
        <div className="thin-scrollbar overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-[11px]">
            <thead>
              <tr className="border-y border-slate-200 bg-slate-50 text-[10px] uppercase tracking-[0.03em] text-slate-500">
                {["Task", "Exp", "Branch", "CV", "Official", "Delta", "Decision", "Claim", "Artifacts", "Agent", "Updated", "操作"].map((h) => <th key={h} className="px-3 py-2 font-black">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {ledgerRows.map((row) => (
                <tr key={row[1]} data-ui-component="dense-table-row" className="border-b border-slate-100 bg-white hover:bg-blue-50/40">
                  {row.map((cell, index) => (
                    <td key={`${row[1]}-${index}`} className={cn("px-3 py-2 font-bold text-slate-700", index === 5 && (String(cell).startsWith("+") ? "text-emerald-700" : "text-red-600"), index < 2 && mono)}>{cell}</td>
                  ))}
                  <td className="px-3 py-2">
                    <div className="flex gap-1">
                      <Button size="icon" variant="secondary" data-ui-action={`experiments_view_${row[1]}`}><Search className="h-3.5 w-3.5" /></Button>
                      <Button size="icon" variant="secondary" data-ui-action={`experiments_open_folder_${row[1]}`}><FileText className="h-3.5 w-3.5" /></Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </Page>
  );
}

export function ResearchTasks(props: ScreenProps) {
  const summaryTasks = props.summary?.tasks ?? [];
  const tasks = summaryTasks.length > 0
    ? summaryTasks.map((t: any) => [
        t.id ?? "unknown",
        t.name ?? t.id ?? "Untitled",
        t.task_type ?? "tabular",
        t.metric ?? "accuracy",
        t.priority ?? "P1",
        t.status ?? "pending",
        t.owner ?? "Workstation",
        t.resource ?? "GPU",
        "open"
      ] as const)
    : [
        ["loading", "Loading tasks from API...", "", "", "", "loading", "API", "", "pending"]
      ];
  return (
    <Page title="任务队列" subtitle="任务配置、上下文与运行入口">
      <LiveRunEvidencePanel {...props} />
      <div className="grid gap-2 xl:grid-cols-[minmax(0,1.42fr)_minmax(420px,0.95fr)]">
        <Panel
          title="Task Queue"
          description="编排与监控科研任务的全生命周期"
          action={
            <div className="flex flex-wrap gap-1">
              <StatusBadge tone="blue">总计 {tasks.length}</StatusBadge>
              <Button size="icon" variant="ghost" data-ui-action="tasks_refresh_queue"><RefreshCw className="h-4 w-4" /></Button>
            </div>
          }
        >
          <div className="thin-scrollbar overflow-x-auto">
            <table className="w-full min-w-[820px] text-left text-[11px]">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[10px] uppercase tracking-[0.03em] text-slate-500">
                  {["Task", "Competition", "Type", "Metric", "Priority", "Status", "Owner", "Resource", "Gate"].map((h) => <th key={h} className="px-3 py-2 font-black">{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {tasks.map((task) => {
                  const selected = props.selectedTask === task[0];
                  return (
                    <tr
                      key={task[0]}
                      data-ui-component="dense-table-row"
                      data-ui-action={`tasks_select_${task[0]}`}
                      data-ui-skip-action="true"
                      data-selected-task={selected ? "true" : "false"}
                      onClick={() => props.setSelectedTask(task[0])}
                      className={cn("cursor-pointer border-b border-slate-100 bg-white hover:bg-blue-50/50", selected && "bg-blue-50/80")}
                    >
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2">
                          <span className={cn("h-3.5 w-3.5 rounded-full border", selected ? "border-blue-600 bg-blue-600 shadow-[inset_0_0_0_3px_white]" : "border-slate-300 bg-white")} />
                          <span className={cn(mono, "font-black text-slate-900")}>{task[0]}</span>
                        </div>
                      </td>
                      <td className="px-3 py-3 font-bold text-slate-700">{task[1]}</td>
                      <td className="px-3 py-3 font-bold text-slate-700">{task[2]}</td>
                      <td className="px-3 py-3 font-bold text-slate-700">{task[3]}</td>
                      <td className="px-3 py-3"><StatusBadge tone={task[4] === "P0" ? "red" : task[4] === "P1" ? "amber" : "slate"}>{task[4]}</StatusBadge></td>
                      <td className="px-3 py-3"><StatusBadge tone={toneFor(task[5])}>{task[5]}</StatusBadge></td>
                      <td className="px-3 py-3"><StatusBadge tone="slate">{task[6]}</StatusBadge></td>
                      <td className="px-3 py-3"><StatusBadge tone={task[7].includes("A800") ? "green" : task[7].includes("queue") ? "blue" : "slate"}>{task[7]}</StatusBadge></td>
                      <td className="px-3 py-3"><StatusBadge tone={toneFor(task[8])}>{task[8]}</StatusBadge></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between px-3 py-2 text-xs font-bold text-slate-500">
            <span>4 tasks</span>
            <div className="flex items-center gap-2"><Button size="icon" variant="ghost" data-ui-action="tasks_prev_page">‹</Button><span className="rounded border border-slate-200 px-2 py-1">1</span><Button size="icon" variant="ghost" data-ui-action="tasks_next_page">›</Button></div>
          </div>
        </Panel>

        <Panel title="Task Intake / Create Task" description="从数据到提交的标准化入口">
          <div className="grid gap-2 text-xs">
            <FormRow label="Competition URL" value="https://www.kaggle.com/competitions/playground-series-s6e6" action="tasks_open_competition_url" />
            <FormRow label="Dataset path" value="/datasets/playground-series/s6e6/train.csv" action="tasks_choose_dataset_path" />
            <div className="grid gap-2 md:grid-cols-3">
              <SelectLike label="Metric" value="accuracy" />
              <SelectLike label="Time budget" value="6h" />
              <SelectLike label="Submission limit" value="/ day" />
            </div>
            <div className="mt-1 grid gap-2 md:grid-cols-3">
              <Button variant="primary" data-ui-action="tasks_create_workstation_run" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("create_workstation_run", { task_id: props.selectedTask })}>Create Workstation Run</Button>
              <Button variant="secondary" data-ui-action="tasks_dispatch_agents" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("dispatch_agents", { task_id: props.selectedTask })}><Upload className="h-4 w-4" />Dispatch Agents</Button>
              <Button variant="secondary" data-ui-action="tasks_open_context"><FileText className="h-4 w-4" />Open Context</Button>
            </div>
            <div className="mt-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-semibold text-slate-600">官方提交将需要通过 Gate 审批（人审 / 规则校验 / 风险评估）</div>
          </div>
        </Panel>
      </div>

      <div className="grid gap-2 xl:grid-cols-[minmax(0,1.05fr)_minmax(430px,0.95fr)]">
        <div className="space-y-2">
          <Panel title="Selected Task Contract" description="当前任务规范">
            <div className="grid overflow-hidden rounded-md border border-slate-200 text-[11px] md:grid-cols-4">
              {[
                ["selectedTask", props.selectedTask],
                ["task_id", "tsk_20250606_s6e6_7f93c1"],
                ["metric", "accuracy"],
                ["stage", props.selectedStage],
                ["experiment", props.selectedExperiment],
                ["baseline", "0.8743 (CV)"],
                ["current best", "0.8921 (CV)"],
                ["target rank (pct)", "top 10%"]
              ].map(([label, value]) => (
                <button key={label} data-ui-action={`tasks_contract_${label.replace(/[^a-z0-9]+/gi, "_")}`} className="min-h-[54px] border-b border-r border-slate-200 bg-white p-3 text-left last:border-r-0 hover:bg-blue-50">
                  <div className="font-black text-slate-500">{label}</div>
                  <div className={cn("mt-1 font-black text-slate-950", mono)}>{value}</div>
                </button>
              ))}
            </div>
          </Panel>
          <Panel title="Agent Assignment Board" description="多智能体协同执行与审计" action={<Button size="sm" variant="secondary" data-ui-action="tasks_view_agent_logs"><FileText className="h-3.5 w-3.5" />查看日志</Button>}>
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
              <div className="grid gap-2 md:grid-cols-4 xl:grid-cols-5">
                {[
                  ["Orchestrator", "running", "Task Contract + Context", "plan.json", 65],
                  ["Data Audit", "verified", "/datasets/.../train.csv", "data_audit.json", 100],
                  ["Research", "running", "research_brief.md", "literature_brief.md", 72],
                  ["Feature", "running", "data_audit.json", "features.parquet", 58],
                  ["Model", "waiting", "features.parquet", "model_select.json", 0],
                  ["Code", "running", "model_select.json", "train.py", 41],
                  ["HPC", "running", "train.py", "gpu_job_manifest.yaml", 28],
                  ["Validation", "waiting", "gpu_job_manifest.yaml", "metrics.json", 0],
                  ["Report", "verified", "metrics.json", "teacher_report.md", 100],
                  ["Claim Audit", "blocked", "teacher_report.md", "claim_audit.json", 0]
                ].map(([name, status, input, output, progress]) => (
                  <AgentAssignmentCard key={name as string} name={name as string} status={status as string} input={input as string} output={output as string} progress={Number(progress)} />
                ))}
              </div>
              <div className="mt-3 flex flex-wrap justify-center gap-3 rounded-md border border-slate-200 bg-white px-3 py-2 text-[11px] font-bold text-slate-600">
                <StatusDot tone="blue" label="running 运行中" />
                <StatusDot tone="amber" label="waiting 等待中" />
                <StatusDot tone="green" label="verified 已验证" />
                <StatusDot tone="red" label="blocked 阻塞" />
              </div>
            </div>
          </Panel>
        </div>

        <aside className="space-y-2">
          <Panel title="Validation Contract" action={<Button size="sm" variant="secondary" data-ui-action="tasks_copy_validation_contract"><FileText className="h-3.5 w-3.5" />复制</Button>}>
            <div className="space-y-2 text-[11px] leading-5 text-slate-700">
              <div><span className="font-black text-slate-950">hypothesis</span><br />通过分层特征与模型集成，提升 CV accuracy &gt;= +0.004</div>
              <div><span className="font-black text-slate-950">acceptance criteria</span><br />CV 提升 &gt;= 0.004；官方 response 缺失时不得计算 Official/CV gap</div>
              <div><span className="font-black text-slate-950">risk checklist</span></div>
              {["preserve best-so-far", "public/CV gap watch", "submission schema check", "human gate before official submit"].map((x, i) => (
                <div key={x} className="flex items-center gap-2"><CheckCircle2 className={cn("h-3.5 w-3.5", i === 3 ? "text-amber-500" : "text-emerald-500")} />{x}</div>
              ))}
            </div>
          </Panel>
          <Panel title="Task Context & Resources" action={<Button size="sm" variant="secondary" data-ui-action="tasks_refresh_resources"><RefreshCw className="h-3.5 w-3.5" />刷新</Button>}>
            {[
              ["Dataset readiness", "/datasets/.../s6e6", "就绪", "green"],
              ["Kaggle API readiness", "额度 98% 剩余", "就绪", "green"],
              ["HPC / GPU readiness", "1x A800 可用", "就绪", "green"],
              ["DeepSeek / Code Agent cache", "命中率 92%", "就绪", "green"],
              ["Human Gate status", "人工审核通道", "受控", "amber"]
            ].map(([name, detail, state, tone]) => (
              <button key={name} data-ui-action={`tasks_resource_${name.replace(/[^a-z0-9]+/gi, "_")}`} className="mb-1 flex w-full items-center justify-between gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-left text-[11px] font-bold hover:bg-blue-50">
                <span className="text-slate-700">{name}</span>
                <span className="min-w-0 flex-1 truncate text-right text-slate-500">{detail}</span>
                <StatusBadge tone={tone as StatusTone}>{state}</StatusBadge>
              </button>
            ))}
          </Panel>
          <div className="grid gap-2 md:grid-cols-2">
            <Panel title="Recent Failure">
              <div className="rounded-md border border-red-100 bg-red-50 p-3 text-xs font-semibold leading-5 text-red-800">
                上一次失败（exp_20250606_181233）<br />CV 下降: -0.0018（较 best）<br />原因: 过拟合，public/CV gap 扩大<br />时间: 2025-06-06 18:35:12
              </div>
            </Panel>
            <Panel title="Rollback Suggestion">
              <div className="rounded-md border border-amber-100 bg-amber-50 p-3 text-xs font-semibold leading-5 text-amber-800">
                建议：保持提交，回滚至 <span className={mono}>exp_20250606_174512</span><br />并将任务路由至 Validation Agent
              </div>
            </Panel>
          </div>
          <Panel title="Next Step">
            <div className="grid gap-2 md:grid-cols-[1fr_auto]">
              <div className="text-xs font-semibold text-slate-600">下一步：验证集评估与误差分析</div>
              <Button variant="primary" data-ui-action="tasks_route_validation_agent" data-ui-skip-action="true" onClick={() => props.runWorkstationAction?.("route_to_validation_agent", { task_id: props.selectedTask })}>Route to Validation Agent <ArrowRight className="h-4 w-4" /></Button>
            </div>
          </Panel>
        </aside>
      </div>
    </Page>
  );
}

export function WorkflowGraph(props: ScreenProps) {
  return (
    <Page title="流程编排" subtitle="多 Agent 工作流、任务分派、状态回退和证据交接">
      <LiveRunEvidencePanel {...props} />
      <AgentGraph />
      <GatePipeline />
      <Panel title="Workflow Contract" description="每个阶段输入、输出、失败回退">
        <DenseTable headers={["stage", "input", "output", "fallback"]} rows={[["Research", "task spec", "plan.json", "return_to_research"], ["Data Audit", "dataset", "data_audit.json", "return_to_data"], ["Code", "plan + context", "train.py", "return_to_code"], ["HPC", "manifest", "metrics.json", "return_to_hpc"], ["Report", "evidence", "teacher_report.pdf", "return_to_report"]]} />
      </Panel>
    </Page>
  );
}

export function SettingsCenter(props: ScreenProps) {
  const locale = props.locale ?? "zh-CN";
  const [activeSection, setActiveSection] = useState("Account");
  const [settingsFeedback, setSettingsFeedback] = useState("当前设置视图：Account");
  const connectorStatus = props.summary?.connector_status as Record<string, unknown> | undefined;
  const connectorState = (name: string, fallback: string) => {
    const raw = connectorStatus?.[name];
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      const record = raw as Record<string, unknown>;
      return String(record.status ?? record.state ?? fallback);
    }
    return fallback;
  };
  const toneForState = (state: string): StatusTone => {
    const normalized = state.toLowerCase();
    if (/(ready|passed|online|configured|enabled)/.test(normalized)) return "green";
    if (/(blocked|closed|pending|warning|needs|expired|rejected)/.test(normalized)) return "amber";
    if (/(failed|error|missing|not_configured|offline)/.test(normalized)) return "red";
    return "slate";
  };
  const connectorRows = [
    ["Kaggle", connectorState("kaggle", props.summary?.kaggle_dpapi_readiness?.configured ? "ready" : "unknown")],
    ["HPC / GPU", connectorState("hpc_gpu_ssh", "configured_channels_closed")],
    ["DeepSeek", connectorState("deepseek", "ready")],
    ["Claude Code", connectorState("claude_code", "ready_via_deepseek_fallback")],
    ["RAG / Vector", connectorState("rag", "not_configured")],
    ["报告导出", "ready"]
  ];
  const setLanguage = (language: Locale) => {
    props.setLocale?.(language);
    void props.runWorkstationAction?.("language_select", { task_id: props.selectedTask, language });
  };
  const recordSettingsAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, { task_id: props.selectedTask, ...metadata });
    setSettingsFeedback(`${action}: ${String(metadata.section ?? metadata.theme ?? metadata.language ?? "recorded")}`);
  };
  const sections = ["Account", "Profile", "Language & Region", "Appearance", "Security & Credentials", "Resource Connectors", "Agent Runtime", "Kaggle & Submission", "GPU / HPC", "RAG & Knowledge Base", "Notifications", "Audit Log", "Design Governance", "Backup & Export", "Advanced"];
  const sectionPanels: Record<string, { title: string; description: string; rows: string[][] }> = {
    Account: {
      title: "A. Account & Role",
      description: "登录方式、会话状态、角色与权限概览",
      rows: [["当前状态", "已登录 (2FA)", "Owner"], ["最近登录", "2026-06-25 10:38:21", "可信设备"], ["活跃设备", "3 台", "可审计"], ["权限范围", "全局权限", "Research Admin"]],
    },
    Profile: {
      title: "B. Profile",
      description: "用户身份、汇报角色与联系方式",
      rows: [["显示名", "科研管理员", "可编辑"], ["邮箱", "ra_admin@researchlab.ai", "已验证"], ["组织", "Research Lab", "默认"], ["角色标签", "Owner / Research Admin", "同步中"]],
    },
    "Language & Region": {
      title: "C. Language & Region",
      description: "语言、时区、格式与度量单位",
      rows: [["语言", locale === "zh-CN" ? "简体中文" : "English", "即时生效"], ["时区", "Asia/Shanghai (UTC+8)", "已锁定"], ["日期格式", "YYYY-MM-DD HH:mm", "中文区"], ["度量单位", "metric", "默认"]],
    },
    Appearance: {
      title: "D. Appearance",
      description: "主题、密度、字体、编辑器与图表",
      rows: [["主题", "浅色 / 黑色", "可切换"], ["字体", "Inter + Mono", "已加载"], ["密度", "科研高密度", "默认"], ["代码编辑器", "审计注释开启", "已启用"]],
    },
    "Security & Credentials": {
      title: "E. Security & Credentials",
      description: "密钥保险库与凭据管理",
      rows: [["保险库", "可信任 (12)", "AES-256-GCM"], ["凭据扫描", "通过 (2h ago)", "自动"], ["需轮换", "1 项", "需要人工确认"], ["已过期", "1 项", "阻断敏感动作"]],
    },
    "Resource Connectors": {
      title: "F. Resource Connectors",
      description: "Kaggle、DeepSeek、HPC、RAG 与报告导出连接状态",
      rows: connectorRows.map(([name, state]) => [name, state, toneForState(state)]),
    },
    "Agent Runtime": {
      title: "G. Agent Runtime Settings",
      description: "模型、缓存、重试与策略",
      rows: [["DeepSeek", connectorState("deepseek", "ready"), "主代码 Agent"], ["缓存命中", "86%", "目标 >= 80%"], ["重试策略", "2 次自动回退", "已启用"], ["成本守护", "开启", "批量生成受控"]],
    },
    "Kaggle & Submission": {
      title: "H. Kaggle & Submission Policy",
      description: "官方提交策略与人类 Gate 规则",
      rows: [["Kaggle API", connectorState("kaggle", "ready"), "凭据脱敏"], ["官方提交", "人工 Gate 必须启用", "默认阻断"], ["Leaderboard claim", "必须绑定 response", "禁止代理分数冒充"], ["提交预算", "保守批量", "任务级限制"]],
    },
    "GPU / HPC": {
      title: "I. GPU / HPC Gateway Settings",
      description: "远程计算网关与资源模板",
      rows: [["HPC / GPU", connectorState("hpc_gpu_ssh", "configured_channels_closed"), "资源门禁"], ["作业模板", "manifest only", "禁止旁路"], ["产物回传", "stdout/stderr/metrics", "必须登记"], ["取消策略", "timeout + cancel record", "已定义"]],
    },
    "RAG & Knowledge Base": {
      title: "J. RAG & Knowledge Base",
      description: "文献、论文、知识库与 Agent 上下文",
      rows: [["向量库", connectorState("rag", "not_configured"), "可配置"], ["文献证据", "claim-bound", "必须引用"], ["Agent Context", "bounded", "按阶段裁剪"], ["报告引用", "citation audit", "需要审计"]],
    },
    Notifications: {
      title: "K. Notifications & Alerts",
      description: "通知渠道与告警规则",
      rows: [["站内信", "开启", "实时"], ["邮件摘要", "开启", "每日"], ["GPU 阻断", "高优先级", "立即提醒"], ["Gate 待审", "高优先级", "立即提醒"]],
    },
    "Audit Log": {
      title: "L. Audit Log",
      description: "所有设置变更都进入 evidence ledger",
      rows: [["最近变更", "5 条", "已记录"], ["审计范围", "设置 / 凭据 / Gate", "全覆盖"], ["导出", "CSV / bundle", "可用"], ["保留策略", "项目周期内", "默认"]],
    },
    "Design Governance": {
      title: "M. Design Governance",
      description: "视觉 token、组件规范、页面基线与 API 契约",
      rows: [["页面覆盖", "10/10", "设计基线"], ["组件一致性", "94%", "继续优化"], ["Figma Gate", "metadata verified", "可编辑结构"], ["UI 合约", "151 actions", "已通过"]],
    },
    "Backup & Export": {
      title: "N. Backup & Export",
      description: "配置备份、报告导出与证据包",
      rows: [["配置备份", "daily snapshot", "开启"], ["报告导出", "PDF / DOCX", "Gate 控制"], ["证据包", "artifact manifest", "可生成"], ["恢复点", "最近 7 天", "保留"]],
    },
    Advanced: {
      title: "O. Advanced",
      description: "高级策略、实验开关与危险操作",
      rows: [["实验开关", "feature flags", "受控"], ["危险操作", "disabled by default", "Gate 必须"], ["调试日志", "verbose optional", "默认关闭"], ["系统版本", "v2.3.1", "Asia/Shanghai"]],
    },
  };
  const selectSection = (section: string) => {
    setActiveSection(section);
    recordSettingsAction("open_settings_section", { section });
  };
  const activePanel = sectionPanels[activeSection] ?? sectionPanels.Account;
  return (
    <Page title="系统设置" subtitle="账号、语言、主题、凭据、资源与工作站偏好">
      <LiveRunEvidencePanel {...props} />
      <div className="grid gap-2 xl:grid-cols-[1fr_0.85fr_1.35fr_1.3fr]">
        <Panel title="账户与角色" description="Account & Role" className="min-h-[146px]">
          <div className="flex items-center gap-3">
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-base font-black text-white">RA</span>
            <div className="min-w-0">
              <div className="text-base font-black text-slate-950">科研管理员</div>
              <div className="text-xs font-semibold text-slate-500">ra_admin@researchlab.ai</div>
              <div className="mt-1.5 flex gap-1.5"><StatusBadge tone="blue">Owner</StatusBadge><StatusBadge tone="green">Research Admin</StatusBadge></div>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-4 text-xs font-black"><span className="text-emerald-700">已登录</span><span className="text-emerald-700">2FA 已启用</span></div>
        </Panel>
        <Panel title="工作区安全" description="Workspace Security" className="min-h-[146px]">
          <Row label="密钥保险库" value="可信任 (12)" />
          <Row label="加密强度" value="AES-256-GCM" />
          <Row label="凭据扫描" value="通过 (2h ago)" />
          <Row label="审计日志" value="实时记录中" />
        </Panel>
        <Panel title="运行时连接器" description="Runtime Connectors" className="min-h-[146px]">
          <div className="grid gap-x-6 md:grid-cols-2">
            {connectorRows.map(([name, state]) => (
              <Row key={name} label={name} value={<StatusBadge tone={toneForState(state)}>{state}</StatusBadge>} />
            ))}
          </div>
        </Panel>
        <Panel title="偏好设置" description="Preferences" className="min-h-[146px]">
          <Row label="语言 / Language" value={locale === "zh-CN" ? "简体中文" : "English"} />
          <div className="my-1 grid grid-cols-2 gap-1.5">
            <button data-ui-action="settings_language_zh_cn" data-ui-skip-action="true" onClick={() => setLanguage("zh-CN")} className={cn("h-7 rounded-md border text-[11px] font-black", locale === "zh-CN" ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-600")}>简体中文</button>
            <button data-ui-action="settings_language_en_us" data-ui-skip-action="true" onClick={() => setLanguage("en-US")} className={cn("h-7 rounded-md border text-[11px] font-black", locale === "en-US" ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-600")}>English</button>
          </div>
          <Row label="主题 / Theme" value={<span className="inline-flex gap-1.5"><button data-ui-action="settings_theme_light" data-ui-skip-action="true" onClick={() => recordSettingsAction("settings_theme_change", { theme: "light" })} className="inline-flex h-6 items-center gap-1 rounded-md border border-blue-200 bg-blue-50 px-2 text-[11px] font-black text-blue-700"><Sun className="h-3 w-3" />浅色</button><button data-ui-action="settings_theme_dark" data-ui-skip-action="true" onClick={() => recordSettingsAction("settings_theme_change", { theme: "dark" })} className="inline-flex h-6 items-center gap-1 rounded-md border border-slate-200 bg-white px-2 text-[11px] font-black text-slate-600"><Moon className="h-3 w-3" />黑色</button></span>} />
          <Row label="时区 / Timezone" value="Asia/Shanghai (UTC+8)" />
          <Row label="通知 / Notifications" value="即时推送 + 邮件摘要" />
        </Panel>
      </div>

      <div className="grid gap-2 xl:grid-cols-[230px_1fr]">
        <Panel title="设置分类" description="Settings Sections" className="h-[330px] overflow-hidden">
          {sections.map((x, i) => (
            <button key={x} data-ui-action={`open_settings_section_${x.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`} data-ui-skip-action="true" data-active-section={activeSection === x ? "true" : "false"} onClick={() => selectSection(x)} className={cn("mb-0.5 flex h-6 w-full items-center gap-2 rounded-md px-2.5 text-left text-xs font-black", activeSection === x ? "bg-blue-50 text-blue-700 ring-1 ring-blue-100" : "text-slate-700 hover:bg-slate-50")}>
              <Settings className="h-3.5 w-3.5" />{x}
            </button>
          ))}
        </Panel>
        <div className="space-y-2">
          <Panel title={activePanel.title} description={activePanel.description}>
            <div className="grid gap-3 xl:grid-cols-[250px_0.75fr_0.85fr_1.2fr]">
              <div className="rounded-md border border-slate-200 p-3">
                <div className="flex items-center gap-3"><span className="flex h-11 w-11 items-center justify-center rounded-full bg-blue-600 text-base font-black text-white">RA</span><div><div className="font-black text-slate-950">科研管理员</div><div className="text-xs text-slate-500">ra_admin@researchlab.ai</div></div></div>
                <Row label="User ID" value="usr_3f7a9c2b8d1e" monoValue />
                <div className="mt-1.5 flex gap-1.5"><StatusBadge tone="blue">Owner</StatusBadge><StatusBadge tone="green">Research Admin</StatusBadge></div>
              </div>
              <DenseTable headers={["设置项", "当前值", "状态"]} rows={activePanel.rows.slice(0, 4)} />
              <DenseTable headers={["门禁", "策略"]} rows={[["变更生效", activeSection.includes("Credential") || activeSection.includes("Kaggle") || activeSection.includes("GPU") ? "需要人工 Gate" : "即时或保存后生效"], ["审计记录", "写入 evidence ledger"], ["后端接口", "POST /api/workstation-actions"], ["当前 section", activeSection]]} />
              <DenseTable headers={["角色", "用户数", "权限范围"]} rows={[["Owner", "1", "全局权限"], ["Research Admin", "3", "资源与凭据"], ["Reviewer", "5", "审核与评论"], ["Agent Operator", "8", "运行与监控"]]} />
            </div>
          </Panel>
          <div className="grid gap-1.5 xl:grid-cols-2">
            <SettingsTile icon={Languages} section="Language & Region" active={activeSection === "Language & Region"} onSelect={selectSection} title="B. Language & Region" value="简体中文 / Asia-Shanghai" detail="语言、时区、格式与度量单位。" />
            <SettingsTile icon={Sun} section="Appearance" active={activeSection === "Appearance"} onSelect={selectSection} title="C. Appearance" value="浅色 / 黑色 / Inter" detail="主题、密度、字体、编辑器与图表。" />
            <SettingsTile icon={KeyRound} section="Security & Credentials" active={activeSection === "Security & Credentials"} onSelect={selectSection} title="D. Security & Credentials" value="1 需轮换，1 已过期" detail="密钥保险库与凭据管理。" warning />
            <SettingsTile icon={Network} section="Resource Connectors" active={activeSection === "Resource Connectors"} onSelect={selectSection} title="E. Resource Connectors" value="1 需重连，0 异常" detail="资源与服务连接状态。" warning />
            <SettingsTile icon={Bot} section="Agent Runtime" active={activeSection === "Agent Runtime"} onSelect={selectSection} title="F. Agent Runtime Settings" value="缓存命中 86% / 成本守护开启" detail="模型、缓存、重试与策略。" />
            <SettingsTile icon={ShieldCheck} section="Kaggle & Submission" active={activeSection === "Kaggle & Submission"} onSelect={selectSection} title="G. Kaggle & Submission Policy" value="人类 Gate 必须启用" detail="提交策略与人类 Gate 规则。" warning />
            <SettingsTile icon={Server} section="GPU / HPC" active={activeSection === "GPU / HPC"} onSelect={selectSection} title="H. GPU / HPC Gateway Settings" value="HPC 已连接" detail="远程计算网关与资源模板。" />
            <SettingsTile icon={Bell} section="Notifications" active={activeSection === "Notifications"} onSelect={selectSection} title="I. Notifications & Alerts" value="邮件 + 站内信" detail="通知渠道与告警规则。" />
            <SettingsTile icon={Layers3} section="Design Governance" active={activeSection === "Design Governance"} onSelect={selectSection} title="J. Design Governance" value="页面覆盖 10/10 / 组件一致 94%" detail="视觉 token、组件规范、页面基线与 API 契约。" />
          </div>
        </div>
      </div>

      <Panel title="最近设置变更审计 / Audit Log" description="每一项设置变更都写入 evidence ledger">
        <DenseTable headers={["时间 (UTC+8)", "变更人", "设置项", "旧状态", "新状态", "原因", "审批"]} rows={[["2026-06-25 10:34:11", "ra_admin", "kaggle.submission.require_gate", "false", "true", "启用官方提交人工 Gate", "owner_approve"], ["2026-06-25 10:10:02", "ra_admin", "security.credential.rag_embedding", "expired", "rotated", "RAG 嵌入凭据轮换", "owner_approve"], ["2026-06-24 21:55:33", "ops_agent", "resource.hpc.retry_policy", "max_retry=3", "max_retry=5", "提高 HPC 重试策略", "auto_approved"], ["2026-06-24 18:22:19", "ra_admin", "agent.runtime.cache_target", "0.80", "0.85", "提升缓存命中目标", "owner_approve"], ["2026-06-24 16:05:47", "ra_admin", "appearance.theme", "light", "light", "无", "N/A"]]} />
      </Panel>

      <DesignGovernancePanel />

      <div className="sticky bottom-0 z-10 mt-2 grid gap-2 rounded-md border border-slate-200 bg-white/96 p-2 backdrop-blur md:grid-cols-[1fr_auto_auto_auto_auto]">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-600"><CheckCircle2 className="h-4 w-4 text-blue-600" /><span>{settingsFeedback}。部分设置变更需要重新验证凭据或人工审批后生效。</span></div>
        <Button variant="primary" data-ui-action="save_settings_changes" data-ui-skip-action="true" onClick={() => recordSettingsAction("save_settings_changes")}><CheckCircle2 className="h-4 w-4" />保存更改</Button>
        <Button variant="secondary" data-ui-action="cancel_settings_changes" data-ui-skip-action="true" onClick={() => recordSettingsAction("cancel_settings_changes")}>取消</Button>
        <Button variant="secondary" data-ui-action="test_all_connectors" data-ui-skip-action="true" onClick={() => recordSettingsAction("test_all_connectors")}><Activity className="h-4 w-4" />测试所有连接</Button>
        <Button variant="secondary" data-ui-action="rotate_credentials_batch" data-ui-skip-action="true" onClick={() => recordSettingsAction("rotate_credentials_batch")}><RefreshCw className="h-4 w-4" />批量轮换凭据</Button>
      </div>
    </Page>
  );
}

function DesignGovernancePanel() {
  const tokenRows = [
    ["Primary", "#2563EB", "primary action / active navigation"],
    ["Success", "#10B981", "verified / completed"],
    ["Warning", "#F59E0B", "human gate / pending approval"],
    ["Danger", "#EF4444", "failed / blocked"],
    ["Neutral", "#64748B", "metadata / secondary text"],
    ["Research Purple", "#7C3AED", "experimental emphasis / special annotations"]
  ];
  const pages = [
    ["Overview / Mission Control", "/api/v1/overview", "done", "95%", "100%", "低"],
    ["Agent Runtime", "/api/v1/agent/runtime", "done", "95%", "92%", "低"],
    ["Workflow Control", "/api/v1/workflows", "api-ready", "80%", "70%", "中"],
    ["Code Agent IDE", "/api/v1/code/ide", "done", "90%", "85%", "低"],
    ["Report Studio", "/api/v1/reports", "needs-polish", "78%", "72%", "中"],
    ["GPU / HPC", "/api/v1/hpc/status", "api-ready", "84%", "76%", "中"],
    ["System Settings", "/api/v1/settings", "done", "88%", "88%", "低"]
  ];
  return (
    <Panel
      title="设计治理 / Design Governance"
      description="已降级为系统设置内治理面板：统一视觉 token、组件规则、页面基线与前后端接口契约"
      action={
        <div className="flex flex-wrap gap-1">
          <Button size="sm" variant="secondary" data-ui-action="design_export_spec"><Download className="h-3.5 w-3.5" />导出设计规范</Button>
          <Button size="sm" variant="secondary" data-ui-action="design_generate_contract"><Code2 className="h-3.5 w-3.5" />生成前端契约</Button>
          <Button size="sm" variant="secondary" data-ui-action="design_run_ui_audit"><Activity className="h-3.5 w-3.5" />运行 UI 体检</Button>
          <Button size="sm" variant="secondary" data-ui-action="design_sync_figma"><PaletteIcon />同步 Figma</Button>
        </div>
      }
    >
      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        <DesignMetric icon={BookOpen} label="页面覆盖率" value="10 / 10" status="verified" tone="blue" />
        <DesignMetric icon={Layers3} label="组件一致性" value="94%" status="needs polish" tone="amber" />
        <DesignMetric icon={Network} label="API 对接准备度" value="87%" status="backend-ready" tone="blue" />
        <DesignMetric icon={ShieldCheck} label="可访问性检查" value="91%" status="passed" tone="green" />
        <DesignMetric icon={LineChart} label="视觉还原度" value="82%" status="improving" tone="blue" />
        <DesignMetric icon={AlertTriangle} label="风险项" value="3" status="review" tone="red" />
      </div>
      <div className="mt-2 grid gap-2 xl:grid-cols-[1fr_1.1fr]">
        <div className="rounded-md border border-slate-200 bg-white">
          <div className="border-b border-slate-200 px-3 py-2 text-sm font-black text-slate-950">Design Tokens</div>
          <div className="thin-scrollbar overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-[11px]">
              <thead><tr className="bg-slate-50 text-[10px] uppercase text-slate-500">{["Token", "Value", "Usage"].map((h) => <th key={h} className="px-3 py-2 font-black">{h}</th>)}</tr></thead>
              <tbody>
                {tokenRows.map(([name, value, usage]) => (
                  <tr key={name} className="border-t border-slate-100 hover:bg-blue-50/40">
                    <td className="px-3 py-2 font-black text-slate-800">{name}</td>
                    <td className="px-3 py-2"><span className="inline-flex items-center gap-2"><span className="h-3 w-10 rounded-sm border border-slate-200" style={{ backgroundColor: value }} /> <span className={mono}>{value}</span></span></td>
                    <td className="px-3 py-2 font-semibold text-slate-600">{usage}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid gap-2 border-t border-slate-200 p-2 md:grid-cols-3">
            <DesignPill title="Radius" values={["4px", "6px", "8px"]} />
            <DesignPill title="Spacing" values={["8", "16", "24", "32"]} />
            <DesignPill title="Elevation" values={["Border", "Soft", "Modal"]} />
          </div>
        </div>
        <div className="rounded-md border border-slate-200 bg-white">
          <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
            <div className="text-sm font-black text-slate-950">Component Library</div>
            <div className="flex gap-1"><StatusBadge tone="blue">Agent 组件</StatusBadge><StatusBadge tone="amber">Gate 组件</StatusBadge><StatusBadge tone="green">数据组件</StatusBadge></div>
          </div>
          <div className="grid gap-2 p-3 md:grid-cols-2">
            <div>
              <div className="mb-2 text-xs font-black text-slate-700">Buttons</div>
              <div className="flex flex-wrap gap-2"><Button size="sm" variant="primary" data-ui-action="design_component_primary_button">Primary Button</Button><Button size="sm" variant="secondary" data-ui-action="design_component_secondary_button">Secondary Button</Button><Button size="sm" variant="ghost" data-ui-action="design_component_ghost_button">Ghost Button</Button><Button size="sm" disabled data-ui-action="design_component_disabled_button">Disabled</Button></div>
              <div className="mt-3 text-xs font-black text-slate-700">Status Badges</div>
              <div className="mt-2 flex flex-wrap gap-1"><StatusBadge tone="green">Ready</StatusBadge><StatusBadge tone="blue">Running</StatusBadge><StatusBadge tone="red">Blocked</StatusBadge><StatusBadge tone="amber">Needs Gate</StatusBadge></div>
            </div>
            <div className="grid gap-2">
              {["Research Agent /agent/research/*", "Code Agent /agent/code/*", "Validation Agent /agent/validation/*", "Plan Gate /gate/plan/*", "HPC Gate /gate/hpc/*", "Evidence Row /evidence/*"].map((x, i) => (
                <button key={x} data-ui-action={`design_component_${i}`} className="flex items-center justify-between rounded-md border border-slate-200 px-3 py-2 text-left text-[11px] font-bold hover:bg-blue-50">
                  <span>{x}</span>
                  <StatusBadge tone={i < 3 ? "blue" : i < 5 ? "amber" : "green"}>{i < 3 ? "Card" : i < 5 ? "Open" : "Row Click"}</StatusBadge>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="mt-2 grid gap-2 xl:grid-cols-[1fr_360px]">
        <div className="rounded-md border border-slate-200 bg-white">
          <div className="border-b border-slate-200 px-3 py-2 text-sm font-black text-slate-950">Page Baseline Matrix</div>
          <DenseTable headers={["Page", "Primary API", "Status", "Design Fidelity", "Interaction", "Risk"]} rows={pages} />
        </div>
        <div className="rounded-md border border-slate-200 bg-white">
          <div className="border-b border-slate-200 px-3 py-2 text-sm font-black text-slate-950">Quality Gate</div>
          <div className="p-2">
            {[
              ["Layout no overflow", "Passed", "green"],
              ["Text no overlap", "Passed", "green"],
              ["Click target >= 40px", "Passed", "green"],
              ["Real state binding", "Warning", "amber"],
              ["No fake Kaggle score", "Failed", "red"],
              ["No secret exposure", "Passed", "green"],
              ["Mobile responsive", "Passed", "green"]
            ].map(([label, state, tone]) => <GateRow key={label} label={label} status={state} tone={tone as StatusTone} />)}
          </div>
        </div>
      </div>
    </Panel>
  );
}

function DesignMetric({ icon: Icon, label, value, status, tone }: { icon: React.ElementType; label: string; value: string; status: string; tone: StatusTone }) {
  return (
    <button data-ui-action={`design_metric_${label.replace(/[^a-z0-9]+/gi, "_")}`} className="rounded-md border border-slate-200 bg-white p-2.5 text-left hover:border-blue-200 hover:bg-blue-50/35">
      <div className="flex items-center justify-between gap-2"><Icon className="h-4 w-4 text-blue-600" /><MiniLine tone={tone} /></div>
      <div className="mt-2 text-[11px] font-black text-slate-500">{label}</div>
      <div className="mt-1 flex items-end justify-between gap-2"><span className="text-[24px] font-black leading-6 text-slate-950">{value}</span><StatusBadge tone={tone}>{status}</StatusBadge></div>
    </button>
  );
}

function DesignPill({ title, values }: { title: string; values: string[] }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
      <div className="mb-1 text-[10px] font-black uppercase text-slate-500">{title}</div>
      <div className="flex flex-wrap gap-1">{values.map((x) => <span key={x} className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] font-black text-slate-700">{x}</span>)}</div>
    </div>
  );
}

function PaletteIcon() {
  return <span className="h-3.5 w-3.5 rounded-sm bg-[linear-gradient(135deg,#2563eb,#10b981,#f59e0b,#ef4444)]" />;
}


function SettingsTile({ icon: Icon, section, active, onSelect, title, value, detail, warning = false }: { icon: React.ElementType; section: string; active: boolean; onSelect: (section: string) => void; title: string; value: string; detail: string; warning?: boolean }) {
  return (
    <button
      className={cn("flex w-full items-center gap-2.5 rounded-md border bg-white p-2.5 text-left transition hover:border-blue-200 hover:bg-blue-50/40", active ? "border-blue-300 bg-blue-50 ring-1 ring-blue-100" : "border-slate-200")}
      data-ui-action={`open_settings_section_${section.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`}
      data-ui-skip-action="true"
      data-active-section={active ? "true" : "false"}
      onClick={() => onSelect(section)}
    >
      <span className={cn("flex h-8 w-8 items-center justify-center rounded-md border", warning ? "border-amber-100 bg-amber-50 text-amber-700" : "border-blue-100 bg-blue-50 text-blue-700")}><Icon className="h-4 w-4" /></span>
      <div className="min-w-0 flex-1"><div className="text-sm font-black leading-4 text-slate-950">{title}</div><div className="mt-0.5 text-xs font-bold text-slate-600">{value}</div><div className="mt-0.5 text-xs text-slate-500">{detail}</div></div>
      <ChevronRight className="h-4 w-4 text-slate-400" />
    </button>
  );
}

export function DesignSystem(props: ScreenProps) {
  return (
    <Page title="设计系统" subtitle="Research OS 视觉 token、组件规则与页面基线">
      <LiveRunEvidencePanel {...props} />
      <div className="grid gap-3 md:grid-cols-6">
        {(["blue", "green", "amber", "red", "slate", "purple"] as StatusTone[]).map((tone) => <Card key={tone}><CardContent className="p-4"><StatusBadge tone={tone}>{tone}</StatusBadge><div className="mt-3 h-12 rounded-md border border-slate-200 bg-slate-50" /></CardContent></Card>)}
      </div>
      <Panel title="页面基线" description="当前设计成果图覆盖的核心页面">
        <DenseTable headers={["page", "role", "status"]} rows={[["Mission Control", "总控", "done"], ["Data / Kaggle", "数据提交", "done"], ["Report Studio", "报告", "done"], ["Code Agent IDE", "代码", "done"], ["GPU / HPC", "算力", "done"], ["Evidence Ledger", "证据", "done"], ["Literature / RAG", "知识库", "done"], ["Agent Runtime", "运行时", "done"], ["Settings", "系统设置", "done"]]} />
      </Panel>
    </Page>
  );
}



