"use client";

import {
  ArrowRight,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ClipboardCheck,
  Code2,
  Cpu,
  Database,
  FileCheck2,
  FileText,
  FolderOpen,
  GitBranch,
  LineChart,
  Play,
  ShieldCheck,
  SlidersHorizontal,
  Upload
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import type { WorkstationSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

type Locale = "zh-CN" | "en-US";
type Connector = Record<string, unknown> | undefined;
type WorkstationPage = "code" | "report" | "runtime" | "evidence" | "gpu" | "gates" | "data" | "workflow" | "control";

type OverviewBoardEnhancedProps = {
  locale?: Locale;
  summary?: WorkstationSummary | null;
  selectedTask?: string;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
};

function ui(locale: Locale | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

function normalizeTaskId(taskId?: string) {
  const value = taskId || "playground_series_s6e6";
  return value === "house-prices" ? "house_prices" : value;
}

function connectorEntry(summary: WorkstationSummary | null | undefined, key: string) {
  return (summary?.connector_status as Record<string, Connector> | undefined)?.[key];
}

function isConfigured(entry: Connector) {
  return Boolean(entry?.configured);
}

function pickRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function scoreText(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(5);
  if (typeof value === "string" && value.trim()) return value;
  return "pending";
}

function compactValue(value: unknown, fallback = "n/a") {
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : fallback;
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  return fallback;
}

function statusTone(configured?: unknown, state?: string): StatusTone {
  const normalized = String(state ?? "").toLowerCase();
  if (configured && (normalized.includes("verified") || normalized.includes("ready") || normalized.includes("passed"))) return "green";
  if (configured) return "blue";
  if (normalized.includes("blocked") || normalized.includes("failed")) return "red";
  if (normalized.includes("not configured")) return "slate";
  return "amber";
}

function statusLabel(locale: Locale | undefined, configured?: unknown, state?: string) {
  const normalized = String(state ?? "").toLowerCase();
  if (configured && normalized.includes("verified")) return ui(locale, "Verified", "已验证");
  if (configured && (normalized.includes("ready") || normalized.includes("passed"))) return ui(locale, "Ready", "就绪");
  if (configured) return ui(locale, "Configured", "已配置");
  if (normalized.includes("not configured")) return ui(locale, "Not configured", "未配置");
  if (normalized.includes("blocked")) return ui(locale, "Blocked", "阻断");
  return ui(locale, "Pending", "待确认");
}

function runTone(status?: unknown): StatusTone {
  const value = String(status ?? "").toLowerCase();
  if (value.includes("complete") || value.includes("passed") || value.includes("ready")) return "green";
  if (value.includes("running") || value.includes("submitted")) return "blue";
  if (value.includes("blocked") || value.includes("failed") || value.includes("regression")) return "red";
  return "amber";
}

function openWorkstationPage(page: WorkstationPage) {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  url.searchParams.set("page", page);
  url.hash = "";
  window.history.pushState(null, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "blue"
}: {
  icon: typeof CheckCircle2;
  label: string;
  value: React.ReactNode;
  detail: string;
  tone?: StatusTone;
}) {
  const toneClass: Record<StatusTone, string> = {
    blue: "bg-blue-50 text-blue-700 border-blue-100",
    green: "bg-emerald-50 text-emerald-700 border-emerald-100",
    amber: "bg-amber-50 text-amber-700 border-amber-100",
    red: "bg-red-50 text-red-700 border-red-100",
    slate: "bg-slate-100 text-slate-600 border-slate-200",
    purple: "bg-violet-50 text-violet-700 border-violet-100"
  };
  return (
    <Card className="min-h-[112px]">
      <CardContent className="flex h-full items-center gap-4 p-4">
        <span className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-md border", toneClass[tone])}>
          <Icon className="h-5 w-5" />
        </span>
        <div className="min-w-0">
          <div className="text-xs font-bold text-slate-500">{label}</div>
          <div className="mt-1 truncate text-2xl font-bold tracking-normal text-slate-950">{value}</div>
          <div className="mt-1 truncate text-xs font-medium text-slate-500">{detail}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function SmallRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex min-w-0 items-start justify-between gap-3 border-b border-slate-100 py-2 text-xs last:border-b-0">
      <span className="shrink-0 font-semibold text-slate-500">{label}</span>
      <span className="min-w-0 break-words text-right font-bold text-slate-900">{compactValue(value)}</span>
    </div>
  );
}

function WorkspaceCard({
  title,
  description,
  status,
  tone,
  icon: Icon,
  page,
  actionLabel,
  testId
}: {
  title: string;
  description: string;
  status: string;
  tone: StatusTone;
  icon: typeof CheckCircle2;
  page: WorkstationPage;
  actionLabel: string;
  testId: string;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 shadow-[0_1px_1px_rgba(15,23,42,0.025)]">
      <div className="flex items-start justify-between gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-blue-100 bg-blue-50 text-blue-700">
          <Icon className="h-5 w-5" />
        </span>
        <StatusBadge tone={tone}>{status}</StatusBadge>
      </div>
      <div className="mt-4 text-sm font-bold text-slate-950">{title}</div>
      <p className="mt-2 min-h-12 text-xs leading-5 text-slate-600">{description}</p>
      <button
        className="mt-4 flex h-9 w-full items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-800 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        onClick={() => openWorkstationPage(page)}
        data-testid={testId}
        data-ui-skip-action="true"
      >
        {actionLabel}
        <ArrowRight className="h-4 w-4" />
      </button>
    </div>
  );
}

export function OverviewBoardEnhanced({
  locale,
  summary,
  selectedTask = "playground_series_s6e6",
  runWorkstationAction,
  runLocalExperiment
}: OverviewBoardEnhancedProps) {
  const normalizedTask = normalizeTaskId(selectedTask);
  const runtime = pickRecord(summary?.runtime);
  const scoreGate = pickRecord(runtime.score_improvement_gate);
  const recoveryPlan = pickRecord(runtime.score_regression_recovery_plan ?? runtime.score_regression_diagnosis);
  const readiness = summary?.kaggle_new_competition_readiness;
  const runs = summary?.runs ?? [];
  const workstationRuns = runs.filter((run) => run.workstation_run);
  const activeRun = workstationRuns[0] ?? runs[0];
  const gates = summary?.gates ?? [];
  const evidence = summary?.evidence ?? [];
  const approvedGates = gates.filter((gate) => /approved|passed|verified/i.test(String(gate.status ?? gate.decision ?? ""))).length;
  const pendingGates = gates.filter((gate) => /pending|waiting|manual/i.test(String(gate.status ?? gate.decision ?? ""))).length;
  const gpu = connectorEntry(summary, "gpu");
  const deepseek = connectorEntry(summary, "deepseek");
  const kaggle = connectorEntry(summary, "kaggle");
  const codeAgent = connectorEntry(summary, "code_agent");
  const currentBest = scoreGate.historical_best_public_score ?? recoveryPlan.historical_best_public_score ?? "0.96659";
  const latestScore = scoreGate.current_public_score ?? recoveryPlan.public_score ?? "0.95295";
  const scoreGap = typeof currentBest === "number" && typeof latestScore === "number"
    ? latestScore - currentBest
    : recoveryPlan.gap_to_best ?? "-0.01364";
  const officialSubmitAllowed = scoreGate.status === "passed" && gates.some((gate) => /submission/i.test(String(gate.gate_type ?? gate.id ?? "")) && /approved|passed/i.test(String(gate.status ?? gate.decision ?? "")));

  const metrics = [
    {
      icon: ClipboardCheck,
      label: ui(locale, "Task status", "任务状态"),
      value: activeRun?.status ?? "ready",
      detail: ui(locale, "workstation run state", "工作站 run 状态"),
      tone: runTone(activeRun?.status)
    },
    {
      icon: LineChart,
      label: ui(locale, "Official best", "官方最佳"),
      value: scoreText(currentBest),
      detail: ui(locale, "protected by score gate", "由分数门禁保护"),
      tone: "green"
    },
    {
      icon: ShieldCheck,
      label: ui(locale, "Latest score", "最新提交分"),
      value: scoreText(latestScore),
      detail: `${ui(locale, "gap", "差距")} ${compactValue(scoreGap)}`,
      tone: runTone(scoreGap)
    },
    {
      icon: FolderOpen,
      label: ui(locale, "Evidence files", "证据文件"),
      value: evidence.length,
      detail: ui(locale, "artifacts bound to claims", "产物绑定结论"),
      tone: evidence.length ? "green" : "amber"
    },
    {
      icon: Bot,
      label: ui(locale, "Agent runtime", "Agent 运行"),
      value: runtime.agent_trace ? "trace" : "mapped",
      detail: ui(locale, "failures and retries visible", "失败与重试可见"),
      tone: runtime.agent_trace ? "green" : "blue"
    },
    {
      icon: Cpu,
      label: ui(locale, "GPU/HPC", "GPU/HPC"),
      value: statusLabel(locale, isConfigured(gpu), String(gpu?.state ?? "")),
      detail: ui(locale, "SSH/CUDA gated", "SSH/CUDA 受控"),
      tone: statusTone(isConfigured(gpu), String(gpu?.state ?? ""))
    }
  ] as const;

  const workflow = [
    [ui(locale, "Task", "任务"), FileText, "done"],
    [ui(locale, "Data", "数据"), Database, readiness ? "done" : "pending"],
    [ui(locale, "Plan", "计划"), ClipboardCheck, activeRun ? "done" : "pending"],
    [ui(locale, "Code", "代码"), Code2, isConfigured(codeAgent) ? "done" : "manual"],
    [ui(locale, "GPU", "GPU"), Cpu, isConfigured(gpu) ? "done" : "manual"],
    [ui(locale, "Validate", "验证"), ShieldCheck, scoreGate.status ? "done" : "manual"],
    [ui(locale, "Submit", "提交"), Upload, officialSubmitAllowed ? "done" : "manual"],
    [ui(locale, "Report", "报告"), FileCheck2, evidence.length ? "done" : "pending"]
  ] as const;

  const resources = [
    { key: "gpu", name: "HPC / GPU", icon: Cpu, entry: gpu, evidence: "SSH + CUDA smoke" },
    { key: "deepseek", name: "DeepSeek", icon: BrainCircuit, entry: deepseek, evidence: ui(locale, "LLM backend", "大模型后端") },
    { key: "code_agent", name: "Claude Code Agent", icon: Code2, entry: codeAgent, evidence: ui(locale, "draft / patch gate", "草稿 / 补丁门禁") },
    { key: "kaggle", name: "Kaggle API", icon: Database, entry: kaggle, evidence: ui(locale, "download / submit gated", "下载 / 提交受控") }
  ];

  const workspaceCards: Array<{
    title: string;
    description: string;
    status: string;
    tone: StatusTone;
    icon: typeof CheckCircle2;
    page: WorkstationPage;
    actionLabel: string;
    testId: string;
  }> = [
    {
      title: ui(locale, "Code Agent Workspace", "代码 Agent 工作区"),
      description: ui(locale, "DeepSeek-backed Claude Code drafts code and patches; execution remains gated.", "DeepSeek 后端驱动 Claude Code 起草代码和补丁；执行仍由 Gate 控制。"),
      status: isConfigured(codeAgent) ? ui(locale, "configured", "已配置") : ui(locale, "needs check", "待检查"),
      tone: isConfigured(codeAgent) ? "green" : "amber",
      icon: Code2,
      page: "code" as const,
      actionLabel: ui(locale, "Open Code Agent", "进入代码 Agent"),
      testId: "open-code-agent-workspace"
    },
    {
      title: ui(locale, "Report Studio", "报告工作室"),
      description: ui(locale, "Generate teacher-ready evidence reports from real runs, logs and metrics.", "基于真实 run、日志和指标生成教师可读证据报告。"),
      status: evidence.length ? ui(locale, "evidence linked", "证据已绑定") : ui(locale, "draftable", "可起草"),
      tone: evidence.length ? "green" : "blue",
      icon: FileCheck2,
      page: "report" as const,
      actionLabel: ui(locale, "Open Reports", "进入报告"),
      testId: "open-report-studio"
    },
    {
      title: ui(locale, "Agent Runtime", "Agent 运行时"),
      description: ui(locale, "Inspect action logs, traces, retries and recovery decisions.", "查看 action log、trace、重试和回退决策。"),
      status: activeRun ? ui(locale, "run tracked", "Run 已记录") : ui(locale, "waiting", "待运行"),
      tone: activeRun ? "green" : "slate",
      icon: Bot,
      page: "runtime" as const,
      actionLabel: ui(locale, "Open Runtime", "查看运行时"),
      testId: "open-agent-runtime"
    },
    {
      title: ui(locale, "EvoMind Gateway", "EvoMind 工作站入口"),
      description: ui(locale, "Command the workstation with natural language; all actions are gated and logged.", "用自然语言指挥工作站；所有动作受 Gate 控制并记录。"),
      status: ui(locale, "ready", "就绪"),
      tone: "blue",
      icon: Bot,
      page: "control" as const,
      actionLabel: ui(locale, "Open EvoMind", "进入 EvoMind"),
      testId: "open-ai-control"
    },
    {
      title: ui(locale, "Integrity Gates", "完整性 Gate"),
      description: ui(locale, "Code, HPC execution, report and official submission stay human-gated.", "代码、HPC 执行、报告和官方提交都保留人工门禁。"),
      status: pendingGates ? ui(locale, "manual pending", "待人工") : ui(locale, "tracked", "已跟踪"),
      tone: pendingGates ? "amber" : "green",
      icon: ShieldCheck,
      page: "gates" as const,
      actionLabel: ui(locale, "Open Gates", "进入 Gate"),
      testId: "open-integrity-gates"
    }
  ];

  return (
    <div className="space-y-4" data-testid="enhanced-overview-board">
      <section className="grid gap-4 xl:grid-cols-[1.2fr_.8fr]">
        <Card className="border-slate-200 overflow-hidden">
          <CardHeader className="bg-slate-50 border-b border-slate-100 pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>{ui(locale, "EvoMind Gateway", "EvoMind 工作站入口")}</CardTitle>
                <CardDescription>{ui(locale, "Initialize runs and manage gates for task execution.", "初始化运行，管理任务执行门禁。")}</CardDescription>
              </div>
              <div className="inline-flex items-center gap-2 rounded-md border border-emerald-100 bg-emerald-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-emerald-700">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-600 animate-pulse" />
                System Active
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-4 sm:p-5">
            <div className="grid gap-3 sm:grid-cols-3">
              <button
                className="flex flex-col items-start rounded-md border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-blue-300 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                onClick={() => runWorkstationAction?.("create_workstation_run", {
                  task_id: readiness?.task_id ?? normalizedTask,
                  config_path: readiness?.config_path ?? "configs/generated/playground_series_s6e6.yaml",
                  competition_slug: readiness?.competition_slug ?? "playground-series-s6e6",
                  trigger: "overview_board",
                  objective: "Launch a workstation-controlled validation run"
                })}
                data-testid="create-workstation-run"
                data-ui-skip-action="true"
              >
                <GitBranch className="mb-3 h-5 w-5 text-blue-600" />
                <span className="text-sm font-bold text-slate-900">{ui(locale, "Create Run", "创建 Run")}</span>
                <span className="mt-1 text-xs leading-5 text-slate-500">{ui(locale, "Launch a workstation-controlled validation run", "启动受控的验证运行流程")}</span>
              </button>
              
              <button
                className="flex flex-col items-start rounded-md border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-amber-300 hover:bg-amber-50 focus:outline-none focus:ring-2 focus:ring-amber-500"
                onClick={() => runWorkstationAction?.("prepare_hpc_execution_gate", {
                  task_id: activeRun?.task_id ?? readiness?.task_id ?? normalizedTask,
                  run_id: activeRun?.id,
                  template: "connection_smoke"
                })}
                data-testid="prepare-hpc-execution-gate"
                data-ui-skip-action="true"
              >
                <Cpu className="mb-3 h-5 w-5 text-amber-600" />
                <span className="text-sm font-bold text-slate-900">{ui(locale, "Prepare HPC Gate", "准备 HPC Gate")}</span>
                <span className="mt-1 text-xs leading-5 text-slate-500">{ui(locale, "Request execution authorization on cluster", "申请计算集群的执行授权")}</span>
              </button>
              
              <button
                className="flex flex-col items-start rounded-md border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-purple-300 hover:bg-purple-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
                onClick={() => runWorkstationAction?.("prepare_score_improvement_plan", {
                  competition_slug: readiness?.competition_slug ?? "playground-series-s6e6",
                  task_id: readiness?.task_id ?? normalizedTask
                })}
                data-testid="prepare-score-improvement-plan"
                data-ui-skip-action="true"
              >
                <SlidersHorizontal className="mb-3 h-5 w-5 text-purple-600" />
                <span className="text-sm font-bold text-slate-900">{ui(locale, "Score Plan", "提分计划")}</span>
                <span className="mt-1 text-xs leading-5 text-slate-500">{ui(locale, "Configure score recovery parameters", "配置分数回退的恢复策略")}</span>
              </button>
            </div>

            <div className="mt-4 grid gap-3 border-t border-slate-100 pt-4 sm:grid-cols-2">
              <div className="rounded-md border border-slate-200 bg-white p-3 shadow-sm">
                <div className="mb-1 flex items-center justify-between">
                  <div className="text-xs font-semibold text-slate-500">{ui(locale, "Historical best", "历史最佳")}</div>
                  <div className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">{ui(locale, "Protected baseline", "受保护基线")}</div>
                </div>
                <div className="font-mono text-2xl font-bold text-slate-950">{scoreText(currentBest)}</div>
              </div>
              <div className="rounded-md border border-slate-200 bg-white p-3 shadow-sm">
                <div className="mb-1 flex items-center justify-between">
                  <div className="text-xs font-semibold text-slate-500">{ui(locale, "Latest official score", "最新官方分数")}</div>
                  <div className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">{ui(locale, "Requires review", "需审查")}</div>
                </div>
                <div className="font-mono text-2xl font-bold text-slate-950">{scoreText(latestScore)}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="flex flex-col border-slate-200">
          <CardHeader className="bg-slate-50 border-b border-slate-100 pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>{ui(locale, "System Constraints", "系统约束")}</CardTitle>
                <CardDescription>{ui(locale, "Hard limits on agent actions.", "Agent 行为硬性边界。")}</CardDescription>
              </div>
              <StatusBadge tone={officialSubmitAllowed ? "green" : "amber"}>
                {officialSubmitAllowed ? ui(locale, "Submit ready", "可提交") : ui(locale, "Submit gated", "提交受控")}
              </StatusBadge>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-4">
            <div className="grid gap-2">
              {[
                ui(locale, "Codex only supervises; workstation executes.", "Codex 只监督；工作站负责真实执行。"),
                ui(locale, "Long GPU jobs require explicit approval.", "长 GPU 作业必须获取明确审批。"),
                ui(locale, "Reports must bind claims to metric artifacts.", "报告必须绑定具体的指标与产物。"),
                ui(locale, "Low-score commits do not overwrite baseline.", "低分提交绝对不会覆盖当前基线。")
              ].map((item) => (
                <div key={item} className="flex items-start gap-2 rounded border border-slate-200 bg-slate-50 p-2.5 text-xs font-medium text-slate-700">
                  <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
                  <span className="leading-5">{item}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
        {metrics.map((item) => (
          <MetricCard key={item.label} {...item} />
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.05fr_.95fr]">
        <Card>
          <CardHeader>
            <CardTitle>{ui(locale, "Core research workflow", "核心工作流")}</CardTitle>
            <CardDescription>{ui(locale, "Every stage exposes state; blocked stages stay visibly blocked.", "每个阶段显示真实状态；阻断阶段不会伪装通过。")}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="thin-scrollbar overflow-x-auto pb-1">
              <div className="grid min-w-[820px] grid-cols-8 gap-2">
                {workflow.map(([title, Icon, status], index) => (
                  <div key={title} className="relative">
                    {index < workflow.length - 1 ? <div className="absolute left-[calc(50%+28px)] top-8 h-px w-[calc(100%-40px)] bg-slate-200" /> : null}
                    <div className="relative rounded-md border border-slate-200 bg-white p-3 text-center">
                      <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-md border border-blue-100 bg-blue-50 text-blue-700">
                        <Icon className="h-5 w-5" />
                      </div>
                      <div className="mt-3 truncate text-sm font-bold text-slate-950">{title}</div>
                      <StatusBadge tone={status === "done" ? "green" : status === "manual" ? "amber" : "slate"} className="mt-2">
                        {status === "done" ? ui(locale, "done", "已完成") : status === "manual" ? ui(locale, "gate", "需门禁") : ui(locale, "pending", "待处理")}
                      </StatusBadge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="kaggle-new-competition-readiness-card">
          <CardHeader>
            <CardTitle>{ui(locale, "Kaggle readiness", "Kaggle 接入能力")}</CardTitle>
            <CardDescription>{ui(locale, "Readiness is displayed, not simulated.", "展示真实 readiness，不模拟成功。")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
              <SmallRow label={ui(locale, "Task", "任务")} value={readiness?.task_id ?? normalizedTask} />
              <SmallRow label={ui(locale, "Competition", "比赛")} value={readiness?.competition_slug ?? "playground-series-s6e6"} />
              <SmallRow label={ui(locale, "Metric", "指标")} value={readiness?.metric ?? "balanced_accuracy"} />
              <SmallRow label={ui(locale, "Target", "目标")} value={readiness?.target ?? "class"} />
              <SmallRow label={ui(locale, "Train rows", "训练行数")} value={readiness?.train_rows ?? "pending"} />
              <SmallRow label={ui(locale, "Test rows", "测试行数")} value={readiness?.test_rows ?? "pending"} />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => runWorkstationAction?.("generate_teacher_evidence_bundle", { task_id: activeRun?.task_id ?? readiness?.task_id ?? normalizedTask })}
                data-testid="generate-teacher-evidence-bundle"
                data-ui-skip-action="true"
              >
                <FileCheck2 className="h-4 w-4" />
                {ui(locale, "Evidence Bundle", "证据包")}
              </Button>
              <Button size="sm" variant="secondary" onClick={() => runLocalExperiment?.(readiness?.task_id ?? normalizedTask)} data-testid="run-kaggle-readiness-task" data-ui-skip-action="true">
                <Play className="h-4 w-4" />
                {ui(locale, "Local Score", "本地跑分")}
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>{ui(locale, "Operational workspaces", "核心能力工作区")}</CardTitle>
          <CardDescription>
            {ui(locale, "Overview is only the launch surface; code, reports, agents, evidence and gates remain directly accessible.", "总览只是启动面板；代码、报告、Agent、证据和 Gate 都可直接进入。")}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {workspaceCards.map((item) => (
            <WorkspaceCard key={item.title} {...item} />
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{ui(locale, "Resource readiness", "资源就绪状态")}</CardTitle>
          <CardDescription>{ui(locale, "External compute and APIs remain gated by real connector state.", "外部算力与 API 只依据真实 connector 状态展示。")}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {resources.map((resource) => {
            const Icon = resource.icon;
            const state = String(resource.entry?.state ?? (isConfigured(resource.entry) ? "configured" : "not configured"));
            return (
              <div key={resource.key} className="rounded-md border border-slate-200 bg-white p-3">
                <div className="flex items-start justify-between gap-2">
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-slate-50 text-slate-700">
                    <Icon className="h-4 w-4" />
                  </span>
                  <StatusBadge tone={statusTone(isConfigured(resource.entry), state)}>
                    {statusLabel(locale, isConfigured(resource.entry), state)}
                  </StatusBadge>
                </div>
                <div className="mt-3 text-sm font-bold text-slate-950">{resource.name}</div>
                <div className="mt-1 text-xs font-semibold text-slate-500">{resource.evidence}</div>
                <div className="mt-2 min-h-10 break-words text-xs leading-5 text-slate-600">{state}</div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
