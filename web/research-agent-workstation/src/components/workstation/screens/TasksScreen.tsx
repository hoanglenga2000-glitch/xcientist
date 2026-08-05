"use client";

import {
  Activity,
  CheckCircle2,
  FileText,
  ListTodo,
  Lock,
  Play,
  RefreshCw,
  Send,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile } from "../primitives/Layout";
import { StatusBadgeV2, type StatusTone } from "../primitives/StatusBadge";
import { t } from "../localization";

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
    action: string; taskId?: string; request?: Record<string, unknown>;
    response?: Record<string, unknown>; message: string; artifact?: string | null; at: string;
  } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

function taskStatusTone(status: string): StatusTone {
  const s = status.toLowerCase();
  if (s.includes("complete") || s.includes("done")) return "verified";
  if (s.includes("running") || s.includes("active")) return "running";
  if (s.includes("blocked") || s.includes("locked")) return "blocked";
  if (s.includes("ready") || s.includes("queued")) return "ready";
  if (s.includes("pending")) return "pending";
  if (s.includes("failed")) return "failed";
  return "unknown";
}

export function TasksScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", setSelectedTask } = props;
  const tasks = (summary?.tasks ?? []).filter((task) => task.status !== "archived");
  const ready = tasks.filter((t) => ["ready", "queued", "pending"].includes(taskStatusTone(t.status))).length;
  const blocked = tasks.filter((t) => taskStatusTone(t.status) === "blocked").length;
  const completed = tasks.filter((t) => taskStatusTone(t.status) === "verified").length;

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Task Queue", "任务队列")}
        subtitle={t(locale, "All tasks in the research loop", "研究闭环中的全部任务")}
        breadcrumb={`${t(locale, "Research Loop", "研究循环")} > ${t(locale, "Task Queue", "任务队列")}`}
        primaryAction={
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              data-ui-action="tasks_refresh_queue"
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <RefreshCw className="h-3 w-3" /> {t(locale, "Refresh", "刷新")}
            </button>
            <button
              type="button"
              data-ui-action="tasks_create_workstation_run"
              disabled={!props.selectedTask}
              className="flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
            >
              <Play className="h-3 w-3" /> {t(locale, "Create Run", "创建运行")}
            </button>
            <button
              type="button"
              data-ui-action="tasks_dispatch_agents"
              disabled={!props.selectedTask}
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
            >
              <Send className="h-3 w-3" /> {t(locale, "Dispatch", "调度")}
            </button>
            <button
              type="button"
              data-ui-action="tasks_open_context"
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <FileText className="h-3 w-3" /> {t(locale, "Context", "上下文")}
            </button>
          </div>
        }
      />

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label={t(locale, "Total Tasks", "总任务")} value={tasks.length} icon={ListTodo} tone="blue" />
        <MetricTile label={t(locale, "Ready", "就绪")} value={ready} icon={Activity} tone="green" />
        <MetricTile label={t(locale, "Blocked", "阻塞")} value={blocked} icon={Lock} tone={blocked > 0 ? "red" : "neutral"} />
        <MetricTile label={t(locale, "Completed", "已完成")} value={completed} icon={CheckCircle2} tone="green" />
      </div>

      {/* Task table */}
      <Panel title={t(locale, "Tasks", "任务列表")}>
        {tasks.length === 0 ? (
          <div className="py-6 text-center text-sm text-ink-muted">{t(locale, "No tasks found", "无任务")}</div>
        ) : (
          <>
            {/* Desktop table */}
            <div className="hidden sm:block overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-edge text-left text-ink-muted">
                    <th className="pb-2 pr-3 font-medium">{t(locale, "Name", "名称")}</th>
                    <th className="pb-2 pr-3 font-medium">{t(locale, "Type", "类型")}</th>
                    <th className="pb-2 pr-3 font-medium">{t(locale, "Status", "状态")}</th>
                    <th className="pb-2 pr-3 font-medium">{t(locale, "Metric", "指标")}</th>
                    <th className="pb-2 font-medium">{t(locale, "Priority", "优先级")}</th>
                  </tr>
                </thead>
                <tbody>
                  {tasks.map((task) => (
                    <tr
                      key={task.id}
                      data-ui-action={`tasks_select_${task.id}`}
                      data-selected-task={props.selectedTask === task.id ? "true" : "false"}
                      aria-selected={props.selectedTask === task.id}
                      className={cn(
                        "border-b border-edge/50 hover:bg-surface-sunken cursor-pointer transition-colors",
                        props.selectedTask === task.id && "bg-accent-light/50"
                      )}
                      onClick={() => setSelectedTask(task.id)}
                    >
                      <td className="py-2 pr-3 font-medium text-ink">{task.name}</td>
                      <td className="py-2 pr-3 text-ink-secondary">{task.task_type}</td>
                      <td className="py-2 pr-3">
                        <StatusBadgeV2 tone={taskStatusTone(task.status)} size="xs">{task.status}</StatusBadgeV2>
                      </td>
                      <td className="py-2 pr-3 font-mono text-ink-secondary">{task.metric ?? "—"}</td>
                      <td className="py-2 text-ink-secondary">{task.priority ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Mobile cards */}
            <div className="space-y-2 sm:hidden">
              {tasks.map((task) => (
                <button
                  key={task.id}
                  type="button"
                  data-ui-action={`tasks_select_${task.id}`}
                  data-selected-task={props.selectedTask === task.id ? "true" : "false"}
                  aria-pressed={props.selectedTask === task.id}
                  className={cn(
                    "w-full rounded-md border border-edge p-3 text-left hover:bg-surface-sunken",
                    props.selectedTask === task.id && "bg-accent-light/50 border-accent/40"
                  )}
                  onClick={() => setSelectedTask(task.id)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-ink">{task.name}</span>
                    <StatusBadgeV2 tone={taskStatusTone(task.status)} size="xs">{task.status}</StatusBadgeV2>
                  </div>
                  <div className="mt-1 flex gap-3 text-2xs text-ink-muted">
                    <span>{task.task_type}</span>
                    <span>{task.metric ?? "—"}</span>
                    <span>{task.priority ?? "—"}</span>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </Panel>
    </div>
  );
}
