import type { WorkstationSummary } from "@/lib/api/types";

export type TaskSignalSource = "current_run" | "scientist_terminal_turn" | "scientist_context_packet" | "terminal_agent";

export type TaskSignal = {
  taskId: string;
  source: TaskSignalSource;
  observedAt: string | null;
  key: string;
};

const taskIdAliases = new Map<string, string>([
  ["house-prices", "house_prices"],
  ["evomind_qwen7b_finetune", "evomind-qwen7b-finetune"],
]);

export function normalizeTaskId(taskId: string | null | undefined): string {
  const value = String(taskId ?? "").trim();
  return taskIdAliases.get(value) ?? value;
}

export function runtimeForTask(
  summary: WorkstationSummary | null | undefined,
  taskId: string | null | undefined,
): NonNullable<WorkstationSummary["runtime"]> | null {
  const selectedKey = taskComparisonKey(taskId);
  if (!summary || !selectedKey) return null;

  const taskRuntime = Object.entries(summary.runtime_by_task ?? {}).find(([key, runtime]) => {
    const runtimeTaskId = runtime.current_run?.task_id ?? runtime.task_id ?? key;
    return taskComparisonKey(key) === selectedKey || taskComparisonKey(runtimeTaskId) === selectedKey;
  })?.[1];
  if (taskRuntime) return taskRuntime;

  const currentRuntime = summary.runtime;
  const currentTaskId = currentRuntime?.current_run?.task_id ?? currentRuntime?.task_id;
  return currentRuntime && taskComparisonKey(currentTaskId) === selectedKey ? currentRuntime : null;
}

function taskComparisonKey(taskId: string | null | undefined): string {
  return normalizeTaskId(taskId).replace(/-/g, "_");
}

function timestamp(value: string | null | undefined): number {
  if (!value) return Number.NEGATIVE_INFINITY;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

export function latestTaskSignal(summary: WorkstationSummary | null | undefined): TaskSignal | null {
  if (!summary) return null;
  const currentRun = summary.runtime?.current_run;
  const currentRunTaskId = normalizeTaskId(currentRun?.task_id);
  if (currentRunTaskId) {
    return {
      taskId: currentRunTaskId,
      source: "current_run",
      observedAt: currentRun?.updated_at ?? null,
      key: `current_run:${currentRunTaskId}:${currentRun?.run_id ?? "unknown"}`,
    };
  }

  const candidates = [
    {
      source: "scientist_terminal_turn" as const,
      taskId: summary.scientist_terminal_turn?.selected_task,
      observedAt: summary.scientist_terminal_turn?.generated_at,
      priority: 3,
    },
    {
      source: "scientist_context_packet" as const,
      taskId: summary.scientist_context_packet?.selected_task,
      observedAt: summary.scientist_context_packet?.generated_at,
      priority: 2,
    },
    {
      source: "terminal_agent" as const,
      taskId: summary.terminal_agent?.task_id,
      observedAt: summary.terminal_agent?.latest_run_mtime,
      priority: 1,
    },
  ]
    .map((candidate) => ({ ...candidate, taskId: normalizeTaskId(candidate.taskId) }))
    .filter((candidate) => candidate.taskId.length > 0)
    .sort((left, right) => timestamp(right.observedAt) - timestamp(left.observedAt) || right.priority - left.priority);

  const latest = candidates[0];
  if (!latest) return null;
  return {
    taskId: latest.taskId,
    source: latest.source,
    observedAt: latest.observedAt ?? null,
    key: `${latest.source}:${latest.taskId}:${latest.observedAt ?? "undated"}`,
  };
}

export function recordMatchesTask(record: unknown, selectedTask: string | null | undefined): boolean {
  const taskId = taskComparisonKey(selectedTask);
  if (!taskId) return true;
  if (!record || typeof record !== "object") return false;
  const candidate = record as Record<string, unknown>;
  return taskComparisonKey(String(candidate.task_id ?? candidate.taskId ?? candidate.id ?? "")) === taskId;
}
