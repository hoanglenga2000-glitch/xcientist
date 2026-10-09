export type RunProgress = {
  schema?: string; work_kind?: string; phase?: string; freshness?: string; stalled?: boolean;
  heartbeat_at?: number | null; last_progress_at?: number | null;
  completed_units?: number | null; total_units?: number | null; unit?: string | null;
  percent?: number | null; detail?: string; source?: string; objective_outcome?: string; execution_terminal?: boolean;
};

export function runProgressView(progress: RunProgress | undefined, status: string, now: number) {
  const terminal = ["completed", "failed", "blocked", "cancelled"].includes(status);
  const waiting = status === "waiting_approval";
  const paused = status === "paused";
  const pausing = status === "pausing";
  const heartbeat = progress?.heartbeat_at;
  const age = typeof heartbeat === "number" && Number.isFinite(heartbeat) && heartbeat > 0 ? now / 1000 - heartbeat : null;
  const stale = !terminal && !waiting && !paused && (age === null || age > 45 || age < -45);
  const governed = progress?.source === "managed_adapter";
  const observed = governed || progress?.source === "executor";
  const modelWait = observed && progress?.phase === "research_model_call";
  const kind = terminal ? "ended" : paused ? "paused" : pausing ? "pausing" : waiting ? "waiting_approval" : stale ? "stale"
    : modelWait ? "model_wait" : governed ? progress?.work_kind ?? "executing" : "executing";
  const total = progress?.total_units, completed = progress?.completed_units;
  const measured = governed && !terminal && !waiting && !paused && !stale
    && typeof total === "number" && Number.isFinite(total) && total > 0
    && typeof completed === "number" && Number.isFinite(completed) && completed >= 0 && completed <= total;
  return {
    kind, terminal, stale,
    percent: measured ? completed / total * 100 : null,
    detailKey: terminal ? status : paused ? "paused" : pausing ? "pausing" : waiting ? "waiting_approval" : stale ? "stale" : modelWait ? "model_wait"
      : observed && progress?.phase === "waiting_for_compute" ? "waiting_for_compute" : "generic",
    stalled: !terminal && !waiting && !paused && !stale && progress?.stalled === true,
    rawDiagnostic: progress?.detail?.trim() || "",
  };
}
