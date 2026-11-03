// t238 C7c-6 (U2/U3): tenant GPU training state for one run. Pure copy + decision helpers (node-tested).
// The runtime returns only this tenant's jobs and no node / job id / host / card details.
export type GpuJobState = 'queued' | 'running' | 'succeeded' | 'cancelled' | 'timeout' | 'unavailable' | 'failed';
export type GpuJob = {
  run_id: string; state: GpuJobState; label?: string; notice?: string; can_cancel?: boolean;
  position?: number | null; eta_minutes?: number; eta_is_estimate?: boolean; elapsed_minutes?: number; max_minutes?: number; preparing?: boolean; retry_label?: string; paused?: boolean; badge?: string; billed_minutes?: number;
};
export type GpuBudget = { remaining_minutes: number; daily_limit_minutes: number; daily_used_minutes: number; max_run_minutes?: number };
export type GpuStatusPayload = { schema?: string; enabled: boolean; available?: boolean; jobs: GpuJob[]; budget: GpuBudget | null };

export const GPU_COPY = {
  heading: '训练状态',
  cancelQueued: '取消排队',
  giveUpHeld: '不训练了',
  confirmHeldTitle: '不再训练？',
  confirmHeldBody: '这次训练不会再自动开始。你之后随时可以重新发起。',
  confirmHeldYes: '不训练了',
  confirmHeldNo: '先留着',
  cancelRunning: '停止训练',
  cancelling: '正在取消…',
  cancelPreparing: '取消',
  confirmPreparingTitle: '取消准备？',
  confirmPreparingBody: '准备会停止，不占用你的训练时长，之后可以重新发起。',
  confirmPreparingYes: '取消准备',
  confirmPreparingNo: '继续准备',
  atCap: '已到时间上限，正在停止…',
  retryFailed: '没能重新发起，请刷新后再试。',
  confirmQueuedTitle: '取消排队？',
  confirmRunningTitle: '停止这次训练？',
  confirmQueuedBody: '取消后会失去现在的位置，要重新排队。',
  confirmRunningBody: '训练会立即停止，已用的时间不会退回。',
  confirmQueuedYes: '取消排队',
  confirmRunningYes: '停止训练',
  confirmQueuedNo: '继续等待',
  confirmRunningNo: '继续训练',
  budget: (remaining: number, limit: number, perRun?: number) => `今天还可以训练 ${remaining} 分钟（每天 ${limit} 分钟）${perRun && perRun > 0 ? `，每次最长 ${perRun} 分钟` : ''}`,
  nearCap: (max: number) => `快到 ${max} 分钟了，到点会自动停止。`,
  estimate: '等待时间是估计值，会随前面的训练变化。',
  cancelFailed: '没能取消，请刷新后再试。',
  refresh: '刷新',
} as const;

/** U3: error code → one short sentence + the next action (no codes in the UI). */
export const GPU_ERROR_COPY: Record<string, { line: string; action: 'retry_later' | 'wait' | 'reduce' | 'tomorrow' | 'none' }> = {
  breaker_open: { line: '训练暂时用不了，你的排队位置还在。恢复后在这个任务里重新发起就行。', action: 'retry_later' },
  pool_unavailable: { line: '训练暂时不可用，请稍后再试。', action: 'retry_later' },
  quota_exhausted: { line: '你今天的训练时长已用完，明天可以继续训练。', action: 'tomorrow' },
  quota_exhausted_rolling: { line: '你最近一段时间的训练时长已用完，过几天可以继续训练。', action: 'retry_later' },
  queue_full: { line: '现在排队的训练太多，请等前面的训练结束后再提交。', action: 'wait' },
  insufficient_gpu_memory: { line: '现在训练排满了，这次训练还没有开始。请稍后在这个任务里重新发起。', action: 'wait' },
  timeout: { line: '这次训练超过了最长时间，已自动停止。可以缩小数据后重新开始。', action: 'reduce' },
};

export function gpuErrorLine(code: string): string {
  return (GPU_ERROR_COPY[code] ?? GPU_ERROR_COPY.pool_unavailable).line;
}

/** The line shown for a job; the runtime label is preferred, a local fallback never shows codes. */
export function gpuJobLine(job: GpuJob): string {
  if (job.label) return job.label;
  switch (job.state) {
    case 'queued': {
      const position = Math.max(1, job.position ?? 1);
      return job.eta_minutes == null ? `排队中（第 ${position} 位）` : `排队中（第 ${position} 位，预计等待约 ${Math.max(1, job.eta_minutes)} 分钟）`;
    }
    case 'running': return job.preparing ? '正在准备训练环境' : `训练中（已运行 ${Math.max(0, job.elapsed_minutes ?? 0)} 分，最长 ${Math.max(0, job.max_minutes ?? 0)} 分）`;
    case 'succeeded': return '已完成';
    case 'cancelled': return '已取消';
    case 'timeout': return '到时间了，已停止';
    case 'unavailable': return '暂不可用';
    default: return '没有完成';
  }
}

/** True when a running job has used 80% or more of its time limit (and still has time left). */
export const isNearCap = (job: GpuJob) => job.state === 'running' && !job.preparing && Number(job.max_minutes) > 0 && Number(job.elapsed_minutes) >= Number(job.max_minutes) * 0.8 && Number(job.elapsed_minutes) < Number(job.max_minutes);

/** True when a running job has reached its time limit: it is being stopped, not "nearly" stopped. */
export const isAtCap = (job: GpuJob) => job.state === 'running' && !job.preparing && Number(job.max_minutes) > 0 && Number(job.elapsed_minutes) >= Number(job.max_minutes);

/** The short status word for the task title chip. ONE source: the `badge` the server derives from the same job as the card (live and terminal). The local fallback only serves a payload without it. C7c-FIX5c C6: terminal states also read `badge` (was null → fell back to runStatusWord and mismatched failed/lost/timeout/reject cards). */
export function gpuBadge(job: GpuJob | null | undefined): string | null {
  if (!job) return null;
  if (job.badge) return job.badge;
  if (job.state === 'running' && job.label === GPU_COPY.cancelling) return '正在取消';
  if (job.state === 'queued') return job.retry_label ? '需要你重新发起' : job.paused ? '训练暂时用不了' : '排队中'; // a held place or a paused card keeps state=queued, but its title chip must agree with its card
  if (job.state === 'running') return job.preparing ? '准备中' : '训练中';
  // terminal fallback when an older payload omitted badge: agree with the card label (trim the ellipsis)
  if (job.label) return job.label.replace(/…$/, '').trim() || null;
  return null;
}

export const isLiveGpuJob = (job: GpuJob | null | undefined) => Boolean(job && (job.state === 'queued' || job.state === 'running'));

/** The newest job of this run, or null when the scheduler is off / the run never asked for the card. */
export function gpuJobForRun(payload: GpuStatusPayload | null, runId: string): GpuJob | null {
  if (!payload?.enabled) return null;
  return payload.jobs.find(job => job.run_id === runId) ?? null;
}

export const gpuStatusUrl = (runId: string) => `/api/assistant/gpu/me?run_id=${encodeURIComponent(runId)}`;
export const gpuCancelUrl = (runId: string) => `/api/assistant/gpu/runs/${encodeURIComponent(runId)}/cancel`;
