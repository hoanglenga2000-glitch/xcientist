// t238 C7c-6: mock /api/assistant/gpu/me payloads for the shots script (page.route) and component tests.
import type { GpuStatusPayload } from './gpu-status';

const budget = { remaining_minutes: 75, daily_limit_minutes: 120, daily_used_minutes: 45, max_run_minutes: 30 };
const one = (runId: string, job: GpuStatusPayload['jobs'][number]): GpuStatusPayload => ({ schema: 'evomind.gpu_tenant.v1', enabled: true, available: true, jobs: [{ ...job, run_id: runId }], budget });

export const GPU_SHOT_FIXTURES = {
  queued: (runId: string) => one(runId, { run_id: runId, state: 'queued', position: 2, eta_minutes: 14, eta_is_estimate: true, can_cancel: true, label: '排队中（第 2 位，预计等待约 14 分钟）', badge: '排队中' }),
  queuedVram: (runId: string) => one(runId, { run_id: runId, state: 'queued', position: 1, eta_minutes: 1, eta_is_estimate: true, can_cancel: true, label: '排队中（第 1 位，预计等待约 1 分钟）', notice: '排队中，轮到你就开始。', badge: '排队中' }),
  breaker: (runId: string) => ({ ...one(runId, { run_id: runId, state: 'queued', position: 1, can_cancel: true, label: '训练暂时用不了', paused: true, notice: '你的排队位置还在，恢复后在这个任务里重新发起就行。', badge: '训练暂时用不了' }), available: false }),
  held: (runId: string) => one(runId, { run_id: runId, state: 'queued', position: 1, can_cancel: true, label: '需要你重新发起', notice: '你的排队位置还在，但没有任务在等它。在这个任务里重新发起就行。', retry_label: '重新发起', badge: '需要你重新发起' }),
  preparing: (runId: string) => one(runId, { run_id: runId, state: 'running', elapsed_minutes: 0, preparing: true, can_cancel: true, label: '正在准备训练环境', notice: '第一次使用要多花一点时间。你可以先离开这个页面，回来就能看到最新状态。准备期间不占用你的训练时长。', badge: '准备中' }),
  preparingSlow: (runId: string) => one(runId, { run_id: runId, state: 'running', elapsed_minutes: 0, preparing: true, can_cancel: true, label: '正在准备训练环境', notice: '这次比平时慢，我们还在继续。准备期间不占用你的训练时长。', badge: '准备中' }),
  prepFailed: (runId: string) => one(runId, { run_id: runId, state: 'failed', can_cancel: false, label: '环境没有准备好', notice: '已经准备好的部分会保留，重试会接着来，不用从头开始。这次没有开始训练，没有占用你的训练时长。', retry_label: '重新准备', badge: '环境没有准备好' }),
  failed: (runId: string) => one(runId, { run_id: runId, state: 'failed', can_cancel: false, label: '没有完成', badge: '没有完成' }),
  atCap: (runId: string) => one(runId, { run_id: runId, state: 'running', elapsed_minutes: 30, max_minutes: 30, can_cancel: true, label: '训练中（已运行 30 分，最长 30 分）', badge: '训练中' }),
  running: (runId: string) => one(runId, { run_id: runId, state: 'running', elapsed_minutes: 7, max_minutes: 30, can_cancel: true, label: '训练中（已运行 7 分，最长 30 分）', badge: '训练中' }),
  succeeded: (runId: string) => one(runId, { run_id: runId, state: 'succeeded', can_cancel: false, label: '已完成', badge: '已完成' }),
  cancelled: (runId: string) => one(runId, { run_id: runId, state: 'cancelled', can_cancel: false, label: '已取消', notice: '没有占用你的训练时长。', badge: '已取消' }),
  timeout: (runId: string) => one(runId, { run_id: runId, state: 'timeout', can_cancel: false, label: '到时间了，已停止', notice: '这次训练超过了最长时间，已自动停止。可以缩小数据后重新开始。', badge: '到时间了，已停止' }),
  unavailable: (runId: string) => ({ ...one(runId, { run_id: runId, state: 'unavailable', can_cancel: false, label: '今天的训练时长已用完', notice: '你今天的训练时长已用完，明天可以继续训练。', badge: '今天的训练时长已用完' }), budget: { remaining_minutes: 0, daily_limit_minutes: 120, daily_used_minutes: 120 } }),
  freeing: (runId: string) => one(runId, { run_id: runId, state: 'queued', position: 1, eta_minutes: 3, eta_is_estimate: true, can_cancel: true, label: '正在腾出位置，预计 3 分钟内开始。', badge: '排队中' }),
  etaUnknown: (runId: string) => one(runId, { run_id: runId, state: 'queued', position: 2, can_cancel: true, label: '排队中（第 2 位）', badge: '排队中' }),
  nearCap: (runId: string) => one(runId, { run_id: runId, state: 'running', elapsed_minutes: 28, max_minutes: 30, can_cancel: true, label: '训练中（已运行 28 分，最长 30 分）', badge: '训练中' }),
} as const;
