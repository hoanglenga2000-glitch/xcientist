"use client";

import { useEffect, useState } from "react";
import { Activity, AlertCircle, Clock3 } from "lucide-react";
import { runProgressView } from "./run-progress-presentation";
import type { RunProgress } from "./run-progress-presentation";
export type { RunProgress } from "./run-progress-presentation";

const labels: Record<string, [string, string]> = {
  preparing: ["准备资源", "Preparing resources"], downloading: ["下载模型", "Downloading model"],
  installing: ["安装依赖", "Installing dependencies"], training: ["训练模型", "Training model"],
  inference: ["运行推理", "Running inference"], evaluating: ["评测结果", "Evaluating results"],
  executing: ["执行任务，阶段待确认", "Executing; phase unverified"], waiting_approval: ["等待确认", "Waiting for approval"],
  ended: ["本次执行已结束", "Execution ended"], stale: ["状态待确认", "Awaiting status"],
  model_wait: ["模型服务处理中", "Waiting for model service"],
  paused: ["任务已暂停", "Task paused"], pausing: ["正在安全暂停", "Pausing at a safe boundary"],
};

const details: Record<string, [string, string]> = {
  completed: ["本次执行已结束。模型与研究结论是否有效，请以独立验证记录为准。", "Execution ended. Use independent verification records to assess model and research claims."],
  failed: ["本次执行失败，现场已保留。请查看错误详情后再决定是否重试。", "Execution failed; its state is preserved. Inspect the error before retrying."],
  blocked: ["任务需要处理，已有结果已保留。可在同一任务中补充信息后继续。", "The task needs attention; existing results are preserved. Continue in the same Run."],
  cancelled: ["任务已取消，已经产生的结果仍会保留。", "The task was cancelled; existing results remain available."],
  waiting_approval: ["请核对具体操作与参数后确认，确认后将继续当前任务。", "Review the exact action and parameters before confirming continuation of this Run."],
  stale: ["尚未收到新的执行证据。页面保持运行状态不代表模型正在训练。", "No fresh execution evidence. A running page does not prove model training."],
  model_wait: ["正在等待模型服务返回；这一步不是 GPU 模型训练。", "Waiting for the model service response; this step is not GPU model fitting."],
  waiting_for_compute: ["正在等待受管算力队列分配执行位置。", "Waiting for an execution slot in the managed compute queue."],
  paused: ["已停止派发新工作。现有结果与预算记录保留，可在同一任务中恢复。", "New dispatch is paused. Existing results and accounting are preserved for same-Run resume."],
  pausing: ["已停止派发新工作，正在等待已启动的操作结算；不会强制终止远端进程。", "New dispatch is blocked while existing operations settle; remote processes are not force-stopped."],
  generic: ["当前阶段按执行器记录展示，原始说明可在技术详情中查看。", "The stage reflects executor records. Expand technical details for the original description."],
};

export function RunProgressPanel({ progress, status, locale }: { progress?: RunProgress; status: string; locale: "zh-CN" | "en-US" }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 15_000); return () => window.clearInterval(timer); }, []);
  const zh = locale === "zh-CN";
  const view = runProgressView(progress, status, now);
  const { terminal, stale, percent } = view;
  const title = (labels[view.kind] ?? labels.executing)[zh ? 0 : 1];
  const detail = (details[view.detailKey] ?? details.generic)[zh ? 0 : 1];
  return <section className="mt-3 space-y-2 rounded-lg border border-edge bg-surface-sunken/50 p-3" aria-label={zh ? "真实执行进度" : "Execution progress"} data-ui-execution-progress>
    <div className="flex items-center gap-2 text-xs font-semibold text-ink">{stale ? <AlertCircle className="h-4 w-4 text-warning" /> : <Activity className="h-4 w-4 text-accent" />}{title}</div>
    {percent !== null ? <div><div className="mb-1 flex justify-between text-[11px] tabular-nums text-ink-muted"><span>{progress?.completed_units?.toLocaleString()} / {progress?.total_units?.toLocaleString()} {progress?.unit}</span><span>{percent.toFixed(1)}%</span></div><progress aria-label={title} max={100} value={percent} className="h-1.5 w-full accent-accent" /></div> : null}
    <p className="text-[11px] leading-5 text-ink-muted">{view.stalled ? (zh ? "执行器仍在线，但近期没有新增进展。" : "The executor is online, but there has been no recent progress. ") : ""}{detail}</p>
    {view.rawDiagnostic ? <details className="text-[11px] text-ink-muted"><summary className="cursor-pointer">{zh ? "技术详情（最近一条执行记录）" : "Technical details (latest execution record)"}</summary><p className="mt-1 break-words whitespace-pre-wrap">{view.rawDiagnostic}</p></details> : null}
    {progress?.heartbeat_at ? <div className="flex items-center gap-1 text-[10px] text-ink-muted"><Clock3 className="h-3 w-3" />{zh ? "最近心跳" : "Last heartbeat"} {new Date(progress.heartbeat_at * 1000).toLocaleTimeString(locale)}</div> : null}
    {terminal && progress?.objective_outcome !== "verified" ? <p className="text-[11px] text-warning-text">{zh ? "执行结束不代表研究目标已通过验收。" : "Execution ending does not establish research acceptance."}</p> : null}
  </section>;
}
