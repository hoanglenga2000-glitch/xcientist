'use client';
import { useState } from 'react';
import { restoreRunMessages } from '../screens/assistant-history';
import { answerStarted, confirmPending, attachmentChips, deliveryMode, filesSummaryLine, LIVE_RUN_STATUSES, stuckSecondsLabel, stuckSurface, isStuckBanner, thinkingRowState, settledStepStatus, trainingFailed, TRAINING_FAILED_STATUS, stoppedByUser, runStatusWord, cancelledTrainingDelivery, resultArtifacts, CANCELLED_RESULTS_EMPTY } from './task-presentation';
import { blockedAssistantLine, blockedProgressCopy, INTERNAL_RUN_ERROR_LINE, isConfigModelFailure, isModelFailure, modelFailureCode, servedModelLabel, isProcessLeak, streamPhaseState, toolStepDetail, toolStepLabel, visibleMessageText } from './task-presentation';
import { taskStatus, type TaskRun } from './types';
import { ERROR_COPY } from './error-copy';
import { explicitModelErrorLine, failoverNotice, SWITCH_TO_FALLBACK } from './model-failure';
import { MarkdownMessage } from './MarkdownMessage';
import { MessageActions } from './MessageActions';
import { MemoryChip, MemoryUsed } from './MemorySettings';
import { ProjectFilesNotice, projectFilesNotice } from './ProjectFilesNotice';
import { ExperiencePlan } from './ExperiencePlan';
import { DeepEvoMark } from '../DeepEvoBrand';
import { AttachmentChip, FileSources } from './AttachmentChip';
import { ThinkingRow } from './ThinkingRow';
import { GpuJobStatus } from './GpuJobStatus';
import type { GpuJob } from './gpu-status';

const LIVE_PROCESS = new Set(LIVE_RUN_STATUSES);

/** C6-brand H_b7: small-cut mark (20px ⇒ 5 seams) in brand ink. C6-answer: the loader lives in the ThinkingRow, so one mark spins. */
function Speaker() {
  return <div className="tw-speaker"><span className="tw-avatar" aria-hidden="true"><DeepEvoMark size={20} className="tw-avatar-mark" /></span><small>DeepEvo</small></div>;
}

/** User text stays plain except `inline code`, styled like the assistant's; fences stay as typed. */
function UserText({ text }: { text: string }) {
  return text.split(/(`[^`\n]+`)/).map((part, index) => index % 2 ? <code key={index}>{part.slice(1, -1)}</code> : part);
}

/** C6b H5-4 / H5-7: Chinese reason + 「换个模型继续」+「再试一次」; tech codes only under 技术详情. No English exception names outside details. */
// t238 C7e A2: an explicit pick with a classified failure gets its own line and, when a listed alternate exists,
// 「换到备选模型」 (the same question again on that model, never an automatic switch).
function ModelFailCard({ run, onResume, fallback }: { run: TaskRun; onResume?: () => Promise<void> | void; fallback?: { label: string; onSwitch: () => Promise<void> | void } | null }) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const config = isConfigModelFailure(run);
  const code = modelFailureCode(run) || (/^[A-Za-z0-9_]{1,80}$/.test(run.error_class || '') ? run.error_class : '') || 'unknown';
  const line = explicitModelErrorLine(run) ?? (config ? ERROR_COPY.model_identity_unconfirmed.line : INTERNAL_RUN_ERROR_LINE);
  const modelsHref = `/workspace?settings=models${run.user_task_id ? `&return_task=${encodeURIComponent(run.user_task_id)}` : ''}`;
  async function retry() {
    if (busy) return;
    if (!onResume) { window.location.reload(); return; }
    setBusy(true); setFailed(false);
    try { await onResume(); } catch { setFailed(true); } finally { setBusy(false); }
  }
  async function switchModel() {
    if (busy || !fallback) return;
    setBusy(true); setFailed(false);
    try { await fallback.onSwitch(); } catch { setFailed(true); } finally { setBusy(false); }
  }
  return <div className="tw-error tw-model-error" role="alert" data-testid="tw-model-fail">
    {config ? <h2>模型暂时不可用</h2> : null}
    <p>{line}</p>
    <div className="tw-fail-actions">
      {fallback && fallback.label && fallback.label !== '备选模型' ? <button type="button" className="tw-primary" disabled={busy} data-testid="tw-switch-fallback" onClick={() => void switchModel()}>{SWITCH_TO_FALLBACK}「{fallback.label}」</button>
        : <a className="tw-primary" href={modelsHref}>换个模型继续</a>}
      <button type="button" className="tw-text-button" disabled={busy} onClick={() => void retry()}>再试一次</button>
    </div>
    {failed && <p className="tw-muted">没能完成，请稍后再点一次。</p>}
    <details className="tw-details"><summary>技术详情</summary><p>{code}</p>{config && (run.error_message || run.answer) ? <p>{run.error_message || run.answer}</p> : null}</details>
  </div>;
}

/** C6b H5-1..H5-3: amber stuck banner only when freshness is stale; actions 停止 / 再试一次. */
function StuckBanner({ run, onStop, onRetry }: { run: TaskRun; onStop?: () => void; onRetry?: () => void }) {
  if (!isStuckBanner(run.progress, run.status)) return null;
  const age = run.progress?.heartbeat_age_seconds;
  return <div className="tw-stuck-banner" role="status" data-testid="tw-stuck-banner">
    <span>{stuckSecondsLabel(age)}</span>
    <span className="tw-stuck-actions">
      {onStop && <button type="button" className="tw-text-button" onClick={onStop}>停止</button>}
      {onRetry && <button type="button" className="tw-text-button" onClick={onRetry}>再试一次</button>}
    </span>
  </div>;
}

export function TaskRunView({ view, run, memoryEnabled = true, projectId, onRetry, onResume, onStop, onRetryTraining, onGpuJob, gpuJob = null, fallback }: { gpuJob?: GpuJob | null; view: string; run: TaskRun | null; title?: string; taskState?: string; memoryEnabled?: boolean; projectId?: string | null; onRetry?: () => void; onResume?: () => Promise<void> | void; onStop?: () => void; onRetryTraining?: () => Promise<void> | void; onGpuJob?: (job: GpuJob | null) => void; fallback?: { label: string; onSwitch: () => Promise<void> | void } | null }) {
  // Empty tabs stay blank; the page eyebrow already names the status.
  if (!run) return null;
  const unavailable = isModelFailure(run);
  // C6b H5-4: 「换个模型继续」primary + 「再试一次」secondary; tech under 技术详情. Never auto-switch models.
  const modelHelp = <ModelFailCard run={run} onResume={onResume} fallback={fallback} />;
  if (view === 'results') {
    // C7c-FIX5c C2 CLEARING: cancelled training must not look like a successful file delivery.
    const cleared = cancelledTrainingDelivery(run, gpuJob);
    const files = resultArtifacts(run, gpuJob);
    const evidence = cleared ? [] : (run.artifact_evidence ?? []);
    return <section data-testid="tw-results">
      {files.map(file => <AttachmentChip key={file.id} file={file} />)}
      {!files.length && <div className="tw-empty"><h2>{cleared ? CANCELLED_RESULTS_EMPTY : '还没有结果文件'}</h2></div>}
      <FileSources files={files} />
      {Boolean(evidence.length) && <details className="tw-details"><summary>中间文件与执行记录（{evidence.length}）</summary>{evidence.map(file => <AttachmentChip key={file.id} file={file} />)}</details>}
    </section>;
  }
  const progressCopy = unavailable ? null : blockedProgressCopy(run);
  if (view === 'progress') return <section><h2>{trainingFailed(run) ? TRAINING_FAILED_STATUS : stoppedByUser(run) ? runStatusWord(run, taskStatus) : progressCopy?.status || taskStatus(run.status)}</h2>
    {unavailable ? modelHelp : progressCopy && <p role="alert" className="tw-error">{progressCopy.action}</p>}
    <GpuJobStatus runId={run.id} live={LIVE_PROCESS.has(run.status)} hideSettled={confirmPending(run)} onRetry={onRetryTraining} onJob={onGpuJob} />
    <ol className="tw-timeline">{(run.plan?.steps ?? []).map(step => {
      const detail = toolStepDetail(step.detail);
      const shownStatus = settledStepStatus(run, step);
      const statusLabel = ({ pending: '尚未执行', running: '正在处理', completed: '已完成', failed: '失败', skipped: '已跳过' } as Record<string, string>)[shownStatus] ?? toolStepLabel(shownStatus);
      return <li key={step.id}><div><h3>{toolStepLabel(step.label, step.status)}</h3>{detail && <p>{detail}</p>}</div><span>{statusLabel}</span></li>;
    })}</ol>
    {!run.plan?.steps?.length && <p className="tw-muted">当前运行尚未提供步骤记录，不推测正在执行的阶段。</p>}
    <details className="tw-details"><summary>运行信息</summary><dl><dt>实际模型</dt><dd>{run.model ? (servedModelLabel(run.model_provider, run.model) || '平台模型') : '尚无记录'}</dd></dl></details>
  </section>;
  const served = servedModelLabel(run.model_provider, run.model);
  const servedId = [run.model_provider, run.model].filter(Boolean).join('/');
  const phase = streamPhaseState(run);
  const surface = stuckSurface(run);
  // C6-brand H_b12/H_b13: while the run is live its answer bubble exists from the first moment (ThinkingRow) and the
  // stream grows inside it; once settled an empty answer leaves no bubble at all.
  const live = !unavailable && LIVE_PROCESS.has(run.status);
  const answerId = run.message_history?.schema === 'evomind.assistant_message_history.v1' ? run.message_history.active_message_id : `assistant_${run.id}`;
  const restored = restoreRunMessages(run, '继续执行此任务');
  if (live && !restored.some(message => message.id === answerId && message.role === 'assistant')) restored.push({ id: answerId, role: 'assistant', content: '', status: 'streaming', runId: run.id });
  const messages = restored
    .map(message => live && message.id === answerId && message.role === 'assistant' ? { ...message, content: run.streaming_text || message.content } : message)
    .filter(message => (message.content || (live && message.id === answerId)) && !(unavailable && message.role === 'assistant' && message.content.includes('Model execution is unavailable:')))
    .map(message => ({ ...message, content: visibleMessageText(message.content) }));
  // C6-answer H_a8: the turn's last assistant message is the answer — never filed away as process chatter.
  const lastReply = messages.filter(message => message.role === 'assistant' && message.content).at(-1)?.id;
  const leaked = (message: { id: string; role: string; content: string }) => message.role === 'assistant'
    && isProcessLeak(message.content, { last: message.id === lastReply || message.id === answerId });
  const process = messages.filter(leaked).map(message => message.content);
  const shown = messages.filter(message => !leaked(message));
  // SPEC §5.6 / H_a10–H_a12: deliverables are attachments under the answer, only in a file mode. An answer-less file run
  // still gets one inline line above its chips.
  // C7c-FIX5i / FIX5c C2 CLEARING: cancelled training never presents outputs as success (chat + results + 「生成文件（N）」).
  const files = resultArtifacts(run, gpuJob);
  const chips = attachmentChips(deliveryMode(run), files);
  if (!live && files.length && !shown.some(message => message.role === 'assistant' && message.content && message.id === answerId)) {
    shown.push({ id: answerId, role: 'assistant', content: filesSummaryLine(files.map(file => file.name)), status: 'complete', runId: run.id });
  }
  const answerShown = shown.some(message => message.role === 'assistant' && message.id === answerId);
  const filesNotice = projectFilesNotice(run);
  // 已记住 sits under this run's last answer; off or nothing saved ⇒ no chip.
  const chipAt = memoryEnabled && run.memory_saved?.length ? shown.filter(message => message.role === 'assistant' && message.content).at(-1)?.id : undefined;
  const latestAssistant = shown.filter(message => message.role === 'assistant' && message.content).at(-1)?.id;
  // C6c H7-5: 用到的记忆 under the same last answer; 无痕 / 不使用记忆 / off / nothing injected ⇒ no block.
  const used = memoryEnabled && (run.memory_mode ?? 'on') === 'on' ? run.memory_used ?? [] : [];
  const rowText = thinkingRowState(run).text;
  // C6-answer: the ThinkingRow owns the run's process and its one time display. It sits in the answer bubble, above the
  // answer and on its left edge; with no bubble (a settled run without text) it stands alone.
  const row = <ThinkingRow run={run} answerStarted={answerStarted(run)} notes={process} gpuJob={gpuJob} />;
  const switched = failoverNotice(run);
  return <section className="tw-messages" aria-label="任务对话">{served && <p className="tw-served-model" data-served-model={servedId}>{served}</p>}
    <GpuJobStatus runId={run.id} live={live} hideSettled={confirmPending(run)} onRetry={onRetryTraining} onJob={onGpuJob} />
    {switched && <p className="tw-muted tw-failover-notice" role="status" data-testid="tw-failover-notice">{switched}</p>}
    {shown.map(message => {
      const content = message.content;
      const streaming = live && message.id === answerId;
      const answer = message.id === answerId;
      if (message.role !== 'assistant') return content ? <article className={`tw-message tw-${message.role}`} key={message.id}><small>你</small><p><UserText text={content} /></p></article> : null;
      return <article className={`tw-message tw-assistant${streaming ? ' is-live' : ''}`} key={message.id}><Speaker />
        {answer && row}
        {content && <MarkdownMessage text={content} answer={answer} />}
        {answer && chips.shown.length > 0 && <div className="tw-attachments" data-testid="tw-attachments">
          {chips.shown.map(file => <AttachmentChip key={file.id} file={file} inlinePreview />)}
          {chips.more.length > 0 && <details className="tw-details tw-attachments-more"><summary>还有 {chips.more.length} 个文件</summary>{chips.more.map(file => <AttachmentChip key={file.id} file={file} />)}</details>}
          <FileSources files={files} />
        </div>}
        {!streaming && content && <MessageActions text={content} latest={message.id === latestAssistant} />}
        {message.id === chipAt && <MemoryChip saved={run.memory_saved!} />}{message.id === latestAssistant && <MemoryUsed used={used} />}</article>;
    })}
    {!answerShown && rowText && <div className="tw-message tw-thinking-standalone">{row}</div>}
    {run.plan_proposed && <ExperiencePlan key={run.id} runId={run.id} plan={run.plan_proposed} decision={run.plan_decision} onDecided={onRetry} />}
    {filesNotice && <ProjectFilesNotice notice={filesNotice} projectId={projectId} onRetry={onRetry} />}
    {surface === 'card' && phase.copy && <article className="tw-message tw-assistant"><Speaker /><p>{blockedAssistantLine(phase.copy.status)}</p></article>}
    {!unavailable && <StuckBanner run={run} onStop={onStop} onRetry={onResume ? () => { void onResume(); } : onRetry} />}
    {unavailable && modelHelp}
    {!rowText && !run.answer && !run.streaming_text && !phase.copy && <p role="status" className="tw-muted">{taskStatus(run.status)}</p>}
  </section>;
}
