'use client';
// t238 C7c-6: the run's GPU training state (queue position + ETA estimate / running / done), remaining minutes and cancel.
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { GPU_COPY, gpuCancelUrl, gpuJobForRun, gpuJobLine, gpuStatusUrl, isAtCap, isLiveGpuJob, isNearCap, type GpuJob, type GpuStatusPayload } from './gpu-status';
import { jsonPost, taskRequest } from './types';
import { revealBelowBar } from './chat-scroll';
import { restoreFocus, type DocLike, type FocusLike } from './focus-return';

const POLL_MS = 15000;
const POLL_FAST_MS = 1500; // waiting / preparing / cancelling: the card follows the job within about two seconds (training start, cancel)

export function GpuJobStatus({ runId, live, onRetry, onJob, hideSettled }: { runId: string; live: boolean; hideSettled?: boolean; onRetry?: () => Promise<void> | void; onJob?: (job: GpuJob | null) => void }) {
  const [payload, setPayload] = useState<GpuStatusPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [confirming, setConfirming] = useState(false);
  const card = useRef<HTMLElement>(null);
  const keep = useRef<HTMLButtonElement>(null);
  const yes = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (confirming) keep.current?.focus(); }, [confirming]);
  // M-3: however the confirmation closes (Esc, keep, click outside, confirm) focus goes back to the button that opened it.
  const cancelOpener = useRef<HTMLButtonElement>(null);
  const wasConfirming = useRef(false);
  useEffect(() => { if (wasConfirming.current && !confirming) restoreFocus(document as unknown as DocLike, [cancelOpener.current as FocusLike | null]); wasConfirming.current = confirming; }, [confirming]);
  const read = useCallback(async () => {
    try { setPayload(await taskRequest<GpuStatusPayload>(gpuStatusUrl(runId))); } catch { /* keep the last state */ }
  }, [runId]);
  const job: GpuJob | null = gpuJobForRun(payload, runId);
  const polling = live || isLiveGpuJob(job);
  const fast = Boolean(job && (job.state === 'queued' || (job.state === 'running' && (job.preparing || job.label === GPU_COPY.cancelling))));
  useEffect(() => {
    void read();
    if (!polling) return undefined;
    const timer = window.setInterval(() => void read(), fast ? POLL_FAST_MS : POLL_MS);
    return () => window.clearInterval(timer);
  }, [read, polling, fast]);
  const jobRef = useRef(onJob);
  jobRef.current = onJob;
  const stateKey = job ? `${job.state}|${job.preparing ? 1 : 0}|${job.badge ?? ''}` : '';
  useEffect(() => { jobRef.current?.(job); }, [stateKey]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => jobRef.current?.(null), []);
  // H3: a card that is waiting (queued / preparing) comes into view under the sticky bar when it appears or changes state.
  const waitingKey = job && (job.state === 'queued' || job.state === 'running') ? stateKey : ''; // 393: the waiting, preparing and training cards sit whole on the first screen
  useEffect(() => { if (waitingKey) revealBelowBar(card.current); }, [waitingKey]);
  if (!job || (hideSettled && !isLiveGpuJob(job) && !job.retry_label)) return null; // a new confirmation is waiting: the finished card of an earlier step is not the run's state
  async function retry() {
    setBusy(true); setError('');
    try { await onRetry?.(); await read(); }
    catch { setError(GPU_COPY.retryFailed); }
    finally { setBusy(false); }
  }
  async function cancel() {
    setConfirming(false); setBusy(true); setError('');
    try { await taskRequest(gpuCancelUrl(runId), jsonPost({})); await read(); }
    catch { setError(GPU_COPY.cancelFailed); }
    finally { setBusy(false); }
  }
  const budget = payload?.budget;
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setConfirming(false); }
    else if (event.key === 'Tab') { event.preventDefault(); (document.activeElement === keep.current && yes.current ? yes.current : keep.current)?.focus(); }
  }
  const queued = job.state === 'queued';
  const held = queued && Boolean(job.retry_label);
  const preparing = job.state === 'running' && Boolean(job.preparing);
  const atCap = isAtCap(job);
  return <section ref={card} className="tw-gpu" data-testid="tw-gpu-status" aria-labelledby="tw-gpu-heading">
    <h2 id="tw-gpu-heading" className="tw-gpu-heading">{GPU_COPY.heading}</h2>
    <p className="tw-gpu-line" role="status" data-state={job.state}>{gpuJobLine(job)}</p>
    {job.notice && <p className="tw-gpu-notice">{job.notice}</p>}
    {job.state === 'queued' && job.eta_is_estimate && <p className="tw-gpu-hint">{GPU_COPY.estimate}</p>}
    {atCap && <p className="tw-gpu-hint" data-testid="tw-gpu-at-cap">{GPU_COPY.atCap}</p>}
    {isNearCap(job) && <p className="tw-gpu-hint" data-testid="tw-gpu-near-cap">{GPU_COPY.nearCap(job.max_minutes ?? 0)}</p>}
    {budget && budget.remaining_minutes > 0 && <p className="tw-gpu-hint" data-testid="tw-gpu-budget">{GPU_COPY.budget(budget.remaining_minutes, budget.daily_limit_minutes, budget.max_run_minutes)}</p>}
    {job.retry_label && onRetry && <button type="button" className="tw-button" data-testid="tw-gpu-retry" disabled={busy} onClick={() => void retry()}>{job.retry_label}</button>}
    {job.can_cancel && !atCap && <button ref={cancelOpener} type="button" className="tw-gpu-cancel" data-testid="tw-gpu-cancel" disabled={busy} onClick={() => setConfirming(true)}>{busy ? GPU_COPY.cancelling : held ? GPU_COPY.giveUpHeld : queued ? GPU_COPY.cancelQueued : preparing ? GPU_COPY.cancelPreparing : GPU_COPY.cancelRunning}</button>}
    {confirming && job.can_cancel && <div className="tw-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); setConfirming(false); } }}>
      <div className="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-gpu-cancel-title" aria-describedby="tw-gpu-cancel-body" data-testid="tw-gpu-cancel-dialog" onKeyDown={keys}>
        <h2 id="tw-gpu-cancel-title">{held ? GPU_COPY.confirmHeldTitle : queued ? GPU_COPY.confirmQueuedTitle : preparing ? GPU_COPY.confirmPreparingTitle : GPU_COPY.confirmRunningTitle}</h2>
        <div id="tw-gpu-cancel-body" className="tw-dialog-body"><p>{held ? GPU_COPY.confirmHeldBody : queued ? GPU_COPY.confirmQueuedBody : preparing ? GPU_COPY.confirmPreparingBody : GPU_COPY.confirmRunningBody}</p></div>
        <div className="tw-dialog-actions">
          <button ref={keep} type="button" className="tw-button tw-nowrap" onClick={() => setConfirming(false)}>{held ? GPU_COPY.confirmHeldNo : queued ? GPU_COPY.confirmQueuedNo : preparing ? GPU_COPY.confirmPreparingNo : GPU_COPY.confirmRunningNo}</button>
          <button ref={yes} type="button" className="tw-button tw-danger-solid tw-nowrap" data-testid="tw-gpu-cancel-confirm" disabled={busy} onClick={() => void cancel()}>{held ? GPU_COPY.confirmHeldYes : queued ? GPU_COPY.confirmQueuedYes : preparing ? GPU_COPY.confirmPreparingYes : GPU_COPY.confirmRunningYes}</button>
        </div>
      </div></div>}
    {error && <p role="alert" className="tw-error">{error}</p>}
  </section>;
}
