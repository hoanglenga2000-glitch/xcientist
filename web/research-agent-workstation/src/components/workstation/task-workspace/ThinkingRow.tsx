'use client';
import { useEffect, useId, useRef, useState } from 'react';
import { AUTO_COLLAPSE_MS, elapsedSince, isLiveTask, liveClock, sanitizeThinkingSummary, secondsLabel, thinkingAnnouncement, thinkingRowOpen, thinkingRowState, timelineRows } from './task-presentation';
import type { TaskRun } from './types';
import type { GpuJob } from './gpu-status';
import { DeepEvoLoader, DeepEvoMark } from '../DeepEvoBrand';

/** Re-render once a second while a clock is on screen. */
function useNow(live: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [live]);
  return now;
}

/** SPEC §5.4: the real timeline — `✓ 标签 · 用时` per row; the row count is the header's N 步. Live view keeps the last 3. */
function Timeline({ run, all }: { run: TaskRun; all: boolean }) {
  const rows = timelineRows(run.activity ?? [], all);
  if (!rows.length) return null;
  return <ol className="tw-activity" data-testid="tw-activity-timeline">{rows.map((row, index) => {
    const time = row.status === 'running' ? liveClock(elapsedSince(row.started_at)) : typeof row.duration_ms === 'number' ? secondsLabel(row.duration_ms) : null;
    return <li key={`${index}:${row.label}`} className={`tw-activity-row is-${row.status}`} data-testid="tw-activity-row" data-activity-status={row.status}>
      <span className="tw-activity-mark" aria-hidden="true">{row.status === 'completed' ? '✓' : row.status === 'running' ? '›' : ''}</span>
      <span className="tw-activity-text"><span className="tw-activity-label">{row.label}{row.status === 'failed' ? ` · ${row.error || '这一步没有完成'}` : ''}</span>
        {time !== null && <span className="tw-activity-time">{time}</span>}</span>
    </li>;
  })}</ol>;
}

/**
 * t237 C6-answer ThinkingRow (CP-A3): the one process line above the answer, replacing the progress card, the process
 * panel, the thinking summary and the footer 「工作了 Ns」. The time shows here only.
 * Live: loader + 「思考中…」/「正在{动作}…」/「正在整理答案…」 + clock, open until the first answer character, then folds
 * after AUTO_COLLAPSE_MS. A user toggle is final. History starts folded.
 */
export function ThinkingRow({ run, answerStarted, notes = [], gpuJob = null }: { run: TaskRun; answerStarted: boolean; notes?: string[]; gpuJob?: GpuJob | null }) {
  const live = isLiveTask(run.status);
  const now = useNow(live);
  const state = thinkingRowState(run, now, gpuJob);
  const regionId = useId();
  const [manual, setManual] = useState<boolean | null>(null);
  // Open only for a run that is live when first seen; a finished run (history) mounts folded.
  const [autoOpen, setAutoOpen] = useState(() => live && !answerStarted);
  const folding = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!autoOpen || (live && !answerStarted)) return;
    folding.current = window.setTimeout(() => setAutoOpen(false), AUTO_COLLAPSE_MS);
    return () => window.clearTimeout(folding.current);
  }, [autoOpen, live, answerStarted]);
  if (!state.text) return null;
  const open = thinkingRowOpen(manual, autoOpen);
  const summary = sanitizeThinkingSummary(run.reasoning_summary);
  return <div className={`tw-thinking-row is-${state.phase}${state.live ? ' is-live' : ''}`} data-testid="tw-thinking-row" data-phase={state.phase}>
    <button type="button" className="tw-process-line" data-testid="tw-process-line" aria-expanded={open} aria-controls={regionId} onClick={() => setManual(!open)}>
      <span className="tw-thinking-icon" aria-hidden="true">{state.live && state.phase !== 'queued'
        ? <DeepEvoLoader size={16} className="tw-thinking-loader" />
        : <DeepEvoMark size={16} className="tw-thinking-mark" />}</span>
      <span className="tw-thinking-text" key={state.text}>{state.text}</span>
      {state.clock && <span className="tw-thinking-clock" aria-hidden="true"> · {state.clock}</span>}
      <span className="tw-thinking-chevron" aria-hidden="true">›</span>
    </button>
    <span className="tw-sr-only" role="status" aria-live="polite">{thinkingAnnouncement(state.phase)}</span>
    <div id={regionId} className="tw-thinking-detail" hidden={!open}>
      <Timeline run={run} all={!state.live} />
      {summary && <p className="tw-thinking-body" data-testid="tw-thinking-summary">{summary}</p>}
      {notes.map((text, index) => <p className="tw-process-note" key={index}>{text}</p>)}
    </div>
  </div>;
}
