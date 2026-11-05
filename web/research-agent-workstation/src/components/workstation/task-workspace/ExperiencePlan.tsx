'use client';
import { useEffect, useRef, useState } from 'react';
import { BookMarked, X } from 'lucide-react';
import { jsonPost, taskRequest } from './types';

/** t237 C3.11 (spec 06): experience entry only — a draft intent and a plan card that records a decision. Nothing here trains or reserves compute. */
export type PlanProposed = { references: Array<{ title: string; summary: string }>; steps: string[]; needs_compute: boolean };
export type PlanDecision = '' | 'confirm' | 'cancel';

export const EXPERIENCE_COPY = {
  line: '从已有经验开始一次模型开发。', start: '用经验库开发', empty: '还没有经验。', startChat: '开始对话', chip: '经验库', chipRemove: '经验库，移出',
  plan: '计划', ready: '计划已生成', references: '参考的经验', noReferences: '没有找到相近的经验。', steps: '步骤', resources: '需要的资源',
  noCompute: '不使用算力', compute: '需要算力，确认后另行审批', confirm: '确认计划', cancel: '取消', confirmed: '已确认', cancelled: '已取消计划。', failed: '没能保存，请重试',
} as const;
export const EXPERIENCE_DRAFT_HREF = '/workspace?new=1&intent=experience';

/** The only accepted intent value; anything else is dropped rather than sent. */
export const experienceIntent = (params: URLSearchParams): '' | 'experience' => params.get('intent') === 'experience' ? 'experience' : '';

/** `count` null = summary unreadable: shown as the 有经验 state, no error on the entry page. */
/** Exactly one state (spec 06 §2). The tab already reads 进化, so the section is labelled, not headed. */
export function ExperienceEntryView({ count }: { count: number | null }) {
  return <section className="tw-experience-entry" aria-label="进化">{count === 0
    ? <div className="tw-empty-hint"><p>{EXPERIENCE_COPY.empty}</p><a className="tw-button" href="/workspace?new=1">{EXPERIENCE_COPY.startChat}</a></div>
    : <><p className="tw-experience-line">{EXPERIENCE_COPY.line}</p><a className="tw-primary" href={EXPERIENCE_DRAFT_HREF}>{EXPERIENCE_COPY.start}</a></>}
  </section>;
}

export function ExperienceEntry() {
  const [count, setCount] = useState<number | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    void taskRequest<{ count?: number }>('/api/assistant/experience/summary', { signal: abort.signal })
      .then(body => { if (!abort.signal.aborted && typeof body.count === 'number') setCount(body.count); }).catch(() => undefined);
    return () => abort.abort();
  }, []);
  return <ExperienceEntryView count={count} />;
}

export function ExperienceChip({ onRemove }: { onRemove: () => void }) {
  return <span className="tw-chip tw-experience-chip"><BookMarked size={12} aria-hidden="true" /><span className="tw-chip-name">{EXPERIENCE_COPY.chip}</span>
    <button type="button" aria-label={EXPERIENCE_COPY.chipRemove} onClick={onRemove}><X size={12} aria-hidden="true" /></button></span>;
}

/** Same inline-confirm card as 等你确认. After a decision the buttons go and focus moves to the status line.
 *  The ring (only ever on :focus-visible) is asked for after keyboard activation; a click, a script or a plan that was
 *  already decided on mount moves focus without it, so 已确认 never looks like a focused input. */
export function PlanCard({ plan, decision, busy = false, onDecide }: { plan: PlanProposed; decision: PlanDecision; busy?: boolean; onDecide: (decision: 'confirm' | 'cancel') => void }) {
  const status = useRef<HTMLParagraphElement>(null);
  const decided = useRef(decision);
  const viaKeyboard = useRef(false);
  useEffect(() => { if (decision && !decided.current) status.current?.focus({ focusVisible: viaKeyboard.current } as FocusOptions); decided.current = decision; }, [decision]);
  // A pending card lifts its button row above the sticky composer (html scroll-padding-bottom = composer + 16).
  // Two frames: the composer's ResizeObserver has set --tw-composer-h by then.
  const actions = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (decision) return;
    let frame = requestAnimationFrame(() => { frame = requestAnimationFrame(() => actions.current?.scrollIntoView({ block: 'end' })); });
    return () => cancelAnimationFrame(frame);
  }, [decision]);
  const line = decision && <p ref={status} className="tw-plan-status" role="status" tabIndex={-1}>{decision === 'confirm' ? EXPERIENCE_COPY.confirmed : EXPERIENCE_COPY.cancelled}</p>;
  if (decision === 'cancel') return <section className="tw-confirm tw-inline-confirm tw-plan-card" aria-label={EXPERIENCE_COPY.plan} onKeyDown={() => { viaKeyboard.current = true; }} onPointerDown={() => { viaKeyboard.current = false; }}>{line}</section>;
  const references = plan.references ?? [];
  return <section className="tw-confirm tw-inline-confirm tw-plan-card" aria-label={EXPERIENCE_COPY.plan} onKeyDown={() => { viaKeyboard.current = true; }} onPointerDown={() => { viaKeyboard.current = false; }}><h2>{EXPERIENCE_COPY.plan}</h2>
    {!decision && <p className="tw-sr-only" role="status">{EXPERIENCE_COPY.ready}</p>}
    <h3>{EXPERIENCE_COPY.references}</h3>{references.length ? <ul className="tw-plan-refs">{references.map((item, index) => <li key={index}><span>{item.title}</span><span className="tw-muted">{item.summary}</span></li>)}</ul>
      : <p>{EXPERIENCE_COPY.noReferences}</p>}
    <h3>{EXPERIENCE_COPY.steps}</h3><ol>{(plan.steps ?? []).map((step, index) => <li key={index}>{step}</li>)}</ol>
    <h3>{EXPERIENCE_COPY.resources}</h3><p>{plan.needs_compute ? EXPERIENCE_COPY.compute : EXPERIENCE_COPY.noCompute}</p>
    {line || <div ref={actions} className="tw-actions"><button type="button" className="tw-primary" disabled={busy || undefined} onClick={() => onDecide('confirm')}>{EXPERIENCE_COPY.confirm}</button>
      <button type="button" disabled={busy || undefined} onClick={() => onDecide('cancel')}>{EXPERIENCE_COPY.cancel}</button></div>}
  </section>;
}

/** Records the decision (POST /plan), then re-reads the run so a compute ask shows its 等你确认 card. */
export function ExperiencePlan({ runId, plan, decision: initial, onDecided }: { runId: string; plan: PlanProposed; decision?: string; onDecided?: () => void }) {
  const [recorded, setRecorded] = useState<PlanDecision>(initial === 'confirm' || initial === 'cancel' ? initial : '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function decide(decision: 'confirm' | 'cancel') {
    setBusy(true); setError('');
    try {
      const result = await taskRequest<{ decision: PlanDecision }>(`/api/assistant/runs/${encodeURIComponent(runId)}/plan`, jsonPost({ decision }));
      setRecorded(result.decision === 'confirm' || result.decision === 'cancel' ? result.decision : decision);
      onDecided?.();
    } catch { setError(EXPERIENCE_COPY.failed); } finally { setBusy(false); }
  }
  return <><PlanCard plan={plan} decision={recorded} busy={busy} onDecide={next => void decide(next)} />{error && <p role="alert" className="tw-error">{error}</p>}</>;
}
