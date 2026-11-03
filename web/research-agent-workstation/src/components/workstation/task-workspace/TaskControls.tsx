'use client';
import { useEffect, useRef, useState } from 'react';
import { confirmActionLabel, confirmNoOutboundCopy, confirmSpendLabel, isModelFailure, separateConfirmCopy, STOPPABLE_RUN_STATUSES, stoppedByUser, trainingFailed, usesSeparateConfirm } from './task-presentation';
import { jsonPost, taskRequest, type TaskRun } from './types';
import { revealBelowBar } from './chat-scroll';

function targetLabel(approval: NonNullable<TaskRun['approvals']>[number]) {
  const target = String(approval.normalized_arguments?.path || approval.normalized_arguments?.destination || '').replaceAll('\\', '/');
  const root = String(approval.impact_scope?.workspace_root || '').replaceAll('\\', '/').replace(/\/+$/, '');
  if (root && target.startsWith(`${root}/`)) return `本次运行的 ${target.slice(root.length + 1)}`;
  if (!target) return '见操作详情';
  if (/^(?:[A-Za-z]:|\/)/.test(target)) return '运行目录之外的位置，请展开详情核对';
  return target;
}

export function TaskControls({ run, onChanged }: { run: TaskRun; onChanged: () => void }) {
  const [consent, setConsent] = useState('');
  const [resuming, setResuming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const inFlight = useRef(false);
  const resumeKey = useRef('');
  async function submit(path: string, body: object) {
    if (inFlight.current) return;
    inFlight.current = true; setBusy(true); setError('');
    try {
      await taskRequest(path, jsonPost(body));
      setConsent(''); setResuming(false); setNotice('操作已提交，正在读取服务端状态。'); onChanged();
    } catch (cause) { setError(cause instanceof Error ? cause.message : '结果尚待确认，请重新读取状态，不要重复开始。'); }
    finally { inFlight.current = false; setBusy(false); }
  }
  const pending = (run.approvals ?? []).filter(item => item.status === 'pending');
  const inline = pending.filter(item => !usesSeparateConfirm(item));
  const separate = pending.filter(item => usesSeparateConfirm(item));
  // H3: a confirmation that waits on the reader is brought into view below the sticky 停止/暂停 bar when it first appears.
  const confirmKey = separate.map(item => item.id).join('|');
  const confirmCard = useRef<HTMLElement | null>(null);
  useEffect(() => { if (confirmKey) revealBelowBar(confirmCard.current); }, [confirmKey]);
  const canStop = STOPPABLE_RUN_STATUSES.includes(run.status);
  const canPause = canStop;
  const canResume = ['paused', 'blocked', 'failed'].includes(run.status)
    && !['execution_reconciliation_required', 'approval_execution_unconfirmed'].includes(run.error_class || '') && !pending.length
    // A model failure keeps only the fail card (换个模型继续 / 再试一次); resuming would hit the same model again.
    && !isModelFailure(run)
    // 已完成 / 已取消 / a declined step: 检查并继续 would contradict the status word (the follow-up goes in the composer).
    && !stoppedByUser(run) && !trainingFailed(run)
    // Critiquito F5-7: the CTA follows training_outcome, not status alone — a failed page whose training already succeeded stays terminal (paused may still continue).
    && !(run.training_outcome === 'failed' && run.status !== 'paused')
    && !(run.training_outcome === 'succeeded' && run.status !== 'paused');
  // C6c H_c6c_s3: the section is display:contents so 停止 sticks to the top of the chat scroll (first fold @393).
  // H_c6c_s1: no help text under 停止 / 暂停 — the button weight carries the difference.
  return <section className="tw-run-ops" aria-label="运行操作">
    <div className="tw-actions tw-run-controls">{canStop && <button type="button" className="tw-primary tw-stop" data-testid="tw-stop" disabled={busy} onClick={() => void submit(`/api/assistant/runs/${run.id}/actions`, { action: 'cancel' })}>停止</button>}
      {canPause && <button type="button" className="tw-text-button tw-pause" data-testid="tw-pause" disabled={busy} onClick={() => void submit(`/api/assistant/runs/${run.id}/actions`, { action: 'pause' })}>暂停</button>}
      {canResume && <button className="tw-primary" disabled={busy} onClick={() => { setResuming(!resuming); resumeKey.current ||= crypto.randomUUID(); }}>检查并继续</button>}
    </div>
    {inline.map(approval => {
      const outbound = confirmNoOutboundCopy(approval);
      const spend = confirmSpendLabel(approval);
      return <section className="tw-confirm tw-inline-confirm" aria-label="等你确认" key={approval.id}><h2>等你确认</h2>
      <p className="tw-confirm-label">要做什么</p><p>{confirmActionLabel(approval.tool_name)}</p>
      {spend ? <><p className="tw-confirm-label">花多少</p><p>{spend}</p></> : null}
      {outbound ? <><p className="tw-confirm-label">外发</p><p>{outbound}</p></> : null}
      <div className="tw-actions"><button className="tw-primary" disabled={busy} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: true, note: '用户同意按当前额度继续' })}>同意并继续</button><button disabled={busy} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: false, note: '用户要求调整方案' })}>调整方案</button></div>
    </section>;
    })}
    {separate.map((approval, index) => { const copy = separateConfirmCopy(approval); return <section ref={index === 0 ? confirmCard : undefined} className="tw-confirm" aria-label="待你确认" key={approval.id}><h2>待你确认：{copy.title}</h2>
      <p className="tw-operation-target">目标：{copy.target ?? targetLabel(approval)}</p>
      <p>{copy.reversible ?? (approval.reversible ? '这一步可以撤销，撤销后请核对实际结果。' : '这一步不保证可以撤销，请先确认目标。')}{copy.refusal}</p>
      <p>{copy.scope}</p>
      {copy.showDetails && <details className="tw-details"><summary>参数与影响详情</summary><pre className="tw-file-preview">{JSON.stringify({ operation: approval.tool_name, arguments: approval.normalized_arguments, impact: approval.impact_scope, risk: approval.risk_level }, null, 2)}</pre></details>}
      <label><input type="checkbox" checked={consent === approval.id} disabled={busy} onChange={event => setConsent(event.target.checked ? approval.id : '')} />{copy.consent}</label>
      <div className="tw-actions"><button disabled={busy || consent !== approval.id} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: true, note: '用户已核对任务内目标和影响' })}>{copy.approve}</button><button disabled={busy} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: false, note: '用户在任务内拒绝此操作' })}>{copy.reject}</button></div>
    </section>; })}
    {resuming && <section className="tw-confirm"><h2>继续当前运行？</h2><p>会沿用这次运行绑定的模型版本和输入，可能继续产生服务商费用。模型设置的新版本不会替换此运行。</p><div className="tw-actions"><button disabled={busy} onClick={() => void submit(`/api/assistant/runs/${run.id}/actions`, { action: 'resume', idempotency_key: resumeKey.current })}>确认继续当前运行</button><button onClick={() => setResuming(false)}>暂不继续</button></div></section>}
    {error && <p role="alert" className="tw-error">{error}<button disabled={busy} onClick={onChanged}>重新读取状态</button></p>}<p role="status">{notice}</p>
  </section>;
}
