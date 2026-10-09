'use client';
import { useRef, useState } from 'react';
import { jsonPost, taskRequest, type TaskRun } from './types';

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
  const canPause = ['queued', 'running', 'planning', 'verifying', 'recovering', 'waiting_approval'].includes(run.status);
  const canResume = ['paused', 'blocked', 'failed'].includes(run.status)
    && !['execution_reconciliation_required', 'approval_execution_unconfirmed'].includes(run.error_class || '') && !pending.length;
  return <section aria-label="运行操作">
    {pending.map(approval => <section className="tw-confirm" aria-label="待你确认" key={approval.id}><h2>待你确认：{approval.tool_name === 'file_delete' ? '删除文件' : '执行一项受控操作'}</h2>
      <p className="tw-operation-target">目标：{targetLabel(approval)}</p>
      <p>{approval.reversible ? '操作标记为可恢复，恢复仍需核对实际结果。' : '这项操作不保证可撤销，请先确认目标。'}拒绝后不会执行这项操作，任务会保留并等待调整。</p>
      <p>仅确认当前目标和参数；涉及资源的操作仍须符合已批准上限，不会因本次确认扩大权限。</p>
      <details className="tw-details"><summary>参数与影响详情</summary><pre className="tw-file-preview">{JSON.stringify({ operation: approval.tool_name, arguments: approval.normalized_arguments, impact: approval.impact_scope, risk: approval.risk_level }, null, 2)}</pre></details>
      <label><input type="checkbox" checked={consent === approval.id} disabled={busy} onChange={event => setConsent(event.target.checked ? approval.id : '')} />我已核对目标与影响</label>
      <div className="tw-actions"><button disabled={busy || consent !== approval.id} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: true, note: '用户已核对任务内目标和影响' })}>确认执行</button><button disabled={busy} onClick={() => void submit(`/api/assistant/approvals/${approval.id}`, { approved: false, note: '用户在任务内拒绝此操作' })}>拒绝这项操作</button></div>
    </section>)}
    <div className="tw-actions">{canPause && <button disabled={busy} onClick={() => void submit(`/api/assistant/runs/${run.id}/actions`, { action: 'pause' })}>暂停任务</button>}
      {canResume && <button disabled={busy} onClick={() => { setResuming(!resuming); resumeKey.current ||= crypto.randomUUID(); }}>检查并继续</button>}
    </div>
    {canPause && <p className="tw-muted">暂停会阻止新操作；已发出的操作需等待安全结束，不强制终止服务器作业。</p>}
    {resuming && <section className="tw-confirm"><h2>继续当前运行？</h2><p>会沿用这次运行绑定的模型版本和输入，可能继续产生服务商费用。模型设置的新版本不会替换此运行。</p><div className="tw-actions"><button disabled={busy} onClick={() => void submit(`/api/assistant/runs/${run.id}/actions`, { action: 'resume', idempotency_key: resumeKey.current })}>确认继续当前运行</button><button onClick={() => setResuming(false)}>暂不继续</button></div></section>}
    {error && <p role="alert" className="tw-error">{error}<button disabled={busy} onClick={onChanged}>重新读取状态</button></p>}<p role="status">{notice}</p>
  </section>;
}
