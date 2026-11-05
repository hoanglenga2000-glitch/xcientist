'use client';
import { useEffect, useState } from 'react';
import { allocationStatusText, type AllocationCheck } from '@/lib/allocation-status';
import { ThemeSegment } from '../theme/ThemeSegment';
import { BudgetUsage, type BudgetPayload } from './BudgetUsage';
import { ModelSettings } from './ModelSettings';
import { ArchivedConversations } from './ArchivedConversations';
import { MemorySettings } from './MemorySettings';
import { jsonPost, taskRequest } from './types';

const groups = [['models', '模型'], ['memory', '记忆'], ['resources', '计算资源与额度'], ['account', '账户与外观'], ['archived', '已归档']] as const;

function ResourceSettings() {
  const [check, setCheck] = useState<AllocationCheck | null>(null);
  const [statusLabel, setStatusLabel] = useState('');
  const [budget, setBudget] = useState<BudgetPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const abort = new AbortController(); setError('');
    void taskRequest<AllocationCheck & { status_label?: string }>('/api/hpc/byoa/binding', { signal: abort.signal }).then(result => {
      if (abort.signal.aborted) return;
      setCheck(result);
      setStatusLabel(result.status_label || '');
      setLoaded(true);
    }).catch(cause => { if (!abort.signal.aborted) setError(cause.message); });
    void taskRequest<{ budget?: BudgetPayload }>('/api/assistant/execution-budget', { signal: abort.signal }).then(result => {
      if (!abort.signal.aborted && result.budget) setBudget(result.budget);
    }).catch(() => undefined);
    return () => abort.abort();
  }, [revision]);
  return <section className="tw-set-page">
    <p className="tw-set-group-title">我的计算资源</p>
    <div className="tw-set-card">
      <div className="tw-set-row"><div className="tw-set-row-main"><div className="tw-set-row-title">{statusLabel || allocationStatusText(check, loaded)}</div></div>
        <div className="tw-set-row-end"><button type="button" className="tw-text-button" onClick={() => setRevision(value => value + 1)}>刷新</button></div></div>
    </div>
    {error && <p role="alert" className="tw-error">{error}</p>}
    <p className="tw-set-group-title">计算资源与额度</p>
    {budget ? <BudgetUsage budget={budget} /> : <p className="tw-muted">暂时没有额度数据。</p>}
  </section>;
}

function AccountSettings() {
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const abort = new AbortController();
    void taskRequest<{ username: string }>('/api/session/status', { signal: abort.signal }).then(result => {
      if (!abort.signal.aborted) setUsername(result.username);
    }).catch(cause => { if (!abort.signal.aborted) setError(cause.message); });
    return () => abort.abort();
  }, []);
  async function logout() {
    setBusy(true); setError('');
    try {
      await taskRequest('/api/auth/logout', jsonPost({}));
      for (const storage of [window.localStorage, window.sessionStorage]) {
        for (const key of Object.keys(storage)) if (key.startsWith('evomind.') || key.startsWith('evomind_')) storage.removeItem(key);
      }
      window.location.replace('/login?next=%2Fworkspace');
    } catch (cause) { setError(cause instanceof Error ? cause.message : '退出未完成，请重试。'); setBusy(false); }
  }
  const initial = (username || '账').slice(0, 1).toUpperCase();
  return <section className="tw-set-page">
    <h2 className="tw-set-group-title">当前账户：{username || '正在读取…'}</h2>
    <div className="tw-set-card">
      <div className="tw-set-row">
        <span className="tw-user-avatar tw-set-avatar" aria-hidden="true">{initial}</span>
        <div className="tw-set-row-main">
          <div className="tw-set-row-title">{username || '正在读取…'}</div>
          <div className="tw-set-row-desc">账号由管理员开通</div>
        </div>
      </div>
      <div className="tw-set-row">
        <div className="tw-set-row-main"><div className="tw-set-row-title">退出登录</div></div>
        <div className="tw-set-row-end"><button type="button" className="tw-text-danger" disabled={busy} onClick={() => void logout()}>{busy ? '正在退出…' : '退出登录'}</button></div>
      </div>
    </div>
    <p className="tw-set-group-title">外观</p>
    <div className="tw-set-card">
      <div className="tw-set-row">
        <div className="tw-set-row-main"><div className="tw-set-row-title">主题</div></div>
        <div className="tw-set-row-end"><ThemeSegment id="workspace-theme" />{/* 撤销未保存的外观 */}</div>
      </div>
    </div>
    {error && <p role="alert" className="tw-error">{error}</p>}
  </section>;
}

export function WorkspaceSettings({ group, returnTask, isolated, embedded = false, onConversationsChanged }: { group: string; returnTask: string; isolated: boolean; embedded?: boolean; onConversationsChanged?: () => void }) {
  return <section className="tw-settings">
    {!embedded && <a className="tw-back" href={returnTask ? `/workspace?task=${encodeURIComponent(returnTask)}` : '/workspace'}>{returnTask ? '← 返回原任务' : '← 对话'}</a>}
    {!embedded && <h1>设置</h1>}
    <div className={`tw-settings-shell${embedded ? ' is-embedded' : ''}`}>
      {!embedded && <nav className="tw-settings-nav" aria-label="设置分组">{groups.map(([id, label]) => (
        <a key={id} aria-current={group === id ? 'page' : undefined}
          href={`/workspace?settings=${id}${returnTask ? `&return_task=${encodeURIComponent(returnTask)}` : ''}`}>{label}</a>
      ))}</nav>}
      <div className="tw-settings-content">
        {group === 'resources' ? <ResourceSettings /> : group === 'account' ? <AccountSettings /> : group === 'archived' ? <ArchivedConversations onChanged={onConversationsChanged} /> : group === 'memory' ? <MemorySettings /> : <ModelSettings returnTask={returnTask} isolated={isolated} embedded />}
      </div>
    </div>
  </section>;
}
