'use client';
import { useEffect, useState } from 'react';
import { useTheme, type ThemeMode } from '../theme/ThemeProvider';
import { ModelSettings } from './ModelSettings';
import { jsonPost, taskRequest } from './types';

const groups = [['models', '模型'], ['resources', '计算资源与额度'], ['account', '账户与外观']] as const;
function ResourceSettings() {
  const [binding, setBinding] = useState<{ state: string } | null>(null);
  const [localMode, setLocalMode] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const abort = new AbortController(); setError('');
    void taskRequest<{ binding: { state: string } | null; personal_resources?: string }>('/api/hpc/byoa/binding', { signal: abort.signal }).then(result => {
      if (!abort.signal.aborted) { setBinding(result.binding); setLocalMode(result.personal_resources === 'unavailable_in_local_mode'); setLoaded(true); }
    }).catch(cause => { if (!abort.signal.aborted) setError(cause.message); });
    return () => abort.abort();
  }, [revision]);
  return <section><h2>我的计算资源</h2>
    <p>{!loaded ? '正在读取账户配置…' : localMode ? '本机模式不接入个人计算资源。需要训练时，请使用管理员开通的账户登录。' : !binding ? '尚未设置计算资源。需要训练时，请联系管理员开通个人资源。' : binding.state === 'active' ? '已登记个人资源。是否可执行仍需每次连接检查，此处不表示 GPU 已连接。' : '资源暂不可使用，请联系管理员核对。'}</p>
    <button onClick={() => setRevision(value => value + 1)}>刷新资源状态</button>
    {error && <p role="alert" className="tw-error">{error}</p>}
    <h2 className="tw-section-title">额度由管理员管理</h2><p className="tw-muted">任务需要计算资源时，会单独说明执行上限并请求确认。这里不把平台预算当作个人余额。</p>
    <p className="tw-muted">本地候选暂未开放资源接入或预算修改；不会读取凭据、连接服务器或启动训练。</p>
  </section>;
}
function AccountSettings() {
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const theme = useTheme();
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
      // A full document change closes streams and discards component drafts.
      // Remove only this product's keys, never other applications' storage.
      for (const storage of [window.localStorage, window.sessionStorage]) {
        for (const key of Object.keys(storage)) if (key.startsWith('evomind.') || key.startsWith('evomind_')) storage.removeItem(key);
      }
      window.location.replace('/login?next=%2Fworkspace');
    } catch (cause) { setError(cause instanceof Error ? cause.message : '退出未完成，请重试。'); setBusy(false); }
  }
  return <section><h2>当前账户：{username || '正在读取…'}</h2><p className="tw-muted">任务、资料和个人模型只属于此账户。账号由管理员开通；重置密码请联系管理员，重置后旧会话失效。</p>
    <button onClick={() => void logout()} disabled={busy}>{busy ? '正在退出…' : '退出登录'}</button>
    <h2 className="tw-section-title">外观</h2><label htmlFor="workspace-theme">选择主题</label>
    <select id="workspace-theme" value={theme.mode} onChange={event => theme.previewMode(event.target.value as ThemeMode)}><option value="dark">深色</option><option value="light">浅色</option><option value="system">跟随系统</option></select>
    <div className="tw-actions"><button disabled={theme.saving || !theme.ready} onClick={() => { setError(''); void theme.saveMode().then(() => setNotice('外观已保存到当前账户。')).catch(() => setError('外观保存失败，当前预览保留。请重试。')); }}>保存外观</button><button onClick={theme.resetMode}>撤销未保存的外观</button></div>
    <p className="tw-muted">界面语言：简体中文。英文界面尚未完成，不提供无效切换。</p>
    {error && <p role="alert" className="tw-error">{error}</p>}<p role="status">{notice}</p>
  </section>;
}
export function WorkspaceSettings({ group, returnTask, isolated }: { group: string; returnTask: string; isolated: boolean }) {
  return <section><a className="tw-back" href={returnTask ? `/workspace?task=${encodeURIComponent(returnTask)}` : '/workspace'}>{returnTask ? '← 返回原任务' : '← 我的任务'}</a>
    <h1>设置</h1><nav className="tw-tabs" aria-label="设置分组">{groups.map(([id, label]) => <a key={id} aria-current={group === id ? 'page' : undefined} href={`/workspace?settings=${id}${returnTask ? `&return_task=${encodeURIComponent(returnTask)}` : ''}`}>{label}</a>)}</nav>
    {group === 'resources' ? <ResourceSettings /> : group === 'account' ? <AccountSettings /> : <ModelSettings returnTask={returnTask} isolated={isolated} embedded />}
  </section>;
}
