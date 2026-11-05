'use client';
import { useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import { Eye, EyeOff, MoreHorizontal } from 'lucide-react';
import { MenuItem, type MenuAnchor } from './ConversationMenu';
import { PopMenu } from './ProjectMenu';
import { jsonPost, jsonRequest, taskRequest, TaskRequestError } from './types';
import { ErrorNotice } from './ErrorNotice';
import { DeleteModelDialog } from './DeleteModelDialog';

export type ModelProfile = { id: string; version: number; name: string; provider: string; base_url: string; model: string; enabled: boolean; verified_at: string; used_at: string };
export type ModelList = { profiles: ModelProfile[]; default_profile_id: string; platform_model?: { provider: string; model: string } | null; platform_models?: Array<{ id: string; provider: string; model: string; label: string }> | null };
const blank = { name: '', provider: 'openai', base_url: '', model: '', api_key: '' };
const PROVIDERS = [
  { value: 'openai', label: 'OpenAI 兼容' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'anthropic', label: 'Anthropic' },
] as const;

function providerLabel(value: string) {
  return PROVIDERS.find(item => item.value === value)?.label || value;
}

/** CP2 H2-5: a version conflict refreshes the list; never shows the code or version. */
export const MODEL_STATUS_UPDATED = '状态已更新，已刷新列表';

type ModelAction = 'edit' | 'disable' | 'enable' | 'test' | 'default' | 'delete';
/** CP2 H2-1: 停用 / 启用 are plain and reversible; 删除 is the one danger item and lives only in this menu. */
export function modelMenuItems(profile: Pick<ModelProfile, 'enabled'>, isolated: boolean): Array<{ label: string; action: ModelAction; danger?: boolean }> {
  return [
    { label: '编辑', action: 'edit' },
    ...(profile.enabled ? [
      { label: '停用', action: 'disable' as const },
      ...(isolated ? [] : [{ label: '测试连接', action: 'test' as const }]),
      { label: '设为默认', action: 'default' as const },
    ] : [{ label: '启用', action: 'enable' as const }]),
    { label: '删除', action: 'delete', danger: true },
  ];
}

/** CP2 H2-2: a disabled row is greyed and says only 已停用; the switch itself turns it back on. */
export function ModelRow({ profile, isDefault, busy, menuOpen, onToggle, onMenu }: { profile: ModelProfile; isDefault: boolean; busy: boolean; menuOpen: boolean; onToggle: () => void; onMenu: (trigger: HTMLElement) => void }) {
  return <div className={`tw-set-row${profile.enabled ? '' : ' is-disabled'}`}>
    <div className="tw-set-row-main">
      <div className="tw-set-row-title">{profile.name}{isDefault ? ' · 默认' : ''}
        {!profile.enabled && <span className="tw-status-pill tw-status-idle" data-testid="tw-model-disabled-label">已停用</span>}</div>
      <div className="tw-set-row-desc">{providerLabel(profile.provider)} · {profile.model}</div>
    </div>
    <div className="tw-set-row-end">
      <button type="button" role="switch" className="tw-switch" aria-checked={profile.enabled} aria-label={profile.enabled ? '停用' : '启用'}
        disabled={busy} onClick={onToggle} />
      <button type="button" className="tw-row-more" aria-haspopup="menu" aria-expanded={menuOpen}
        aria-label={`${profile.name} 更多`} onClick={event => onMenu(event.currentTarget)}>
        <MoreHorizontal size={16} />
      </button>
    </div>
  </div>;
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function ModelSettings({ returnTask, isolated, embedded = false }: { returnTask: string; isolated: boolean; embedded?: boolean }) {
  const [list, setList] = useState<ModelList>({ profiles: [], default_profile_id: '' });
  const [form, setForm] = useState(blank);
  const [editing, setEditing] = useState<ModelProfile | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [providerOpen, setProviderOpen] = useState(false);
  const [providerActive, setProviderActive] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState<ModelProfile | null>(null);
  const [consent, setConsent] = useState(false);
  const [menuFor, setMenuFor] = useState('');
  const [menuAnchor, setMenuAnchor] = useState<MenuAnchor | null>(null);
  const [deleting, setDeleting] = useState<ModelProfile | null>(null);
  const testKey = useRef('');
  const again = useRef<() => void>(() => reload());
  const panelRef = useRef<HTMLElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const providerOpenRef = useRef(false);
  const providerListId = useId();
  providerOpenRef.current = providerOpen;

  async function load() { setList(await taskRequest<ModelList>('/api/assistant/model-profiles')); }
  function reload() { setError(null); again.current = reload; void load().catch(setError); }
  useEffect(() => { reload(); }, []);

  function openAdd(from?: HTMLElement | null) {
    returnFocusRef.current = from ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setEditing(null); setForm(blank); setShowKey(false); setProviderOpen(false);
    setProviderActive(0); setPanelOpen(true); setError(null); setNotice('');
  }
  function openEdit(profile: ModelProfile, from?: HTMLElement | null) {
    returnFocusRef.current = from ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setEditing(profile); setForm({ name: profile.name, provider: profile.provider, base_url: profile.base_url, model: profile.model, api_key: '' });
    setShowKey(false); setProviderOpen(false);
    setProviderActive(Math.max(0, PROVIDERS.findIndex(item => item.value === profile.provider)));
    setPanelOpen(true); setError(null); setMenuFor('');
  }
  function closePanel() {
    setPanelOpen(false); setEditing(null); setForm(blank); setProviderOpen(false); setShowKey(false);
    const target = returnFocusRef.current;
    queueMicrotask(() => { target?.focus(); });
  }

  useEffect(() => {
    if (!panelOpen) return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusables = () => [...panel.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(el => {
      const style = window.getComputedStyle(el);
      return style.visibility !== 'hidden' && style.display !== 'none';
    });
    const frame = requestAnimationFrame(() => { focusables()[0]?.focus(); });
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        if (providerOpenRef.current) { setProviderOpen(false); return; }
        closePanel();
        return;
      }
      if (event.key !== 'Tab') return;
      const items = focusables();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    panel.addEventListener('keydown', onKey);
    return () => {
      cancelAnimationFrame(frame);
      panel.removeEventListener('keydown', onKey);
    };
  }, [panelOpen]);

  useEffect(() => {
    if (!providerOpen) return;
    document.getElementById(providerListId)?.focus();
  }, [providerOpen, providerListId]);

  function chooseProvider(value: string, index: number) {
    setForm(current => ({ ...current, provider: value }));
    setProviderActive(index);
    setProviderOpen(false);
    document.getElementById('model-provider')?.focus();
  }

  function onProviderTriggerKey(event: ReactKeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      const idx = Math.max(0, PROVIDERS.findIndex(item => item.value === form.provider));
      setProviderActive(event.key === 'ArrowUp' ? idx : idx);
      setProviderOpen(true);
    }
  }

  function onProviderListKey(event: ReactKeyboardEvent<HTMLUListElement>) {
    if (event.key === 'Escape') {
      event.preventDefault();
      setProviderOpen(false);
      document.getElementById('model-provider')?.focus();
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setProviderActive(index => Math.min(PROVIDERS.length - 1, index + 1));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setProviderActive(index => Math.max(0, index - 1));
      return;
    }
    if (event.key === 'Home') {
      event.preventDefault();
      setProviderActive(0);
      return;
    }
    if (event.key === 'End') {
      event.preventDefault();
      setProviderActive(PROVIDERS.length - 1);
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      const item = PROVIDERS[providerActive];
      if (item) chooseProvider(item.value, providerActive);
    }
  }

  async function save() {
    if (busy) return;
    setBusy(true); setError(null); setNotice(''); again.current = () => void save();
    try {
      await taskRequest('/api/assistant/model-profiles' + (editing ? '/' + editing.id : ''), jsonPost({ ...form, ...(editing ? { version: editing.version } : {}) }));
      closePanel(); setNotice('已加密保存。尚未测试连接，也没有启动任务。'); await load();
    } catch (cause) { setError(cause); }
    finally { setBusy(false); }
  }

  /** CP2 H2-5 / L8: a stale version or a double click refreshes the list with one plain line. */
  async function settle(cause: unknown) {
    if (cause instanceof TaskRequestError && cause.status === 409) {
      setNotice(MODEL_STATUS_UPDATED);
      await load().catch(setError);
    } else setError(cause);
  }

  async function command(profile: ModelProfile, action: 'default' | 'disable' | 'enable' | 'test') {
    if (busy || (action === 'test' && (!consent || isolated))) return;
    setBusy(true); setError(null); again.current = () => void command(profile, action);
    try {
      const result = await taskRequest<{ status?: string }>(`/api/assistant/model-profiles/${profile.id}/${action}`, jsonPost({ version: profile.version,
        ...(action === 'test' ? { confirm_cost: true, idempotency_key: testKey.current } : {}) }));
      if (action === 'test') {
        setNotice(result.status === 'passed' ? '连接测试通过；这不代表任务执行成功。' : `连接未通过（${result.status}），没有切换到其他模型。`);
        if (result.status !== 'pending') setTesting(null);
      } else setNotice(action === 'default' ? '已设为新任务的默认选择，已有运行不变。' : action === 'enable' ? '已启用。' : '已停用；后续模型请求将停止，历史记录保留。');
      await load();
    } catch (cause) { await settle(cause); }
    finally { setBusy(false); setMenuFor(''); }
  }

  async function remove(profile: ModelProfile) {
    if (busy) return;
    setBusy(true); setError(null); again.current = () => void remove(profile);
    try {
      await taskRequest(`/api/assistant/model-profiles/${profile.id}`, jsonRequest('DELETE', { version: profile.version }));
      setNotice('已删除。');
      await load();
    } catch (cause) { await settle(cause); }
    finally { setBusy(false); setDeleting(null); menuTrigger.current?.focus(); }
  }

  function choose(profile: ModelProfile, action: ModelAction) {
    if (action === 'edit') openEdit(profile, menuTrigger.current);
    else if (action === 'test') { setTesting(profile); setConsent(false); testKey.current = crypto.randomUUID(); setMenuFor(''); }
    else if (action === 'delete') { setDeleting(profile); setMenuFor(''); }
    else void command(profile, action);
  }

  const menuTrigger = useRef<HTMLElement | null>(null);
  function openMenu(profile: ModelProfile, trigger: HTMLElement) {
    const rect = trigger.getBoundingClientRect();
    menuTrigger.current = trigger;
    setMenuAnchor({ left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom });
    setMenuFor(profile.id);
  }

  const activeOptionId = `${providerListId}-${PROVIDERS[providerActive]?.value ?? 'openai'}`;

  const formPanel = panelOpen ? (
    <div className="tw-sheet-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); closePanel(); } }}>
      <aside ref={panelRef} className="tw-side-panel" role="dialog" aria-modal="true" aria-label={editing ? '修改模型配置' : '添加个人模型'}>
        <header className="tw-side-panel-head"><h2>{editing ? '修改模型配置' : '添加个人模型'}</h2>
          <button type="button" className="tw-text-button" onClick={closePanel}>关闭</button></header>
        <form className="tw-settings-form tw-set-form" onSubmit={event => { event.preventDefault(); void save(); }}>
          <label htmlFor="model-name">配置名称</label>
          <input id="model-name" required maxLength={80} value={form.name} onChange={event => setForm({ ...form, name: event.target.value })} placeholder="例如：我的研究模型" />
          <label id="model-provider-label" htmlFor="model-provider">服务类型</label>
          <div className="tw-custom-select">
            <button type="button" id="model-provider" className="tw-custom-select-trigger" aria-haspopup="listbox" aria-expanded={providerOpen}
              aria-controls={providerOpen ? providerListId : undefined}
              aria-labelledby="model-provider-label model-provider" onClick={() => {
                const idx = Math.max(0, PROVIDERS.findIndex(item => item.value === form.provider));
                setProviderActive(idx);
                setProviderOpen(open => !open);
              }} onKeyDown={onProviderTriggerKey}>
              {providerLabel(form.provider)}
            </button>
            {providerOpen && <ul id={providerListId} className="tw-custom-select-list" role="listbox" tabIndex={-1}
              aria-labelledby="model-provider-label" aria-activedescendant={activeOptionId} onKeyDown={onProviderListKey}>
              {PROVIDERS.map((item, index) => (
                <li key={item.value} id={`${providerListId}-${item.value}`} role="option" aria-selected={form.provider === item.value}
                  data-active={index === providerActive ? 'true' : undefined}
                  onClick={() => chooseProvider(item.value, index)}>{item.label}</li>
              ))}
            </ul>}
          </div>
          <label htmlFor="model-address">服务地址</label>
          <input id="model-address" required type="url" value={form.base_url} onChange={event => setForm({ ...form, base_url: event.target.value })} placeholder="https://api.example.com/v1" />
          <label htmlFor="model-id">模型名称</label>
          <input id="model-id" required value={form.model} onChange={event => setForm({ ...form, model: event.target.value })} placeholder="模型名称" />
          <label htmlFor="model-key">API 密钥{editing ? '（留空保留原密钥）' : ''}</label>
          <div className="tw-secret-field">
            <input id="model-key" type={showKey ? 'text' : 'password'} autoComplete="off" required={!editing} value={form.api_key}
              onChange={event => setForm({ ...form, api_key: event.target.value })} />
            <button type="button" className="tw-secret-toggle" aria-label={showKey ? '隐藏密钥' : '显示密钥'} onClick={() => setShowKey(value => !value)}>
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
          <div className="tw-side-panel-actions">
            <button className="tw-primary" disabled={busy}>{busy ? '正在处理…' : '保存'}</button>
            <button type="button" disabled={busy} onClick={closePanel}>取消</button>
          </div>
        </form>
      </aside>
    </div>
  ) : null;

  return <section className="tw-set-page">
    {!embedded && <><a className="tw-back" href={returnTask ? `/workspace?task=${encodeURIComponent(returnTask)}` : '/workspace'}>{returnTask ? '← 返回原任务' : '← 对话'}</a><p className="tw-eyebrow">设置</p><h1>我的模型</h1></>}
    <p className="tw-set-group-title">我的模型</p>
    {!list.profiles.length ? (
      <div className="tw-empty tw-set-empty"><p>还没有个人模型。</p>
        <button type="button" className="tw-button" onClick={event => openAdd(event.currentTarget)}>添加个人模型</button></div>
    ) : (
      <div className="tw-set-card">
        {list.profiles.map(profile => <ModelRow key={profile.id} profile={profile} isDefault={list.default_profile_id === profile.id} busy={busy}
          menuOpen={menuFor === profile.id} onToggle={() => void command(profile, profile.enabled ? 'disable' : 'enable')} onMenu={trigger => openMenu(profile, trigger)} />)}
        <div className="tw-set-row"><div className="tw-set-row-main" />
          <div className="tw-set-row-end"><button type="button" className="tw-button" onClick={event => openAdd(event.currentTarget)}>添加个人模型</button></div></div>
      </div>
    )}
    {isolated && <p className="tw-muted">隔离验收环境禁止外部模型请求。可保存测试配置；真实服务测试待单独授权。</p>}
    {testing && <section className="tw-confirm" aria-label="测试连接确认"><h3>测试 {testing.name}</h3><p>向此配置发送一次简短请求，最多 16 个输出 token、30 秒；每小时最多 3 次，不自动重试。</p><label><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} />我确认可能产生服务商费用</label><div className="tw-actions"><button disabled={!consent || busy} onClick={() => void command(testing, 'test')}>确认测试一次</button><button disabled={busy} onClick={() => setTesting(null)}>取消</button></div></section>}
    {menuFor && menuAnchor && (() => {
      const profile = list.profiles.find(item => item.id === menuFor);
      if (!profile) return null;
      return <PopMenu id="tw-model-menu" label={`${profile.name} 更多`} anchor={menuAnchor} returnFocus={menuTrigger.current} onClose={() => setMenuFor('')}>
        {modelMenuItems(profile, isolated).map(item => <MenuItem key={item.action} label={item.label} danger={item.danger} onSelect={() => choose(profile, item.action)} />)}
      </PopMenu>;
    })()}
    {deleting && <DeleteModelDialog name={deleting.name} busy={busy} onCancel={() => { setDeleting(null); menuTrigger.current?.focus(); }} onConfirm={() => void remove(deleting)} />}
    {formPanel}
    <ErrorNotice error={error} onAction={action => action === '刷新' ? reload() : again.current()} /><p role="status">{notice}</p>
  </section>;
}
