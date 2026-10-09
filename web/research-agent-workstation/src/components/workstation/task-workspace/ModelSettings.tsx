'use client';
import { useEffect, useRef, useState } from 'react';
import { jsonPost, taskRequest } from './types';

export type ModelProfile = { id: string; version: number; name: string; provider: string; base_url: string; model: string; enabled: boolean; verified_at: string; used_at: string };
export type ModelList = { profiles: ModelProfile[]; default_profile_id: string };
const blank = { name: '', provider: 'openai', base_url: '', model: '', api_key: '' };

export function ModelSettings({ returnTask, isolated, embedded = false }: { returnTask: string; isolated: boolean; embedded?: boolean }) {
  const [list, setList] = useState<ModelList>({ profiles: [], default_profile_id: '' });
  const [form, setForm] = useState(blank);
  const [editing, setEditing] = useState<ModelProfile | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState<ModelProfile | null>(null);
  const [consent, setConsent] = useState(false);
  const testKey = useRef('');
  async function load() { setList(await taskRequest<ModelList>('/api/assistant/model-profiles')); }
  useEffect(() => { void load().catch(cause => setError(cause.message)); }, []);

  async function save() {
    if (busy) return;
    setBusy(true); setError(''); setNotice('');
    try {
      await taskRequest('/api/assistant/model-profiles' + (editing ? '/' + editing.id : ''), jsonPost({ ...form, ...(editing ? { version: editing.version } : {}) }));
      setForm(blank); setEditing(null); setNotice('已加密保存。尚未测试连接，也没有启动任务。'); await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败，配置输入保留在本页。'); }
    finally { setBusy(false); }
  }

  async function command(profile: ModelProfile, action: 'default' | 'disable' | 'test') {
    if (busy || (action === 'test' && (!consent || isolated))) return;
    setBusy(true); setError('');
    try {
      const result = await taskRequest<{ status?: string }>(`/api/assistant/model-profiles/${profile.id}/${action}`, jsonPost({ version: profile.version,
        ...(action === 'test' ? { confirm_cost: true, idempotency_key: testKey.current } : {}) }));
      if (action === 'test') {
        setNotice(result.status === 'passed' ? '连接测试通过；尚不代表任务执行成功。' : `连接未通过（${result.status}），没有切换到其他模型。`);
        if (result.status !== 'pending') setTesting(null);
      } else setNotice(action === 'default' ? '已设为新任务的默认选择，已有运行不变。' : '已停用；后续模型请求将停止，历史记录保留。');
      await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : '操作失败，请重试。'); }
    finally { setBusy(false); }
  }
  function edit(profile: ModelProfile) { setEditing(profile); setForm({ name: profile.name, provider: profile.provider, base_url: profile.base_url, model: profile.model, api_key: '' }); setError(''); }

  return <section>{!embedded && <><a className="tw-back" href={returnTask ? `/workspace?task=${encodeURIComponent(returnTask)}` : '/workspace'}>{returnTask ? '← 返回原任务' : '← 我的任务'}</a><p className="tw-eyebrow">设置</p><h1>我的模型</h1></>}<p className="tw-muted">配置自己的模型服务。保存不会发出请求；测试和任务执行可能按服务商规则计费。</p>
    <div className="tw-settings-grid"><section><h2>已保存的模型</h2>{!list.profiles.length && <p className="tw-muted">尚未添加个人模型。填写右侧配置后保存。</p>}
      {list.profiles.map(profile => <article className="tw-model-card" key={profile.id}><h3>{profile.name} {list.default_profile_id === profile.id ? '· 默认' : ''}</h3>
        <p>{profile.model} · v{profile.version} · {profile.enabled ? '可选择' : '已停用'}</p><dl><dt>密钥</dt><dd>已加密保存，不回传浏览器</dd><dt>连接测试</dt><dd>{profile.verified_at ? `通过 · ${new Date(profile.verified_at).toLocaleString('zh-CN')}` : '尚未测试'}</dd><dt>{isolated ? '隔离脚本调用（非真实模型）' : '任务模型调用'}</dt><dd>{profile.used_at ? `成功 · ${new Date(profile.used_at).toLocaleString('zh-CN')}` : '尚无成功记录'}</dd></dl>
        <div className="tw-actions"><button disabled={busy} onClick={() => edit(profile)}>修改配置</button><button disabled={busy || !profile.enabled} onClick={() => void command(profile, 'default')}>设为默认</button>
          <button disabled={busy || !profile.enabled || isolated} onClick={() => { setTesting(profile); setConsent(false); testKey.current = crypto.randomUUID(); }}>测试连接</button><button disabled={busy || !profile.enabled} onClick={() => void command(profile, 'disable')}>停用</button></div>
      </article>)}
      {isolated && <p className="tw-muted">隔离验收环境禁止外部模型请求。可保存测试配置；真实服务测试待单独授权。</p>}
      {testing && <section className="tw-confirm" aria-label="测试连接确认"><h3>测试 {testing.name}</h3><p>向此配置发送一次简短请求，最多 16 个输出 token、30 秒；每小时最多 3 次，不自动重试。</p><label><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} />我确认可能产生服务商费用</label><div className="tw-actions"><button disabled={!consent || busy} onClick={() => void command(testing, 'test')}>确认测试一次</button><button disabled={busy} onClick={() => setTesting(null)}>取消</button></div></section>}
    </section><form className="tw-settings-form" onSubmit={event => { event.preventDefault(); void save(); }}><h2>{editing ? '修改模型配置' : '添加个人模型'}</h2>
      <label htmlFor="model-name">配置名称</label><input id="model-name" required maxLength={80} value={form.name} onChange={event => setForm({ ...form, name: event.target.value })} placeholder="例如：我的研究模型" />
      <label htmlFor="model-provider">服务类型</label><select id="model-provider" value={form.provider} onChange={event => setForm({ ...form, provider: event.target.value })}><option value="openai">OpenAI 兼容 · Chat Completions</option><option value="deepseek">DeepSeek · Chat Completions</option><option value="anthropic">Anthropic · Messages</option></select>
      <label htmlFor="model-address">服务地址</label><input id="model-address" required type="url" value={form.base_url} onChange={event => setForm({ ...form, base_url: event.target.value })} placeholder="https://api.example.com/v1" /><p className="tw-muted">仅公网 HTTPS、443 端口；不接受内网、重定向或带密钥的网址。Anthropic 填根地址，不含 /v1/messages。</p>
      <label htmlFor="model-id">模型名称</label><input id="model-id" required value={form.model} onChange={event => setForm({ ...form, model: event.target.value })} placeholder="填写服务商提供的完整名称" />
      <label htmlFor="model-key">API 密钥{editing ? '（留空保留原密钥）' : ''}</label><input id="model-key" type="password" autoComplete="off" required={!editing} value={form.api_key} onChange={event => setForm({ ...form, api_key: event.target.value })} />
      <p className="tw-muted">密钥只用于服务端受限加密存储，不写入浏览器持久化或运行记录。修改会创建新版本，已有运行不会改用新配置。</p>
      <div className="tw-actions"><button className="tw-primary" disabled={busy}>{busy ? '正在处理…' : '加密保存'}</button>{editing && <button type="button" disabled={busy} onClick={() => { setEditing(null); setForm(blank); }}>取消修改</button>}</div>
    </form></div>{error && <p role="alert" className="tw-error">{error}</p>}<p role="status">{notice}</p>
  </section>;
}
