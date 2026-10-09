'use client';
import { useEffect, useRef, useState } from 'react';
import { Plus, ListTodo, ArrowUpRight, Search, BookOpen, Settings2, FolderOpen, PanelLeft } from 'lucide-react';
import { TaskDetail, statusLabels } from './TaskDetail';
import type { PreviewTask } from '../model';

async function request(path: string, init: RequestInit = {}) {
  const response = await fetch(`/api/preview/${path}`, { ...init, cache: 'no-store', signal: init.signal ?? AbortSignal.timeout(10000) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message ?? `暂时无法完成操作（${response.status}）。请稍后重试，输入已保留。`);
  return payload;
}
const draftKey = (task: string) => `deepevo-usability-draft:${task}`;

export function PreviewWorkspace() {
  const [tasks, setTasks] = useState<PreviewTask[]>([]);
  const [task, setTask] = useState<PreviewTask | null>(null);
  const [selection, setSelection] = useState('');
  const [tab, setTab] = useState('conversation');
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState('');
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [menu, setMenu] = useState(false);
  const [failure, setFailure] = useState('');
  const pending = useRef<{ fingerprint: string; requestId: string } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams(window.location.search);
    const id = params.get('task') ?? '';
    setSelection(id); setCreating(params.get('new') === '1');
    setTab(['progress', 'results'].includes(params.get('view') ?? '') ? params.get('view')! : 'conversation');
    try { setDraft(sessionStorage.getItem(draftKey(id || 'new')) ?? ''); } catch { /* Memory-only draft still works. */ }
    setLoading(true); setError(''); setTask(null);
    (async () => {
      await request('session', { signal: controller.signal });
      const listing = await request('tasks', { signal: controller.signal });
      if (controller.signal.aborted) return;
      setTasks(listing.tasks);
      if (id) {
        const detail = await request(`tasks/${encodeURIComponent(id)}`, { signal: controller.signal });
        if (!controller.signal.aborted) setTask(detail);
      }
    })().catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '加载失败，请重试。'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [revision]);
  function updateDraft(value: string) {
    setDraft(value);
    try { sessionStorage.setItem(draftKey(selection || 'new'), value); } catch { /* No storage access required. */ }
  }
  async function submit() {
    if (!draft.trim() || busy) return;
    setBusy(true); setError('');
    const fingerprint = JSON.stringify([task?.id, task?.version, draft]);
    if (pending.current?.fingerprint !== fingerprint) pending.current = { fingerprint, requestId: crypto.randomUUID() };
    try {
      if (creating) {
        const created = await request('tasks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: draft.trim().slice(0, 42), requestId: pending.current.requestId }) });
        try { sessionStorage.setItem(draftKey(created.id), draft); sessionStorage.removeItem(draftKey('new')); } catch { /* Keep navigation usable. */ }
        window.location.assign(`/?task=${encodeURIComponent(created.id)}`);
      } else if (task) {
        if (failure === 'offline') throw new Error('样例：网络已断开。恢复网络后重试，输入已保留。');
        const updated = await request(`tasks/${encodeURIComponent(task.id)}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-preview-failure': failure }, signal: AbortSignal.timeout(failure === 'timeout' ? 600 : 10000), body: JSON.stringify({ text: draft, version: task.version, requestId: pending.current.requestId }) });
        setTask(updated); updateDraft(''); pending.current = null;
      }
    } catch (cause) { setError(cause instanceof Error && cause.name === 'TimeoutError' ? '请求超时，输入已保留。请重试；相同请求不会重复记录。' : cause instanceof TypeError ? '网络连接失败。请检查连接后重试，输入已保留。' : cause instanceof Error ? cause.message : '暂时无法保存，请重试。输入已保留。'); }
    finally { setBusy(false); }
  }
  return <div className="workspace">
    <a className="skip-link" href="#main">跳到主要内容</a>
    <aside className={`sidebar ${menu ? 'open' : ''}`} aria-label="主导航"><a className="brand" href="/">DeepEvo<span>深度进化</span></a>
      <a className="new-task" href="/?new=1"><Plus size={18} />新任务</a>
      <a className="nav-current" href="/"><ListTodo size={18} />我的任务</a>
      <div className="upcoming"><p>其余入口 · 下一阶段</p><span><FolderOpen size={17} />资料与数据</span><span><BookOpen size={17} />文献检索</span><span><FolderOpen size={17} />成果</span><span><Settings2 size={17} />设置</span></div>
      <div className="sidebar-note"><strong>先把一件事做清楚</strong><p>需求、进度和结果，都在同一个任务里。</p><small>文献保留独立入口；模型、计算资源与预算集中到设置。此版尚未接入。</small></div>
    </aside>
    <div className="work-area"><div className="preview-strip"><button className="mobile-menu" aria-label="展开导航" aria-expanded={menu} onClick={() => setMenu(!menu)}><PanelLeft size={19} /></button><span><strong>交互预览</strong> · 独立样例，未连接真实模型或算力</span><span className="local-only">仅本机 · 重启清空样例</span></div>
      <main id="main" tabIndex={-1}>
        {loading ? <p role="status" className="empty-state">正在打开你的预览任务…</p> : selection ? task ? <TaskDetail task={task} tab={tab} draft={draft} busy={busy} error={error} setDraft={updateDraft} send={submit} reload={() => setRevision(value => value + 1)} /> : <div className="empty-state" role="alert"><h1>暂时无法打开任务</h1><p>{error}</p><button onClick={() => setRevision(value => value + 1)}>重试</button><a href="/">返回我的任务</a></div> : creating ? <section className="new-pane"><a className="back-link" href="/">← 我的任务</a><p className="eyebrow">新任务 · 样例</p><h1>你想完成什么？</h1><p className="muted">先描述问题与期望结果，不用填写任务编号，也不用先建项目。</p><form className="composer" onSubmit={event => { event.preventDefault(); submit(); }}><label htmlFor="new-request">任务需求</label><textarea id="new-request" autoFocus value={draft} maxLength={4000} onChange={event => updateDraft(event.target.value)} placeholder="例如：帮我整理工业缺陷检测的方法，输出一份阅读提纲。" /><div className="composer-bottom"><span>仅保存预览草稿，不调用模型<br />请勿输入密钥或敏感资料</span><button className="primary" type="submit" disabled={busy || !draft.trim()}>{busy ? '正在保存…' : '保存预览草稿'}</button></div>{error && <p role="alert" className="error">{error}</p>}</form></section> : <section className="task-list"><p className="eyebrow">你的工作</p><h1>我的任务</h1><p className="muted">打开一项工作，继续对话、查看进度或找到结果。</p>
          <label className="task-search"><Search size={18} /><span className="sr-only">搜索任务</span><input aria-label="搜索任务" placeholder="搜索任务名称" value={query} onChange={event => setQuery(event.target.value)} /></label>
          <div className="list-heading"><h2>最近任务</h2><span>以下均为隔离样例</span></div>
          {error && <p role="alert" className="error">{error}<button onClick={() => setRevision(value => value + 1)}>重新加载</button></p>}
          {tasks.filter(item => item.title.includes(query.trim())).map(item => <a className="task-row" key={item.id} href={`/?task=${encodeURIComponent(item.id)}`}><div className="task-symbol"><ListTodo size={22} /></div><div className="task-summary"><h3>{item.title}</h3><p>{item.status === 'needs_input' ? '下一步：补充你关注的应用场景' : item.result ? '已有文件，可打开查看和下载' : '草稿已保存，还没有开始执行'}</p></div><div className="task-meta"><span className={`status ${item.status}`}>{statusLabels[item.status]}</span><small>{new Date(item.updated).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })}</small></div><ArrowUpRight size={18} /></a>)}
          {!tasks.filter(item => item.title.includes(query.trim())).length && <div className="empty-state"><h3>{query ? '没有匹配的任务' : '还没有任务'}</h3><p>{query ? '换个关键词，或清空搜索。' : '点击“新任务”，先描述你想完成什么。'}</p></div>}
          <p className="list-footnote">本版只验证“我的任务 → 对话 → 进度 → 结果”。样例内容不代表真实研究结果。</p>
        </section>}
        {!loading && task && <details className="technical preview-testing"><summary>预览测试（不属于正式产品）</summary><p>故障选项只影响下一次样例回复，不会访问真实服务。测试后选“正常”即可重试。</p><label htmlFor="failure">模拟请求结果 </label><select id="failure" value={failure} onChange={event => setFailure(event.target.value)}><option value="">正常</option><option value="401">会话过期（401）</option><option value="403">没有权限（403）</option><option value="409">版本冲突（409）</option><option value="503">服务故障（503）</option><option value="timeout">超时</option><option value="offline">离线提示</option></select></details>}
      </main>
    </div>
  </div>;
}
