'use client';
import { useEffect, useRef, useState } from 'react';
import { BookOpen, FolderOpen, ListTodo, Menu, Plus, Search, Settings2, X } from 'lucide-react';
import { TaskList } from './TaskList';
import { TaskSession } from './TaskSession';
import { WorkspaceSettings } from './WorkspaceSettings';
import { TaskFiles } from './TaskFiles';
import { LegacyRecord, WorkspaceRecords } from './WorkspaceRecords';
import { jsonPost, taskRequest, taskStatus, type UserTask, type TaskDetail } from './types';

export function TaskWorkspace({ isolated = false }: { isolated?: boolean }) {
  const [tasks, setTasks] = useState<UserTask[]>([]);
  const [task, setTask] = useState<UserTask | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [selection, setSelection] = useState('');
  const [creating, setCreating] = useState(false);
  const [settings, setSettings] = useState('');
  const [section, setSection] = useState('');
  const [legacyRun, setLegacyRun] = useState('');
  const [legacyTask, setLegacyTask] = useState('');
  const [view, setView] = useState('conversation');
  const [returnTask, setReturnTask] = useState('');
  const [draft, setDraft] = useState('');
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const creationKey = useRef('');
  const latestDraft = useRef(draft); latestDraft.current = draft;
  const creatingNow = useRef(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const sidebar = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!menu) return;
    sidebar.current?.querySelector<HTMLElement>('a')?.focus();
    function escape(event: KeyboardEvent) { if (event.key === 'Escape') { setMenu(false); menuButton.current?.focus(); } }
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [menu]);
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams(window.location.search);
    const id = params.get('task') ?? '';
    setSettings(params.get('settings') || '');
    setSection(params.get('section') || ''); setLegacyRun(params.get('legacy_run') || ''); setLegacyTask(params.get('legacy_task') || '');
    setView(['progress', 'results', 'more'].includes(params.get('view') || '') ? params.get('view')! : 'conversation');
    setReturnTask(params.get('return_task') ?? '');
    setSelection(id); setCreating(params.get('new') === '1'); setTask(null); setDetail(null); setLoading(true); setError('');
    (async () => {
      const list = await taskRequest<{ tasks: UserTask[] }>('/api/assistant/tasks', { signal: controller.signal });
      if (controller.signal.aborted) return;
      setTasks(list.tasks);
      if (id) {
        const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${encodeURIComponent(id)}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setTask(detail.task); setDetail(detail); setDraft(detail.task.draft); }
      }
    })().catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '读取失败，请重试。'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh]);

  async function save() {
    if (creatingNow.current || busy || (creating && !draft.trim())) return;
    creatingNow.current = true;
    setBusy(true); setError(''); setNotice('');
    try {
      if (creating) {
        const location = new URL(window.location.href);
        creationKey.current ||= location.searchParams.get('draft_request') || crypto.randomUUID();
        location.searchParams.set('draft_request', creationKey.current);
        window.history.replaceState(null, '', location);
        const created = await taskRequest<{ task: UserTask }>('/api/assistant/tasks', jsonPost({ title: draft.trim().split('\n')[0].slice(0, 80), draft, idempotency_key: creationKey.current }));
        window.history.replaceState(null, '', `/workspace?task=${encodeURIComponent(created.task.id)}`);
        setSelection(created.task.id); setTask(created.task); setTasks(current => [created.task, ...current.filter(item => item.id !== created.task.id)]);
        setDetail({ task: created.task, runs: [], files: [] }); setCreating(false);
      } else if (task) {
        const updated = await taskRequest<{ task: UserTask }>(`/api/assistant/tasks/${task.id}/draft`, jsonPost({ title: task.title, draft, version: task.version }));
        setTask(updated.task); setNotice('草稿已保存，尚未执行。');
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败，输入已保留。'); }
    finally { creatingNow.current = false; setBusy(false); }
  }

  useEffect(() => {
    if (!creating || !draft.trim() || busy || error) return;
    const timer = window.setTimeout(() => { void save(); }, 800);
    return () => window.clearTimeout(timer);
  }, [creating, draft, busy, error]);

  return <div className="task-workspace">
    <a className="tw-skip" href="#task-main">跳到主要内容</a>
    <aside id="workspace-navigation" ref={sidebar} className={`tw-sidebar ${menu ? 'tw-open' : ''}`} aria-label="主导航">
      <a className="tw-brand" href="/workspace">DeepEvo<small>深度进化</small></a>
      <a className="tw-new" href="/workspace?new=1"><Plus size={18} />新任务</a>
      <nav><a aria-current={!settings && !section && !creating ? 'page' : undefined} href="/workspace"><ListTodo size={18} />我的任务</a>
        <a aria-current={section === 'data' ? 'page' : undefined} href="/workspace?section=data"><FolderOpen size={18} />资料与数据</a><a aria-current={section === 'literature' ? 'page' : undefined} href="/workspace?section=literature"><BookOpen size={18} />文献检索</a>
        <a aria-current={section === 'results' ? 'page' : undefined} href="/workspace?section=results"><FolderOpen size={18} />成果</a><a aria-current={settings ? 'page' : undefined} href={`/workspace?settings=models${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`}><Settings2 size={18} />设置</a></nav>
      <p className="tw-sidebar-note">需求、进度和结果，都在同一个任务里。<br /><a href="/workspace?section=projects">管理项目</a></p>
    </aside>
    <div className="tw-work-area"><header className="tw-topbar"><button ref={menuButton} className="tw-menu" aria-controls="workspace-navigation" aria-label={menu ? '收起导航' : '展开导航'} aria-expanded={menu} onClick={() => setMenu(!menu)}>{menu ? <X size={20} /> : <Menu size={20} />}</button>
      <span>{isolated ? '隔离验收环境 · 不连接真实模型或算力' : '你的研究工作空间'}</span><a href={`/workspace?settings=account${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`}>账户与设置</a></header>
      <main id="task-main" tabIndex={-1}>
        {settings ? <WorkspaceSettings group={settings} returnTask={returnTask} isolated={isolated} /> : loading ? <p role="status">正在读取你的任务…</p> : legacyRun ? <LegacyRecord id={legacyRun} tasks={tasks} view={view} /> : legacyTask ? <section><h1>请先核实旧记录</h1><p>此链接使用旧工具编号，尚没有可靠的个人任务关联。原数据保留，不会按名称或默认任务认领。</p><p>请让管理员核实归属后再关联；其他个人任务仍可正常使用。</p><a className="tw-button" href="/workspace">我的任务</a></section> : section === 'data' ? <TaskFiles /> : ['results', 'projects'].includes(section) ? <WorkspaceRecords projects={section === 'projects'} /> : section === 'literature' ? <section><h1>文献检索</h1><p>个人文献功能仍在迁移，暂未开放检索和导入。旧文献保留，需先核实个人任务关联，不能借用其他任务。</p><a className="tw-button" href="/workspace">返回我的任务</a><p className="tw-muted">这是本地候选的明确缺项，不表示正式站功能已经迁移完成。</p></section> : selection && !task ? <section className="tw-empty" role="alert"><h1>暂时无法打开任务</h1><p>{error}</p><button onClick={() => setRefresh(value => value + 1)}>重试</button><a href="/workspace">返回我的任务</a></section> : detail ? <TaskSession key={detail.task.id} initial={detail} initialDraft={latestDraft.current} /> : creating ? <section>
          <a className="tw-back" href="/workspace">← 我的任务</a><p className="tw-eyebrow">{creating ? '新任务' : taskStatus(task!.status)}</p>
          <h1>{creating ? '你想完成什么？' : task!.title}</h1><p className="tw-muted">先描述问题与期望结果，不需要填写任务编号，也不用先建项目。</p>
          <form className="tw-composer" onSubmit={event => { event.preventDefault(); void save(); }}><label htmlFor="task-draft">任务需求</label>
            <textarea id="task-draft" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} placeholder="例如：整理工业缺陷检测的方法，输出一份阅读提纲。" />
            <div className="tw-composer-bottom"><span>输入后自动保存，不调用模型。保存后可选资料和模型，再明确开始。<br />请勿在对话中输入密钥。</span><button className="tw-primary" disabled={busy || (creating && !draft.trim())}>{busy ? '正在保存…' : '保存草稿'}</button></div>
          </form><p role="status">{notice}</p>{error && <p role="alert" className="tw-error">{error}</p>}
        </section> : <section><p className="tw-eyebrow">你的工作</p><h1>我的任务</h1><p className="tw-muted">打开一项工作，继续对话、查看进度或找到结果。</p>
          <label className="tw-search"><Search size={18} /><span className="tw-sr-only">搜索任务</span><input aria-label="搜索任务" placeholder="搜索任务名称" value={query} onChange={event => setQuery(event.target.value)} /></label>
          {error && <p role="alert" className="tw-error">{error}<button onClick={() => setRefresh(value => value + 1)}>重新加载</button></p>}
          <TaskList tasks={tasks} query={query} />
        </section>}
      </main>
    </div>
  </div>;
}
