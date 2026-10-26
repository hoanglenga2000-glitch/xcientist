'use client';
import { useEffect, useRef, useState } from 'react';
import { FolderOpen, ListTodo, Menu, MessageSquare, Plus, Search, UserRound, X } from 'lucide-react';
import { TaskList } from './TaskList';
import { TaskSession } from './TaskSession';
import { WorkspaceSettings } from './WorkspaceSettings';
import { TaskFiles } from './TaskFiles';
import { LegacyRecord, WorkspaceRecords } from './WorkspaceRecords';
import type { ModelList } from './ModelSettings';
import { CHAT_SUGGESTIONS, CHAT_TITLE, conversationGroup, isLiveTask, platformOptionLabel, sortTasksByUpdatedAt } from './task-presentation';
import { jsonPost, taskRequest, type UserTask, type TaskDetail, type UserFile } from './types';
import { AttachButton, AttachmentChips, useComposerUpload } from './ComposerAttach';
import { ModelOptions } from './ModelOptions';
import { carryAttachments } from './upload-attachment';
import { defaultModelChoice, INTENSITY_LEVELS, LAST_MODEL_KEY, NEXT_INTENSITY_KEY, NEXT_MODEL_KEY, type Intensity } from './composer-options';

const WORK_FILTERS = [['', '全部'], ['active', '进行中'], ['confirm', '待确认'], ['stuck', '停住了'], ['done', '已交付']] as const;

function matchesWorkFilter(status: string, filter: string) {
  if (filter === 'active') return ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing'].includes(status);
  if (filter === 'confirm') return status === 'waiting_approval';
  if (filter === 'stuck') return ['blocked', 'failed', 'paused', 'undelivered'].includes(status);
  if (filter === 'done') return ['delivered', 'completed'].includes(status);
  return true;
}

function WorkBoard({ tasks, tab, isolated, initialFilter }: { tasks: UserTask[]; tab: string; isolated: boolean; initialFilter: string }) {
  const [filter, setFilter] = useState(initialFilter);
  const [query, setQuery] = useState('');
  const visible = tasks.filter(task => matchesWorkFilter(task.status, filter));
  return <section>
    <h1>工作</h1>
    <nav className="tw-tabs" aria-label="工作分页">
      <a aria-current={!tab || tab === 'tasks' ? 'page' : undefined} href="/workspace?section=work">任务</a>
      <a aria-current={tab === 'resources' ? 'page' : undefined} href="/workspace?section=work&tab=resources">计算资源</a>
      <a aria-current={tab === 'evolution' ? 'page' : undefined} href="/workspace?section=work&tab=evolution">进化</a>
    </nav>
    {tab === 'resources' ? <WorkspaceSettings group="resources" returnTask="" isolated={isolated} embedded /> : tab === 'evolution' ? <section><h2>进化</h2><p className="tw-muted">在对话里说出想改进的目标，进化会跟着那次对话推进。</p></section> : <>
      <div className="tw-filters" role="group" aria-label="任务状态">{WORK_FILTERS.map(([id, label]) => <button key={id || 'all'} type="button" aria-pressed={filter === id} onClick={() => setFilter(current => current === id ? '' : id)}>{label} {tasks.filter(task => matchesWorkFilter(task.status, id)).length}</button>)}</div>
      <label className="tw-search"><Search size={18} /><span className="tw-sr-only">搜索任务</span><input aria-label="搜索任务" placeholder="搜索任务名称" value={query} onChange={event => setQuery(event.target.value)} /></label>
      <TaskList tasks={visible} query={query} showPending={false} constrained={Boolean(filter)} />
      <p className="tw-muted"><a href="/workspace?section=projects">管理项目</a> · <a href="/workspace?section=results">查看成果记录</a></p>
    </>}
  </section>;
}

export function TaskWorkspace({ isolated = false }: { isolated?: boolean }) {
  const [tasks, setTasks] = useState<UserTask[]>([]);
  const [task, setTask] = useState<UserTask | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [selection, setSelection] = useState('');
  const [creating, setCreating] = useState(false);
  const [settings, setSettings] = useState('');
  const [section, setSection] = useState('');
  const [workTab, setWorkTab] = useState('');
  const [legacyRun, setLegacyRun] = useState('');
  const [legacyTask, setLegacyTask] = useState('');
  const [view, setView] = useState('conversation');
  const [returnTask, setReturnTask] = useState('');
  const [draft, setDraft] = useState('');
  const [historyQuery, setHistoryQuery] = useState('');
  const [homeModel, setHomeModel] = useState('');
  const [homeIntensity, setHomeIntensity] = useState<Intensity>('中');
  const [homeFiles, setHomeFiles] = useState<UserFile[]>([]);
  const homeUpload = useComposerUpload(file => setHomeFiles(current => [...current.filter(item => item.id !== file.id), file]));
  const [modelList, setModelList] = useState<ModelList>({ profiles: [], default_profile_id: '' });
  const [pendingOnly, setPendingOnly] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [locationKey, setLocationKey] = useState('');
  const [draftTask, setDraftTask] = useState('');
  const creationKey = useRef('');
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
    const onPop = () => setLocationKey(window.location.search);
    let opened = false;
    const onShow = () => {
      if (!opened) { opened = true; return; }
      setLocationKey(window.location.search);
    };
    window.addEventListener('popstate', onPop);
    window.addEventListener('pageshow', onShow);
    return () => {
      window.removeEventListener('popstate', onPop);
      window.removeEventListener('pageshow', onShow);
    };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void taskRequest<ModelList>('/api/assistant/model-profiles', { signal: controller.signal }).then(value => {
      if (controller.signal.aborted) return;
      setModelList(value);
      setHomeModel(current => defaultModelChoice(value, current, window.sessionStorage.getItem(LAST_MODEL_KEY)));
    }).catch(() => undefined);
    return () => controller.abort();
  }, [settings]);
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams(window.location.search);
    const id = params.get('task') ?? '';
    setSettings(params.get('settings') || '');
    setSection(params.get('section') || '');
    setWorkTab(params.get('tab') || '');
    setLegacyRun(params.get('legacy_run') || '');
    setLegacyTask(params.get('legacy_task') || '');
    setView(['progress', 'results', 'more'].includes(params.get('view') || '') ? params.get('view')! : 'conversation');
    setReturnTask(params.get('return_task') ?? '');
    setPendingOnly(params.get('pending') === '1');
    setSelection(id); setCreating(params.get('new') === '1'); setTask(null); setDetail(null); setDraftTask(''); setLoading(true); setError('');
    (async () => {
      const list = await taskRequest<{ tasks: UserTask[] }>('/api/assistant/tasks', { signal: controller.signal });
      if (controller.signal.aborted) return;
      setTasks(list.tasks);
      if (id) {
        const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${encodeURIComponent(id)}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setTask(detail.task); setDetail(detail); setDraft(detail.task.draft); setDraftTask(detail.task.id); }
      }
    })().catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '读取失败，请重试。'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh, locationKey]);

  async function save() {
    if (creatingNow.current || busy || homeUpload.busy || (creating && !draft.trim())) return;
    creatingNow.current = true;
    setBusy(true); setError(''); setNotice('');
    try {
      if (creating) {
        const location = new URL(window.location.href);
        const start = location.searchParams.get('start') === '1';
        creationKey.current ||= location.searchParams.get('draft_request') || crypto.randomUUID();
        location.searchParams.set('draft_request', creationKey.current);
        window.history.replaceState(null, '', location);
        const created = await taskRequest<{ task: UserTask }>('/api/assistant/tasks', jsonPost({ title: draft.trim().split('\n')[0].slice(0, 80), draft, idempotency_key: creationKey.current }));
        window.history.replaceState(null, '', `/workspace?task=${encodeURIComponent(created.task.id)}${start ? '&start=1' : ''}`);
        carryAttachments(created.task.id, homeFiles.map(file => file.id)); setHomeFiles([]);
        window.sessionStorage.setItem(NEXT_INTENSITY_KEY, homeIntensity);
        setSelection(created.task.id); setTask(created.task); setTasks(current => [created.task, ...current.filter(item => item.id !== created.task.id)]);
        setDetail({ task: created.task, runs: [], files: [] }); setDraftTask(created.task.id); setCreating(false);
      } else if (task) {
        const updated = await taskRequest<{ task: UserTask }>(`/api/assistant/tasks/${task.id}/draft`, jsonPost({ title: task.title, draft, version: task.version }));
        setTask(updated.task); setNotice('草稿已保存，尚未执行。');
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败，输入已保留。'); }
    finally { creatingNow.current = false; setBusy(false); }
  }

  useEffect(() => {
    if (!creating || !draft.trim() || busy || error || homeUpload.busy) return;
    const timer = window.setTimeout(() => { void save(); }, 800);
    return () => window.clearTimeout(timer);
  }, [creating, draft, busy, error, homeUpload.busy]);

  function openDraft(text: string, start: boolean) {
    if (start && homeModel) window.sessionStorage.setItem(NEXT_MODEL_KEY, homeModel);
    if (start) window.sessionStorage.setItem(NEXT_INTENSITY_KEY, homeIntensity);
    setDraft(text);
    const url = new URL(window.location.href);
    url.searchParams.set('new', '1');
    url.searchParams.delete('task');
    url.searchParams.delete('section');
    url.searchParams.delete('settings');
    if (start) url.searchParams.set('start', '1'); else url.searchParams.delete('start');
    window.history.pushState(null, '', url);
    setLocationKey(url.search);
  }

  const library = section === 'data' || section === 'library';
  const workOpen = section === 'work' || (pendingOnly && !selection && !settings && !creating && !legacyRun && !legacyTask && !section);
  const chatCurrent = !settings && !creating && !section && !legacyRun && !legacyTask && !workOpen;
  const needle = historyQuery.trim().toLocaleLowerCase();
  const history = sortTasksByUpdatedAt(tasks).filter(item => item.title.toLocaleLowerCase().includes(needle));
  const groups = (['今天', '昨天', '更早'] as const).map(label => ({ label, items: history.filter(item => conversationGroup(item.updated_at) === label) })).filter(group => group.items.length);
  const live = sortTasksByUpdatedAt(tasks.filter(item => isLiveTask(item.status)));
  const attachButton = <AttachButton disabled={homeUpload.busy} onFiles={files => void homeUpload.upload(files)} />;
  const attachChips = <AttachmentChips files={homeFiles} status={homeUpload.status} error={homeUpload.error} disabled={busy} onRemove={id => setHomeFiles(current => current.filter(item => item.id !== id))} />;
  const intensityPicker = <span className="tw-intensity" role="group" aria-label="强度">强度{INTENSITY_LEVELS.map(level => <button type="button" key={level} aria-pressed={homeIntensity === level} onClick={() => setHomeIntensity(level)}>{level}</button>)}</span>;
  const settingsHref = `/workspace?settings=models${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`;

  return <div className="task-workspace">
    <a className="tw-skip" href="#task-main">跳到主要内容</a>
    <aside id="workspace-navigation" ref={sidebar} className={`tw-sidebar ${menu ? 'tw-open' : ''}`} aria-label="主导航">
      <a className="tw-brand" href="/workspace">DeepEvo<small>深度进化</small></a>
      <a className="tw-new" href="/workspace?new=1"><Plus size={18} />新对话</a>
      <label className="tw-side-search"><Search size={16} /><span className="tw-sr-only">搜索对话</span><input aria-label="搜索对话" placeholder="搜索" value={historyQuery} onChange={event => setHistoryQuery(event.target.value)} /></label>
      <nav>
        <a aria-current={chatCurrent ? 'page' : undefined} href="/workspace"><MessageSquare size={18} />对话</a>
        <a aria-current={workOpen ? 'page' : undefined} href="/workspace?section=work"><ListTodo size={18} />工作</a>
        <a aria-current={library ? 'page' : undefined} href="/workspace?section=data"><FolderOpen size={18} />资料</a>
        <a aria-current={settings ? 'page' : undefined} href={settingsHref}><UserRound size={18} />我的</a>
      </nav>
      <div className="tw-history">{groups.map(group => <div className="tw-history-group" key={group.label}><p className="tw-history-label">{group.label}</p>{group.items.map(item => <a key={item.id} aria-current={item.id === selection ? 'page' : undefined} href={`/workspace?task=${encodeURIComponent(item.id)}`}>{item.title}</a>)}</div>)}</div>
      {live.length > 0 && <a className="tw-live" href={`/workspace?task=${encodeURIComponent(live[0].id)}`}><span className="tw-dot is-current" /><strong>{live.length} 项进行中</strong><span>{live[0].title}</span></a>}
    </aside>
    <div className="tw-work-area"><header className="tw-topbar"><button ref={menuButton} className="tw-menu" aria-controls="workspace-navigation" aria-label={menu ? '收起导航' : '展开导航'} aria-expanded={menu} onClick={() => setMenu(!menu)}>{menu ? <X size={20} /> : <Menu size={20} />}</button>
      <span>{isolated ? '隔离验收环境 · 不连接真实模型或算力' : '深度进化'}</span><a href={`/workspace?settings=account${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`}>账户</a></header>
      <main id="task-main" tabIndex={-1}>
        {settings ? <WorkspaceSettings group={settings} returnTask={returnTask} isolated={isolated} /> : loading ? <p role="status">正在读取你的任务…</p> : legacyRun ? <LegacyRecord id={legacyRun} tasks={tasks} view={view} /> : legacyTask ? <section><h1>请先核实旧记录</h1><p>此链接使用旧工具编号，尚没有可靠的个人任务关联。原数据保留，不会按名称或默认任务认领。</p><p>请让管理员核实归属后再关联；其他个人任务仍可正常使用。</p><a className="tw-button" href="/workspace">返回对话</a></section> : workOpen ? <WorkBoard tasks={tasks} tab={workTab} isolated={isolated} initialFilter={pendingOnly ? 'confirm' : ''} /> : library ? <section><h1>资料</h1><TaskFiles /><h2>文献</h2><p className="tw-muted">文献按对话整理。</p><a className="tw-button" href="/workspace?section=literature">打开文献</a></section> : ['results', 'projects'].includes(section) ? <WorkspaceRecords projects={section === 'projects'} /> : section === 'literature' ? <section><h1>文献检索</h1><p>个人文献功能仍在迁移，暂未开放检索和导入。旧文献保留，需先核实个人任务关联，不能借用其他任务。</p><a className="tw-button" href="/workspace">返回对话</a><p className="tw-muted">这是本地候选的明确缺项，不表示正式站功能已经迁移完成。</p></section> : selection && !task ? <section className="tw-empty" role="alert"><h1>暂时无法打开任务</h1><p>{error}</p><button onClick={() => setRefresh(value => value + 1)}>重试</button><a href="/workspace">返回对话</a></section> : detail ? <TaskSession key={detail.task.id} initial={detail} initialDraft={draftTask === detail.task.id ? draft : undefined} /> : creating ? <section className="tw-chat">
          <a className="tw-back" href="/workspace">← 对话</a>
          <h1 className="tw-hero">{CHAT_TITLE}</h1>
          <form className="tw-composer" onSubmit={event => { event.preventDefault(); void save(); }}><label className="tw-sr-only" htmlFor="task-draft">任务需求</label>
            <textarea id="task-draft" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} placeholder="说说你要推进的事" />
            {attachChips}
            <div className="tw-composer-bottom">{attachButton}{intensityPicker}<button className="tw-primary" disabled={busy || homeUpload.busy || !draft.trim()}>{busy ? '正在保存…' : '保存草稿'}</button></div>
          </form>{error && <p role="alert" className="tw-error">{error}</p>}{notice && <p role="status">{notice}</p>}
        </section> : <section className="tw-chat tw-chat-empty">
          <div className="tw-hero-block"><span className="tw-mark" aria-hidden="true"><MessageSquare size={16} /></span><h1 className="tw-hero">{CHAT_TITLE}</h1>
            <div className="tw-suggest">{CHAT_SUGGESTIONS.map(text => <button type="button" key={text} onClick={() => setDraft(text)}>{text}</button>)}</div>
          </div>
          <div className="tw-composer-float"><form className="tw-composer" onSubmit={event => { event.preventDefault(); if (draft.trim()) openDraft(draft.trim(), true); }}>
            <label className="tw-sr-only" htmlFor="home-draft">说说你要推进的事</label>
            <textarea id="home-draft" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} placeholder="说说你要推进的事" />
            {attachChips}
            <div className="tw-composer-bottom">{attachButton}<label className="tw-sr-only" htmlFor="home-model">本次使用的模型</label><select id="home-model" value={homeModel} onChange={event => { setHomeModel(event.target.value); window.sessionStorage.setItem(LAST_MODEL_KEY, event.target.value); }}><ModelOptions list={modelList} value={homeModel} platformLabel={platformOptionLabel(modelList.platform_model, null)} /></select>{intensityPicker}<button className="tw-primary tw-send" aria-label="发送" disabled={!draft.trim() || !homeModel || homeUpload.busy}>发送</button></div>
          </form></div>
          {error && <p role="alert" className="tw-error">{error}<button onClick={() => setRefresh(value => value + 1)}>重新加载</button></p>}
        </section>}
      </main>
    </div>
  </div>;
}
