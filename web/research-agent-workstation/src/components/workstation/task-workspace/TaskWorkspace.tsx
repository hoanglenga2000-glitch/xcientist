'use client';
import { useEffect, useRef, useState } from 'react';
import { ArrowUp, BookOpen, CircleDollarSign, FolderKanban, FolderOpen, ListTodo, Menu, MessageSquare, Plus, Search, UserRound, X } from 'lucide-react';
import { TaskList } from './TaskList';
import { TaskSession, useComposerHeight } from './TaskSession';
import { SidebarHistory } from './SidebarHistory';
import { ProjectPage } from './ProjectPage';
import { DraftProjectChip, draftTaskBody, isSendKey } from './ProjectDraft';
import { ExperienceChip, ExperienceEntry, experienceIntent } from './ExperiencePlan';
import { DeleteProjectDialog, PROJECT_DELETE_COPY } from './DeleteProjectDialog';
import { deleteProject, restoreProject } from './project-folders';
import { showToast } from './Toast';
import { ToastHost } from './Toast';
import { COPY, listConversations } from './conversation-menu';
import { WorkspaceSettings } from './WorkspaceSettings';
import { TaskFiles } from './TaskFiles';
import { LegacyRecord, WorkspaceRecords } from './WorkspaceRecords';
import type { ModelList } from './ModelSettings';
import { CHAT_SUGGESTIONS, CHAT_TITLE, isLiveTask } from './task-presentation';
import { LiveCounts } from './LiveCounts';
import { NotificationsBell, reconcileNotifications } from './NotificationsBell';
import { jsonPost, taskRequest, type Project, type UserTask, type TaskDetail, type TaskRun, type UserFile } from './types';
import { AttachButton, AttachmentChips, useComposerUpload } from './ComposerAttach';
import { IntensityPicker, ModelPicker } from './ComposerPicker';
import { carryAttachments } from './upload-attachment';
import { defaultModelChoice, LAST_MODEL_KEY, pickerModels, NEXT_INTENSITY_KEY, NEXT_MODEL_KEY, type Intensity } from './composer-options';
import { THEME_OPTIONS, useOptionalTheme, type ThemeMode } from '../theme/ThemeProvider';
import { MenuItem, type MenuAnchor } from './ConversationMenu';
import { PopMenu } from './ProjectMenu';
import { DeepEvoLoader, DeepEvoLockup, DeepEvoMark } from '../DeepEvoBrand';

function SidebarUserBlock() {
  const [name, setName] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const [anchor, setAnchor] = useState<MenuAnchor | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const theme = useOptionalTheme();
  useEffect(() => {
    const abort = new AbortController();
    void taskRequest<{ username?: string }>('/api/session/status', { signal: abort.signal }).then(result => {
      if (!abort.signal.aborted && result.username) setName(result.username);
    }).catch(() => {});
    return () => abort.abort();
  }, []);
  function openAppearance() {
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setAnchor({ left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom });
    setMenuOpen(true);
  }
  return <div className="tw-user-block">
    <span className="tw-user-avatar" aria-hidden="true"><UserRound size={16} /></span>
    <span className="tw-user-name">{name || '账户'}</span>
    <button ref={triggerRef} type="button" className="tw-user-appearance" data-slot="appearance" aria-haspopup="menu"
      aria-expanded={menuOpen} aria-label="外观" title="外观" onClick={openAppearance}>外观</button>
    {menuOpen && anchor && <PopMenu id="tw-appearance-menu" label="外观" anchor={anchor} returnFocus={triggerRef.current} onClose={() => setMenuOpen(false)}>
      {THEME_OPTIONS.map(option => (
        <MenuItem key={option.value} role="menuitemradio" label={option.label} checked={theme?.mode === option.value}
          onSelect={() => { theme?.chooseMode(option.value as ThemeMode); setMenuOpen(false); }} />
      ))}
    </PopMenu>}
  </div>;
}

const LIVE_POLL_MS = 20_000;
const WORK_FILTERS = [['', '全部'], ['active', '进行中'], ['confirm', '待确认'], ['stuck', '停住了'], ['done', '已交付']] as const;

function matchesWorkFilter(status: string, filter: string) {
  if (filter === 'active') return isLiveTask(status);
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
    {tab === 'resources' ? <WorkspaceSettings group="resources" returnTask="" isolated={isolated} embedded /> : tab === 'evolution' ? <ExperienceEntry /> : <>
      <div className="tw-filters" role="group" aria-label="任务状态">{WORK_FILTERS.map(([id, label]) => <button key={id || 'all'} type="button" aria-pressed={filter === id} onClick={() => setFilter(current => current === id ? '' : id)}>{label} {tasks.filter(task => matchesWorkFilter(task.status, id)).length}</button>)}</div>
      <label className="tw-search"><Search size={18} /><span className="tw-sr-only">搜索任务</span><input aria-label="搜索任务" placeholder="搜索任务名称" value={query} onChange={event => setQuery(event.target.value)} /></label>
      <TaskList tasks={visible} query={query} showPending={false} constrained={Boolean(filter)} />
      <p className="tw-muted"><a href="/workspace?section=results">查看成果记录</a></p>
    </>}
  </section>;
}

/** C5a L9 (393): with the soft keyboard up the visual viewport is shorter; the shell follows it so the composer stays above the keys and only the message list shrinks. */
export function useViewportHeight() {
  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) return;
    const root = document.documentElement;
    const sync = () => { root.style.setProperty('--tw-vh', `${Math.round(viewport.height)}px`); if (window.scrollY) window.scrollTo(0, 0); };
    sync();
    viewport.addEventListener('resize', sync); viewport.addEventListener('scroll', sync);
    return () => { viewport.removeEventListener('resize', sync); viewport.removeEventListener('scroll', sync); root.style.removeProperty('--tw-vh'); };
  }, []);
}

export function TaskWorkspace({ isolated = false }: { isolated?: boolean }) {
  const [tasks, setTasks] = useState<UserTask[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [archivedCount, setArchivedCount] = useState(0);
  const [listRevision, setListRevision] = useState(0);
  const [task, setTask] = useState<UserTask | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [initialRun, setInitialRun] = useState<TaskRun | null>(null);
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
  const [draftSavedFaded, setDraftSavedFaded] = useState(false);
  useEffect(() => {
    if (notice !== '草稿已保存') { setDraftSavedFaded(false); return; }
    setDraftSavedFaded(false);
    const timer = window.setTimeout(() => setDraftSavedFaded(true), 2000);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [locationKey, setLocationKey] = useState('');
  const [draftTask, setDraftTask] = useState('');
  const [projectParam, setProjectParam] = useState('');
  const [draftProject, setDraftProject] = useState('');
  const [draftIntent, setDraftIntent] = useState('');
  // t238 C7c-A3 #1 (Critiquito C7b addendum): the model picker never opens by itself — closed by default, opens on
  // click only (C4b E2's intent: closed, focus in the composer input).
  const draftPickerAutoOpen = false;
  const [deleting, setDeleting] = useState<{ project: Project; trigger: HTMLElement | null } | null>(null);
  const creationKey = useRef('');
  const creatingNow = useRef(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const sidebar = useRef<HTMLElement>(null);
  const homeComposer = useRef<HTMLDivElement>(null);
  const draftComposer = useRef<HTMLFormElement>(null);
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
      const choices = pickerModels(value);
      setModelList(choices);
      setHomeModel(current => defaultModelChoice(choices, current, window.sessionStorage.getItem(LAST_MODEL_KEY)));
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
    setProjectParam(params.get('project') ?? '');
    setDraftProject(params.get('new') === '1' ? params.get('project') ?? '' : '');
    setDraftIntent(params.get('new') === '1' ? experienceIntent(params) : '');
    const runParam = params.get('assistant_run') || '';
    setSelection(id); setCreating(params.get('new') === '1'); setTask(null); setDetail(null); setInitialRun(null); setDraftTask(''); setLoading(true); setError('');
    (async () => {
      // C6e soft s_d_live_file_body: fetch task list, detail, and deep-linked run in parallel so the answer
      // paints inside VERIFY's open_live wait (serial list→detail→run was missing the body node).
      const [list, detail, prefetched] = await Promise.all([
        taskRequest<{ tasks: UserTask[] }>('/api/assistant/tasks', { signal: controller.signal }),
        id ? taskRequest<TaskDetail>(`/api/assistant/tasks/${encodeURIComponent(id)}`, { signal: controller.signal }) : Promise.resolve(null),
        id && runParam ? taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(runParam)}`, { signal: controller.signal }).catch(() => null) : Promise.resolve(null),
      ]);
      if (controller.signal.aborted) return;
      setTasks(list.tasks); reconcileNotifications(list.tasks, true);
      if (detail) {
        setTask(detail.task); setDetail(detail); setDraft(detail.task.draft); setDraftTask(detail.task.id);
        const owned = Boolean(prefetched && detail.runs.some(item => item.id === prefetched.id));
        setInitialRun(owned ? prefetched : null);
      }
    })().catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '读取失败，请重试。'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh, locationKey]);

  // Folders and 已归档（n） are secondary: the conversation list never waits for them.
  useEffect(() => {
    const controller = new AbortController();
    void taskRequest<{ projects: Project[] }>('/api/assistant/projects', { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setProjects(value.projects); }).catch(() => undefined);
    void listConversations({ archived: '1' }, controller.signal)
      .then(value => { if (!controller.signal.aborted) setArchivedCount(value.length); }).catch(() => undefined);
    return () => controller.abort();
  }, [refresh, locationKey, listRevision]);
  // 我的 → 已归档 changed something: re-read only the sidebar list.
  useEffect(() => {
    if (!listRevision) return;
    const controller = new AbortController();
    void taskRequest<{ tasks: UserTask[] }>('/api/assistant/tasks', { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setTasks(value.tasks); }).catch(() => undefined);
    return () => controller.abort();
  }, [listRevision]);

  // C6e G14 (H10-6): runs keep going server-side. While one is live, a light list read keeps statuses current and feeds the
  // bell; coming back to the tab re-reads once and lists what finished meanwhile without replaying browser notifications.
  const anyLive = tasks.some(item => !item.deleted && isLiveTask(item.status));
  useEffect(() => {
    let controller: AbortController | null = null;
    const read = (historical: boolean) => {
      controller?.abort(); controller = new AbortController();
      const signal = controller.signal;
      void taskRequest<{ tasks: UserTask[] }>('/api/assistant/tasks', { signal }).then(value => {
        if (signal.aborted) return;
        const fresh = new Map(value.tasks.map(item => [item.id, item]));
        setTasks(current => current.map(item => {
          const next = fresh.get(item.id);
          return next ? { ...item, status: next.status, latest_run: next.latest_run } : item;
        }));
        reconcileNotifications(value.tasks, historical);
      }).catch(() => undefined);
    };
    const onVisible = () => { if (document.visibilityState === 'visible') read(true); };
    document.addEventListener('visibilitychange', onVisible);
    const timer = anyLive ? window.setInterval(() => read(false), LIVE_POLL_MS) : undefined;
    return () => { document.removeEventListener('visibilitychange', onVisible); window.clearInterval(timer); controller?.abort(); };
  }, [anyLive]);

  function navigate(href: string) {
    window.history.pushState(null, '', href);
    setMenu(false);
    setLocationKey(`nav:${href}:${Date.now()}`);
  }
  function leaveOpenConversation() {
    window.history.pushState(null, '', '/workspace');
    setLocationKey(`left:${Date.now()}`);
  }
  // A session write (draft autosave) bumps the version; keep the sidebar copy current so its next PATCH is not stale.
  function sessionTask(next: UserTask) {
    setTasks(current => current.some(item => item.id === next.id && item.version < next.version)
      ? current.map(item => item.id === next.id ? { ...item, ...next, status: item.status } : item) : current);
  }

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
        const created = await taskRequest<{ task: UserTask }>('/api/assistant/tasks', jsonPost(draftTaskBody(draft, creationKey.current, draftProject)));
        // TaskSession sends intent=experience with the first run and keeps the 经验库 chip until then.
        window.history.replaceState(null, '', `/workspace?task=${encodeURIComponent(created.task.id)}${start ? '&start=1' : ''}${draftIntent ? '&intent=experience' : ''}`);
        carryAttachments(created.task.id, homeFiles.map(file => file.id)); setHomeFiles([]);
        window.sessionStorage.setItem(NEXT_INTENSITY_KEY, homeIntensity);
        if (homeModel) window.sessionStorage.setItem(NEXT_MODEL_KEY, homeModel);
        setSelection(created.task.id); setTask(created.task); setTasks(current => [created.task, ...current.filter(item => item.id !== created.task.id)]);
        setDetail({ task: created.task, runs: [], files: [] }); setDraftTask(created.task.id); setCreating(false);
      } else if (task) {
        const updated = await taskRequest<{ task: UserTask }>(`/api/assistant/tasks/${task.id}/draft`, jsonPost({ title: task.title, draft, version: task.version }));
        setTask(updated.task); setNotice('草稿已保存');
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败，输入已保留。'); }
    finally { creatingNow.current = false; setBusy(false); }
  }

  useEffect(() => {
    if (!creating || !draft.trim() || busy || error || homeUpload.busy) return;
    const timer = window.setTimeout(() => { void save(); }, 800);
    return () => window.clearTimeout(timer);
  }, [creating, draft, busy, error, homeUpload.busy]);

  // 用经验库开发 lands in the composer input with the 经验库 chip, not in the model picker (C4b E2).
  useEffect(() => {
    if (creating && draftIntent) document.getElementById('task-draft')?.focus();
  }, [creating, draftIntent, loading]);

  /** Enter in the draft: create the task and start its first run (TaskSession picks up start=1). */
  function sendDraft() {
    if (!draft.trim()) return;
    const url = new URL(window.location.href);
    url.searchParams.set('start', '1');
    window.history.replaceState(null, '', url);
    void save();
  }
  function removeDraftIntent() {
    setDraftIntent('');
    const url = new URL(window.location.href);
    url.searchParams.delete('intent');
    window.history.replaceState(null, '', url);
  }
  function removeDraftProject() {
    setDraftProject('');
    const url = new URL(window.location.href);
    url.searchParams.delete('project');
    window.history.replaceState(null, '', url);
  }
  async function confirmDelete() {
    if (!deleting) return;
    const { project, trigger } = deleting;
    setDeleting(null);
    try { await deleteProject(project.id); }
    catch { showToast({ message: COPY.saveFailed }); trigger?.focus(); return; }
    setProjects(list => list.filter(item => item.id !== project.id));
    setTasks(list => list.map(item => item.project_id === project.id ? { ...item, project_id: null } : item));
    document.getElementById('task-main')?.focus();
    if (projectView === project.id || draftProject === project.id) navigate('/workspace');
    else setListRevision(value => value + 1);
    showToast({ message: PROJECT_DELETE_COPY.toast, action: { label: COPY.undo, run: () => void undoDelete(project) } });
  }
  /** Restore brings back the project, its instructions, file bindings and every membership; re-read the list for them. */
  async function undoDelete(project: Project) {
    try {
      const restored = await restoreProject(project.id);
      setProjects(list => [restored, ...list.filter(item => item.id !== restored.id)]);
      setListRevision(value => value + 1);
    } catch { showToast({ message: COPY.saveFailed }); }
  }

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
  // /workspace?project={id} (02 L1); ?new=1&project={id} is the draft, not the page.
  const projectView = !creating && !selection && !settings && !section ? projectParam : '';
  const workOpen = section === 'work' || (pendingOnly && !selection && !settings && !creating && !legacyRun && !legacyTask && !section);
  const chatCurrent = !settings && !creating && !section && !legacyRun && !legacyTask && !workOpen && !projectView;
  const present = tasks.filter(item => !item.deleted);
  const draftProjectInfo = creating ? projects.find(item => item.id === draftProject) : undefined;
  const sessionProjectId = detail ? tasks.find(item => item.id === detail.task.id)?.project_id ?? detail.task.project_id : null;
  const sessionProject = sessionProjectId ? projects.find(item => item.id === sessionProjectId) : undefined;
  useViewportHeight();
  useComposerHeight(creating ? draftComposer : homeComposer, `${creating}:${Boolean(detail)}:${loading}:${settings}:${section}`);
  const attachButton = <AttachButton disabled={homeUpload.busy} onFiles={files => void homeUpload.upload(files)} />;
  const attachChips = <AttachmentChips files={homeFiles} status={homeUpload.status} error={homeUpload.error} errorCode={homeUpload.errorCode} disabled={busy} onRemove={id => setHomeFiles(current => current.filter(item => item.id !== id))} />;
  // Status / error / training hint visible → an open picker closes so it never covers them.
  const composerCloseSignal = [homeUpload.status, homeUpload.error].filter(Boolean).join('|');
  const pickModel = (value: string) => { setHomeModel(value); window.sessionStorage.setItem(LAST_MODEL_KEY, value); };
  const intensityPicker = <IntensityPicker id={creating ? 'draft-intensity' : 'home-intensity'} value={homeIntensity} onChange={setHomeIntensity} />;
  const settingsHref = `/workspace?settings=models${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`;

  return <div className="task-workspace">
    <a className="tw-skip" href="#task-main">跳到主要内容</a>
    <aside id="workspace-navigation" ref={sidebar} className={`tw-sidebar ${menu ? 'tw-open' : ''}`} aria-label="主导航">
      <div className="tw-side-top">
      <a className="tw-brand" href="/workspace" aria-label="DeepEvo 深度进化"><DeepEvoLockup width={168} className="tw-brand-lockup" decorative /><DeepEvoMark size={32} className="tw-brand-mark" /></a>
      <a className="tw-new" href="/workspace?new=1"><Plus size={18} />新对话</a>
      <label className="tw-side-search"><Search size={16} /><span className="tw-sr-only">搜索对话</span><input aria-label="搜索对话" placeholder="搜索" value={historyQuery} onChange={event => setHistoryQuery(event.target.value)}
        onKeyDown={event => { if (event.key === 'Escape' && historyQuery) { event.stopPropagation(); setHistoryQuery(''); } }} /></label>
      <nav>
        <a aria-current={chatCurrent ? 'page' : undefined} href="/workspace"><MessageSquare size={18} />对话</a>
        <a aria-current={workOpen ? 'page' : undefined} href="/workspace?section=work"><ListTodo size={18} />工作</a>
        <a aria-current={library ? 'page' : undefined} href="/workspace?section=data"><FolderOpen size={18} />资料</a>
        <a aria-current={settings ? 'page' : undefined} href={settingsHref}><UserRound size={18} />我的</a>
      </nav>
      </div>
      <SidebarHistory tasks={tasks} setTasks={setTasks} projects={projects} setProjects={setProjects} archivedCount={archivedCount} setArchivedCount={setArchivedCount}
        selection={selection} query={historyQuery} onLeaveOpen={leaveOpenConversation} onNavigate={navigate} openProject={projectView} currentProject={task?.project_id ?? ''}
        onDeleteProject={(project, trigger) => setDeleting({ project, trigger })} />
      <LiveCounts tasks={present} />
      <SidebarUserBlock />
    </aside>
    <div className="tw-work-area"><header className="tw-topbar"><button ref={menuButton} className="tw-menu" aria-controls="workspace-navigation" aria-label={menu ? '收起导航菜单' : '展开导航菜单'} aria-expanded={menu} onClick={() => setMenu(!menu)}>{menu ? <X size={20} /> : <Menu size={20} />}</button>
      <span>{isolated ? '隔离验收环境 · 不连接真实模型或算力' : '深度进化'}</span><NotificationsBell onNavigate={navigate} /><a href={`/workspace?settings=account${selection ? `&return_task=${encodeURIComponent(selection)}` : ''}`}>账户</a></header>
      <main id="task-main" tabIndex={-1}>
        {settings ? <WorkspaceSettings group={settings} returnTask={returnTask} isolated={isolated} onConversationsChanged={() => setListRevision(value => value + 1)} /> : projectView ? <ProjectPage key={projectView} projectId={projectView} tasks={tasks} setTasks={setTasks} projects={projects} setProjects={setProjects}
          setArchivedCount={setArchivedCount} selection={selection} onLeaveOpen={leaveOpenConversation} onNavigate={navigate} /> : loading ? <p role="status" className="tw-loading"><DeepEvoLoader size={48} /><span>正在读取你的任务…</span></p> : legacyRun ? <LegacyRecord id={legacyRun} tasks={tasks} view={view} /> : legacyTask ? <section><h1>请先核实旧记录</h1><p>此链接使用旧工具编号，尚没有可靠的个人任务关联。原数据保留，不会按名称或默认任务认领。</p><p>请让管理员核实归属后再关联；其他个人任务仍可正常使用。</p><a className="tw-button" href="/workspace">返回对话</a></section> : workOpen ? <WorkBoard tasks={present} tab={workTab} isolated={isolated} initialFilter={pendingOnly ? 'confirm' : ''} /> : library ? <section><h1>资料</h1><TaskFiles /><h2>文献</h2><p className="tw-muted">文献按对话整理。</p><a className="tw-button" href="/workspace?section=literature">打开文献</a></section> : ['results', 'projects'].includes(section) ? <WorkspaceRecords projects={section === 'projects'} /> : section === 'literature' ? <section><h1>文献检索</h1><p>个人文献功能仍在迁移，暂未开放检索和导入。旧文献保留，需先核实个人任务关联，不能借用其他任务。</p><a className="tw-button" href="/workspace">返回对话</a><p className="tw-muted">这是本地候选的明确缺项，不表示正式站功能已经迁移完成。</p></section> : selection && !task ? <section className="tw-empty" role="alert"><h1>暂时无法打开任务</h1><p>{error}</p><button onClick={() => setRefresh(value => value + 1)}>重试</button><a href="/workspace">返回对话</a></section> : detail ? <TaskSession key={detail.task.id} initial={detail} initialRun={initialRun} initialDraft={draftTask === detail.task.id ? draft : undefined}
          meta={tasks.find(item => item.id === detail.task.id)} onTaskChange={sessionTask} project={sessionProject} /> : creating ? <section className="tw-chat">
          <div className="tw-chat-scroll"><a className="tw-back" href="/workspace">← 对话</a>
          <h1 className="tw-hero">{CHAT_TITLE}</h1></div>
          <form ref={draftComposer} className="tw-composer" onSubmit={event => { event.preventDefault(); void save(); }}>
            {(draftProjectInfo || draftIntent) && <DraftProjectChip project={draftProjectInfo} onRemove={removeDraftProject}>{draftIntent && <ExperienceChip onRemove={removeDraftIntent} />}</DraftProjectChip>}<label className="tw-sr-only" htmlFor="task-draft">任务需求</label>
            <textarea id="task-draft" value={draft} maxLength={20000} onChange={event => { setDraft(event.target.value); setError(''); }} placeholder="说说你要推进的事"
              onKeyDown={event => { if (isSendKey(event)) { event.preventDefault(); sendDraft(); } }} />
            {attachChips}
            <div className="tw-composer-bottom">{attachButton}<div className="tw-composer-send"><ModelPicker id="draft-model" list={modelList} value={homeModel} autoOpen={draftPickerAutoOpen} closeSignal={composerCloseSignal} onChange={pickModel} />{intensityPicker}</div>{notice === '草稿已保存' ? <p className={`tw-save-status${draftSavedFaded ? ' is-faded' : ''}`} role="status">草稿已保存</p> : null}</div>
          </form>{error && <p role="alert" className="tw-error">{error}</p>}{notice && notice !== '草稿已保存' ? <p role="status">{notice}</p> : null}
        </section> : <section className="tw-chat tw-chat-empty">
          <div className="tw-hero-block"><span className="tw-mark" aria-hidden="true"><MessageSquare size={16} /></span><h1 className="tw-hero">{CHAT_TITLE}</h1>
            <div className="tw-suggest">{CHAT_SUGGESTIONS.map((text, index) => {
              const Icon = [BookOpen, FolderKanban, CircleDollarSign][index] ?? MessageSquare;
              return <button type="button" key={text} onClick={() => setDraft(text)}><Icon size={16} aria-hidden="true" /><span>{text}</span></button>;
            })}</div>
          </div>
          <div ref={homeComposer} className="tw-composer-float"><form className="tw-composer" onSubmit={event => { event.preventDefault(); if (draft.trim()) openDraft(draft.trim(), true); }}>
            <label className="tw-sr-only" htmlFor="home-draft">说说你要推进的事</label>
            <textarea id="home-draft" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} placeholder="说说你要推进的事" />
            {attachChips}
            <div className="tw-composer-bottom">{attachButton}<div className="tw-composer-send"><ModelPicker id="home-model" list={modelList} value={homeModel} onChange={pickModel} />{intensityPicker}<button type="submit" className="tw-send-icon tw-send" aria-label="发送" disabled={!draft.trim() || !homeModel || homeUpload.busy}><ArrowUp size={16} aria-hidden="true" /></button></div></div>
          </form></div>
          {error && <p role="alert" className="tw-error">{error}<button onClick={() => setRefresh(value => value + 1)}>重新加载</button></p>}
        </section>}
      </main>
    </div>
    {deleting && <DeleteProjectDialog project={deleting.project} onCancel={() => { const { trigger } = deleting; setDeleting(null); trigger?.focus(); }} onConfirm={() => void confirmDelete()}
      count={Math.max(deleting.project.conversation_count ?? 0, tasks.filter(item => item.project_id === deleting.project.id && !item.deleted).length)} />}
    <ToastHost />
  </div>;
}
