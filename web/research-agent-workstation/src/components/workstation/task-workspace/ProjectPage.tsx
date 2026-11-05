'use client';
import { useEffect, useRef, useState, type Dispatch, type MouseEvent, type SetStateAction } from 'react';
import { MoreHorizontal, Plus, Trash2, Upload } from 'lucide-react';
import { ConversationMenu, MenuItem, type MenuAnchor, type MenuCommand } from './ConversationMenu';
import { useConversationActions } from './conversation-actions';
import { COPY, createProject } from './conversation-menu';
import { ErrorNotice } from './ErrorNotice';
import { MANAGE_FILES } from './ComposerAttach';
import { uploadFailureAction } from './upload-failure';
import { ProjectMemoryToggle } from './MemorySettings';
import { PopMenu } from './ProjectMenu';
import {
  bindProjectFile, listProjectFiles, patchProject, projectDraftHref, readProject, unbindProjectFile, type ProjectFile,
} from './project-folders';
import { sortTasksByUpdatedAt, statusPillClass } from './task-presentation';
import { showToast } from './Toast';
import { TaskRequestError, taskRequest, taskStatus, type Project, type UserFile, type UserTask } from './types';
import { isUploadLimitCode } from './error-copy';
import { uploadUserFile, type UploadJob } from './upload-attachment';

/** 02 §3.2: ≤8000 characters, counter only above 7000, autosave 800ms after typing stops, 已保存 fades after 2s. */
export const INSTRUCTIONS_MAX = 8000;
export const AUTOSAVE_MS = 800;
export const SAVED_FADE_MS = 2000;
export const instructionsCounter = (length: number) => length > 7000 ? `${length} / ${INSTRUCTIONS_MAX}` : '';

export function fileMeta(file: Pick<ProjectFile, 'size' | 'created_at'>) {
  const size = file.size < 1024 ? `${file.size} B` : file.size < 1024 ** 2 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / 1024 ** 2).toFixed(1)} MB`;
  return `${size} · ${new Date(file.created_at).toLocaleDateString('zh-CN')}`;
}

type SaveState = 'idle' | 'saving' | 'saved' | 'faded' | 'failed';
type UploadRow = { key: string; name: string; state: 'uploading' | 'failed'; message?: string; code?: string; file?: File; job?: { current: UploadJob | null } };
type RowMenu = { key: string; anchor: MenuAnchor; returnFocus: HTMLElement | null };
const box = (node: Element): MenuAnchor => { const rect = node.getBoundingClientRect(); return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom }; };
/** The copy with the higher version wins: the sidebar and this page both PATCH the same project. */
const newer = (left: Project | null, right: Project | undefined) => !left ? right ?? null : !right ? left : (right.version ?? 0) > (left.version ?? 0) ? { ...left, ...right } : left;

export function SaveStatus({ state, onRetry }: { state: SaveState; onRetry: () => void }) {
  return <><p className={`tw-save-status${state === 'faded' ? ' is-faded' : ''}`} role="status">{state === 'saving' ? '正在保存…' : state === 'saved' || state === 'faded' ? '已保存' : ''}</p>
    {state === 'failed' && <p className="tw-error tw-inline-error">没能保存<button type="button" onClick={onRetry}>重试</button></p>}</>;
}

/** h1 inline rename (02 §4): Enter saves, Esc restores, an empty name cancels. */
export function ProjectTitle({ name, editing, onEdit, onDone }: { name: string; editing: boolean; onEdit: () => void; onDone: (value: string) => void }) {
  if (!editing) return <h1><button type="button" className="tw-title-edit" aria-label="重命名项目" onClick={onEdit}>{name}</button></h1>;
  return <h1><input className="tw-title-input" aria-label="项目名称" maxLength={120} defaultValue={name} autoFocus
    onFocus={event => event.currentTarget.select()} onBlur={event => onDone(event.currentTarget.value)}
    onKeyDown={event => {
      if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur(); }
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); event.currentTarget.value = name; event.currentTarget.blur(); }
    }} /></h1>;
}

/** t238 C7b S1/S2: quota → 管理文件 (go to 资料), disk → 稍后再试 (dismiss), never a 重试 that re-413s; buttons never wrap. */
function FailedUpload({ message, code, onRetry }: { message?: string; code?: string; onRetry?: () => void }) {
  const action = uploadFailureAction(code);
  return <span className="tw-error tw-inline-error tw-row-error"><span className="tw-row-error-text">{message ?? '上传失败'}</span>
    {action === '管理文件' ? <a className="tw-button tw-nowrap" href={MANAGE_FILES.href}>{action}</a>
      : <button type="button" className="tw-nowrap" onClick={onRetry}>{action}</button>}</span>;
}

export function ProjectFileRow({ file, upload, menuOpen, onRemove, onMenu, onRetry }: {
  file?: ProjectFile; upload?: Pick<UploadRow, 'name' | 'state' | 'message' | 'code'>; menuOpen?: boolean; onRemove?: () => void; onMenu?: (event: MouseEvent<HTMLButtonElement>) => void; onRetry?: () => void;
}) {
  const name = file?.name ?? upload?.name ?? '';
  return <li className="tw-file tw-project-file"><div className="tw-project-file-text"><span className="tw-project-file-name">{name}</span>
    {file ? <small>{fileMeta(file)}</small> : upload?.state === 'uploading' ? <small>正在上传…</small>
      : <FailedUpload message={upload?.message} code={upload?.code} onRetry={onRetry} />}</div>
    {file && <><button type="button" className="tw-file-remove" onClick={onRemove}>移出项目</button>
      <button type="button" className="tw-file-more" aria-haspopup="menu" aria-expanded={Boolean(menuOpen)} aria-label={`文件操作：${name}`} onClick={onMenu}><MoreHorizontal size={24} aria-hidden="true" /></button></>}
  </li>;
}

type Props = {
  projectId: string; tasks: UserTask[]; setTasks: Dispatch<SetStateAction<UserTask[]>>; projects: Project[]; setProjects: Dispatch<SetStateAction<Project[]>>;
  setArchivedCount: Dispatch<SetStateAction<number>>; selection: string; onLeaveOpen: () => void; onNavigate: (href: string) => void;
  /** Tests and a warm cache: skip the first read. */
  initial?: { project: Project; files: ProjectFile[] };
};

export function ProjectPage({ projectId, tasks, setTasks, projects, setProjects, setArchivedCount, selection, onLeaveOpen, onNavigate, initial }: Props) {
  const [loaded, setLoaded] = useState<Project | null>(initial?.project ?? null);
  const [files, setFiles] = useState<ProjectFile[]>(initial?.files ?? []);
  const [error, setError] = useState<unknown>(null);
  const [revision, setRevision] = useState(0);
  const [instructions, setInstructions] = useState(initial?.project.instructions ?? '');
  const [saveState, setSaveState] = useState<SaveState>('idle');
  const [editingTitle, setEditingTitle] = useState(false);
  const [uploads, setUploads] = useState<UploadRow[]>([]);
  const [library, setLibrary] = useState<UserFile[] | null>(null);
  const [menu, setMenu] = useState<RowMenu | null>(null);
  const [renaming, setRenaming] = useState('');
  const project = newer(loaded, projects.find(item => item.id === projectId));
  const current = useRef(project);
  current.current = project;
  const text = useRef(instructions);
  const saved = useRef(initial?.project.instructions ?? '');
  const timer = useRef<number | undefined>(undefined);
  const fade = useRef<number | undefined>(undefined);
  const inflight = useRef(false);
  const again = useRef(false);
  const alive = useRef(new AbortController());
  const { save, archive, remove, download } = useConversationActions({ setTasks, setArchivedCount, selection, onLeaveOpen });

  useEffect(() => {
    const controller = new AbortController(); alive.current = controller;
    if (!initial || revision) {
      setError(null);
      void Promise.all([readProject(projectId, controller.signal), listProjectFiles(projectId, controller.signal)]).then(([value, list]) => {
        if (controller.signal.aborted) return;
        setLoaded(value); setFiles(list);
        setInstructions(value.instructions ?? ''); text.current = saved.current = value.instructions ?? '';
      }).catch(cause => { if (!controller.signal.aborted) setError(cause); });
    }
    return () => { controller.abort(); window.clearTimeout(timer.current); window.clearTimeout(fade.current); };
  }, [projectId, revision]);

  function remember(next: Project) {
    setLoaded(next);
    setProjects(list => list.map(item => item.id === next.id ? { ...item, ...next } : item));
  }
  async function saveInstructions() {
    window.clearTimeout(timer.current);
    const value = text.current;
    if (!current.current || value === saved.current) return;
    if (inflight.current) { again.current = true; return; }
    inflight.current = true; window.clearTimeout(fade.current); setSaveState('saving');
    try {
      remember(await patchProject(current.current, { instructions: value }));
      saved.current = value; setSaveState('saved');
      fade.current = window.setTimeout(() => setSaveState(state => state === 'saved' ? 'faded' : state), SAVED_FADE_MS);
    } catch {
      setSaveState('failed');
      // A stale version (409) or a lost answer: take the server copy's version, keep the text, let 重试 send it again.
      void readProject(projectId).then(value => setLoaded(old => old && { ...old, version: value.version })).catch(() => undefined);
    } finally {
      inflight.current = false;
      if (again.current) { again.current = false; void saveInstructions(); }
    }
  }
  function typeInstructions(value: string) {
    setInstructions(value); text.current = value;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void saveInstructions(), AUTOSAVE_MS);
  }
  async function rename(value: string) {
    setEditingTitle(false);
    const name = value.trim();
    if (!project || !name || name === project.name) return;
    const before = project;
    remember({ ...project, name });
    try { remember(await patchProject(before, { name })); }
    catch { remember(before); showToast({ message: COPY.saveFailed }); }
  }

  const add = (file: ProjectFile) => setFiles(list => [...list.filter(item => item.id !== file.id), file]);
  async function upload(file: File, key = crypto.randomUUID(), job: { current: UploadJob | null } = { current: null }) {
    setUploads(list => [...list.filter(item => item.key !== key), { key, name: file.name, state: 'uploading', file, job }]);
    try {
      const attachment = await uploadUserFile(file, alive.current.signal, undefined, job);
      add(await bindProjectFile(projectId, attachment.id));
      setUploads(list => list.filter(item => item.key !== key));
    } catch (cause) {
      const limited = cause instanceof TaskRequestError && isUploadLimitCode(cause.code);
      const message = limited ? cause.message : undefined;
      const code = limited ? cause.code : undefined;
      if (!alive.current.signal.aborted) setUploads(list => list.map(item => item.key === key ? { ...item, state: 'failed', message, code } : item));
    }
  }
  async function unbind(file: ProjectFile) {
    setMenu(null);
    setFiles(list => list.filter(item => item.id !== file.id));
    try { await unbindProjectFile(projectId, file.id); }
    catch { add(file); showToast({ message: COPY.saveFailed }); return; }
    showToast({ message: '已移出', action: { label: COPY.undo, run: () => void bindProjectFile(projectId, file.id).then(add).catch(() => showToast({ message: COPY.saveFailed })) } });
  }
  async function bindFromLibrary(file: UserFile) {
    setLibrary(null);
    try { add(await bindProjectFile(projectId, file.id)); }
    catch { showToast({ message: COPY.saveFailed }); }
  }
  function toggleLibrary() {
    if (library) { setLibrary(null); return; }
    setLibrary([]);
    void taskRequest<{ files: UserFile[] }>('/api/assistant/files', { signal: alive.current.signal }).then(value => setLibrary(open => open && value.files))
      .catch(() => { setLibrary(null); showToast({ message: COPY.saveFailed }); });
  }

  function command(task: UserTask, action: MenuCommand) {
    setMenu(null);
    if (action === 'rename') setRenaming(task.id);
    else if (action === 'pin') void save(task, { pinned: !task.pinned_at }, { pinned_at: task.pinned_at ? null : new Date().toISOString() });
    else if (action === 'export') download(task.id);
    else if (action === 'archive') void archive(task);
    else void remove(task);
  }
  async function move(task: UserTask, target: Project | null) {
    setMenu(null);
    const next = await save(task, { project_id: target?.id ?? null }, { project_id: target?.id ?? null });
    if (next && target) showToast({ message: COPY.moved(target.name) });
  }
  function renameTask(task: UserTask, value: string) {
    setRenaming('');
    const title = value.trim();
    if (title && title !== task.title) void save(task, { title }, { title });
  }

  if (!project) return <section className="tw-project"><a className="tw-back" href="/workspace">← 对话</a>
    {error ? <ErrorNotice error={error} onAction={() => setRevision(value => value + 1)} /> : <p role="status">正在读取项目…</p>}</section>;
  const draftHref = projectDraftHref(project.id);
  const openDraft = (event: MouseEvent) => { if (event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey) { event.preventDefault(); onNavigate(draftHref); } };
  const conversations = sortTasksByUpdatedAt(tasks.filter(item => item.project_id === project.id && !item.archived_at && !item.deleted));
  const counter = instructionsCounter(instructions.length);
  const menuTask = menu && conversations.find(item => 'task:' + item.id === menu.key);
  const menuFile = menu && files.find(item => 'file:' + item.id === menu.key);
  const openMenu = (key: string, event: MouseEvent<HTMLButtonElement>) => {
    const node = event.currentTarget;
    setMenu(open => open?.key === key ? null : { key, anchor: box(node), returnFocus: node });
  };
  const bound = new Set(files.map(item => item.id));

  return <section className="tw-project"><a className="tw-back" href="/workspace">← 对话</a>
    <div className="tw-project-head"><ProjectTitle name={project.name} editing={editingTitle} onEdit={() => setEditingTitle(true)} onDone={value => void rename(value)} />
      <a className="tw-primary tw-project-new" href={draftHref} aria-label="新对话" onClick={openDraft}><Plus size={14} aria-hidden="true" /><span className="tw-project-new-label">新对话</span></a></div>
    <section className="tw-project-section" aria-labelledby="project-instructions-title"><h2 id="project-instructions-title">项目指令</h2><textarea className="tw-instructions" aria-labelledby="project-instructions-title" maxLength={INSTRUCTIONS_MAX}
      value={instructions} onChange={event => typeInstructions(event.target.value)} onBlur={() => void saveInstructions()} />
      <div className="tw-instructions-foot"><SaveStatus state={saveState} onRetry={() => void saveInstructions()} />{counter && <small className="tw-counter">{counter}</small>}</div>
      <ProjectMemoryToggle projectId={project.id} />
    </section>
    <section className="tw-project-section" aria-labelledby="project-files-title"><h2 id="project-files-title">项目文件</h2>
      {!files.length && !uploads.length && <p className="tw-empty-hint">还没有文件。</p>}
      {(files.length > 0 || uploads.length > 0) && <ul className="tw-project-files">
        {files.map(file => <ProjectFileRow key={file.id} file={file} menuOpen={menu?.key === 'file:' + file.id} onRemove={() => void unbind(file)} onMenu={event => openMenu('file:' + file.id, event)} />)}
        {uploads.map(item => <ProjectFileRow key={item.key} upload={item} onRetry={() => item.file && void upload(item.file, item.key, item.job)} />)}
      </ul>}
      <div className="tw-project-file-actions"><label className="tw-upload"><Upload size={16} aria-hidden="true" />上传文件<input className="tw-sr-only" type="file" aria-label="上传文件"
        onChange={event => { for (const file of event.target.files ?? []) void upload(file); event.target.value = ''; }} /></label>
        <button type="button" className="tw-button" aria-expanded={Boolean(library)} onClick={toggleLibrary}>从资料库添加</button></div>
      {library && <ul className="tw-library-pick" aria-label="从资料库添加">{library.filter(item => !bound.has(item.id)).map(item => <li key={item.id}>
        <button type="button" className="tw-pick-file" onClick={() => void bindFromLibrary(item)}><span className="tw-project-file-name">{item.name}</span><small>{fileMeta({ size: item.bytes, created_at: item.created_at })}</small></button></li>)}
        {!library.some(item => !bound.has(item.id)) && <li><p className="tw-empty-hint">还没有文件。</p></li>}</ul>}
    </section>
    <section className="tw-project-section" aria-labelledby="project-conversations-title"><h2 id="project-conversations-title">对话</h2>
      {conversations.length ? conversations.map(item => <article className="tw-task-row" key={item.id}><div className="tw-task-summary">
        {renaming === item.id ? <input className="tw-rename" aria-label={COPY.renameLabel} defaultValue={item.title} maxLength={120} autoFocus onFocus={event => event.currentTarget.select()}
          onBlur={event => renameTask(item, event.currentTarget.value)} onKeyDown={event => {
            if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur(); }
            if (event.key === 'Escape') { event.preventDefault(); event.currentTarget.value = item.title; setRenaming(''); }
          }} /> : <h3><a href={`/workspace?task=${encodeURIComponent(item.id)}`}>{item.title}</a></h3>}
        <p className="tw-task-line"><span className={statusPillClass(taskStatus(item.status))}>{taskStatus(item.status)}</span><time dateTime={item.updated_at}>{new Date(item.updated_at).toLocaleString('zh-CN', { dateStyle: 'short', timeStyle: 'short' })}</time></p></div>
        <button type="button" className="tw-row-more tw-task-more" aria-haspopup="menu" aria-expanded={menu?.key === 'task:' + item.id} aria-label={COPY.trigger(item.title)}
          onClick={event => openMenu('task:' + item.id, event)}><MoreHorizontal size={24} aria-hidden="true" /></button></article>)
        : <><p className="tw-empty-hint">还没有对话。</p><a className="tw-button" href={draftHref} onClick={openDraft}>新对话</a></>}
    </section>
    {menu && menuTask && <ConversationMenu id="tw-project-conversation-menu" task={menuTask} projects={projects} anchor={menu.anchor} returnFocus={menu.returnFocus}
      onClose={() => setMenu(null)} onCommand={action => command(menuTask, action)} onMove={target => void move(menuTask, target)}
      onCreateProject={async name => {
        const created = await createProject(name).catch(() => null);
        if (!created) { showToast({ message: COPY.saveFailed }); return; }
        setProjects(list => [created, ...list.filter(item => item.id !== created.id)]);
        await move(menuTask, created);
      }} />}
    {menu && menuFile && <PopMenu id="tw-project-file-menu" label={`文件操作：${menuFile.name}`} anchor={menu.anchor} returnFocus={menu.returnFocus} onClose={() => setMenu(null)}>
      <MenuItem icon={<Trash2 size={16} aria-hidden="true" />} label="移出项目" onSelect={() => void unbind(menuFile)} /></PopMenu>}
  </section>;
}
