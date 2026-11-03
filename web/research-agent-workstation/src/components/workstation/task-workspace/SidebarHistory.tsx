'use client';
import { useEffect, useRef, useState, type Dispatch, type KeyboardEvent, type MouseEvent, type SetStateAction, type TouchEvent } from 'react';
import { ChevronRight, Folder, MoreHorizontal, Pin, Plus } from 'lucide-react';
import { ConversationMenu, type MenuAnchor, type MenuCommand } from './ConversationMenu';
import {
  COPY, contextMenuDecision, createProject, LONG_PRESS_MS, listConversations, type LongPress, SEARCH_DEBOUNCE_MS, searchOrTitles, sidebarSections, snippetParts,
} from './conversation-menu';
import { useConversationActions } from './conversation-actions';
import { ProjectMenu, type ProjectCommand } from './ProjectMenu';
import { folderConvoHintId, folderExpandHintId, folderExpanded, folderMoreHintId, folderRows, folderTitleId, patchProject, PROJECT_COPY, PROJECT_EXPAND_KEY, projectDraftHref, projectHref, readExpand, writeExpand } from './project-folders';
import { isAwaitingConfirm, isLiveTask } from './task-presentation';
import { showToast } from './Toast';
import type { Project, UserTask } from './types';

type Props = {
  tasks: UserTask[]; setTasks: Dispatch<SetStateAction<UserTask[]>>; projects: Project[]; setProjects: Dispatch<SetStateAction<Project[]>>;
  archivedCount: number; setArchivedCount: Dispatch<SetStateAction<number>>; selection: string; query: string; onLeaveOpen: () => void;
  /** In-app navigation (pushState); a plain link load when absent. */
  onNavigate?: (href: string) => void;
  /** The project page that is open (`?project=`), and the open conversation's project when the list does not carry it. */
  openProject?: string; currentProject?: string;
  onDeleteProject?: (project: Project, trigger: HTMLElement | null) => void;
};
type FolderMenu = { project: Project; anchor: MenuAnchor; returnFocus: HTMLElement | null };
const FOLDER_MENU_ID = 'tw-project-menu';
type OpenMenu = { key: string; task: UserTask; anchor: MenuAnchor; returnFocus: HTMLElement | null };
const MENU_ID = 'tw-conversation-menu';
const point = (x: number, y: number): MenuAnchor => ({ left: x, top: y, right: x, bottom: y });
const box = (node: Element): MenuAnchor => { const rect = node.getBoundingClientRect(); return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom }; };

export function SidebarHistory({ tasks, setTasks, projects, setProjects, archivedCount, setArchivedCount, selection, query, onLeaveOpen, onNavigate, openProject = '', currentProject = '', onDeleteProject }: Props) {
  const [results, setResults] = useState<UserTask[] | null>(null);
  const [menu, setMenu] = useState<OpenMenu | null>(null);
  const [editing, setEditing] = useState('');
  const [folderMenu, setFolderMenu] = useState<FolderMenu | null>(null);
  const [stored, setStored] = useState<Record<string, boolean>>({});
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const [naming, setNaming] = useState(false);
  const [roving, setRoving] = useState('');
  const list = useRef<HTMLDivElement>(null);
  const press = useRef<LongPress>({ x: 0, y: 0, fired: false });
  useEffect(() => () => window.clearTimeout(press.current.timer), []);
  const focusAfter = useRef('');
  useEffect(() => { setStored(readExpand(window.localStorage.getItem(PROJECT_EXPAND_KEY))); }, []);
  const searching = query.trim();
  const currentTasks = useRef(tasks);
  currentTasks.current = tasks;

  useEffect(() => {
    if (!searching) { setResults(null); return; }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      // Endpoint down: keep the old title filter, silently.
      void searchOrTitles(listConversations({ q: searching }, controller.signal), () => currentTasks.current, searching)
        .then(found => { if (!controller.signal.aborted) setResults(found); });
    }, SEARCH_DEBOUNCE_MS);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [searching]);
  useEffect(() => {
    if (!focusAfter.current || editing) return;
    list.current?.querySelector<HTMLElement>(`a[data-row="${focusAfter.current}"]`)?.focus();
    focusAfter.current = '';
  }, [editing]);

  const { save, archive, remove, download } = useConversationActions({ setTasks, setResults, setArchivedCount, selection, onLeaveOpen });
  async function move(task: UserTask, project: Project | null) {
    setMenu(null);
    const next = await save(task, { project_id: project?.id ?? null }, { project_id: project?.id ?? null });
    if (next && project) showToast({ message: COPY.moved(project.name) });
  }
  function go(event: MouseEvent, href: string) {
    if (!onNavigate || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onNavigate(href);
  }
  /** Writes only what the user clicked; the forced open of the current project is never stored (L2). */
  function toggle(id: string, open: boolean) {
    setToggled(current => ({ ...current, [id]: open }));
    const next = writeExpand(readExpand(window.localStorage.getItem(PROJECT_EXPAND_KEY)), id, open);
    setStored(next);
    try { window.localStorage.setItem(PROJECT_EXPAND_KEY, JSON.stringify(next)); } catch { /* private mode: this page still remembers */ }
  }
  async function newProject(value: string) {
    const name = value.trim();
    if (!name) { setNaming(false); return; }
    const project = await createProject(name).catch(() => null);
    if (!project) { showToast({ message: COPY.saveFailed }); return; }
    setNaming(false);
    setProjects(current => [project, ...current.filter(item => item.id !== project.id)]);
    if (onNavigate) onNavigate(projectHref(project.id)); else window.location.assign(projectHref(project.id));
  }
  async function renameProject(project: Project, value: string) {
    setEditing('');
    const name = value.trim();
    if (!name || name === project.name) return;
    const local = (fields: Partial<Project>) => setProjects(current => current.map(item => item.id === project.id ? { ...item, ...fields } : item));
    local({ name });
    try { local(await patchProject(project, { name })); }
    catch { local(project); showToast({ message: COPY.saveFailed }); }
  }
  function projectCommand(project: Project, action: ProjectCommand, trigger: HTMLElement | null) {
    setFolderMenu(null);
    if (action === 'rename') setEditing('folder:' + project.id);
    else if (action === 'new') { if (onNavigate) onNavigate(projectDraftHref(project.id)); else window.location.assign(projectDraftHref(project.id)); }
    else onDeleteProject?.(project, trigger);
  }
  function openFolderMenu(project: Project, anchor: MenuAnchor, returnFocus: HTMLElement | null) {
    setMenu(null);
    setFolderMenu(current => current?.project.id === project.id ? null : { project, anchor, returnFocus });
  }
  function folderContextMenu(event: MouseEvent<HTMLAnchorElement>, project: Project) {
    event.preventDefault();
    const decision = contextMenuDecision(press.current, Date.now());
    window.clearTimeout(decision.clearTimer);
    press.current = decision.press;
    if (decision.open) openFolderMenu(project, point(event.clientX, event.clientY), event.currentTarget);
  }
  function folderTouch(event: TouchEvent<HTMLAnchorElement>, project: Project) {
    touch.start(event, (anchor, node) => openFolderMenu(project, anchor, node));
  }
  function command(task: UserTask, key: string, action: MenuCommand) {
    setMenu(null);
    if (action === 'rename') { focusAfter.current = key; setEditing(key); }
    else if (action === 'pin') void save(task, { pinned: !task.pinned_at }, { pinned_at: task.pinned_at ? null : new Date().toISOString() });
    else if (action === 'export') download(task.id);
    else if (action === 'archive') void archive(task);
    else void remove(task);
  }
  function rename(task: UserTask, value: string) {
    setEditing('');
    const title = value.trim();
    if (title && title !== task.title) void save(task, { title }, { title });
  }

  function openMenu(key: string, task: UserTask, anchor: MenuAnchor, returnFocus: HTMLElement | null) {
    setFolderMenu(null);
    setMenu(current => current?.key === key ? null : { key, task, anchor, returnFocus });
  }
  function rowKeys(event: KeyboardEvent<HTMLAnchorElement>, key: string, task: UserTask) {
    const links = [...list.current?.querySelectorAll<HTMLAnchorElement>('a[data-row]') ?? []];
    const at = links.indexOf(event.currentTarget);
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = links[(at + (event.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length];
      next?.focus();
    } else if (event.key === 'F2') { event.preventDefault(); focusAfter.current = key; setEditing(key); }
    else if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) {
      event.preventDefault();
      const rect = box(event.currentTarget);
      openMenu(key, task, point(rect.left + 12, rect.bottom), event.currentTarget);
    }
  }
  const touch = {
    start(event: TouchEvent<HTMLAnchorElement>, open: (anchor: MenuAnchor, node: HTMLElement) => void) {
      const node = event.currentTarget;
      const { clientX: x, clientY: y } = event.touches[0];
      window.clearTimeout(press.current.timer);
      press.current = { x, y, fired: false, timer: window.setTimeout(() => { press.current.fired = true; press.current.at = Date.now(); press.current.timer = undefined; open(point(x, y), node); }, LONG_PRESS_MS) };
    },
    move(event: TouchEvent) {
      const { clientX: x, clientY: y } = event.touches[0];
      if (Math.hypot(x - press.current.x, y - press.current.y) > 10) { window.clearTimeout(press.current.timer); press.current.timer = undefined; }
    },
    end(event: TouchEvent) { window.clearTimeout(press.current.timer); press.current.timer = undefined; if (press.current.fired) event.preventDefault(); },
  };
  function contextMenu(event: MouseEvent<HTMLAnchorElement>, key: string, task: UserTask) {
    event.preventDefault();
    const decision = contextMenuDecision(press.current, Date.now());
    window.clearTimeout(decision.clearTimer);
    press.current = decision.press;
    if (decision.open) openMenu(key, task, point(event.clientX, event.clientY), event.currentTarget);
  }
  function click(event: MouseEvent) { if (press.current.fired) { event.preventDefault(); press.current.fired = false; } }

  const latest = (task: UserTask) => tasks.find(item => item.id === task.id) ?? results?.find(item => item.id === task.id) ?? task;
  const view = results ? null : sidebarSections(tasks, projects);
  const shown = results?.filter(item => !item.deleted) ?? [];
  const forced = currentProject || tasks.find(item => item.id === selection)?.project_id || '';
  const isOpen = (id: string) => folderExpanded(id, stored, toggled, forced);
  const keys = results ? shown.map(item => 'result:' + item.id)
    : [...view!.pinned.map(item => 'pinned:' + item.id), ...view!.projects.flatMap(folder => isOpen(folder.project.id) ? folderRows(folder.items).shown.map(item => `project:${item.id}`) : []),
      ...view!.groups.flatMap(group => group.items.map(item => 'time:' + item.id))];
  const active = keys.includes(roving) ? roving : keys.find(key => key.endsWith(':' + selection)) ?? keys[0];

  function row(task: UserTask, key: string, options: { pinMark?: boolean; result?: boolean } = {}) {
    const open = menu?.key === key;
    const stop = key === active ? 0 : -1;
    if (editing === key) {
      return <div className="tw-history-row is-editing" key={key}><input className="tw-rename" aria-label={COPY.renameLabel} defaultValue={task.title} maxLength={120} autoFocus
        onFocus={event => event.currentTarget.select()} onBlur={event => rename(task, event.currentTarget.value)}
        onKeyDown={event => {
          if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur(); }
          if (event.key === 'Escape') { event.preventDefault(); event.currentTarget.value = task.title; setEditing(''); }
        }} /></div>;
    }
    return <div className={`tw-history-row${open ? ' is-open' : ''}${options.result ? ' is-result' : ''}`} key={key}>
      <a href={`/workspace?task=${encodeURIComponent(task.id)}`} data-row={key} tabIndex={stop} title={task.title} aria-current={task.id === selection ? 'page' : undefined}
        onFocus={() => setRoving(key)} onKeyDown={event => rowKeys(event, key, task)} onClick={click}
        onContextMenu={event => contextMenu(event, key, task)}
        onTouchStart={event => touch.start(event, (anchor, node) => openMenu(key, task, anchor, node))} onTouchMove={touch.move} onTouchEnd={touch.end} onTouchCancel={touch.end}>
        {options.result ? <SearchResult task={task} query={searching} /> : <>{options.pinMark && task.pinned_at && <Pin size={12} className="tw-row-pin" aria-hidden="true" />}<span className="tw-row-title">{task.title}</span></>}
      </a>
      {isLiveTask(task.status) && <span className="tw-dot is-current tw-row-live" aria-hidden="true" />}
      {isAwaitingConfirm(task.status) && <span className="tw-confirm-mark tw-row-live" aria-hidden="true">!</span>}
      <button type="button" className="tw-row-more" tabIndex={stop} aria-haspopup="menu" aria-expanded={open} aria-controls={open ? MENU_ID : undefined}
        aria-label={COPY.trigger(task.title)} onClick={event => openMenu(key, task, box(event.currentTarget), event.currentTarget)}><MoreHorizontal size={24} aria-hidden="true" /></button>
    </div>;
  }

  function folder(project: Project, items: UserTask[]) {
    const open = isOpen(project.id);
    const menuOpen = folderMenu?.project.id === project.id;
    const groupId = `tw-folder-${project.id}`;
    const { shown: rows, more } = folderRows(items);
    if (editing === 'folder:' + project.id) {
      return <div className="tw-folder" key={project.id}><div className="tw-history-row is-editing"><input className="tw-rename" aria-label={PROJECT_COPY.name} defaultValue={project.name} maxLength={120} autoFocus
        onFocus={event => event.currentTarget.select()} onBlur={event => void renameProject(project, event.currentTarget.value)}
        onKeyDown={event => {
          if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur(); }
          if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); event.currentTarget.value = project.name; setEditing(''); }
        }} /></div></div>;
    }
    const titleId = folderTitleId(project.id);
    const expandHint = folderExpandHintId(project.id);
    const moreHint = folderMoreHintId(project.id);
    const convoHint = folderConvoHintId(project.id);
    return <div className="tw-folder" key={project.id}>
      <div className={`tw-folder-row${menuOpen ? ' is-open' : ''}`}>
        <span id={expandHint} className="tw-sr-only">{PROJECT_COPY.expandHint}</span>
        <button type="button" className="tw-folder-chevron" aria-expanded={open} aria-controls={groupId} aria-labelledby={`${expandHint} ${titleId}`} onClick={() => toggle(project.id, !open)}><ChevronRight size={12} aria-hidden="true" /></button>
        <a href={projectHref(project.id)} className="tw-folder-link" title={project.name} aria-current={openProject === project.id ? 'page' : undefined}
          onClick={event => { click(event); go(event, projectHref(project.id)); }}
          onKeyDown={event => {
            if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) { event.preventDefault(); const rect = box(event.currentTarget); openFolderMenu(project, point(rect.left + 12, rect.bottom), event.currentTarget); }
            else if (event.key === 'F2') { event.preventDefault(); setEditing('folder:' + project.id); }
          }}
          onContextMenu={event => folderContextMenu(event, project)}
          onTouchStart={event => folderTouch(event, project)} onTouchMove={touch.move} onTouchEnd={touch.end} onTouchCancel={touch.end}>
          <Folder size={16} aria-hidden="true" /><span id={titleId} className="tw-row-title">{project.name}</span>{items.length > 0 && <small className="tw-folder-count">{items.length}</small>}</a>
        <span id={moreHint} className="tw-sr-only">{PROJECT_COPY.moreHint}</span>
        <button type="button" className="tw-row-more" aria-haspopup="menu" aria-expanded={menuOpen} aria-controls={menuOpen ? FOLDER_MENU_ID : undefined}
          aria-labelledby={`${moreHint} ${titleId}`} onClick={event => openFolderMenu(project, box(event.currentTarget), event.currentTarget)}><MoreHorizontal size={24} aria-hidden="true" /></button>
      </div>
      {open && <div className="tw-folder-items" id={groupId} role="group" aria-labelledby={`${titleId} ${convoHint}`}>
        <span id={convoHint} className="tw-sr-only">{PROJECT_COPY.convoHint}</span>
        {rows.map(item => row(item, 'project:' + item.id, { pinMark: true }))}
        {more && <a className="tw-folder-all" href={projectHref(project.id)} onClick={event => go(event, projectHref(project.id))}>{PROJECT_COPY.all(items.length)}</a>}
      </div>}
    </div>;
  }

  return <div className="tw-history" ref={list}>
    {results ? <div className="tw-history-group">{shown.length ? shown.map(item => row(item, 'result:' + item.id, { result: true })) : <p className="tw-history-empty">{COPY.noResults}</p>}</div> : <>
      {view!.pinned.length > 0 && <div className="tw-history-group"><p className="tw-history-label">置顶</p>{view!.pinned.map(item => row(item, 'pinned:' + item.id))}</div>}
      <div className="tw-history-group"><div className="tw-history-head"><p className="tw-history-label">{PROJECT_COPY.group}</p><button type="button" className="tw-group-add" aria-label={PROJECT_COPY.create} onClick={() => setNaming(true)}><Plus size={14} aria-hidden="true" /></button></div>
        {naming && <div className="tw-history-row is-editing"><input className="tw-rename" aria-label={PROJECT_COPY.create} placeholder={PROJECT_COPY.name} maxLength={120} autoFocus
          onKeyDown={event => {
            if (event.key === 'Enter') { event.preventDefault(); void newProject(event.currentTarget.value); }
            if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setNaming(false); }
          }} onBlur={event => { if (!event.currentTarget.value.trim()) setNaming(false); }} /></div>}
        {view!.projects.map(({ project, items }) => folder(project, items))}</div>
      {view!.groups.map(group => <div className="tw-history-group" key={group.label}><p className="tw-history-label">{group.label}</p>{group.items.map(item => row(item, 'time:' + item.id))}</div>)}
      {archivedCount > 0 && <a className="tw-archived-link" href="/workspace?settings=archived">{COPY.archivedLink(archivedCount)}</a>}
    </>}
    {menu && <ConversationMenu id={MENU_ID} task={latest(menu.task)}
      projects={projects} anchor={menu.anchor} returnFocus={menu.returnFocus} onClose={() => setMenu(null)}
      onCommand={action => command(latest(menu.task), menu.key, action)}
      onMove={project => void move(latest(menu.task), project)}
      onCreateProject={async name => {
        const project = await createProject(name).catch(() => null);
        if (!project) { showToast({ message: COPY.saveFailed }); return; }
        setProjects(current => [project, ...current.filter(item => item.id !== project.id)]);
        await move(latest(menu.task), project);
      }} />}
    {folderMenu && <ProjectMenu id={FOLDER_MENU_ID} project={projects.find(item => item.id === folderMenu.project.id) ?? folderMenu.project} anchor={folderMenu.anchor}
      returnFocus={folderMenu.returnFocus} onClose={() => setFolderMenu(null)}
      onCommand={action => projectCommand(projects.find(item => item.id === folderMenu.project.id) ?? folderMenu.project, action, folderMenu.returnFocus)} />}
  </div>;
}

export function SearchResult({ task, query }: { task: UserTask; query: string }) {
  return <><span className="tw-result-head"><span className="tw-row-title">{task.title}</span>{task.archived_at && <span className="tw-tag">{COPY.archivedTag}</span>}</span>
    {task.snippet && <span className="tw-result-snippet">{snippetParts(task.snippet, query).map((part, index) => part.match ? <strong key={index}>{part.text}</strong> : <span key={index}>{part.text}</span>)}</span>}</>;
}
