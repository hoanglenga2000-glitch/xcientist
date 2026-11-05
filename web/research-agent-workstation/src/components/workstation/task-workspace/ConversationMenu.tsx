'use client';
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Archive, ArchiveRestore, Check, ChevronLeft, ChevronRight, Download, FolderInput, Pencil, Pin, PinOff, Plus, Trash2 } from 'lucide-react';
import { COPY, menuEntries, type MenuEntry } from './conversation-menu';
import type { Project, UserTask } from './types';

export type MenuAnchor = { left: number; top: number; right: number; bottom: number };
export type MenuCommand = Exclude<MenuEntry['id'], 'move'>;
type Props = {
  id: string; task: UserTask; projects: Project[]; anchor: MenuAnchor; returnFocus: HTMLElement | null;
  onClose: () => void; onCommand: (command: MenuCommand) => void; onMove: (project: Project | null) => void; onCreateProject: (name: string) => Promise<void>;
};

const GAP = 8;
const ICONS = { rename: Pencil, pin: Pin, unpin: PinOff, move: FolderInput, export: Download, archive: Archive, unarchive: ArchiveRestore, delete: Trash2 };
const narrow = () => typeof window !== 'undefined' && window.matchMedia('(max-width:760px)').matches;
const items = (root: HTMLElement | null) => [...root?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not([aria-disabled="true"])') ?? []];

/** Place a fixed popover inside the viewport: below the anchor, flipped up when there is no room (spec §3.3/§3.4). */
export function place(node: HTMLElement, anchor: MenuAnchor) {
  const { width, height } = node.getBoundingClientRect();
  const left = Math.max(GAP, Math.min(anchor.left, window.innerWidth - width - GAP));
  const below = anchor.bottom + 4;
  const top = below + height <= window.innerHeight - GAP ? below : Math.max(GAP, anchor.top - height - 4);
  Object.assign(node.style, { left: `${left}px`, top: `${top}px` });
}

/** Desktop submenu: to the right of the menu overlapping 4px, or to the left when the right would clip. */
function placeSubmenu(node: HTMLElement, menu: HTMLElement, item: HTMLElement) {
  const { width, height } = node.getBoundingClientRect();
  const parent = menu.getBoundingClientRect();
  const right = parent.right - 4;
  const left = right + width <= window.innerWidth - GAP ? right : Math.max(GAP, parent.left - width + 4);
  const top = Math.max(GAP, Math.min(item.getBoundingClientRect().top - 4, window.innerHeight - height - GAP));
  Object.assign(node.style, { left: `${left}px`, top: `${top}px` });
}

export function MenuItem({ icon, label, onSelect, danger, ariaLabel, submenu, expanded, checked, disabled, role = 'menuitem' }: {
  icon?: ReactNode; label: string; onSelect: () => void; danger?: boolean; ariaLabel?: string; submenu?: boolean; expanded?: boolean;
  checked?: boolean; disabled?: boolean; role?: 'menuitem' | 'menuitemradio';
}) {
  return <button type="button" role={role} tabIndex={-1} className={`tw-menu-item${danger ? ' is-danger' : ''}`} aria-label={ariaLabel}
    aria-haspopup={submenu ? 'menu' : undefined} aria-expanded={submenu ? Boolean(expanded) : undefined}
    aria-checked={role === 'menuitemradio' ? Boolean(checked) : undefined} aria-disabled={disabled || undefined}
    onClick={() => { if (!disabled) onSelect(); }}>
    {icon}<span className="tw-menu-label">{label}</span>{submenu && <ChevronRight size={16} aria-hidden="true" className="tw-menu-end" />}
    {checked && <Check size={16} aria-hidden="true" className="tw-menu-end" />}
  </button>;
}

/** The top-level entries (spec §4.2), shared by the live menu and the static tests. */
export function MenuEntries({ task, subOpen, onCommand, onOpenSubmenu }: { task: UserTask; subOpen: boolean; onCommand: (command: MenuCommand) => void; onOpenSubmenu: () => void }) {
  return <>{menuEntries(task).map(entry => {
    const key = entry.id === 'pin' && task.pinned_at ? 'unpin' : entry.id === 'archive' && task.archived_at ? 'unarchive' : entry.id;
    const Icon = ICONS[key];
    return <div key={entry.id} role="none" className="tw-menu-row">{entry.separatorBefore && <div role="separator" className="tw-menu-sep" />}
      <MenuItem icon={<Icon size={16} aria-hidden="true" />} label={entry.label} danger={entry.danger} submenu={entry.submenu} expanded={entry.submenu && subOpen}
        ariaLabel={entry.danger ? COPY.remove(task.title) : undefined}
        onSelect={() => entry.id === 'move' ? onOpenSubmenu() : onCommand(entry.id as MenuCommand)} /></div>;
  })}</>;
}

export function ProjectEntries({ task, projects, inline, onBack, onMove, onCreateProject }: {
  task: UserTask; projects: Project[]; inline: boolean; onBack: () => void; onMove: (project: Project | null) => void; onCreateProject: (name: string) => Promise<void>;
}) {
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { if (naming) input.current?.focus(); }, [naming]);
  async function create() {
    if (busy || !name.trim()) return;
    setBusy(true);
    try { await onCreateProject(name.trim()); } finally { setBusy(false); }
  }
  return <>
    {inline && <MenuItem icon={<ChevronLeft size={16} aria-hidden="true" />} label="移到项目" onSelect={onBack} />}
    {task.project_id && <MenuItem label="移出项目" onSelect={() => onMove(null)} />}
    <div role="none" className="tw-menu-scroll">{projects.map(project => <MenuItem key={project.id} role="menuitemradio" label={project.name}
      checked={project.id === task.project_id} disabled={project.id === task.project_id} onSelect={() => onMove(project)} />)}</div>
    <div role="separator" className="tw-menu-sep" />
    {naming ? <input ref={input} className="tw-menu-input" aria-label="新建项目" placeholder={COPY.projectName} value={name} maxLength={120} disabled={busy}
      onChange={event => setName(event.target.value)}
      onKeyDown={event => {
        event.stopPropagation();
        if (event.key === 'Enter') { event.preventDefault(); void create(); }
        if (event.key === 'Escape') { event.preventDefault(); setNaming(false); setName(''); }
      }} />
      : <MenuItem icon={<Plus size={16} aria-hidden="true" />} label="新建项目" onSelect={() => setNaming(true)} />}
  </>;
}

export function ConversationMenu({ id, task, projects, anchor, returnFocus, onClose, onCommand, onMove, onCreateProject }: Props) {
  const menu = useRef<HTMLDivElement>(null);
  const submenu = useRef<HTMLDivElement>(null);
  const [sub, setSub] = useState(false);
  const [inline] = useState(narrow);
  const close = (restore = true) => { onClose(); if (restore) returnFocus?.focus(); };
  const moveItem = () => menu.current?.querySelector<HTMLElement>('[aria-haspopup="menu"]') ?? null;

  useLayoutEffect(() => { if (menu.current) place(menu.current, anchor); }, [anchor, sub]);
  useLayoutEffect(() => {
    const item = moveItem();
    if (sub && !inline && submenu.current && menu.current && item) placeSubmenu(submenu.current, menu.current, item);
  }, [sub, inline, projects.length]);
  useEffect(() => { items(sub && !inline ? submenu.current : menu.current)[0]?.focus(); }, [sub]);
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      // The trigger itself toggles the menu on click.
      if (!menu.current?.contains(target) && !submenu.current?.contains(target) && !returnFocus?.contains(target)) close(false);
    };
    const resize = () => close(false);
    document.addEventListener('pointerdown', outside, true);
    window.addEventListener('resize', resize);
    return () => { document.removeEventListener('pointerdown', outside, true); window.removeEventListener('resize', resize); };
  }, []);

  function keys(event: KeyboardEvent<HTMLDivElement>, inSubmenu: boolean) {
    const list = items(event.currentTarget);
    const at = list.indexOf(document.activeElement as HTMLElement);
    const focus = (index: number) => { event.preventDefault(); list[(index + list.length) % list.length]?.focus(); };
    if (event.key === 'ArrowDown') focus(at + 1);
    else if (event.key === 'ArrowUp') focus(at - 1);
    else if (event.key === 'Home') focus(0);
    else if (event.key === 'End') focus(list.length - 1);
    else if (event.key === 'ArrowRight' && !inSubmenu && document.activeElement === moveItem()) { event.preventDefault(); setSub(true); }
    else if (event.key === 'ArrowLeft' && (inSubmenu || (inline && sub))) { event.preventDefault(); back(); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
    else if (event.key === 'Tab') { event.preventDefault(); close(); }
  }
  function back() { setSub(false); window.requestAnimationFrame(() => moveItem()?.focus()); }

  const projectEntries = <ProjectEntries task={task} projects={projects} inline={inline} onBack={back} onMove={onMove} onCreateProject={onCreateProject} />;
  return <>
    <div ref={menu} id={id} className="tw-menu-pop" role="menu" aria-label={COPY.menu(task.title)} onKeyDown={event => keys(event, false)}>
      {sub && inline ? projectEntries : <MenuEntries task={task} subOpen={sub} onCommand={onCommand} onOpenSubmenu={() => setSub(true)} />}
    </div>
    {sub && !inline && <div ref={submenu} className="tw-menu-pop tw-submenu" role="menu" aria-label="移到项目" onKeyDown={event => keys(event, true)}>{projectEntries}</div>}
  </>;
}
