'use client';
import { useEffect, useLayoutEffect, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { Pencil, Plus, Trash2 } from 'lucide-react';
import { MenuItem, place, type MenuAnchor } from './ConversationMenu';
import { PROJECT_COPY } from './project-folders';
import type { Project } from './types';

export type ProjectCommand = 'rename' | 'new' | 'delete';
const items = (root: HTMLElement | null) => [...root?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []];

/** 02 §2: 重命名 · 新对话 · 删除项目; only 删除项目 carries the danger text (01 §0). */
export function ProjectMenuEntries({ project, onCommand }: { project: Project; onCommand: (command: ProjectCommand) => void }) {
  return <>
    <MenuItem icon={<Pencil size={16} aria-hidden="true" />} label={PROJECT_COPY.rename} onSelect={() => onCommand('rename')} />
    <MenuItem icon={<Plus size={16} aria-hidden="true" />} label={PROJECT_COPY.newChat} onSelect={() => onCommand('new')} />
    <div role="separator" className="tw-menu-sep" />
    <MenuItem icon={<Trash2 size={16} aria-hidden="true" />} label={PROJECT_COPY.remove} danger onSelect={() => onCommand('delete')} />
  </>;
}

/** A one-level anchored menu (01 §4.3 keys); focus goes to the first item and back to the trigger on Esc. */
export function PopMenu({ id, label, anchor, returnFocus, onClose, children }: {
  id: string; label: string; anchor: MenuAnchor; returnFocus: HTMLElement | null; onClose: () => void; children: ReactNode;
}) {
  const menu = useRef<HTMLDivElement>(null);
  const close = (restore = true) => { onClose(); if (restore) returnFocus?.focus(); };
  useLayoutEffect(() => { if (menu.current) place(menu.current, anchor); }, [anchor]);
  useEffect(() => { items(menu.current)[0]?.focus(); }, []);
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!menu.current?.contains(target) && !returnFocus?.contains(target)) close(false);
    };
    const resize = () => close(false);
    document.addEventListener('pointerdown', outside, true);
    window.addEventListener('resize', resize);
    return () => { document.removeEventListener('pointerdown', outside, true); window.removeEventListener('resize', resize); };
  }, []);
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    const list = items(event.currentTarget);
    const at = list.indexOf(document.activeElement as HTMLElement);
    const focus = (index: number) => { event.preventDefault(); list[(index + list.length) % list.length]?.focus(); };
    if (event.key === 'ArrowDown') focus(at + 1);
    else if (event.key === 'ArrowUp') focus(at - 1);
    else if (event.key === 'Home') focus(0);
    else if (event.key === 'End') focus(list.length - 1);
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
    else if (event.key === 'Tab') { event.preventDefault(); close(); }
  }
  return <div ref={menu} id={id} className="tw-menu-pop" role="menu" aria-label={label} onKeyDown={keys}>{children}</div>;
}

export function ProjectMenu({ id, project, anchor, returnFocus, onClose, onCommand }: {
  id: string; project: Project; anchor: MenuAnchor; returnFocus: HTMLElement | null; onClose: () => void; onCommand: (command: ProjectCommand) => void;
}) {
  return <PopMenu id={id} label={PROJECT_COPY.menu} anchor={anchor} returnFocus={returnFocus} onClose={onClose}>
    <ProjectMenuEntries project={project} onCommand={onCommand} />
  </PopMenu>;
}
