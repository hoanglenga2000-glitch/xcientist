'use client';
import { useEffect, useRef, type KeyboardEvent } from 'react';
import type { Project } from './types';
import { useReturnFocus } from './focus-return';

/** Exactly the 02 §7 strings. */
export const PROJECT_DELETE_COPY = {
  title: (name: string) => `删除项目「${name}」？`, line: (count: number) => `${count} 个对话会移回对话列表，文件仍在资料里。`,
  cancel: '取消', confirm: '删除项目', toast: '已删除项目',
} as const;

/** 02 §5: the only confirm in the sidebar flows. 取消 is focused first, Tab stays inside, Esc cancels; no red anywhere. */
export function DeleteProjectDialog({ project, count, onCancel, onConfirm }: { project: Pick<Project, 'id' | 'name'>; count: number; onCancel: () => void; onConfirm: () => void }) {
  useReturnFocus();
  const cancel = useRef<HTMLButtonElement>(null);
  const confirm = useRef<HTMLButtonElement>(null);
  useEffect(() => { cancel.current?.focus(); }, []);
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onCancel(); }
    else if (event.key === 'Tab') {
      event.preventDefault();
      (document.activeElement === cancel.current ? confirm.current : cancel.current)?.focus();
    }
  }
  return <div className="tw-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); onCancel(); } }}>
    <div className="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-delete-project-title" aria-describedby="tw-delete-project-line" onKeyDown={keys}>
      <h2 id="tw-delete-project-title">{PROJECT_DELETE_COPY.title(project.name)}</h2><p id="tw-delete-project-line">{PROJECT_DELETE_COPY.line(count)}</p>
      <div className="tw-dialog-actions"><button ref={cancel} type="button" className="tw-button" onClick={onCancel}>{PROJECT_DELETE_COPY.cancel}</button>
        <button ref={confirm} type="button" className="tw-button" onClick={onConfirm}>{PROJECT_DELETE_COPY.confirm}</button></div>
    </div></div>;
}
