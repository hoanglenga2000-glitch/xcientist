'use client';
import { Folder, X } from 'lucide-react';
import type { ReactNode } from 'react';
import type { Project } from './types';

/** 02 §3.5: the draft's project, removable before the first send. Other draft chips (经验库) sit beside it as children. */
export function DraftProjectChip({ project, onRemove, children }: { project?: Pick<Project, 'id' | 'name'>; onRemove: () => void; children?: ReactNode }) {
  return <div className="tw-chips tw-draft-project">{project && <span className="tw-chip"><Folder size={12} aria-hidden="true" /><span className="tw-chip-name">{project.name}</span>
    <button type="button" aria-label={`项目：${project.name}，移出`} onClick={onRemove}><X size={12} aria-hidden="true" /></button></span>}{children}</div>;
}

/** POST tasks body for a draft; `project_id` only when the draft belongs to a project. */
export function draftTaskBody(draft: string, key: string, projectId: string) {
  return { title: draft.trim().split('\n')[0].slice(0, 80), draft, idempotency_key: key, ...(projectId ? { project_id: projectId } : {}) };
}

/** Enter sends; Shift+Enter and IME composition keep typing. */
export const isSendKey = (event: { key: string; shiftKey: boolean; nativeEvent: { isComposing?: boolean } }) =>
  event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing;
