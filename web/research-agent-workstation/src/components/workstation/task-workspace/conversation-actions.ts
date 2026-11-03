'use client';
import type { Dispatch, SetStateAction } from 'react';
import {
  COPY, ConversationError, deleteConversation, exportConversation, patchConversation, readConversation, restoreConversation, type TaskPatch,
} from './conversation-menu';
import { showToast } from './Toast';
import type { UserTask } from './types';

/** Server copies replace ours field by field; the work status stays our own (PATCH answers carry no run state). */
function merge(list: UserTask[], next: UserTask, insert: boolean) {
  if (!list.some(item => item.id === next.id)) return insert ? [next, ...list] : list;
  return list.map(item => item.id === next.id ? { ...item, ...next, status: item.status } : item);
}

type Options = {
  setTasks: Dispatch<SetStateAction<UserTask[]>>; setResults?: Dispatch<SetStateAction<UserTask[] | null>>;
  setArchivedCount: Dispatch<SetStateAction<number>>; selection: string; onLeaveOpen: () => void;
};

/** The 01 menu actions on a conversation, shared by the sidebar and the project page. */
export function useConversationActions({ setTasks, setResults = () => undefined, setArchivedCount, selection, onLeaveOpen }: Options) {
  const local = (id: string, fields: Partial<UserTask>) => {
    const apply = (items: UserTask[]) => items.map(item => item.id === id ? { ...item, ...fields } : item);
    setTasks(apply); setResults(current => current && apply(current));
  };
  const accept = (next: UserTask) => {
    setTasks(current => merge(current, next, !next.archived_at)); setResults(current => current && merge(current, next, false));
  };
  async function refresh(id: string) {
    try { accept(await readConversation(id)); }
    catch { setTasks(current => current.filter(item => item.id !== id)); }
  }
  /** Optimistic write. Failure rolls back with 没能保存，请重试; a 409 also re-reads that row. */
  async function save(task: UserTask, change: TaskPatch, fields: Partial<UserTask>) {
    local(task.id, fields);
    try {
      const next = await patchConversation(task, change);
      accept(next);
      return next;
    } catch (cause) {
      local(task.id, task);
      showToast({ message: COPY.saveFailed });
      if (cause instanceof ConversationError && cause.status === 409) void refresh(task.id);
      return null;
    }
  }
  async function archive(task: UserTask) {
    const archiving = !task.archived_at;
    const next = await save(task, { archived: archiving }, { archived_at: archiving ? new Date().toISOString() : null, archived: archiving });
    if (!next) return;
    setArchivedCount(count => Math.max(0, count + (archiving ? 1 : -1)));
    if (archiving) showToast({ message: COPY.archived, action: { label: COPY.undo, run: () => void archive(next) } });
  }
  async function remove(task: UserTask) {
    local(task.id, { deleted: true });
    try { await deleteConversation(task.id); }
    catch { local(task.id, { deleted: false }); showToast({ message: COPY.saveFailed }); return; }
    if (task.id === selection) onLeaveOpen();
    showToast({ message: COPY.deleted, action: { label: COPY.undo, run: () => void restoreConversation(task.id)
      .then(restored => { accept({ ...restored, deleted: false }); local(task.id, { deleted: false }); })
      .catch(() => showToast({ message: COPY.saveFailed })) } });
  }
  function download(id: string) {
    void exportConversation(id).catch(() => showToast({ message: COPY.exportFailed, action: { label: COPY.retry, run: () => download(id) } }));
  }
  return { local, accept, refresh, save, archive, remove, download };
}
