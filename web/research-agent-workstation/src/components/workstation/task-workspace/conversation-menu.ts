import { conversationGroup, sortTasksByUpdatedAt } from './task-presentation';
import { jsonRequest, type Project, type UserTask } from './types';

/** Touch: hold a history row this long to open its menu (spec §3.3). */
export const LONG_PRESS_MS = 500;
/** Android Chrome fires `contextmenu` right after a long press; ignore it this long so the open menu stays put. */
export const LONG_PRESS_CONTEXTMENU_GUARD_MS = 800;
/** True when a `contextmenu` at `now` is the echo of a long press that opened the menu at `lastLongPressAt`. */
export function shouldIgnoreContextMenu(lastLongPressAt: number | undefined, now: number): boolean {
  if (!lastLongPressAt) return false;
  const elapsed = now - lastLongPressAt;
  return elapsed >= 0 && elapsed < LONG_PRESS_CONTEXTMENU_GUARD_MS;
}
/** A row's touch long-press state; `timer` is set only while the long-press timer is still pending. */
export type LongPress = { timer?: number; x: number; y: number; fired: boolean; at?: number };
/**
 * What a `contextmenu` at `now` does. Some Android versions fire it before our timer: then it claims the press
 * (cancel the timer, mark it fired, stamp `at`) so the timer cannot toggle the menu closed again. Otherwise only the guard applies.
 */
export function contextMenuDecision(press: LongPress, now: number): { open: boolean; clearTimer?: number; press: LongPress } {
  if (press.timer !== undefined) return { open: true, clearTimer: press.timer, press: { ...press, timer: undefined, fired: true, at: now } };
  return { open: !shouldIgnoreContextMenu(press.at, now), press };
}
/** Sidebar search waits this long after the last keystroke (spec §7). */
export const SEARCH_DEBOUNCE_MS = 250;
/** Undo window for delete / archive (spec §5). */
export const TOAST_MS = 5000;

/** Exactly the §9 strings. */
export const COPY = {
  deleted: '已删除', archived: '已归档', undo: '撤销', exportFailed: '导出失败', retry: '重试', saveFailed: '没能保存，请重试',
  moved: (project: string) => `已移到「${project}」`, archivedLink: (count: number) => `已归档（${count}）`,
  noResults: '没有找到', archivedTag: '已归档', projectName: '项目名称', renameLabel: '对话名称',
  trigger: (title: string) => `对话操作：${title}`, menu: (title: string) => `${title} 的操作`, remove: (title: string) => `删除 ${title}`,
} as const;

const byPinnedDesc = (left: UserTask, right: UserTask) => String(right.pinned_at || '').localeCompare(String(left.pinned_at || ''));
const live = (task: UserTask) => !task.archived_at && !task.deleted;

export type SidebarSections = {
  pinned: UserTask[];
  projects: Array<{ project: Project; items: UserTask[] }>;
  groups: Array<{ label: '今天' | '昨天' | '更早'; items: UserTask[] }>;
};

/**
 * 置顶 / 项目 / 今天 / 昨天 / 更早 (spec §2). A pinned conversation in a project shows twice: in 置顶 and first in its
 * folder, because the folder is always the full list of the project. Project conversations never enter the time groups.
 * Every project is a folder, an empty one too (02 §2).
 */
export function sidebarSections(tasks: UserTask[], projects: Project[], now = Date.now()): SidebarSections {
  const visible = sortTasksByUpdatedAt(tasks.filter(live));
  const known = new Set(projects.map(project => project.id));
  const inProject = (task: UserTask) => Boolean(task.project_id && known.has(task.project_id));
  const pinned = visible.filter(task => task.pinned_at).sort(byPinnedDesc);
  const folders = projects.map(project => {
    const items = visible.filter(task => task.project_id === project.id);
    return { project, items: [...items.filter(task => task.pinned_at).sort(byPinnedDesc), ...items.filter(task => !task.pinned_at)] };
  });
  const rest = visible.filter(task => !task.pinned_at && !inProject(task));
  const groups = (['今天', '昨天', '更早'] as const).map(label => ({ label, items: rest.filter(task => conversationGroup(task.updated_at, now) === label) }))
    .filter(group => group.items.length);
  return { pinned, projects: folders, groups };
}

export type MenuEntry = { id: 'rename' | 'pin' | 'move' | 'export' | 'archive' | 'delete'; label: string; separatorBefore?: boolean; danger?: boolean; submenu?: boolean };

/** Fixed order (spec §4.2); only labels flip with state. */
export function menuEntries(task: Pick<UserTask, 'pinned_at' | 'archived_at'>): MenuEntry[] {
  return [
    { id: 'rename', label: '重命名' },
    { id: 'pin', label: task.pinned_at ? '取消置顶' : '置顶' },
    { id: 'move', label: '移到项目', submenu: true },
    { id: 'export', label: '导出' },
    { id: 'archive', label: task.archived_at ? '取消归档' : '归档', separatorBefore: true },
    { id: 'delete', label: '删除', danger: true },
  ];
}

/** Split a snippet so each case-insensitive occurrence of the query can be bolded. */
export function snippetParts(snippet: string, query: string): Array<{ text: string; match: boolean }> {
  const needle = query.trim().replace(/\s+/g, ' ');
  if (!needle) return snippet ? [{ text: snippet, match: false }] : [];
  const parts: Array<{ text: string; match: boolean }> = [];
  const lower = snippet.toLocaleLowerCase();
  const target = needle.toLocaleLowerCase();
  let at = 0;
  for (let hit = lower.indexOf(target); hit >= 0; hit = lower.indexOf(target, at)) {
    if (hit > at) parts.push({ text: snippet.slice(at, hit), match: false });
    parts.push({ text: snippet.slice(hit, hit + needle.length), match: true });
    at = hit + needle.length;
  }
  if (at < snippet.length) parts.push({ text: snippet.slice(at), match: false });
  return parts;
}

/** Client-side fallback when the search endpoint is unavailable: title only, no snippet. */
export function titleMatches(tasks: UserTask[], query: string) {
  const needle = query.trim().toLocaleLowerCase();
  return sortTasksByUpdatedAt(tasks.filter(task => live(task) && task.title.toLocaleLowerCase().includes(needle)));
}

/** Server search, or the title filter over the list as it is when the request fails (never a stale copy). */
export function searchOrTitles(found: Promise<UserTask[]>, current: () => UserTask[], query: string) {
  return found.catch(() => titleMatches(current(), query));
}

export class ConversationError extends Error {
  constructor(readonly status: number) { super('conversation_request_failed'); }
}

export async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, cache: 'no-store' });
  if (!response.ok) throw new ConversationError(response.status);
  return await response.json() as T;
}

const taskPath = (id: string, tail = '') => `/api/assistant/tasks/${encodeURIComponent(id)}${tail}`;
export type TaskPatch = { title?: string; pinned?: boolean; archived?: boolean; project_id?: string | null };

export const patchConversation = (task: Pick<UserTask, 'id' | 'version'>, change: TaskPatch) =>
  call<{ task: UserTask }>(taskPath(task.id), jsonRequest('PATCH', { ...change, version: task.version })).then(result => result.task);
export const deleteConversation = (id: string) => call<{ deleted: boolean }>(taskPath(id), jsonRequest('DELETE'));
export const restoreConversation = (id: string) => call<{ task: UserTask }>(taskPath(id, '/restore'), jsonRequest('POST')).then(result => result.task);
export const readConversation = (id: string) => call<{ task: UserTask }>(taskPath(id)).then(result => result.task);
export const listConversations = (params: Record<string, string>, signal?: AbortSignal) =>
  call<{ tasks: UserTask[] }>(`/api/assistant/tasks?${new URLSearchParams(params)}`, { signal }).then(result => result.tasks);
export const createProject = (name: string) =>
  call<{ project: Project }>('/api/assistant/projects', jsonRequest('POST', { name, idempotency_key: crypto.randomUUID() })).then(result => result.project);

/** RFC 5987 `filename*` first, then the plain fallback. */
export function dispositionFilename(header: string | null, fallback: string) {
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(header || '')?.[1];
  if (encoded) { try { return decodeURIComponent(encoded); } catch { /* fall through */ } }
  return /filename="([^"]+)"/i.exec(header || '')?.[1] || fallback;
}

export async function exportConversation(id: string) {
  const response = await fetch(taskPath(id, '/export?format=md'), { cache: 'no-store' });
  if (!response.ok) throw new ConversationError(response.status);
  const url = URL.createObjectURL(await response.blob());
  const link = Object.assign(document.createElement('a'), { href: url, download: dispositionFilename(response.headers.get('content-disposition'), '对话.md') });
  document.body.append(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
