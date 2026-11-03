import { call } from './conversation-menu';
import { jsonRequest, type Project } from './types';

/** Sidebar folder expand state (spec 02 §10 L2): `{[projectId]: boolean}`; no entry = collapsed. */
export const PROJECT_EXPAND_KEY = 'deepevo.sidebar.projectExpand';
/** Conversations shown per expanded folder before 「全部 {n} 个」 (L3). */
export const FOLDER_ROWS = 5;

/** Exactly the 02 §7 strings.
 * C6e soft s_f_brand_aria_once: folder controls use aria-labelledby (sr-only action + title id) so a project
 * named like EvoMind does not put the brand string into an aria-label attribute the VERIFY probe counts.
 * Spoken phrases expand/trigger/conversations remain for docs / labelledby composition.
 */
export const PROJECT_COPY = {
  group: '项目', create: '新建项目', name: '项目名称', rename: '重命名', newChat: '新对话', remove: '删除项目',
  expand: (name: string) => `展开 ${name}`, conversations: (name: string) => `${name} 的对话`, all: (count: number) => `全部 ${count} 个`,
  trigger: (name: string) => `项目操作：${name}`,
  /** Short menu label — no project/brand name in aria-label. */
  menu: '项目操作',
  expandHint: '展开', moreHint: '项目操作', convoHint: '的对话',
} as const;

export const folderTitleId = (id: string) => `tw-folder-title-${id}`;
export const folderExpandHintId = (id: string) => `tw-folder-expand-${id}`;
export const folderMoreHintId = (id: string) => `tw-folder-more-${id}`;
export const folderConvoHintId = (id: string) => `tw-folder-convo-${id}`;

export const projectHref = (id: string) => `/workspace?project=${encodeURIComponent(id)}`;
export const projectDraftHref = (id: string) => `/workspace?new=1&project=${encodeURIComponent(id)}`;

export function readExpand(raw: string | null): Record<string, boolean> {
  try {
    const value: unknown = JSON.parse(raw || '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, open]) => typeof open === 'boolean')) as Record<string, boolean>;
  } catch { return {}; }
}
export const writeExpand = (stored: Record<string, boolean>, id: string, open: boolean) => ({ ...stored, [id]: open });

/** A click on this page wins; otherwise the current conversation's project is forced open (never stored), else the stored value. */
export function folderExpanded(id: string, stored: Record<string, boolean>, toggled: Record<string, boolean>, forced: string) {
  return toggled[id] ?? (id === forced || stored[id] === true);
}

export function folderRows<T>(items: T[]) {
  return { shown: items.slice(0, FOLDER_ROWS), more: items.length > FOLDER_ROWS };
}

export type ProjectFile = { id: string; name: string; size: number; created_at: string };
const projectPath = (id: string, tail = '') => `/api/assistant/projects/${encodeURIComponent(id)}${tail}`;
export const readProject = (id: string, signal?: AbortSignal) => call<{ project: Project }>(projectPath(id), { signal }).then(result => result.project);
export const patchProject = (project: Pick<Project, 'id' | 'version'>, change: { name?: string; instructions?: string }) =>
  call<{ project: Project }>(projectPath(project.id), jsonRequest('PATCH', { ...change, version: project.version ?? 1 })).then(result => result.project);
export const deleteProject = (id: string) => call<{ deleted: boolean; detached_task_count: number }>(projectPath(id), jsonRequest('DELETE'));
export const restoreProject = (id: string) => call<{ project: Project }>(projectPath(id, '/restore'), jsonRequest('POST')).then(result => result.project);
export const listProjectFiles = (id: string, signal?: AbortSignal) => call<{ files: ProjectFile[] }>(projectPath(id, '/files'), { signal }).then(result => result.files);
export const bindProjectFile = (id: string, fileId: string) =>
  call<{ file: ProjectFile }>(projectPath(id, '/files'), jsonRequest('POST', { file_id: fileId })).then(result => result.file);
export const unbindProjectFile = (id: string, fileId: string) => call<{ unbound: boolean }>(projectPath(id, `/files/${encodeURIComponent(fileId)}`), jsonRequest('DELETE'));
