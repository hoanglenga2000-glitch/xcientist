import { jsonRequest, taskRequest } from './types';

/** t237 C3 long-term memory (spec 03): USER scope, agent auto-records, no manual add. */
export type MemoryKind = 'profile' | 'preference' | 'project' | 'fact' | 'other';
export type Memory = {
  id: string; text: string; source_task_id: string | null; source_task_title: string | null; project_id: string | null; created_at: string; updated_at: string;
  /** t237 C6c: absent on a pre-C6c snapshot ⇒ other / user / unpinned. */
  kind?: MemoryKind; scope?: 'user' | 'project'; pinned?: boolean; last_used_at?: string | null; use_count?: number;
};
export type MemorySaved = { id: string; text: string };

export const MEMORY_TEXT_MAX = 2000;
export const MEMORY_HIGHLIGHT_MS = 1500;
export const MEMORY_COPY = {
  tab: '记忆', title: '记忆', toggle: '使用记忆', pause: '暂停记忆', reset: '重置记忆',
  clearTitle: '清空全部记忆？', clearLine: '清空后不能恢复。', clear: '清空', cancel: '取消',
  empty: '还没有记忆', back: '返回对话', noMatch: '没有匹配的记忆', search: '搜索记忆', offEmpty: '记忆已关闭。', turnOn: '打开记忆',
  pin: (text: string) => `置顶：${text.slice(0, 40)}`, unpin: (text: string) => `取消置顶：${text.slice(0, 40)}`, kind: '分类',
  projectOnly: '仅用本项目记忆', incognito: '无痕', incognitoStatus: '无痕 · 不读不写记忆',
  edit: '编辑', remove: '删除', save: '保存', limit: '最多 2000 字', editLabel: '编辑记忆', more: (text: string) => `更多操作：${text.slice(0, 40)}`,
  deleted: '已删除', undo: '撤销', failed: '没能保存，请重试', loadFailed: '记忆暂时读不了，请稍后重试。',
} as const;

/** S6-2: one Chinese name per kind, product-wide. */
export const MEMORY_KINDS: MemoryKind[] = ['profile', 'preference', 'project', 'fact', 'other'];
export const MEMORY_KIND_LABEL: Record<MemoryKind, string> = { profile: '关于我', preference: '偏好', project: '项目', fact: '事实', other: '其他' };
export const memoryKindLabel = (kind?: string | null) => MEMORY_KIND_LABEL[(kind as MemoryKind) in MEMORY_KIND_LABEL ? kind as MemoryKind : 'other'];
/** H6-2: 全部 / 个人偏好 (preference + profile) / 项目 / 事实 / 置顶; 其他 only under 全部. */
export type MemoryTab = 'all' | 'preference' | 'project' | 'fact' | 'pinned';
export const MEMORY_TABS: Array<[MemoryTab, string]> = [['all', '全部'], ['preference', '个人偏好'], ['project', '项目'], ['fact', '事实'], ['pinned', '置顶']];
const TAB_KINDS: Partial<Record<MemoryTab, string[]>> = { preference: ['preference', 'profile'], project: ['project'], fact: ['fact'] };
/** H6-4: instant client-side filter — tab, then a case-folded substring of the text. */
export function filterMemories<T extends Pick<Memory, 'text' | 'kind' | 'pinned'>>(memories: T[], tab: MemoryTab, query: string) {
  const needle = query.trim().toLocaleLowerCase();
  const kinds = TAB_KINDS[tab];
  return memories.filter(memory => (tab !== 'pinned' || memory.pinned) && (!kinds || kinds.includes(memory.kind || 'other'))
    && (!needle || memory.text.toLocaleLowerCase().includes(needle)));
}
/** H7-5: 「用到的记忆 · N」; each line at most 40 characters. */
export const memoryUsedLabel = (count: number) => `用到的记忆 · ${count}`;
/** C7e-R1 (design soft): internal tool names are 技术详情 material, never a line the tenant reads here. */
const INTERNAL_TOOL_NAMES = /[（(]?\b(?:managed_tensor_train|tenant_tensor_train|hpc_execute_solution|managed_model_download|managed_cancel)\b[)）]?/g;
export const memoryUsedText = (text: string) => { const chars = [...text.replace(INTERNAL_TOOL_NAMES, '').replace(/\s+/g, ' ').trim()]; return chars.length > 40 ? chars.slice(0, 39).join('') + '…' : chars.join(''); };

/** Unset ⇒ ON (c3_contracts.resolve_memory_enabled). */
export const resolveMemoryEnabled = (settings?: { memory_enabled?: boolean | null } | null) => settings?.memory_enabled !== false;
/** [text, truncated?] — the same 2000-char clamp the BE applies. */
export const clampMemoryText = (text: string): [string, boolean] => {
  const chars = [...text];
  return chars.length > MEMORY_TEXT_MAX ? [chars.slice(0, MEMORY_TEXT_MAX).join(''), true] : [text, false];
};
/** Locked: same calendar year「10月26日」, otherwise「2025年10月26日」. */
export function memoryDate(iso: string, now = Date.now()) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const day = `${date.getMonth() + 1}月${date.getDate()}日`;
  return date.getFullYear() === new Date(now).getFullYear() ? day : `${date.getFullYear()}年${day}`;
}
export const memorySource = (memory: Pick<Memory, 'source_task_id' | 'source_task_title'>) =>
  memory.source_task_id && memory.source_task_title ? `来自「${memory.source_task_title}」` : '来自已删除的对话';
/** 03 §2/§12: the source title opens that conversation; a deleted source (no live title) is plain text. */
export const memorySourceHref = (memory: Pick<Memory, 'source_task_id' | 'source_task_title'>) =>
  memory.source_task_id && memory.source_task_title ? `/workspace?task=${encodeURIComponent(memory.source_task_id)}` : null;
/** PATCH returns the stored row; keep the live source the list resolved. */
export const mergeEditedMemory = (current: Memory, edited: Memory): Memory =>
  ({ ...current, ...edited, source_task_id: current.source_task_id, source_task_title: current.source_task_title });

export const memoryChipLabel = (saved: MemorySaved[]) => saved.length > 1 ? `已记住 ${saved.length} 条` : '已记住';
/** Hover title: 120 chars at most. */
export const memoryChipTitle = (saved: MemorySaved[]) => {
  const chars = [...saved.map(item => item.text).join('；')];
  return chars.length > 120 ? chars.slice(0, 119).join('') + '…' : chars.join('');
};
export const memoryHref = (id: string) => `/workspace?settings=memory&memory=${encodeURIComponent(id)}`;

const MEMORY = '/api/assistant/user-memory';
const SETTINGS = '/api/assistant/user-settings';
const item = (id: string) => `${MEMORY}/${encodeURIComponent(id)}`;
export const listMemories = (signal?: AbortSignal) => taskRequest<{ memories: Memory[] }>(MEMORY, { signal }).then(body => body.memories ?? []);
export const getMemorySettings = (signal?: AbortSignal) => taskRequest<{ memory_enabled?: boolean; paused?: boolean }>(SETTINGS, { signal });
export const setMemoryEnabled = (enabled: boolean) => taskRequest<{ memory_enabled: boolean }>(SETTINGS, jsonRequest('PATCH', { memory_enabled: enabled }));
export const setMemoryPaused = (paused: boolean) => taskRequest<{ memory_enabled: boolean; paused: boolean }>(SETTINGS, jsonRequest('PATCH', { paused }));
export const editMemory = (id: string, text: string) => taskRequest<{ memory: Memory; truncated: boolean }>(item(id), jsonRequest('PATCH', { text }));
/** C6c: any non-empty subset of {text, kind, pinned}. */
export const patchMemory = (id: string, body: { text?: string; kind?: MemoryKind; pinned?: boolean }) =>
  taskRequest<{ memory: Memory; truncated: boolean }>(item(id), jsonRequest('PATCH', body));
type ProjectMemorySettings = { use_account_memory: boolean; project_only: boolean };
const projectSettings = (projectId: string) => `/api/assistant/projects/${encodeURIComponent(projectId)}/memory-settings`;
export const getProjectMemorySettings = (projectId: string, signal?: AbortSignal) => taskRequest<ProjectMemorySettings>(projectSettings(projectId), { signal });
export const setProjectMemoryOnly = (projectId: string, projectOnly: boolean) =>
  taskRequest<ProjectMemorySettings>(projectSettings(projectId), jsonRequest('PATCH', { project_only: projectOnly }));
export const deleteMemory = (id: string) => taskRequest(item(id), jsonRequest('DELETE'));
export const restoreMemory = (id: string) => taskRequest<{ memory: Memory }>(`${item(id)}/restore`, jsonRequest('POST'));
export const clearMemories = () => taskRequest<{ cleared: number }>(MEMORY, jsonRequest('DELETE'));

/** Phase 2 我的经验: one row per training attempt, written by the platform (never by the model). */
export type Experience = {
  id: string; title: string; summary: string; outcome: 'success' | 'failed'; task: 'classification' | 'regression';
  created_at: string; success_count: number; use_count: number; fixed_failure_id?: string | null;
};
export type ExperienceSettings = { contribute_global: boolean };
export const EXPERIENCE_COPY = {
  title: '我的经验', line: '每次训练结束后自动记下数据规模、设置和结果，下次训练时作为参考。只有你自己能看到。',
  empty: '还没有训练经验', contribute: '参与匿名统计', contributeLine: '只汇总设置和结果的数字，不含数据和文字；至少 3 个用户出现过才会用到。',
  remove: (title: string) => `删除经验：${title.slice(0, 40)}`, deleted: '已删除', failed: '没能保存，请重试', loadFailed: '经验暂时读不了，请稍后重试。',
  reused: (count: number) => `帮到后续训练 ${count} 次`,
} as const;
const EXPERIENCE = '/api/assistant/experience';
export const listExperiences = (signal?: AbortSignal) =>
  taskRequest<{ experiences?: Experience[]; settings?: ExperienceSettings }>(EXPERIENCE, { signal })
    .then(body => ({ experiences: body.experiences ?? [], settings: { contribute_global: body.settings?.contribute_global !== false } }));
export const deleteExperience = (id: string) => taskRequest<{ id: string; deleted: boolean }>(`${EXPERIENCE}/${encodeURIComponent(id)}`, jsonRequest('DELETE'));
export const setExperienceContribution = (contribute: boolean) =>
  taskRequest<ExperienceSettings>(`${EXPERIENCE}/settings`, jsonRequest('PATCH', { contribute_global: contribute }));
