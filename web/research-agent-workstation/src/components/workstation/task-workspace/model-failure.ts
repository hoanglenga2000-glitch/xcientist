// t238 C7e A2: one short Chinese sentence + next action for an explicit model's failure, and the catalog alternate
// offered as 「换到备选模型」. Error codes never appear here; they stay in the closed 技术详情.
import type { ModelList } from './ModelSettings';
import { extraPlatformModels } from './composer-options';
import type { TaskRun } from './types';
import { productModelLabel } from './task-presentation';

export type ModelErrorKind = 'rate_limited' | 'unavailable' | 'timeout' | 'auth' | 'identity' | 'request' | 'unknown';

export const MODEL_ERROR_COPY: Record<ModelErrorKind, string> = {
  rate_limited: '这个模型现在请求太多，稍等一分钟再试，或者换一个模型。',
  unavailable: '这个模型暂时连不上，可以稍后再试，或者换一个模型。',
  timeout: '这个模型这次响应太慢，可以再试一次，或者换一个模型。',
  auth: '这个模型的访问权限有问题，请检查模型配置，或者换一个模型。',
  identity: '这次回复不是你选的模型给的，对话已保留。可以再试一次，或者换一个模型。',
  request: '这次请求没有被模型接受，对话已保留。可以再试一次，或者换一个模型。',
  unknown: '这次没有拿到模型回复，对话已保留。',
};
export const SWITCH_TO_FALLBACK = '换到备选模型';

const KINDS = new Set(Object.keys(MODEL_ERROR_COPY));

/** The failure kind the runtime classified (`run.model_error.kind`), or null when the run has none. */
export function modelErrorKind(run: Pick<TaskRun, 'model_error'>): ModelErrorKind | null {
  const kind = run.model_error?.kind;
  return kind && KINDS.has(kind) ? kind as ModelErrorKind : null;
}

/** The line for an explicit pick with a classified failure; null keeps the existing C6b copy. */
export function explicitModelErrorLine(run: Pick<TaskRun, 'model_error'>) {
  const kind = modelErrorKind(run);
  return kind && run.model_error?.explicit ? MODEL_ERROR_COPY[kind] : null;
}

/**
 * The listed platform model to offer as 「换到备选模型」: the run's own pre-selected fallback when still listed,
 * else the first listed model other than the failed one (the product default first). Hidden ids are never in the
 * list (the BE drops them), so they can never be offered. Null when nothing else is available.
 */
export function fallbackChoice(list: ModelList, run: Pick<TaskRun, 'model_error' | 'fallback_model_id' | 'model_profile'>): { value: string; label: string } | null {
  if (!run.model_error?.explicit) return null;
  const failed = run.model_error.platform_model_id ?? '';
  const extra = extraPlatformModels(list);
  // t238 F-2 (HARD): names come ONLY from the registry; a server-supplied `label` is never shown (it may be a derived name).
  const label = (model: string) => productModelLabel(model) || '备选模型';
  const own = run.fallback_model_id ? extra.find(item => item.id === run.fallback_model_id && item.id !== failed) : undefined;
  if (own) return { value: 'platform:' + own.id, label: label(own.model) };
  // The product default (`platform`) is an alternate unless it is the model that just failed.
  const base = list.platform_model;
  const failedIsDefault = base && failed && failed === `${base.provider}:${base.model}`;
  if (base && !failedIsDefault && (failed || run.model_profile)) return { value: 'platform', label: label(base.model) };
  const other = extra.find(item => item.id !== failed);
  return other ? { value: 'platform:' + other.id, label: label(other.model) } : null;
}

/** The visible notice after the runtime switched backends (A1); null when it never switched. */
export function failoverNotice(run: Pick<TaskRun, 'model_failover'>) {
  const notice = run.model_failover?.notice;
  return typeof notice === 'string' && notice.startsWith('已切换到备选模型') ? notice : null;
}

export const NO_FALLBACK = '';
export const FALLBACK_MODEL_KEY = 'deepevo.fallback-model';

/** Choices for 「备选模型」: listed platform models other than the explicit pick, plus 「不设置」. Only for an explicit pick. */
export function fallbackGroups(list: ModelList, selected: string): Array<{ label?: string; choices: Array<{ value: string; label: string }> }> | null {
  if (!selected || selected === 'platform') return null;  // the default already fails over by itself (A1)
  const failedId = selected.startsWith('platform:') ? selected.slice('platform:'.length) : '';
  const label = (model: string) => productModelLabel(model) || '备选模型';
  const choices = [
    ...(list.platform_model ? [{ value: `${list.platform_model.provider}:${list.platform_model.model}`, label: label(list.platform_model.model) }] : []),
    ...extraPlatformModels(list).map(item => ({ value: item.id, label: label(item.model) })),
  ].filter((item, index, all) => item.value !== failedId && all.findIndex(other => other.value === item.value) === index);
  if (!choices.length) return null;
  return [{ choices: [{ value: NO_FALLBACK, label: '不设置' }, ...choices] }];
}

/** `fallback_model_id` for run create: only a listed id, only with an explicit pick; undefined otherwise. */
export function fallbackModelId(list: ModelList, selected: string, fallback: string) {
  if (!fallback) return undefined;
  return fallbackGroups(list, selected)?.[0].choices.some(item => item.value === fallback) ? fallback : undefined;
}
