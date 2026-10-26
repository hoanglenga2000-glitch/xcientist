export const INTENSITY_LEVELS = ['低', '中', '高'] as const;
export type Intensity = typeof INTENSITY_LEVELS[number];
const EFFORT: Record<Intensity, 'low' | 'medium' | 'high'> = { 低: 'low', 中: 'medium', 高: 'high' };

export const NEXT_MODEL_KEY = 'deepevo.next-model';
export const NEXT_INTENSITY_KEY = 'deepevo.next-intensity';
export const LAST_MODEL_KEY = 'deepevo.last-model';

export function isIntensity(value: unknown): value is Intensity {
  return INTENSITY_LEVELS.includes(value as Intensity);
}

/** Value sent as `reasoning_effort` on run create. */
export function reasoningEffort(level: string) {
  return EFFORT[isIntensity(level) ? level : '中'];
}

type PlatformModel = { id: string; provider: string; model: string; label?: string };
type Choices = {
  profiles: Array<{ id: string; enabled: boolean }>; default_profile_id: string;
  platform_model?: { provider: string; model: string } | null; platform_models?: PlatformModel[] | null;
};

/** Platform entries other than the default, which is served as `platform`. */
export function extraPlatformModels(list: Pick<Choices, 'platform_model' | 'platform_models'>) {
  const base = list.platform_model;
  return (list.platform_models ?? []).filter(item => item.id && !(base && item.provider === base.provider && item.model === base.model));
}

/** A picker value the API can serve: platform only when the platform model exists, profiles only when enabled. */
export function isSelectableModel(list: Choices, value: string) {
  if (!value) return false;
  if (value === 'platform') return Boolean(list.platform_model);
  if (value.startsWith('platform:')) return platformModelId(list, value) !== undefined;
  return list.profiles.some(item => item.id === value && item.enabled);
}

/** `platform_model_id` for run create; undefined for the default platform model and for profiles. */
export function platformModelId(list: Choices, value: string) {
  if (!value.startsWith('platform:')) return undefined;
  const id = value.slice('platform:'.length);
  return extraPlatformModels(list).some(item => item.id === id) ? id : undefined;
}

/** First usable of: the given preferences, platform, the default profile, the first enabled profile, the first platform entry. */
export function defaultModelChoice(list: Choices, ...preferred: Array<string | null | undefined>) {
  const enabled = list.profiles.filter(item => item.enabled);
  const first = list.platform_model ? '' : extraPlatformModels(list)[0]?.id;
  const candidates = [...preferred, list.platform_model ? 'platform' : '', list.default_profile_id, enabled[0]?.id, first ? 'platform:' + first : ''];
  return candidates.find((value): value is string => isSelectableModel(list, value || '')) ?? '';
}
