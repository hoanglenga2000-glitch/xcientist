import { createHash } from 'node:crypto';

type Principal = { username: string; tenantId: string | null; authentication: string } | null;
export type Preferences = Record<string, Record<string, string>>;
const fields: Record<string, Record<string, readonly string[]>> = {
  general: { theme: ['dark', 'light', 'system'] },
  language: { ui_language: ['zh-CN', 'en-US'], report_language: ['zh-CN', 'en-US'], agent_output_language: ['zh-CN', 'en-US'] },
};
export function preferenceKey(principal: Principal): string {
  if (!principal || !/^[A-Za-z0-9._-]{1,64}$/.test(principal.username)) throw new Error('session_required');
  const local = principal.authentication === 'local_bootstrap' && principal.username === 'local' && principal.tenantId === null;
  if (!local && (principal.authentication !== 'public_password' || !/^tenant_[a-f0-9]{24}$/.test(principal.tenantId ?? ''))) throw new Error('invalid_principal_scope');
  return `user_preferences_v1_${createHash('sha256').update(JSON.stringify([local ? 'local' : principal.tenantId, principal.username])).digest('hex')}`;
}
export function validatePreferences(value: unknown): Preferences {
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Object.keys(value).length) throw new Error('invalid_preferences');
  const result: Preferences = {};
  for (const [section, raw] of Object.entries(value)) {
    if (!Object.hasOwn(fields, section) || !raw || typeof raw !== 'object' || Array.isArray(raw) || !Object.keys(raw).length) throw new Error('invalid_preferences');
    result[section] = {};
    for (const [key, item] of Object.entries(raw)) {
      if (!Object.hasOwn(fields[section], key) || typeof item !== 'string' || !fields[section][key].includes(item)) throw new Error('invalid_preferences');
      result[section][key] = item;
    }
  }
  return result;
}
export function mergePreferences(stored: unknown, patch: Preferences = {}): Preferences {
  let safe: Preferences = {};
  try { safe = validatePreferences(stored); } catch { /* Never return legacy secrets or arbitrary settings. */ }
  return {
    general: { theme: 'dark', ...safe.general, ...patch.general },
    language: { ui_language: 'zh-CN', report_language: 'zh-CN', agent_output_language: 'zh-CN', ...safe.language, ...patch.language },
  };
}
