type Summary = { connector_status?: Record<string, unknown> } | null | undefined;
type Entry = Record<string, unknown>;
export type ConnectorView = { configured: boolean | null; state: string; tone: 'verified' | 'ready' | 'blocked' | 'unknown' | 'stale' };
function entries(summary: Summary): Record<string, Entry> {
  return Object.fromEntries(Object.entries(summary?.connector_status ?? {}).filter(([, value]) => value && typeof value === 'object' && !Array.isArray(value))) as Record<string, Entry>;
}
export function connectorStatus(summary: Summary, key: string): ConnectorView {
  const entry = entries(summary)[key];
  if (!entry) return { configured: null, state: 'unknown', tone: 'unknown' };
  const configured = typeof entry.configured === 'boolean' ? entry.configured : null;
  const state = String(entry.state ?? entry.status ?? 'unknown').toLowerCase();
  const negative = /(?:^|_)(blocked|failed|error|unavailable|denied)(?:_|$)/.test(state);
  if (['gpu', 'local_hpc'].includes(key) && configured === true && !negative && !hpcConnectionVerified(entry)) {
    return { configured, state: 'identity_recheck_required', tone: 'stale' };
  }
  const stale = /(?:^|_)(stale|expired)(?:_|$)/.test(state);
  // Exact positive states: "unverified" and "not_ready" must never become green.
  const tone = negative ? 'blocked' : stale ? 'stale' : configured === true && ['verified', 'ready', 'passed'].includes(state)
    ? 'verified' : configured === true ? 'ready' : 'unknown';
  return { configured, state, tone };
}
export function connectorLabel(locale: string | undefined, view: ConnectorView): string {
  const label = (en: string, zh: string) => locale === 'en-US' ? en : zh;
  if (view.tone === 'blocked') return label('Blocked', '阻断');
  if (view.tone === 'stale') return label('Needs recheck', '待重新核验');
  if (view.configured && view.state === 'verified') return label('Verified', '已验证');
  if (view.configured && ['ready', 'passed'].includes(view.state)) return label('Ready', '就绪');
  if (view.configured) return label('Configured', '已配置');
  if (view.configured === false) return label('Not configured', '未配置');
  return label('Unknown', '未知');
}
export function preferredModelConnector(summary: Summary): { key: string; model: string } | null {
  const candidates = Object.entries(entries(summary)).flatMap(([key, entry]) => {
    const model = typeof entry.model === 'string' ? entry.model.trim() : '';
    if (!model) return [];
    const status = connectorStatus(summary, key);
    return [{ key, model, score: status.tone === 'verified' ? 2 : status.configured ? 1 : 0 }];
  });
  const chosen = candidates.sort((a, b) => b.score - a.score)[0];
  return chosen ? { key: chosen.key, model: chosen.model } : null;
}

/** Connector readiness is narrower than execution budgets or research gates. */
export function connectorHealthReady(summary: Summary): boolean {
  const model = preferredModelConnector(summary);
  return model !== null && ['gpu', 'kaggle', model.key].every(
    key => connectorStatus(summary, key).tone === 'verified',
  );
}

export function hpcConnectionVerified(entry: Record<string, unknown> | undefined, now = Date.now()): boolean {
  if (!entry) return false;
  const expiry = Date.parse(String(entry.proof_expires_at ?? entry.expires_at_utc ?? ''));
  return entry.configured === true && entry.current_gate_ready === true
    && entry.job_container_verified === true && entry.designated_proxy_path_verified === true
    && entry.identity_samples_passed === 5 && entry.live_probe_fresh === true
    && Number.isFinite(expiry) && expiry > now;
}
