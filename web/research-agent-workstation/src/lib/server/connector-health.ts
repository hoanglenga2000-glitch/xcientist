export type ConnectorHealthState = "READY" | "DEGRADED" | "OFFLINE" | "NOT_CONFIGURED";

export type ConnectorHealth = Record<string, unknown> & {
  name: string;
  state: ConnectorHealthState;
  raw_state: string;
  configured: boolean;
  source: "connector_health_service";
  checked_at: string;
};

export function normalizeConnectorHealth(provider: string, raw: Record<string, unknown>): ConnectorHealth {
  const configured = raw.configured === true;
  const rawState = String(raw.state ?? raw.status ?? "unknown");
  const state = rawState.toLowerCase();
  let canonical: ConnectorHealthState;
  if (!configured && (state.includes("not_configured") || state.includes("not configured") || state === "unknown")) {
    canonical = "NOT_CONFIGURED";
  } else if (
    raw.current_allocation_blocked === true
    || state.includes("offline")
    || state.includes("unreachable")
    || state.includes("blocked")
  ) {
    canonical = "OFFLINE";
  } else if (
    configured
    && (
      state === "ready"
      || state.includes(" ready")
      || state.includes("verified")
      || state.includes("passed")
      || state.includes("connected")
      || state === "local"
      || state === "local_workspace"
      || state === "rule_based"
    )
  ) {
    canonical = "READY";
  } else if (configured) {
    canonical = "DEGRADED";
  } else {
    canonical = "NOT_CONFIGURED";
  }
  return {
    ...raw,
    provider,
    name: typeof raw.name === "string" ? raw.name : provider,
    state: canonical,
    raw_state: rawState,
    configured,
    source: "connector_health_service",
    checked_at: new Date().toISOString(),
  };
}

export function normalizeConnectorRegistry(raw: Record<string, Record<string, unknown>>) {
  const registry = Object.fromEntries(Object.entries(raw).map(([provider, entry]) => [provider, normalizeConnectorHealth(provider, entry)]));
  if (registry.gpu) {
    registry.local_hpc = { ...registry.gpu, provider: "local_hpc", name: "HPC GPU" };
  }
  return registry;
}
