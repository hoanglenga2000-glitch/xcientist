export const SUPER_AGENT_RUNTIME_SCHEMA = "evomind.super_agent_runtime.v1" as const;
export const SUPER_AGENT_STATUS_SCHEMA = "evomind.super_agent_status_view.v1" as const;

export type SuperAgentMode = "off" | "shadow" | "enabled";
export type SuperAgentMountHealth = "ready" | "degraded" | "unknown";
export type SuperAgentKernelStatus =
  | "planned"
  | "running"
  | "verifying"
  | "repairing"
  | "canary_verifying"
  | "waiting_exact_gate"
  | "completed"
  | "unknown";

export type SuperAgentNodeStatus =
  | "pending"
  | "running"
  | "verifying"
  | "repairing"
  | "canary_verifying"
  | "waiting_exact_gate"
  | "completed"
  | "unknown";

export type SuperAgentMountView = {
  directory_id: string;
  connector_id: string;
  operations: string[];
  max_files: number;
  max_bytes: number;
  symlink_or_reparse_policy: "deny";
  expires_at: string | null;
  health: SuperAgentMountHealth;
};

export type SuperAgentTaskNodeView = {
  id: string;
  action: string;
  status: SuperAgentNodeStatus;
  dependencies: string[];
  capability_ids: string[];
};

export type SuperAgentExactGateView = {
  id: string;
  node_id: string;
  reason: string;
  required_action: string;
  resume_point: string;
  created_at: string;
};

export type SuperAgentRunView = {
  run_id: string;
  status: SuperAgentKernelStatus;
  objective: string;
  nodes: SuperAgentTaskNodeView[];
  failure_count: number;
  repair_count: number;
  exact_gate: SuperAgentExactGateView | null;
  node_list_truncated: boolean;
};

export type SuperAgentV1Status = {
  ok: true;
  schema: typeof SUPER_AGENT_STATUS_SCHEMA;
  upstream_schema: typeof SUPER_AGENT_RUNTIME_SCHEMA;
  mode: SuperAgentMode;
  enabled: boolean;
  shadow: boolean;
  migration_applied: boolean;
  capability_count: number;
  directory_count: number;
  directory_ids_sha256: string;
  active_run_ids: string[];
  mounts: SuperAgentMountView[];
  runs: SuperAgentRunView[];
  mount_list_truncated: boolean;
  run_list_truncated: boolean;
};

export class SuperAgentStatusContractError extends Error {
  readonly code = "runtime_contract_invalid";

  constructor() {
    super("managed Super Agent status did not satisfy the public contract");
    this.name = "SuperAgentStatusContractError";
  }
}

const ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
const SHA256 = /^[a-f0-9]{64}$/i;
const MAX_MOUNTS = 200;
const MAX_RUNS = 100;
const MAX_NODES = 200;
const MAX_IDS = 500;
const DIRECTORY_OPERATIONS = new Set([
  "health",
  "list",
  "stat",
  "read",
  "hash",
  "write",
  "mkdir",
  "copy",
  "sync",
  "delete",
  "execute",
]);
const KERNEL_STATES = new Set<SuperAgentKernelStatus>([
  "planned",
  "running",
  "verifying",
  "repairing",
  "canary_verifying",
  "waiting_exact_gate",
  "completed",
  "unknown",
]);
const NODE_STATES = new Set<SuperAgentNodeStatus>([
  "pending",
  "running",
  "verifying",
  "repairing",
  "canary_verifying",
  "waiting_exact_gate",
  "completed",
  "unknown",
]);

function invalid(): never {
  throw new SuperAgentStatusContractError();
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid();
  return value as Record<string, unknown>;
}

function boolean(value: unknown): boolean {
  if (typeof value !== "boolean") invalid();
  return value;
}

function count(value: unknown, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0 || Number(value) > maximum) invalid();
  return Number(value);
}

function string(value: unknown, maximum: number, { allowEmpty = false }: { allowEmpty?: boolean } = {}): string {
  if (typeof value !== "string") invalid();
  const normalized = value.trim();
  if ((!allowEmpty && !normalized) || normalized.length > maximum) invalid();
  return normalized;
}

function id(value: unknown): string {
  const normalized = string(value, 128);
  if (!ID.test(normalized)) invalid();
  return normalized;
}

function digest(value: unknown): string {
  const normalized = string(value, 64).toLowerCase();
  if (!SHA256.test(normalized)) invalid();
  return normalized;
}

function isoTimestamp(value: unknown): string {
  const normalized = string(value, 64);
  if (!/(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized) || !Number.isFinite(Date.parse(normalized))) invalid();
  return normalized;
}

function idList(value: unknown, maximum = MAX_IDS): { items: string[]; truncated: boolean } {
  if (!Array.isArray(value)) invalid();
  const items = value.map(id);
  return {
    items: [...new Set(items)].slice(0, maximum),
    truncated: items.length > maximum,
  };
}

function sanitizeDisplayText(value: unknown, maximum: number): string {
  let text = string(value, 4_096, { allowEmpty: true });
  if (text.includes("\uFFFD")) return "[unreadable text redacted]";
  text = text
    .replace(/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/gi, "[credential redacted]")
    .replace(/\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi, "[credential redacted]")
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g, "[credential redacted]")
    .replace(/\b(?:password|passwd|pwd|access[_ -]?token|refresh[_ -]?token|id[_ -]?token|api[_ -]?key|authorization|cookie|secret|private[_ -]?key|\u5bc6\u7801|\u53e3\u4ee4|\u51ed\u636e|\u5bc6\u94a5)\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/gi, "[credential redacted]")
    .replace(/([a-z][a-z0-9+.-]*:\/\/)[^\s/@:]+:[^\s/@]+@/gi, "$1[credential redacted]@")
    .replace(/\b(?:ssh|sftp):\/\/[^\s,;]+/gi, "[infrastructure endpoint redacted]")
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b/g, "[infrastructure endpoint redacted]")
    .replace(/\b[A-Za-z]:[\\/][^\s"'`<>|]+/g, "[path redacted]")
    .replace(/(?:\\\\|(?<!:)\/\/)[^\s"'`<>|]+[\\/][^\s"'`<>|]+/g, "[path redacted]")
    .replace(/\/(?:home|Users|var|tmp|opt|etc|mnt|srv|root|data|workspace|usr|run|bin|sbin|lib|lib64)\/[^\s"'`<>]*/g, "[path redacted]");
  return text.slice(0, maximum);
}

function mountHealth(value: unknown): SuperAgentMountHealth {
  if (value == null) return "unknown";
  const health = record(value);
  if (health.ok === true) return "ready";
  if (health.ok === false) return "degraded";
  return "unknown";
}

function sanitizeMount(value: unknown): SuperAgentMountView {
  const source = record(value);
  if (!Array.isArray(source.operations) || source.operations.length === 0 || source.operations.length > 32) invalid();
  const operations = [...new Set(source.operations.map((operation) => {
    const normalized = string(operation, 32).toLowerCase();
    if (!DIRECTORY_OPERATIONS.has(normalized)) invalid();
    return normalized;
  }))];
  const policy = string(source.symlink_or_reparse_policy, 16);
  if (policy !== "deny") invalid();
  const expiresAt = source.expires_at === "" || source.expires_at == null
    ? null
    : isoTimestamp(source.expires_at);
  return {
    directory_id: id(source.directory_id),
    connector_id: id(source.connector_id),
    operations,
    max_files: count(source.max_files),
    max_bytes: count(source.max_bytes),
    symlink_or_reparse_policy: policy,
    expires_at: expiresAt,
    health: mountHealth(source.health_receipt),
  };
}

function kernelStatus(value: unknown): SuperAgentKernelStatus {
  const normalized = string(value, 32).toLowerCase() as SuperAgentKernelStatus;
  if (!KERNEL_STATES.has(normalized)) invalid();
  return normalized;
}

function nodeStatus(value: unknown): SuperAgentNodeStatus {
  const normalized = string(value, 32).toLowerCase() as SuperAgentNodeStatus;
  if (!NODE_STATES.has(normalized)) invalid();
  return normalized;
}

function sanitizeNode(value: unknown): SuperAgentTaskNodeView {
  const source = record(value);
  return {
    id: id(source.id),
    action: sanitizeDisplayText(source.action, 300),
    status: nodeStatus(source.status),
    dependencies: idList(source.dependencies, 64).items,
    capability_ids: idList(source.capability_ids, 64).items,
  };
}

function sanitizeExactGate(value: unknown, runId: string): SuperAgentExactGateView | null {
  if (value == null) return null;
  const source = record(value);
  if (id(source.run_id) !== runId) invalid();
  return {
    id: id(source.id),
    node_id: id(source.node_id),
    reason: sanitizeDisplayText(source.reason, 512),
    required_action: sanitizeDisplayText(source.required_action, 512),
    resume_point: sanitizeDisplayText(source.resume_point, 128),
    created_at: isoTimestamp(source.created_at),
  };
}

function sanitizeRun(value: unknown): SuperAgentRunView {
  const source = record(value);
  const runId = id(source.run_id);
  if (!Array.isArray(source.nodes)) invalid();
  return {
    run_id: runId,
    status: kernelStatus(source.status),
    objective: sanitizeDisplayText(source.objective, 500),
    nodes: source.nodes.slice(0, MAX_NODES).map(sanitizeNode),
    failure_count: count(source.failure_count, 10_000_000),
    repair_count: count(source.repair_count, 10_000_000),
    exact_gate: sanitizeExactGate(source.exact_gate, runId),
    node_list_truncated: source.nodes.length > MAX_NODES,
  };
}

function assertModeConsistency(mode: SuperAgentMode, enabled: boolean, shadow: boolean): void {
  if (mode === "enabled" && (!enabled || shadow)) invalid();
  if (mode === "shadow" && (enabled || !shadow)) invalid();
  if (mode === "off" && (enabled || shadow)) invalid();
}

/**
 * Project the managed runtime response into the deliberately narrow browser
 * contract. Unknown fields are discarded; connector roots, tenant/project
 * identities, approval fingerprints, credential lease references and raw
 * health receipts are never returned.
 */
export function sanitizeSuperAgentV1Status(value: unknown): SuperAgentV1Status {
  const source = record(value);
  if (source.schema !== SUPER_AGENT_RUNTIME_SCHEMA) invalid();
  const mode = string(source.mode, 16) as SuperAgentMode;
  if (!(["off", "shadow", "enabled"] as const).includes(mode)) invalid();
  const enabled = boolean(source.enabled);
  const shadow = boolean(source.shadow);
  assertModeConsistency(mode, enabled, shadow);

  const capabilityCount = count(source.capability_count, 1_000_000);
  const directoryCount = count(source.directory_count, 1_000_000);
  if (capabilityCount < directoryCount || !Array.isArray(source.directories) || !Array.isArray(source.runs)) invalid();
  if (directoryCount < source.directories.length) invalid();

  const active = idList(source.active_run_ids, MAX_RUNS);
  const mounts = source.directories.slice(0, MAX_MOUNTS).map(sanitizeMount);
  const runs = source.runs.slice(0, MAX_RUNS).map(sanitizeRun);
  const activeIds = new Set(idList(source.active_run_ids, MAX_IDS).items);
  if (runs.some((run) => !activeIds.has(run.run_id))) invalid();

  return {
    ok: true,
    schema: SUPER_AGENT_STATUS_SCHEMA,
    upstream_schema: SUPER_AGENT_RUNTIME_SCHEMA,
    mode,
    enabled,
    shadow,
    migration_applied: boolean(source.migration_applied),
    capability_count: capabilityCount,
    directory_count: directoryCount,
    directory_ids_sha256: digest(source.directory_ids_sha256),
    active_run_ids: active.items,
    mounts,
    runs,
    mount_list_truncated: source.directories.length > MAX_MOUNTS,
    run_list_truncated: active.truncated || source.runs.length > MAX_RUNS,
  };
}

export function isSuperAgentV1Status(value: unknown): value is SuperAgentV1Status {
  try {
    const source = record(value);
    if (source.ok !== true || source.schema !== SUPER_AGENT_STATUS_SCHEMA || source.upstream_schema !== SUPER_AGENT_RUNTIME_SCHEMA) invalid();
    const mode = string(source.mode, 16) as SuperAgentMode;
    if (!(["off", "shadow", "enabled"] as const).includes(mode)) invalid();
    assertModeConsistency(mode, boolean(source.enabled), boolean(source.shadow));
    boolean(source.migration_applied);
    const capabilityCount = count(source.capability_count, 1_000_000);
    const directoryCount = count(source.directory_count, 1_000_000);
    if (capabilityCount < directoryCount) invalid();
    digest(source.directory_ids_sha256);
    const active = idList(source.active_run_ids, MAX_RUNS);
    if (active.truncated) invalid();
    boolean(source.mount_list_truncated);
    boolean(source.run_list_truncated);

    if (!Array.isArray(source.mounts) || source.mounts.length > MAX_MOUNTS || directoryCount < source.mounts.length) invalid();
    for (const value of source.mounts) {
      const mount = record(value);
      id(mount.directory_id);
      id(mount.connector_id);
      if (!Array.isArray(mount.operations) || mount.operations.length === 0 || mount.operations.length > 32) invalid();
      for (const operation of mount.operations) {
        if (!DIRECTORY_OPERATIONS.has(string(operation, 32).toLowerCase())) invalid();
      }
      count(mount.max_files);
      count(mount.max_bytes);
      if (mount.symlink_or_reparse_policy !== "deny") invalid();
      if (mount.expires_at !== null) isoTimestamp(mount.expires_at);
      if (!["ready", "degraded", "unknown"].includes(string(mount.health, 16))) invalid();
    }

    if (!Array.isArray(source.runs) || source.runs.length > MAX_RUNS) invalid();
    const activeIds = new Set(active.items);
    for (const value of source.runs) {
      const run = record(value);
      const runId = id(run.run_id);
      if (!activeIds.has(runId)) invalid();
      kernelStatus(run.status);
      const objective = string(run.objective, 500, { allowEmpty: true });
      if (sanitizeDisplayText(objective, 500) !== objective) invalid();
      count(run.failure_count, 10_000_000);
      count(run.repair_count, 10_000_000);
      boolean(run.node_list_truncated);
      if (!Array.isArray(run.nodes) || run.nodes.length > MAX_NODES) invalid();
      for (const nodeValue of run.nodes) {
        const node = record(nodeValue);
        id(node.id);
        const action = string(node.action, 300, { allowEmpty: true });
        if (sanitizeDisplayText(action, 300) !== action) invalid();
        nodeStatus(node.status);
        if (idList(node.dependencies, 64).truncated || idList(node.capability_ids, 64).truncated) invalid();
      }
      if (run.exact_gate !== null) {
        const gate = record(run.exact_gate);
        id(gate.id);
        id(gate.node_id);
        for (const [field, maximum] of [["reason", 512], ["required_action", 512], ["resume_point", 128]] as const) {
          const text = string(gate[field], maximum, { allowEmpty: true });
          if (sanitizeDisplayText(text, maximum) !== text) invalid();
        }
        isoTimestamp(gate.created_at);
      }
    }
    return true;
  } catch {
    return false;
  }
}
