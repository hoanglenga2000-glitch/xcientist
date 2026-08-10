export const RUNTIME_BUILD_MANIFEST_SCHEMA = "evomind.runtime_build.v1" as const;

export type RuntimeBuildManifest = {
  schema: typeof RUNTIME_BUILD_MANIFEST_SCHEMA;
  commit_hash: string;
  source_dirty: boolean;
  source_tree_sha256: string;
  build_id: string;
  build_time: string;
  backend_version: string;
  frontend_version: string;
  database_schema_version: string;
  database_schema_sha256: string;
};

export type RuntimeHealthObservation = {
  reachable: boolean;
  backend_version?: string;
  commit_hash?: string;
  source_tree_sha256?: string;
};

export type RuntimeIdentityObservation = {
  launched_source_tree_sha256: string;
  frontend_version: string;
  database_schema_sha256: string;
  runtime: RuntimeHealthObservation;
};

export type RuntimeIdentityFailure =
  | "source_tree_mismatch"
  | "frontend_version_mismatch"
  | "database_schema_mismatch"
  | "runtime_unavailable"
  | "backend_version_mismatch"
  | "runtime_commit_mismatch"
  | "runtime_source_tree_mismatch";

export type RuntimeIdentityResult = {
  status: "ready" | "error";
  ready: boolean;
  failures: RuntimeIdentityFailure[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isBoundedString(value: unknown, maximum = 256): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

function isHex(value: unknown, length: number): value is string {
  return typeof value === "string" && new RegExp(`^[0-9a-f]{${length}}$`, "i").test(value);
}

export function parseRuntimeBuildManifest(value: unknown): RuntimeBuildManifest | null {
  if (!isRecord(value)) return null;
  if (value.schema !== RUNTIME_BUILD_MANIFEST_SCHEMA) return null;
  if (!isHex(value.commit_hash, 40)) return null;
  if (typeof value.source_dirty !== "boolean") return null;
  if (!isHex(value.source_tree_sha256, 64)) return null;
  if (!isBoundedString(value.build_id)) return null;
  if (!isBoundedString(value.build_time) || !Number.isFinite(Date.parse(value.build_time))) return null;
  if (!isBoundedString(value.backend_version, 64)) return null;
  if (!isBoundedString(value.frontend_version, 64)) return null;
  if (!isBoundedString(value.database_schema_version)) return null;
  if (!isHex(value.database_schema_sha256, 64)) return null;

  return {
    schema: RUNTIME_BUILD_MANIFEST_SCHEMA,
    commit_hash: value.commit_hash,
    source_dirty: value.source_dirty,
    source_tree_sha256: value.source_tree_sha256,
    build_id: value.build_id,
    build_time: value.build_time,
    backend_version: value.backend_version,
    frontend_version: value.frontend_version,
    database_schema_version: value.database_schema_version,
    database_schema_sha256: value.database_schema_sha256,
  };
}

export function evaluateRuntimeIdentity(
  manifestValue: RuntimeBuildManifest | unknown,
  observation: RuntimeIdentityObservation,
): RuntimeIdentityResult {
  const manifest = parseRuntimeBuildManifest(manifestValue);
  if (!manifest) {
    return { status: "error", ready: false, failures: ["source_tree_mismatch"] };
  }

  const failures: RuntimeIdentityFailure[] = [];
  if (observation.launched_source_tree_sha256 !== manifest.source_tree_sha256) {
    failures.push("source_tree_mismatch");
  }
  if (observation.frontend_version !== manifest.frontend_version) {
    failures.push("frontend_version_mismatch");
  }
  if (observation.database_schema_sha256 !== manifest.database_schema_sha256) {
    failures.push("database_schema_mismatch");
  }
  if (!observation.runtime.reachable) {
    failures.push("runtime_unavailable");
  } else {
    if (observation.runtime.backend_version !== manifest.backend_version) {
      failures.push("backend_version_mismatch");
    }
    if (observation.runtime.commit_hash !== manifest.commit_hash) {
      failures.push("runtime_commit_mismatch");
    }
    if (observation.runtime.source_tree_sha256 !== manifest.source_tree_sha256) {
      failures.push("runtime_source_tree_mismatch");
    }
  }

  return {
    status: failures.length === 0 ? "ready" : "error",
    ready: failures.length === 0,
    failures,
  };
}
