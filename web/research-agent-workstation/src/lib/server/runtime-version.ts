import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import { prisma } from "@/lib/db";
import { runtimeHealth } from "@/lib/server/evomind-runtime";
import {
  evaluateRuntimeIdentity,
  parseRuntimeBuildManifest,
  type RuntimeBuildManifest,
} from "@/lib/server/runtime-version-contract";

type SchemaRow = {
  type: string;
  name: string;
  tbl_name: string;
  sql: string | null;
};

export type RuntimeVersionResponse = {
  status: "ready" | "error";
  ready: boolean;
  commit_hash: string | null;
  build_time: string | null;
  build_id: string | null;
  backend_version: string | null;
  frontend_version: string | null;
  database_schema_version: string | null;
  source_tree_sha256: string | null;
  database_schema_sha256: string | null;
  source_dirty: boolean | null;
  failures: string[];
};

function manifestCandidates() {
  const configured = process.env.EVOMIND_RUNTIME_BUILD_MANIFEST?.trim();
  return [
    configured,
    path.join(process.cwd(), "runtime-build-manifest.json"),
    path.join(process.cwd(), ".next", "runtime-build-manifest.json"),
  ].filter((candidate): candidate is string => Boolean(candidate));
}

async function loadRuntimeBuildManifest(): Promise<RuntimeBuildManifest | null> {
  for (const candidate of manifestCandidates()) {
    try {
      const parsed = parseRuntimeBuildManifest(JSON.parse(await fs.readFile(candidate, "utf-8")));
      if (parsed) return parsed;
    } catch {
      // Continue to the next deterministic location. The result fails closed
      // below when no valid build identity can be loaded.
    }
  }
  return null;
}

function normalizeSchemaSql(value: string | null) {
  return (value ?? "").trim().replace(/\s+/g, " ");
}

async function liveDatabaseSchemaSha256() {
  const rows = await prisma.$queryRawUnsafe<SchemaRow[]>(`
    SELECT type, name, tbl_name, COALESCE(sql, '') AS sql
    FROM sqlite_master
    WHERE type IN ('table', 'index', 'view', 'trigger')
      AND name NOT LIKE 'sqlite_%'
    ORDER BY type, name, tbl_name
  `);
  const digest = createHash("sha256");
  for (const row of rows) {
    digest.update([row.type, row.name, row.tbl_name, normalizeSchemaSql(row.sql)].join("\0"), "utf-8");
    digest.update("\n", "utf-8");
  }
  return digest.digest("hex");
}

export async function runtimeVersionIdentity(): Promise<RuntimeVersionResponse> {
  const manifest = await loadRuntimeBuildManifest();
  if (!manifest) {
    return {
      status: "error",
      ready: false,
      commit_hash: null,
      build_time: null,
      build_id: null,
      backend_version: null,
      frontend_version: null,
      database_schema_version: null,
      source_tree_sha256: null,
      database_schema_sha256: null,
      source_dirty: null,
      failures: ["runtime_build_manifest_invalid"],
    };
  }

  const [runtime, databaseResult] = await Promise.all([
    runtimeHealth(),
    liveDatabaseSchemaSha256()
      .then((sha256) => ({ sha256, failure: null }))
      .catch(() => ({ sha256: "", failure: "database_schema_unavailable" })),
  ]);
  const identity = evaluateRuntimeIdentity(manifest, {
    launched_source_tree_sha256: process.env.EVOMIND_SOURCE_TREE_SHA256?.trim() ?? "",
    frontend_version: process.env.EVOMIND_FRONTEND_VERSION?.trim()
      || process.env.npm_package_version?.trim()
      || "unknown",
    database_schema_sha256: databaseResult.sha256,
    runtime,
  });
  const failures = databaseResult.failure
    ? [databaseResult.failure, ...identity.failures.filter((failure) => failure !== "database_schema_mismatch")]
    : identity.failures;

  return {
    status: failures.length === 0 ? "ready" : "error",
    ready: failures.length === 0,
    commit_hash: manifest.commit_hash,
    build_time: manifest.build_time,
    build_id: manifest.build_id,
    backend_version: manifest.backend_version,
    frontend_version: manifest.frontend_version,
    database_schema_version: manifest.database_schema_version,
    source_tree_sha256: manifest.source_tree_sha256,
    database_schema_sha256: manifest.database_schema_sha256,
    source_dirty: manifest.source_dirty,
    failures,
  };
}
