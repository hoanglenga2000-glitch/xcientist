import { createHash, createPublicKey, randomUUID, verify as verifySignature } from "node:crypto";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { copyFile, lstat, mkdir, open, readFile, readdir, realpath, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const MODULE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const MANIFEST_SCHEMA = "evomind.release.manifest.v2";
export const BUNDLE_SCHEMA = "evomind.windows.bundle.v1";
export const BOOTSTRAP_VERSION = "0.3.0";
export const INSTALL_MARKER_SCHEMA = "evomind.install_marker.v1";
export const TRANSACTION_SCHEMA = "evomind.release_transaction.v1";
export const ACCEPTANCE_SCHEMA = "evomind.release_acceptance.v1";
export const TRANSACTION_PHASES = Object.freeze([
  "prepared",
  "stopped",
  "db_snapshotted",
  "version_staged",
  "python_env_prepared",
  "health_passed",
  "acceptance_advanced",
  "pointer_committed",
]);
export const ROLLBACK_PHASES = Object.freeze([
  "rollback_prepared",
  "rollback_current_stopped",
  "rollback_target_starting",
  "rollback_target_started",
  "rollback_pointer_committed",
]);
const DEFAULT_MANIFEST_URL = process.env.EVOMIND_MANIFEST_URL || "https://releases.evomind.ai/stable/latest.json";
const TRUSTED_PUBLIC_KEY_PATH = path.join(MODULE_ROOT, "keys", "release-ed25519-public.pem");
const DASHBOARD_BOOTSTRAP_MAX_BYTES = 1024;
const DASHBOARD_BOOTSTRAP_TOKEN = /^[A-Za-z0-9_-]{24,256}$/;

let knownFolderCache = null;

function windowsKnownFolderRoots() {
  if (knownFolderCache) return knownFolderCache;
  const command = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); [ordered]@{local=[Environment]::GetFolderPath('LocalApplicationData');roaming=[Environment]::GetFolderPath('ApplicationData')}|ConvertTo-Json -Compress";
  const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", command], {
    encoding: "utf8",
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.error || result.status !== 0) throw new Error("Windows Known Folder lookup failed");
  let parsed;
  try { parsed = JSON.parse(result.stdout); } catch { throw new Error("Windows Known Folder lookup returned malformed JSON"); }
  if (!path.isAbsolute(String(parsed?.local || "")) || !path.isAbsolute(String(parsed?.roaming || ""))) {
    throw new Error("Windows Known Folder lookup returned invalid profile roots");
  }
  knownFolderCache = Object.freeze({ local: path.resolve(parsed.local), roaming: path.resolve(parsed.roaming) });
  return knownFolderCache;
}

export function layout(env = process.env) {
  // EVOMIND_* path overrides are dependency-injection hooks for callers that
  // pass an explicit environment object (tests/verifiers). The installed CLI
  // always derives purge-capable paths from Windows' canonical profile roots.
  const injected = env !== process.env;
  const knownFolders = !injected && process.platform === "win32" ? windowsKnownFolderRoots() : null;
  const localBase = path.resolve((injected ? env.EVOMIND_LOCALAPPDATA : null) || knownFolders?.local || env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"));
  const roamingBase = path.resolve((injected ? env.EVOMIND_APPDATA : null) || knownFolders?.roaming || env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"));
  const root = path.join(localBase, "EvoMind");
  const roaming = path.join(roamingBase, "EvoMind");
  return {
    root,
    versions: path.join(root, "app", "versions"),
    current: path.join(root, "app", "current.json"),
    cache: path.join(root, "cache", "downloads"),
    data: path.join(root, "data"),
    logs: path.join(root, "logs"),
    backups: path.join(root, "backups"),
    lock: path.join(root, ".lifecycle.lock"),
    journal: path.join(root, "app", "transaction.active.json"),
    transactions: path.join(root, "app", "transactions"),
    installMarker: path.join(root, ".install-marker.json"),
    acceptance: path.join(root, ".release-acceptance.json"),
    roaming,
    profiles: path.join(roaming, "profiles"),
    secrets: path.join(roaming, "secrets"),
  };
}

export function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
}

export function signedPayload(manifest) {
  const signature = manifest?.signature;
  const payload = {
    ...manifest,
    signature: {
      algorithm: signature?.algorithm,
      key_id: signature?.key_id,
    },
  };
  return Buffer.from(stableStringify(payload), "utf8");
}

function trustedPublicKey() {
  // Production trust is package-pinned. Tests may inject a key through the
  // function argument, but an ordinary environment variable cannot replace
  // the release trust root.
  if (!existsSync(TRUSTED_PUBLIC_KEY_PATH)) throw new Error("trusted EvoMind release public key is missing");
  return createPublicKey(readFileSync(TRUSTED_PUBLIC_KEY_PATH));
}

function parseSemver(value, label) {
  const match = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(String(value || ""));
  if (!match) throw new Error(`invalid ${label}`);
  return { core: match.slice(1, 4).map(Number), prerelease: match[4]?.split(".") ?? [] };
}

function compareSemver(left, right) {
  const a = parseSemver(left, "bootstrap version");
  const b = parseSemver(right, "minimum bootstrap version");
  for (let index = 0; index < 3; index += 1) {
    if (a.core[index] !== b.core[index]) return a.core[index] > b.core[index] ? 1 : -1;
  }
  if (!a.prerelease.length && !b.prerelease.length) return 0;
  if (!a.prerelease.length) return 1;
  if (!b.prerelease.length) return -1;
  const length = Math.max(a.prerelease.length, b.prerelease.length);
  for (let index = 0; index < length; index += 1) {
    if (a.prerelease[index] === undefined) return -1;
    if (b.prerelease[index] === undefined) return 1;
    if (a.prerelease[index] === b.prerelease[index]) continue;
    const aNumber = /^\d+$/.test(a.prerelease[index]);
    const bNumber = /^\d+$/.test(b.prerelease[index]);
    if (aNumber && bNumber) return Number(a.prerelease[index]) > Number(b.prerelease[index]) ? 1 : -1;
    if (aNumber !== bNumber) return aNumber ? -1 : 1;
    return a.prerelease[index] > b.prerelease[index] ? 1 : -1;
  }
  return 0;
}

function publishedEpoch(value, label = "release published_at") {
  const raw = String(value || "");
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/.test(raw)) {
    throw new Error(`invalid ${label}`);
  }
  const epoch = Date.parse(raw);
  if (!Number.isFinite(epoch)) throw new Error(`invalid ${label}`);
  return epoch;
}

export function verifyManifest(manifest, publicKeyPem = null, bootstrapVersion = BOOTSTRAP_VERSION) {
  if (!manifest || manifest.schema !== MANIFEST_SCHEMA) throw new Error("unsupported release manifest schema");
  parseSemver(manifest.version, "release version");
  parseSemver(manifest.min_bootstrap_version, "minimum bootstrap version");
  publishedEpoch(manifest.published_at);
  if (compareSemver(bootstrapVersion, manifest.min_bootstrap_version) < 0) {
    throw new Error(`release requires bootstrap >= ${manifest.min_bootstrap_version}; current bootstrap is ${bootstrapVersion}`);
  }
  const signature = manifest.signature;
  const signatureKeys = signature && typeof signature === "object" ? Object.keys(signature).sort() : [];
  if (
    !signature
    || signatureKeys.join(",") !== "algorithm,key_id,value"
    || signature.algorithm !== "Ed25519"
    || !/^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$/.test(String(signature.key_id || ""))
    || typeof signature.value !== "string"
  ) throw new Error("release signature is missing or invalid");
  const key = publicKeyPem?.type === "public" ? publicKeyPem : publicKeyPem ? createPublicKey(publicKeyPem) : trustedPublicKey();
  const expectedKeyId = `evomind-${createHash("sha256").update(key.export({ type: "spki", format: "der" })).digest("hex").slice(0, 16)}`;
  if (signature.key_id !== expectedKeyId) throw new Error("release manifest key_id does not match the trusted public key");
  const signatureBytes = Buffer.from(signature.value, "base64");
  if (signatureBytes.length !== 64 || signatureBytes.toString("base64") !== signature.value) throw new Error("release signature encoding is invalid");
  const valid = verifySignature(null, signedPayload(manifest), key, signatureBytes);
  if (!valid) throw new Error("release manifest signature verification failed");
  const platform = manifest.platforms?.["win32-x64"];
  if (
    !platform
    || typeof platform.url !== "string"
    || !platform.url.trim()
    || platform.archive !== "zip"
    || !/^[a-f0-9]{64}$/.test(String(platform.sha256 || ""))
    || !Number.isSafeInteger(platform.bytes)
    || platform.bytes <= 0
  ) {
    throw new Error("win32-x64 release entry is incomplete");
  }
  return manifest;
}

export async function sha256File(filePath) {
  const digest = createHash("sha256");
  await new Promise((resolve, reject) => {
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return digest.digest("hex");
}

async function atomicJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  await rename(temporary, filePath);
  await syncDirectory(path.dirname(filePath));
}

async function readJson(filePath, { allowMissing = true, label = path.basename(filePath) } = {}) {
  let raw;
  try {
    raw = await readFile(filePath, "utf8");
  } catch (error) {
    if (allowMissing && error?.code === "ENOENT") return null;
    throw new Error(`${label} could not be read`, { cause: error });
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} is malformed JSON`, { cause: error });
  }
}

async function syncDirectory(directory) {
  // Windows does not consistently permit opening directories for fsync. The
  // file itself is always flushed; directory sync is an additional durability
  // barrier on platforms that expose it.
  let handle;
  try {
    handle = await open(directory, "r");
    await handle.sync();
  } catch (error) {
    if (process.platform !== "win32") throw error;
  } finally {
    await handle?.close().catch(() => {});
  }
}

const WINDOWS_DEVICE_NAME = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i;

export function safeReleaseRelative(value) {
  const raw = String(value ?? "");
  if (!raw || raw.includes("\0") || raw.startsWith("/") || raw.startsWith("\\") || /^[A-Za-z]:/.test(raw)) {
    throw new Error(`unsafe release path: ${raw}`);
  }
  const parts = raw.replaceAll("\\", "/").split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) throw new Error(`unsafe release path: ${raw}`);
  for (const part of parts) {
    if (part.includes(":")) throw new Error(`alternate data stream is forbidden: ${raw}`);
    if (/[. ]$/.test(part)) throw new Error(`trailing dot or space is forbidden: ${raw}`);
    if (WINDOWS_DEVICE_NAME.test(part)) throw new Error(`reserved Windows device name is forbidden: ${raw}`);
  }
  return parts.join(path.sep);
}

function canonicalCase(value) {
  return path.resolve(value).replaceAll("/", path.sep).toLocaleLowerCase("en-US");
}

function assertContained(root, target, { allowRoot = false } = {}) {
  const base = path.resolve(root);
  const candidate = path.resolve(target);
  const foldedBase = canonicalCase(base);
  const foldedCandidate = canonicalCase(candidate);
  if ((!allowRoot && foldedCandidate === foldedBase) || (foldedCandidate !== foldedBase && !foldedCandidate.startsWith(`${foldedBase}${path.sep}`))) {
    throw new Error(`path escapes managed root: ${candidate}`);
  }
  return candidate;
}

async function assertNoReparse(root, target, { allowMissingLeaf = true } = {}) {
  const base = path.resolve(root);
  const candidate = assertContained(base, target, { allowRoot: true });
  const relative = path.relative(base, candidate);
  const segments = relative ? relative.split(path.sep) : [];
  let cursor = base;
  for (let index = 0; index <= segments.length; index += 1) {
    if (index > 0) cursor = path.join(cursor, segments[index - 1]);
    const info = await lstat(cursor).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
    if (!info) {
      if (!allowMissingLeaf || index < segments.length) return candidate;
      return candidate;
    }
    if (info.isSymbolicLink()) throw new Error(`reparse/symbolic path is forbidden: ${cursor}`);
    if (index < segments.length && !info.isDirectory()) throw new Error(`managed path ancestor is not a directory: ${cursor}`);
  }
  const resolvedBase = await realpath(base).catch(() => base);
  const resolvedCandidate = await realpath(candidate).catch(() => candidate);
  assertContained(resolvedBase, resolvedCandidate, { allowRoot: true });
  return candidate;
}

async function assertTreeNoReparse(target) {
  const info = await lstat(target).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (!info) return;
  if (info.isSymbolicLink()) throw new Error(`managed tree contains a reparse/symbolic path: ${target}`);
  if (!info.isDirectory()) return;
  for (const entry of await readdir(target, { withFileTypes: true })) {
    const child = path.join(target, entry.name);
    const childInfo = await lstat(child);
    if (childInfo.isSymbolicLink()) throw new Error(`managed tree contains a reparse/symbolic path: ${child}`);
    if (childInfo.isDirectory()) await assertTreeNoReparse(child);
  }
}

function pidAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error?.code === "EPERM"; }
}

function lifecycleProcessIdentity(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  if (process.platform === "win32") {
    const script = [
      "$ErrorActionPreference='Stop'",
      `$p=Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}'`,
      "if($null -eq $p){exit 3}",
      "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)",
      "[ordered]@{pid=[int]$p.ProcessId;creation_time=$p.CreationDate.ToUniversalTime().ToString('o');executable=[string]$p.ExecutablePath}|ConvertTo-Json -Compress",
    ].join(";");
    const result = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
      encoding: "utf8", windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.status === 3) return null;
    if (result.error || result.status !== 0) throw new Error("lifecycle process identity lookup failed");
    let identity;
    try { identity = JSON.parse(result.stdout); } catch { throw new Error("lifecycle process identity lookup returned malformed JSON"); }
    if (
      identity?.pid !== pid
      || typeof identity.creation_time !== "string"
      || !identity.creation_time
      || typeof identity.executable !== "string"
      || !path.isAbsolute(identity.executable)
    ) throw new Error("lifecycle process identity lookup returned incomplete data");
    return {
      pid,
      creation_time: identity.creation_time,
      executable: path.resolve(identity.executable),
    };
  }
  if (!pidAlive(pid)) return null;
  return {
    pid,
    creation_time: pid === process.pid ? String(Math.floor(Date.now() - process.uptime() * 1000)) : null,
    executable: pid === process.pid ? path.resolve(process.execPath) : null,
  };
}

function sameLifecycleProcess(left, right) {
  return Boolean(
    left
    && right
    && left.pid === right.pid
    && left.creation_time
    && left.creation_time === right.creation_time
    && left.executable
    && canonicalCase(left.executable) === canonicalCase(right.executable)
  );
}

async function acquireLock(paths) {
  await mkdir(path.dirname(paths.lock), { recursive: true });
  for (let attempt = 0; attempt < 2; attempt += 1) {
    let createdHandle = null;
    try {
      const handle = await open(paths.lock, "wx", 0o600);
      createdHandle = handle;
      const owner = lifecycleProcessIdentity(process.pid);
      if (!owner) {
        throw new Error("could not bind the lifecycle lock to the current process identity");
      }
      const nonce = randomUUID();
      await handle.writeFile(`${JSON.stringify({ schema: "evomind.lifecycle_lock.v2", owner, nonce, started_at: new Date().toISOString(), started_ms: Date.now() })}\n`);
      await handle.sync();
      return { handle, nonce };
    } catch (error) {
      if (createdHandle) {
        await createdHandle.close().catch(() => {});
        await rm(paths.lock, { force: true }).catch(() => {});
      }
      if (error?.code !== "EEXIST") throw error;
      const body = await readJson(paths.lock);
      const lockInfo = await stat(paths.lock).catch(() => null);
      const age = body?.started_ms ? Date.now() - Number(body.started_ms) : lockInfo ? Date.now() - lockInfo.mtimeMs : 0;
      if (!Number.isFinite(age) || age < -60_000) throw new Error("existing EvoMind lifecycle lock has an invalid timestamp");
      if (body?.schema === "evomind.lifecycle_lock.v2") {
        if (
          !body.owner
          || !Number.isSafeInteger(body.owner.pid)
          || typeof body.owner.creation_time !== "string"
          || typeof body.owner.executable !== "string"
          || !/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(String(body.nonce || ""))
        ) throw new Error("existing EvoMind lifecycle lock is malformed");
        const observed = lifecycleProcessIdentity(body.owner.pid);
        if (observed && sameLifecycleProcess(observed, body.owner)) {
          throw new Error("another EvoMind lifecycle operation is running");
        }
      } else if (body?.schema === "evomind.lifecycle_lock.v1" && pidAlive(Number(body.pid)) && age < 15 * 60_000) {
        throw new Error("a recent legacy EvoMind lifecycle operation may still be running");
      } else if (!body) {
        throw new Error("an unreadable EvoMind lifecycle lock exists");
      } else if (!["evomind.lifecycle_lock.v1", "evomind.lifecycle_lock.v2"].includes(body.schema)) {
        throw new Error("existing EvoMind lifecycle lock schema is invalid");
      }
      const stale = `${paths.lock}.stale-${Date.now()}-${randomUUID()}`;
      await rename(paths.lock, stale).catch((renameError) => {
        if (renameError?.code !== "ENOENT") throw renameError;
      });
      await syncDirectory(path.dirname(paths.lock));
    }
  }
  throw new Error("failed to recover a stale EvoMind lifecycle lock");
}

async function withLock(paths, callback) {
  const { handle, nonce } = await acquireLock(paths);
  try {
    const recovery = await recoverPendingTransaction(paths);
    return await callback(recovery);
  } finally {
    await handle.close();
    const owner = await readJson(paths.lock);
    if (owner?.schema !== "evomind.lifecycle_lock.v2" || owner.nonce !== nonce) {
      throw new Error("EvoMind lifecycle lock identity changed before release");
    }
    await rm(paths.lock, { force: true });
    await syncDirectory(path.dirname(paths.lock));
  }
}

export async function recoverReleaseState(paths = layout()) {
  return withLock(paths, async (recovery) => ({ ok: true, status: "recovered", ...recovery }));
}

function sourceKind(value) {
  if (/^[A-Za-z]:[\\/]/.test(String(value)) || String(value).startsWith("\\\\")) return "file:";
  try { return new URL(value).protocol; } catch { return "file:"; }
}

async function loadManifest(source, { publicKeyPem = null, bootstrapVersion = BOOTSTRAP_VERSION } = {}) {
  const target = source || DEFAULT_MANIFEST_URL;
  const protocol = sourceKind(target);
  let raw;
  if (protocol === "file:") {
    const filePath = target.startsWith("file:") ? fileURLToPath(target) : path.resolve(target);
    raw = await readFile(filePath, "utf8");
  } else {
    const url = new URL(target);
    if (url.protocol !== "https:" && !(url.protocol === "http:" && ["127.0.0.1", "localhost"].includes(url.hostname))) throw new Error("manifest URL must use HTTPS");
    const response = await fetch(url, { headers: { Accept: "application/json", "User-Agent": "@evomind-ai/cli" }, redirect: "follow" });
    if (!response.ok) throw new Error(`manifest download failed: HTTP ${response.status}`);
    const finalUrl = new URL(response.url);
    if (finalUrl.protocol !== "https:" && !(finalUrl.protocol === "http:" && ["127.0.0.1", "localhost"].includes(finalUrl.hostname))) {
      throw new Error("manifest redirect left the authenticated transport policy");
    }
    raw = await response.text();
  }
  if (Buffer.byteLength(raw) > 4 * 1024 * 1024) throw new Error("release manifest exceeds 4 MiB");
  return verifyManifest(JSON.parse(raw), publicKeyPem, bootstrapVersion);
}

async function download(source, target, expectedBytes) {
  const protocol = sourceKind(source);
  await mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${randomUUID()}.download`;
  try {
    if (protocol === "file:") {
      await copyFile(source.startsWith("file:") ? fileURLToPath(source) : path.resolve(source), temporary);
    } else {
      const url = new URL(source);
      if (url.protocol !== "https:" && !(url.protocol === "http:" && ["127.0.0.1", "localhost"].includes(url.hostname))) throw new Error("bundle URL must use HTTPS");
      const response = await fetch(url, { redirect: "follow", headers: { "User-Agent": "@evomind-ai/cli" } });
      if (!response.ok || !response.body) throw new Error(`bundle download failed: HTTP ${response.status}`);
      const finalUrl = new URL(response.url);
      if (finalUrl.protocol !== "https:" && !(finalUrl.protocol === "http:" && ["127.0.0.1", "localhost"].includes(finalUrl.hostname))) {
        throw new Error("bundle redirect left the authenticated transport policy");
      }
      const file = await open(temporary, "wx", 0o600);
      let written = 0;
      try {
        for await (const chunk of response.body) {
          written += chunk.length;
          if (written > expectedBytes) throw new Error("bundle exceeds signed byte length");
          await file.write(chunk);
        }
        await file.sync();
      } finally { await file.close(); }
    }
    const actual = await stat(temporary);
    if (actual.size !== expectedBytes) throw new Error("bundle byte length mismatch");
    await rename(temporary, target);
    await syncDirectory(path.dirname(target));
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function verifyArchiveForManifest(manifest, archivePath) {
  const target = path.resolve(String(archivePath || ""));
  if (!archivePath || !path.isAbsolute(target)) throw new Error("release bundle path is invalid");
  const info = await lstat(target).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (!info?.isFile() || info.isSymbolicLink()) throw new Error("release bundle must be a regular file");
  const platform = manifest.platforms["win32-x64"];
  if (info.size !== platform.bytes) throw new Error("release bundle byte length mismatch");
  const digest = await sha256File(target);
  if (digest !== platform.sha256) throw new Error("release bundle SHA-256 mismatch");
  return {
    ok: true,
    status: "verified",
    schema: manifest.schema,
    version: manifest.version,
    published_at: manifest.published_at,
    manifest_key_id: manifest.signature.key_id,
    archive: target,
    bytes: info.size,
    sha256: digest,
  };
}

export async function verifyReleaseEnvelope({
  manifestSource,
  archivePath,
  trustedPublicKey = null,
  bootstrapVersion = BOOTSTRAP_VERSION,
} = {}) {
  if (!manifestSource) throw new Error("release manifest source is required");
  const manifest = await loadManifest(manifestSource, { publicKeyPem: trustedPublicKey, bootstrapVersion });
  return verifyArchiveForManifest(manifest, archivePath);
}

function runPowerShell(script, args = [], { visible = false, env = {} } = {}) {
  const shell = process.env.ComSpec ? "powershell.exe" : "powershell";
  const childEnvironment = { ...process.env, ...env };
  delete childEnvironment.EVOMIND_QA_LAYOUT_CAPABILITY;
  delete childEnvironment.EVOMIND_QA_LAYOUT_NONCE;
  const result = spawnSync(shell, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script, ...args], {
    encoding: "utf8", windowsHide: !visible, stdio: ["ignore", "pipe", "pipe"], maxBuffer: 16 * 1024 * 1024,
    env: childEnvironment,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error((result.stderr || result.stdout || `PowerShell exited ${result.status}`).trim());
  return result.stdout.trim();
}

async function findBundleRoot(stage) {
  const direct = path.join(stage, "release-manifest.json");
  if (existsSync(direct)) return stage;
  const children = await readdir(stage, { withFileTypes: true });
  const matches = children.filter((entry) => entry.isDirectory() && existsSync(path.join(stage, entry.name, "release-manifest.json")));
  if (matches.length !== 1) throw new Error("bundle must contain exactly one release-manifest.json root");
  return path.join(stage, matches[0].name);
}

async function bundleTreeFingerprint(root) {
  const files = [];
  const seen = new Set();
  async function visit(relative = "") {
    const directory = relative ? path.join(root, relative) : root;
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0);
    for (const entry of entries) {
      const childRelative = relative ? path.join(relative, entry.name) : entry.name;
      const safe = safeReleaseRelative(childRelative);
      const folded = safe.replaceAll("/", path.sep).toLocaleLowerCase("en-US");
      if (seen.has(folded)) throw new Error(`bundle contains a case-fold path collision: ${childRelative}`);
      seen.add(folded);
      const target = path.join(root, safe);
      const info = await lstat(target);
      if (info.isSymbolicLink()) throw new Error(`bundle contains a reparse/symbolic path: ${childRelative}`);
      if (info.isDirectory()) await visit(safe);
      else if (info.isFile()) files.push({ path: safe.split(path.sep).join("/"), bytes: info.size, sha256: await sha256File(target) });
      else throw new Error(`bundle contains a non-regular filesystem entry: ${childRelative}`);
    }
  }
  await visit();
  files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  return {
    files,
    sha256: createHash("sha256").update(stableStringify(files)).digest("hex"),
  };
}

async function verifyBundle(root, version) {
  const manifest = await readJson(path.join(root, "release-manifest.json"));
  if (!manifest || manifest.schema !== BUNDLE_SCHEMA || manifest.version !== version || !Array.isArray(manifest.files) || !manifest.files.length) throw new Error("bundle release manifest is invalid");
  const seen = new Set();
  for (const item of manifest.files) {
    if (!item || typeof item.path !== "string" || !/^[a-f0-9]{64}$/.test(String(item.sha256 || ""))) throw new Error("bundle file entry is invalid");
    const relative = safeReleaseRelative(item.path);
    const folded = relative.replaceAll("/", path.sep).toLocaleLowerCase("en-US");
    if (seen.has(folded)) throw new Error(`bundle contains a case-fold path collision: ${item.path}`);
    seen.add(folded);
    const target = assertContained(root, path.join(root, relative));
    await assertNoReparse(root, target, { allowMissingLeaf: false });
    const info = await stat(target).catch(() => null);
    if (!info?.isFile() || info.size !== item.bytes || await sha256File(target) !== item.sha256) throw new Error(`bundle file verification failed: ${item.path}`);
  }
  const tree = await bundleTreeFingerprint(root);
  return { manifest, tree_sha256: tree.sha256, tree_files: tree.files };
}

async function ensureLayout(paths) {
  for (const dir of [paths.root, paths.versions, paths.cache, paths.data, paths.logs, paths.backups, paths.transactions, paths.profiles, paths.secrets]) {
    await mkdir(dir, { recursive: true });
    const info = await lstat(dir);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`managed directory is a reparse/symbolic path: ${dir}`);
  }
}

function managedEnvironment(paths, versionDir) {
  const contentId = path.basename(versionDir);
  return {
    WORKSTATION_DATA_DIR: paths.data,
    WORKSTATION_ROOT: paths.data,
    WORKSTATION_LOGS_DIR: paths.logs,
    WORKSTATION_RUNTIME_DIR: paths.logs,
    WORKSTATION_BACKUPS_DIR: paths.backups,
    EVOMIND_PROFILES_DIR: paths.profiles,
    EVOMIND_SECRETS_DIR: paths.secrets,
    EVOMIND_INSTALL_ROOT: versionDir,
    WORKSTATION_PYTHON: path.join(paths.data, "runtime", "python-env", contentId, "Scripts", "python.exe"),
    LOCALAPPDATA: path.dirname(paths.root),
    APPDATA: path.dirname(paths.roaming),
  };
}

function runBundleScript(versionDir, scriptName, args, paths, options = {}) {
  const script = path.join(versionDir, scriptName);
  if (!existsSync(script)) throw new Error(`${scriptName} is missing from installed bundle`);
  const output = runPowerShell(script, args, {
    ...options,
    env: { ...managedEnvironment(paths, versionDir), ...(options.env ?? {}) },
  });
  return parseLastJson(output);
}

function signalIdentity(status) {
  const outer = unwrapProcessResult(status) ?? status;
  const dashboard = unwrapProcessResult(outer?.dashboard) ?? outer?.dashboard ?? outer;
  const failures = [];
  if (dashboard?.identity_contract !== "evomind.process_identity.v1") failures.push("process identity contract is missing");
  if (dashboard?.dashboard_identity_verified !== true) failures.push("dashboard identity is not verified");
  if (dashboard?.process_port_consistent !== true) failures.push("dashboard PID does not own its recorded listener");
  if (
    dashboard?.runtime_identity_verified !== true
    && (dashboard?.runtime_pid_running === true || (Array.isArray(dashboard?.runtime_listener_pids) && dashboard.runtime_listener_pids.length))
  ) {
    failures.push("runtime identity is not verified");
  }
  return { passed: failures.length === 0, failures, dashboard };
}

function stopBundle(versionDir, paths) {
  const status = runBundleScript(versionDir, "status.ps1", [], paths);
  if (!observedRunning(status)) return { ok: true, status: "not_running" };
  const identity = signalIdentity(status);
  if (!identity.passed) throw new Error(`EvoMind process identity conflict; no signal sent: ${identity.failures.join("; ")}`);
  const result = runBundleScript(versionDir, "stop.ps1", [], paths);
  if (result?.ok === false || !["stopped", "not_running"].includes(String(result?.status || ""))) {
    throw new Error(`EvoMind stop did not release the managed process: ${result?.status ?? "malformed result"}`);
  }
  return result;
}

function unwrapProcessResult(value) {
  if (!value || typeof value !== "object") return null;
  if (value.ok === false || (Number.isInteger(value.exit_code) && value.exit_code !== 0)) return null;
  if (typeof value.output === "string") {
    const parsed = parseLastJson(value.output);
    return parsed && typeof parsed === "object" && !Object.hasOwn(parsed, "output") ? parsed : null;
  }
  return value;
}

function observedRunning(status) {
  const outer = unwrapProcessResult(status) ?? status;
  const dashboard = unwrapProcessResult(outer?.dashboard) ?? outer?.dashboard ?? outer;
  return Boolean(
    dashboard
    && (dashboard.pid_running === true || dashboard.health?.reachable === true)
  );
}

export function validateRuntimeStatus(status, expectedVersion) {
  const failures = [];
  const outer = unwrapProcessResult(status) ?? status;
  if (!outer || typeof outer !== "object") return { passed: false, failures: ["status output is not structured JSON"], dashboard: null };
  if (outer.install) {
    const install = unwrapProcessResult(outer.install);
    if (!install || !["installed", "initialized", "already_current"].includes(String(install.status || ""))) {
      failures.push("installation marker status is not healthy");
    }
  }
  let dashboard = unwrapProcessResult(outer.dashboard) ?? outer.dashboard;
  if (!dashboard && ["running", "started", "already_running"].includes(String(outer.status || ""))) dashboard = outer;
  if (!dashboard || typeof dashboard !== "object") {
    failures.push("dashboard status is missing or malformed");
    return { passed: false, failures, dashboard: null };
  }
  if (dashboard.status !== "running") failures.push(`dashboard status is ${dashboard.status ?? "missing"}`);
  if (dashboard.identity_contract !== "evomind.process_identity.v1") failures.push("dashboard process identity contract is missing");
  if (dashboard.dashboard_identity_verified !== true) failures.push("dashboard identity is not verified");
  if (dashboard.runtime_identity_verified !== true) failures.push("runtime identity is not verified");
  if (dashboard.pid_running !== true) failures.push("dashboard PID is not running");
  if (dashboard.process_port_consistent !== true) failures.push("dashboard PID/listener binding is inconsistent");
  if (dashboard.runtime_process_consistent !== true) failures.push("runtime PID/listener identity is inconsistent");
  const health = dashboard.health;
  if (!health || health.reachable !== true || health.http_status !== 200) failures.push("/api/healthz is not reachable with HTTP 200");
  if (String(health?.version || "") !== String(expectedVersion)) failures.push("dashboard health version does not match current.json");
  if (health?.has_runtime !== true) failures.push("EvoMind runtime is not attached");
  const runtime = health?.runtime;
  if (!runtime || runtime.reachable !== true || runtime.http_status !== 200 || runtime.status !== "ready") {
    failures.push("EvoMind runtime health is not ready");
  }
  if (String(runtime?.version || "") !== String(expectedVersion)) failures.push("EvoMind runtime version does not match current.json");
  return { passed: failures.length === 0, failures, dashboard };
}

async function readOptionalFile(filePath) {
  try { return await readFile(filePath); } catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

async function restoreOptionalFile(filePath, body) {
  if (body === null) {
    await rm(filePath, { force: true });
    return;
  }
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${Date.now()}.restore`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(body);
    await handle.sync();
  } finally {
    await handle.close();
  }
  await rename(temporary, filePath);
  await syncDirectory(path.dirname(filePath));
}

function releasePythonEnvironmentPaths(paths, bundleSha256, transactionId) {
  if (!/^[a-f0-9]{64}$/.test(String(bundleSha256 || ""))) throw new Error("invalid release Python environment content ID");
  if (!/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(String(transactionId || ""))) throw new Error("invalid release Python environment transaction ID");
  const root = path.join(paths.data, "runtime", "python-env");
  return {
    root,
    target: path.join(root, bundleSha256),
    stage: path.join(root, `.stage-${bundleSha256}-${transactionId}`),
    quarantine: path.join(root, `.quarantine-${bundleSha256}-${transactionId}`),
  };
}

async function assertManagedPythonEnvironmentPath(paths, target, { allowMissingLeaf = true } = {}) {
  const root = path.join(paths.data, "runtime", "python-env");
  assertContained(root, target);
  await assertNoReparse(paths.data, target, { allowMissingLeaf });
  return target;
}

async function removeManagedPythonEnvironment(paths, target) {
  const info = await lstat(target).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (!info) return;
  await assertManagedPythonEnvironmentPath(paths, target, { allowMissingLeaf: false });
  await assertTreeNoReparse(target);
  await rm(target, { recursive: true, force: false });
}

async function recoverInstallMutableSideEffects(paths, journal) {
  const expected = releasePythonEnvironmentPaths(paths, journal.target_bundle_sha256, journal.transaction_id);
  for (const key of ["target_python_env", "target_python_env_stage", "target_python_env_quarantine"]) {
    const expectedValue = { target_python_env: expected.target, target_python_env_stage: expected.stage, target_python_env_quarantine: expected.quarantine }[key];
    if (canonicalCase(journal[key]) !== canonicalCase(expectedValue)) throw new Error(`release transaction ${key} binding mismatch`);
  }
  if (journal.python_env_install_started === true) {
    await removeManagedPythonEnvironment(paths, expected.stage);
    await removeManagedPythonEnvironment(paths, expected.target);
  }
  const quarantineInfo = await lstat(expected.quarantine).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (quarantineInfo) {
    if (existsSync(expected.target)) throw new Error("cannot restore quarantined Python environment over an existing target");
    await assertManagedPythonEnvironmentPath(paths, expected.quarantine, { allowMissingLeaf: false });
    await assertTreeNoReparse(expected.quarantine);
    await rename(expected.quarantine, expected.target);
    await syncDirectory(expected.root);
  }
  const runtimeEnv = path.join(paths.data, "config", "runtime.env");
  await restoreOptionalFile(runtimeEnv, decodeOptional(journal.previous_runtime_env));
}

async function finalizeInstallMutableSideEffects(paths, journal) {
  const expected = releasePythonEnvironmentPaths(paths, journal.target_bundle_sha256, journal.transaction_id);
  await removeManagedPythonEnvironment(paths, expected.stage);
  await removeManagedPythonEnvironment(paths, expected.quarantine);
  const targetInfo = await lstat(expected.target).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (!targetInfo?.isDirectory() || targetInfo.isSymbolicLink()) throw new Error("committed release Python environment is missing or invalid");
  await assertManagedPythonEnvironmentPath(paths, expected.target, { allowMissingLeaf: false });
}

async function snapshotDatabase(paths, version) {
  const database = path.join(paths.data, "prisma", "workstation.db");
  const transactionRoot = path.join(paths.backups, "transactions");
  await mkdir(transactionRoot, { recursive: true });
  await assertNoReparse(paths.backups, transactionRoot, { allowMissingLeaf: false });
  const directory = path.join(transactionRoot, `${version}-${Date.now()}-${process.pid}`);
  await mkdir(directory, { recursive: false });
  const files = [];
  for (const suffix of ["", "-wal"]) {
    const source = `${database}${suffix}`;
    const info = await lstat(source).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
    if (!info) continue;
    if (!info.isFile() || info.isSymbolicLink()) throw new Error(`database snapshot source is not a regular file: ${source}`);
    const destination = path.join(directory, `workstation.db${suffix}`);
    await copyFile(source, destination);
    files.push({ suffix, bytes: info.size, sha256: await sha256File(destination) });
  }
  const receipt = {
    schema: "evomind.sqlite_transaction_backup.v1",
    created_at: new Date().toISOString(),
    database_existed: files.some((item) => item.suffix === ""),
    files,
    rollback: "restore_database_and_sidecars_before_restarting_previous_version",
  };
  const receiptPath = path.join(directory, "backup-receipt.json");
  await atomicJson(receiptPath, receipt);
  return { database, directory, receipt: receiptPath, receipt_sha256: await sha256File(receiptPath), ...receipt };
}

async function restoreDatabaseSnapshot(snapshot, paths = null) {
  if (paths) {
    const expectedDatabase = path.resolve(path.join(paths.data, "prisma", "workstation.db"));
    if (canonicalCase(snapshot.database) !== canonicalCase(expectedDatabase)) throw new Error("database rollback target does not match the canonical data layout");
    const transactionRoot = path.join(paths.backups, "transactions");
    assertContained(transactionRoot, snapshot.directory);
    await assertNoReparse(paths.backups, snapshot.directory, { allowMissingLeaf: false });
  }
  const receiptPath = path.resolve(snapshot.receipt || path.join(snapshot.directory, "backup-receipt.json"));
  if (canonicalCase(receiptPath) !== canonicalCase(path.join(snapshot.directory, "backup-receipt.json"))) throw new Error("database backup receipt path is invalid");
  if (!/^[a-f0-9]{64}$/.test(String(snapshot.receipt_sha256 || "")) || await sha256File(receiptPath) !== snapshot.receipt_sha256) {
    throw new Error("database backup receipt hash mismatch");
  }
  const receipt = await readJson(receiptPath);
  if (receipt?.schema !== "evomind.sqlite_transaction_backup.v1" || stableStringify(receipt.files) !== stableStringify(snapshot.files)) {
    throw new Error("database backup receipt content mismatch");
  }
  await mkdir(path.dirname(snapshot.database), { recursive: true });
  const nonce = `${process.pid}.${Date.now()}`;
  const staged = new Map();
  const quarantined = new Map();
  let committed = false;
  for (const item of snapshot.files) {
    const source = path.join(snapshot.directory, `workstation.db${item.suffix}`);
    if (await sha256File(source) !== item.sha256) throw new Error("database rollback snapshot hash mismatch");
    const temporary = `${snapshot.database}${item.suffix}.${nonce}.restore`;
    await copyFile(source, temporary);
    if (await sha256File(temporary) !== item.sha256) throw new Error("staged database rollback hash mismatch");
    staged.set(item.suffix, temporary);
  }
  try {
    for (const suffix of ["", "-wal", "-shm"]) {
      const current = `${snapshot.database}${suffix}`;
      if (!existsSync(current)) continue;
      const quarantine = `${current}.${nonce}.failed`;
      await rename(current, quarantine);
      quarantined.set(suffix, quarantine);
    }
    for (const [suffix, temporary] of staged) await rename(temporary, `${snapshot.database}${suffix}`);
    for (const item of snapshot.files) {
      if (await sha256File(`${snapshot.database}${item.suffix}`) !== item.sha256) throw new Error("restored database hash mismatch");
    }
    committed = true;
  } catch (error) {
    for (const suffix of ["", "-wal", "-shm"]) await rm(`${snapshot.database}${suffix}`, { force: true });
    for (const [suffix, quarantine] of quarantined) {
      if (existsSync(quarantine)) await rename(quarantine, `${snapshot.database}${suffix}`);
    }
    throw error;
  } finally {
    for (const temporary of staged.values()) await rm(temporary, { force: true });
  }
  if (committed) {
    for (const quarantine of quarantined.values()) await rm(quarantine, { force: true });
    await atomicJson(path.join(snapshot.directory, "rollback-receipt.json"), {
      schema: "evomind.sqlite_transaction_rollback.v1",
      restored_at: new Date().toISOString(),
      source_backup_receipt: snapshot.receipt,
      source_backup_receipt_sha256: snapshot.receipt_sha256,
      database_existed_before_transaction: snapshot.database_existed,
      restored_files: snapshot.files,
      verified: true,
    });
  }
}

async function collectLegacyFiles(root, relative = "") {
  const directory = relative ? path.join(root, relative) : root;
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const childRelative = relative ? path.join(relative, entry.name) : entry.name;
    const source = path.join(root, childRelative);
    const info = await lstat(source);
    if (info.isSymbolicLink()) throw new Error(`legacy credential migration rejects links: ${source}`);
    if (info.isDirectory()) files.push(...await collectLegacyFiles(root, childRelative));
    else if (info.isFile()) files.push({ source, relative: childRelative, bytes: info.size });
  }
  return files;
}

async function migrateLegacyProfiles(paths) {
  const legacyRoot = path.join(path.dirname(paths.roaming), "ResearchAgentWorkstation");
  const markerPath = path.join(paths.roaming, ".legacy-migration.json");
  const existing = await readJson(markerPath);
  if (existing) {
    const receipt = typeof existing.receipt === "string"
      ? assertContained(path.join(paths.backups, "profile-migrations"), existing.receipt)
      : null;
    if (existing.schema !== "evomind.dpapi_migration.v2" || !receipt || !existsSync(receipt)) throw new Error("legacy DPAPI migration marker is invalid");
    const audit = await verifyLegacyMigration(paths, existing);
    if (!audit.passed) throw new Error("legacy DPAPI migration audit failed");
    return { status: "already_migrated", migrated_files: existing.migrated_files, receipt, source_preserved: true };
  }
  const rootInfo = await lstat(legacyRoot).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
  if (!rootInfo) return { status: "not_found", migrated_files: 0, source_preserved: true };
  if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) throw new Error("legacy DPAPI root is not a regular directory");

  const candidates = [];
  for (const entry of await readdir(legacyRoot, { withFileTypes: true })) {
    const source = path.join(legacyRoot, entry.name);
    const info = await lstat(source);
    if (info.isSymbolicLink()) throw new Error(`legacy credential migration rejects links: ${source}`);
    const lowerName = entry.name.toLowerCase();
    const credentialName = /(credential|api[_-]?key|token|metadata|current)/i.test(entry.name)
      && !/[.](?:log|pid)$/i.test(entry.name);
    if (info.isFile() && credentialName) candidates.push({ source, relative: entry.name, bytes: info.size, kind: "secret" });
    else if (info.isDirectory() && (
      ["profiles", "secrets"].includes(lowerName)
      || lowerName.startsWith("backup")
      || lowerName.includes("credential")
      || lowerName.includes("generation")
    )) {
      const nested = await collectLegacyFiles(source);
      for (const file of nested) candidates.push({
        ...file,
        relative: ["profiles", "secrets"].includes(lowerName) ? file.relative : path.join(entry.name, file.relative),
        kind: lowerName === "profiles" ? "profile" : "secret",
      });
    }
  }
  if (!candidates.length) return { status: "not_found", migrated_files: 0, source_preserved: true };

  const created = [];
  const records = [];
  try {
    for (const item of candidates.sort((a, b) => a.source.localeCompare(b.source))) {
      const base = item.kind === "profile" ? paths.profiles : paths.secrets;
      const destination = assertContained(base, path.join(base, item.relative));
      const sourceHash = await sha256File(item.source);
      const existingInfo = await lstat(destination).catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
      let disposition = "copied";
      if (existingInfo) {
        if (!existingInfo.isFile() || existingInfo.isSymbolicLink() || await sha256File(destination) !== sourceHash) {
          throw new Error(`legacy credential destination conflict: ${destination}`);
        }
        disposition = "already_identical";
      } else {
        await mkdir(path.dirname(destination), { recursive: true });
        const temporary = `${destination}.${process.pid}.${Date.now()}.migrate`;
        await copyFile(item.source, temporary);
        if (await sha256File(temporary) !== sourceHash) throw new Error("legacy credential copy hash mismatch");
        await rename(temporary, destination);
        created.push(destination);
      }
      records.push({ source: item.source, destination, kind: item.kind, bytes: item.bytes, sha256: sourceHash, disposition });
    }
    const receiptDir = path.join(paths.backups, "profile-migrations");
    await mkdir(receiptDir, { recursive: true });
    const receiptPath = path.join(receiptDir, `legacy-dpapi-${Date.now()}.json`);
    await atomicJson(receiptPath, {
      schema: "evomind.dpapi_migration_receipt.v2",
      migrated_at: new Date().toISOString(),
      source_root: legacyRoot,
      source_preserved: true,
      records,
      rollback: { action: "remove copied destinations only when SHA-256 still matches", destinations: created },
    });
    const marker = {
      schema: "evomind.dpapi_migration.v2",
      migrated_at: new Date().toISOString(),
      migrated_files: records.length,
      receipt: receiptPath,
      source_preserved: true,
    };
    await atomicJson(markerPath, marker);
    return { status: "migrated", ...marker };
  } catch (error) {
    for (const destination of created.reverse()) await rm(destination, { force: true });
    throw error;
  }
}

function expectedInstallMarker(paths) {
  const root = path.resolve(paths.root);
  if (path.basename(root).toLocaleLowerCase("en-US") !== "evomind") throw new Error("managed install root must end in EvoMind");
  return {
    root,
    data: path.join(root, "data"),
    logs: path.join(root, "logs"),
    backups: path.join(root, "backups"),
    roaming: path.resolve(paths.roaming),
    profiles: path.join(path.resolve(paths.roaming), "profiles"),
    secrets: path.join(path.resolve(paths.roaming), "secrets"),
  };
}

function validateInstallMarker(paths, marker) {
  const expected = expectedInstallMarker(paths);
  if (
    marker?.schema !== INSTALL_MARKER_SCHEMA
    || !/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(String(marker.install_id || ""))
    || canonicalCase(marker.root) !== canonicalCase(expected.root)
  ) throw new Error("trusted EvoMind install marker is missing or invalid");
  for (const [key, value] of Object.entries(expected)) {
    if (canonicalCase(marker.paths?.[key]) !== canonicalCase(value)) throw new Error(`install marker ${key} path does not match the canonical layout`);
  }
  if (canonicalCase(paths.data) !== canonicalCase(expected.data)
    || canonicalCase(paths.logs) !== canonicalCase(expected.logs)
    || canonicalCase(paths.backups) !== canonicalCase(expected.backups)
    || canonicalCase(paths.profiles) !== canonicalCase(expected.profiles)
    || canonicalCase(paths.secrets) !== canonicalCase(expected.secrets)) {
    throw new Error("environment-controlled lifecycle paths do not match the trusted install marker");
  }
  return marker;
}

async function ensureInstallMarker(paths) {
  const current = await readJson(paths.installMarker);
  if (current) return validateInstallMarker(paths, current);
  const expected = expectedInstallMarker(paths);
  const marker = {
    schema: INSTALL_MARKER_SCHEMA,
    product: "EvoMind",
    install_id: randomUUID(),
    created_at: new Date().toISOString(),
    root: expected.root,
    paths: expected,
  };
  await atomicJson(paths.installMarker, marker);
  return validateInstallMarker(paths, marker);
}

async function loadInstallMarker(paths) {
  const marker = await readJson(paths.installMarker);
  return validateInstallMarker(paths, marker);
}

function acceptanceEventFromLegacy(current) {
  if (!current?.version) return null;
  if (!/^[a-f0-9]{64}$/.test(String(current.bundle_sha256 || ""))) {
    throw new Error("legacy current pointer is missing its signed archive hash");
  }
  if (!/^[a-f0-9]{64}$/.test(String(current.content_tree_sha256 || ""))) {
    throw new Error("legacy current pointer is missing its trusted content-tree hash");
  }
  return {
    sequence: 1,
    version: current.version,
    published_at: null,
    published_epoch_ms: null,
    bundle_sha256: current.bundle_sha256,
    content_tree_sha256: current.content_tree_sha256,
    manifest_key_id: current.manifest_key_id ?? null,
    manifest_payload_sha256: null,
    accepted_at: new Date().toISOString(),
    source: "adopted_verified_current_pointer",
  };
}

function validateAcceptanceLedger(paths, marker, ledger) {
  if (
    ledger?.schema !== ACCEPTANCE_SCHEMA
    || ledger.install_id !== marker.install_id
    || !Number.isSafeInteger(ledger.sequence)
    || ledger.sequence < 0
    || !Array.isArray(ledger.history)
    || ledger.history.length !== ledger.sequence
  ) throw new Error("EvoMind release acceptance ledger is missing or invalid");
  let previous = null;
  for (let index = 0; index < ledger.history.length; index += 1) {
    const event = ledger.history[index];
    parseSemver(event?.version, "accepted release version");
    if (
      event?.sequence !== index + 1
      || !/^[a-f0-9]{64}$/.test(String(event.bundle_sha256 || ""))
      || !/^[a-f0-9]{64}$/.test(String(event.content_tree_sha256 || ""))
      || !["signed_manifest", "adopted_verified_current_pointer"].includes(event.source)
    ) throw new Error("EvoMind release acceptance history is malformed");
    if (event.source === "signed_manifest") {
      const epoch = publishedEpoch(event.published_at, "accepted release published_at");
      if (
        event.published_epoch_ms !== epoch
        || !/^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$/.test(String(event.manifest_key_id || ""))
        || !/^[a-f0-9]{64}$/.test(String(event.manifest_payload_sha256 || ""))
      ) throw new Error("signed release acceptance history is malformed");
    } else if (event.published_at !== null || event.published_epoch_ms !== null || event.manifest_payload_sha256 !== null) {
      throw new Error("legacy release acceptance history is malformed");
    }
    if (previous) {
      const ordering = compareSemver(event.version, previous.version);
      if (ordering < 0) throw new Error("release acceptance history moved backwards");
      if (ordering === 0) {
        if (event.source !== "signed_manifest") throw new Error("same-version acceptance advancement must be signed");
        const previousEpoch = previous.published_epoch_ms;
        if (previousEpoch !== null && event.published_epoch_ms <= previousEpoch) {
          throw new Error("same-version acceptance history did not advance signed time");
        }
      } else if (
        previous.published_epoch_ms !== null
        && event.published_epoch_ms !== null
        && event.published_epoch_ms < previous.published_epoch_ms
      ) {
        throw new Error("release acceptance history signed time moved backwards");
      }
    }
    previous = event;
  }
  if ((previous === null) !== (ledger.highest === null)) throw new Error("release acceptance high-water marker is inconsistent");
  if (previous && stableStringify(previous) !== stableStringify(ledger.highest)) {
    throw new Error("release acceptance high-water marker does not match its history");
  }
  if (ledger.updated_at) publishedEpoch(ledger.updated_at, "release acceptance updated_at");
  return ledger;
}

async function ensureAcceptanceLedger(paths, marker, current = null) {
  const existing = await readJson(paths.acceptance);
  if (existing) return validateAcceptanceLedger(paths, marker, existing);
  const adopted = acceptanceEventFromLegacy(current);
  const ledger = {
    schema: ACCEPTANCE_SCHEMA,
    install_id: marker.install_id,
    sequence: adopted ? 1 : 0,
    highest: adopted,
    history: adopted ? [adopted] : [],
    updated_at: new Date().toISOString(),
  };
  await atomicJson(paths.acceptance, ledger);
  return validateAcceptanceLedger(paths, marker, ledger);
}

async function loadAcceptanceLedger(paths, marker = null) {
  const installMarker = marker ?? await loadInstallMarker(paths);
  const ledger = await readJson(paths.acceptance, { allowMissing: false, label: "release acceptance ledger" });
  return validateAcceptanceLedger(paths, installMarker, ledger);
}

function assertAcceptancePolicy(ledger, manifest) {
  const highest = ledger.highest;
  if (!highest) return { advancement: true, reason: "first_release" };
  const platform = manifest.platforms["win32-x64"];
  const ordering = compareSemver(manifest.version, highest.version);
  if (ordering < 0) {
    throw new Error(`anti-rollback high-water rejected ${manifest.version}; highest accepted version is ${highest.version}`);
  }
  const epoch = publishedEpoch(manifest.published_at);
  if (ordering === 0 && platform.sha256 === highest.bundle_sha256) {
    return {
      advancement: highest.published_epoch_ms === null || epoch > highest.published_epoch_ms,
      reason: "same_authenticated_content",
    };
  }
  if (ordering === 0 && highest.published_epoch_ms === null) {
    throw new Error("anti-rollback rejected a different same-version build because the adopted high-water has no signed timestamp");
  }
  if (
    highest.published_epoch_ms !== null
    && (ordering === 0 ? epoch <= highest.published_epoch_ms : epoch < highest.published_epoch_ms)
  ) {
    throw new Error("anti-rollback rejected a replayed or non-monotonic signed release");
  }
  return { advancement: true, reason: ordering === 0 ? "newer_same_version_build" : "newer_version" };
}

async function advanceAcceptanceLedger(paths, marker, ledger, manifest, contentTreeSha256) {
  const decision = assertAcceptancePolicy(ledger, manifest);
  if (!decision.advancement) return ledger;
  if (!/^[a-f0-9]{64}$/.test(String(contentTreeSha256 || ""))) {
    throw new Error("release acceptance requires a verified content-tree hash");
  }
  const platform = manifest.platforms["win32-x64"];
  const event = {
    sequence: ledger.sequence + 1,
    version: manifest.version,
    published_at: manifest.published_at,
    published_epoch_ms: publishedEpoch(manifest.published_at),
    bundle_sha256: platform.sha256,
    content_tree_sha256: contentTreeSha256,
    manifest_key_id: manifest.signature.key_id,
    manifest_payload_sha256: createHash("sha256").update(signedPayload(manifest)).digest("hex"),
    accepted_at: new Date().toISOString(),
    source: "signed_manifest",
    reason: decision.reason,
  };
  const updated = {
    ...ledger,
    sequence: event.sequence,
    highest: event,
    history: [...ledger.history, event],
    updated_at: new Date().toISOString(),
  };
  await atomicJson(paths.acceptance, updated);
  return validateAcceptanceLedger(paths, marker, updated);
}

function encodeOptional(body) {
  return body === null ? null : Buffer.from(body).toString("base64");
}

function decodeOptional(body) {
  return body === null ? null : Buffer.from(String(body), "base64");
}

function phaseNumber(phase) {
  const phases = TRANSACTION_PHASES.includes(phase) ? TRANSACTION_PHASES : ROLLBACK_PHASES;
  const index = phases.indexOf(phase);
  if (index < 0) throw new Error(`invalid release transaction phase: ${phase}`);
  return index;
}

async function writeTransactionPhase(paths, journal, phase) {
  if (journal.phase && phaseNumber(phase) < phaseNumber(journal.phase)) throw new Error("release transaction phase cannot move backwards");
  const next = {
    ...journal,
    schema: TRANSACTION_SCHEMA,
    phase,
    sequence: Number(journal.sequence || 0) + 1,
    updated_at: new Date().toISOString(),
  };
  await atomicJson(paths.journal, next);
  Object.assign(journal, next);
  return journal;
}

async function archiveTransaction(paths, journal, outcome, detail = null) {
  const record = { ...journal, outcome, outcome_detail: detail, finished_at: new Date().toISOString() };
  const destination = path.join(paths.transactions, `${journal.transaction_id}.${outcome}.json`);
  await atomicJson(destination, record);
  await rm(paths.journal, { force: true });
  await syncDirectory(path.dirname(paths.journal));
  return destination;
}

async function invokeFault(faultInjector, phase, journal) {
  if (typeof faultInjector === "function") await faultInjector(phase, structuredClone(journal));
}

async function recoverRollbackTransaction(paths, journal) {
  const sourceDir = assertContained(paths.versions, journal.source_directory);
  const targetDir = assertContained(paths.versions, journal.target_directory);
  if (journal.phase === "rollback_pointer_committed") {
    const pointer = await readJson(paths.current);
    if (pointer?.transaction_id !== journal.transaction_id || pointer.version !== journal.target_version) {
      throw new Error("committed rollback pointer identity mismatch");
    }
    const receipt = await archiveTransaction(paths, journal, "rollback_committed_recovered");
    return { recovered: true, action: "finalized_committed_rollback", receipt };
  }
  const errors = [];
  if (journal.target_should_start && phaseNumber(journal.phase) >= phaseNumber("rollback_target_starting")) {
    try {
      await assertInstalledContent(targetDir, journal.target_version, journal.target_content_tree_sha256);
      stopBundle(targetDir, paths);
    } catch (error) { errors.push(`stop rollback target: ${error.message}`); }
  }
  if (!errors.length) {
    try { await restoreOptionalFile(paths.current, decodeOptional(journal.source_pointer)); }
    catch (error) { errors.push(`restore source pointer: ${error.message}`); }
  }
  if (!errors.length && journal.source_was_running) {
    try {
      await assertInstalledContent(sourceDir, journal.source_version, journal.source_content_tree_sha256);
      runBundleScript(sourceDir, "start.ps1", [], paths);
      const status = runBundleScript(sourceDir, "status.ps1", [], paths);
      const health = validateRuntimeStatus(status, journal.source_version);
      if (!health.passed) throw new Error(health.failures.join("; "));
    } catch (error) { errors.push(`restart rollback source: ${error.message}`); }
  }
  if (errors.length) throw new Error(`rollback transaction recovery failed: ${errors.join(" | ")}`);
  const receipt = await archiveTransaction(paths, journal, "rollback_reverted_after_interruption");
  return { recovered: true, action: "reverted_interrupted_rollback", phase: journal.phase, receipt };
}

export async function recoverPendingTransaction(paths = layout()) {
  const journal = await readJson(paths.journal);
  if (!journal) return { recovered: false };
  const rollbackOperation = journal.operation === "rollback";
  if (
    journal.schema !== TRANSACTION_SCHEMA
    || !(rollbackOperation ? ROLLBACK_PHASES : TRANSACTION_PHASES).includes(journal.phase)
    || !/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(String(journal.transaction_id || ""))
    || (!rollbackOperation && !/^[a-f0-9]{64}$/.test(String(journal.target_bundle_sha256 || "")))
  ) {
    throw new Error("active EvoMind release transaction journal is malformed");
  }
  if (journal.install_id) {
    const marker = await loadInstallMarker(paths);
    if (marker.install_id !== journal.install_id) throw new Error("release transaction install identity mismatch");
  }
  if (rollbackOperation) return recoverRollbackTransaction(paths, journal);
  if (journal.phase === "pointer_committed") {
    const pointer = await readJson(paths.current);
    if (
      pointer?.transaction_id !== journal.transaction_id
      || pointer.version !== journal.target_version
      || pointer.bundle_sha256 !== journal.target_bundle_sha256
    ) throw new Error("committed transaction pointer identity mismatch");
    await finalizeInstallMutableSideEffects(paths, journal);
    const receipt = await archiveTransaction(paths, journal, "committed_recovered");
    return { recovered: true, action: "finalized_committed_pointer", receipt };
  }

  const errors = [];
  const destination = journal.destination ? assertContained(paths.versions, journal.destination) : null;
  const expectedDestination = path.join(paths.versions, String(journal.target_version), journal.target_bundle_sha256);
  if (destination && canonicalCase(destination) !== canonicalCase(expectedDestination)) throw new Error("release transaction destination binding mismatch");
  if (destination && journal.destination_created === true && existsSync(destination)) {
    try {
      const status = runBundleScript(destination, "status.ps1", [], paths);
      if (observedRunning(status)) stopBundle(destination, paths);
    } catch (error) {
      errors.push(`stop staged version: ${error.message}`);
    }
  }
  if (!errors.length) {
    try { await recoverInstallMutableSideEffects(paths, journal); }
    catch (error) { errors.push(`restore release mutable state: ${error.message}`); }
  }
  if (!errors.length && journal.database_snapshot) {
    try { await restoreDatabaseSnapshot(journal.database_snapshot, paths); }
    catch (error) { errors.push(`restore database: ${error.message}`); }
  }
  if (!errors.length) {
    try { await restoreOptionalFile(path.join(paths.data, "install-state.json"), decodeOptional(journal.previous_install_state)); }
    catch (error) { errors.push(`restore install state: ${error.message}`); }
  }
  if (!errors.length) {
    try { await restoreOptionalFile(paths.current, decodeOptional(journal.previous_pointer)); }
    catch (error) { errors.push(`restore version pointer: ${error.message}`); }
  }
  if (!errors.length && destination && journal.destination_created === true && existsSync(destination)) {
    try { await rm(destination, { recursive: true, force: false }); }
    catch (error) { errors.push(`remove staged version: ${error.message}`); }
  }
  if (!errors.length && journal.stage) {
    try {
      const safeStage = assertContained(paths.versions, journal.stage);
      if (!path.basename(safeStage).startsWith(".stage-")) throw new Error("journal stage path is not a managed staging directory");
      await rm(safeStage, { recursive: true, force: true });
    } catch (error) { errors.push(`remove extraction stage: ${error.message}`); }
  }
  if (!errors.length && journal.previous_was_running && journal.previous_directory) {
    try {
      const previousDir = assertContained(paths.versions, journal.previous_directory);
      runBundleScript(previousDir, "start.ps1", [], paths);
      const status = runBundleScript(previousDir, "status.ps1", [], paths);
      const health = validateRuntimeStatus(status, journal.previous_version);
      if (!health.passed) throw new Error(health.failures.join("; "));
    } catch (error) { errors.push(`restart previous version: ${error.message}`); }
  }
  if (errors.length) throw new Error(`release transaction recovery failed: ${errors.join(" | ")}`);
  const receipt = await archiveTransaction(paths, journal, "rolled_back_after_interruption");
  return { recovered: true, action: "rolled_back_interrupted_transaction", phase: journal.phase, receipt };
}

export async function installRelease({
  manifestSource,
  version = null,
  paths = layout(),
  trustedPublicKey = null,
  bootstrapVersion = BOOTSTRAP_VERSION,
  faultInjector = null,
} = {}) {
  return withLock(paths, async () => {
    await ensureLayout(paths);
    const installMarker = await ensureInstallMarker(paths);
    const manifest = await loadManifest(manifestSource, { publicKeyPem: trustedPublicKey, bootstrapVersion });
    if (version && manifest.version !== version) throw new Error(`manifest version ${manifest.version} does not match requested ${version}`);
    const platform = manifest.platforms["win32-x64"];
    const previousPointer = await readOptionalFile(paths.current);
    const previous = previousPointer ? await currentInstallation(paths) : null;
    if (previousPointer && !previous?.version) throw new Error("current.json is unreadable or incomplete");
    if (previous?.version) await assertInstalledContent(previous.version_dir, previous.version, previous.content_tree_sha256);
    let acceptanceLedger = await ensureAcceptanceLedger(paths, installMarker, previous);
    if (previous?.version && acceptanceLedger.highest) {
      const currentVsHighWater = compareSemver(previous.version, acceptanceLedger.highest.version);
      if (currentVsHighWater > 0) throw new Error("current version exceeds the persistent release acceptance high-water mark");
      if (currentVsHighWater === 0 && previous.bundle_sha256 !== acceptanceLedger.highest.bundle_sha256) {
        throw new Error("current version content conflicts with the persistent release acceptance high-water mark");
      }
    }
    assertAcceptancePolicy(acceptanceLedger, manifest);
    if (previous?.version && compareSemver(manifest.version, previous.version) < 0) {
      throw new Error(`anti-rollback rejected ${manifest.version}; installed version is ${previous.version}; use the explicit rollback command`);
    }
    const archive = path.join(paths.cache, `EvoMind-win-x64-${manifest.version}.zip`);
    await assertNoReparse(paths.cache, archive);
    const cached = await stat(archive).catch(() => null);
    if (!cached || cached.size !== platform.bytes || await sha256File(archive) !== platform.sha256) {
      await rm(archive, { force: true });
      let bundleSource = platform.url;
      if (sourceKind(bundleSource) === "file:" && manifestSource && !path.isAbsolute(bundleSource)) {
        const manifestPath = manifestSource.startsWith("file:") ? fileURLToPath(manifestSource) : path.resolve(manifestSource);
        bundleSource = path.resolve(path.dirname(manifestPath), bundleSource);
      }
      await download(bundleSource, archive, platform.bytes);
    }
    if (await sha256File(archive) !== platform.sha256) throw new Error("downloaded bundle SHA-256 mismatch");
    if (previous?.bundle_sha256 === platform.sha256 && previous.version === manifest.version) {
      const status = runBundleScript(previous.version_dir, "status.ps1", [], paths);
      const health = validateRuntimeStatus(status, manifest.version);
      if (health.passed) {
        acceptanceLedger = await advanceAcceptanceLedger(
          paths, installMarker, acceptanceLedger, manifest, previous.content_tree_sha256,
        );
        return {
          ok: true,
          status: "already_current",
          version: manifest.version,
          strict_health: health,
          acceptance_high_water: acceptanceLedger.highest,
          root: paths.root,
        };
      }
    }

    const previousDir = previous?.version_dir ?? null;
    const stage = path.join(paths.versions, `.stage-${manifest.version}-${process.pid}-${Date.now()}`);
    await mkdir(stage, { recursive: false });
    const versionRoot = path.join(paths.versions, manifest.version);
    const destination = assertContained(paths.versions, path.join(versionRoot, platform.sha256));
    const installStateSnapshot = await readOptionalFile(path.join(paths.data, "install-state.json"));
    const runtimeEnvSnapshot = await readOptionalFile(path.join(paths.data, "config", "runtime.env"));
    const transactionId = randomUUID();
    const pythonEnvironment = releasePythonEnvironmentPaths(paths, platform.sha256, transactionId);
    const journal = {
      schema: TRANSACTION_SCHEMA,
      transaction_id: transactionId,
      install_id: installMarker.install_id,
      created_at: new Date().toISOString(),
      target_version: manifest.version,
      target_bundle_sha256: platform.sha256,
      target_manifest_key_id: manifest.signature.key_id,
      previous_version: previous?.version ?? null,
      previous_directory: previousDir,
      previous_was_running: false,
      previous_pointer: encodeOptional(previousPointer),
      previous_install_state: encodeOptional(installStateSnapshot),
      previous_runtime_env: encodeOptional(runtimeEnvSnapshot),
      database_snapshot: null,
      stage,
      destination,
      destination_created: false,
      target_python_env: pythonEnvironment.target,
      target_python_env_stage: pythonEnvironment.stage,
      target_python_env_quarantine: pythonEnvironment.quarantine,
      target_python_env_quarantined: false,
      python_env_install_started: false,
      acceptance_sequence_before: acceptanceLedger.sequence,
      sequence: 0,
    };
    try {
      await writeTransactionPhase(paths, journal, "prepared");
      await invokeFault(faultInjector, "prepared", journal);
      runPowerShell(path.join(MODULE_ROOT, "scripts", "expand-archive.ps1"), ["-Archive", archive, "-Destination", stage]);
      const bundleRoot = await findBundleRoot(stage);
      const extractedBundle = await verifyBundle(bundleRoot, manifest.version);
      journal.target_content_tree_sha256 = extractedBundle.tree_sha256;
      if (previousDir && existsSync(previousDir)) {
        const previousStatus = runBundleScript(previousDir, "status.ps1", [], paths);
        journal.previous_was_running = observedRunning(previousStatus);
      }
      // Persist the observed running state before the first signal so a hard
      // exit immediately after stop still restarts the previous version.
      await writeTransactionPhase(paths, journal, "prepared");
      if (journal.previous_was_running) stopBundle(previousDir, paths);
      await writeTransactionPhase(paths, journal, "stopped");
      await invokeFault(faultInjector, "stopped", journal);

      journal.database_snapshot = await snapshotDatabase(paths, manifest.version);
      await writeTransactionPhase(paths, journal, "db_snapshotted");
      await invokeFault(faultInjector, "db_snapshotted", journal);

      await assertNoReparse(paths.versions, versionRoot);
      await mkdir(versionRoot, { recursive: true });
      await assertNoReparse(paths.versions, versionRoot, { allowMissingLeaf: false });
      if (existsSync(destination)) {
        const existingBundle = await verifyBundle(destination, manifest.version);
        if (existingBundle.tree_sha256 !== extractedBundle.tree_sha256) {
          throw new Error("immutable content-addressed version directory differs from the signed archive");
        }
        await rm(stage, { recursive: true, force: true });
      } else {
        if (bundleRoot === stage) await rename(stage, destination);
        else { await rename(bundleRoot, destination); await rm(stage, { recursive: true, force: true }); }
        journal.destination_created = true;
      }
      await writeTransactionPhase(paths, journal, "version_staged");
      await invokeFault(faultInjector, "version_staged", journal);

      await mkdir(pythonEnvironment.root, { recursive: true });
      await assertNoReparse(paths.data, pythonEnvironment.root, { allowMissingLeaf: false });
      const priorPythonEnvironment = await lstat(pythonEnvironment.target)
        .catch((error) => (error?.code === "ENOENT" ? null : Promise.reject(error)));
      if (priorPythonEnvironment) {
        await assertManagedPythonEnvironmentPath(paths, pythonEnvironment.target, { allowMissingLeaf: false });
        await assertTreeNoReparse(pythonEnvironment.target);
        if (existsSync(pythonEnvironment.quarantine)) throw new Error("release Python environment quarantine already exists");
        await rename(pythonEnvironment.target, pythonEnvironment.quarantine);
        await syncDirectory(pythonEnvironment.root);
        journal.target_python_env_quarantined = true;
        await writeTransactionPhase(paths, journal, "version_staged");
      }
      journal.python_env_install_started = true;
      await writeTransactionPhase(paths, journal, "python_env_prepared");
      await invokeFault(faultInjector, "python_env_prepared", journal);

      const installResult = runBundleScript(destination, "install.ps1", [
        "-DataDir", paths.data,
        "-LogsDir", paths.logs,
        "-BackupsDir", paths.backups,
        "-ProfilesDir", paths.profiles,
        "-SecretsDir", paths.secrets,
        "-OfflineOnly",
        "-SkipGatewayStart",
        "-SkipMutableReconciliation",
        "-SkipSecretPrompt",
      ], paths, { env: { EVOMIND_RELEASE_TRANSACTION_ID: journal.transaction_id } });
      if (installResult?.ok !== true || installResult?.status !== "installed") throw new Error("bundle install.ps1 did not return a successful structured receipt");
      if (!existsSync(pythonEnvironment.target) || existsSync(pythonEnvironment.stage)) {
        throw new Error("bundle install.ps1 did not atomically publish its Python environment");
      }
      const startResult = runBundleScript(destination, "start.ps1", [], paths);
      if (startResult?.status === "failed" || startResult?.ok === false) throw new Error("new EvoMind version failed to start");
      const statusResult = runBundleScript(destination, "status.ps1", [], paths);
      const health = validateRuntimeStatus(statusResult, manifest.version);
      if (!health.passed) throw new Error(`new EvoMind version failed strict health verification: ${health.failures.join("; ")}`);
      const installedBundle = await verifyBundle(destination, manifest.version);
      if (installedBundle.tree_sha256 !== extractedBundle.tree_sha256) throw new Error("installed content tree mutated after signed extraction");
      await writeTransactionPhase(paths, journal, "health_passed");
      await invokeFault(faultInjector, "health_passed", journal);

      acceptanceLedger = await advanceAcceptanceLedger(
        paths, installMarker, acceptanceLedger, manifest, extractedBundle.tree_sha256,
      );
      journal.acceptance_sequence_after = acceptanceLedger.sequence;
      journal.acceptance_high_water = acceptanceLedger.highest;
      await writeTransactionPhase(paths, journal, "acceptance_advanced");
      await invokeFault(faultInjector, "acceptance_advanced", journal);

      const legacyMigration = await migrateLegacyProfiles(paths);
      const previousVersion = previous?.version && previous.version !== manifest.version
        ? previous.version
        : previous?.previous_version ?? null;
      const directory = path.relative(paths.versions, destination).split(path.sep).join("/");
      await atomicJson(paths.current, {
        schema: "evomind.current_version.v1",
        version: manifest.version,
        directory,
        previous_version: previousVersion,
        previous: previous?.version ? {
          version: previous.version,
          directory: path.relative(paths.versions, previous.version_dir).split(path.sep).join("/"),
          bundle_sha256: previous.bundle_sha256 ?? null,
          content_tree_sha256: previous.content_tree_sha256 ?? null,
        } : null,
        installed_at: new Date().toISOString(),
        manifest_source: manifestSource || DEFAULT_MANIFEST_URL,
        manifest_key_id: manifest.signature.key_id,
        bundle_sha256: platform.sha256,
        content_tree_sha256: extractedBundle.tree_sha256,
        install_id: installMarker.install_id,
        transaction_id: journal.transaction_id,
        transaction: {
          database_backup_receipt: journal.database_snapshot.receipt,
          database_backup_receipt_sha256: journal.database_snapshot.receipt_sha256,
          install_receipt_status: installResult.status,
          strict_health_verified: true,
          acceptance_sequence: acceptanceLedger.sequence,
        },
        acceptance_high_water: {
          version: acceptanceLedger.highest.version,
          published_at: acceptanceLedger.highest.published_at,
          bundle_sha256: acceptanceLedger.highest.bundle_sha256,
        },
      });
      await writeTransactionPhase(paths, journal, "pointer_committed");
      await invokeFault(faultInjector, "pointer_committed", journal);
      await finalizeInstallMutableSideEffects(paths, journal);
      const transactionReceipt = await archiveTransaction(paths, journal, "committed");
      return {
        ok: true,
        status: previous?.version ? "upgraded" : "installed",
        version: manifest.version,
        previous_version: previous?.version ?? null,
        install_receipt: installResult,
        strict_health: health,
        database_backup_receipt: journal.database_snapshot.receipt,
        database_backup_receipt_sha256: journal.database_snapshot.receipt_sha256,
        transaction_receipt: transactionReceipt,
        acceptance_high_water: acceptanceLedger.highest,
        legacy_profile_migration: legacyMigration,
        root: paths.root,
      };
    } catch (error) {
      let recoveryError = null;
      try { await recoverPendingTransaction(paths); }
      catch (rollbackError) { recoveryError = rollbackError; }
      await rm(stage, { recursive: true, force: true }).catch(() => {});
      const suffix = recoveryError ? `; recovery error: ${recoveryError.message}` : "";
      throw new Error(`${error.message}${suffix}`, { cause: error });
    }
  });
}

export async function currentInstallation(paths = layout()) {
  const current = await readJson(paths.current);
  if (!current?.version) return null;
  if (current.schema !== "evomind.current_version.v1") throw new Error("current.json schema is invalid");
  parseSemver(current.version, "current installation version");
  const relative = current.directory ? safeReleaseRelative(current.directory) : safeReleaseRelative(current.version);
  if (relative.split(path.sep)[0] !== current.version) throw new Error("current version directory is not bound to the selected version");
  if (current.bundle_sha256 && current.directory && !relative.endsWith(current.bundle_sha256)) throw new Error("current version directory is not content-addressed by the signed bundle hash");
  if (current.directory && !/^[a-f0-9]{64}$/.test(String(current.content_tree_sha256 || ""))) throw new Error("current version pointer is missing the signed content-tree binding");
  const versionDir = assertContained(paths.versions, path.join(paths.versions, relative));
  await assertNoReparse(paths.versions, versionDir, { allowMissingLeaf: false });
  if (!existsSync(versionDir)) throw new Error("current version directory is missing");
  if (current.install_id) {
    const marker = await loadInstallMarker(paths);
    if (marker.install_id !== current.install_id) throw new Error("current version pointer install identity mismatch");
  }
  return { ...current, version_dir: versionDir };
}

async function assertInstalledContent(versionDir, version, expectedTreeSha256) {
  if (!/^[a-f0-9]{64}$/.test(String(expectedTreeSha256 || ""))) throw new Error("installed version is missing a trusted content-tree hash");
  const verified = await verifyBundle(versionDir, version);
  if (verified.tree_sha256 !== expectedTreeSha256) throw new Error("installed version content-tree hash mismatch");
  return verified;
}

async function assertInstallationWithinAcceptance(paths, current) {
  const marker = await loadInstallMarker(paths);
  const ledger = await loadAcceptanceLedger(paths, marker);
  if (!ledger.highest) throw new Error("release acceptance ledger has no accepted release");
  const ordering = compareSemver(current.version, ledger.highest.version);
  if (ordering > 0) throw new Error("current version exceeds the persistent release acceptance high-water mark");
  if (ordering === 0 && current.bundle_sha256 !== ledger.highest.bundle_sha256) {
    throw new Error("current version content conflicts with the persistent release acceptance high-water mark");
  }
  return ledger;
}

function parseLastJson(text) {
  const candidates = String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    try { return JSON.parse(candidates[index]); } catch { /* continue */ }
  }
  try { return JSON.parse(text); } catch { return { output: text }; }
}

export async function lifecycle(command, { paths = layout(), visible = false } = {}) {
  const current = await currentInstallation(paths);
  if (!current) return { ok: false, status: "not_installed" };
  await assertInstallationWithinAcceptance(paths, current);
  await assertInstalledContent(current.version_dir, current.version, current.content_tree_sha256);
  const scriptName = { start: "start.ps1", stop: "stop.ps1", status: "status.ps1" }[command];
  if (!scriptName) throw new Error(`unsupported lifecycle command: ${command}`);
  const result = command === "stop"
    ? stopBundle(current.version_dir, paths)
    : runBundleScript(current.version_dir, scriptName, [], paths, { visible });
  if (command === "start") {
    const status = runBundleScript(current.version_dir, "status.ps1", [], paths);
    const strictHealth = validateRuntimeStatus(status, current.version);
    let cleanup = null;
    if (!strictHealth.passed) {
      try { cleanup = stopBundle(current.version_dir, paths); }
      catch (error) { cleanup = { ok: false, error: error.message }; }
    }
    return {
      command,
      version: current.version,
      ...result,
      ok: result?.ok !== false && result?.status !== "failed" && strictHealth.passed,
      strict_health: strictHealth,
      status_probe: status,
      cleanup,
    };
  }
  if (command === "status") {
    const strictHealth = validateRuntimeStatus(result, current.version);
    return { command, version: current.version, ...result, ok: strictHealth.passed, strict_health: strictHealth };
  }
  const stopped = command !== "stop" || ["stopped", "not_running"].includes(String(result?.status || ""));
  return { command, version: current.version, ...result, ok: result?.ok !== false && result?.status !== "failed" && stopped };
}

export async function rollback({ paths = layout(), restart = true, faultInjector = null } = {}) {
  return withLock(paths, async () => {
    const current = await currentInstallation(paths);
    await assertInstallationWithinAcceptance(paths, current);
    const previousRecord = current?.previous?.version ? current.previous : current?.previous_version ? { version: current.previous_version, directory: current.previous_version } : null;
    if (!previousRecord?.version) throw new Error("no previous version is available");
    parseSemver(previousRecord.version, "previous installation version");
    const previousRelative = safeReleaseRelative(previousRecord.directory || previousRecord.version);
    if (previousRelative.split(path.sep)[0] !== previousRecord.version) throw new Error("previous version directory is invalid");
    if (!/^[a-f0-9]{64}$/.test(String(previousRecord.bundle_sha256 || "")) || !previousRelative.endsWith(previousRecord.bundle_sha256)) {
      throw new Error("previous version directory is not bound to its signed archive hash");
    }
    const previousDir = assertContained(paths.versions, path.join(paths.versions, previousRelative));
    await assertNoReparse(paths.versions, previousDir, { allowMissingLeaf: false });
    if (!existsSync(previousDir)) throw new Error("previous version directory is missing");
    await assertInstalledContent(current.version_dir, current.version, current.content_tree_sha256);
    await assertInstalledContent(previousDir, previousRecord.version, previousRecord.content_tree_sha256);
    const currentPointer = await readOptionalFile(paths.current);
    const currentStatus = runBundleScript(current.version_dir, "status.ps1", [], paths);
    const currentWasRunning = observedRunning(currentStatus);
    const marker = await loadInstallMarker(paths);
    const journal = {
      schema: TRANSACTION_SCHEMA,
      operation: "rollback",
      transaction_id: randomUUID(),
      install_id: marker.install_id,
      created_at: new Date().toISOString(),
      source_version: current.version,
      source_directory: current.version_dir,
      source_bundle_sha256: current.bundle_sha256,
      source_content_tree_sha256: current.content_tree_sha256,
      source_pointer: encodeOptional(currentPointer),
      source_was_running: currentWasRunning,
      target_version: previousRecord.version,
      target_directory: previousDir,
      target_bundle_sha256: previousRecord.bundle_sha256,
      target_content_tree_sha256: previousRecord.content_tree_sha256,
      target_should_start: restart,
      sequence: 0,
    };
    try {
      await writeTransactionPhase(paths, journal, "rollback_prepared");
      await invokeFault(faultInjector, "rollback_prepared", journal);
      if (currentWasRunning) stopBundle(current.version_dir, paths);
      await writeTransactionPhase(paths, journal, "rollback_current_stopped");
      await invokeFault(faultInjector, "rollback_current_stopped", journal);
      const result = { ok: true, status: "rolled_back", version: previousRecord.version, previous_version: current.version };
      if (restart) {
        await writeTransactionPhase(paths, journal, "rollback_target_starting");
        await invokeFault(faultInjector, "rollback_target_starting", journal);
        result.restart = runBundleScript(previousDir, "start.ps1", [], paths);
        const previousStatus = runBundleScript(previousDir, "status.ps1", [], paths);
        const health = validateRuntimeStatus(previousStatus, previousRecord.version);
        if (!health.passed) throw new Error(`rollback target failed strict health verification: ${health.failures.join("; ")}`);
        result.strict_health = health;
      }
      await writeTransactionPhase(paths, journal, "rollback_target_started");
      await invokeFault(faultInjector, "rollback_target_started", journal);
      const currentRecord = { ...current };
      delete currentRecord.version_dir;
      await atomicJson(paths.current, {
        ...currentRecord,
        version: previousRecord.version,
        directory: previousRelative.split(path.sep).join("/"),
        bundle_sha256: previousRecord.bundle_sha256 ?? null,
        content_tree_sha256: previousRecord.content_tree_sha256,
        previous_version: current.version,
        previous: {
          version: current.version,
          directory: path.relative(paths.versions, current.version_dir).split(path.sep).join("/"),
          bundle_sha256: current.bundle_sha256 ?? null,
          content_tree_sha256: current.content_tree_sha256,
        },
        transaction_id: journal.transaction_id,
        rolled_back_at: new Date().toISOString(),
      });
      await writeTransactionPhase(paths, journal, "rollback_pointer_committed");
      await invokeFault(faultInjector, "rollback_pointer_committed", journal);
      result.transaction_receipt = await archiveTransaction(paths, journal, "rollback_committed");
      return result;
    } catch (error) {
      let recoveryError = null;
      try { await recoverPendingTransaction(paths); }
      catch (rollbackError) { recoveryError = rollbackError; }
      const suffix = recoveryError ? `; recovery error: ${recoveryError.message}` : "";
      throw new Error(`${error.message}${suffix}`, { cause: error });
    }
  });
}

async function verifyLegacyMigration(paths, marker) {
  if (
    marker?.schema !== "evomind.dpapi_migration.v2"
    || marker.source_preserved !== true
    || typeof marker.receipt !== "string"
  ) return { passed: false, detail: "migration marker is malformed" };
  let receiptPath;
  try { receiptPath = assertContained(path.join(paths.backups, "profile-migrations"), marker.receipt); }
  catch { return { passed: false, detail: "migration receipt escapes the managed backup root" }; }
  const receipt = await readJson(receiptPath);
  if (receipt?.schema !== "evomind.dpapi_migration_receipt.v2" || !Array.isArray(receipt.records)) {
    return { passed: false, detail: "migration receipt is missing or malformed" };
  }
  const failures = [];
  const legacyRoot = path.join(path.dirname(paths.roaming), "ResearchAgentWorkstation");
  for (const record of receipt.records) {
    try {
      if (!record || !["profile", "secret"].includes(record.kind) || !/^[a-f0-9]{64}$/.test(String(record.sha256 || ""))) {
        throw new Error("record schema is invalid");
      }
      const base = record.kind === "profile" ? paths.profiles : paths.secrets;
      const destination = assertContained(base, record.destination);
      const source = assertContained(legacyRoot, record.source);
      const destinationInfo = await lstat(destination);
      if (!destinationInfo.isFile() || destinationInfo.isSymbolicLink() || await sha256File(destination) !== record.sha256) {
        throw new Error("destination hash mismatch");
      }
      const sourceInfo = await lstat(source);
      if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink() || await sha256File(source) !== record.sha256) {
        throw new Error("preserved source hash mismatch");
      }
    } catch (error) {
      failures.push({ destination: record?.destination, reason: error.message });
    }
  }
  return { passed: failures.length === 0 && receipt.records.length === marker.migrated_files, detail: { records: receipt.records.length, failures } };
}

export async function doctor(paths = layout()) {
  const checks = [];
  const current = await currentInstallation(paths).catch((error) => ({ error: error.message }));
  checks.push({ name: "installation", passed: Boolean(current && !current.error), detail: current?.error ?? current?.version ?? "not installed" });
  if (current && !current.error) {
    const acceptanceResult = await assertInstallationWithinAcceptance(paths, current)
      .then((ledger) => ({ passed: true, detail: { sequence: ledger.sequence, highest: ledger.highest } }))
      .catch((error) => ({ passed: false, detail: error.message }));
    checks.push({ name: "release_acceptance", ...acceptanceResult });
    const manifestResult = await verifyBundle(current.version_dir, current.version)
      .then((verified) => verified.tree_sha256 === current.content_tree_sha256
        ? ({ passed: true, detail: { content_tree_sha256: verified.tree_sha256 } })
        : ({ passed: false, detail: "installed content tree does not match current.json" }))
      .catch((error) => ({ passed: false, detail: error.message }));
    checks.push({ name: "bundle_integrity", ...manifestResult });
    const directoryChecks = {};
    for (const [name, directory] of Object.entries({ data: paths.data, logs: paths.logs, backups: paths.backups, profiles: paths.profiles, secrets: paths.secrets })) {
      const info = await lstat(directory).catch(() => null);
      directoryChecks[name] = Boolean(info?.isDirectory() && !info.isSymbolicLink());
    }
    checks.push({ name: "managed_directories", passed: Object.values(directoryChecks).every(Boolean), detail: directoryChecks });
    const database = path.join(paths.data, "prisma", "workstation.db");
    const databaseInfo = await lstat(database).catch(() => null);
    checks.push({ name: "database", passed: Boolean(databaseInfo?.isFile() && !databaseInfo.isSymbolicLink()), detail: database });
    const status = await lifecycle("status", { paths }).catch((error) => ({ ok: false, error: error.message }));
    const runtime = validateRuntimeStatus(status, current.version);
    checks.push({ name: "runtime", passed: runtime.passed, detail: { failures: runtime.failures, status } });
    const migrationMarker = path.join(paths.roaming, ".legacy-migration.json");
    const migration = await readJson(migrationMarker);
    if (migration) {
      const migrationAudit = await verifyLegacyMigration(paths, migration);
      checks.push({
        name: "legacy_profile_migration",
        passed: migrationAudit.passed,
        detail: { migrated_files: migration.migrated_files, receipt: migration.receipt, audit: migrationAudit.detail },
      });
    }
  }
  const passed = checks.every((item) => item.passed);
  return { ok: passed, status: passed ? "ready" : "failed", checks, paths: { root: paths.root, data: paths.data, profiles: paths.profiles } };
}

export async function uninstall({ paths = layout(), purgeData = false } = {}) {
  return withLock(paths, async () => {
    const marker = purgeData ? await loadInstallMarker(paths) : null;
    const current = await currentInstallation(paths);
    if (current) {
      const stopped = await lifecycle("stop", { paths });
      if (!stopped.ok) throw new Error("EvoMind uninstall did not release the managed process");
    }
    const appRoot = assertContained(paths.root, path.join(paths.root, "app"));
    const cacheRoot = assertContained(paths.root, path.join(paths.root, "cache"));
    for (const target of [appRoot, cacheRoot]) await assertNoReparse(paths.root, target);
    for (const target of [appRoot, cacheRoot]) await assertTreeNoReparse(target);
    await rm(appRoot, { recursive: true, force: true });
    await rm(cacheRoot, { recursive: true, force: true });
    if (purgeData) {
      const expected = expectedInstallMarker(paths);
      if (marker.install_id !== (await loadInstallMarker(paths)).install_id) throw new Error("install marker changed during purge");
      for (const target of [expected.data, expected.logs, expected.backups]) {
        const safe = assertContained(expected.root, target);
        await assertNoReparse(expected.root, safe);
        await assertTreeNoReparse(safe);
        await rm(safe, { recursive: true, force: true });
      }
      for (const target of [expected.profiles, expected.secrets]) {
        const safe = assertContained(expected.roaming, target);
        await assertNoReparse(expected.roaming, safe);
        await assertTreeNoReparse(safe);
        await rm(safe, { recursive: true, force: true });
      }
      await rm(paths.acceptance, { force: false });
      await rm(paths.installMarker, { force: false });
    }
    return { ok: true, status: "uninstalled", user_data_preserved: !purgeData, root: paths.root };
  });
}

function strictDashboardControlUrl(dashboard) {
  const match = /^http:\/\/127[.]0[.]0[.]1:([1-9]\d{0,4})\/\?page=assistant$/.exec(String(dashboard?.url ?? ""));
  if (!match) throw new Error("EvoMind dashboard reported an invalid local control URL");
  const port = Number(match[1]);
  if (!Number.isSafeInteger(port) || port > 65_535 || String(port) !== match[1]) {
    throw new Error("EvoMind dashboard reported an invalid local control URL");
  }
  if (dashboard.health_url !== undefined && dashboard.health_url !== `http://127.0.0.1:${port}/api/healthz`) {
    throw new Error("EvoMind dashboard reported an invalid local health URL");
  }
  return { url: match[0], port };
}

function dashboardBootstrapPathHints(...roots) {
  const hints = [];
  const seen = new Set();
  const visit = (value, depth) => {
    if (!value || typeof value !== "object" || depth > 8 || seen.has(value)) return;
    seen.add(value);
    if (Object.hasOwn(value, "bootstrap_url_file")) hints.push(value.bootstrap_url_file);
    for (const child of Object.values(value)) visit(child, depth + 1);
  };
  for (const root of roots) visit(root, 0);
  return hints.filter((value) => value !== null && value !== undefined);
}

async function trustedDashboardRuntimeDirectory(paths, reportedDirectory) {
  const runtimeDirectory = assertContained(paths.root, paths.logs);
  await assertNoReparse(paths.root, runtimeDirectory, { allowMissingLeaf: false });
  const info = await lstat(runtimeDirectory).catch(() => null);
  if (!info?.isDirectory() || info.isSymbolicLink()) {
    throw new Error("EvoMind dashboard runtime directory is not trusted");
  }
  if (
    reportedDirectory !== undefined
    && (typeof reportedDirectory !== "string"
      || !path.isAbsolute(reportedDirectory)
      || canonicalCase(reportedDirectory) !== canonicalCase(runtimeDirectory))
  ) {
    throw new Error("EvoMind dashboard reported an untrusted runtime directory");
  }
  return runtimeDirectory;
}

async function readClaimedBootstrapUrl(claimedPath) {
  const pathInfo = await lstat(claimedPath);
  if (!pathInfo.isFile() || pathInfo.isSymbolicLink() || pathInfo.size <= 0 || pathInfo.size > DASHBOARD_BOOTSTRAP_MAX_BYTES) {
    throw new Error("EvoMind dashboard bootstrap claim is malformed");
  }
  const handle = await open(claimedPath, "r");
  try {
    const info = await handle.stat();
    if (!info.isFile() || info.size <= 0 || info.size > DASHBOARD_BOOTSTRAP_MAX_BYTES) {
      throw new Error("EvoMind dashboard bootstrap claim is malformed");
    }
    const body = Buffer.alloc(DASHBOARD_BOOTSTRAP_MAX_BYTES + 1);
    let bytes = 0;
    while (bytes < body.length) {
      const read = await handle.read(body, bytes, body.length - bytes, bytes);
      if (read.bytesRead === 0) break;
      bytes += read.bytesRead;
    }
    const after = await handle.stat();
    if (bytes <= 0 || bytes > DASHBOARD_BOOTSTRAP_MAX_BYTES || after.size !== info.size || after.mtimeMs !== info.mtimeMs) {
      throw new Error("EvoMind dashboard bootstrap claim changed while being read");
    }
    return body.subarray(0, bytes).toString("utf8");
  } finally {
    await handle.close();
  }
}

async function claimDashboardBootstrapUrl(paths, control, pathHints, dashboard) {
  const runtimeDirectory = await trustedDashboardRuntimeDirectory(paths, dashboard.runtime_dir);
  if (dashboard.bootstrap_pending !== undefined && typeof dashboard.bootstrap_pending !== "boolean") {
    throw new Error("EvoMind dashboard reported malformed bootstrap state");
  }
  const suffix = control.port === 8088 ? "" : `.${control.port}`;
  const source = assertContained(runtimeDirectory, path.join(runtimeDirectory, `dashboard${suffix}.bootstrap.once`));
  for (const hint of pathHints) {
    if (typeof hint !== "string" || !path.isAbsolute(hint) || canonicalCase(hint) !== canonicalCase(source)) {
      throw new Error("EvoMind dashboard bootstrap claim escaped the trusted runtime directory");
    }
  }

  const sourceInfo = await lstat(source).catch((error) => {
    if (error?.code === "ENOENT") return null;
    throw error;
  });
  if (!sourceInfo) {
    if (pathHints.length || dashboard.bootstrap_pending === true) {
      throw new Error("EvoMind dashboard bootstrap claim is missing or was already claimed");
    }
    return null;
  }
  if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink() || sourceInfo.size <= 0 || sourceInfo.size > DASHBOARD_BOOTSTRAP_MAX_BYTES) {
    throw new Error("EvoMind dashboard bootstrap claim is not a small regular file");
  }

  const claimed = assertContained(runtimeDirectory, path.join(runtimeDirectory, `.${path.basename(source)}.claimed-${process.pid}-${randomUUID()}`));
  try {
    await rename(source, claimed);
  } catch (error) {
    if (error?.code === "ENOENT" || error?.code === "EEXIST" || error?.code === "EPERM") {
      throw new Error("EvoMind dashboard bootstrap claim could not be acquired");
    }
    throw error;
  }

  let claimedUrl;
  let readError = null;
  let cleanupError = null;
  try {
    claimedUrl = await readClaimedBootstrapUrl(claimed);
  } catch (error) {
    readError = error;
  } finally {
    try {
      await rm(claimed, { force: false });
      await syncDirectory(runtimeDirectory);
    } catch (error) {
      cleanupError = error;
    }
  }
  if (cleanupError) throw new Error("EvoMind dashboard bootstrap claim could not be destroyed");
  if (readError) throw readError;

  const prefix = `${control.url}#bootstrap=`;
  const token = claimedUrl.startsWith(prefix) ? claimedUrl.slice(prefix.length) : "";
  if (!DASHBOARD_BOOTSTRAP_TOKEN.test(token) || claimedUrl !== `${prefix}${token}`) {
    throw new Error("EvoMind dashboard bootstrap claim did not contain the expected local URL");
  }
  return claimedUrl;
}

function openDashboardUrl(url) {
  runPowerShell(path.join(MODULE_ROOT, "scripts", "open-url.ps1"), ["-Url", url], { visible: true });
}

export async function openWorkstation(paths = layout(), { openUrl = openDashboardUrl } = {}) {
  if (typeof openUrl !== "function") throw new Error("EvoMind dashboard URL opener is invalid");
  const current = await currentInstallation(paths);
  if (!current) throw new Error("EvoMind is not installed");
  let status = await lifecycle("status", { paths }).catch(() => ({ ok: false }));
  let started = null;
  if (!validateRuntimeStatus(status, current.version).passed) {
    started = await lifecycle("start", { paths });
    status = await lifecycle("status", { paths });
    const health = validateRuntimeStatus(status, current.version);
    if (!health.passed) throw new Error(`EvoMind failed strict health verification: ${health.failures.join("; ")}`);
  }
  const dashboard = validateRuntimeStatus(status, current.version).dashboard;
  const control = strictDashboardControlUrl(dashboard);
  const pathHints = dashboardBootstrapPathHints(status, started);
  let target = await claimDashboardBootstrapUrl(paths, control, pathHints, dashboard) || control.url;
  try {
    await openUrl(target);
  } catch {
    throw new Error("EvoMind dashboard could not be opened");
  } finally {
    target = "";
  }
  return { ok: true, status: "opened", url: control.url, version: status.version };
}

export { loadManifest, verifyBundle };
