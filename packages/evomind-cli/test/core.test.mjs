import assert from "node:assert/strict";
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import { cp, mkdtemp, mkdir, readFile, readdir, rm, stat, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { pathToFileURL } from "node:url";

import {
  BUNDLE_SCHEMA,
  BOOTSTRAP_VERSION,
  MANIFEST_SCHEMA,
  currentInstallation,
  doctor,
  installRelease,
  layout,
  lifecycle,
  openWorkstation,
  recoverPendingTransaction,
  rollback,
  safeReleaseRelative,
  sha256File,
  signedPayload,
  stableStringify,
  uninstall,
  verifyManifest,
  verifyBundle,
} from "../src/core.mjs";

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
const testKeyId = `evomind-${createHash("sha256").update(publicKey.export({ type: "spki", format: "der" })).digest("hex").slice(0, 16)}`;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const fixtureStartBootstrapToken = "fixture_start_bootstrap_token_0123456789-ABCDEFG";
const hardExitChildTimeoutMs = 120_000;

function psQuote(value) {
  return String(value).replaceAll("'", "''");
}

async function fixture(root, version, {
  healthy = true,
  minBootstrapVersion = "0.3.0",
  publishedAt = "2026-08-02T00:00:00Z",
  variant = "default",
  bootstrapOnStart = false,
} = {}) {
  const safeVariant = String(variant).replaceAll(/[^A-Za-z0-9_-]/g, "-");
  const bundle = path.join(root, `bundle-${version}-${healthy ? "healthy" : "unhealthy"}-${safeVariant}`);
  await mkdir(bundle, { recursive: true });
  const escapedVersion = psQuote(version);
  const bootstrapStart = bootstrapOnStart
    ? `$bootstrapPath=Join-Path $env:WORKSTATION_RUNTIME_DIR 'dashboard.bootstrap.once'\n[IO.File]::WriteAllText($bootstrapPath,'http://127.0.0.1:8088/?page=assistant#bootstrap=${fixtureStartBootstrapToken}',[Text.UTF8Encoding]::new($false))\n`
    : "$bootstrapPath=$null\n";
  const scripts = {
    "install.ps1": `param([string]$DataDir,[string]$LogsDir,[string]$BackupsDir,[string]$ProfilesDir,[string]$SecretsDir,[switch]$OfflineOnly,[switch]$SkipGatewayStart,[switch]$SkipMutableReconciliation,[switch]$SkipSecretPrompt)\nforeach($dir in @($DataDir,$LogsDir,$BackupsDir,$ProfilesDir,$SecretsDir)){New-Item -ItemType Directory -Force -Path $dir|Out-Null}\nNew-Item -ItemType Directory -Force -Path (Join-Path $DataDir 'prisma')|Out-Null\nNew-Item -ItemType Directory -Force -Path (Split-Path -Parent $env:WORKSTATION_PYTHON)|Out-Null\n[IO.File]::WriteAllText($env:WORKSTATION_PYTHON,'fixture-python-${escapedVersion}',[Text.UTF8Encoding]::new($false))\n[IO.File]::WriteAllText((Join-Path $DataDir 'prisma\\workstation.db'),'${escapedVersion}',[Text.UTF8Encoding]::new($false))\n[IO.File]::WriteAllText((Join-Path $DataDir 'install-state.json'),'{\"version\":\"${escapedVersion}\"}',[Text.UTF8Encoding]::new($false))\n@{ok=$true;status='installed';version='${escapedVersion}';data_dir=$DataDir;logs_dir=$LogsDir;backups_dir=$BackupsDir;profiles_dir=$ProfilesDir;secrets_dir=$SecretsDir}|ConvertTo-Json -Compress\n`,
    "start.ps1": `New-Item -ItemType Directory -Force -Path $env:WORKSTATION_DATA_DIR|Out-Null\n[IO.File]::WriteAllText((Join-Path $env:WORKSTATION_DATA_DIR 'runtime-version'),'${escapedVersion}',[Text.UTF8Encoding]::new($false))\n${bootstrapStart}@{ok=$true;status='started';version='${escapedVersion}';bootstrap_url_file=$bootstrapPath}|ConvertTo-Json -Compress\n`,
    "stop.ps1": `Remove-Item -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'runtime-version') -Force -ErrorAction SilentlyContinue\nRemove-Item -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'legacy-pid-missing') -Force -ErrorAction SilentlyContinue\n@{ok=$true;status='stopped';version='${escapedVersion}'}|ConvertTo-Json -Compress\n`,
    "status.ps1": `$running=(Test-Path -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'runtime-version')) -and ((Get-Content -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'runtime-version') -Raw) -eq '${escapedVersion}')\n$forced=Test-Path -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'force-unhealthy')\n$legacyPidMissing=Test-Path -LiteralPath (Join-Path $env:WORKSTATION_DATA_DIR 'legacy-pid-missing')\n$bootstrapHintFile=Join-Path $env:WORKSTATION_DATA_DIR 'bootstrap-path-hint'\n$dashboardUrlFile=Join-Path $env:WORKSTATION_DATA_DIR 'dashboard-url-override'\n$bootstrapHint=$(if(Test-Path -LiteralPath $bootstrapHintFile){[IO.File]::ReadAllText($bootstrapHintFile)}else{$null})\n$dashboardUrl=$(if(Test-Path -LiteralPath $dashboardUrlFile){[IO.File]::ReadAllText($dashboardUrlFile)}else{'http://127.0.0.1:8088/?page=assistant'})\n$bootstrapPending=Test-Path -LiteralPath (Join-Path $env:WORKSTATION_RUNTIME_DIR 'dashboard.bootstrap.once')\n$health=$null\nif($running -and -not $forced -and $${healthy}){$health=@{reachable=$true;http_status=200;version='${escapedVersion}';has_runtime=$true;runtime=@{reachable=$true;http_status=200;status='ready';version='${escapedVersion}'}}}\n$pidBound=[bool]($running -and -not $legacyPidMissing)\n$dashboard=@{status=$(if($running){'running'}else{'not_reachable'});pid_running=$pidBound;process_port_consistent=$pidBound;runtime_process_consistent=$pidBound;identity_contract='evomind.process_identity.v1';dashboard_identity_verified=$pidBound;runtime_identity_verified=$pidBound;runtime_pid_running=$pidBound;runtime_listener_pids=$(if($pidBound){@(12345)}else{@()});url=$dashboardUrl;health_url='http://127.0.0.1:8088/api/healthz';runtime_dir=$env:WORKSTATION_RUNTIME_DIR;bootstrap_pending=$bootstrapPending;health=$health}\nif($null -ne $bootstrapHint){$dashboard['bootstrap_url_file']=$bootstrapHint}\n@{status='ok';install=@{ok=$true;exit_code=0;output=(@{status='installed'}|ConvertTo-Json -Compress)};dashboard=@{ok=$true;exit_code=0;output=($dashboard|ConvertTo-Json -Depth 6 -Compress)}}|ConvertTo-Json -Depth 8 -Compress\n`,
  };
  for (const [name, source] of Object.entries(scripts)) await writeFile(path.join(bundle, name), source, "utf8");
  await writeFile(path.join(bundle, "build-id.txt"), `${version}:${variant}\n`, "utf8");
  const files = [];
  for (const name of [...Object.keys(scripts), "build-id.txt"].sort()) {
    files.push({ path: name, bytes: (await stat(path.join(bundle, name))).size, sha256: await sha256File(path.join(bundle, name)) });
  }
  await writeFile(path.join(bundle, "release-manifest.json"), `${JSON.stringify({ schema: BUNDLE_SCHEMA, version, files }, null, 2)}\n`, "utf8");
  const archive = path.join(root, `EvoMind-win-x64-${version}-${healthy ? "healthy" : "unhealthy"}-${safeVariant}.zip`);
  const compressed = spawnSync("tar.exe", ["-a", "-c", "-f", archive, "-C", root, path.basename(bundle)], { encoding: "utf8", windowsHide: true });
  assert.equal(compressed.status, 0, compressed.stderr);
  const platform = { url: archive, bytes: (await stat(archive)).size, sha256: await sha256File(archive), archive: "zip" };
  const unsigned = {
    schema: MANIFEST_SCHEMA,
    version,
    channel: "test",
    published_at: publishedAt,
    min_bootstrap_version: minBootstrapVersion,
    platforms: { "win32-x64": platform },
  };
  const signatureMetadata = { algorithm: "Ed25519", key_id: testKeyId };
  const authenticated = { ...unsigned, signature: signatureMetadata };
  const manifest = {
    ...unsigned,
    signature: {
      ...signatureMetadata,
      value: sign(null, Buffer.from(stableStringify(authenticated)), privateKey).toString("base64"),
    },
  };
  assert.deepEqual(signedPayload(manifest), Buffer.from(stableStringify(authenticated)));
  const manifestPath = path.join(root, `latest-${version}-${healthy ? "healthy" : "unhealthy"}-${safeVariant}.json`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return { archive, bundle, manifest, manifestPath };
}

test("manifest binds key_id, enforces minimum bootstrap, and ignores trust-root environment override", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-cli-signature-"));
  const packageMetadata = JSON.parse(await readFile(path.join(repoRoot, "packages", "evomind-cli", "package.json"), "utf8"));
  assert.equal(BOOTSTRAP_VERSION, packageMetadata.version);
  const { manifest } = await fixture(root, "0.3.0");
  const futureBootstrap = await fixture(root, "0.3.1", { minBootstrapVersion: "0.4.0" });
  assert.equal(verifyManifest(manifest, publicKey).version, "0.3.0");
  assert.throws(() => verifyManifest({ ...manifest, version: "9.9.9" }, publicKey), /signature verification failed/);
  assert.throws(() => verifyManifest({ ...manifest, signature: { ...manifest.signature, key_id: "other-key" } }, publicKey), /key_id does not match/);
  assert.throws(() => verifyManifest({ ...manifest, schema: "evomind.release.manifest.v1" }, publicKey), /unsupported release manifest schema/);
  assert.throws(() => verifyManifest({ ...manifest, min_bootstrap_version: "0.4.0" }, publicKey), /requires bootstrap/);
  assert.throws(() => verifyManifest(futureBootstrap.manifest, publicKey), /requires bootstrap/);
  const malicious = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" });
  const maliciousPath = path.join(root, "malicious.pem");
  await writeFile(maliciousPath, malicious);
  const previousOverride = process.env.EVOMIND_TRUSTED_PUBLIC_KEY_FILE;
  process.env.EVOMIND_TRUSTED_PUBLIC_KEY_FILE = maliciousPath;
  try {
    assert.throws(() => verifyManifest(manifest), /key_id does not match|signature verification failed/);
  } finally {
    if (previousOverride === undefined) delete process.env.EVOMIND_TRUSTED_PUBLIC_KEY_FILE;
    else process.env.EVOMIND_TRUSTED_PUBLIC_KEY_FILE = previousOverride;
  }
});

test("release signer emits the same key_id-bound payload verified by the bootstrapper", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-cli-signer-"));
  const bundle = path.join(root, "bundle.zip");
  const manifestPath = path.join(root, "latest.json");
  const privateKeyPath = path.join(root, "release-private.pem");
  await writeFile(bundle, "signed bundle fixture", "utf8");
  await writeFile(privateKeyPath, privateKey.export({ type: "pkcs8", format: "pem" }));
  const signed = spawnSync(process.execPath, [
    path.join(repoRoot, "scripts", "sign_evomind_release_manifest.mjs"),
    "--bundle", bundle,
    "--output", manifestPath,
    "--version", "0.3.0",
    "--url", "https://releases.example.invalid/releases/0.3.0/bundle.zip",
    "--min-bootstrap-version", "0.3.0",
    "--published-at", "2026-08-03T00:00:00Z",
  ], {
    encoding: "utf8",
    windowsHide: true,
    env: { ...process.env, EVOMIND_RELEASE_PRIVATE_KEY_FILE: privateKeyPath },
  });
  assert.equal(signed.status, 0, signed.stderr);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.equal(manifest.signature.key_id, testKeyId);
  assert.equal(manifest.published_at, "2026-08-03T00:00:00Z");
  assert.equal(verifyManifest(manifest, publicKey).version, "0.3.0");
});

test("install and upgrade are transactional, migrate DPAPI files, and doctor checks nested health", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-cli-lifecycle-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const legacy = path.join(root, "roaming", "ResearchAgentWorkstation");
  await mkdir(path.join(legacy, "profiles", "job90353"), { recursive: true });
  await writeFile(path.join(legacy, "profiles", "job90353", "credential.xml"), "dpapi-profile", "utf8");
  await writeFile(path.join(legacy, "openai_api_key.xml"), "dpapi-root-secret", "utf8");

  const first = await fixture(root, "0.3.0");
  const failedUpgrade = await fixture(root, "0.4.0", { healthy: false });
  const installed = await installRelease({ manifestSource: first.manifestPath, paths, trustedPublicKey: publicKey });
  assert.equal(installed.status, "installed");
  assert.equal(installed.strict_health.passed, true);
  assert.equal(installed.legacy_profile_migration.status, "migrated");
  const migrationReceiptRaw = await readFile(installed.legacy_profile_migration.receipt, "utf8");
  const migrationReceipt = JSON.parse(migrationReceiptRaw);
  assert.equal(migrationReceipt.schema, "evomind.dpapi_migration_receipt.v2");
  assert.equal(migrationReceipt.source_preserved, true);
  assert.equal(migrationReceipt.records.length, 2);
  assert.equal(migrationReceipt.rollback.destinations.length, 2);
  assert.equal(migrationReceiptRaw.includes("dpapi-profile"), false);
  assert.equal(migrationReceiptRaw.includes("dpapi-root-secret"), false);
  assert.equal((await currentInstallation(paths)).version, "0.3.0");
  assert.equal(await readFile(path.join(paths.data, "prisma", "workstation.db"), "utf8"), "0.3.0");
  assert.deepEqual(JSON.parse(await readFile(path.join(paths.data, "install-state.json"), "utf8")), { version: "0.3.0" });
  assert.equal(await readFile(path.join(paths.profiles, "job90353", "credential.xml"), "utf8"), "dpapi-profile");
  assert.equal(await readFile(path.join(paths.secrets, "openai_api_key.xml"), "utf8"), "dpapi-root-secret");
  assert.equal(await readFile(path.join(legacy, "openai_api_key.xml"), "utf8"), "dpapi-root-secret");
  assert.equal((await doctor(paths)).ok, true);
  await writeFile(path.join(paths.secrets, "openai_api_key.xml"), "tampered", "utf8");
  const tamperedProfileDoctor = await doctor(paths);
  assert.equal(tamperedProfileDoctor.checks.find((item) => item.name === "legacy_profile_migration")?.passed, false);
  await writeFile(path.join(paths.secrets, "openai_api_key.xml"), "dpapi-root-secret", "utf8");

  await assert.rejects(
    installRelease({ manifestSource: failedUpgrade.manifestPath, paths, trustedPublicKey: publicKey }),
    /strict health verification/,
  );
  assert.equal((await currentInstallation(paths)).version, "0.3.0");
  assert.equal(await readFile(path.join(paths.data, "prisma", "workstation.db"), "utf8"), "0.3.0");
  assert.deepEqual(JSON.parse(await readFile(path.join(paths.data, "install-state.json"), "utf8")), { version: "0.3.0" });
  const transactionDirectories = await readdir(path.join(paths.backups, "transactions"));
  const rollbackReceipts = await Promise.all(transactionDirectories.map(async (name) => {
    const receipt = path.join(paths.backups, "transactions", name, "rollback-receipt.json");
    return readFile(receipt, "utf8").then(JSON.parse).catch(() => null);
  }));
  assert.equal(rollbackReceipts.some((receipt) => receipt?.verified === true), true);
  assert.equal((await doctor(paths)).ok, true);

  await writeFile(path.join(paths.data, "force-unhealthy"), "1", "utf8");
  const unhealthyDoctor = await doctor(paths);
  assert.equal(unhealthyDoctor.ok, false);
  assert.equal(unhealthyDoctor.checks.find((item) => item.name === "runtime")?.passed, false);
  await rm(path.join(paths.data, "force-unhealthy"));

  const second = await fixture(root, "0.4.0", { healthy: true });
  await writeFile(path.join(paths.data, "legacy-pid-missing"), "legacy-layout", "utf8");
  await assert.rejects(
    installRelease({ manifestSource: second.manifestPath, paths, trustedPublicKey: publicKey }),
    /process identity conflict/,
  );
  assert.equal(await readFile(path.join(paths.data, "legacy-pid-missing"), "utf8"), "legacy-layout");
  await rm(path.join(paths.data, "legacy-pid-missing"));
  const upgraded = await installRelease({ manifestSource: second.manifestPath, paths, trustedPublicKey: publicKey });
  assert.equal(upgraded.status, "upgraded");
  await assert.rejects(
    installRelease({ manifestSource: first.manifestPath, paths, trustedPublicKey: publicKey }),
    /anti-rollback.*rejected/,
  );
  assert.equal((await currentInstallation(paths)).previous_version, "0.3.0");
  const rolled = await rollback({ paths, restart: true });
  assert.equal(rolled.version, "0.3.0");
  assert.equal(rolled.strict_health.passed, true);
  assert.equal(Object.hasOwn(JSON.parse(await readFile(paths.current, "utf8")), "version_dir"), false);
  const removed = await uninstall({ paths, purgeData: false });
  assert.equal(removed.user_data_preserved, true);
  assert.equal(await readFile(path.join(paths.data, "prisma", "workstation.db"), "utf8"), "0.4.0");
  const attackerData = path.join(root, "attacker-controlled-data");
  await mkdir(attackerData);
  await writeFile(path.join(attackerData, "keep.txt"), "must survive", "utf8");
  await assert.rejects(
    uninstall({ paths: { ...paths, data: attackerData }, purgeData: true }),
    /environment-controlled lifecycle paths|install marker data path/,
  );
  assert.equal(await readFile(path.join(attackerData, "keep.txt"), "utf8"), "must survive");
  const purged = await uninstall({ paths, purgeData: true });
  assert.equal(purged.user_data_preserved, false);
  await assert.rejects(readFile(path.join(paths.data, "prisma", "workstation.db")), /ENOENT/);
});

test("open claims dashboard bootstrap URLs once without exposing the fragment", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-cli-open-bootstrap-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const release = await fixture(root, "0.3.0", { variant: "open-bootstrap", bootstrapOnStart: true });
  await installRelease({ manifestSource: release.manifestPath, paths, trustedPublicKey: publicKey });

  const baseUrl = "http://127.0.0.1:8088/?page=assistant";
  const onceFile = path.join(paths.logs, "dashboard.bootstrap.once");
  const hintFile = path.join(paths.data, "bootstrap-path-hint");
  const dashboardUrlOverride = path.join(paths.data, "dashboard-url-override");
  const token = "test-bootstrap-token-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const fragmentUrl = `${baseUrl}#bootstrap=${token}`;
  const opened = [];

  await writeFile(onceFile, fragmentUrl, { encoding: "utf8", mode: 0o600 });
  await writeFile(hintFile, onceFile, "utf8");
  const result = await openWorkstation(paths, { openUrl: async (url) => opened.push(url) });
  assert.deepEqual(opened, [fragmentUrl]);
  assert.deepEqual(result, { ok: true, status: "opened", url: baseUrl, version: "0.3.0" });
  assert.equal(JSON.stringify(result).includes(token), false);
  assert.equal(JSON.stringify(result).includes("#bootstrap"), false);
  assert.equal(await stat(onceFile).then(() => true, () => false), false);
  assert.equal((await readdir(paths.logs)).some((name) => name.includes(".claimed-")), false);
  const openedBeforeReplay = opened.length;
  await assert.rejects(
    openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
    /bootstrap claim is missing or was already claimed/,
  );
  assert.equal(opened.length, openedBeforeReplay);
  for (const entry of await readdir(paths.logs, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const logBody = await readFile(path.join(paths.logs, entry.name), "utf8");
    assert.equal(logBody.includes(token), false);
    assert.equal(logBody.includes("#bootstrap"), false);
  }
  await rm(hintFile);

  opened.length = 0;
  const withoutToken = await openWorkstation(paths, { openUrl: async (url) => opened.push(url) });
  assert.deepEqual(opened, [baseUrl]);
  assert.equal(withoutToken.url, baseUrl);

  for (const body of [
    `${baseUrl}&next=outside#bootstrap=${token}`,
    `${baseUrl}#bootstrap=${token}&extra=value`,
    `http://127.0.0.1:8089/?page=assistant#bootstrap=${token}`,
    `${baseUrl}#bootstrap=too-short`,
    `${baseUrl}#bootstrap=${token}!`,
  ]) {
    await writeFile(onceFile, body, "utf8");
    await assert.rejects(
      openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
      /bootstrap claim did not contain the expected local URL/,
    );
    assert.equal(await stat(onceFile).then(() => true, () => false), false);
  }

  await writeFile(onceFile, "x".repeat(1025), "utf8");
  await assert.rejects(
    openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
    /small regular file/,
  );
  await rm(onceFile);
  await mkdir(onceFile);
  await assert.rejects(
    openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
    /small regular file/,
  );
  await rm(onceFile, { recursive: true });
  const symlinkTarget = path.join(root, "bootstrap-symlink-target.txt");
  await writeFile(symlinkTarget, fragmentUrl, "utf8");
  let symlinkCreated = false;
  try {
    await symlink(symlinkTarget, onceFile, "file");
    symlinkCreated = true;
  } catch (error) {
    if (!["EPERM", "EACCES", "UNKNOWN"].includes(error?.code)) throw error;
  }
  if (symlinkCreated) {
    await assert.rejects(
      openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
      /small regular file/,
    );
    await rm(onceFile);
  }
  await rm(symlinkTarget);

  const outsideClaim = path.join(root, "outside.bootstrap.once");
  await writeFile(outsideClaim, fragmentUrl, "utf8");
  await writeFile(hintFile, outsideClaim, "utf8");
  await assert.rejects(
    openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
    /escaped the trusted runtime directory/,
  );
  assert.equal(await readFile(outsideClaim, "utf8"), fragmentUrl);
  await rm(hintFile);
  await rm(outsideClaim);

  for (const unsafeUrl of [
    "http://localhost:8088/?page=assistant",
    "http://127.0.0.1:8088/?page=assistant&extra=1",
    "https://127.0.0.1:8088/?page=assistant",
    "http://127.0.0.1:8088/?page=assistant#bootstrap=fragment",
  ]) {
    await writeFile(dashboardUrlOverride, unsafeUrl, "utf8");
    await assert.rejects(
      openWorkstation(paths, { openUrl: async (url) => opened.push(url) }),
      /invalid local control URL/,
    );
  }
  await rm(dashboardUrlOverride);

  const concurrentToken = "concurrent_bootstrap_token_0123456789-ABCDEFG";
  const concurrentUrl = `${baseUrl}#bootstrap=${concurrentToken}`;
  const concurrentOpened = [];
  await writeFile(onceFile, concurrentUrl, "utf8");
  const concurrent = await Promise.allSettled([
    openWorkstation(paths, { openUrl: async (url) => concurrentOpened.push(url) }),
    openWorkstation(paths, { openUrl: async (url) => concurrentOpened.push(url) }),
  ]);
  assert.equal(concurrentOpened.filter((url) => url === concurrentUrl).length, 1);
  assert.equal(JSON.stringify(concurrent).includes(concurrentToken), false);
  assert.equal(await stat(onceFile).then(() => true, () => false), false);

  const openerFailureToken = "opener_failure_bootstrap_token_0123456789-ABCDE";
  await writeFile(onceFile, `${baseUrl}#bootstrap=${openerFailureToken}`, "utf8");
  await assert.rejects(
    openWorkstation(paths, { openUrl: async (url) => { throw new Error(url); } }),
    (error) => {
      assert.equal(error.message, "EvoMind dashboard could not be opened");
      assert.equal(error.message.includes(openerFailureToken), false);
      assert.equal(error.message.includes("#bootstrap"), false);
      return true;
    },
  );
  assert.equal(await stat(onceFile).then(() => true, () => false), false);

  const stopped = await lifecycle("stop", { paths });
  assert.equal(stopped.ok, true);
  const startedOpen = [];
  await openWorkstation(paths, { openUrl: async (url) => startedOpen.push(url) });
  assert.deepEqual(startedOpen, [`${baseUrl}#bootstrap=${fixtureStartBootstrapToken}`]);
  assert.equal(await stat(onceFile).then(() => true, () => false), false);

  await uninstall({ paths, purgeData: true });
});

test("Windows release paths reject drive-relative, ADS, devices, trailing aliases, and case collisions", async () => {
  assert.equal(safeReleaseRelative("app/server.js"), path.join("app", "server.js"));
  for (const candidate of ["C:escape.txt", "C:\\escape.txt", "app/file.txt:stream", "app/CON", "app/name. ", "../escape"] ) {
    assert.throws(() => safeReleaseRelative(candidate), /unsafe|forbidden|stream|device|trailing/);
  }
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-case-collision-"));
  const bundle = path.join(root, "bundle");
  await mkdir(path.join(bundle, "app"), { recursive: true });
  await writeFile(path.join(bundle, "app", "A.txt"), "a");
  const digest = await sha256File(path.join(bundle, "app", "A.txt"));
  await writeFile(path.join(bundle, "release-manifest.json"), JSON.stringify({
    schema: BUNDLE_SCHEMA,
    version: "0.3.0",
    files: [
      { path: "app/A.txt", bytes: 1, sha256: digest },
      { path: "app/a.txt", bytes: 1, sha256: digest },
    ],
  }));
  await assert.rejects(verifyBundle(bundle, "0.3.0"), /case-fold path collision/);
});

test("signed content binding rejects a pre-created self-consistent destination", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-precreated-content-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const signed = await fixture(root, "0.3.0", { variant: "signed" });
  const attacker = await fixture(root, "0.3.0", { variant: "attacker" });
  const destination = path.join(paths.versions, "0.3.0", signed.manifest.platforms["win32-x64"].sha256);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(attacker.bundle, destination, { recursive: true, errorOnExist: true });
  await assert.rejects(
    installRelease({ manifestSource: signed.manifestPath, paths, trustedPublicKey: publicKey }),
    /immutable content-addressed version directory differs/,
  );
  assert.equal(await readFile(path.join(destination, "build-id.txt"), "utf8"), "0.3.0:attacker\n");
});

test("lifecycle lock distinguishes PID reuse and fails closed on malformed identity", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-lock-identity-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const release = await fixture(root, "0.3.0", { variant: "lock-identity" });
  await mkdir(paths.root, { recursive: true });
  await writeFile(paths.lock, JSON.stringify({
    schema: "evomind.lifecycle_lock.v2",
    owner: { pid: process.pid, creation_time: "reused-pid", executable: process.execPath },
    nonce: "00000000-0000-4000-8000-000000000000",
    started_at: new Date().toISOString(),
    started_ms: Date.now(),
  }));
  const installed = await installRelease({ manifestSource: release.manifestPath, paths, trustedPublicKey: publicKey });
  assert.equal(installed.status, "installed");
  assert.equal((await readdir(paths.root)).some((name) => name.startsWith(".lifecycle.lock.stale-")), true);

  await writeFile(paths.lock, JSON.stringify({ schema: "evomind.lifecycle_lock.v2", owner: { pid: process.pid } }));
  await assert.rejects(uninstall({ paths, purgeData: true }), /lifecycle lock is malformed/);
  await rm(paths.lock);
  await uninstall({ paths, purgeData: true });
});

test("persistent acceptance high-water rejects downgrade and same-version replay after rollback", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-acceptance-ledger-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const baseline = await fixture(root, "0.3.0", { publishedAt: "2026-08-02T00:00:00Z", variant: "baseline" });
  const oldBuild = await fixture(root, "0.4.0", { publishedAt: "2026-08-02T01:00:00Z", variant: "old" });
  const newBuild = await fixture(root, "0.4.0", { publishedAt: "2026-08-02T02:00:00Z", variant: "new" });

  await installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey });
  await installRelease({ manifestSource: oldBuild.manifestPath, paths, trustedPublicKey: publicKey });
  await rollback({ paths, restart: true });
  await assert.rejects(
    installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey }),
    /anti-rollback high-water rejected/,
  );
  const upgraded = await installRelease({ manifestSource: newBuild.manifestPath, paths, trustedPublicKey: publicKey });
  assert.equal(upgraded.acceptance_high_water.bundle_sha256, newBuild.manifest.platforms["win32-x64"].sha256);
  await assert.rejects(
    installRelease({ manifestSource: oldBuild.manifestPath, paths, trustedPublicKey: publicKey }),
    /replayed or non-monotonic/,
  );

  const trustedLedger = await readFile(paths.acceptance);
  await writeFile(paths.acceptance, "{truncated", "utf8");
  assert.equal((await doctor(paths)).checks.find((item) => item.name === "release_acceptance")?.passed, false);
  await assert.rejects(
    installRelease({ manifestSource: newBuild.manifestPath, paths, trustedPublicKey: publicKey }),
    /release-acceptance\.json is malformed JSON/,
  );
  await writeFile(paths.acceptance, trustedLedger);

  const preserved = await uninstall({ paths, purgeData: false });
  assert.equal(preserved.user_data_preserved, true);
  await assert.rejects(
    installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey }),
    /anti-rollback high-water rejected/,
  );
  await uninstall({ paths, purgeData: true });
  const fresh = await installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey });
  assert.equal(fresh.status, "installed");
  await uninstall({ paths, purgeData: true });
});

test("rollback verifies the trusted content tree before executing the target", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-rollback-integrity-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const baseline = await fixture(root, "0.3.0", { variant: "rollback-source" });
  const treatment = await fixture(root, "0.4.0", { publishedAt: "2026-08-02T01:00:00Z", variant: "rollback-current" });
  await installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey });
  await installRelease({ manifestSource: treatment.manifestPath, paths, trustedPublicKey: publicKey });
  const current = await currentInstallation(paths);
  const previousDir = path.join(paths.versions, ...current.previous.directory.split("/"));
  const startPath = path.join(previousDir, "start.ps1");
  const original = await readFile(startPath);
  await writeFile(startPath, `${original.toString("utf8")}\n# tampered\n`, "utf8");
  await assert.rejects(rollback({ paths, restart: true }), /bundle file verification failed|content-tree hash mismatch/);
  await writeFile(startPath, original);
  await uninstall({ paths, purgeData: true });
});

test("durable journal recovers real hard exits at every release phase", { timeout: 900_000 }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-cli-hard-kill-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const baseline = await fixture(root, "0.3.0");
  const treatment = await fixture(root, "0.4.0");
  await installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey });
  const publicKeyPath = path.join(root, "test-public.pem");
  await writeFile(publicKeyPath, publicKey.export({ type: "spki", format: "pem" }));
  const coreUrl = pathToFileURL(path.join(repoRoot, "packages", "evomind-cli", "src", "core.mjs")).href;
  const childSource = `
    import { readFile } from "node:fs/promises";
    const [coreUrl, manifest, keyPath, localBase, roamingBase, phase] = process.argv.slice(1);
    const { installRelease, layout } = await import(coreUrl);
    const key = await readFile(keyPath, "utf8");
    await installRelease({
      manifestSource: manifest,
      paths: layout({ EVOMIND_LOCALAPPDATA: localBase, EVOMIND_APPDATA: roamingBase }),
      trustedPublicKey: key,
      faultInjector: async (observed) => { if (observed === phase) process.exit(91); },
    });
  `;
  for (const phase of [
    "prepared", "stopped", "db_snapshotted", "version_staged", "python_env_prepared",
    "health_passed", "acceptance_advanced", "pointer_committed",
  ]) {
    const child = spawnSync(process.execPath, [
      "--input-type=module", "-e", childSource,
      coreUrl, treatment.manifestPath, publicKeyPath,
      path.join(root, "local"), path.join(root, "roaming"), phase,
    ], { encoding: "utf8", windowsHide: true, timeout: hardExitChildTimeoutMs });
    assert.equal(child.status, 91, `${phase}: status=${child.status} signal=${child.signal} error=${child.error?.message || ""}\n${child.stderr || child.stdout}`);
    assert.equal(await stat(paths.journal).then(() => true, () => false), true, phase);
    if (phase === "acceptance_advanced") {
      const recovered = await recoverPendingTransaction(paths);
      assert.equal(recovered.recovered, true, phase);
    } else {
      const recoveryManifest = phase === "pointer_committed" ? treatment.manifestPath : baseline.manifestPath;
      const recovered = await installRelease({ manifestSource: recoveryManifest, paths, trustedPublicKey: publicKey });
      assert.equal(recovered.status, "already_current", phase);
    }
    assert.equal((await currentInstallation(paths)).version, phase === "pointer_committed" ? "0.4.0" : "0.3.0");
    assert.equal(await stat(paths.journal).then(() => true, () => false), false, phase);
    if (phase === "pointer_committed") break;
  }
  const staleLocks = (await readdir(paths.root)).filter((name) => name.startsWith(".lifecycle.lock.stale-"));
  assert.ok(staleLocks.length >= 8);
  await uninstall({ paths, purgeData: true });
});

test("durable rollback journal recovers hard exits without process-pointer split", { timeout: 600_000 }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-rollback-hard-kill-"));
  const paths = layout({ EVOMIND_LOCALAPPDATA: path.join(root, "local"), EVOMIND_APPDATA: path.join(root, "roaming") });
  const baseline = await fixture(root, "0.3.0", { variant: "rollback-base" });
  const treatment = await fixture(root, "0.4.0", { publishedAt: "2026-08-02T01:00:00Z", variant: "rollback-treatment" });
  await installRelease({ manifestSource: baseline.manifestPath, paths, trustedPublicKey: publicKey });
  await installRelease({ manifestSource: treatment.manifestPath, paths, trustedPublicKey: publicKey });
  const coreUrl = pathToFileURL(path.join(repoRoot, "packages", "evomind-cli", "src", "core.mjs")).href;
  const childSource = `
    const [coreUrl, localBase, roamingBase, phase] = process.argv.slice(1);
    const { layout, rollback } = await import(coreUrl);
    await rollback({
      paths: layout({ EVOMIND_LOCALAPPDATA: localBase, EVOMIND_APPDATA: roamingBase }),
      restart: true,
      faultInjector: async (observed) => { if (observed === phase) process.exit(92); },
    });
  `;
  for (const phase of [
    "rollback_prepared", "rollback_current_stopped", "rollback_target_starting",
    "rollback_target_started", "rollback_pointer_committed",
  ]) {
    const child = spawnSync(process.execPath, [
      "--input-type=module", "-e", childSource,
      coreUrl, path.join(root, "local"), path.join(root, "roaming"), phase,
    ], { encoding: "utf8", windowsHide: true, timeout: hardExitChildTimeoutMs });
    assert.equal(child.status, 92, `${phase}: status=${child.status} signal=${child.signal} error=${child.error?.message || ""}\n${child.stderr || child.stdout}`);
    const recovered = await recoverPendingTransaction(paths);
    assert.equal(recovered.recovered, true, phase);
    assert.equal((await currentInstallation(paths)).version, phase === "rollback_pointer_committed" ? "0.3.0" : "0.4.0");
    if (phase === "rollback_pointer_committed") break;
  }
  await uninstall({ paths, purgeData: true });
});
