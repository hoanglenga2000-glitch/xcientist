#!/usr/bin/env node
import { createHash, createPrivateKey, createPublicKey, sign } from "node:crypto";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

function stable(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
}

function value(flag, fallback = "") {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const bundleValue = value("--bundle");
const outputValue = value("--output");
const version = value("--version");
const url = value("--url");
const channel = value("--channel", "stable");
const minBootstrapVersion = value("--min-bootstrap-version", "0.3.0");
const publishedAt = value("--published-at", new Date().toISOString());
const privateKeyPath = process.env.EVOMIND_RELEASE_PRIVATE_KEY_FILE;
if (!bundleValue || !outputValue || !version || !url || !privateKeyPath) throw new Error("--bundle, --output, --version, --url and EVOMIND_RELEASE_PRIVATE_KEY_FILE are required");
const semver = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
if (!semver.test(version) || !semver.test(minBootstrapVersion)) throw new Error("release and minimum bootstrap versions must be valid SemVer");
if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/.test(publishedAt) || !Number.isFinite(Date.parse(publishedAt))) {
  throw new Error("--published-at must be an explicit ISO-8601 timestamp");
}
const releaseUrl = new URL(url);
if (releaseUrl.protocol !== "https:") throw new Error("release bundle URL must use HTTPS");
const bundle = path.resolve(bundleValue);
const output = path.resolve(outputValue);
const bytes = (await stat(bundle)).size;
const bundleBody = await readFile(bundle);
const bundleHash = createHash("sha256").update(bundleBody).digest("hex");
const privateKey = createPrivateKey(await readFile(path.resolve(privateKeyPath)));
const publicDer = createPublicKey(privateKey).export({ type: "spki", format: "der" });
const keyId = `evomind-${createHash("sha256").update(publicDer).digest("hex").slice(0, 16)}`;
const unsigned = {
  schema: "evomind.release.manifest.v2",
  version,
  channel,
  published_at: publishedAt,
  min_bootstrap_version: minBootstrapVersion,
  platforms: {
    "win32-x64": { url, sha256: bundleHash, bytes, archive: "zip" },
  },
};
const signatureMetadata = { algorithm: "Ed25519", key_id: keyId };
const authenticated = { ...unsigned, signature: signatureMetadata };
const signature = sign(null, Buffer.from(stable(authenticated), "utf8"), privateKey).toString("base64");
const manifest = { ...unsigned, signature: { ...signatureMetadata, value: signature } };
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
const manifestHash = createHash("sha256").update(await readFile(output)).digest("hex");
await writeFile(`${output}.sha256`, `${manifestHash}  ${path.basename(output)}\n`, "ascii");
process.stdout.write(`${JSON.stringify({ ok: true, output, version, key_id: keyId, bundle_sha256: bundleHash, bundle_bytes: bytes, manifest_sha256: manifestHash }, null, 2)}\n`);
