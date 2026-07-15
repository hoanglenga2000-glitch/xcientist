import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workstationRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tempRoot = await mkdtemp(path.join(tmpdir(), "evomind-upgrade-campaign-"));
const fixtureRoot = path.join(tempRoot, "source-fixture");
const profileRoot = path.join(tempRoot, "profile");
const portProbe = createServer();

await new Promise((resolve, reject) => {
  portProbe.once("error", reject);
  portProbe.listen(0, "127.0.0.1", resolve);
});
const address = portProbe.address();
assert(address && typeof address === "object");
const port = address.port;
await new Promise((resolve) => portProbe.close(resolve));

const fixturePackage = path.join(fixtureRoot, "src", "xsci");
await mkdir(fixturePackage, { recursive: true });
await mkdir(profileRoot, { recursive: true });
await writeFile(path.join(fixturePackage, "__init__.py"), "", "utf8");
await writeFile(
  path.join(fixturePackage, "kaggle.py"),
  [
    "import json",
    "print(json.dumps({",
    "    'ok': False,",
    "    'status': 'blocked',",
    "    'error': 'Upgrade campaign requires a Git repository',",
    "    'parity_claim_allowed': True,",
    "    'score_cap': 100,",
    "    'blockers': ['missing_git_repository'],",
    "}))",
    "raise SystemExit(1)",
    "",
  ].join("\n"),
  "utf8",
);

const isolatedDirectories = {
  USERPROFILE: profileRoot,
  APPDATA: path.join(profileRoot, "AppData", "Roaming"),
  LOCALAPPDATA: path.join(profileRoot, "AppData", "Local"),
  TEMP: path.join(profileRoot, "Temp"),
  TMP: path.join(profileRoot, "Temp"),
};
await Promise.all(Object.values(isolatedDirectories).map((directory) => mkdir(directory, { recursive: true })));

const env = {
  ...process.env,
  ...isolatedDirectories,
  NODE_ENV: "production",
  WORKSTATION_ROOT: fixtureRoot,
  WORKSTATION_PYTHON: process.env.CONTRACT_TEST_PYTHON ?? "python",
  XSCI_HOME: path.join(profileRoot, ".xsci"),
};
const nextBin = path.join(workstationRoot, "node_modules", "next", "dist", "bin", "next");
const child = spawn(process.execPath, [nextBin, "start", workstationRoot, "--hostname", "127.0.0.1", "--port", String(port)], {
  cwd: workstationRoot,
  env,
  stdio: ["ignore", "pipe", "pipe"],
});

let output = "";
child.stdout.on("data", (chunk) => { output += chunk.toString(); });
child.stderr.on("data", (chunk) => { output += chunk.toString(); });

async function waitForResponse(url, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Next.js exited before readiness:\n${output}`);
    try {
      return await fetch(url);
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw new Error(`Next.js did not become ready:\n${output}`);
}

try {
  const response = await waitForResponse(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`);
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.ok, true, "the completed status lookup must be consumable by readJson");
  assert.equal(payload.action, "scientist_upgrade_campaign_status");
  assert.equal(payload.scientist_upgrade_campaign.ok, false);
  assert.equal(payload.scientist_upgrade_campaign.status, "blocked");
  assert.equal(payload.scientist_upgrade_campaign.parity_claim_allowed, false);
  assert.equal(payload.scientist_upgrade_campaign.score_cap, 84);
  assert.deepEqual(payload.scientist_upgrade_campaign.blockers, ["missing_git_repository"]);
  assert.equal(payload.no_training_started, true);
  assert.equal(payload.official_submit, "blocked_until_explicit_human_approval");
  console.log("upgrade campaign fail-closed status contract passed");
} finally {
  child.kill();
  await Promise.race([
    new Promise((resolve) => child.once("exit", resolve)),
    new Promise((resolve) => setTimeout(resolve, 5_000)),
  ]);
  if (child.exitCode === null) child.kill("SIGKILL");
  await rm(tempRoot, { recursive: true, force: true });
}
