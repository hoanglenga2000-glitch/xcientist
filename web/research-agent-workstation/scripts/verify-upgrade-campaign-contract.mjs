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
const modePath = path.join(fixtureRoot, "fixture-mode.txt");
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
    "import os",
    "import sys",
    "from pathlib import Path",
    "mode = Path(os.environ['WORKSTATION_ROOT'], 'fixture-mode.txt').read_text(encoding='utf-8').strip()",
    "blocked = {",
    "    'tool': 'research_parity_gate',",
    "    'status': 'blocked',",
    "    'parity_claim_allowed': False,",
    "    'score_cap': 84,",
    "    'certification': {",
    "        'tool': 'capability_certification_status',",
    "        'status': 'not_certified',",
    "        'verified': False,",
    "        'release_allowed': False,",
    "        'parity_claim_allowed': False,",
    "        'artifact_path': os.environ['WORKSTATION_ROOT'],",
    "    },",
    "    'upgrade_campaign': {",
    "        'tool': 'scientist_upgrade_campaign_status',",
    "        'status': 'not_run',",
    "        'active_and_verified': False,",
    "        'artifact_path': os.environ['WORKSTATION_ROOT'],",
    "    },",
    "    'certification_campaign_source_binding_verified': False,",
    "    'certification_campaign_artifact_binding_verified': False,",
    "    'blockers': [",
    "        'external_capability_certification_not_verified',",
    "        'active_self_upgrade_campaign_not_verified',",
    "    ],",
    "    'claim': 'research parity is not externally certified',",
    "}",
    "if mode == 'valid_blocked_success':",
    "    print(json.dumps(blocked))",
    "    raise SystemExit(0)",
    "if mode == 'valid_blocked_error':",
    "    print(json.dumps(blocked))",
    "    raise SystemExit(1)",
    "if mode == 'contradictory':",
    "    blocked['parity_claim_allowed'] = True",
    "    blocked['score_cap'] = 100",
    "    print(json.dumps(blocked))",
    "    raise SystemExit(1)",
    "if mode == 'command_blocked':",
    "    print(json.dumps({",
    "        'ok': False,",
    "        'tool': 'scientist_upgrade_campaign',",
    "        'action': 'run',",
    "        'status': 'blocked',",
    "        'error': 'internal path: ' + os.environ['WORKSTATION_ROOT'],",
    "    }))",
    "    raise SystemExit(1)",
    "if mode == 'command_wrong_action':",
    "    print(json.dumps({",
    "        'ok': False,",
    "        'tool': 'scientist_upgrade_campaign',",
    "        'action': 'promote',",
    "        'status': 'blocked',",
    "        'error': 'internal path: ' + os.environ['WORKSTATION_ROOT'],",
    "    }))",
    "    raise SystemExit(1)",
    "if mode in {'valid_run_success', 'forged_run_success'}:",
    "    print(json.dumps({",
    "        'schema': 'evomind.self_upgrade_controller.v1',",
    "        'campaign_id': 'upgrade-fixture',",
    "        'status': 'awaiting_human_promotion' if mode == 'valid_run_success' else 'active',",
    "        'human_gate': 'explicit_human_approval_required',",
    "        'main_worktree_modified': False,",
    "        'no_training_started': True,",
    "        'manifest_path': os.environ['WORKSTATION_ROOT'],",
    "    }))",
    "    raise SystemExit(0)",
    "print('malformed internal path: ' + os.environ['WORKSTATION_ROOT'])",
    "raise SystemExit(1)",
    "",
  ].join("\n"),
  "utf8",
);
await writeFile(modePath, "valid_blocked_success", "utf8");

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
  for (const mode of ["valid_blocked_success", "valid_blocked_error"]) {
    await writeFile(modePath, mode, "utf8");
    const response = await waitForResponse(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`);
    assert.equal(response.status, 200);
    const payload = await response.json();
    assert.equal(payload.ok, true, "the completed status lookup must be consumable by readJson");
    assert.equal(payload.action, "scientist_upgrade_campaign_status");
    assert.equal(payload.scientist_upgrade_campaign.ok, false);
    assert.equal(payload.scientist_upgrade_campaign.status, "blocked");
    assert.equal(payload.scientist_upgrade_campaign.parity_claim_allowed, false);
    assert.equal(payload.scientist_upgrade_campaign.score_cap, 84);
    assert.deepEqual(payload.scientist_upgrade_campaign.blockers, [
      "external_capability_certification_not_verified",
      "active_self_upgrade_campaign_not_verified",
    ]);
    assert.equal(payload.no_training_started, true);
    assert.equal(payload.official_submit, "blocked_until_explicit_human_approval");
    assert.equal(JSON.stringify(payload).includes(tempRoot), false);
  }

  await writeFile(modePath, "valid_blocked_error", "utf8");
  const postStatusResponse = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: `http://127.0.0.1:${port}` },
    body: JSON.stringify({ action: "status" }),
  });
  assert.equal(postStatusResponse.status, 200);
  const postStatusPayload = await postStatusResponse.json();
  assert.equal(postStatusPayload.ok, true);
  assert.equal(postStatusPayload.scientist_upgrade_campaign.status, "blocked");

  for (const mode of ["contradictory", "malformed"]) {
    await writeFile(modePath, mode, "utf8");
    const response = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`);
    assert.equal(response.status, 500);
    const payload = await response.json();
    assert.equal(payload.ok, false);
    assert.equal(payload.error, "Upgrade campaign status failed");
    assert.equal(JSON.stringify(payload).includes(tempRoot), false);
  }

  await writeFile(modePath, "command_blocked", "utf8");
  const postResponse = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: `http://127.0.0.1:${port}` },
    body: JSON.stringify({ action: "run" }),
  });
  assert.equal(postResponse.status, 409);
  const postPayload = await postResponse.json();
  assert.equal(postPayload.ok, false);
  assert.equal(postPayload.error, "Upgrade campaign command was blocked");
  assert.equal(postPayload.scientist_upgrade_campaign.status, "blocked");
  assert.equal("error" in postPayload.scientist_upgrade_campaign, false);
  assert.equal(JSON.stringify(postPayload).includes(tempRoot), false);

  await writeFile(modePath, "command_wrong_action", "utf8");
  const wrongActionResponse = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: `http://127.0.0.1:${port}` },
    body: JSON.stringify({ action: "run" }),
  });
  assert.equal(wrongActionResponse.status, 409);
  const wrongActionPayload = await wrongActionResponse.json();
  assert.equal(wrongActionPayload.error, "Upgrade campaign command failed");
  assert.equal("scientist_upgrade_campaign" in wrongActionPayload, false);
  assert.equal(JSON.stringify(wrongActionPayload).includes(tempRoot), false);

  await writeFile(modePath, "valid_run_success", "utf8");
  const validRunResponse = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: `http://127.0.0.1:${port}` },
    body: JSON.stringify({ action: "run" }),
  });
  assert.equal(validRunResponse.status, 200);
  const validRunPayload = await validRunResponse.json();
  assert.equal(validRunPayload.ok, true);
  assert.equal(validRunPayload.scientist_upgrade_campaign.action, "run");
  assert.equal(validRunPayload.scientist_upgrade_campaign.status, "awaiting_human_promotion");
  assert.equal(JSON.stringify(validRunPayload).includes(tempRoot), false);

  await writeFile(modePath, "forged_run_success", "utf8");
  const forgedRunResponse = await fetch(`http://127.0.0.1:${port}/api/scientist/upgrade-campaign`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: `http://127.0.0.1:${port}` },
    body: JSON.stringify({ action: "run" }),
  });
  assert.equal(forgedRunResponse.status, 409);
  const forgedRunPayload = await forgedRunResponse.json();
  assert.equal(forgedRunPayload.error, "Upgrade campaign command failed");
  assert.equal(JSON.stringify(forgedRunPayload).includes(tempRoot), false);
  console.log("upgrade campaign strict fail-closed transport contract passed");
} finally {
  child.kill();
  await Promise.race([
    new Promise((resolve) => child.once("exit", resolve)),
    new Promise((resolve) => setTimeout(resolve, 5_000)),
  ]);
  if (child.exitCode === null) child.kill("SIGKILL");
  await rm(tempRoot, { recursive: true, force: true });
}
