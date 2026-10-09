import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import path from "node:path";
import test from "node:test";

const server = await fs.readFile(path.join(process.cwd(), "src/lib/server/assistant-run-secrets.ts"), "utf8");
const panel = await fs.readFile(path.join(process.cwd(), "src/components/workstation/screens/AssistantControlledCredentials.tsx"), "utf8");
const runs = await fs.readFile(path.join(process.cwd(), "src/app/api/assistant/runs/route.ts"), "utf8");
const actions = await fs.readFile(path.join(process.cwd(), "src/app/api/assistant/runs/[runId]/actions/route.ts"), "utf8");
const intent = await fs.readFile(path.join(process.cwd(), "src/app/api/assistant/run-secrets/intent/route.ts"), "utf8");
const envelope = await fs.readFile(path.join(process.cwd(), "src/app/api/assistant/run-secrets/envelope/route.ts"), "utf8");
const assistant = await fs.readFile(path.join(process.cwd(), "src/components/workstation/screens/AssistantScreen.tsx"), "utf8");

test("run secret channel uses server intent hybrid encryption and opaque references", () => {
  assert.match(server, /const INTENT_TTL_MS = 2 \* 60 \* 60 \* 1000/);
  assert.match(panel, /RSA-OAEP-256\+A256GCM/);
  assert.match(panel, /AES-GCM/);
  assert.match(panel, /additionalData: aad/);
  assert.match(panel, /evomind\.hybrid_run_secret_envelope\.v1/);
  assert.match(server, /randomBytes\(32\)\.toString\("base64url"\)/);
  assert.match(server, /secret_ref_sha256/);
  assert.match(server, /writeExclusiveBytes/);
  assert.match(server, /writeExclusiveBytes\(pendingEnvelopePath\(secretRef\), envelopeBytes\)/);
  assert.match(server, /run_secret_intent_consumed/);
  assert.match(server, /run_secret_ref_mismatch_or_expired/);
  assert.match(intent, /issueRunSecretIntent/);
  assert.match(envelope, /storeRunSecretEnvelope/);
  assert.match(panel, /kaggle_api/);
  assert.match(panel, /Three independent one-time Kaggle references/);
  assert.match(panel, /const count = purpose === "kaggle_api" \? 3 : 1/);
});

test("plaintext credentials never enter the assistant Run request or tool arguments", () => {
  assert.match(server, /plaintext_secret_rejected/);
  assert.match(panel, /secret_ref/);
  assert.match(runs, /validateRunSecretRefs/);
  assert.match(actions, /validateRunSecretRefs/);
  assert.match(runs, /secret_refs: secretRefs/);
  assert.match(actions, /secret_refs: secretRefs/);
  assert.doesNotMatch(panel, /localStorage\.setItem\([^)]*(password|username|host)/i);
  assert.match(panel, /setPassword\(""\)/);
  assert.match(panel, /commitSealed/);
  assert.match(panel, /sealedBatchRef/);
  assert.doesNotMatch(panel.match(/sealPending: async[\s\S]*?commitSealed:/)?.[0] ?? "", /setHost\(""\)|setUsername\(""\)|setRemotePath\("\/"\)/);
});

test("resume actions can attach only newly sealed opaque references", () => {
  assert.match(assistant, /sealPending\(\)/);
  assert.match(assistant, /body: JSON\.stringify\(\{ action, secret_refs: secretRefs \}\)/);
  assert.doesNotMatch(assistant, /body: JSON\.stringify\(\{ action \}\)/);
});

test("controlled credentials can attach to one explicit approval Run without creating a Run", () => {
  assert.match(actions, /managed_hpc_identity: managedHpcIdentity/);
  assert.match(assistant, /const CREDENTIAL_RUN_ID = \/\^run_\[0-9a-f\]\{32\}\$\//);
  assert.match(assistant, /assistant_attach_credentials_to_run/);
  assert.match(assistant, /action: "attach_secret_refs", secret_refs: secretRefs/);
  assert.match(assistant, /secretRefs\.length !== 1/);
  assert.match(assistant, /isConfirmedControlledSecretBinding\(run, credentialRunId, "weather4cast_sftp", 1\)/);
  assert.match(assistant, /isExactPendingWeatherAcceleration\(currentRun\)/);
  assert.match(assistant, /commitSealed\(secretRefs\)/);
  const attachBlock = assistant.match(/const attachCredentialsToPendingRun[\s\S]*?\}, \[apiError/)?.[0] ?? "";
  assert.doesNotMatch(attachBlock, /fetch\("\/api\/assistant\/runs"/);
  assert.doesNotMatch(attachBlock, /prompt|conversation_id|instruction/);
  assert.match(attachBlock, /applySnapshot\(run/);
});

test("existing Run identity rebinds only from the server-bound action route", () => {
  assert.match(actions, /const serverBody = \{ \.\.\.body \};/);
  assert.match(actions, /delete serverBody\.managed_hpc_identity/);
  assert.match(actions, /managed_hpc_identity: managedHpcIdentity/);
  assert.match(actions, /idempotency_key/);
  assert.doesNotMatch(actions, /body\.managed_hpc_identity/);
});
