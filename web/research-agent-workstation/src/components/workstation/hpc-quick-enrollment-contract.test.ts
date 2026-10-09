import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const SRC = path.resolve(process.cwd(), "src");

async function source(relativePath: string) {
  return readFile(path.join(SRC, relativePath), "utf8");
}

test("Data and Kaggle exposes the persistent AIMSLAB quick enrollment entry", async () => {
  const screen = await source("components/workstation/screens/DataKaggleScreen.tsx");
  assert.match(screen, /import \{ TenantHpcQuickEnrollment \}/);
  assert.match(screen, /<TenantHpcQuickEnrollment\s*\/>/);
});

test("Assistant exposes a direct route to the secure HPC quick enrollment surface", async () => {
  const assistantCredentials = await source("components/workstation/screens/AssistantControlledCredentials.tsx");
  assert.match(assistantCredentials, /data-ui-action="assistant_open_hpc_quick_enrollment"/);
  assert.match(assistantCredentials, /searchParams\.set\("page", "data"\)/);
  assert.match(assistantCredentials, /高级：AIMSLAB 快速换绑/);
});

test("quick enrollment form bypasses global capture auditing so controlled inputs receive change events", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  const shell = await source("components/workstation/AppShell.tsx");
  assert.match(component, /data-testid="tenant-hpc-quick-enrollment" data-ui-skip-action="true"/);
  assert.match(component, /aria-label="确认退役旧 allocation 并创建新 Generation"/);
  assert.match(shell, /const isImplicitFormControl =/);
  assert.match(shell, /if \(isImplicitFormControl && !hasExplicitAuditMarker\) return;/);
});

test("quick enrollment keeps a 60 second activation target and resumes provisioning after reload", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  assert.match(component, /const ACTIVATION_POLL_MS = 2_000;/);
  assert.match(component, /const QUICK_TARGET_MS = 60_000;/);
  assert.match(component, /binding\.state !== "provisioning"/);
  assert.match(component, /void watchActivation\(binding\.job_id, binding\.allocation_generation, Date\.now\(\)\)/);
  assert.match(component, /60 秒窗口内仍处于 provisioning/);
});

test("quick enrollment persists only non-secret job state and session identifiers", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  const persistentWriter = component.match(/function writePersistentState[\s\S]*?\n\}/)?.[0] ?? "";
  const sessionWriter = component.match(/function writeSessionIdentifiers[\s\S]*?\n\}/)?.[0] ?? "";
  assert.match(persistentWriter, /localStorage\.setItem/);
  assert.doesNotMatch(persistentWriter, /password|secret/i);
  assert.match(sessionWriter, /JSON\.stringify\(\{ role, proxyUser \}\)/);
  assert.doesNotMatch(sessionWriter, /password|secret/i);
  assert.doesNotMatch(component, /localStorage\.setItem\([^\n]+(?:proxyPassword|hpcPassword)/);
  assert.doesNotMatch(component, /sessionStorage\.setItem\([^\n]+(?:proxyPassword|hpcPassword)/);
});

test("quick enrollment encrypts secrets in browser memory and clears password state", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  assert.match(component, /importKey\("jwk"[\s\S]*name: "RSA-OAEP", hash: "SHA-256"/);
  assert.match(component, /generateKey\(\{ name: "AES-GCM", length: 256 \}/);
  assert.match(component, /algorithm: "RSA-OAEP-256\+A256GCM"/);
  assert.ok((component.match(/setProxyPassword\(""\)/g) ?? []).length >= 2);
  assert.ok((component.match(/setHpcPassword\(""\)/g) ?? []).length >= 2);
});

test("quick replacement retires the old profile before requesting a server-issued identity", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  const quickReplace = component.match(/const quickReplace = async \(\) => \{[\s\S]*?\n  \};/)?.[0] ?? "";
  const retireAt = quickReplace.indexOf("if (needsRetirement) await retireCurrent()");
  const enrollAt = quickReplace.indexOf("await submitEncryptedEnvelope()");
  assert.ok(retireAt >= 0 && enrollAt > retireAt);
  assert.match(component, /action: "retire"/);
  assert.match(component, /administrator_reclaim_confirmed: true/);

  const identityRoute = await source("app/api/hpc/byoa/enrollment-identity/route.ts");
  assert.match(identityRoute, /"allocation_generation" in body/);
  assert.match(identityRoute, /"profile_instance_id" in body/);
  assert.match(identityRoute, /"nonce" in body/);
  assert.match(identityRoute, /server_issued_identity_required/);
  assert.match(identityRoute, /server_issued: true/);
});

test("quick enrollment trims the pasted allocation id and says which requirement is missing", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  // A pasted binding id can carry invisible whitespace; validating the raw
  // string used to leave the submit button disabled with no explanation.
  assert.match(component, /test\(allocation\.trim\(\)\)/);
  assert.match(component, /allocation_binding_id: allocation\.trim\(\)/);
  assert.match(component, /data-testid="tenant-hpc-quick-missing"/);
  assert.match(component, /const missingRequirements = useMemo/);
  assert.match(component, /const canSubmit = missingRequirements\.length === 0;/);
});

test("quick replacement can recover a retryable provisioning failure in the same submit", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  assert.match(component, /const replaceFailedProvisioningBinding = async \(\) =>/);
  assert.match(component, /if \(binding\.state === "provisioning"\) \{[\s\S]*replaceFailedProvisioningBinding\(\)/);
  assert.doesNotMatch(component, /existing_provisioning_must_finish_or_be_marked_failed_before_replacement/);
  const quickReplace = component.match(/const quickReplace = async \(\) => \{[\s\S]*?\n  \};/)?.[0] ?? "";
  assert.match(quickReplace, /if \(needsRetirement\) await retireCurrent\(\)/);
  assert.match(quickReplace, /await submitEncryptedEnvelope\(\)/);
});

test("failed provisioning replacement remains fail closed and lifecycle managed", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  const route = await source("app/api/hpc/byoa/binding/route.ts");
  assert.match(component, /action: "replace_failed_provisioning"/);
  assert.match(route, /assertReplaceableFailedProvisioning/);
  assert.match(route, /failed_provisioning_receipt_exists/);
  assert.match(route, /failed_provisioning_identity_evidence_exists/);
  assert.match(route, /credential_decrypted:false,network_accessed:false/);
});

test("quick enrollment exposes only sanitized retry state and hides destructive recovery until retryable", async () => {
  const component = await source("components/workstation/TenantHpcQuickEnrollment.tsx");
  const route = await source("app/api/hpc/byoa/binding/route.ts");
  assert.match(route, /PUBLIC_ACTIVATION_STATES/);
  assert.match(route, /safeActivationError/);
  assert.match(route, /receipt_exists/);
  assert.match(route, /evidence_exists/);
  assert.doesNotMatch(route, /proxy_password|hpc_password|encrypted_envelope/);
  assert.match(component, /tenant-hpc-activation-retry-status/);
  assert.match(component, /retryableActivation \?/);
  assert.match(component, /指定代理暂时无法连接 HPC SSH 网关/);
  assert.doesNotMatch(component, /activation\.error_code\s*}\s*<\/p>/);
});
