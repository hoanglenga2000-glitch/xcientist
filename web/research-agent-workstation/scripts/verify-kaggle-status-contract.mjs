import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import ts from "typescript";

const sourcePath = path.resolve("src/lib/server/kaggle-status.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
  fileName: sourcePath,
}).outputText;
const module = { exports: {} };
new Function("exports", "module", compiled)(module.exports, module);
const { deriveKaggleAuthState } = module.exports;
const now = Date.parse("2026-07-12T02:00:00Z");
const freshGeneratedAt = "2026-07-12T01:59:00Z";
const staleGeneratedAt = "2026-07-12T00:00:00Z";

assert.deepEqual(deriveKaggleAuthState(null, false), {
  status: "not_configured",
  configured: false,
  authenticated: false,
  evidenceCredentialStatus: "not_configured",
});
assert.equal(deriveKaggleAuthState(null, true, now).status, "configured_unverified");
assert.equal(deriveKaggleAuthState({ credential_installed: true, authenticated: false }, false, now).status, "not_configured");
assert.equal(deriveKaggleAuthState({
  status: "passed",
  authenticated: true,
  credential_installed: true,
  credential_status: "configured_dpapi_unverified",
  verification_method: "dpapi_status_only",
  generated_at: freshGeneratedAt,
}, true, now).status, "configured_unverified");
assert.equal(deriveKaggleAuthState({
  status: "passed",
  authenticated: true,
  credential_installed: true,
  credential_status: "authenticated_real_api",
  verification_method: "dpapi_status_and_real_api_smoke",
  generated_at: freshGeneratedAt,
}, false, now).status, "not_configured");
assert.equal(deriveKaggleAuthState({
  status: "passed",
  authenticated: true,
  credential_installed: true,
  credential_status: "authenticated_real_api",
  verification_method: "dpapi_status_and_real_api_smoke",
  generated_at: freshGeneratedAt,
}, true, now).status, "authenticated");
assert.equal(deriveKaggleAuthState({
  status: "passed",
  authenticated: true,
  credential_installed: true,
  credential_status: "authenticated_real_api",
  verification_method: "dpapi_status_and_real_api_smoke",
  generated_at: staleGeneratedAt,
}, true, now).status, "configured_unverified");

const connectorSourcePath = path.resolve("src/lib/connector-status.ts");
const connectorSource = fs.readFileSync(connectorSourcePath, "utf8");
const connectorCompiled = ts.transpileModule(connectorSource, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
  fileName: connectorSourcePath,
}).outputText;
const connectorModule = { exports: {} };
new Function("exports", "module", connectorCompiled)(connectorModule.exports, connectorModule);
const { deriveConnectorDisplays } = connectorModule.exports;

const unconfigured = deriveConnectorDisplays({
  gpu: { state: "GPU Environment Created / Web Terminal Ready / External SSH Pending", configured: false, current_gate_ready: false },
  kaggle: { state: "configured_unverified", configured: true, authenticated: false, ready: false },
  deepseek: { state: "Not Configured", configured: false },
  code_agent: { state: "Not Configured", configured: false },
}, "zh-CN");
assert.equal(unconfigured.gpu.state, "SSH 待验证");
assert.equal(unconfigured.gpu.tone, "amber");
assert.equal(unconfigured.kaggle.state, "待认证");
assert.equal(unconfigured.kaggle.tone, "amber");
assert.equal(unconfigured.deepseek.state, "未配置");
assert.equal(unconfigured.deepseek.tone, "red");
assert.equal(unconfigured.humanGate.state, "未验证");
assert.equal(unconfigured.humanGate.tone, "slate");

const authenticated = deriveConnectorDisplays({
  gpu: { state: "ready", configured: true, current_gate_ready: true },
  kaggle: { state: "authenticated", configured: true, authenticated: true, ready: true },
  deepseek: { state: "verified", configured: true },
  code_agent: { state: "configured", configured: true },
}, "en-US");
assert.equal(authenticated.gpu.state, "Verified Ready");
assert.equal(authenticated.kaggle.state, "Authenticated");
assert.equal(authenticated.deepseek.state, "Runtime Unverified");
assert.equal(authenticated.deepseek.tone, "amber");
assert.equal(authenticated.codeAgent.state, "Runtime Unverified");
assert.equal(authenticated.codeAgent.tone, "amber");

const failClosed = deriveConnectorDisplays({
  gpu: { state: "failed", configured: true, current_gate_ready: true },
  kaggle: { state: "error", configured: true, authenticated: true, ready: true },
  deepseek: { state: "invalid credential", configured: true, runtime_verified: true },
  code_agent: { state: "blocked", configured: true, smoke_passed: true },
}, "en-US");
for (const value of [failClosed.gpu, failClosed.kaggle, failClosed.deepseek, failClosed.codeAgent]) {
  assert.equal(value.state, "Blocked");
  assert.equal(value.tone, "red");
}

const contradictory = deriveConnectorDisplays({
  gpu: { state: "ready", configured: false, current_gate_ready: true },
  kaggle: { state: "authenticated", configured: false, authenticated: true, ready: true },
  deepseek: { state: "verified", configured: false, runtime_verified: true },
  code_agent: { state: "verified", configured: false, authenticated: true },
}, "en-US");
for (const value of [contradictory.gpu, contradictory.kaggle, contradictory.deepseek, contradictory.codeAgent]) {
  assert.notEqual(value.tone, "green");
}

const controlled = deriveConnectorDisplays({
  kaggle: { state: "configured_unverified", configured: true, human_gate_required_for_submission: true },
}, "en-US");
assert.equal(controlled.humanGate.state, "Controlled");
assert.equal(controlled.humanGate.tone, "amber");

console.log("Kaggle status contract: PASS");
