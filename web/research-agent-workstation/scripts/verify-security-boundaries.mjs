import assert from "node:assert/strict";
import fs from "node:fs";
import module from "node:module";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const nativeRequire = module.createRequire(import.meta.url);

function loadTypeScript(relativePath, injectedModules = {}) {
  const source = fs.readFileSync(path.join(root, relativePath), "utf8");
  const output = ts.transpileModule(source, {
    compilerOptions: {
      esModuleInterop: true,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022
    }
  }).outputText;
  const exports = {};
  const loadedModule = { exports };
  const localRequire = (specifier) => injectedModules[specifier] ?? nativeRequire(specifier);
  new Function("require", "module", "exports", output)(localRequire, loadedModule, exports);
  return loadedModule.exports;
}

const boundary = loadTypeScript("src/lib/security/request-boundary.ts");
assert.equal(boundary.normalizeTaskId("house-prices"), "house_prices");
assert.equal(boundary.normalizeTaskId("task_01.v2"), "task_01.v2");
for (const value of [".", "..", "../x", "..\\x", "%2e%2e", "x%2fy", "x%5cy", "C:escape", "NUL.txt", "foo.", "Foo"]) {
  assert.throws(() => boundary.normalizeTaskId(value), /Invalid task ID/);
}
assert.equal(boundary.isLoopbackHostHeader("127.0.0.1:8088"), true);
assert.equal(boundary.isLoopbackHostHeader("[::1]:8088"), true);
assert.equal(boundary.isLoopbackHostHeader("attacker.invalid:8088"), false);
assert.equal(boundary.isLoopbackHostHeader("attacker@127.0.0.1:8088"), false);
assert.equal(boundary.isLoopbackHostHeader("127.0.0.1:0"), false);
assert.equal(boundary.isAllowedBrowserOrigin("http://127.0.0.1:8088", "127.0.0.1:8088"), true);
assert.equal(boundary.isAllowedBrowserOrigin("https://attacker.invalid", "127.0.0.1:8088"), false);
assert.equal(boundary.isAllowedBrowserOrigin(null, "127.0.0.1:8088"), false);
assert.equal(boundary.isAllowedMutationSource(null, "127.0.0.1:8088", "same-origin"), true);
assert.equal(boundary.isAllowedMutationSource(null, "127.0.0.1:8088", "cross-site"), false);

const networkBoundary = loadTypeScript("src/lib/security/network-boundary.ts");
assert.equal(networkBoundary.requireNetworkHost("127.0.0.1"), "127.0.0.1");
assert.equal(networkBoundary.requireNetworkHost("proxy.internal"), "proxy.internal");
assert.equal(networkBoundary.requireTcpPort("7890"), "7890");
for (const value of ["127.0.0.1&whoami", "proxy host", "-bad.internal", "bad..internal"]) {
  assert.throws(() => networkBoundary.requireNetworkHost(value), /Invalid/);
}
for (const value of ["0", "65536", "7890&whoami", "abc"]) {
  assert.throws(() => networkBoundary.requireTcpPort(value), /Invalid/);
}
assert.throws(() => networkBoundary.requireOptionalProxyUsername("user&whoami"), /Invalid/);
assert.throws(() => networkBoundary.requireShellSafePath("C:\\Python&whoami.exe"), /Invalid/);

process.env.WORKSTATION_ROOT = root;
const paths = loadTypeScript("src/lib/server/paths.ts", {
  "@/lib/security/request-boundary": boundary
});
assert.equal(paths.resolveWorkspacePath("workspace/tasks/task_01").startsWith(root), true);
for (const value of ["../outside", "..\\outside", "/absolute", "C:\\outside", "safe/../../outside", "safe:stream"]) {
  assert.throws(() => paths.resolveWorkspacePath(value), /Workspace path/);
}

const sshGateway = fs.readFileSync(path.join(root, "src/lib/server/gpu-ssh-gateway.ts"), "utf8");
assert.equal(sshGateway.includes("StrictHostKeyChecking=" + "accept-new"), false);
assert.equal(sshGateway.includes("trust_on_" + "first_use"), false);
assert.equal(sshGateway.includes("cmd /" + "c"), false);
assert.equal(sshGateway.includes("requireNetworkHost"), true);
assert.equal(sshGateway.includes("/hpc2" + "hdd/home/"), false);
assert.equal(sshGateway.includes("aims" + "lab"), false);
assert.equal(sshGateway.includes("jing" + "hw"), false);
assert.equal(sshGateway.includes("dedicated project directory"), true);

const runsSource = fs.readFileSync(path.join(root, "src/lib/server/runs.ts"), "utf8");
assert.equal(runsSource.includes('WORKSTATION_DISABLE_LOCAL_TRAINING === "0"'), false);
assert.equal(runsSource.includes('compute.execution_mode === "local"'), false);
assert.match(runsSource, /async function localTrainingFallbackDisabled\(\)\s*{\s*return true;\s*}/);

const actionsSource = fs.readFileSync(path.join(root, "src/lib/server/workstation-actions.ts"), "utf8");
assert.equal(actionsSource.includes('local_training_enabled: executionMode === "local"'), false);
assert.equal(actionsSource.includes("blocked_local_training_disabled"), true);

console.log("security boundary checks passed");
