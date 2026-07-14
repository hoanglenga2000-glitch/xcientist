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

const providerBoundary = loadTypeScript("src/lib/security/provider-boundary.ts");
const previousAllowedOrigins = process.env.DEEPSEEK_ALLOWED_ORIGINS;
process.env.DEEPSEEK_ALLOWED_ORIGINS = "https://deepseek-proxy.example,http://127.0.0.1:11434";
try {
  assert.equal(
    providerBoundary.resolveDeepSeekEndpoint("https://api.deepseek.com/v1").chatCompletionsUrl,
    "https://api.deepseek.com/v1/chat/completions"
  );
  assert.equal(
    providerBoundary.resolveDeepSeekEndpoint("https://deepseek-proxy.example").chatCompletionsUrl,
    "https://deepseek-proxy.example/chat/completions"
  );
  assert.equal(
    providerBoundary.resolveDeepSeekEndpoint("http://127.0.0.1:11434/v1").chatCompletionsUrl,
    "http://127.0.0.1:11434/v1/chat/completions"
  );
  for (const value of [
    "http://api.deepseek.com",
    "https://unapproved.example",
    "http://localhost:11434",
    "http://127.0.0.1:11435",
    "https://user:secret@api.deepseek.com",
    "https://api.deepseek.com?token=secret",
    "https://api.deepseek.com#secret"
  ]) {
    assert.throws(() => providerBoundary.resolveDeepSeekEndpoint(value), /provider_/);
  }
} finally {
  if (previousAllowedOrigins === undefined) delete process.env.DEEPSEEK_ALLOWED_ORIGINS;
  else process.env.DEEPSEEK_ALLOWED_ORIGINS = previousAllowedOrigins;
}
assert.deepEqual(
  await providerBoundary.readBoundedProviderJson(new Response('{"ok":true}', {
    headers: { "content-type": "application/json" }
  }), 64),
  { ok: true }
);
await assert.rejects(
  providerBoundary.readBoundedProviderJson(new Response("0123456789"), 4),
  /provider_response_too_large/
);
await assert.rejects(
  providerBoundary.readBoundedProviderJson(new Response("{}", {
    headers: { "content-length": "100" }
  }), 4),
  /provider_response_too_large/
);
await assert.rejects(
  providerBoundary.readBoundedProviderJson(new Response("not-json"), 64),
  /provider_response_invalid_json/
);
await assert.rejects(
  providerBoundary.readBoundedProviderJson(new Response(new Uint8Array([
    0x7b, 0x22, 0x76, 0x22, 0x3a, 0x22, 0xc3, 0x28, 0x22, 0x7d
  ])), 64),
  /provider_response_invalid_utf8/
);
await assert.rejects(
  providerBoundary.readBoundedProviderJson(new Response("[]"), 64),
  /provider_response_invalid_json/
);
const remoteRequestId = "remote-secret-request-id";
const providerHttpError = providerBoundary.providerHttpFailure("deepseek", new Response("{}", {
  status: 429,
  headers: { "x-request-id": remoteRequestId }
}));
assert.equal(providerHttpError, "deepseek_http_429_request_present");
assert.equal(providerHttpError.includes(remoteRequestId), false);
assert.equal(
  providerBoundary.providerHttpFailure("deepseek", new Response("{}", { status: 503 })),
  "deepseek_http_503_request_unavailable"
);

const xmlBoundary = loadTypeScript("src/lib/security/xml.ts");
assert.equal(xmlBoundary.decodeXmlText("&lt;safe&gt;&amp;"), "<safe>&");
assert.equal(xmlBoundary.decodeXmlText("&amp;lt;script&amp;gt;"), "&lt;script&gt;");
assert.equal(xmlBoundary.decodeXmlText("<![CDATA[&lt;literal&gt;]]>"), "&lt;literal&gt;");

process.env.WORKSTATION_ROOT = root;
const stableFile = loadTypeScript("src/lib/server/stable-file.ts");
const paths = loadTypeScript("src/lib/server/paths.ts", {
  "@/lib/security/request-boundary": boundary,
  "@/lib/server/stable-file": stableFile
});
assert.equal(paths.resolveWorkspacePath("workspace/tasks/task_01").startsWith(root), true);
for (const value of ["../outside", "..\\outside", "/absolute", "C:\\outside", "safe/../../outside", "safe:stream"]) {
  assert.throws(() => paths.resolveWorkspacePath(value), /Workspace path/);
}

const stableRoot = fs.mkdtempSync(path.join(root, ".stable-file-test-"));
try {
  const regularPath = path.join(stableRoot, "regular.txt");
  fs.writeFileSync(regularPath, "stable-content", "utf8");
  const stableRead = await stableFile.readStableRegularTextFile(regularPath, {
    allowedRoot: stableRoot,
    maxBytes: 1024
  });
  assert.equal(stableRead.text, "stable-content");
  assert.match(stableRead.sha256, /^[0-9a-f]{64}$/);
  const privatePath = path.join(stableRoot, "private", "atomic.txt");
  const privateWrite = await stableFile.writeAtomicPrivateTextFile(privatePath, "atomic-content", {
    allowedRoot: stableRoot,
    maxBytes: 1024
  });
  assert.equal(privateWrite.text, "atomic-content");
  assert.match(privateWrite.sha256, /^[0-9a-f]{64}$/);
  await assert.rejects(
    stableFile.readStableRegularTextFile(regularPath, { allowedRoot: stableRoot, maxBytes: 4 }),
    /byte limit/
  );
  const invalidUtf8Path = path.join(stableRoot, "invalid-utf8.txt");
  fs.writeFileSync(invalidUtf8Path, Buffer.from([0xc3, 0x28]));
  await assert.rejects(
    stableFile.readStableRegularTextFile(invalidUtf8Path, { allowedRoot: stableRoot, maxBytes: 16 }),
    /valid UTF-8/
  );
  const hardLinkSource = path.join(root, ".stable-file-hardlink-source.txt");
  const hardLinkPath = path.join(stableRoot, "hardlink.txt");
  fs.writeFileSync(hardLinkSource, "outside-hardlink", "utf8");
  fs.linkSync(hardLinkSource, hardLinkPath);
  await assert.rejects(
    stableFile.readStableRegularTextFile(hardLinkPath, { allowedRoot: stableRoot, maxBytes: 1024 }),
    /hard-linked/
  );
  fs.rmSync(hardLinkPath, { force: true });
  fs.rmSync(hardLinkSource, { force: true });
  if (process.platform === "win32") {
    const outsideDir = fs.mkdtempSync(path.join(root, ".stable-file-outside-"));
    const outsidePath = path.join(outsideDir, "outside.txt");
    const linkDir = path.join(stableRoot, "linked-outside");
    try {
      fs.writeFileSync(outsidePath, "outside", "utf8");
      fs.symlinkSync(outsideDir, linkDir, "junction");
      await assert.rejects(
        stableFile.readStableRegularTextFile(path.join(linkDir, "outside.txt"), {
          allowedRoot: stableRoot,
          maxBytes: 1024
        }),
        /junction|symlink|escapes/
      );
    } finally {
      fs.rmSync(linkDir, { recursive: true, force: true });
      fs.rmSync(outsideDir, { recursive: true, force: true });
    }
  } else {
    const outsidePath = path.join(root, ".stable-file-outside.txt");
    const linkPath = path.join(stableRoot, "link.txt");
    try {
      fs.writeFileSync(outsidePath, "outside", "utf8");
      fs.symlinkSync(outsidePath, linkPath, "file");
      await assert.rejects(
        stableFile.readStableRegularTextFile(linkPath, { allowedRoot: stableRoot, maxBytes: 1024 }),
        /non-symlink regular file/
      );
    } finally {
      fs.rmSync(linkPath, { force: true });
      fs.rmSync(outsidePath, { force: true });
    }
  }
  const outsideWriteDir = fs.mkdtempSync(path.join(root, ".stable-write-outside-"));
  const linkedWriteDir = path.join(stableRoot, "linked-write-outside");
  try {
    fs.symlinkSync(outsideWriteDir, linkedWriteDir, process.platform === "win32" ? "junction" : "dir");
    await assert.rejects(
      stableFile.writeAtomicPrivateTextFile(path.join(linkedWriteDir, "escaped.txt"), "blocked", {
        allowedRoot: stableRoot,
        maxBytes: 1024
      }),
      /junction|symlink|escapes/
    );
    assert.equal(fs.existsSync(path.join(outsideWriteDir, "escaped.txt")), false);
  } finally {
    fs.rmSync(linkedWriteDir, { recursive: true, force: true });
    fs.rmSync(outsideWriteDir, { recursive: true, force: true });
  }
} finally {
  fs.rmSync(stableRoot, { recursive: true, force: true });
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
assert.equal(actionsSource.includes("readStableRegularTextFile"), true);
assert.equal(actionsSource.includes("quality_artifact_sha256"), true);
assert.equal(actionsSource.includes("quality_gate_action_id"), true);
assert.equal(actionsSource.includes('error: "python_syntax_check_failed"'), true);
assert.equal(actionsSource.includes('error: "patch_python_syntax_check_failed"'), true);
assert.equal(actionsSource.includes("## Patch Excerpt"), false);
assert.equal(actionsSource.includes("Patch content is not duplicated into review summaries."), true);
assert.equal(actionsSource.includes(".syntax_${randomUUID()}"), true);
assert.equal(actionsSource.includes("code_quality_check_${reviewRevisionId}.json"), true);
assert.equal(actionsSource.includes("writeAtomicPrivateTextFile(scratchFile"), true);

const runContractSource = fs.readFileSync(path.join(root, "src/lib/server/workstation-run-contract.ts"), "utf8");
assert.equal(runContractSource.includes("readStableRegularTextFile"), true);
assert.equal(runContractSource.includes("quality_gate_sha256"), true);
const hpcGateFactory = runContractSource
  .split("export async function createHpcExecutionGate", 2)[1]
  .split("export async function generateTeacherEvidenceBundle", 1)[0];
assert.equal(hpcGateFactory.includes("reviewer: null"), true);
assert.equal(hpcGateFactory.includes("decidedAt: null"), true);

const reportRouteSource = fs.readFileSync(path.join(root, "src/app/api/tasks/[taskId]/report/route.ts"), "utf8");
assert.equal(reportRouteSource.includes("readStableRegularTextFile"), true);
assert.equal(reportRouteSource.includes("fs.stat(absolutePath)"), false);
for (const reportRoute of [
  "src/app/api/tasks/[taskId]/report/route.ts",
  "src/app/api/tasks/[taskId]/generate-report-draft/route.ts",
  "src/app/api/reports/[reportId]/insert-figure/route.ts"
]) {
  const source = fs.readFileSync(path.join(root, reportRoute), "utf8");
  assert.equal(source.includes("randomUUID()"), true);
  assert.equal(source.includes("ensurePrivateDirectory"), true);
  assert.equal(source.includes("writeAtomicPrivateTextFile"), true);
  assert.equal(source.includes("markdown_sha256"), true);
  assert.equal(source.includes("html_sha256"), true);
  assert.equal(source.includes("fs.writeFile"), false);
}
const insertFigureRouteSource = fs.readFileSync(path.join(root, "src/app/api/reports/[reportId]/insert-figure/route.ts"), "utf8");
assert.equal(insertFigureRouteSource.includes("prisma.report.updateMany"), true);
assert.equal(insertFigureRouteSource.includes("updatedAt: report.updatedAt"), true);
assert.equal(insertFigureRouteSource.includes("report_revision_conflict"), true);

const literatureSource = fs.readFileSync(path.join(root, "src/app/api/literature/search/route.ts"), "utf8");
assert.equal(literatureSource.includes('"configs"'), false);
assert.equal(literatureSource.includes('"workspace/workstation_runs"'), false);
assert.equal(literatureSource.includes('path.join("workspace", "tasks", taskId, "reports")'), true);
assert.equal(literatureSource.includes("readStableRegularTextFile"), true);
assert.equal(literatureSource.includes("MAX_LOCAL_WALK_ENTRIES"), true);
assert.equal(literatureSource.includes("MAX_LOCAL_WALK_DEPTH"), true);
assert.equal(literatureSource.includes("MAX_ARXIV_RESPONSE_BYTES"), true);
assert.equal(literatureSource.includes('redirect: "error"'), true);
assert.equal(literatureSource.includes("response.text()"), false);
assert.equal(literatureSource.includes("containsSecretMaterial(query)"), true);

const hpcGateSource = fs.readFileSync(path.join(root, "src/lib/server/hpc-execution-gate.ts"), "utf8");
const gpuGatewaySource = fs.readFileSync(path.join(root, "src/lib/server/gpu-ssh-gateway.ts"), "utf8");
assert.equal(hpcGateSource.includes("passedCodeQualityGateByBinding"), true);
assert.equal(hpcGateSource.includes("latestPassedCodeQualityGate"), true);
assert.equal(gpuGatewaySource.includes("validateHpcExecutionGate"), true);

const sessionSource = fs.readFileSync(path.join(root, "src/lib/server/claude-agent-sessions.ts"), "utf8");
assert.equal(sessionSource.includes("randomUUID()"), true);
assert.equal(sessionSource.includes("Math.random()"), false);
assert.equal(sessionSource.includes("normalizeClaudeSessionId"), true);
assert.equal(sessionSource.includes("fetch(config.chatCompletionsUrl"), true);
assert.equal(sessionSource.includes("config.baseUrl.replace"), false);
assert.equal(sessionSource.includes('redirect: "error"'), true);
assert.equal(sessionSource.includes("readBoundedProviderJson(response)"), true);
assert.equal(sessionSource.includes('providerHttpFailure("deepseek_code_agent", response)'), true);
assert.equal(sessionSource.includes('safeProviderFailure(error, "deepseek_code_agent_session_failed")'), true);
assert.equal(sessionSource.includes("response.json()"), false);
assert.equal(sessionSource.includes("JSON.stringify(payload.error)"), false);
assert.equal(sessionSource.includes('error: lastError.message'), false);

console.log("security boundary checks passed");
