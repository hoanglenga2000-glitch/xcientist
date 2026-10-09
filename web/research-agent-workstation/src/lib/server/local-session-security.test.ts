import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = process.cwd();

test("all authenticated mutations are length-bounded and assistant chunks are the only binary exception", async () => {
  const source = await readFile(path.join(root, "src", "proxy.ts"), "utf-8");
  assert.match(source, /if \(rawLength === null\)[\s\S]*content_length_required/);
  assert.match(source, /Number\(rawLength\) > MAX_BODY_BYTES/);
  assert.match(source, /contentType !== "application\/json"/);
  assert.match(source, /allowsMultipart && contentType === "multipart\/form-data"/);
  assert.match(source, /allowsBinaryChunk && contentType === "application\/octet-stream"/);
  assert.match(source, /\/api\\\/assistant\\\/uploads/);
  assert.doesNotMatch(source, /rawLength === null && transferEncoding/);
  assert.match(source, /forwardedHeaders\.delete\(LOCAL_AUTOMATION_VERIFIED_HEADER\)/);
  assert.match(source, /if \(localAutomation\) forwardedHeaders\.set\(LOCAL_AUTOMATION_VERIFIED_HEADER, "1"\)/);
});

test("Next Proxy preserves a full upload chunk while the application boundary remains stricter", async () => {
  const config = await readFile(path.join(root, "next.config.mjs"), "utf-8");
  const proxy = await readFile(path.join(root, "src", "proxy.ts"), "utf-8");
  assert.match(config, /proxyClientMaxBodySize: 17 \* 1024 \* 1024/);
  assert.match(proxy, /MAX_BODY_BYTES = 16 \* 1024 \* 1024/);
});

test("the browser session wrapper supplies an empty JSON envelope and fixed fetch controls", async () => {
  const source = await readFile(
    path.join(root, "src", "components", "workstation", "LocalSessionBootstrap.tsx"),
    "utf-8",
  );
  assert.match(source, /inheritedBody === null \? "\{\}"/);
  assert.match(source, /headers\.set\("Content-Type", "application\/json"\)/);
  assert.match(source, /cache: "no-store", redirect: "error"/);
  assert.match(source, /sessionStorage\.getItem\(CSRF_STORAGE_KEY\) \|\| csrfToken/);
  assert.match(source, /headers\.set\("x-evomind-csrf", latestCsrf\)/);
});

test("assistant explains stale local sessions instead of reporting a generic stream failure", async () => {
  const source = await readFile(
    path.join(root, "src", "components", "workstation", "screens", "AssistantScreen.tsx"),
    "utf-8",
  );
  assert.match(source, /response\.status === 401 \|\| code === "session_required"/);
  assert.match(source, /response\.status === 403 \|\| code === "csrf_rejected"/);
  assert.match(source, /本地会话已失效/);
  assert.match(source, /sessionStorage\.removeItem\(LOCAL_CSRF_STORAGE_KEY\)/);
  assert.match(source, /const friendly = await assistantStreamErrorMessage\(response, locale\)/);
});

test("runtime API requests cannot spawn an untracked replacement process", async () => {
  const source = await readFile(path.join(root, "src", "lib", "server", "evomind-runtime.ts"), "utf-8");
  const paths = await readFile(path.join(root, "src", "lib", "server", "paths.ts"), "utf-8");
  const route = await readFile(path.join(root, "src", "app", "api", "runtime", "[...path]", "route.ts"), "utf-8");
  assert.doesNotMatch(source, /child_process|\bspawn\s*\(/);
  assert.match(paths, /WORKSTATION_DATA_DIR \?\? workspaceRoot/);
  assert.match(paths, /runtimeRoot = path\.join\(workstationDataRoot, "workspace", "runtime"\)/);
  assert.match(source, /import \{ runtimeRoot \} from "@\/lib\/server\/paths"/);
  assert.doesNotMatch(source, /path\.join\(workspaceRoot, "workspace", "runtime"\)/);
  assert.match(source, /Runtime process creation and recovery belong exclusively to the signed/);
  assert.match(route, /code: "runtime_not_ready"[\s\S]*status: 503/);
});

test("assistant compatibility stream delegates to the durable runtime without pinning a model or spawning a subprocess", async () => {
  const route = await readFile(
    path.join(root, "src", "app", "api", "assistant", "stream", "route.ts"),
    "utf-8",
  );
  assert.match(route, /runtimeJson<RunSnapshot>\("\/v1\/runs"/);
  assert.match(route, /\/v1\/runs\/\$\{encodeURIComponent\(run\.id\)\}\/events/);
  assert.match(route, /model_observed/);
  assert.match(route, /artifact_published/);
  assert.match(route, /approval_required/);
  assert.doesNotMatch(route, /child_process|spawn\s*\(|assistantProcessEnv/);
  assert.doesNotMatch(route, /OPENAI_MODEL|gpt-5\.6-sol|DEEPSEEK_MODEL/);
});
