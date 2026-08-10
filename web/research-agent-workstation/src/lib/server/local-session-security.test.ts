import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = process.cwd();

test("all authenticated mutations are length-bounded and JSON-only", async () => {
  const source = await readFile(path.join(root, "src", "proxy.ts"), "utf-8");
  assert.match(source, /if \(rawLength === null\)[\s\S]*content_length_required/);
  assert.match(source, /Number\(rawLength\) > MAX_BODY_BYTES/);
  assert.match(source, /contentType !== "application\/json"/);
  assert.match(source, /allowsMultipart && contentType === "multipart\/form-data"/);
  assert.doesNotMatch(source, /rawLength === null && transferEncoding/);
  assert.match(source, /forwardedHeaders\.delete\(LOCAL_AUTOMATION_VERIFIED_HEADER\)/);
  assert.match(source, /if \(localAutomation\) forwardedHeaders\.set\(LOCAL_AUTOMATION_VERIFIED_HEADER, "1"\)/);
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
  assert.match(source, /throw new Error\(await assistantStreamErrorMessage\(response, locale\)\)/);
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

test("assistant subprocess receives only the authenticated request session for loopback tools", async () => {
  const route = await readFile(
    path.join(root, "src", "app", "api", "assistant", "stream", "route.ts"),
    "utf-8",
  );
  assert.match(route, /SESSION_COOKIE/);
  assert.match(route, /CSRF_HEADER/);
  assert.match(route, /EVOMIND_INTERNAL_SESSION_COOKIE/);
  assert.match(route, /EVOMIND_INTERNAL_CSRF/);
  assert.match(route, /EVOMIND_INTERNAL_ORIGIN/);
  assert.match(route, /EVOLUTION_PRIMARY_PROVIDER:\s*"openai"/);
  assert.match(route, /EVOLUTION_PROVIDER_STRICT:\s*"true"/);
  assert.match(route, /OPENAI_BASE_URL:\s*"http:\/\/127\.0\.0\.1:65068\/v1"/);
  assert.match(route, /OPENAI_MODEL:\s*"gpt-5\.6-sol"/);
  assert.match(route, /OPENAI_REASONING_EFFORT:\s*"low"/);
  assert.match(route, /OPENAI_SERVICE_TIER:\s*"priority"/);
  assert.match(route, /LEGACY_GPU_ENV_KEYS/);
  assert.match(route, /for \(const key of LEGACY_GPU_ENV_KEYS\) delete env\[key\]/);
  assert.match(route, /assistantProcessEnv\(\)/);
  assert.match(route, /assistant_progress_timeout/);
  assert.match(route, /Date\.now\(\) - lastProgressAt >= 150000/);
  assert.doesNotMatch(route, /demo_timeout_fallback/);
  assert.doesNotMatch(route, /WORKSTATION_SESSION_SECRET/);
});
