import { spawn } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { createServer } from "node:net";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { applyReleaseCheckAuth } from "./workstation_test_session.mjs";
import {
  WORKSPACE_VIEWS,
  clickSelector,
  createDraftTask,
  currentView,
  evaluate,
  inspectWorkspace,
  openView,
  setDesktopViewport,
  shellOk,
  sleep,
  waitForWorkspace
} from "./workstation_workspace_probe.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const requireFromWeb = createRequire(new URL("../web/research-agent-workstation/package.json", import.meta.url));
const WebSocketClient = globalThis.WebSocket ?? requireFromWeb("ws");
const outJson = join(root, "workspace", "workstation_click_smoke_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_CLICK_SMOKE_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");

async function allocateCdpPort() {
  if (process.env.WORKSTATION_CDP_PORT) return Number(process.env.WORKSTATION_CDP_PORT);
  return await new Promise((resolvePort, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const selected = typeof address === "object" && address ? address.port : 0;
      server.close((error) => error ? reject(error) : resolvePort(selected));
    });
  });
}

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser"
].filter(Boolean);

function findChrome() {
  return chromeCandidates.find((candidate) => candidate && existsSync(candidate)) ?? null;
}

async function fetchJson(url, timeoutMs = 8000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

async function stopBrowser(process) {
  if (!process || process.killed) return;
  process.kill();
  await new Promise((resolveStop) => {
    const timer = setTimeout(resolveStop, 2500);
    process.once("exit", () => {
      clearTimeout(timer);
      resolveStop();
    });
  });
}

async function cleanupUserDataDir(userDataDir) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      await rm(userDataDir, { recursive: true, force: true });
      return null;
    } catch (error) {
      if (error?.code !== "EBUSY" && error?.code !== "EPERM") throw error;
      await sleep(400 + attempt * 300);
    }
  }
  return `cleanup_deferred:${userDataDir}`;
}

class CdpClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
  }

  async connect() {
    this.socket = new WebSocketClient(this.wsUrl);
    await new Promise((resolveConnect, reject) => {
      const timer = setTimeout(() => reject(new Error("CDP websocket connection timeout")), 10000);
      this.socket.addEventListener("open", () => {
        clearTimeout(timer);
        resolveConnect();
      }, { once: true });
      this.socket.addEventListener("error", (event) => {
        clearTimeout(timer);
        reject(new Error(`CDP websocket error: ${event.message ?? "unknown"}`));
      }, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const payload = JSON.parse(String(event.data));
      if (payload.id && this.pending.has(payload.id)) {
        const { resolve: resolvePending, reject } = this.pending.get(payload.id);
        this.pending.delete(payload.id);
        if (payload.error) reject(new Error(payload.error.message ?? JSON.stringify(payload.error)));
        else resolvePending(payload.result ?? {});
        return;
      }
      this.events.push(payload);
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    const message = JSON.stringify({ id, method, params });
    return new Promise((resolveSend, reject) => {
      this.pending.set(id, { resolve: resolveSend, reject });
      this.socket.send(message);
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP command timeout: ${method}`));
        }
      }, 12000);
    });
  }

  close() {
    this.socket?.close();
  }
}

async function waitForChrome(portNumber) {
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      return await fetchJson(`http://127.0.0.1:${portNumber}/json/version`, 2000);
    } catch {
      await sleep(200);
    }
  }
  throw new Error("Chrome DevTools endpoint did not become ready.");
}

// Views of the shipped /workspace UI. The legacy ?page= dashboard is retired:
// app/page.tsx redirects every ?page= link to /workspace, which the
// legacy_redirect target below verifies in a real browser.
const pageTargets = [...WORKSPACE_VIEWS.map((view) => view.id), "legacy_redirect"];

// Safe clicks: real mouse clicks on shipped navigation controls. Each must land
// on the expected view (re-derived from the DOM) with the expected sidebar item
// marked aria-current. None of them starts a model run, training or submission.
const safeClicks = [
  { page: "tasks", selector: "#workspace-navigation nav a[href='/workspace?section=data']", expectedPage: "data", expectedNav: "/workspace?section=data" },
  { page: "data", selector: "#workspace-navigation nav a[href='/workspace?section=literature']", expectedPage: "literature", expectedNav: "/workspace?section=literature" },
  { page: "literature", selector: "#workspace-navigation nav a[href='/workspace?section=results']", expectedPage: "results", expectedNav: "/workspace?section=results" },
  { page: "results", selector: "#workspace-navigation nav a[href^='/workspace?settings=models']", expectedPage: "settings_models", expectedNav: "/workspace?settings=models" },
  { page: "settings_models", selector: "main#task-main nav[aria-label='设置分组'] a[href^='/workspace?settings=resources']", expectedPage: "settings_resources", expectedNav: "/workspace?settings=models" },
  { page: "settings_resources", selector: "header.tw-topbar a[href^='/workspace?settings=account']", expectedPage: "settings_account", expectedNav: "/workspace?settings=models" },
  { page: "settings_account", selector: "#workspace-navigation .tw-sidebar-note a[href='/workspace?section=projects']", expectedPage: "projects", expectedNav: null },
  { page: "projects", selector: "#workspace-navigation a.tw-new", expectedPage: "new_task", expectedNav: null },
  { page: "new_task", selector: "main#task-main a.tw-back[href='/workspace']", expectedPage: "tasks", expectedNav: "/workspace" },
  { page: "data", selector: "#workspace-navigation a.tw-brand", expectedPage: "tasks", expectedNav: "/workspace" }
];

// Controls that must refuse to act until the person supplies what they need.
// "开始任务" is the only control that can trigger a (billable) model run; it
// must stay disabled until a model is explicitly chosen. task_session targets
// are checked on a draft task created through the UI during this smoke.
const blockedControls = [
  { page: "new_task", selector: "main#task-main form.tw-composer button.tw-primary", reason: "empty_draft_cannot_be_saved" },
  { page: "projects", selector: "main#task-main form.tw-settings-form button", reason: "empty_project_name_cannot_be_created" },
  { page: "task_session", selector: "main#task-main form.tw-composer button[type='submit'].tw-primary", reason: "model_run_requires_explicit_model_choice" }
];

const viewPaths = Object.fromEntries(WORKSPACE_VIEWS.map((view) => [view.id, view.path]));
const viewNav = { ...Object.fromEntries(WORKSPACE_VIEWS.map((view) => [view.id, view.nav])), task_session: "/workspace" };
let draftTask = null;

function pathFor(page) {
  if (page === "task_session") return draftTask?.taskId ? `/workspace?task=${encodeURIComponent(draftTask.taskId)}` : null;
  return viewPaths[page] ?? null;
}

async function inspectPage(client, page) {
  if (page === "legacy_redirect") {
    // Old bookmarks (/?page=overview) must still land on the shipped workspace.
    await client.send("Page.navigate", { url: `${baseUrl}/?page=overview` });
    const ready = await waitForWorkspace(client, "tasks");
    const info = await inspectWorkspace(client);
    return {
      page,
      ok: ready && info.url.startsWith("/workspace") && info.activePage === "tasks" && shellOk(info) && !info.hasErrorText && info.alerts.length === 0,
      ...info,
      activePage: info.url.startsWith("/workspace") && info.activePage === "tasks" ? "legacy_redirect" : info.activePage
    };
  }
  const ready = await openView(client, baseUrl, pathFor(page), page);
  const info = await inspectWorkspace(client);
  const expectedNav = viewNav[page];
  const navOk = expectedNav === null ? info.navCurrent.length === 0 : info.navCurrent.length === 1 && info.navCurrent[0].startsWith(expectedNav);
  return {
    page,
    // A view passes only when the shipped shell rendered, the DOM shows this
    // view's own content, the matching sidebar item is current, no alert or
    // error text is visible, and the view rendered substantive text.
    ok: ready && info.activePage === page && shellOk(info) && navOk && info.actionCount >= 10 && info.buttonCount >= 1 && info.mainTextSize >= 20 && info.textSize >= 80 && !info.hasErrorText && info.alerts.length === 0,
    navOk,
    ...info
  };
}

async function clickAndInspect(client, item) {
  const opened = await openView(client, baseUrl, pathFor(item.page), item.page);
  const result = opened ? await clickSelector(client, item.selector) : { clicked: false, reason: "start_view_not_ready" };
  const landed = result.clicked ? await waitForWorkspace(client, item.expectedPage) : false;
  const info = await inspectWorkspace(client);
  const navOk = item.expectedNav === null ? info.navCurrent.length === 0 : info.navCurrent.length === 1 && info.navCurrent[0].startsWith(item.expectedNav);
  return {
    ...item,
    clicked: Boolean(result.clicked),
    label: result.label ?? null,
    reason: result.reason ?? null,
    activePage: info.activePage,
    url: info.url,
    navCurrent: info.navCurrent,
    ok: Boolean(result.clicked) && landed && info.activePage === item.expectedPage && navOk && !info.hasErrorText && info.alerts.length === 0
  };
}

async function inspectBlockedControl(client, item) {
  const path = pathFor(item.page);
  if (!path) return { ...item, found: false, reason: "draft_task_not_created", ok: false };
  const opened = await openView(client, baseUrl, path, item.page);
  const result = await evaluate(client, `(() => {
    const target = document.querySelector(${JSON.stringify(item.selector)});
    if (!target) return { found: false, reason: 'selector_not_found' };
    return {
      found: true,
      label: (target.textContent || target.getAttribute('aria-label') || '').trim().slice(0, 80),
      disabled: Boolean(target.disabled) || target.getAttribute('aria-disabled') === 'true'
    };
  })()`);
  const activePage = await currentView(client);
  // The gate must hold under a real click too: clicking a disabled control
  // must not navigate away or start anything.
  let afterClickView = activePage;
  if (result.found) {
    await clickSelector(client, item.selector);
    await new Promise((resolveWait) => setTimeout(resolveWait, 600));
    afterClickView = await currentView(client);
  }
  const info = await inspectWorkspace(client);
  return {
    ...item,
    ...result,
    activePage,
    afterClickView,
    ok: opened && Boolean(result.found) && Boolean(result.disabled) && afterClickView === item.page && !info.hasErrorText && info.alerts.length === 0
  };
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_click_smoke.v3",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_unavailable",
      chrome: null,
      page_results: [],
      click_results: [],
      blocked_control_results: [],
      failed_pages: pageTargets,
      failed_clicks: safeClicks.map((item) => item.selector),
      failed_blocked_controls: blockedControls.map((item) => item.selector),
      claim_boundary: "No Chromium-compatible browser was found, so real click smoke could not run."
    };
  }

  const port = await allocateCdpPort();

  const userDataDir = join(root, "workspace", `.chrome-click-smoke-${Date.now()}`);
  await mkdir(userDataDir, { recursive: true });
  const chromeProcess = spawn(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    "about:blank"
  ], { stdio: ["ignore", "ignore", "pipe"] });
  const chromeStderr = [];
  chromeProcess.stderr?.on("data", (chunk) => chromeStderr.push(String(chunk)));

  let client;
  let cleanupWarning = null;
  try {
    const version = await waitForChrome(port);
    const tabs = await fetchJson(`http://127.0.0.1:${port}/json`);
    const tab = tabs.find((item) => item.type === "page") ?? tabs[0];
    client = new CdpClient(tab.webSocketDebuggerUrl ?? version.webSocketDebuggerUrl);
    await client.connect();
    await client.send("Page.enable");
    await client.send("Network.enable");
    // Session first (real signed-in principal), automation header fallback.
    const authMode = await applyReleaseCheckAuth(client, root, baseUrl);
    process.env.WORKSTATION_RELEASE_CHECK_AUTH_MODE = authMode;
    await client.send("Runtime.enable");
    await client.send("Log.enable");
    await setDesktopViewport(client);

    const pageResults = [];
    for (const page of pageTargets) pageResults.push(await inspectPage(client, page));

    const clickResults = [];
    for (const item of safeClicks) clickResults.push(await clickAndInspect(client, item));

    // Core flow: describe a task -> it auto-saves as a draft (no model call)
    // and opens in its own session view.
    const draftTitle = `发布验收冒烟草稿 ${createdAt}`;
    draftTask = await createDraftTask(client, baseUrl, draftTitle);
    clickResults.push({
      page: "new_task",
      selector: "textarea#task-draft",
      expectedPage: "task_session",
      clicked: true,
      label: "输入需求并自动保存草稿",
      reason: draftTask.ok ? null : (draftTask.reason ?? "draft_not_saved"),
      activePage: draftTask.ok ? "task_session" : await currentView(client),
      url: draftTask.url ?? null,
      ok: Boolean(draftTask.ok)
    });
    if (draftTask.ok) pageResults.push(await inspectPage(client, "task_session"));
    else pageResults.push({ page: "task_session", ok: false, activePage: null, actionCount: 0, buttonCount: 0, textSize: 0, hasErrorText: false, reason: "draft_task_not_created" });

    const blockedResults = [];
    for (const item of blockedControls) blockedResults.push(await inspectBlockedControl(client, item));

    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });
    const failedPages = pageResults.filter((item) => !item.ok).map((item) => item.page);
    const failedClicks = clickResults.filter((item) => !item.ok).map((item) => item.selector);
    const failedBlockedControls = blockedResults.filter((item) => !item.ok).map((item) => item.selector);
    return {
      schema: "academic_research_os.workstation_click_smoke.v3",
      created_at: createdAt,
      base_url: baseUrl,
      ui: "workspace",
      auth_mode: authMode,
      status: authMode === "local_session" && failedPages.length === 0 && failedClicks.length === 0 && failedBlockedControls.length === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: authMode === "local_session" ? null : "release_check_session_missing",
      chrome,
      draft_task_id: draftTask?.taskId ?? null,
      page_results: pageResults,
      click_results: clickResults,
      blocked_control_results: blockedResults,
      failed_pages: failedPages,
      failed_clicks: failedClicks,
      failed_blocked_controls: failedBlockedControls,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This smoke signs in with the release-check session and drives the shipped /workspace UI in a real headless Chromium with real mouse clicks. It verifies every workspace view renders its own content inside the shipped shell, sidebar/top-bar/settings navigation, the legacy ?page= redirect, creating a draft task through the composer (drafts never call a model), and that saving an empty draft, creating an unnamed project and starting a model run without an explicit model choice stay disabled. It does not send a task, call a model, start training or GPU jobs, submit to Kaggle or write to Figma."
    };
  } catch (error) {
    return {
      schema: "academic_research_os.workstation_click_smoke.v3",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_cdp_unavailable",
      chrome,
      chrome_stderr_tail: chromeStderr.join("").slice(-4000),
      error: String(error?.message ?? error),
      page_results: [], click_results: [], blocked_control_results: [],
      failed_pages: pageTargets,
      failed_clicks: safeClicks.map((item) => item.selector),
      failed_blocked_controls: blockedControls.map((item) => item.selector)
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdown(report) {
  const lines = [
    "# \u5de5\u4f5c\u7ad9\u771f\u5b9e\u70b9\u51fb\u5192\u70df\u6d4b\u8bd5",
    "",
    `- \u751f\u6210\u65f6\u95f4\uff1a\`${report.created_at}\``,
    `- \u5de5\u4f5c\u7ad9\u5730\u5740\uff1a\`${report.base_url}\``,
    `- \u72b6\u6001\uff1a\`${report.status}\``,
    `- \u6d4f\u89c8\u5668\uff1a\`${report.chrome ?? "not_found"}\``,
    `- \u9875\u9762\u76f4\u8fbe\u6570\uff1a\`${report.page_results?.length ?? 0}\``,
    `- \u5b89\u5168\u70b9\u51fb\u6570\uff1a\`${report.click_results?.length ?? 0}\``,
    `- \u963b\u65ad\u63a7\u4ef6\u68c0\u67e5\u6570\uff1a\`${report.blocked_control_results?.length ?? 0}\``,
    `- \u5931\u8d25\u9875\u9762\uff1a\`${report.failed_pages?.join(", ") || "none"}\``,
    `- \u5931\u8d25\u70b9\u51fb\uff1a\`${report.failed_clicks?.join(", ") || "none"}\``,
    `- \u9519\u8bef\u5f00\u653e\u7684\u963b\u65ad\u63a7\u4ef6\uff1a\`${report.failed_blocked_controls?.join(", ") || "none"}\``,
    `- \u8fd0\u884c\u65f6\u9519\u8bef\u6570\uff1a\`${report.runtime_error_count ?? 0}\``,
    "",
    "## \u9875\u9762\u76f4\u8fbe",
    "",
    "| view | ok | detected view | controls | buttons | text size |",
    "| --- | --- | --- | ---: | ---: | ---: |"
  ];
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.page}\` | \`${item.ok}\` | \`${item.activePage}\` | ${item.actionCount} | ${item.buttonCount} | ${item.textSize} |`);
  }
  lines.push("", "## \u5b89\u5168\u70b9\u51fb", "", "| selector | from | expected | active | clicked | ok |", "| --- | --- | --- | --- | --- | --- |");
  for (const item of report.click_results ?? []) {
    lines.push(`| \`${item.selector}\` | \`${item.page}\` | \`${item.expectedPage}\` | \`${item.activePage}\` | \`${item.clicked}\` | \`${item.ok}\` |`);
  }
  lines.push("", "## \u963b\u65ad\u63a7\u4ef6", "", "| selector | page | found | disabled | ok |", "| --- | --- | --- | --- | --- |");
  for (const item of report.blocked_control_results ?? []) {
    lines.push(`| \`${item.selector}\` | \`${item.page}\` | \`${item.found}\` | \`${item.disabled}\` | \`${item.ok}\` |`);
  }
  lines.push("", "## Claim Boundary", "", report.claim_boundary, "");
  return lines.join("\n");
}

const report = await run();
if (writeReport) {
  await mkdir(dirname(outJson), { recursive: true });
  await mkdir(dirname(outMd), { recursive: true });
  await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(outMd, `\ufeff${toMarkdown(report)}`, "utf8");
}

console.log(JSON.stringify({
  status: report.status,
  failed_pages: report.failed_pages,
  failed_clicks: report.failed_clicks,
  failed_page_details: report.page_results?.filter((item) => !item.ok) ?? [],
  failed_click_details: report.click_results?.filter((item) => !item.ok) ?? [],
  failed_blocked_controls: report.failed_blocked_controls,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_click_smoke_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_CLICK_SMOKE_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
