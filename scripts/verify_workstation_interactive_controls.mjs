import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { createServer } from "node:net";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { applyReleaseCheckAuth } from "./workstation_test_session.mjs";
import {
  WORKSPACE_VIEWS,
  clearInput,
  clickSelector,
  createDraftTask,
  evaluate,
  inspectWorkspace,
  openView,
  pressKey,
  setDesktopViewport,
  shellOk,
  sleep,
  typeInto,
  waitFor
} from "./workstation_workspace_probe.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const requireFromWeb = createRequire(new URL("../web/research-agent-workstation/package.json", import.meta.url));
const WebSocketClient = globalThis.WebSocket ?? requireFromWeb("ws");
const outJson = join(root, "workspace", "workstation_interactive_controls_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_INTERACTIVE_CONTROLS_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");

async function allocateCdpPort() {
  if (process.env.WORKSTATION_CONTROL_AUDIT_CDP_PORT) return Number(process.env.WORKSTATION_CONTROL_AUDIT_CDP_PORT);
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
    return new Promise((resolveSend, reject) => {
      this.pending.set(id, { resolve: resolveSend, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
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

// Every view of the shipped /workspace UI plus the session view of a draft task.
const pageTargets = [...WORKSPACE_VIEWS.map((view) => view.id), "task_session"];
const viewPaths = Object.fromEntries(WORKSPACE_VIEWS.map((view) => [view.id, view.path]));
let sessionTaskId = null;

function pathFor(page) {
  if (page === "task_session") return sessionTaskId ? `/workspace?task=${encodeURIComponent(sessionTaskId)}` : null;
  return viewPaths[page];
}

// Contract for every visible control of the shipped UI: it has an accessible
// name, is keyboard reachable (or explicitly disabled), and links stay inside
// the product (workspace views, the skip target, verified file downloads).
const CONTROL_AUDIT = `(() => {
  const selector = 'button, a[href], input, select, textarea, summary, [role="button"], [tabindex]:not([tabindex="-1"])';
  const nameOf = (el) => {
    const labelledBy = (el.getAttribute('aria-labelledby') || '').split(/\\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ');
    const forLabel = el.id ? Array.from(document.querySelectorAll('label')).filter(l => l.htmlFor === el.id).map(l => l.textContent).join(' ') : '';
    const wrapping = el.closest('label')?.textContent || '';
    return (el.getAttribute('aria-label') || labelledBy || forLabel || wrapping || el.textContent || el.getAttribute('title') || '').replace(/\\s+/g, ' ').trim();
  };
  const allowedHref = (href) => {
    if (href === '#task-main') return !!document.getElementById('task-main');
    const url = new URL(href, location.origin);
    if (url.origin !== location.origin) return false;
    return url.pathname === '/workspace' || url.pathname === '/login' || /^\\/api\\/assistant\\/files\\/[^/]+$/.test(url.pathname);
  };
  const items = Array.from(document.querySelectorAll(selector)).map((el, index) => {
    const tag = el.tagName.toLowerCase();
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const skipLink = el.classList.contains('tw-skip');
    const visible = skipLink || (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none');
    const disabled = Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true';
    const label = nameOf(el).slice(0, 120);
    const href = el.getAttribute('href');
    const problems = [];
    if (!label) problems.push('missing_accessible_name');
    if (!disabled && el.tabIndex < 0) problems.push('not_keyboard_reachable');
    if (tag === 'a' && href && !allowedHref(href)) problems.push('link_leaves_product_or_unknown_route');
    return { index, tag, type: el.getAttribute('type'), label, href, disabled, visible, problems, ok: problems.length === 0 };
  }).filter(item => item.visible);
  return {
    total: items.length,
    named: items.filter(item => item.label).length,
    links: items.filter(item => item.tag === 'a').length,
    disabled: items.filter(item => item.disabled).length,
    missing: items.filter(item => !item.ok),
    sample: items.slice(0, 12)
  };
})()`;

async function inspectControls(client, page) {
  const path = pathFor(page);
  if (!path) return { page, ok: false, activePage: null, total: 0, named: 0, links: 0, disabled: 0, missing: [], reason: "no_task_for_session_view" };
  const ready = await openView(client, baseUrl, path, page);
  const info = await inspectWorkspace(client);
  const audit = await evaluate(client, CONTROL_AUDIT);
  return {
    page,
    ok: ready && info.activePage === page && shellOk(info) && audit.total >= 10 && audit.missing.length === 0 && !info.hasErrorText && info.alerts.length === 0,
    activePage: info.activePage,
    alerts: info.alerts,
    ...audit
  };
}

// Behaviour probes: the control must visibly respond, not merely exist.
async function probe(name, page, fn) {
  try {
    const details = await fn();
    return { name, page, ...details, ok: Boolean(details.ok) };
  } catch (error) {
    return { name, page, ok: false, error: String(error?.message ?? error) };
  }
}

const setNativeValue = (selector, value) => `(() => {
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return false;
  const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? 'change' : 'input', { bubbles: true }));
  return true;
})()`;

async function behaviourProbes(client) {
  const results = [];

  results.push(await probe("skip_link_keyboard", "tasks", async () => {
    await openView(client, baseUrl, pathFor("tasks"), "tasks");
    await evaluate(client, "document.activeElement?.blur(); window.focus(); true");
    await pressKey(client, "Tab", "Tab", 9);
    const first = await evaluate(client, "document.activeElement?.classList.contains('tw-skip') ?? false");
    await pressKey(client, "Enter", "Enter", 13);
    const landed = await waitFor(client, "location.hash === '#task-main' && document.activeElement?.id === 'task-main'", 3000);
    return { firstTabStopIsSkipLink: first, focusMovedToMain: landed, ok: first && landed };
  }));

  results.push(await probe("search_filters_task_list", "tasks", async () => {
    await openView(client, baseUrl, pathFor("tasks"), "tasks");
    const rows = "document.querySelectorAll('main#task-main .tw-task-list a.tw-task-row').length";
    const before = await evaluate(client, rows);
    const firstTitle = await evaluate(client, "document.querySelector('main#task-main .tw-task-row h2')?.textContent ?? ''");
    await typeInto(client, "input[aria-label='搜索任务']", `zz-无此任务-${Date.now()}`);
    const emptied = await waitFor(client, `${rows} === 0 && Array.from(document.querySelectorAll('main#task-main .tw-empty h2')).some(n => n.textContent === '没有匹配的任务')`, 3000);
    await clearInput(client, "input[aria-label='搜索任务']");
    const restored = await waitFor(client, `${rows} === ${before}`, 3000);
    let matched = true;
    if (before > 0 && firstTitle) {
      await typeInto(client, "input[aria-label='搜索任务']", firstTitle.slice(0, 12));
      matched = await waitFor(client, `${rows} >= 1 && Array.from(document.querySelectorAll('main#task-main .tw-task-row h2')).some(n => n.textContent.includes(${JSON.stringify(firstTitle.slice(0, 12))}))`, 3000);
      await clearInput(client, "input[aria-label='搜索任务']");
    }
    return { rowsBefore: before, emptyStateOnNoMatch: emptied, restoredAfterClear: restored, matchedExistingTitle: matched, ok: before >= 1 && emptied && restored && matched };
  }));

  results.push(await probe("mobile_menu_toggle", "tasks", async () => {
    await client.send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    try {
      await openView(client, baseUrl, pathFor("tasks"), "tasks");
      const state = "(() => { const b = document.querySelector('header.tw-topbar button.tw-menu'); const a = document.getElementById('workspace-navigation'); return { expanded: b?.getAttribute('aria-expanded'), label: b?.getAttribute('aria-label'), open: a?.classList.contains('tw-open'), sidebarVisible: a ? getComputedStyle(a).display !== 'none' : false, focusInSidebar: a?.contains(document.activeElement) ?? false, focusOnButton: document.activeElement === b }; })()";
      const closed = await evaluate(client, state);
      const click = await clickSelector(client, "header.tw-topbar button.tw-menu");
      await waitFor(client, "document.querySelector('header.tw-topbar button.tw-menu')?.getAttribute('aria-expanded') === 'true'", 3000);
      const opened = await evaluate(client, state);
      await pressKey(client, "Escape", "Escape", 27);
      await waitFor(client, "document.querySelector('header.tw-topbar button.tw-menu')?.getAttribute('aria-expanded') === 'false'", 3000);
      const reclosed = await evaluate(client, state);
      return {
        closed, opened, reclosed, clicked: click.clicked,
        ok: click.clicked && closed.expanded === "false" && !closed.sidebarVisible &&
          opened.expanded === "true" && opened.open && opened.sidebarVisible && opened.label === "收起导航" && opened.focusInSidebar &&
          reclosed.expanded === "false" && !reclosed.sidebarVisible && reclosed.focusOnButton
      };
    } finally {
      await setDesktopViewport(client);
    }
  }));

  results.push(await probe("task_session_tabs", "task_session", async () => {
    await openView(client, baseUrl, pathFor("task_session"), "task_session");
    const visited = [];
    for (const [label, view, marker] of [
      ["进度", "progress", "Array.from(document.querySelectorAll('main#task-main .tw-empty h2')).some(n => n.textContent === '尚未开始')"],
      ["文件与结果", "results", "Array.from(document.querySelectorAll('main#task-main .tw-empty h2')).some(n => n.textContent === '尚未开始')"],
      ["更多", "more", "Array.from(document.querySelectorAll('main#task-main h2')).some(n => n.textContent === '关联已有记录')"],
      ["对话", "conversation", "!!document.querySelector('main#task-main form.tw-composer textarea#task-message')"]
    ]) {
      const selector = `main#task-main nav.tw-tabs[aria-label='任务内容'] button:nth-child(${["对话", "进度", "文件与结果", "更多"].indexOf(label) + 1})`;
      const click = await clickSelector(client, selector);
      const ok = click.clicked && click.label === label && await waitFor(client, `(() => { const cur = document.querySelector("main#task-main nav.tw-tabs button[aria-current='page']"); const v = new URLSearchParams(location.search).get('view') || 'conversation'; return cur?.textContent === ${JSON.stringify(label)} && v === ${JSON.stringify(view)} && ${marker}; })()`, 5000);
      visited.push({ label, view, ok });
    }
    return { visited, ok: visited.every((item) => item.ok) };
  }));

  results.push(await probe("start_requires_explicit_model", "task_session", async () => {
    await openView(client, baseUrl, pathFor("task_session"), "task_session");
    const submit = "main#task-main form.tw-composer button[type='submit'].tw-primary";
    const state = `(() => ({ disabled: document.querySelector(${JSON.stringify(submit)})?.disabled ?? null, model: document.querySelector('select#task-model')?.value ?? null, draft: (document.querySelector('textarea#task-message')?.value || '').trim().length }))()`;
    const initial = await evaluate(client, state);
    const hasPlatform = await evaluate(client, "Array.from(document.querySelectorAll('select#task-model option')).some(o => o.value === 'platform')");
    await evaluate(client, setNativeValue("select#task-model", "platform"));
    await sleep(300);
    const chosen = await evaluate(client, state);
    await evaluate(client, setNativeValue("select#task-model", ""));
    await sleep(300);
    const reset = await evaluate(client, state);
    // Nothing is sent: the probe never clicks the submit button.
    return {
      initial, chosen, reset, hasPlatform,
      ok: initial.model === "" && initial.disabled === true && hasPlatform && chosen.model === "platform" &&
        chosen.disabled === (chosen.draft === 0) && reset.disabled === true
    };
  }));

  results.push(await probe("details_disclosure", "task_session", async () => {
    await openView(client, baseUrl, pathFor("task_session"), "task_session");
    const summary = "main#task-main details.tw-details > summary";
    const isOpen = "(document.querySelector('main#task-main details.tw-details')?.open ?? null)";
    const before = await evaluate(client, isOpen);
    await clickSelector(client, summary);
    const opened = await waitFor(client, `${isOpen} === true`, 2000);
    await clickSelector(client, summary);
    const closed = await waitFor(client, `${isOpen} === false`, 2000);
    return { before, opened, closed, ok: before === false && opened && closed };
  }));

  results.push(await probe("project_name_gate", "projects", async () => {
    await openView(client, baseUrl, pathFor("projects"), "projects");
    const button = "(document.querySelector('main#task-main form.tw-settings-form button')?.disabled ?? null)";
    const empty = await evaluate(client, button);
    await typeInto(client, "input#new-project", "验收临时名称");
    const enabled = await waitFor(client, `${button} === false`, 2000);
    await clearInput(client, "input#new-project");
    const disabledAgain = await waitFor(client, `${button} === true`, 2000);
    // The form is never submitted, so no project is created.
    return { emptyDisabled: empty, enabledWithName: enabled, disabledAfterClear: disabledAgain, ok: empty === true && enabled && disabledAgain };
  }));

  results.push(await probe("theme_preview_and_undo", "settings_account", async () => {
    await openView(client, baseUrl, pathFor("settings_account"), "settings_account");
    const mode = "(document.documentElement.dataset.themeMode ?? null)";
    const original = await evaluate(client, "document.querySelector('select#workspace-theme')?.value ?? null");
    const target = original === "light" ? "dark" : "light";
    await evaluate(client, setNativeValue("select#workspace-theme", target));
    const previewed = await waitFor(client, `${mode} === ${JSON.stringify(target)} && document.documentElement.dataset.theme === ${JSON.stringify(target)}`, 3000);
    const undo = await evaluate(client, "Array.from(document.querySelectorAll('main#task-main button')).find(b => b.textContent === '撤销未保存的外观') ? true : false");
    if (undo) {
      await evaluate(client, "(() => { const b = Array.from(document.querySelectorAll('main#task-main button')).find(n => n.textContent === '撤销未保存的外观'); b.setAttribute('data-probe-target', 'undo-theme'); return true; })()");
      await clickSelector(client, "[data-probe-target='undo-theme']");
    }
    const reverted = await waitFor(client, `${mode} === ${JSON.stringify(original)} && document.querySelector('select#workspace-theme')?.value === ${JSON.stringify(original)}`, 3000);
    // Only previewed and undone; "保存外观" is never clicked.
    return { original, target, previewed, undoFound: undo, reverted, ok: Boolean(original) && previewed && undo && reverted };
  }));

  return results;
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_interactive_controls.v2",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_unavailable",
      chrome: null,
      page_results: [],
      behaviour_results: [],
      failed_pages: pageTargets,
      missing_control_count: null,
      claim_boundary: "No Chromium-compatible browser was found, so the control audit could not run."
    };
  }

  const port = await allocateCdpPort();

  const userDataDir = join(root, "workspace", `.chrome-control-audit-${Date.now()}`);
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
    const authMode = await applyReleaseCheckAuth(client, root, baseUrl);
    process.env.WORKSTATION_RELEASE_CHECK_AUTH_MODE = authMode;
    await client.send("Runtime.enable");
    await client.send("Log.enable");
    await setDesktopViewport(client);

    // The session view needs a task owned by this account: reuse the newest
    // one, or create a draft through the UI (drafts never call a model).
    await openView(client, baseUrl, "/workspace", "tasks");
    sessionTaskId = await evaluate(client, "fetch('/api/assistant/tasks', { cache: 'no-store' }).then(r => r.ok ? r.json() : { tasks: [] }).then(body => body.tasks?.[0]?.id ?? null)");
    let createdDraft = null;
    if (!sessionTaskId) {
      createdDraft = await createDraftTask(client, baseUrl, `交互控件审计草稿 ${createdAt}`);
      sessionTaskId = createdDraft.ok ? createdDraft.taskId : null;
    }

    const pageResults = [];
    for (const page of pageTargets) pageResults.push(await inspectControls(client, page));
    const behaviourResults = await behaviourProbes(client);

    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });
    const failedPages = pageResults.filter((item) => !item.ok).map((item) => item.page);
    const missingControlCount = pageResults.reduce((total, item) => total + item.missing.length, 0);
    const failedBehaviours = behaviourResults.filter((item) => !item.ok).map((item) => item.name);
    return {
      schema: "academic_research_os.workstation_interactive_controls.v2",
      created_at: createdAt,
      base_url: baseUrl,
      ui: "workspace",
      auth_mode: authMode,
      status: authMode === "local_session" && failedPages.length === 0 && missingControlCount === 0 && failedBehaviours.length === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: authMode === "local_session" ? null : "release_check_session_missing",
      chrome,
      session_task_id: sessionTaskId,
      created_draft_task: Boolean(createdDraft?.ok),
      page_results: pageResults,
      behaviour_results: behaviourResults,
      failed_pages: failedPages,
      failed_behaviours: failedBehaviours,
      missing_control_count: missingControlCount,
      failed_count: failedPages.length + failedBehaviours.length,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This audit signs in with the release-check session and inspects every visible control of each shipped /workspace view in a real headless Chromium: each must have an accessible name, be keyboard reachable or explicitly disabled, and link only to product routes. Behaviour probes verify the controls respond: keyboard skip link, task search filtering, the mobile navigation toggle (open, focus, Escape), task session tabs, the explicit-model gate on 开始任务, disclosure widgets, the project-name gate and theme preview/undo. It never sends a task, saves settings, creates a project, calls a model, starts training or GPU jobs, submits to Kaggle or writes to Figma."
    };
  } catch (error) {
    return {
      schema: "academic_research_os.workstation_interactive_controls.v2",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_cdp_unavailable",
      chrome,
      chrome_stderr_tail: chromeStderr.join("").slice(-4000),
      error: String(error?.message ?? error),
      page_results: [], behaviour_results: [], failed_pages: pageTargets, missing_control_count: null
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdownCleanUtf8(report) {
  const lines = [
    "# Workstation Interactive Controls Audit",
    "",
    `- created_at: \`${report.created_at}\``,
    `- base_url: \`${report.base_url}\``,
    `- status: \`${report.status}\``,
    `- auth_mode: \`${report.auth_mode ?? "unknown"}\``,
    `- browser: \`${report.chrome ?? "not_found"}\``,
    `- failed_pages: \`${report.failed_pages?.join(", ") || "none"}\``,
    `- failed_behaviours: \`${report.failed_behaviours?.join(", ") || "none"}\``,
    `- missing_control_count: \`${report.missing_control_count ?? "unknown"}\``,
    `- runtime_error_count: \`${report.runtime_error_count ?? 0}\``,
    "",
    "## View Coverage",
    "",
    "| view | ok | detected view | controls | named | links | disabled | failing |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.page}\` | \`${item.ok}\` | \`${item.activePage}\` | ${item.total} | ${item.named} | ${item.links} | ${item.disabled} | ${item.missing.length} |`);
  }
  lines.push("", "## Behaviour Probes", "", "| probe | view | ok |", "| --- | --- | --- |");
  for (const item of report.behaviour_results ?? []) lines.push(`| \`${item.name}\` | \`${item.page}\` | \`${item.ok}\` |`);
  lines.push("", "## Failing Controls", "");
  const missing = (report.page_results ?? []).flatMap((item) => item.missing.map((control) => ({ page: item.page, ...control })));
  if (missing.length === 0) {
    lines.push("none", "");
  } else {
    lines.push("| view | tag | label | href | problems |", "| --- | --- | --- | --- | --- |");
    for (const item of missing.slice(0, 100)) {
      lines.push(`| \`${item.page}\` | \`${item.tag}\` | \`${item.label || "-"}\` | \`${item.href || "-"}\` | \`${item.problems.join(", ")}\` |`);
    }
  }
  lines.push("", "## Claim Boundary", "", report.claim_boundary, "");
  return lines.join("\n");
}

const report = await run();
if (writeReport) {
  await mkdir(dirname(outJson), { recursive: true });
  await mkdir(dirname(outMd), { recursive: true });
  await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(outMd, `\ufeff${toMarkdownCleanUtf8(report)}`, "utf8");
}

console.log(JSON.stringify({
  status: report.status,
  auth_mode: report.auth_mode ?? null,
  failed_pages: report.failed_pages,
  failed_page_details: report.page_results?.filter((item) => !item.ok) ?? [],
  failed_behaviours: report.behaviour_results?.filter((item) => !item.ok) ?? [],
  missing_control_count: report.missing_control_count,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_interactive_controls_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_INTERACTIVE_CONTROLS_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
