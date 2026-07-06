import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const outJson = join(root, "workspace", "workstation_click_smoke_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_CLICK_SMOKE_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");
const port = Number(process.env.WORKSTATION_CDP_PORT ?? String(9223 + (process.pid % 1000)));

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
].filter(Boolean);

const pageTargets = [
  "overview",
  "control",
  "tasks",
  "data",
  "gpu",
  "evidence",
  "literature",
  "workflow",
  "code",
  "runtime",
  "experiments",
  "report",
  "gates",
  "settings"
];

const safeClicks = [
  { page: "overview", selector: "[data-ui-action='topbar_open_code']", expectedPage: "code" },
  { page: "code", selector: "[data-ui-action='topbar_open_report']", expectedPage: "report" },
  { page: "report", selector: "[data-ui-action='topbar_open_gpu']", expectedPage: "gpu" },
  { page: "gpu", selector: "[data-ui-action='topbar_open_gates']", expectedPage: "gates" },
  { page: "overview", selector: "[data-ui-action='mission_open_evidence_ledger']", expectedPage: "evidence" },
  { page: "overview", selector: "[data-ui-action='mission_view_workflow_details']", expectedPage: "workflow" },
  { page: "overview", selector: "[data-ui-action='mission_view_all_claim_audits']", expectedPage: "gates" },
  { page: "tasks", selector: "[data-ui-action='tasks_open_context']", expectedPage: "tasks" },
  { page: "tasks", selector: "[data-ui-action='tasks_view_agent_logs']", expectedPage: "runtime" },
  { page: "tasks", selector: "[data-ui-action='tasks_copy_validation_contract']", expectedPage: "tasks" },
  { page: "code", selector: "[data-ui-action='code_filter_task']", expectedPage: "code" },
  { page: "code", selector: "[data-ui-action='open_code_folder_experiments']", expectedPage: "code" },
  { page: "code", selector: "[data-ui-action='ask_code_agent']", expectedPage: "code" },
  { page: "code", selector: "[data-ui-action='request_code_quality_gate']", expectedPage: "gates" },
  { page: "evidence", selector: "[data-ui-action='apply_evidence_filters']", expectedPage: "evidence" },
  { page: "evidence", selector: "[data-ui-action='open_evidence_lineage_graph']", expectedPage: "evidence" },
  { page: "evidence", selector: "[data-ui-action='preview_selected_artifact']", expectedPage: "evidence" },
  { page: "literature", selector: "[data-ui-action='literature_refresh_library']", expectedPage: "literature" },
  { page: "literature", selector: "[data-ui-action='rag_send_code_agent']", expectedPage: "code" },
  { page: "runtime", selector: "[data-ui-action='runtime_refresh_5s']", expectedPage: "runtime" },
  { page: "runtime", selector: "[data-ui-action='runtime_open_agent_artifact']", expectedPage: "evidence" },
  { page: "experiments", selector: "[data-ui-action='experiments_filter_graph']", expectedPage: "experiments" },
  { page: "experiments", selector: "[data-ui-action='experiments_open_artifacts']", expectedPage: "evidence" },
  { page: "report", selector: "[data-ui-action='report_add_section']", expectedPage: "report" },
  { page: "report", selector: "[data-ui-action='report_export_draft_pdf']", expectedPage: "report" },
  { page: "gates", selector: "[data-ui-action='request_gate_revision']", expectedPage: "gates" },
  { page: "settings", selector: "[data-ui-action='settings_language_en_us']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='settings_language_zh_cn']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='settings_theme_light']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='settings_theme_dark']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='open_settings_section_security_credentials']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='save_settings_changes']", expectedPage: "settings" },
  { page: "settings", selector: "[data-ui-action='test_all_connectors']", expectedPage: "settings" }
];

const blockedControls = [
  { page: "gpu", selector: "[data-ui-action='blocked_start_training']" },
  { page: "code", selector: "[data-ui-action='blocked_send_to_hpc']" },
  { page: "gates", selector: "[data-ui-action='blocked_allow_official_submit']" },
  { page: "report", selector: "[data-ui-action='blocked_final_report_export']" },
  { page: "evidence", selector: "[data-ui-action='blocked_final_evidence_approval']" }
];

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

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
    this.socket = new WebSocket(this.wsUrl);
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

async function waitForPage(client) {
  for (let attempt = 0; attempt < 80; attempt++) {
    const result = await client.send("Runtime.evaluate", {
      expression: "document.readyState === 'complete' && !!document.querySelector('[data-ui-component=\"workstation-page\"]')",
      returnByValue: true
    });
    if (result.result?.value === true) return;
    await sleep(150);
  }
  throw new Error("Page shell did not become ready.");
}

async function evalValue(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text ?? "Runtime evaluation failed.");
  }
  return result.result?.value;
}

async function navigate(client, page) {
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  await waitForPage(client);
  await sleep(250);
}

async function inspectPage(client, page) {
  await navigate(client, page);
  const info = await evalValue(client, `(() => {
    const marker = document.querySelector('[data-ui-component="workstation-page"]');
    return {
      activePage: marker?.getAttribute('data-ui-page') ?? null,
      actionCount: document.querySelectorAll('[data-ui-action]').length,
      buttonCount: document.querySelectorAll('button').length,
      textSize: document.body.innerText.length,
      hasErrorText: /Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(document.body.innerText)
    };
  })()`);
  return {
    page,
    ok: info.activePage === page && info.actionCount >= 5 && info.buttonCount >= 3 && info.textSize >= 1000 && !info.hasErrorText,
    ...info
  };
}

async function clickAndInspect(client, item) {
  await navigate(client, item.page);
  const result = await evalValue(client, `(() => {
    const target = document.querySelector(${JSON.stringify(item.selector)});
    if (!target) return { clicked: false, reason: 'selector_not_found' };
    target.click();
    return { clicked: true, label: (target.textContent || target.getAttribute('aria-label') || '').trim().slice(0, 80) };
  })()`);
  await sleep(650);
  const activePage = await evalValue(client, `document.querySelector('[data-ui-component="workstation-page"]')?.getAttribute('data-ui-page') ?? null`);
  const hasErrorText = await evalValue(client, `/Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(document.body.innerText)`);
  return {
    ...item,
    clicked: Boolean(result.clicked),
    label: result.label ?? null,
    reason: result.reason ?? null,
    activePage,
    ok: Boolean(result.clicked) && activePage === item.expectedPage && !hasErrorText
  };
}

async function inspectBlockedControl(client, item) {
  await navigate(client, item.page);
  const result = await evalValue(client, `(() => {
    const target = document.querySelector(${JSON.stringify(item.selector)});
    if (!target) return { found: false, reason: 'selector_not_found' };
    const style = getComputedStyle(target);
    return {
      found: true,
      label: (target.textContent || target.getAttribute('aria-label') || '').trim().slice(0, 80),
      disabled: Boolean(target.disabled) || target.getAttribute('aria-disabled') === 'true',
      pointerEvents: style.pointerEvents,
      cursor: style.cursor
    };
  })()`);
  const hasErrorText = await evalValue(client, `/Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(document.body.innerText)`);
  return {
    ...item,
    ...result,
    ok: Boolean(result.found) && Boolean(result.disabled) && !hasErrorText
  };
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_click_smoke.v2",
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

  const userDataDir = join(root, "workspace", `.chrome-click-smoke-${Date.now()}`);
  await mkdir(userDataDir, { recursive: true });
  const chromeProcess = spawn(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    `${baseUrl}/?page=overview`
  ], { stdio: "ignore" });

  let client;
  let cleanupWarning = null;
  try {
    const version = await waitForChrome(port);
    const tabs = await fetchJson(`http://127.0.0.1:${port}/json`);
    const tab = tabs.find((item) => item.type === "page") ?? tabs[0];
    client = new CdpClient(tab.webSocketDebuggerUrl ?? version.webSocketDebuggerUrl);
    await client.connect();
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Log.enable");

    const pageResults = [];
    for (const page of pageTargets) pageResults.push(await inspectPage(client, page));

    const clickResults = [];
    for (const item of safeClicks) clickResults.push(await clickAndInspect(client, item));

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
      schema: "academic_research_os.workstation_click_smoke.v2",
      created_at: createdAt,
      base_url: baseUrl,
      status: failedPages.length === 0 && failedClicks.length === 0 && failedBlockedControls.length === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: null,
      chrome,
      page_results: pageResults,
      click_results: clickResults,
      blocked_control_results: blockedResults,
      failed_pages: failedPages,
      failed_clicks: failedClicks,
      failed_blocked_controls: failedBlockedControls,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This smoke uses a real headless Chromium browser and safe clicks only. It verifies direct navigation, safe UI actions, and blocked training/submission controls. It does not start training, GPU jobs, Kaggle submission, or Figma writes."
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
    "| page | ok | active page | actions | buttons | text size |",
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
  failed_blocked_controls: report.failed_blocked_controls,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_click_smoke_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_CLICK_SMOKE_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
