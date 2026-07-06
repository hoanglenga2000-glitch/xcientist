import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const outJson = join(root, "workspace", "workstation_interactive_controls_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_INTERACTIVE_CONTROLS_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");
const port = Number(process.env.WORKSTATION_CONTROL_AUDIT_CDP_PORT ?? "9224");

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

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
].filter(Boolean);

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

async function evalValue(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text ?? "Runtime evaluation failed.");
  return result.result?.value;
}

async function waitForPage(client) {
  for (let attempt = 0; attempt < 80; attempt++) {
    const ok = await evalValue(client, "document.readyState === 'complete' && !!document.querySelector('[data-ui-component=\"workstation-page\"]')");
    if (ok) return;
    await sleep(150);
  }
  throw new Error("Page shell did not become ready.");
}

async function navigate(client, page) {
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  await waitForPage(client);
  await sleep(300);
}

async function inspectControls(client, page) {
  await navigate(client, page);
  const result = await evalValue(client, `(() => {
    const activePage = document.querySelector('[data-ui-component="workstation-page"]')?.getAttribute('data-ui-page') ?? null;
    const selector = [
      'button',
      'a[href]',
      'input',
      'select',
      'textarea',
      '[role="button"]',
      '[tabindex]:not([tabindex="-1"])',
      '[data-ui-action]',
      '[data-testid]'
    ].join(',');
    const items = Array.from(document.querySelectorAll(selector)).map((el, index) => {
      const tag = el.tagName.toLowerCase();
      const label = (el.getAttribute('aria-label') || el.textContent || el.getAttribute('placeholder') || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
      const disabled = Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true';
      const action = el.getAttribute('data-ui-action');
      const testid = el.getAttribute('data-testid');
      const component = el.getAttribute('data-ui-component');
      const href = el.getAttribute('href');
      const role = el.getAttribute('role');
      const type = el.getAttribute('type');
      const hasInlineHandler = Array.from(el.getAttributeNames()).some((name) => name.toLowerCase().startsWith('on'));
      const hasRouteContract = Boolean(action || testid || component || href || disabled || ['input','select','textarea'].includes(tag));
      const rect = el.getBoundingClientRect();
      return {
        index,
        tag,
        type,
        label,
        action,
        testid,
        component,
        href,
        role,
        disabled,
        hasInlineHandler,
        visible: rect.width > 0 && rect.height > 0,
        ok: hasRouteContract
      };
    }).filter((item) => item.visible);
    const missing = items.filter((item) => !item.ok);
    return {
      activePage,
      total: items.length,
      withAction: items.filter((item) => item.action).length,
      withTestId: items.filter((item) => item.testid).length,
      disabled: items.filter((item) => item.disabled).length,
      missing,
      sample: items.slice(0, 12)
    };
  })()`);
  return {
    page,
    ok: result.activePage === page && result.missing.length === 0,
    ...result
  };
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_interactive_controls.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_unavailable",
      page_results: [],
      failed_pages: pageTargets,
      missing_control_count: null,
      claim_boundary: "No Chromium-compatible browser was found, so interactive controls audit could not run."
    };
  }

  const userDataDir = join(root, "workspace", `.chrome-control-audit-${Date.now()}`);
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
    for (const page of pageTargets) pageResults.push(await inspectControls(client, page));

    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });
    const failedPages = pageResults.filter((item) => !item.ok).map((item) => item.page);
    const missingControlCount = pageResults.reduce((total, item) => total + item.missing.length, 0);
    return {
      schema: "academic_research_os.workstation_interactive_controls.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: failedPages.length === 0 && missingControlCount === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: null,
      chrome,
      page_results: pageResults,
      failed_pages: failedPages,
      missing_control_count: missingControlCount,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This audit inspects visible interactive controls after client-side routing has settled. A control passes if it has data-ui-action, data-testid, data-ui-component, href, input semantics, or an explicit disabled state. It does not start training, GPU jobs, Kaggle submission, or Figma writes."
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdown(report) {
  const lines = [
    "# 工作站交互控件审计",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 工作站地址：\`${report.base_url}\``,
    `- 状态：\`${report.status}\``,
    `- 浏览器：\`${report.chrome ?? "not_found"}\``,
    `- 失败页面：\`${report.failed_pages?.join(", ") || "none"}\``,
    `- 缺少契约的控件数：\`${report.missing_control_count ?? "unknown"}\``,
    `- 运行时错误数：\`${report.runtime_error_count ?? 0}\``,
    "",
    "## 页面覆盖",
    "",
    "| page | ok | controls | data-ui-action | data-testid | disabled | missing |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
  ];
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.page}\` | \`${item.ok}\` | ${item.total} | ${item.withAction} | ${item.withTestId} | ${item.disabled} | ${item.missing.length} |`);
  }
  lines.push("", "## 缺少契约的控件", "");
  const missing = (report.page_results ?? []).flatMap((item) => item.missing.map((control) => ({ page: item.page, ...control })));
  if (missing.length === 0) {
    lines.push("none", "");
  } else {
    lines.push("| page | tag | label | action | testid |", "| --- | --- | --- | --- | --- |");
    for (const item of missing.slice(0, 100)) {
      lines.push(`| \`${item.page}\` | \`${item.tag}\` | \`${item.label || "-"}\` | \`${item.action || "-"}\` | \`${item.testid || "-"}\` |`);
    }
  }
  lines.push("", "## Claim Boundary", "", report.claim_boundary, "");
  return lines.join("\n");
}

function toMarkdownClean(report) {
  const lines = [
    "# 工作站交互控件审计",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 工作站地址：\`${report.base_url}\``,
    `- 状态：\`${report.status}\``,
    `- 浏览器：\`${report.chrome ?? "not_found"}\``,
    `- 失败页面：\`${report.failed_pages?.join(", ") || "none"}\``,
    `- 缺少交互契约的控件数：\`${report.missing_control_count ?? "unknown"}\``,
    `- 运行时错误数：\`${report.runtime_error_count ?? 0}\``,
    "",
    "## 页面覆盖",
    "",
    "| page | ok | controls | data-ui-action | data-testid | disabled | missing |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.page}\` | \`${item.ok}\` | ${item.total} | ${item.withAction} | ${item.withTestId} | ${item.disabled} | ${item.missing.length} |`);
  }
  lines.push("", "## 缺少契约的控件", "");
  const missing = (report.page_results ?? []).flatMap((item) => item.missing.map((control) => ({ page: item.page, ...control })));
  if (missing.length === 0) {
    lines.push("none", "");
  } else {
    lines.push("| page | tag | label | action | testid |", "| --- | --- | --- | --- | --- |");
    for (const item of missing.slice(0, 100)) {
      lines.push(`| \`${item.page}\` | \`${item.tag}\` | \`${item.label || "-"}\` | \`${item.action || "-"}\` | \`${item.testid || "-"}\` |`);
    }
  }
  lines.push("", "## 声明边界", "", report.claim_boundary, "");
  return lines.join("\n");
}

const report = await run();
if (writeReport) {
  await mkdir(dirname(outJson), { recursive: true });
  await mkdir(dirname(outMd), { recursive: true });
  await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(outMd, `\ufeff${toMarkdownClean(report)}`, "utf8");
}

console.log(JSON.stringify({
  status: report.status,
  failed_pages: report.failed_pages,
  missing_control_count: report.missing_control_count,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_interactive_controls_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_INTERACTIVE_CONTROLS_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
