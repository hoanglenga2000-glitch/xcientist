import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const outJson = join(root, "workspace", "workstation_responsive_smoke_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_RESPONSIVE_SMOKE_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");
const port = Number(process.env.WORKSTATION_RESPONSIVE_CDP_PORT ?? String(9323 + (process.pid % 1000)));

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

const viewports = [
  { name: "desktop", width: 1440, height: 900, mobile: false, minTextSize: 1000 },
  { name: "laptop", width: 1366, height: 768, mobile: false, minTextSize: 1000 },
  { name: "tablet", width: 834, height: 1112, mobile: false, minTextSize: 900 },
  { name: "mobile", width: 390, height: 844, mobile: true, minTextSize: 700 }
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

async function setViewport(client, viewport) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: 1,
    mobile: viewport.mobile
  });
}

async function inspectPage(client, viewport, page) {
  await setViewport(client, viewport);
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  await waitForPage(client);
  await sleep(350);
  const info = await evalValue(client, `(() => {
    const pageEl = document.querySelector('[data-ui-component="workstation-page"]');
    const activePage = pageEl?.getAttribute('data-ui-page') ?? null;
    const body = document.body;
    const doc = document.documentElement;
    const maxDocumentWidth = Math.max(body.scrollWidth, doc.scrollWidth, body.offsetWidth, doc.offsetWidth);
    const horizontalOverflow = maxDocumentWidth - window.innerWidth;
    const visibleButtons = Array.from(document.querySelectorAll('button')).filter((el) => {
      const rect = el.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    }).length;
    const visibleActions = Array.from(document.querySelectorAll('[data-ui-action]')).filter((el) => {
      const rect = el.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    }).length;
    const textSize = body.innerText.length;
    const hasRuntimeError = /Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(body.innerText);
    const offscreenWideElements = Array.from(document.querySelectorAll('body *')).filter((el) => {
      const rect = el.getBoundingClientRect();
      return rect.width > window.innerWidth + 80 && rect.height > 0;
    }).slice(0, 10).map((el) => ({
      tag: el.tagName.toLowerCase(),
      className: String(el.className || '').slice(0, 120),
      width: Math.round(el.getBoundingClientRect().width)
    }));
    return {
      activePage,
      innerWidth: window.innerWidth,
      scrollWidth: maxDocumentWidth,
      horizontalOverflow,
      visibleButtons,
      visibleActions,
      textSize,
      hasRuntimeError,
      offscreenWideElements
    };
  })()`);
  const allowedOverflow = viewport.mobile ? 32 : 24;
  return {
    viewport: viewport.name,
    page,
    ok:
      info.activePage === page &&
      info.visibleButtons >= 3 &&
      info.visibleActions >= 5 &&
      info.textSize >= viewport.minTextSize &&
      info.horizontalOverflow <= allowedOverflow &&
      !info.hasRuntimeError,
    allowedOverflow,
    ...info
  };
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_responsive_smoke.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_unavailable",
      page_results: [],
      failed_results: [],
      runtime_error_count: null,
      claim_boundary: "No Chromium-compatible browser was found, so responsive smoke could not run."
    };
  }

  const userDataDir = join(root, "workspace", `.chrome-responsive-smoke-${Date.now()}`);
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
    for (const viewport of viewports) {
      for (const page of pageTargets) {
        pageResults.push(await inspectPage(client, viewport, page));
      }
    }

    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });
    const failedResults = pageResults.filter((item) => !item.ok);
    return {
      schema: "academic_research_os.workstation_responsive_smoke.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: failedResults.length === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: null,
      chrome,
      viewport_count: viewports.length,
      page_count: pageTargets.length,
      check_count: pageResults.length,
      page_results: pageResults,
      failed_results: failedResults,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This responsive smoke checks desktop, laptop, tablet, and mobile viewport rendering after client-side routing has settled. It verifies active page, visible controls, text volume, runtime errors, and horizontal overflow. It does not start training, GPU jobs, Kaggle submission, or Figma writes."
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdown(report) {
  const lines = [
    "# 工作站多视口响应式 Smoke",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 工作站地址：\`${report.base_url}\``,
    `- 状态：\`${report.status}\``,
    `- 浏览器：\`${report.chrome ?? "not_found"}\``,
    `- 视口数：\`${report.viewport_count ?? 0}\``,
    `- 页面数：\`${report.page_count ?? 0}\``,
    `- 检查总数：\`${report.check_count ?? 0}\``,
    `- 失败总数：\`${report.failed_results?.length ?? 0}\``,
    `- 运行时错误数：\`${report.runtime_error_count ?? 0}\``,
    "",
    "## 失败项",
    ""
  ];
  if (!report.failed_results?.length) {
    lines.push("none", "");
  } else {
    lines.push("| viewport | page | overflow | buttons | actions | text |", "| --- | --- | ---: | ---: | ---: | ---: |");
    for (const item of report.failed_results) {
      lines.push(`| \`${item.viewport}\` | \`${item.page}\` | ${item.horizontalOverflow} | ${item.visibleButtons} | ${item.visibleActions} | ${item.textSize} |`);
    }
  }
  lines.push("## 页面矩阵", "", "| viewport | page | ok | overflow | buttons | actions | text |", "| --- | --- | --- | ---: | ---: | ---: | ---: |");
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.viewport}\` | \`${item.page}\` | \`${item.ok}\` | ${item.horizontalOverflow} | ${item.visibleButtons} | ${item.visibleActions} | ${item.textSize} |`);
  }
  lines.push("", "## Claim Boundary", "", report.claim_boundary, "");
  return lines.join("\n");
}

function toMarkdownClean(report) {
  const lines = [
    "# 工作站多视口响应式 Smoke",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 工作站地址：\`${report.base_url}\``,
    `- 状态：\`${report.status}\``,
    `- 浏览器：\`${report.chrome ?? "not_found"}\``,
    `- 视口数：\`${report.viewport_count ?? 0}\``,
    `- 页面数：\`${report.page_count ?? 0}\``,
    `- 检查总数：\`${report.check_count ?? 0}\``,
    `- 失败总数：\`${report.failed_results?.length ?? 0}\``,
    `- 运行时错误数：\`${report.runtime_error_count ?? 0}\``,
    "",
    "## 失败项",
    "",
  ];
  if (!report.failed_results?.length) {
    lines.push("none", "");
  } else {
    lines.push("| viewport | page | overflow | buttons | actions | text |", "| --- | --- | ---: | ---: | ---: | ---: |");
    for (const item of report.failed_results) {
      lines.push(`| \`${item.viewport}\` | \`${item.page}\` | ${item.horizontalOverflow} | ${item.visibleButtons} | ${item.visibleActions} | ${item.textSize} |`);
    }
  }
  lines.push("## 页面矩阵", "", "| viewport | page | ok | overflow | buttons | actions | text |", "| --- | --- | --- | ---: | ---: | ---: | ---: |");
  for (const item of report.page_results ?? []) {
    lines.push(`| \`${item.viewport}\` | \`${item.page}\` | \`${item.ok}\` | ${item.horizontalOverflow} | ${item.visibleButtons} | ${item.visibleActions} | ${item.textSize} |`);
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
  check_count: report.check_count ?? 0,
  failed_count: report.failed_results?.length ?? 0,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_responsive_smoke_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_RESPONSIVE_SMOKE_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
