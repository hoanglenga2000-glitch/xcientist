import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const requireFromWorkstation = createRequire(join(root, "web", "research-agent-workstation", "package.json"));
const WebSocketClient = globalThis.WebSocket ?? requireFromWorkstation("next/dist/compiled/ws");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const outDir = resolve(process.argv.includes("--out-dir")
  ? process.argv[process.argv.indexOf("--out-dir") + 1]
  : join(root, "artifacts", "design-qa", "implementation"));
const port = Number(process.env.WORKSTATION_DESIGN_QA_CDP_PORT ?? String(10623 + (process.pid % 1000)));

const targets = [
  { page: "assistant", viewport: "desktop", width: 1440, height: 900 },
  { page: "overview", viewport: "desktop", width: 1440, height: 900 },
  { page: "control", viewport: "desktop", width: 1440, height: 900 },
  { page: "runtime", viewport: "desktop", width: 1440, height: 900 },
  { page: "report", viewport: "desktop", width: 1440, height: 900 },
  { page: "settings", viewport: "desktop", width: 1440, height: 900 },
  { page: "assistant", viewport: "mobile", width: 390, height: 844 },
  { page: "overview", viewport: "mobile", width: 390, height: 844 },
  { page: "runtime", viewport: "mobile", width: 390, height: 844 },
  { page: "report", viewport: "mobile", width: 390, height: 844 },
];

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function findChrome() {
  return chromeCandidates.find((candidate) => candidate && existsSync(candidate)) ?? null;
}

async function fetchJson(url, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
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
        const pending = this.pending.get(payload.id);
        this.pending.delete(payload.id);
        if (payload.error) pending.reject(new Error(payload.error.message ?? JSON.stringify(payload.error)));
        else pending.resolve(payload.result ?? {});
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
      }, 15000);
    });
  }

  close() {
    this.socket?.close();
  }
}

async function waitForChrome(portNumber) {
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      return await fetchJson(`http://127.0.0.1:${portNumber}/json/version`, 1500);
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
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text ?? "Runtime evaluation failed.");
  return result.result?.value;
}

async function waitForStablePage(client, page) {
  for (let attempt = 0; attempt < 100; attempt++) {
    const ready = await evalValue(client, `(async () => {
      await document.fonts.ready;
      const workstation = document.querySelector('[data-ui-component="workstation-page"]');
      return document.readyState === 'complete'
        && workstation?.getAttribute('data-ui-page') === ${JSON.stringify(page)}
        && document.documentElement.dataset.theme === 'dark';
    })()`);
    if (ready) {
      await sleep(page === "report" ? 3500 : page === "assistant" ? 1400 : 900);
      return;
    }
    await sleep(150);
  }
  throw new Error(`Page ${page} did not reach a stable dark-theme state.`);
}

async function captureTarget(client, target) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: target.width,
    height: target.height,
    deviceScaleFactor: 1,
    mobile: target.viewport === "mobile",
    screenWidth: target.width,
    screenHeight: target.height,
  });
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${target.page}` });
  await waitForStablePage(client, target.page);

  const metrics = await evalValue(client, `(() => {
    const root = document.documentElement;
    const body = document.body;
    const visibleControls = Array.from(document.querySelectorAll('button,a[href],input,select,textarea,[role="button"]'))
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
      });
    const undersized = visibleControls.map((element) => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        action: element.getAttribute('data-ui-action'),
        label: (element.getAttribute('aria-label') || element.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 80),
        width: Math.round(rect.width * 10) / 10,
        height: Math.round(rect.height * 10) / 10,
        minWidth: style.minWidth,
        minHeight: style.minHeight,
      };
    }).filter((item) => item.width < 44 || item.height < 44);
    return {
      theme: root.dataset.theme,
      colorScheme: getComputedStyle(root).colorScheme,
      clientWidth: root.clientWidth,
      scrollWidth: Math.max(root.scrollWidth, body.scrollWidth),
      horizontalOverflow: Math.max(root.scrollWidth, body.scrollWidth) > root.clientWidth + 1,
      shellBackgroundImage: getComputedStyle(body).backgroundImage,
      visibleControlCount: visibleControls.length,
      undersizedControls: undersized,
    };
  })()`);

  const screenshot = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  const filename = `${target.page}-${target.viewport}.png`;
  await writeFile(join(outDir, filename), Buffer.from(screenshot.data, "base64"));
  return { ...target, filename, ...metrics };
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

async function cleanupDirectory(path) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      await rm(path, { recursive: true, force: true });
      return null;
    } catch (error) {
      if (error?.code !== "EBUSY" && error?.code !== "EPERM") throw error;
      await sleep(350 + attempt * 250);
    }
  }
  return `cleanup_deferred:${path}`;
}

async function run() {
  const chrome = findChrome();
  if (!chrome) throw new Error("No Chromium-compatible browser was found.");
  await mkdir(outDir, { recursive: true });
  const userDataDir = join(root, "workspace", `.chrome-design-qa-${Date.now()}`);
  await mkdir(userDataDir, { recursive: true });
  const chromeProcess = spawn(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    `${baseUrl}/?page=assistant`,
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
    const results = [];
    for (const target of targets) results.push(await captureTarget(client, target));
    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const value = JSON.stringify(event.params ?? {});
      const ignored = /favicon\.ico/.test(value) && /404|Not Found/.test(value);
      return !ignored && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error|Uncaught/i.test(value));
    });
    const report = {
      schema: "evomind.workstation.design_qa_capture.v1",
      created_at: new Date().toISOString(),
      base_url: baseUrl,
      status: results.every((item) => !item.horizontalOverflow) && runtimeErrors.length === 0 ? "passed" : "failed",
      results,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 20),
      cleanup_warning: cleanupWarning,
    };
    await writeFile(join(outDir, "capture-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
    return report;
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupDirectory(userDataDir);
  }
}

try {
  const report = await run();
  console.log(JSON.stringify({
    status: report.status,
    captures: report.results.length,
    overflow_failures: report.results.filter((item) => item.horizontalOverflow).map((item) => `${item.page}-${item.viewport}`),
    runtime_error_count: report.runtime_error_count,
    out_dir: outDir,
  }, null, 2));
  process.exitCode = report.status === "passed" ? 0 : 1;
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
}
