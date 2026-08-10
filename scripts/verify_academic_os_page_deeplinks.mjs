import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argValue = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
};
const baseUrl = argValue("--url", "http://127.0.0.1:8088").replace(/\/$/, "");
const outDirArg = argValue("--out-dir", path.join(root, "docs", "evomind_pages_20260708"));
const outDir = path.isAbsolute(outDirArg) ? outDirArg : path.resolve(root, outDirArg);
const port = Number(process.env.EVOMIND_DEEPLINK_CDP_PORT ?? String(10400 + (process.pid % 1000)));

const pages = {
  overview: [["Research Overview", "科研总览"], ["Research workstation operating status and mission control", "科研工作站运行态势与闭环总控"], ["Research Loop Stage", "研究循环阶段"]],
  control: [["EvoMind Gateway", "EvoMind 工作站入口"], ["Scientist Autopilot", "科学家诊断"], ["Scientist Action Queue", "科学家行动队列"]],
  tasks: [["Task Queue", "任务队列"], ["All tasks in the research loop", "研究闭环中的全部任务"], ["Tasks", "任务列表"]],
  experiments: [["Experiment Ledger", "实验台账"], ["Score Trend", "分数趋势"], ["Run Ledger", "运行台账"]],
  evolution: [["Evolution Engine", "自进化引擎"], ["Search graph", "搜索图"], ["Retrospective memory", "回顾记忆"]],
  workflow: [["Workflow Graph", "工作流图"], ["Agent Handoff Chain", "Agent 交接链"], ["Gates & Fallbacks", "门控与回退"]],
  runtime: [
    ["Agent Runtime", "Agent 运行时"],
    ["Dynamic Task Graph", "动态任务图"],
    ["Current Run Event Stream", "当前运行事件流"],
    ["HPC Runtime", "HPC 运行环境"],
    ["Reviewer", "独立审核"],
    ["Report & Deliverables", "报告与交付物"],
    ["Official Kaggle Submission", "Kaggle 正式提交", "Model Publication Gate", "模型发布 Gate"]
  ],
  data: [["Data / Kaggle", "数据 / Kaggle"], ["Dataset audit", "数据集审计"], ["Kaggle Submission", "Kaggle 提交"]],
  code: [["Code Agent IDE", "代码 Agent IDE"], ["Agent Session", "Agent 会话"], ["Quality Gate", "质量门"]],
  literature: [["Literature / RAG", "文献 / RAG"], ["Verified Search & Import", "真实检索与文献导入"], ["Task Papers", "当前任务论文"], ["Agent Evidence Workflow", "Agent 证据工作流"], ["Independent Reviewer", "独立 Reviewer"]],
  report: [
    ["Report Studio", "报告工作室"],
    ["Independent Review"],
    ["Generate Scientific Report", "生成科研报告", "科研报告"],
    ["Continue with natural language"]
  ],
  gpu: [["GPU / HPC"], ["HPC Connector Detail", "HPC 连接器详情"], ["Job History", "作业历史"]],
  evidence: [["Evidence Ledger", "证据账本"], ["Claim Boundary", "声明边界"], ["Evidence Records", "证据记录"]],
  gates: [["Integrity Gates", "完整性闸门"], ["Gate Pipeline", "闸门流水线"], ["Gate Decisions", "闸门决议"]],
  settings: [["Settings", "设置"], ["General", "通用"], ["Governance", "治理"]]
};

const browserCandidates = [
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

function findBrowser() {
  return browserCandidates.find((candidate) => candidate && existsSync(candidate)) ?? null;
}

async function fetchJson(url, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function ensureReachable(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error(`Dashboard is not reachable: HTTP ${response.status}`);
}

async function readAutomationToken() {
  const parsedBase = new URL(baseUrl);
  if (parsedBase.protocol !== "http:" || parsedBase.hostname !== "127.0.0.1") {
    throw new Error("Deep-link authentication is restricted to the loopback dashboard.");
  }
  const dashboardPort = Number(parsedBase.port || "80");
  const suffix = dashboardPort === 8088 ? "" : `.${dashboardPort}`;
  const runtimeDir = process.env.WORKSTATION_RUNTIME_DIR
    ? path.resolve(process.env.WORKSTATION_RUNTIME_DIR)
    : path.join(root, "web", "research-agent-workstation", ".runtime-logs");
  const source = path.join(runtimeDir, `dashboard${suffix}.automation.token`);
  const token = (await readFile(source, "ascii").catch(() => "")).trim();
  if (!/^[A-Za-z0-9_-]{24,256}$/.test(token)) {
    throw new Error("Dashboard local automation token is missing or malformed.");
  }
  return token;
}

async function establishLocalSession(client) {
  await client.send("Page.navigate", { url: `${baseUrl}/?page=assistant` });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    await sleep(150);
    const state = await evalValue(client, `(async () => {
      if (document.readyState !== 'complete') return { ready: false };
      const response = await fetch('/api/session/status', { cache: 'no-store', credentials: 'same-origin' }).catch(() => null);
      if (!response?.ok) return { ready: false, status: response?.status ?? null };
      const payload = await response.json().catch(() => ({}));
      return { ready: payload?.authenticated === true || Boolean(payload?.csrf_token), status: response.status };
    })()`);
    if (state?.ready === true) return;
  }
  throw new Error("Chromium could not establish the dashboard local session.");
}

async function waitForChrome(portNumber, browserProcess) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      return await fetchJson(`http://127.0.0.1:${portNumber}/json/version`, 2000);
    } catch {
      if (browserProcess.exitCode !== null) break;
      await sleep(200);
    }
  }
  throw new Error("Chromium DevTools endpoint did not become ready.");
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
      this.socket.addEventListener("error", () => {
        clearTimeout(timer);
        reject(new Error("CDP websocket connection failed"));
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
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolveSend, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.has(id)) return;
        this.pending.delete(id);
        reject(new Error(`CDP command timeout: ${method}`));
      }, 15000);
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolveSend(value);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        }
      });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.socket?.close();
  }
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

async function setViewport(client, width, height, mobile) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile
  });
}

async function waitForPage(client, page) {
  let lastState = null;
  for (let attempt = 0; attempt < 80; attempt += 1) {
    await sleep(150);
    const state = await evalValue(client, `(() => {
      const pageRoot = document.querySelector('[data-ui-component="workstation-page"]');
      return {
        documentReady: document.readyState === 'complete',
        uiReady: pageRoot?.getAttribute('data-ui-ready') === 'true',
        ariaBusy: pageRoot?.getAttribute('aria-busy') ?? null,
        page: pageRoot?.getAttribute('data-ui-page') ?? null,
        task: pageRoot?.getAttribute('data-ui-task') ?? '',
        textSize: document.body?.innerText?.length ?? 0,
        h1Count: document.querySelectorAll('main h1').length,
        headingCount: document.querySelectorAll('main h1, main h2, main [role="heading"]').length
      };
    })()`);
    lastState = state;
    const ready = state?.documentReady
      && state.uiReady
      && state.ariaBusy === "false"
      && state.page === page
      && state.task.length > 0
      && state.textSize >= 120
      && state.headingCount >= 1;
    if (ready) return state;
  }
  throw new Error(`Page did not reach its UI readiness contract: ${page}; state=${JSON.stringify(lastState)}`);
}

async function navigate(client, page) {
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  return await waitForPage(client, page);
}

async function captureScreenshot(client, outputPath) {
  const capture = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false
  });
  const bytes = Buffer.from(capture.data, "base64");
  if (bytes.length < 10000) {
    throw new Error(`Screenshot is too small: ${outputPath} (${bytes.length} bytes)`);
  }
  await writeFile(outputPath, bytes);
  return bytes.length;
}

function runtimeErrors(events) {
  return events.filter((event) => {
    const method = event.method ?? "";
    const params = event.params ?? {};
    const text = JSON.stringify(params);
    if (/favicon\.ico/.test(text) && /404|Not Found/.test(text)) return false;
    if (
      method === "Log.entryAdded"
      && params.entry?.url === `${baseUrl}/api/session/status`
      && params.entry?.level === "error"
      && /status of 401 \(Unauthorized\)/.test(String(params.entry?.text ?? ""))
    ) return false;
    if (method === "Runtime.exceptionThrown") return true;
    if (method === "Runtime.consoleAPICalled" && params.type === "error") return true;
    if (method === "Log.entryAdded" && params.entry?.level === "error") return true;
    return /ChunkLoadError|Hydration failed|Unhandled Runtime Error|Internal Server Error/i.test(text);
  });
}

async function stopBrowser(browserProcess, client) {
  if (client) {
    await Promise.race([client.send("Browser.close"), sleep(2000)]).catch(() => undefined);
    client.close();
  }
  if (browserProcess.exitCode === null) {
    browserProcess.kill();
    for (let attempt = 0; attempt < 20 && browserProcess.exitCode === null; attempt += 1) {
      await sleep(100);
    }
  }
}

async function run() {
  const browser = findBrowser();
  if (!browser) throw new Error(`No supported Chromium browser found: ${browserCandidates.join(", ")}`);
  await ensureReachable(baseUrl);
  const automationToken = await readAutomationToken();
  await mkdir(outDir, { recursive: true });

  const profileDir = await mkdtemp(path.join(os.tmpdir(), "evomind-deeplinks-"));
  const browserProcess = spawn(browser, [
    "--headless=new",
    "--disable-gpu",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-component-update",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-features=Translate,MediaRouter,BackForwardCache",
    "--disable-sync",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profileDir}`,
    "about:blank"
  ], { stdio: "ignore" });

  let client = null;
  try {
    const version = await waitForChrome(port, browserProcess);
    const targets = await fetchJson(`http://127.0.0.1:${port}/json/list`);
    const target = targets.find((item) => item.type === "page") ?? targets[0];
    if (!target?.webSocketDebuggerUrl && !version.webSocketDebuggerUrl) {
      throw new Error("Chromium page target has no websocket debugger URL.");
    }
    client = new CdpClient(target?.webSocketDebuggerUrl ?? version.webSocketDebuggerUrl);
    await client.connect();
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Log.enable");
    await client.send("Network.enable");
    await client.send("Network.setExtraHTTPHeaders", {
      headers: { "x-evomind-local-automation": automationToken }
    });
    await establishLocalSession(client);

    const artifacts = [];
    const missing = {};
    const pageResults = [];
    for (const [page, termGroups] of Object.entries(pages)) {
      const eventStart = client.events.length;
      await setViewport(client, 1440, 1100, false);
      const desktopState = await navigate(client, page);
      const visibleText = await evalValue(client, "document.body.innerText");
      const dom = await evalValue(client, "document.documentElement.outerHTML");
      const pageMissing = termGroups
        .filter((group) => !group.some((term) => visibleText.includes(term)))
        .map((group) => group.join(" | "));
      if (pageMissing.length > 0) missing[page] = pageMissing;

      const domPath = path.join(outDir, `${page}.html`);
      const desktopPath = path.join(outDir, `${page}_desktop.png`);
      const mobilePath = path.join(outDir, `${page}_mobile.png`);
      await writeFile(domPath, dom, "utf8");
      const desktopBytes = await captureScreenshot(client, desktopPath);

      await setViewport(client, 390, 1000, true);
      const mobileState = await navigate(client, page);
      const mobileBytes = await captureScreenshot(client, mobilePath);
      const pageErrors = runtimeErrors(client.events.slice(eventStart));

      artifacts.push(
        path.relative(root, domPath),
        path.relative(root, desktopPath),
        path.relative(root, mobilePath)
      );
      pageResults.push({
        page,
        desktop_text_size: desktopState.textSize,
        mobile_text_size: mobileState.textSize,
        desktop_png_bytes: desktopBytes,
        mobile_png_bytes: mobileBytes,
        missing_terms: pageMissing,
        runtime_error_count: pageErrors.length
      });
    }

    const errors = runtimeErrors(client.events);
    const indexPath = path.join(outDir, "acceptance_index.json");
    const indexArtifact = path.relative(root, indexPath);
    artifacts.push(indexArtifact);
    const report = {
      status: Object.keys(missing).length === 0 && errors.length === 0 ? "passed" : "failed",
      schema: "evomind.page_deeplinks.cdp.v1",
      created_at: new Date().toISOString(),
      dashboard_url: baseUrl,
      browser,
      browser_protocol_version: version["Protocol-Version"] ?? null,
      page_count: Object.keys(pages).length,
      artifact_count: artifacts.length,
      artifacts,
      page_results: pageResults,
      missing,
      runtime_error_count: errors.length,
      runtime_errors: errors.slice(0, 10)
    };
    await writeFile(indexPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    console.log(JSON.stringify({
      status: report.status,
      dashboard_url: report.dashboard_url,
      browser: report.browser,
      page_count: report.page_count,
      artifact_count: report.artifact_count,
      missing: report.missing,
      runtime_error_count: report.runtime_error_count,
      out_dir: path.relative(root, outDir)
    }, null, 2));
    return report.status === "passed" ? 0 : 1;
  } finally {
    await stopBrowser(browserProcess, client);
    await rm(profileDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }).catch(() => undefined);
  }
}

try {
  process.exitCode = await run();
} catch (error) {
  console.error(JSON.stringify({
    status: "failed",
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : null
  }, null, 2));
  process.exitCode = 1;
}
