import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const outJson = join(root, "workspace", "workstation_stateful_interactions_20260701.json");
const outMd = join(root, "reports", "WORKSTATION_STATEFUL_INTERACTIONS_20260701.md");

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8088";
const writeReport = process.argv.includes("--write-report");
const port = Number(process.env.WORKSTATION_STATEFUL_CDP_PORT ?? String(9623 + (process.pid % 1000)));

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
].filter(Boolean);

const checks = [
  {
    name: "settings_language_region_section",
    page: "settings",
    selector: "[data-ui-action='open_settings_section_language_region']",
    activeSelector: "[data-ui-action='open_settings_section_language_region'][data-active-section='true']",
    mustContain: ["C. Language & Region", "当前 section", "Language & Region"]
  },
  {
    name: "settings_gpu_hpc_section",
    page: "settings",
    selector: "[data-ui-action='open_settings_section_gpu_hpc']",
    activeSelector: "[data-ui-action='open_settings_section_gpu_hpc'][data-active-section='true']",
    mustContain: ["GPU / HPC Gateway Settings", "GPU / HPC", "需要人工 Gate"]
  },
  {
    name: "settings_design_governance_section",
    page: "settings",
    selector: "[data-ui-action='open_settings_section_design_governance']",
    activeSelector: "[data-ui-action='open_settings_section_design_governance'][data-active-section='true']",
    mustContain: ["Design Governance", "页面覆盖", "组件一致"]
  },
  {
    name: "runtime_llm_cache_tab",
    page: "runtime",
    selector: "[data-ui-action='runtime_trace_tab_llm_cache']",
    activeSelector: "[data-ui-action='runtime_trace_tab_llm_cache'][data-active-trace-tab='true']",
    mustContain: ["prompt_cache_lookup", "bounded_context_hash", "deepseek_batch_cache"]
  },
  {
    name: "runtime_tool_calls_tab",
    page: "runtime",
    beforeSelector: "[data-ui-action='runtime_trace_tab_llm_cache']",
    selector: "[data-ui-action='runtime_trace_tab_tool_calls']",
    activeSelector: "[data-ui-action='runtime_trace_tab_tool_calls'][data-active-trace-tab='true']",
    mustContain: ["read_artifact", "generate_code", "create_hpc_manifest"]
  },
  {
    name: "evidence_validation_detail_tab",
    page: "evidence",
    selector: "[data-ui-action='artifact_detail_tab_1']",
    activeSelector: "[data-ui-action='artifact_detail_tab_1'][data-active-artifact-tab='true']",
    mustContain: ["Validation Detail / 验证详情", "required_artifacts_present", "validation_contract_1001"]
  },
  {
    name: "evidence_dependency_tab",
    page: "evidence",
    selector: "[data-ui-action='artifact_detail_tab_2']",
    activeSelector: "[data-ui-action='artifact_detail_tab_2'][data-active-artifact-tab='true']",
    mustContain: ["Dependency Graph / 依赖关系", "Task -> Run -> Experiment", "gate_regression_v1"]
  },
  {
    name: "code_agent_features_file",
    page: "code",
    selector: "[data-ui-action='open_code_file_features_py']",
    activeSelector: "[data-ui-action='open_code_file_features_py'][data-selected-file='true']",
    mustContain: ["features.py", "build_feature_matrix"]
  },
  {
    name: "tasks_selected_task_row",
    page: "tasks",
    selector: "[data-ui-action='tasks_select_titanic']",
    activeSelector: "[data-ui-action='tasks_select_titanic'][data-selected-task='true']",
    mustContain: ["titanic", "Create Workstation Run"]
  }
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

async function navigate(client, page) {
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  for (let attempt = 0; attempt < 80; attempt++) {
    const ready = await evalValue(client, `document.readyState === 'complete' && document.querySelector('[data-ui-component="workstation-page"]')?.getAttribute('data-ui-page') === ${JSON.stringify(page)}`);
    if (ready) return;
    await sleep(150);
  }
  throw new Error(`Page ${page} did not become ready.`);
}

async function clickSelector(client, selector) {
  return evalValue(client, `(() => {
    const target = document.querySelector(${JSON.stringify(selector)});
    if (!target) return { ok: false, reason: "selector_not_found" };
    target.scrollIntoView({ block: "center", inline: "center" });
    target.click();
    return { ok: true, label: (target.textContent || target.getAttribute("aria-label") || "").trim().slice(0, 80) };
  })()`);
}

async function waitForSelector(client, selector, timeoutMs = 5000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const found = await evalValue(client, `Boolean(document.querySelector(${JSON.stringify(selector)}))`);
    if (found) return true;
    await sleep(150);
  }
  return false;
}

async function runCheck(client, check) {
  await navigate(client, check.page);
  await sleep(250);
  if (check.beforeSelector) {
    await waitForSelector(client, check.beforeSelector);
    await clickSelector(client, check.beforeSelector);
    await sleep(250);
  }
  await waitForSelector(client, check.selector);
  const clickResult = await clickSelector(client, check.selector);
  await sleep(450);
  const state = await evalValue(client, `(() => {
    const body = document.body.innerText;
    return {
      clicked: ${JSON.stringify(Boolean(clickResult.ok))},
      clickReason: ${JSON.stringify(clickResult.reason ?? null)},
      activeMarker: Boolean(document.querySelector(${JSON.stringify(check.activeSelector)})),
      textMatches: ${JSON.stringify(check.mustContain)}.map((text) => ({ text, ok: body.includes(text) })),
      hasRuntimeError: /Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(body)
    };
  })()`);
  return {
    ...check,
    clicked: state.clicked,
    click_reason: state.clickReason,
    active_marker: state.activeMarker,
    text_matches: state.textMatches,
    has_runtime_error: state.hasRuntimeError,
    ok: Boolean(state.clicked) && state.activeMarker && state.textMatches.every((item) => item.ok) && !state.hasRuntimeError
  };
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  if (!chrome) {
    return {
      schema: "academic_research_os.workstation_stateful_interactions.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: "blocked",
      blocker: "browser_unavailable",
      results: [],
      failed_checks: checks.map((item) => item.name),
      claim_boundary: "No Chromium-compatible browser was found, so stateful interaction smoke could not run."
    };
  }

  const userDataDir = join(root, "workspace", `.chrome-stateful-smoke-${Date.now()}`);
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

    const results = [];
    for (const check of checks) results.push(await runCheck(client, check));
    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });
    const failedChecks = results.filter((item) => !item.ok).map((item) => item.name);
    return {
      schema: "academic_research_os.workstation_stateful_interactions.v1",
      created_at: createdAt,
      base_url: baseUrl,
      status: failedChecks.length === 0 && runtimeErrors.length === 0 ? "passed" : "failed",
      blocker: null,
      chrome,
      results,
      failed_checks: failedChecks,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      cleanup_warning: cleanupWarning,
      claim_boundary: "This smoke verifies visible state changes after safe UI clicks. It does not start training, GPU jobs, Kaggle submissions, or backend mutations."
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdown(report) {
  const lines = [
    "# 工作站状态型交互 Smoke",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 工作站地址：\`${report.base_url}\``,
    `- 状态：\`${report.status}\``,
    `- 失败检查：\`${report.failed_checks?.join(", ") || "none"}\``,
    `- 运行时错误数：\`${report.runtime_error_count ?? 0}\``,
    "",
    "| check | page | clicked | active marker | text ok | ok |",
    "| --- | --- | --- | --- | --- | --- |",
  ];
  for (const item of report.results ?? []) {
    const textOk = item.text_matches?.every((match) => match.ok) ?? false;
    lines.push(`| \`${item.name}\` | \`${item.page}\` | \`${item.clicked}\` | \`${item.active_marker}\` | \`${textOk}\` | \`${item.ok}\` |`);
  }
  lines.push("", "## 声明边界", "", report.claim_boundary, "");
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
  failed_checks: report.failed_checks,
  runtime_error_count: report.runtime_error_count ?? 0,
  json: writeReport ? "workspace/workstation_stateful_interactions_20260701.json" : null,
  md: writeReport ? "reports/WORKSTATION_STATEFUL_INTERACTIONS_20260701.md" : null
}, null, 2));

process.exit(report.status === "passed" ? 0 : 1);
