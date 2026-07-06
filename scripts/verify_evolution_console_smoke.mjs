// Real-browser smoke for the Evolution Console closed-loop controls.
//
// Drives a headless Chromium via CDP to:
//   1. navigate to ?page=evolution and wait for the SPA shell,
//   2. confirm the new closed-loop controls render (task picker, engine/runner
//      selects, iterations, MCGS toggle, plan/approve buttons),
//   3. confirm the /api/evolution/configs route feeds the picker with real configs,
//   4. run a GPU-free "Plan cycle (no training)" and assert it stops at
//      awaiting_approval (no training launched, official submit stays disabled),
//   5. capture a screenshot as evidence.
//
// It never clicks the armed "Confirm: run real training" button, so it cannot
// start GPU jobs, local training, or Kaggle submission.
//
// Usage: node scripts/verify_evolution_console_smoke.mjs --base-url http://127.0.0.1:8090 [--write-report]
import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
const outJson = join(root, "workspace", `evolution_console_smoke_${stamp}.json`);
const outMd = join(root, "reports", `EVOLUTION_CONSOLE_SMOKE_${stamp}.md`);
const outPng = join(root, "workspace", `evolution_console_smoke_${stamp}.png`);

const baseUrl = process.argv.includes("--base-url")
  ? process.argv[process.argv.indexOf("--base-url") + 1]
  : "http://127.0.0.1:8090";
const writeReport = process.argv.includes("--write-report");
const port = Number(process.env.WORKSTATION_CDP_PORT ?? String(9323 + (process.pid % 1000)));

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
].filter(Boolean);

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
function findChrome() {
  return chromeCandidates.find((c) => c && existsSync(c)) ?? null;
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
async function stopBrowser(proc) {
  if (!proc || proc.killed) return;
  proc.kill();
  await new Promise((r) => {
    const t = setTimeout(r, 2500);
    proc.once("exit", () => {
      clearTimeout(t);
      r();
    });
  });
}
async function cleanupUserDataDir(dir) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      await rm(dir, { recursive: true, force: true });
      return null;
    } catch (error) {
      if (error?.code !== "EBUSY" && error?.code !== "EPERM") throw error;
      await sleep(400 + attempt * 300);
    }
  }
  return `cleanup_deferred:${dir}`;
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
    await new Promise((res, rej) => {
      const timer = setTimeout(() => rej(new Error("CDP websocket connection timeout")), 10000);
      this.socket.addEventListener("open", () => {
        clearTimeout(timer);
        res();
      }, { once: true });
      this.socket.addEventListener("error", (e) => {
        clearTimeout(timer);
        rej(new Error(`CDP websocket error: ${e.message ?? "unknown"}`));
      }, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const payload = JSON.parse(String(event.data));
      if (payload.id && this.pending.has(payload.id)) {
        const { resolve: res, reject: rej } = this.pending.get(payload.id);
        this.pending.delete(payload.id);
        if (payload.error) rej(new Error(payload.error.message ?? JSON.stringify(payload.error)));
        else res(payload.result ?? {});
        return;
      }
      this.events.push(payload);
    });
  }
  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((res, rej) => {
      this.pending.set(id, { resolve: res, reject: rej });
      this.socket.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          rej(new Error(`CDP command timeout: ${method}`));
        }
      }, 15000);
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
  const result = await client.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text ?? "Runtime evaluation failed.");
  return result.result?.value;
}

async function waitForEvolutionConsole(client) {
  for (let attempt = 0; attempt < 100; attempt++) {
    const ready = await evalValue(client, `(() => {
      const shell = document.querySelector('[data-ui-component="workstation-page"]');
      if (!shell || shell.getAttribute('data-ui-page') !== 'evolution') return false;
      return document.body.innerText.includes('Closed loop');
    })()`);
    if (ready === true) return;
    await sleep(200);
  }
  throw new Error("Evolution console did not render the closed-loop panel.");
}

async function run() {
  const chrome = findChrome();
  const createdAt = new Date().toISOString();
  const base = {
    schema: "academic_research_os.evolution_console_smoke.v1",
    created_at: createdAt,
    base_url: baseUrl,
    chrome
  };
  if (!chrome) {
    return {
      ...base,
      status: "blocked",
      blocker: "browser_unavailable",
      claim_boundary: "No Chromium-compatible browser found, so the console smoke could not run."
    };
  }

  // Pre-flight: the configs API must serve real task configs the picker reads.
  let configsApi = null;
  try {
    configsApi = await fetchJson(`${baseUrl}/api/evolution/configs`, 8000);
  } catch (error) {
    return { ...base, status: "failed", blocker: "configs_api_unreachable", error: String(error?.message ?? error) };
  }

  const userDataDir = join(root, "workspace", `.chrome-evolution-smoke-${Date.now()}`);
  await mkdir(userDataDir, { recursive: true });
  const chromeProcess = spawn(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--window-size=1440,2200",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    `${baseUrl}/?page=evolution`
  ], { stdio: "ignore" });

  let client;
  let cleanupWarning = null;
  try {
    const version = await waitForChrome(port);
    const tabs = await fetchJson(`http://127.0.0.1:${port}/json`);
    const tab = tabs.find((t) => t.type === "page") ?? tabs[0];
    client = new CdpClient(tab.webSocketDebuggerUrl ?? version.webSocketDebuggerUrl);
    await client.connect();
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Log.enable");

    await waitForEvolutionConsole(client);

    // The task picker is populated by an async /api/evolution/configs fetch that
    // resolves shortly after the panel first paints. Wait for it before asserting,
    // otherwise we race the fetch and see an empty option set.
    for (let attempt = 0; attempt < 60; attempt++) {
      const populated = await evalValue(client, `(() => {
        const selects = Array.from(document.querySelectorAll('select'));
        return selects.some((s) => Array.from(s.options).some((o) => ['nyc_taxi','aerial_cactus','spooky_author'].includes(o.value)));
      })()`);
      if (populated === true) break;
      await sleep(300);
    }

    // Inspect the rendered controls. The picker is the first <select> whose option
    // set matches the API config task ids; assert the full control surface exists.
    const controls = await evalValue(client, `(() => {
      const selects = Array.from(document.querySelectorAll('select'));
      const options = selects.map((s) => Array.from(s.options).map((o) => o.value));
      const text = document.body.innerText;
      const findBtn = (label) => Array.from(document.querySelectorAll('button')).some((b) => (b.textContent || '').includes(label));
      const taskSelect = selects.find((s) => Array.from(s.options).some((o) => ['nyc_taxi','aerial_cactus','spooky_author'].includes(o.value)));
      return {
        selectCount: selects.length,
        taskOptions: taskSelect ? Array.from(taskSelect.options).map((o) => o.value) : [],
        engineOptions: options.find((o) => o.includes('research_os') && o.includes('legacy')) ?? [],
        runnerOptions: options.find((o) => o.includes('gpu') && o.includes('local')) ?? [],
        hasIterations: !!document.querySelector('input[type="number"]'),
        hasMcgsToggle: text.includes('MCGS search') && !!document.querySelector('input[type="checkbox"]'),
        hasPlanButton: findBtn('Plan cycle'),
        hasApproveButton: findBtn('Approve & run'),
        submitDisabledBadge: text.includes('Kaggle submit disabled')
      };
    })()`);

    // Run the GPU-free plan cycle by clicking "Plan cycle (no training)".
    const clicked = await evalValue(client, `(() => {
      const btn = Array.from(document.querySelectorAll('button')).find((b) => (b.textContent || '').includes('Plan cycle'));
      if (!btn) return { clicked: false };
      btn.click();
      return { clicked: true };
    })()`);

    // Wait for the cycle result banner (awaiting_approval stage) to appear.
    let planResult = { appeared: false, text: "" };
    for (let attempt = 0; attempt < 60; attempt++) {
      const info = await evalValue(client, `(() => {
        const text = document.body.innerText;
        const appeared = text.includes('awaiting_approval') || text.includes('Plan ready') || text.includes('plan only');
        return { appeared, hasError: /Cycle failed|Application error|Internal Server Error/i.test(text) };
      })()`);
      if (info.appeared) {
        planResult = { appeared: true };
        break;
      }
      if (info.hasError) {
        planResult = { appeared: false, error: "cycle error banner shown" };
        break;
      }
      await sleep(500);
    }

    // Confirm state after plan: no training was launched, submit stays disabled.
    const postState = await evalValue(client, `(() => {
      const text = document.body.innerText;
      return {
        awaitingApproval: text.includes('awaiting_approval') || text.includes('plan only'),
        submitDisabled: !text.includes('Official submit ON'),
        hasRuntimeError: /Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error/i.test(text)
      };
    })()`);

    // Screenshot evidence.
    let screenshotSaved = null;
    try {
      const shot = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
      if (shot?.data) {
        await mkdir(dirname(outPng), { recursive: true });
        await writeFile(outPng, Buffer.from(shot.data, "base64"));
        screenshotSaved = outPng;
      }
    } catch {
      screenshotSaved = null;
    }

    const runtimeErrors = client.events.filter((event) => {
      const method = event.method ?? "";
      const text = JSON.stringify(event.params ?? {});
      const favicon404 = /favicon\.ico/.test(text) && /404|Not Found/.test(text);
      return !favicon404 && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error/i.test(text));
    });

    const checks = {
      configs_api_ok: Boolean(configsApi?.ok) && (configsApi?.count ?? 0) > 0,
      picker_has_real_tasks: controls.taskOptions.length >= 3,
      engine_options_ok: controls.engineOptions.includes("research_os") && controls.engineOptions.includes("legacy"),
      runner_options_ok: controls.runnerOptions.includes("gpu") && controls.runnerOptions.includes("local"),
      iterations_input_ok: controls.hasIterations,
      mcgs_toggle_ok: controls.hasMcgsToggle,
      plan_button_ok: controls.hasPlanButton,
      approve_button_ok: controls.hasApproveButton,
      submit_disabled_badge_ok: controls.submitDisabledBadge,
      plan_cycle_clicked: Boolean(clicked.clicked),
      plan_awaiting_approval: planResult.appeared === true && postState.awaitingApproval,
      submit_stayed_disabled: postState.submitDisabled,
      no_runtime_errors: runtimeErrors.length === 0 && !postState.hasRuntimeError
    };
    const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([k]) => k);

    return {
      ...base,
      status: failed.length === 0 ? "passed" : "failed",
      blocker: null,
      configs_api: { ok: configsApi?.ok, count: configsApi?.count, task_ids: (configsApi?.configs ?? []).map((c) => c.task_id) },
      controls,
      plan_cycle: { clicked: Boolean(clicked.clicked), ...planResult, post_state: postState },
      checks,
      failed_checks: failed,
      runtime_error_count: runtimeErrors.length,
      runtime_errors: runtimeErrors.slice(0, 10),
      screenshot: screenshotSaved,
      cleanup_warning: cleanupWarning,
      claim_boundary:
        "Real headless Chromium smoke. Verifies the evolution console renders the closed-loop controls, the /api/evolution/configs route feeds real task configs, and the GPU-free 'Plan cycle' stops at awaiting_approval with official submit disabled. It never clicks the armed 'Confirm: run real training' button, so it starts no GPU/local training and no Kaggle submission."
    };
  } finally {
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupUserDataDir(userDataDir);
  }
}

function toMarkdown(report) {
  const lines = [
    "# Evolution Console Closed-Loop Smoke",
    "",
    `- created_at: \`${report.created_at}\``,
    `- base_url: \`${report.base_url}\``,
    `- status: \`${report.status}\``,
    `- browser: \`${report.chrome ?? "not_found"}\``,
    `- configs served: \`${report.configs_api?.count ?? 0}\` (${(report.configs_api?.task_ids ?? []).join(", ") || "none"})`,
    `- failed checks: \`${report.failed_checks?.join(", ") || "none"}\``,
    `- runtime errors: \`${report.runtime_error_count ?? 0}\``,
    `- screenshot: \`${report.screenshot ?? "none"}\``,
    "",
    "## Checks",
    "",
    "| check | ok |",
    "| --- | --- |"
  ];
  for (const [k, v] of Object.entries(report.checks ?? {})) lines.push(`| \`${k}\` | \`${v}\` |`);
  lines.push("", "## Claim Boundary", "", report.claim_boundary ?? "", "");
  return lines.join("\n");
}

const report = await run();
if (writeReport) {
  await mkdir(dirname(outJson), { recursive: true });
  await mkdir(dirname(outMd), { recursive: true });
  await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(outMd, `﻿${toMarkdown(report)}`, "utf8");
}
console.log(JSON.stringify({
  status: report.status,
  blocker: report.blocker ?? null,
  failed_checks: report.failed_checks ?? [],
  runtime_error_count: report.runtime_error_count ?? 0,
  screenshot: report.screenshot ?? null,
  json: writeReport ? outJson : null
}, null, 2));
process.exit(report.status === "passed" ? 0 : 1);
