import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { mkdir, rm, writeFile } from "node:fs/promises";
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
  : join(root, "artifacts", "theme-locale-acceptance"));
const outJson = join(outDir, "theme-locale-acceptance.json");
const outMd = join(root, "reports", "WORKSTATION_THEME_LOCALE_ACCEPTANCE_V4.md");
const port = Number(process.env.WORKSTATION_THEME_LOCALE_CDP_PORT ?? String(11623 + (process.pid % 1000)));

const chromeCandidates = [
  process.env.WORKSTATION_BROWSER,
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);

const traceBootstrap = `(() => {
  const trace = [];
  Object.defineProperty(window, "__evomindThemeAcceptanceTrace", {
    value: trace,
    configurable: false,
    enumerable: false,
    writable: false,
  });
  const sample = (event) => {
    const root = document.documentElement;
    const body = document.body;
    let rootStyle = null;
    let bodyStyle = null;
    try {
      rootStyle = root ? getComputedStyle(root) : null;
      bodyStyle = body ? getComputedStyle(body) : null;
    } catch (_) {}
    trace.push({
      event,
      at: performance.now(),
      readyState: document.readyState,
      theme: root?.dataset?.theme ?? null,
      themeMode: root?.dataset?.themeMode ?? null,
      classDark: root?.classList?.contains("dark") ?? false,
      colorScheme: rootStyle?.colorScheme ?? root?.style?.colorScheme ?? null,
      bodyBackground: bodyStyle?.backgroundColor ?? null,
    });
  };
  sample("bootstrap");
  const observeRoot = () => {
    if (!document.documentElement) return false;
    new MutationObserver(() => sample("root-attribute-mutation"))
      .observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme", "data-theme-mode", "style"] });
    return true;
  };
  if (!observeRoot()) {
    const rootObserver = new MutationObserver(() => {
      if (observeRoot()) rootObserver.disconnect();
    });
    rootObserver.observe(document, { childList: true, subtree: true });
  }
  document.addEventListener("readystatechange", () => sample("ready:" + document.readyState));
  document.addEventListener("DOMContentLoaded", () => sample("dom-content-loaded"), { once: true });
  window.addEventListener("load", () => sample("load"), { once: true });
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) sample("paint:" + entry.name);
    }).observe({ type: "paint", buffered: true });
  } catch (_) {}
  requestAnimationFrame(() => {
    sample("raf:1");
    requestAnimationFrame(() => sample("raf:2"));
  });
})();`;

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function findChrome() {
  return chromeCandidates.find((candidate) => candidate && existsSync(candidate)) ?? null;
}

async function fetchJson(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${JSON.stringify(payload)}`);
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

async function getSettings() {
  return fetchJson(`${baseUrl}/api/settings`);
}

async function patchSettings(settings) {
  return fetchJson(`${baseUrl}/api/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings }),
  });
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
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      await rm(path, { recursive: true, force: true });
      return null;
    } catch (error) {
      if (error?.code !== "EBUSY" && error?.code !== "EPERM") throw error;
      await sleep(300 + attempt * 250);
    }
  }
  return `cleanup_deferred:${path}`;
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
      return await fetchJson(`http://127.0.0.1:${portNumber}/json/version`, {}, 1500);
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

async function setSystemScheme(client, scheme) {
  await client.send("Emulation.setEmulatedMedia", {
    media: "",
    features: [{ name: "prefers-color-scheme", value: scheme }],
  });
}

async function setLocalTheme(client, mode) {
  const origin = new URL(baseUrl).origin;
  try {
    await client.send("DOMStorage.setDOMStorageItem", {
      storageId: { securityOrigin: origin, isLocalStorage: true },
      key: "evomind.theme",
      value: mode,
    });
  } catch (error) {
    if (!/Frame not found for the given storage id/i.test(String(error))) throw error;
    await client.send("Page.navigate", { url: `${baseUrl}/?theme_locale_storage_bootstrap=1` });
    await waitFor(client, `document.readyState === "complete"`, "storage bootstrap page", 15000);
    await evalValue(client, `localStorage.setItem("evomind.theme", ${JSON.stringify(mode)})`);
  }
}

async function navigateSettings(client, marker) {
  const url = `${baseUrl}/?page=settings&theme_locale_acceptance=${encodeURIComponent(marker)}&t=${Date.now()}`;
  await client.send("Page.navigate", { url });
  await waitFor(client, `document.readyState === "complete"
    && document.querySelector('[data-ui-component="workstation-page"]')?.getAttribute('data-ui-page') === "settings"
    && document.querySelector('[data-ui-action="save_settings_changes"]')?.disabled === false
    && document.querySelector('[data-ui-action="settings_theme_light"]')?.disabled === false
    && document.querySelector('[data-ui-action="settings_language_en_us"]')?.disabled === false`, `settings page ${marker}`, 15000);
  await sleep(150);
}

async function waitFor(client, expression, label, timeoutMs = 10000) {
  const started = Date.now();
  let lastValue = null;
  while (Date.now() - started < timeoutMs) {
    lastValue = await evalValue(client, `Boolean(${expression})`);
    if (lastValue) return true;
    await sleep(120);
  }
  throw new Error(`Timed out waiting for ${label}; last=${JSON.stringify(lastValue)}`);
}

async function clickAction(client, action) {
  const result = await evalValue(client, `(() => {
    const target = document.querySelector('[data-ui-action=${JSON.stringify(action)}]');
    if (!target) return { ok: false, reason: "missing" };
    target.scrollIntoView({ block: "center", inline: "center" });
    target.click();
    return { ok: true, disabled: Boolean(target.disabled), label: (target.textContent || target.getAttribute("aria-label") || "").replace(/\\s+/g, " ").trim() };
  })()`);
  if (!result?.ok) throw new Error(`Action ${action} was not found.`);
  return result;
}

async function snapshot(client, label) {
  return evalValue(client, `(() => {
    const root = document.documentElement;
    const rootStyle = getComputedStyle(root);
    const bodyStyle = getComputedStyle(document.body);
    const button = (action) => document.querySelector('[data-ui-action="' + action + '"]');
    const text = document.body?.innerText ?? "";
    return {
      label: ${JSON.stringify(label)},
      href: location.href,
      readyState: document.readyState,
      theme: root.dataset.theme ?? null,
      themeMode: root.dataset.themeMode ?? null,
      classDark: root.classList.contains("dark"),
      inlineColorScheme: root.style.colorScheme || null,
      computedColorScheme: rootStyle.colorScheme,
      bodyBackground: bodyStyle.backgroundColor,
      canvasToken: rootStyle.getPropertyValue("--canvas").trim(),
      localStorageTheme: localStorage.getItem("evomind.theme"),
      htmlLang: root.lang,
      prefersLight: matchMedia("(prefers-color-scheme: light)").matches,
      selectedTheme: ["light", "dark", "system"].find((mode) => button("settings_theme_" + mode)?.getAttribute("aria-pressed") === "true") ?? null,
      selectedLocale: button("settings_language_en_us")?.getAttribute("aria-pressed") === "true" ? "en-US"
        : button("settings_language_zh_cn")?.getAttribute("aria-pressed") === "true" ? "zh-CN" : null,
      hasEnglishSettingsCopy: text.includes("Settings") && text.includes("General") && text.includes("Appearance"),
      hasChineseSettingsCopy: text.includes("设置") && text.includes("通用") && text.includes("外观"),
      feedback: document.querySelector('[role="status"]')?.textContent?.trim() ?? null,
      statusMessages: Array.from(document.querySelectorAll('[role="status"]')).map((node) => node.textContent?.trim() ?? "").filter(Boolean),
      saveDisabled: Boolean(button("save_settings_changes")?.disabled),
      trace: (window.__evomindThemeAcceptanceTrace ?? []).slice(-40),
      paintEntries: performance.getEntriesByType("paint").map((entry) => ({ name: entry.name, startTime: entry.startTime })),
    };
  })()`);
}

async function screenshot(client, filename) {
  const capture = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  await writeFile(join(outDir, filename), Buffer.from(capture.data, "base64"));
  return filename;
}

function settingsPair(payload) {
  return {
    theme: payload.settings?.general?.theme ?? null,
    locale: payload.settings?.language?.ui_language ?? null,
  };
}

function isDarkBackground(value) {
  const match = String(value ?? "").match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
  if (!match) return false;
  const [, red, green, blue] = match.map(Number);
  return red < 40 && green < 50 && blue < 50;
}

function tracePaints(snapshotValue) {
  return (snapshotValue.trace ?? []).filter((item) => String(item.event).startsWith("paint:"));
}

function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function toMarkdown(report) {
  const lines = [
    "# EvoMind Theme / Locale 端到端验收",
    "",
    `- 生成时间：\`${report.created_at}\``,
    `- 生产地址：\`${report.base_url}\``,
    `- 最终状态：\`${report.status}\``,
    `- 原始设置：\`${report.initial_settings.theme} / ${report.initial_settings.locale}\``,
    `- 恢复后设置：\`${report.restored_settings?.theme ?? "unknown"} / ${report.restored_settings?.locale ?? "unknown"}\``,
    `- 运行时错误：\`${report.runtime_error_count}\``,
    "",
    "## 断言矩阵",
    "",
    "| 断言 | 状态 | 证据 |",
    "| --- | --- | --- |",
  ];
  for (const check of report.checks) {
    const evidence = String(check.evidence ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
    lines.push(`| \`${check.name}\` | \`${check.ok ? "passed" : "failed"}\` | ${evidence} |`);
  }
  lines.push(
    "",
    "## 视觉证据",
    "",
    ...report.screenshots.map((file) => `- \`${file}\``),
    "",
    "## 完整证据",
    "",
    `- JSON：\`${outJson}\``,
    `- 初始 HTML 根节点：\`${report.html_evidence.root_tag}\``,
    `- 主题预水合脚本位于 \`<body>\` 之前：\`${report.html_evidence.init_before_body}\``,
    `- 恢复精确匹配：\`${report.restore_exact_match}\``,
    "",
    "## 边界",
    "",
    "验收只操作 Settings 中的主题与语言偏好；没有触发训练、GPU 作业、Kaggle 提交或其他业务动作。验收结束后已恢复原始设置。",
    "",
    `final result: ${report.status}`,
    "",
  );
  return lines.join("\n");
}

async function run() {
  const createdAt = new Date().toISOString();
  const initialPayload = await getSettings();
  const initialGeneral = structuredClone(initialPayload.settings?.general ?? {});
  const initialLanguage = structuredClone(initialPayload.settings?.language ?? {});
  const initialPair = settingsPair(initialPayload);
  if (!["dark", "light", "system"].includes(initialPair.theme)) throw new Error(`Unexpected initial theme: ${initialPair.theme}`);
  if (!["zh-CN", "en-US"].includes(initialPair.locale)) throw new Error(`Unexpected initial locale: ${initialPair.locale}`);

  const html = await (await fetch(`${baseUrl}/?page=settings&theme_locale_html_probe=1`)).text();
  const rootTag = html.match(/<html[^>]*>/i)?.[0] ?? "missing";
  const initIndex = html.indexOf("evomind.theme");
  const bodyIndex = html.indexOf("<body");
  const htmlEvidence = {
    root_tag: rootTag,
    has_static_dark_fallback: /data-theme="dark"/i.test(rootTag) && /class="dark"/i.test(rootTag),
    init_script_index: initIndex,
    body_index: bodyIndex,
    init_before_body: initIndex >= 0 && bodyIndex >= 0 && initIndex < bodyIndex,
  };

  const chrome = findChrome();
  if (!chrome) throw new Error("No Chromium-compatible browser was found.");
  await mkdir(outDir, { recursive: true });
  await mkdir(dirname(outMd), { recursive: true });
  const userDataDir = join(root, "workspace", `.chrome-theme-locale-${Date.now()}`);
  await mkdir(userDataDir, { recursive: true });
  const chromeProcess = spawn(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--window-size=1440,900",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    "about:blank",
  ], { stdio: "ignore" });

  const checks = [];
  const checkpoints = [];
  const screenshots = [];
  let client = null;
  let cleanupWarning = null;
  let restoredPayload = null;
  let restoreExactMatch = false;
  let fatalError = null;

  const addCheck = (name, ok, evidence, details = undefined) => {
    checks.push({ name, ok: Boolean(ok), evidence, ...(details === undefined ? {} : { details }) });
  };

  try {
    const version = await waitForChrome(port);
    const tabs = await fetchJson(`http://127.0.0.1:${port}/json`);
    const tab = tabs.find((item) => item.type === "page") ?? tabs[0];
    client = new CdpClient(tab.webSocketDebuggerUrl ?? version.webSocketDebuggerUrl);
    await client.connect();
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Log.enable");
    await client.send("DOMStorage.enable");
    await client.send("Network.enable");
    await client.send("Page.addScriptToEvaluateOnNewDocument", { source: traceBootstrap });
    await client.send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false,
      screenWidth: 1440,
      screenHeight: 900,
    });

    await setSystemScheme(client, "dark");
    await setLocalTheme(client, initialPair.theme);
    await navigateSettings(client, "cold-initial");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === ${JSON.stringify(initialPair.theme)}
       && document.documentElement.lang === ${JSON.stringify(initialPair.locale)}
       && document.querySelector('[data-ui-action="settings_theme_${initialPair.theme}"]')?.getAttribute('aria-pressed') === "true"
       && document.querySelector('[data-ui-action="settings_language_${initialPair.locale === "en-US" ? "en_us" : "zh_cn"}"]')?.getAttribute('aria-pressed') === "true"`,
      "initial persisted theme and locale");
    const initialState = await snapshot(client, "initial-cold-load");
    checkpoints.push(initialState);
    screenshots.push(await screenshot(client, "01-initial-cold-load.png"));
    const expectedInitialResolved = initialPair.theme === "system" ? "dark" : initialPair.theme;
    const initialPaints = tracePaints(initialState);
    const initialPaintsCorrect = initialPaints.length > 0 && initialPaints.every((item) =>
      item.theme === expectedInitialResolved
      && item.themeMode === initialPair.theme
      && item.colorScheme === expectedInitialResolved
      && (expectedInitialResolved !== "dark" || isDarkBackground(item.bodyBackground)));
    addCheck("initial_html_dark_fallback", htmlEvidence.has_static_dark_fallback && htmlEvidence.init_before_body,
      `root=${rootTag}; init_before_body=${htmlEvidence.init_before_body}`);
    addCheck("initial_no_theme_flash", initialPaintsCorrect,
      `expected=${initialPair.theme}/${expectedInitialResolved}; paints=${JSON.stringify(initialPaints)}`,
      { paintEntries: initialState.paintEntries, trace: initialState.trace });
    addCheck("initial_theme_locale_contract",
      initialState.themeMode === initialPair.theme
      && initialState.theme === expectedInitialResolved
      && initialState.htmlLang === initialPair.locale
      && initialState.localStorageTheme === initialPair.theme
      && initialState.inlineColorScheme === expectedInitialResolved
      && initialState.computedColorScheme === expectedInitialResolved,
      JSON.stringify(initialState));

    await clickAction(client, "settings_theme_light");
    await waitFor(client,
      `document.documentElement.dataset.theme === "light"
       && document.documentElement.dataset.themeMode === "light"
       && document.querySelector('[data-ui-action="settings_theme_light"]')?.getAttribute('aria-pressed') === "true"`,
      "light preview");
    const lightPreview = await snapshot(client, "light-preview-unsaved");
    const serverAfterLightPreview = await getSettings();
    checkpoints.push(lightPreview);
    addCheck("light_instant_preview",
      lightPreview.theme === "light"
      && lightPreview.themeMode === "light"
      && !lightPreview.classDark
      && lightPreview.inlineColorScheme === "light"
      && lightPreview.computedColorScheme === "light"
      && lightPreview.localStorageTheme === initialPair.theme
      && settingsPair(serverAfterLightPreview).theme === initialPair.theme,
      `ui=${JSON.stringify(lightPreview)}; server=${JSON.stringify(settingsPair(serverAfterLightPreview))}`);

    await clickAction(client, "settings_language_en_us");
    await waitFor(client,
      `document.documentElement.lang === "en-US"
       && document.querySelector('[data-ui-action="settings_language_en_us"]')?.getAttribute('aria-pressed') === "true"
       && document.body.innerText.includes("Settings")
       && document.body.innerText.includes("Appearance")`,
      "English preview");
    const englishPreview = await snapshot(client, "english-preview-unsaved");
    const serverAfterEnglishPreview = await getSettings();
    checkpoints.push(englishPreview);
    screenshots.push(await screenshot(client, "02-light-english-preview.png"));
    addCheck("locale_instant_preview_and_html_lang",
      englishPreview.selectedLocale === "en-US"
      && englishPreview.htmlLang === "en-US"
      && englishPreview.hasEnglishSettingsCopy
      && settingsPair(serverAfterEnglishPreview).locale === initialPair.locale,
      `ui=${JSON.stringify(englishPreview)}; server=${JSON.stringify(settingsPair(serverAfterEnglishPreview))}`);

    await clickAction(client, "save_settings_changes");
    for (let attempt = 0; attempt < 80; attempt++) {
      const payload = await getSettings();
      if (settingsPair(payload).theme === "light" && settingsPair(payload).locale === "en-US") break;
      await sleep(150);
    }
    try {
      await waitFor(client,
        `localStorage.getItem("evomind.theme") === "light"
         && !document.querySelector('[data-ui-action="save_settings_changes"]')?.disabled`,
        "light and English save completion");
    } catch (error) {
      const diagnosticState = await snapshot(client, "light-english-save-timeout");
      const diagnosticServer = await getSettings();
      throw new Error(`${error instanceof Error ? error.message : String(error)}; ui=${JSON.stringify(diagnosticState)}; server=${JSON.stringify(settingsPair(diagnosticServer))}`);
    }
    const persistedLightPayload = await getSettings();
    const persistedLightState = await snapshot(client, "light-english-saved");
    checkpoints.push(persistedLightState);
    addCheck("save_light_english_to_server_and_storage",
      settingsPair(persistedLightPayload).theme === "light"
      && settingsPair(persistedLightPayload).locale === "en-US"
      && persistedLightState.localStorageTheme === "light"
      && persistedLightState.statusMessages.some((message) => /saved|已保存/i.test(message)),
      `ui=${JSON.stringify(persistedLightState)}; server=${JSON.stringify(settingsPair(persistedLightPayload))}`);

    await navigateSettings(client, "reload-light-en");
    await waitFor(client,
      `document.documentElement.dataset.theme === "light"
       && document.documentElement.dataset.themeMode === "light"
       && document.documentElement.lang === "en-US"
       && document.querySelector('[data-ui-action="settings_language_en_us"]')?.getAttribute('aria-pressed') === "true"`,
      "light and English refresh persistence");
    const lightReload = await snapshot(client, "light-english-after-reload");
    checkpoints.push(lightReload);
    screenshots.push(await screenshot(client, "03-light-english-after-reload.png"));
    const lightReloadPaints = tracePaints(lightReload);
    addCheck("refresh_persists_light_english",
      lightReload.theme === "light"
      && lightReload.themeMode === "light"
      && lightReload.htmlLang === "en-US"
      && lightReload.localStorageTheme === "light"
      && lightReload.selectedLocale === "en-US",
      JSON.stringify(lightReload));
    addCheck("light_reload_no_opposite_theme_paint",
      lightReloadPaints.length > 0 && lightReloadPaints.every((item) => item.theme === "light" && item.themeMode === "light" && item.colorScheme === "light"),
      JSON.stringify(lightReloadPaints), { trace: lightReload.trace });

    await clickAction(client, "settings_theme_dark");
    await clickAction(client, "settings_language_zh_cn");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "dark"
       && document.documentElement.lang === "zh-CN"`,
      "unsaved dark and Chinese preview before cancel");
    await clickAction(client, "cancel_settings_changes");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "light"
       && document.documentElement.dataset.theme === "light"
       && document.documentElement.lang === "en-US"
       && document.querySelector('[data-ui-action="settings_theme_light"]')?.getAttribute('aria-pressed') === "true"
       && document.querySelector('[data-ui-action="settings_language_en_us"]')?.getAttribute('aria-pressed') === "true"`,
      "cancel restoration to saved light and English");
    const cancelLightEnglish = await snapshot(client, "cancel-restored-light-english");
    const serverAfterCancel = await getSettings();
    checkpoints.push(cancelLightEnglish);
    addCheck("cancel_restores_saved_theme_and_locale",
      cancelLightEnglish.themeMode === "light"
      && cancelLightEnglish.theme === "light"
      && cancelLightEnglish.htmlLang === "en-US"
      && cancelLightEnglish.selectedLocale === "en-US"
      && cancelLightEnglish.localStorageTheme === "light"
      && settingsPair(serverAfterCancel).theme === "light"
      && settingsPair(serverAfterCancel).locale === "en-US",
      `ui=${JSON.stringify(cancelLightEnglish)}; server=${JSON.stringify(settingsPair(serverAfterCancel))}`);

    await setSystemScheme(client, "light");
    await clickAction(client, "settings_theme_system");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "system"
       && document.documentElement.dataset.theme === "light"
       && matchMedia("(prefers-color-scheme: light)").matches`,
      "system light preview");
    const systemLight = await snapshot(client, "system-preview-light-os");
    checkpoints.push(systemLight);
    addCheck("system_preview_resolves_light",
      systemLight.themeMode === "system"
      && systemLight.theme === "light"
      && systemLight.inlineColorScheme === "light"
      && systemLight.computedColorScheme === "light"
      && systemLight.localStorageTheme === "light",
      JSON.stringify(systemLight));

    await setSystemScheme(client, "dark");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "system"
       && document.documentElement.dataset.theme === "dark"
       && document.documentElement.classList.contains("dark")
       && getComputedStyle(document.documentElement).colorScheme === "dark"`,
      "live system switch to dark");
    const systemDark = await snapshot(client, "system-live-dark-os");
    checkpoints.push(systemDark);
    screenshots.push(await screenshot(client, "04-system-live-dark-os.png"));
    addCheck("system_color_change_updates_live",
      systemDark.themeMode === "system"
      && systemDark.theme === "dark"
      && systemDark.classDark
      && systemDark.inlineColorScheme === "dark"
      && systemDark.computedColorScheme === "dark"
      && isDarkBackground(systemDark.bodyBackground),
      JSON.stringify(systemDark));

    await setSystemScheme(client, "light");
    await waitFor(client, `document.documentElement.dataset.theme === "light"`, "live system switch back to light");
    await clickAction(client, "settings_language_zh_cn");
    await clickAction(client, "save_settings_changes");
    for (let attempt = 0; attempt < 80; attempt++) {
      const payload = await getSettings();
      if (settingsPair(payload).theme === "system" && settingsPair(payload).locale === "zh-CN") break;
      await sleep(150);
    }
    await waitFor(client,
      `localStorage.getItem("evomind.theme") === "system"
       && document.documentElement.dataset.themeMode === "system"
       && document.documentElement.lang === "zh-CN"`,
      "system and Chinese save completion");
    const persistedSystemPayload = await getSettings();
    const persistedSystemState = await snapshot(client, "system-chinese-saved");
    checkpoints.push(persistedSystemState);
    addCheck("save_system_chinese_to_server_and_storage",
      settingsPair(persistedSystemPayload).theme === "system"
      && settingsPair(persistedSystemPayload).locale === "zh-CN"
      && persistedSystemState.localStorageTheme === "system"
      && persistedSystemState.themeMode === "system"
      && persistedSystemState.htmlLang === "zh-CN",
      `ui=${JSON.stringify(persistedSystemState)}; server=${JSON.stringify(settingsPair(persistedSystemPayload))}`);

    await navigateSettings(client, "reload-system-zh-light-os");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "system"
       && document.documentElement.dataset.theme === "light"
       && document.documentElement.lang === "zh-CN"
       && localStorage.getItem("evomind.theme") === "system"`,
      "system and Chinese refresh persistence");
    const systemReloadLight = await snapshot(client, "system-chinese-after-reload-light-os");
    checkpoints.push(systemReloadLight);
    const systemReloadPaints = tracePaints(systemReloadLight);
    addCheck("refresh_persists_system_chinese",
      systemReloadLight.themeMode === "system"
      && systemReloadLight.theme === "light"
      && systemReloadLight.htmlLang === "zh-CN"
      && systemReloadLight.selectedLocale === "zh-CN"
      && systemReloadLight.localStorageTheme === "system",
      JSON.stringify(systemReloadLight));
    addCheck("system_reload_prepaint_resolves_os_theme",
      systemReloadPaints.length > 0 && systemReloadPaints.every((item) => item.theme === "light" && item.themeMode === "system" && item.colorScheme === "light"),
      JSON.stringify(systemReloadPaints), { trace: systemReloadLight.trace });

    await setSystemScheme(client, "dark");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "system"
       && document.documentElement.dataset.theme === "dark"`,
      "persisted system live switch after reload");
    const systemReloadDark = await snapshot(client, "system-after-reload-dark-os");
    checkpoints.push(systemReloadDark);
    addCheck("persisted_system_tracks_os_after_reload",
      systemReloadDark.themeMode === "system"
      && systemReloadDark.theme === "dark"
      && systemReloadDark.classDark
      && systemReloadDark.computedColorScheme === "dark",
      JSON.stringify(systemReloadDark));

    await clickAction(client, "settings_theme_light");
    await clickAction(client, "settings_language_en_us");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "light" && document.documentElement.lang === "en-US"`,
      "unsaved light and English before system cancel");
    await clickAction(client, "cancel_settings_changes");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === "system"
       && document.documentElement.dataset.theme === "dark"
       && document.documentElement.lang === "zh-CN"`,
      "cancel restoration to system and Chinese");
    const cancelSystemChinese = await snapshot(client, "cancel-restored-system-chinese");
    checkpoints.push(cancelSystemChinese);
    addCheck("cancel_restores_saved_system_mode",
      cancelSystemChinese.themeMode === "system"
      && cancelSystemChinese.theme === "dark"
      && cancelSystemChinese.htmlLang === "zh-CN"
      && cancelSystemChinese.selectedLocale === "zh-CN"
      && cancelSystemChinese.localStorageTheme === "system",
      JSON.stringify(cancelSystemChinese));

    const initialThemeAction = `settings_theme_${initialPair.theme}`;
    const initialLocaleAction = `settings_language_${initialPair.locale === "en-US" ? "en_us" : "zh_cn"}`;
    await clickAction(client, initialThemeAction);
    await clickAction(client, initialLocaleAction);
    await clickAction(client, "save_settings_changes");
    for (let attempt = 0; attempt < 80; attempt++) {
      const payload = await getSettings();
      if (settingsPair(payload).theme === initialPair.theme && settingsPair(payload).locale === initialPair.locale) break;
      await sleep(150);
    }
    restoredPayload = await getSettings();
    restoreExactMatch = jsonEqual(restoredPayload.settings?.general ?? {}, initialGeneral)
      && jsonEqual(restoredPayload.settings?.language ?? {}, initialLanguage);
    await setLocalTheme(client, initialPair.theme);
    await navigateSettings(client, "restored-original");
    await waitFor(client,
      `document.documentElement.dataset.themeMode === ${JSON.stringify(initialPair.theme)}
       && document.documentElement.lang === ${JSON.stringify(initialPair.locale)}`,
      "restored original settings reload");
    const restoredState = await snapshot(client, "restored-original-final");
    checkpoints.push(restoredState);
    screenshots.push(await screenshot(client, "05-restored-original-final.png"));
    addCheck("original_settings_restored",
      restoreExactMatch
      && restoredState.themeMode === initialPair.theme
      && restoredState.htmlLang === initialPair.locale
      && restoredState.localStorageTheme === initialPair.theme,
      `exact=${restoreExactMatch}; ui=${JSON.stringify(restoredState)}; server=${JSON.stringify(settingsPair(restoredPayload))}`);
  } catch (error) {
    fatalError = error instanceof Error ? error.stack ?? error.message : String(error);
  } finally {
    try {
      const current = await getSettings();
      const exact = jsonEqual(current.settings?.general ?? {}, initialGeneral)
        && jsonEqual(current.settings?.language ?? {}, initialLanguage);
      if (!exact) await patchSettings({ general: initialGeneral, language: initialLanguage });
      restoredPayload = await getSettings();
      restoreExactMatch = jsonEqual(restoredPayload.settings?.general ?? {}, initialGeneral)
        && jsonEqual(restoredPayload.settings?.language ?? {}, initialLanguage);
      if (client) await setLocalTheme(client, initialPair.theme).catch(() => undefined);
    } catch (restoreError) {
      fatalError = `${fatalError ?? ""}\nRESTORE ERROR: ${restoreError instanceof Error ? restoreError.stack ?? restoreError.message : String(restoreError)}`.trim();
    }
    client?.close();
    await stopBrowser(chromeProcess);
    cleanupWarning = await cleanupDirectory(userDataDir);
  }

  const runtimeErrors = (client?.events ?? []).filter((event) => {
    const method = event.method ?? "";
    const value = JSON.stringify(event.params ?? {});
    const ignored = /favicon\.ico/.test(value) && /404|Not Found/.test(value);
    return !ignored && (method.includes("exception") || /ChunkLoadError|Hydration failed|Internal Server Error|Uncaught/i.test(value));
  });
  addCheck("browser_runtime_console_clean", runtimeErrors.length === 0,
    `runtime_error_count=${runtimeErrors.length}`, runtimeErrors.slice(0, 20));
  addCheck("cleanup_restored_exact_original_settings", restoreExactMatch,
    `initial=${JSON.stringify(initialPair)}; restored=${JSON.stringify(settingsPair(restoredPayload ?? { settings: {} }))}`);

  const failedChecks = checks.filter((item) => !item.ok).map((item) => item.name);
  const status = !fatalError && failedChecks.length === 0 && restoreExactMatch ? "passed" : "failed";
  const report = {
    schema: "evomind.workstation.theme_locale_acceptance.v1",
    created_at: createdAt,
    base_url: baseUrl,
    status,
    fatal_error: fatalError,
    initial_settings: initialPair,
    restored_settings: settingsPair(restoredPayload ?? { settings: {} }),
    restore_exact_match: restoreExactMatch,
    html_evidence: htmlEvidence,
    checks,
    failed_checks: failedChecks,
    checkpoints,
    screenshots,
    runtime_error_count: runtimeErrors.length,
    runtime_errors: runtimeErrors.slice(0, 20),
    cleanup_warning: cleanupWarning,
  };
  await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(outMd, `\ufeff${toMarkdown(report)}`, "utf8");
  return report;
}

try {
  const report = await run();
  console.log(JSON.stringify({
    status: report.status,
    failed_checks: report.failed_checks,
    runtime_error_count: report.runtime_error_count,
    restore_exact_match: report.restore_exact_match,
    json: outJson,
    markdown: outMd,
  }, null, 2));
  process.exitCode = report.status === "passed" ? 0 : 1;
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
}
