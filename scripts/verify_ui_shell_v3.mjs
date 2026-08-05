/* V3 UI Shell acceptance: geometry, H1, overflow, drawer, aria, hydration console. */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { once } from "node:events";
import { writeFile, mkdir, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const ROOT = path.resolve(process.cwd());
const baseUrl = process.env.WORKSTATION_URL ?? "http://127.0.0.1:8088";
const OUT_DIR = path.join(ROOT, "reports", "ui_v3_acceptance");
const SHOT_DIR = path.join(ROOT, "reports", "ui_v3_screenshots");

const pages = ["overview","control","tasks","experiments","evolution","workflow","runtime","data","code","literature","report","gpu","evidence","gates","settings"];
const viewports = [
  { name: "1920x1080", width: 1920, height: 1080, mobile: false },
  { name: "1440x1000", width: 1440, height: 1000, mobile: false },
  { name: "1024x800", width: 1024, height: 800, mobile: false },
  { name: "768x1024", width: 768, height: 1024, mobile: false },
  { name: "390x844", width: 390, height: 844, mobile: true },
  { name: "375x812", width: 375, height: 812, mobile: true },
  { name: "351x844", width: 351, height: 844, mobile: true },
];

const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  process.env.CHROME_PATH,
].filter(Boolean);

function findChrome() { return chromeCandidates.find((c) => c && existsSync(c)) ?? null; }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function stopChrome(proc, profileDir) {
  if (proc.exitCode === null) {
    try { proc.kill(); } catch { /* already stopped */ }
    await Promise.race([once(proc, "exit"), sleep(5000)]).catch(() => undefined);
  }
  await rm(profileDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}

async function waitForDevtools(port, proc) {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return await res.json();
    } catch { /* retry */ }
    if (proc.exitCode !== null) break;
    await sleep(250);
  }
  throw new Error("Chrome DevTools endpoint did not become ready.");
}

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 1; this.pending = new Map(); this.consoleErrors = []; }
  async connect() {
    this.sock = new WebSocket(this.ws);
    await new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error("ws timeout")), 10000);
      this.sock.addEventListener("open", () => { clearTimeout(t); res(); }, { once: true });
      this.sock.addEventListener("error", () => { clearTimeout(t); rej(new Error("ws error")); }, { once: true });
    });
    this.sock.addEventListener("message", (ev) => {
      const p = JSON.parse(String(ev.data));
      if (p.id && this.pending.has(p.id)) {
        const { res, rej } = this.pending.get(p.id);
        this.pending.delete(p.id);
        if (p.error) rej(new Error(p.error.message)); else res(p.result ?? {});
        return;
      }
      if (p.method === "Runtime.exceptionThrown") this.consoleErrors.push("exception:" + (p.params?.exceptionDetails?.text ?? ""));
      if (p.method === "Runtime.consoleAPICalled" && p.params?.type === "error") {
        const txt = (p.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" ");
        this.consoleErrors.push("console.error:" + txt);
      }
    });
  }
  send(method, params = {}) {
    const id = this.id++;
    this.sock.send(JSON.stringify({ id, method, params }));
    return new Promise((res, rej) => this.pending.set(id, { res, rej }));
  }
  async eval(expr) {
    const r = await this.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text ?? "eval failed");
    return r.result?.value;
  }
}

const GEOM = `(() => {
  const de = document.documentElement;
  const aside = document.querySelector('aside.shell-sidebar');
  const main = document.querySelector('main');
  const aR = aside ? aside.getBoundingClientRect() : null;
  const mR = main ? main.getBoundingClientRect() : null;
  const h1s = document.querySelectorAll('main h1');
  const isLocallyScrollable = (el) => {
    const scroller = el.closest('[data-ui-local-scroll="true"], .overflow-x-auto, .overflow-auto');
    if (!scroller) return false;
    const r = scroller.getBoundingClientRect();
    return r.left >= -1 && r.right <= de.clientWidth + 1 && scroller.scrollWidth > scroller.clientWidth;
  };
  const wide = Array.from(document.querySelectorAll('main *')).filter(el => {
    const r = el.getBoundingClientRect();
    if (r.height <= 0 || r.width <= 0) return false;
    const outside = r.right > de.clientWidth + 1 || r.left < -1;
    return outside && !isLocallyScrollable(el);
  });
  const duplicateIds = Array.from(document.querySelectorAll('[id]'))
    .map((el) => el.id)
    .filter((id, index, ids) => id && ids.indexOf(id) !== index);
  const ariaCurrent = document.querySelectorAll('[aria-current="page"]').length;
  const ariaExpanded = document.querySelectorAll('[aria-expanded]').length;
  return {
    vw: window.innerWidth, scrollW: de.scrollWidth, clientW: de.clientWidth,
    overflowX: de.scrollWidth > de.clientWidth,
    sidebarRight: aR ? Math.round(aR.right) : null,
    mainLeft: mR ? Math.round(mR.left) : null,
    match: (aR && mR) ? Math.round(Math.abs(aR.right - mR.left) * 100) / 100 : null,
    h1Count: h1s.length,
    wideCount: wide.length,
    wideSamples: wide.slice(0, 8).map((el) => ({
      tag: el.tagName.toLowerCase(),
      className: typeof el.className === 'string' ? el.className.slice(0, 160) : '',
      text: (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 100),
      right: Math.round(el.getBoundingClientRect().right),
    })),
    duplicateIds: Array.from(new Set(duplicateIds)),
    ariaCurrent, ariaExpanded,
    activePage: document.querySelector('[data-ui-component="workstation-page"]')?.getAttribute('data-ui-page') ?? null,
  };
})()`;

async function inspect(client, page) {
  const health = await fetch(baseUrl).catch(() => null);
  if (!health?.ok) throw new Error(`workstation service unavailable before ${page}`);
  await client.send("Page.navigate", { url: `${baseUrl}/?page=${page}` });
  let ready = false;
  for (let i = 0; i < 50; i++) {
    await sleep(150);
    const ok = await client.eval(`document.readyState==='complete' && !!document.querySelector('[data-ui-component="workstation-page"]') && document.querySelector('[data-ui-component="workstation-page"]').getAttribute('data-ui-page')==='${page}' && document.querySelectorAll('main h1').length>0`);
    if (ok) { ready = true; break; }
  }
  if (!ready) throw new Error(`page did not become ready: ${page}`);
  await sleep(400);
  return await client.eval(GEOM);
}

async function run() {
  const chrome = findChrome();
  if (!chrome) throw new Error("chrome not found");
  await mkdir(OUT_DIR, { recursive: true });
  await mkdir(SHOT_DIR, { recursive: true });
  const port = 9333;
  const results = [];
  const consoleByPage = {};

  for (const vp of viewports) {
    const profileDir = await mkdtemp(path.join(os.tmpdir(), `evomind-ui-v3-${vp.name}-`));
    const proc = spawn(chrome, [
      "--headless=new", "--disable-gpu", "--hide-scrollbars",
      "--disable-background-networking", "--disable-component-update", "--disable-extensions",
      "--disable-sync", "--no-first-run", "--no-default-browser-check",
      `--remote-debugging-port=${port}`,
      `--window-size=${vp.width},${vp.height}`,
      "--user-data-dir=" + profileDir,
      "about:blank",
    ], { stdio: "ignore" });
    let client = null;
    try {
      const version = await waitForDevtools(port, proc);
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = targets.find((t) => t.type === "page");
      client = new Cdp(page.webSocketDebuggerUrl);
      await client.connect();
      await client.send("Runtime.enable");
      await client.send("Page.enable");
      await client.send("Emulation.setDeviceMetricsOverride", { width: vp.width, height: vp.height, deviceScaleFactor: 1, mobile: vp.mobile });

      for (const pg of pages) {
        client.consoleErrors = [];
        let g;
        try { g = await inspect(client, pg); } catch (e) {
          g = { error: String(e) };
          if (g.error.includes("workstation service unavailable")) throw e;
        }
        // sidebar collapse geometry (desktop only)
        let collapsed = null;
        if (!vp.mobile && !g.error) {
          try {
            await client.eval(`document.querySelector('[data-ui-action="toggle_desktop_sidebar"]')?.click()`);
            await sleep(400);
            collapsed = await client.eval(GEOM);
            await client.eval(`document.querySelector('[data-ui-action="toggle_desktop_sidebar"]')?.click()`);
            await sleep(300);
          } catch { /* ignore */ }
        }
        // Navigation drawer check at every viewport below the desktop breakpoint.
        let drawer = null;
        let evidenceOverlay = null;
        if (vp.width < 1024 && pg === "overview") {
          try {
            const mainTopBefore = await client.eval(`Math.round(document.querySelector('main')?.getBoundingClientRect().top || 0)`);
            await client.eval(`document.querySelector('[data-ui-action="toggle_mobile_navigation"]')?.click()`);
            await sleep(500);
            drawer = await client.eval(`(()=>{
              const d=document.querySelector('.shell-drawer[role="dialog"]');
              const b=document.querySelector('.shell-drawer-backdrop');
              const main=document.querySelector('main');
              const focusables=Array.from(d?.querySelectorAll('button,a,input,select,textarea,[tabindex]:not([tabindex="-1"])')||[]).filter(el=>!el.disabled);
              const duplicateIds=Array.from(document.querySelectorAll('[id]')).map(el=>el.id).filter((id,index,ids)=>id&&ids.indexOf(id)!==index);
              const first=focusables[0]; const last=focusables[focusables.length-1];
              last?.focus();
              document.dispatchEvent(new KeyboardEvent('keydown',{key:'Tab',bubbles:true}));
              return {
                dialog: !!d, ariaModal: d?.getAttribute('aria-modal'), backdrop: !!b,
                lock: document.body.getAttribute('data-nav-lock'), drawerW: Math.round(d?.getBoundingClientRect().width||0),
                mainTop: Math.round(main?.getBoundingClientRect().top||0), focusedInside: !!d?.contains(document.activeElement),
                tabWrapped: document.activeElement===first, duplicateIds:Array.from(new Set(duplicateIds)),
              };
            })()`);
            drawer.mainShift = Math.abs(drawer.mainTop - mainTopBefore);
            const drawerCapture = await client.send("Page.captureScreenshot", { format: "png" });
            drawer.screenshot = path.join(SHOT_DIR, `${vp.name}_${pg}_navigation-drawer.png`);
            await writeFile(drawer.screenshot, Buffer.from(drawerCapture.data, "base64"));
            await client.eval(`document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
            await sleep(300);
            drawer.afterEsc = await client.eval(`(()=>({ dialogGone: !document.querySelector('.shell-drawer[role="dialog"]'), unlocked: !document.body.getAttribute('data-nav-lock'), focusRestored: document.activeElement?.getAttribute('data-ui-action')==='toggle_mobile_navigation' }))()`);
            await client.eval(`document.querySelector('[data-ui-action="toggle_mobile_navigation"]')?.click()`);
            await sleep(200);
            await client.eval(`document.querySelector('.shell-drawer-backdrop')?.click()`);
            await sleep(200);
            drawer.afterBackdrop = await client.eval(`(()=>({ dialogGone: !document.querySelector('.shell-drawer[role="dialog"]'), unlocked: !document.body.getAttribute('data-nav-lock') }))()`);
          } catch (e) { drawer = { error: String(e) }; }
        }
        if (pg === "overview") {
          try {
            await client.eval(`document.querySelector('[data-ui-action="open_evidence_overlay"]')?.click()`);
            await sleep(300);
            evidenceOverlay = await client.eval(`(()=>{
              const mobile=window.innerWidth<1024;
              const dialog=document.querySelector('.mobile-evidence-drawer[role="dialog"]');
              const rail=document.querySelector('[role="complementary"][data-ui-component="evidence-rail"]');
              return { mobile, dialog:!!dialog, ariaModal:dialog?.getAttribute('aria-modal'), rail:!!rail, lock:document.body.getAttribute('data-evidence-lock'), focusedInside:!!dialog?.contains(document.activeElement) };
            })()`);
            const evidenceCapture = await client.send("Page.captureScreenshot", { format: "png" });
            evidenceOverlay.screenshot = path.join(SHOT_DIR, `${vp.name}_${pg}_evidence-overlay.png`);
            await writeFile(evidenceOverlay.screenshot, Buffer.from(evidenceCapture.data, "base64"));
            await client.eval(`document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
            await sleep(200);
            evidenceOverlay.afterEsc = await client.eval(`(()=>({closed:!document.querySelector('.mobile-evidence-drawer,[data-ui-component="evidence-rail"]'),unlocked:!document.body.getAttribute('data-evidence-lock')}))()`);
          } catch (e) { evidenceOverlay = { error: String(e) }; }
        }
        const hydrationErrors = client.consoleErrors.filter((e) => /hydration|Hydration|did not match|Minified React error #4(18|23)/.test(e));
        const errors = client.consoleErrors;
        consoleByPage[`${vp.name}/${pg}`] = errors;
        // screenshot
        let shot = null;
        try {
          const cap = await client.send("Page.captureScreenshot", { format: "png" });
          shot = path.join(SHOT_DIR, `${vp.name}_${pg}.png`);
          await writeFile(shot, Buffer.from(cap.data, "base64"));
        } catch { /* ignore */ }
        results.push({ viewport: vp.name, viewportWidth: vp.width, page: pg, ...g, collapsed, drawer, evidenceOverlay, hydrationErrorCount: hydrationErrors.length, consoleErrorCount: errors.length, consoleErrors: errors.slice(0, 5), screenshot: shot });
      }
    } catch (e) {
      results.push({ viewport: vp.name, error: String(e) });
    } finally {
      if (client) {
        await Promise.race([client.send("Browser.close"), sleep(2000)]).catch(() => undefined);
      }
      await stopChrome(proc, profileDir);
    }
  }

  // assertions
  const failures = [];
  for (const r of results) {
    if (r.error) { failures.push(`${r.viewport}/${r.page ?? "?"}: ${r.error}`); continue; }
    if (r.viewportWidth >= 1024) {
      if (r.match !== null && r.match > 1) failures.push(`${r.viewport}/${r.page}: main.left vs sidebar.right mismatch ${r.match}px`);
      if (r.collapsed && r.collapsed.match !== null && r.collapsed.match > 1) failures.push(`${r.viewport}/${r.page}: collapsed mismatch ${r.collapsed.match}px`);
    }
    if (r.overflowX) failures.push(`${r.viewport}/${r.page}: horizontal overflow scrollW=${r.scrollW} clientW=${r.clientW}`);
    if (r.wideCount > 0) failures.push(`${r.viewport}/${r.page}: ${r.wideCount} unclipped viewport violations ${JSON.stringify(r.wideSamples?.slice(0, 3) ?? [])}`);
    if ((r.duplicateIds?.length ?? 0) > 0) failures.push(`${r.viewport}/${r.page}: duplicate ids ${r.duplicateIds.join(',')}`);
    if (r.h1Count !== 1) failures.push(`${r.viewport}/${r.page}: h1Count=${r.h1Count}`);
    if (r.hydrationErrorCount > 0) failures.push(`${r.viewport}/${r.page}: ${r.hydrationErrorCount} hydration errors`);
    if (r.consoleErrorCount > 0) failures.push(`${r.viewport}/${r.page}: ${r.consoleErrorCount} console errors`);
    if (r.drawer) {
      if (r.drawer.error) failures.push(`${r.viewport}/${r.page}: drawer error ${r.drawer.error}`);
      if (!r.drawer.dialog || r.drawer.ariaModal !== 'true' || !r.drawer.backdrop || r.drawer.lock !== 'true') failures.push(`${r.viewport}/${r.page}: incomplete navigation drawer contract`);
      if (!r.drawer.focusedInside || !r.drawer.tabWrapped || r.drawer.mainShift > 1) failures.push(`${r.viewport}/${r.page}: navigation focus/layout contract failed`);
      if ((r.drawer.duplicateIds?.length ?? 0) > 0) failures.push(`${r.viewport}/${r.page}: drawer duplicate ids ${r.drawer.duplicateIds.join(',')}`);
      if (!r.drawer.afterEsc?.dialogGone || !r.drawer.afterEsc?.unlocked || !r.drawer.afterEsc?.focusRestored) failures.push(`${r.viewport}/${r.page}: navigation Escape/focus restore failed`);
      if (!r.drawer.afterBackdrop?.dialogGone || !r.drawer.afterBackdrop?.unlocked) failures.push(`${r.viewport}/${r.page}: navigation backdrop close failed`);
    }
    if (r.evidenceOverlay) {
      if (r.evidenceOverlay.error) failures.push(`${r.viewport}/${r.page}: evidence overlay error ${r.evidenceOverlay.error}`);
      if (r.evidenceOverlay.mobile && (!r.evidenceOverlay.dialog || r.evidenceOverlay.ariaModal !== 'true' || r.evidenceOverlay.lock !== 'true' || !r.evidenceOverlay.focusedInside)) failures.push(`${r.viewport}/${r.page}: mobile evidence dialog contract failed`);
      if (!r.evidenceOverlay.mobile && !r.evidenceOverlay.rail) failures.push(`${r.viewport}/${r.page}: desktop evidence rail missing`);
      if (!r.evidenceOverlay.afterEsc?.closed || !r.evidenceOverlay.afterEsc?.unlocked) failures.push(`${r.viewport}/${r.page}: evidence Escape/scroll unlock failed`);
    }
  }

  const report = {
    schema: "evomind.ui_shell_v3_acceptance.v1",
    created_at: new Date().toISOString(),
    base_url: baseUrl,
    viewport_count: viewports.length,
    page_count: pages.length,
    check_count: results.length,
    failure_count: failures.length,
    status: failures.length === 0 ? "passed" : "failed",
    failures,
    results,
  };
  await writeFile(path.join(OUT_DIR, "ui_v3_acceptance.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, checks: report.check_count, failures: report.failure_count, failures_list: failures.slice(0, 30) }, null, 2));
  process.exit(failures.length === 0 ? 0 : 1);
}

run().catch((e) => { console.error(e); process.exit(2); });
