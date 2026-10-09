// Shared browser probes for the shipped /workspace UI (TaskWorkspace).
//
// The release smokes (verify_workstation_click_smoke.mjs and
// verify_workstation_interactive_controls.mjs) drive a real headless
// Chromium over CDP. Everything here asserts markers rendered by the shipped
// components in web/research-agent-workstation/src/components/workstation/
// task-workspace/ — the sidebar (#workspace-navigation), the top bar
// (.tw-topbar), the main region (#task-main) and the per-view content — so a
// blank page, an error page, a permission error or a stale route cannot pass.
// Nothing here starts a model run, a training job or a submission.

export const ERROR_PATTERN = "Application error|Unhandled Runtime Error|Hydration failed|ChunkLoadError|Internal Server Error|task_principal_required|session_required|csrf_rejected";

// Views of the shipped workspace. `path` is what a person opens; the view id
// is re-derived from the DOM (detectView) so the URL alone can never pass.
export const WORKSPACE_VIEWS = [
  { id: "tasks", path: "/workspace", nav: "/workspace" },
  { id: "new_task", path: "/workspace?new=1", nav: null },
  { id: "data", path: "/workspace?section=data", nav: "/workspace?section=data" },
  { id: "literature", path: "/workspace?section=literature", nav: "/workspace?section=literature" },
  { id: "results", path: "/workspace?section=results", nav: "/workspace?section=results" },
  { id: "projects", path: "/workspace?section=projects", nav: null },
  { id: "settings_models", path: "/workspace?settings=models", nav: "/workspace?settings=models" },
  { id: "settings_account", path: "/workspace?settings=account", nav: "/workspace?settings=models" },
  { id: "settings_resources", path: "/workspace?settings=resources", nav: "/workspace?settings=models" }
];

// Runs in the page. Returns the view the DOM actually shows, or null.
const DETECT_VIEW = `(() => {
  const main = document.querySelector('.task-workspace main#task-main');
  if (!main) return null;
  const h1 = (main.querySelector('h1')?.textContent || '').trim();
  const settingsTab = main.querySelector('nav[aria-label="设置分组"] a[aria-current="page"]');
  if (h1 === '设置' && settingsTab) {
    const group = new URL(settingsTab.getAttribute('href'), location.origin).searchParams.get('settings');
    return group ? 'settings_' + group : null;
  }
  if (main.querySelector('form.tw-composer textarea#task-draft')) return 'new_task';
  if (main.querySelector('nav.tw-tabs[aria-label="任务内容"]')) return 'task_session';
  if (main.querySelector('section[aria-label="我的资料"] input[type="file"][aria-label="上传资料"]')) return 'data';
  if (h1 === '文献检索') return 'literature';
  if (h1 === '管理项目' && main.querySelector('input#new-project')) return 'projects';
  if (h1 === '成果') return 'results';
  if (h1 === '我的任务' && main.querySelector('input[aria-label="搜索任务"]') && main.querySelector('.tw-task-list')) return 'tasks';
  return null;
})()`;

// Per-view readiness beyond the marker: data that must have loaded.
const VIEW_READY = {
  settings_account: `(() => { const h = Array.from(document.querySelectorAll('main#task-main h2')).find(n => n.textContent.startsWith('当前账户：')); return !!h && !h.textContent.includes('正在读取'); })()`,
  results: `!Array.from(document.querySelectorAll('main#task-main [role="status"]')).some(n => n.textContent.includes('正在读取'))`,
  projects: `!Array.from(document.querySelectorAll('main#task-main [role="status"]')).some(n => n.textContent.includes('正在读取'))`
};

export function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

export async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text ?? "Runtime evaluation failed.");
  return result.result?.value;
}

export async function waitFor(client, expression, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await evaluate(client, expression)) return true;
    } catch (error) {
      // A broken probe expression must fail loudly, never read as "not yet".
      if (/SyntaxError/.test(String(error?.message ?? error))) throw error;
      // Otherwise navigation is in progress and the execution context was replaced.
    }
    await sleep(150);
  }
  return false;
}

export async function setDesktopViewport(client) {
  await client.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 960, deviceScaleFactor: 1, mobile: false });
}

export async function currentView(client) {
  return await evaluate(client, DETECT_VIEW);
}

/** Wait until the shell is mounted, the task list finished loading and the view is recognisable. */
export async function waitForWorkspace(client, expectedView = null, timeoutMs = 30000) {
  const ready = await waitFor(client, `(() => {
    if (document.readyState !== 'complete') return false;
    if (!document.querySelector('.task-workspace aside#workspace-navigation') || !document.querySelector('.task-workspace main#task-main')) return false;
    const loading = Array.from(document.querySelectorAll('main#task-main [role="status"]')).some(n => n.textContent.includes('正在读取你的任务'));
    if (loading) return false;
    const view = ${DETECT_VIEW};
    if (${JSON.stringify(expectedView)} !== null && view !== ${JSON.stringify(expectedView)}) {
      return new RegExp(${JSON.stringify(ERROR_PATTERN)}, 'i').test(document.body.innerText) || !!document.querySelector('main#task-main [role="alert"]');
    }
    if (view === null) return false;
    ${Object.entries(VIEW_READY).map(([id, check]) => `if (view === ${JSON.stringify(id)}) return ${check};`).join("\n    ")}
    return true;
  })()`, timeoutMs);
  return ready;
}

export async function openView(client, baseUrl, path, expectedView) {
  await client.send("Page.navigate", { url: `${baseUrl}${path}` });
  await sleep(100);
  return await waitForWorkspace(client, expectedView);
}

/** Facts about the rendered page used by the render smoke and the control audit. */
export async function inspectWorkspace(client) {
  return await evaluate(client, `(() => {
    const visible = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
    const aside = document.querySelector('.task-workspace aside#workspace-navigation[aria-label="主导航"]');
    const navLinks = aside ? Array.from(aside.querySelectorAll('nav a[href]')) : [];
    const current = navLinks.filter(a => a.getAttribute('aria-current') === 'page').map(a => a.getAttribute('href'));
    const alerts = Array.from(document.querySelectorAll('[role="alert"]')).filter(visible).map(n => n.textContent.trim()).filter(Boolean);
    const main = document.querySelector('main#task-main');
    return {
      activePage: ${DETECT_VIEW},
      url: location.pathname + location.search,
      shell: {
        workspace: !!document.querySelector('div.task-workspace'),
        skipLink: !!document.querySelector('a.tw-skip[href="#task-main"]'),
        sidebar: !!aside,
        brand: !!aside?.querySelector('a.tw-brand[href="/workspace"]'),
        newTask: !!aside?.querySelector('a.tw-new[href="/workspace?new=1"]'),
        navLinkCount: navLinks.length,
        topbar: !!document.querySelector('header.tw-topbar'),
        menuButton: !!document.querySelector('header.tw-topbar button.tw-menu[aria-controls="workspace-navigation"]'),
        accountLink: !!document.querySelector('header.tw-topbar a[href^="/workspace?settings=account"]'),
        main: !!main
      },
      navCurrent: current,
      heading: (main?.querySelector('h1')?.textContent || '').trim(),
      actionCount: document.querySelectorAll('a[href], button, input, select, textarea, summary').length,
      buttonCount: document.querySelectorAll('button').length,
      textSize: (main?.innerText || '').length + (aside?.innerText || '').length,
      mainTextSize: (main?.innerText || '').length,
      alerts,
      hasErrorText: new RegExp(${JSON.stringify(ERROR_PATTERN)}, 'i').test(document.body.innerText)
    };
  })()`);
}

export function shellOk(info) {
  const s = info?.shell ?? {};
  return Boolean(s.workspace && s.skipLink && s.sidebar && s.brand && s.newTask && s.navLinkCount >= 5 && s.topbar && s.menuButton && s.accountLink && s.main);
}

/** Real mouse click at the element's centre (CDP input events, not a synthetic DOM click). */
export async function clickSelector(client, selector) {
  const box = await evaluate(client, `(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    el.scrollIntoView({ block: 'center', inline: 'center' });
    const r = el.getBoundingClientRect();
    const label = (el.getAttribute('aria-label') || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 80);
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const hit = document.elementFromPoint(x, y);
    return { x, y, w: r.width, h: r.height, label, hittable: !!hit && (hit === el || el.contains(hit)) };
  })()`);
  if (!box) return { clicked: false, reason: "selector_not_found" };
  if (!(box.w > 0 && box.h > 0)) return { clicked: false, reason: "not_visible", label: box.label };
  if (!box.hittable) return { clicked: false, reason: "covered_by_other_element", label: box.label };
  for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) {
    await client.send("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: type === "mouseMoved" ? 0 : 1 });
  }
  return { clicked: true, label: box.label };
}

export async function typeInto(client, selector, text) {
  const focused = await evaluate(client, `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return false; el.focus(); return document.activeElement === el; })()`);
  if (!focused) return false;
  await client.send("Input.insertText", { text });
  return true;
}

export async function pressKey(client, key, code = key, keyCode = 0) {
  await client.send("Input.dispatchKeyEvent", { type: "keyDown", key, code, windowsVirtualKeyCode: keyCode });
  await client.send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode: keyCode });
}

export async function clearInput(client, selector) {
  await evaluate(client, `(() => { const el = document.querySelector(${JSON.stringify(selector)}); el?.focus(); el?.select?.(); return true; })()`);
  await pressKey(client, "Backspace", "Backspace", 8);
}

/**
 * Create a draft task the way a person does: open "新任务", type a request
 * and let the composer auto-save. Saving a draft never calls a model; the
 * task opens in its session view with "开始任务" still gated on a model choice.
 */
export async function createDraftTask(client, baseUrl, title) {
  const opened = await openView(client, baseUrl, "/workspace?new=1", "new_task");
  if (!opened) return { ok: false, reason: "new_task_view_not_ready" };
  const typed = await typeInto(client, "textarea#task-draft", title);
  if (!typed) return { ok: false, reason: "draft_textarea_not_focusable" };
  const saved = await waitFor(client, `/^\\/workspace$/.test(location.pathname) && new URLSearchParams(location.search).has('task') && !!document.querySelector('main#task-main nav.tw-tabs[aria-label="任务内容"]')`, 20000);
  const state = await evaluate(client, `(() => ({
    url: location.pathname + location.search,
    taskId: new URLSearchParams(location.search).get('task'),
    heading: (document.querySelector('main#task-main h1')?.textContent || '').trim(),
    alerts: Array.from(document.querySelectorAll('main#task-main [role="alert"]')).map(n => n.textContent.trim()).filter(Boolean)
  }))()`);
  return { ok: Boolean(saved && state.taskId && state.heading === title && state.alerts.length === 0), ...state };
}
