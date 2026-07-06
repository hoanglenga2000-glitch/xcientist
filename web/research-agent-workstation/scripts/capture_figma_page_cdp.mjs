import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const chromePath = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const page = process.env.CAPTURE_PAGE || "mission";
const captureId = process.env.FIGMA_CAPTURE_ID;
const endpoint = process.env.FIGMA_ENDPOINT;
const port = Number(process.env.CDP_PORT || 9333);
const profile = process.env.CDP_PROFILE || `D:\\桌面\\codex\\科研港科技\\docs\\ui-editable-capture-20260627\\cdp-profile-${page}`;

if (!captureId || !endpoint) {
  throw new Error("FIGMA_CAPTURE_ID and FIGMA_ENDPOINT are required.");
}

const targetUrl = `http://127.0.0.1:8099/?page=${encodeURIComponent(page)}&capture=cdp`;

const chrome = spawn(chromePath, [
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--window-size=1672,941",
  targetUrl
], {
  detached: true,
  stdio: "ignore"
});
chrome.unref();

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

for (let i = 0; i < 30; i += 1) {
  try {
    await getJson(`http://127.0.0.1:${port}/json/version`);
    break;
  } catch {
    await delay(500);
  }
}

await delay(2500);
const targets = await getJson(`http://127.0.0.1:${port}/json/list`);
const target = targets.find((item) => item.type === "page" && item.url.includes(`page=${page}`)) ?? targets.find((item) => item.type === "page");
if (!target?.webSocketDebuggerUrl) {
  throw new Error(`No debuggable page target found for ${page}.`);
}

let nextId = 1;
const pending = new Map();
const ws = new WebSocket(target.webSocketDebuggerUrl);

function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
}

await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, { once: true });
  ws.addEventListener("error", reject, { once: true });
});

ws.addEventListener("message", (event) => {
  const payload = JSON.parse(event.data);
  if (!payload.id || !pending.has(payload.id)) return;
  const { resolve, reject } = pending.get(payload.id);
  pending.delete(payload.id);
  if (payload.error) reject(new Error(JSON.stringify(payload.error)));
  else resolve(payload.result);
});

await send("Runtime.enable");
await send("Page.enable");
await send("Page.bringToFront");
await delay(3000);

const expression = `
  (async () => {
    const scriptText = await fetch('https://mcp.figma.com/mcp/html-to-design/capture.js').then((r) => r.text());
    const script = document.createElement('script');
    script.textContent = scriptText;
    document.head.appendChild(script);
    const started = Date.now();
    while (!window.figma?.captureForDesign) {
      if (Date.now() - started > 15000) throw new Error('captureForDesign was not installed');
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
    return await window.figma.captureForDesign({
      captureId: '${captureId}',
      endpoint: '${endpoint}',
      selector: 'body'
    });
  })()
`;

const result = await send("Runtime.evaluate", {
  expression,
  awaitPromise: true,
  returnByValue: true
});

await delay(1000);
ws.close();

console.log(JSON.stringify({
  page,
  captureId,
  endpoint,
  targetUrl,
  result: result.result?.value ?? result.result?.description ?? result
}, null, 2));
