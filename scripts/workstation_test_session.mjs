import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

// Browser release checks authenticate the same way a person does: with the
// local session obtained once through the server's one-time bootstrap URL
// (see workstation_local_auth.py `establish`). The local automation header
// is only a fallback; it deliberately carries no user principal, so pages
// that read personal data show task_principal_required under it.

const SESSION_COOKIE = "evomind_local_session";
const VALUE_PATTERN = /^[A-Za-z0-9._-]{20,2048}$/;
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{24,256}$/;

function runtimeDir(root) {
  return process.env.WORKSTATION_RUNTIME_DIR
    ? resolve(process.env.WORKSTATION_RUNTIME_DIR)
    : join(root, "web", "research-agent-workstation", ".runtime-logs");
}

function loopbackPort(baseUrl) {
  try {
    const parsed = new URL(baseUrl);
    if (parsed.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(parsed.hostname)) return null;
    return Number(parsed.port || 80);
  } catch {
    return null;
  }
}

function suffix(port) {
  return port === 8088 ? "" : `.${port}`;
}

export async function localTestSession(root, baseUrl) {
  const port = loopbackPort(baseUrl);
  if (port === null) return null;
  try {
    const payload = JSON.parse(await readFile(join(runtimeDir(root), `dashboard${suffix(port)}.test-session.json`), "utf8"));
    if (payload?.port !== port || !VALUE_PATTERN.test(payload?.cookie ?? "")) return null;
    return { cookie: payload.cookie };
  } catch {
    return null;
  }
}

export async function localAutomationToken(root, baseUrl) {
  const port = loopbackPort(baseUrl);
  if (port === null) return "";
  try {
    const token = (await readFile(join(runtimeDir(root), `dashboard${suffix(port)}.automation.token`), "ascii")).trim();
    return TOKEN_PATTERN.test(token) ? token : "";
  } catch {
    return "";
  }
}

/** Apply release-check credentials to a CDP client; returns the auth mode. */
export async function applyReleaseCheckAuth(client, root, baseUrl) {
  const session = await localTestSession(root, baseUrl);
  if (session) {
    const ok = await client.send("Network.setCookie", {
      name: SESSION_COOKIE,
      value: session.cookie,
      url: new URL(baseUrl).origin,
      httpOnly: true,
      sameSite: "Strict",
      path: "/",
    });
    if (ok?.success === false) throw new Error("test_session_cookie_rejected");
    return "local_session";
  }
  const token = await localAutomationToken(root, baseUrl);
  if (token) {
    await client.send("Network.setExtraHTTPHeaders", { headers: { "x-evomind-local-automation": token } });
    return "local_automation_header";
  }
  return "none";
}
