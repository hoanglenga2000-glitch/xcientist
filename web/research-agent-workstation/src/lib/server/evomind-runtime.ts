import fs from "node:fs";
import path from "node:path";
import { runtimeRoot } from "@/lib/server/paths";

const port = Number(process.env.EVOMIND_RUNTIME_PORT ?? "8765");
const baseUrl = `http://127.0.0.1:${port}`;
const tokenPath = path.join(runtimeRoot, "runtime.token");

function token() {
  return fs.readFileSync(tokenPath, "ascii").trim();
}

async function ready() {
  if (!fs.existsSync(tokenPath)) return false;
  try {
    const response = await fetch(`${baseUrl}/v1/health`, { headers: { Authorization: `Bearer ${token()}` }, cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function ensureEvoMindRuntime() {
  if (await ready()) return { baseUrl, token: token() };
  // Runtime process creation and recovery belong exclusively to the signed
  // lifecycle manager, which records PID, creation time, executable,
  // command-line, cwd, listener and release nonce.  An API request must never
  // create an untracked replacement after a crash.
  throw new Error("EvoMind Runtime is not managed and ready");
}
