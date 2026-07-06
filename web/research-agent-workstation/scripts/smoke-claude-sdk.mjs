// Smoke test: does the Claude Agent SDK reach opus-4-8 through the lt4net gateway?
// Read-only tools, single logical turn. No writes, no commits. Reversible.
// Loads creds from web/research-agent-workstation/.env exactly as the workstation
// does at runtime.
//
// Run from the workstation dir:  npm run smoke:sdk
// Or directly:                   node scripts/smoke-claude-sdk.mjs
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, ".."); // web/research-agent-workstation
const repoRoot = path.resolve(here, "..", "..", ".."); // repo root

// Minimal .env loader (no external dep). Only sets keys not already in env.
function loadEnv(file) {
  let text = "";
  try { text = readFileSync(file, "utf-8"); } catch { return; }
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = val;
  }
}

loadEnv(path.join(appRoot, ".env"));

const model = process.env.CLAUDE_CODE_MODEL || "claude-opus-4-8";
const baseUrl = process.env.ANTHROPIC_BASE_URL || "(unset -> official api)";
const hasKey = Boolean(process.env.ANTHROPIC_API_KEY);

function mask(u) {
  try { const x = new URL(u); return `${x.protocol}//${x.host}${x.pathname}`; } catch { return u; }
}

console.log("=== SDK opus smoke config ===");
console.log("model            :", model);
console.log("ANTHROPIC_BASE_URL:", mask(baseUrl));
console.log("ANTHROPIC_API_KEY :", hasKey ? "present (masked)" : "MISSING");
console.log("node             :", process.version);
console.log("");

if (!hasKey) { console.error("No ANTHROPIC_API_KEY; aborting."); process.exit(2); }

const controller = new AbortController();
const t0 = Date.now();
const timeoutMs = 90_000;
const timer = setTimeout(() => controller.abort(), timeoutMs);

const collected = { textParts: [], toolUses: [], messages: 0, resultMeta: null, firstChunkMs: null };

try {
  const { query } = await import("@anthropic-ai/claude-agent-sdk");
  const iterable = query({
    prompt:
      "You are running a connectivity smoke test inside the Research Agent Workstation. " +
      "Use the Glob tool ONCE to list up to 5 files matching 'src/research_os/*.py', " +
      "then in one short sentence state the model name you are running as and confirm you reached the gateway. " +
      "Do not write or edit anything.",
    options: {
      abortController: controller,
      cwd: repoRoot,
      model,
      maxTurns: 3,
      permissionMode: "dontAsk",
      tools: ["Read", "Grep", "Glob"],
      disallowedTools: ["Edit", "MultiEdit", "Write", "Bash", "NotebookEdit"],
      persistSession: false,
      env: {
        ...process.env,
        ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY,
        ANTHROPIC_BASE_URL: process.env.ANTHROPIC_BASE_URL,
        CLAUDE_AGENT_SDK_CLIENT_APP: "research-agent-workstation-smoke/0.1.0"
      }
    }
  });

  for await (const message of iterable) {
    collected.messages += 1;
    if (collected.firstChunkMs === null) collected.firstChunkMs = Date.now() - t0;
    const m = message?.message;
    for (const block of m?.content ?? []) {
      if (block.type === "text" && block.text) collected.textParts.push(block.text);
      if (block.type === "tool_use") collected.toolUses.push(block.name);
    }
    if (message?.type === "result") {
      collected.resultMeta = {
        subtype: message.subtype,
        is_error: message.is_error,
        num_turns: message.num_turns,
        duration_ms: message.duration_ms,
        total_cost_usd: message.total_cost_usd,
        usage: message.usage,
        model: message.modelUsage ? Object.keys(message.modelUsage) : undefined
      };
    }
  }

  clearTimeout(timer);
  console.log("=== RESULT: SUCCESS ===");
  console.log("messages streamed :", collected.messages);
  console.log("first chunk ms    :", collected.firstChunkMs);
  console.log("total ms          :", Date.now() - t0);
  console.log("tool_use calls    :", JSON.stringify(collected.toolUses));
  console.log("result meta       :", JSON.stringify(collected.resultMeta, null, 2));
  console.log("");
  console.log("--- assistant text ---");
  console.log(collected.textParts.join("\n").slice(0, 1200));
  process.exit(0);
} catch (err) {
  clearTimeout(timer);
  console.log("=== RESULT: FAILED ===");
  console.log("aborted (timeout) :", controller.signal.aborted);
  console.log("elapsed ms        :", Date.now() - t0);
  console.log("messages before err:", collected.messages);
  console.log("tool_use calls    :", JSON.stringify(collected.toolUses));
  console.log("error name        :", err?.name);
  console.log("error message     :", (err?.message || String(err)).slice(0, 800));
  process.exit(1);
}
