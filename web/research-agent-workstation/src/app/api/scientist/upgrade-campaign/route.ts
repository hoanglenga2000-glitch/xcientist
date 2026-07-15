import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const defaultRequestPath = ".xsci/scientist_upgrade_campaign_request.json";

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || "python";
}

function localPath(value: unknown, fallback?: string) {
  const text = typeof value === "string" && value.trim() ? value.trim() : fallback;
  if (!text) return undefined;
  return resolveWorkspacePath(text);
}

async function invokeCampaign(args: string[], timeoutSeconds: number) {
  const pythonPath = [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  const { stdout } = await execFileAsync(
    pythonExecutable(),
    ["-m", "xsci.kaggle", "upgrade-campaign", ...args, "--json"],
    {
      cwd: workspaceRoot,
      timeout: (timeoutSeconds + 30) * 1000,
      maxBuffer: 4 * 1024 * 1024,
      windowsHide: true,
      env: { ...process.env, PYTHONPATH: pythonPath, PYTHONUTF8: "1" }
    }
  );
  return sanitizeClientJson(JSON.parse(stdout) as Record<string, unknown>) as Record<string, unknown>;
}

function campaignResultFromError(error: unknown) {
  const stdout = typeof (error as { stdout?: unknown })?.stdout === "string"
    ? String((error as { stdout: string }).stdout)
    : "";
  try {
    return stdout
      ? sanitizeClientJson(JSON.parse(stdout) as Record<string, unknown>) as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const result = await invokeCampaign(["status"], 60);
    return NextResponse.json({
      ok: true,
      action: "scientist_upgrade_campaign_status",
      scientist_upgrade_campaign: result,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    });
  } catch (error) {
    const cliResult = campaignResultFromError(error);
    if (cliResult) {
      return NextResponse.json({
        // The lookup succeeded; the nested campaign remains explicitly blocked.
        ok: true,
        action: "scientist_upgrade_campaign_status",
        error: typeof cliResult.error === "string" ? cliResult.error : "Upgrade campaign is blocked",
        scientist_upgrade_campaign: {
          ...cliResult,
          parity_claim_allowed: false,
          score_cap: 84
        },
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      });
    }
    const message = error instanceof Error ? error.message : "Upgrade campaign status failed";
    return NextResponse.json({
      ok: false,
      action: "scientist_upgrade_campaign_status",
      error: message,
      scientist_upgrade_campaign: { status: "unavailable", parity_claim_allowed: false },
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({})) as Record<string, unknown>;
  const action = typeof body.action === "string" ? body.action.trim().toLowerCase() : "status";
  if (!["status", "run", "promote", "rollback"].includes(action)) {
    return NextResponse.json({ ok: false, error: "Unsupported upgrade campaign action" }, { status: 400 });
  }
  const timeoutSeconds = Math.max(30, Math.min(1800, Number(body.timeout_seconds ?? 300) || 300));
  const args = [action];
  if (action !== "status") {
    try {
      args.push("--request", localPath(body.request_path, defaultRequestPath) as string);
      const manifestPath = localPath(body.manifest_path);
      if (manifestPath) args.push("--manifest", manifestPath);
    } catch {
      return NextResponse.json({ ok: false, error: "Request and manifest paths must stay inside the workspace" }, { status: 400 });
    }
  }
  if (action === "promote") {
    if (body.human_approved !== true) {
      return NextResponse.json({ ok: false, error: "Explicit human approval is required for promotion" }, { status: 409 });
    }
    args.push("--human-approved");
  }
  args.push("--timeout", String(timeoutSeconds));

  try {
    const result = await invokeCampaign(args, timeoutSeconds);
    const ok = action === "status" || [
      "awaiting_human_promotion",
      "held_no_strict_improvement",
      "active",
      "rolled_back"
    ].includes(String(result.status ?? ""));
    return NextResponse.json({
      ok,
      action: `scientist_upgrade_campaign_${action}`,
      scientist_upgrade_campaign: result,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: ok ? 200 : 409 });
  } catch (error) {
    const cliResult = campaignResultFromError(error);
    const message = typeof cliResult?.error === "string"
      ? cliResult.error
      : error instanceof Error
        ? error.message
        : "Upgrade campaign command failed";
    return NextResponse.json({
      ok: false,
      action: `scientist_upgrade_campaign_${action}`,
      error: message,
      ...(cliResult ? { scientist_upgrade_campaign: cliResult } : {}),
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: 409 });
  }
}
