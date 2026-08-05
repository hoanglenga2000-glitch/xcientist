import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const defaultRequestPath = ".xsci/scientist_upgrade_campaign_request.json";
const blockedScoreCap = 84;
const controllerSchema = "evomind.self_upgrade_controller.v1";

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

function recordValue(value: unknown) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function parseCampaignResult(stdout: string) {
  if (!stdout) return null;
  try {
    return recordValue(sanitizeClientJson(JSON.parse(stdout)));
  } catch {
    return null;
  }
}

function campaignResultFromError(error: unknown) {
  const stdout = typeof (error as { stdout?: unknown })?.stdout === "string"
    ? String((error as { stdout: string }).stdout)
    : "";
  return parseCampaignResult(stdout);
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function booleanValue(value: unknown) {
  return value === true;
}

function sameStrings(actual: unknown[], expected: string[]) {
  return actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function normalizeCampaignStatus(value: unknown) {
  const result = recordValue(sanitizeClientJson(value));
  const certification = recordValue(result?.certification);
  const campaign = recordValue(result?.upgrade_campaign);
  const blockers = result?.blockers;
  if (
    !result
    || result.tool !== "research_parity_gate"
    || !certification
    || !campaign
    || !Array.isArray(blockers)
    || blockers.some((blocker) => typeof blocker !== "string" || !blocker)
  ) {
    return null;
  }

  if (
    certification.tool !== "capability_certification_status"
    || campaign.tool !== "scientist_upgrade_campaign_status"
    || typeof certification.verified !== "boolean"
    || typeof certification.release_allowed !== "boolean"
    || typeof certification.parity_claim_allowed !== "boolean"
    || typeof campaign.active_and_verified !== "boolean"
    || typeof result.certification_campaign_source_binding_verified !== "boolean"
    || typeof result.certification_campaign_artifact_binding_verified !== "boolean"
  ) {
    return null;
  }

  const certificationVerified = certification.verified === true;
  const campaignVerified = campaign.active_and_verified === true;
  if (
    certificationVerified !== (
      certification.status === "certified"
      && certification.release_allowed === true
      && certification.parity_claim_allowed === true
    )
    || campaignVerified !== (campaign.status === "active_verified")
  ) {
    return null;
  }

  const sourceBindingVerified = result.certification_campaign_source_binding_verified === true;
  const artifactBindingVerified = result.certification_campaign_artifact_binding_verified === true;
  if ((!certificationVerified || !campaignVerified) && (sourceBindingVerified || artifactBindingVerified)) {
    return null;
  }
  const expectedBlockers = [
    ...(!certificationVerified ? ["external_capability_certification_not_verified"] : []),
    ...(!campaignVerified ? ["active_self_upgrade_campaign_not_verified"] : []),
    ...(certificationVerified && campaignVerified && !sourceBindingVerified
      ? ["certification_campaign_source_mismatch"]
      : []),
    ...(certificationVerified && campaignVerified && !artifactBindingVerified
      ? ["certification_campaign_artifact_mismatch"]
      : [])
  ];
  if (!sameStrings(blockers, expectedBlockers)) return null;

  const scoreCap = result.score_cap;
  if (result.status === "blocked") {
    if (
      result.parity_claim_allowed !== false
      || scoreCap !== blockedScoreCap
      || blockers.length === 0
      || result.claim !== "research parity is not externally certified"
    ) {
      return null;
    }
  } else if (result.status === "certified_research_parity") {
    if (
      result.parity_claim_allowed !== true
      || scoreCap !== 100
      || blockers.length !== 0
      || certification.verified !== true
      || campaign.active_and_verified !== true
      || result.certification_campaign_source_binding_verified !== true
      || result.certification_campaign_artifact_binding_verified !== true
      || result.claim !== "externally certified non-inferiority against named Codex and Claude baselines"
    ) {
      return null;
    }
  } else {
    return null;
  }

  const resultSha256 = stringValue(certification.result_sha256);
  return {
    ok: result.status === "certified_research_parity",
    tool: "research_parity_gate",
    status: result.status,
    parity_claim_allowed: result.parity_claim_allowed,
    score_cap: scoreCap,
    certification: {
      tool: "capability_certification_status",
      status: certification.status,
      verified: certificationVerified,
      release_allowed: certification.release_allowed,
      parity_claim_allowed: certification.parity_claim_allowed,
      ...(resultSha256.match(/^[0-9a-f]{64}$/) ? { result_sha256: resultSha256 } : {}),
      source_identity_matches: booleanValue(certification.source_identity_matches)
    },
    upgrade_campaign: {
      tool: "scientist_upgrade_campaign_status",
      status: campaign.status,
      campaign_status: stringValue(campaign.campaign_status),
      active_and_verified: campaignVerified,
      champion_ref: stringValue(campaign.champion_ref) || "refs/evomind/champion",
      promotion_verified: booleanValue(campaign.promotion_verified),
      rollback_verified: booleanValue(campaign.rollback_verified),
      strict_improvement_verified: booleanValue(campaign.strict_improvement_verified),
      champion_ref_matches: booleanValue(campaign.champion_ref_matches)
    },
    certification_campaign_source_binding_verified: sourceBindingVerified,
    certification_campaign_artifact_binding_verified: artifactBindingVerified,
    blockers: [...blockers],
    claim: result.claim
  };
}

function blockedCampaignStatusFromError(error: unknown) {
  const result = normalizeCampaignStatus(campaignResultFromError(error));
  return result?.status === "blocked" ? result : null;
}

function blockedCampaignCommandFromError(error: unknown, expectedAction: string) {
  const result = campaignResultFromError(error);
  if (
    result?.ok !== false
    || result.tool !== "scientist_upgrade_campaign"
    || result.status !== "blocked"
    || result.action !== expectedAction
  ) {
    return null;
  }
  return {
    ok: false,
    tool: "scientist_upgrade_campaign",
    action: result.action,
    status: "blocked"
  };
}

function normalizeCampaignCommand(value: unknown, action: string) {
  const result = recordValue(sanitizeClientJson(value));
  const expectedStatuses: Record<string, string[]> = {
    run: ["awaiting_human_promotion", "held_no_strict_improvement"],
    promote: ["active"],
    rollback: ["rolled_back"]
  };
  const allowed = expectedStatuses[action];
  if (
    !result
    || !allowed
    || result.schema !== controllerSchema
    || !allowed.includes(String(result.status ?? ""))
    || typeof result.campaign_id !== "string"
    || !result.campaign_id
    || result.main_worktree_modified !== false
    || result.no_training_started !== true
  ) {
    return null;
  }
  return {
    ok: true,
    tool: "scientist_upgrade_campaign",
    action,
    status: result.status,
    campaign_id: result.campaign_id,
    human_gate: stringValue(result.human_gate),
    main_worktree_modified: false,
    no_training_started: true
  };
}

export async function GET() {
  try {
    const result = normalizeCampaignStatus(await invokeCampaign(["status"], 60));
    if (!result) throw new Error("invalid upgrade campaign status contract");
    return NextResponse.json({
      ok: true,
      action: "scientist_upgrade_campaign_status",
      scientist_upgrade_campaign: result,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    });
  } catch (error) {
    const cliResult = blockedCampaignStatusFromError(error);
    if (cliResult) {
      return NextResponse.json({
        // The lookup succeeded; the nested campaign remains explicitly blocked.
        ok: true,
        action: "scientist_upgrade_campaign_status",
        error: "Upgrade campaign is blocked",
        scientist_upgrade_campaign: cliResult,
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      });
    }
    return NextResponse.json({
      ok: false,
      action: "scientist_upgrade_campaign_status",
      error: "Upgrade campaign status failed",
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
    const invoked = await invokeCampaign(args, timeoutSeconds);
    const result = action === "status"
      ? normalizeCampaignStatus(invoked)
      : normalizeCampaignCommand(invoked, action);
    if (!result) throw new Error("invalid upgrade campaign result contract");
    return NextResponse.json({
      ok: true,
      action: `scientist_upgrade_campaign_${action}`,
      scientist_upgrade_campaign: result,
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    });
  } catch (error) {
    const statusResult = action === "status" ? blockedCampaignStatusFromError(error) : null;
    if (statusResult) {
      return NextResponse.json({
        ok: true,
        action: "scientist_upgrade_campaign_status",
        error: "Upgrade campaign is blocked",
        scientist_upgrade_campaign: statusResult,
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      });
    }
    if (action === "status") {
      return NextResponse.json({
        ok: false,
        action: "scientist_upgrade_campaign_status",
        error: "Upgrade campaign status failed",
        scientist_upgrade_campaign: { status: "unavailable", parity_claim_allowed: false },
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval"
      }, { status: 500 });
    }
    const cliResult = blockedCampaignCommandFromError(error, action);
    return NextResponse.json({
      ok: false,
      action: `scientist_upgrade_campaign_${action}`,
      error: cliResult ? "Upgrade campaign command was blocked" : "Upgrade campaign command failed",
      ...(cliResult ? { scientist_upgrade_campaign: cliResult } : {}),
      no_training_started: true,
      official_submit: "blocked_until_explicit_human_approval"
    }, { status: 409 });
  }
}
