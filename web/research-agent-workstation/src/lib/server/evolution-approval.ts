import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
// @ts-expect-error Node's strip-types tests require explicit TypeScript suffixes.
import { workspaceRoot } from "./paths.ts";
// @ts-expect-error Node's strip-types tests require explicit TypeScript suffixes.
import { CANONICAL_HASH_SCHEMA, sha256Canonical } from "./evolution-integrity.ts";

const APPROVAL_SCHEMA = "evomind.evolution_approval_plan.v2";
const PLAN_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const TASK_ID_RE = /^[A-Za-z0-9._-]+$/;
const TRANSIENT_REQUEST_KEYS = new Set([
  "approve",
  "plan_id",
  "plan_sha256",
  "request_fingerprint",
  "official_submit_allowed",
]);

type ApprovalStatus = "awaiting_approval" | "approved";

export type EvolutionApprovalPlan = {
  schema: typeof APPROVAL_SCHEMA;
  hash_canonicalization: typeof CANONICAL_HASH_SCHEMA;
  plan_id: string;
  task_id: string;
  status: ApprovalStatus;
  request_fingerprint: string;
  request_contract: Record<string, unknown>;
  plan_sha256: string;
  plan: Record<string, unknown>;
  created_at: string;
  expires_at: string;
  approved_at: string | null;
  receipt_sha256: string;
};

type StorageOptions = {
  storageRoot?: string;
  now?: Date;
  ttlSeconds?: number;
};

function strictJsonSnapshot(value: unknown): unknown {
  // canonicalJson is deliberately stricter than JSON.stringify: it rejects
  // non-finite/unsafe numbers, sparse arrays, symbols, accessors, cycles and
  // non-plain objects. Validate before cloning so two different requests can
  // never collapse to the same persisted approval contract.
  sha256Canonical(value);
  return JSON.parse(JSON.stringify(value)) as unknown;
}

function safeTaskId(taskId: string) {
  if (!TASK_ID_RE.test(taskId) || taskId.includes("..")) {
    throw new Error("Invalid evolution approval task_id.");
  }
  return taskId;
}

export function evolutionRequestContract(input: Record<string, unknown>) {
  const snapshot = strictJsonSnapshot(input) as Record<string, unknown>;
  return Object.fromEntries(
    Object.entries(snapshot)
      .filter(([key]) => !TRANSIENT_REQUEST_KEYS.has(key))
  );
}

export function evolutionRequestFingerprint(input: Record<string, unknown>) {
  return sha256Canonical(evolutionRequestContract(input));
}

function receiptContent(receipt: Omit<EvolutionApprovalPlan, "receipt_sha256"> | EvolutionApprovalPlan) {
  const content = { ...receipt } as Partial<EvolutionApprovalPlan>;
  delete content.receipt_sha256;
  return content;
}

function signedReceipt(receipt: Omit<EvolutionApprovalPlan, "receipt_sha256">): EvolutionApprovalPlan {
  return { ...receipt, receipt_sha256: sha256Canonical(receipt) };
}

function approvalDirectory(taskId: string, root = workspaceRoot) {
  return path.join(root, "workspace", "evolution", "approvals", safeTaskId(taskId));
}

function receiptPath(taskId: string, planId: string, root = workspaceRoot) {
  if (!PLAN_ID_RE.test(planId)) throw new Error("Invalid evolution approval plan_id.");
  const directory = approvalDirectory(taskId, root);
  const candidate = path.join(directory, `${planId}.json`);
  const relative = path.relative(directory, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error("Evolution approval path escaped its task directory.");
  return candidate;
}

async function writeReceiptAtomic(file: string, receipt: EvolutionApprovalPlan) {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await fs.rename(temporary, file).catch(async (error) => {
    await fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  });
}

export async function createEvolutionApprovalPlan(
  taskId: string,
  input: Record<string, unknown>,
  plan: Record<string, unknown>,
  options: StorageOptions = {},
) {
  const now = options.now ?? new Date();
  const ttlSeconds = options.ttlSeconds ?? 60 * 60;
  if (!Number.isFinite(ttlSeconds) || ttlSeconds < 60 || ttlSeconds > 24 * 60 * 60) {
    throw new Error("Evolution approval TTL must be between 60 seconds and 24 hours.");
  }
  const planId = randomUUID();
  const requestContract = evolutionRequestContract(input);
  const planSnapshot = strictJsonSnapshot(plan) as Record<string, unknown>;
  const unsigned: Omit<EvolutionApprovalPlan, "receipt_sha256"> = {
    schema: APPROVAL_SCHEMA,
    hash_canonicalization: CANONICAL_HASH_SCHEMA,
    plan_id: planId,
    task_id: taskId,
    status: "awaiting_approval",
    request_fingerprint: sha256Canonical(requestContract),
    request_contract: requestContract,
    plan_sha256: sha256Canonical(planSnapshot),
    plan: planSnapshot,
    created_at: now.toISOString(),
    expires_at: new Date(now.getTime() + ttlSeconds * 1000).toISOString(),
    approved_at: null,
  };
  const receipt = signedReceipt(unsigned);
  await writeReceiptAtomic(receiptPath(taskId, planId, options.storageRoot), receipt);
  return receipt;
}

export async function consumeEvolutionApprovalPlan(
  taskId: string,
  input: Record<string, unknown>,
  options: StorageOptions = {},
) {
  const planId = typeof input.plan_id === "string" ? input.plan_id : "";
  const claimedPlanSha256 = typeof input.plan_sha256 === "string" ? input.plan_sha256 : "";
  const claimedFingerprint = typeof input.request_fingerprint === "string" ? input.request_fingerprint : "";
  if (!PLAN_ID_RE.test(planId) || !/^[0-9a-f]{64}$/i.test(claimedPlanSha256) || !/^[0-9a-f]{64}$/i.test(claimedFingerprint)) {
    throw new Error("approve=true requires a valid plan_id, plan_sha256, and request_fingerprint.");
  }
  const file = receiptPath(taskId, planId, options.storageRoot);
  const claimLock = `${file}.consume.lock`;
  const claim = `${file}.${process.pid}.${randomUUID()}.consuming`;
  try {
    const lockHandle = await fs.open(claimLock, "wx");
    await lockHandle.close();
  } catch {
    throw new Error("Evolution approval plan does not exist or is already being consumed.");
  }
  try {
    // Moving the only pending receipt to a unique claim path is the atomic
    // single-use boundary. Exactly one concurrent consumer can acquire it.
    await fs.rename(file, claim);
  } catch {
    await fs.rm(claimLock, { force: true }).catch(() => undefined);
    throw new Error("Evolution approval plan does not exist or is already being consumed.");
  }

  let temporary = "";
  try {
    const stat = await fs.lstat(claim);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error("Evolution approval plan binding verification failed.");
    const parsed = JSON.parse(await fs.readFile(claim, "utf8")) as EvolutionApprovalPlan;
    const verifiedReceiptHash = sha256Canonical(receiptContent(parsed));
    const currentFingerprint = evolutionRequestFingerprint(input);
    const now = options.now ?? new Date();
    if (
      parsed.schema !== APPROVAL_SCHEMA
      || parsed.hash_canonicalization !== CANONICAL_HASH_SCHEMA
      || parsed.plan_id !== planId
      || parsed.task_id !== taskId
      || parsed.status !== "awaiting_approval"
      || parsed.receipt_sha256 !== verifiedReceiptHash
      || parsed.plan_sha256 !== sha256Canonical(parsed.plan)
      || parsed.plan_sha256 !== claimedPlanSha256
      || parsed.request_fingerprint !== claimedFingerprint
      || parsed.request_fingerprint !== currentFingerprint
      || parsed.request_fingerprint !== sha256Canonical(parsed.request_contract)
    ) {
      throw new Error("Evolution approval plan binding verification failed.");
    }
    if (!Number.isFinite(Date.parse(parsed.expires_at)) || now.getTime() >= Date.parse(parsed.expires_at)) {
      throw new Error("Evolution approval plan expired.");
    }
    const approved = signedReceipt({
      ...receiptContent(parsed) as Omit<EvolutionApprovalPlan, "receipt_sha256">,
      status: "approved",
      approved_at: now.toISOString(),
    });
    temporary = `${file}.${process.pid}.${randomUUID()}.approved.tmp`;
    await fs.writeFile(temporary, `${JSON.stringify(approved, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await fs.rename(temporary, file);
    temporary = "";
    // The approved receipt is now the authoritative state. Cleanup of the
    // uniquely named claim is best-effort and must never restore an awaiting
    // receipt over the committed approved receipt.
    await fs.rm(claim, { force: true }).catch(() => undefined);
    await fs.rm(claimLock, { force: true }).catch(() => undefined);
    return approved;
  } catch (error) {
    if (temporary) await fs.rm(temporary, { force: true }).catch(() => undefined);
    // Preserve an unconsumed/invalid receipt for audit and deterministic retry.
    await fs.link(claim, file)
      .then(() => fs.rm(claim, { force: true }))
      .catch(() => undefined);
    await fs.rm(claimLock, { force: true }).catch(() => undefined);
    throw error;
  }
}
