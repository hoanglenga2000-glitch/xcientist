import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import {
  selectLatestValidLiteratureManifest,
  type LiteratureManifestCandidate
} from "@/lib/server/literature-manifest-selection";
import { normalizeTaskId, resolveWorkspacePath, stamp } from "@/lib/server/paths";

type JsonRecord = Record<string, unknown>;

type LiteratureActionInput = {
  action: string;
  taskId: string;
  metadata?: Record<string, unknown>;
};

export type LiteratureActionResult = {
  artifactPath: string;
  message: string;
  metadata: Record<string, unknown>;
  result: Record<string, unknown>;
};

const AGENT_ACTIONS = new Set([
  "rag_build_agent_context",
  "rag_send_research_agent",
  "rag_send_code_agent",
  "rag_bind_report_claim",
  "rag_request_citation_audit"
]);

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function validTaskId(rawTaskId: string) {
  const taskId = normalizeTaskId(rawTaskId.trim());
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(taskId)) {
    throw new Error("Invalid task_id. Use letters, numbers, underscores, or hyphens only.");
  }
  return taskId;
}

function toPosix(value: string) {
  return value.replaceAll("\\", "/");
}

function safeTaskRelativePath(taskId: string, candidate: string, allowedRoots: string[]) {
  const normalized = toPosix(candidate.trim());
  if (!normalized || path.isAbsolute(candidate) || path.win32.isAbsolute(candidate) || normalized.startsWith("/")) {
    throw new Error("Artifact path must be a task-relative workspace path.");
  }
  const segments = normalized.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new Error("Artifact path contains an invalid path segment.");
  }
  const taskPrefix = `workspace/tasks/${taskId}/`;
  if (!normalized.startsWith(taskPrefix) || !allowedRoots.some((root) => normalized.startsWith(`${taskPrefix}${root}`))) {
    throw new Error("Artifact path is outside the selected task.");
  }
  const absolute = resolveWorkspacePath(normalized);
  const taskRoot = resolveWorkspacePath(`workspace/tasks/${taskId}`);
  const relativeToTask = path.relative(taskRoot, absolute);
  if (!relativeToTask || relativeToTask.startsWith("..") || path.isAbsolute(relativeToTask)) {
    throw new Error("Artifact path escapes the selected task.");
  }
  return { relative: normalized, absolute };
}

async function atomicWrite(target: string, data: string | Uint8Array) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(temporary, data);
  await fs.rename(temporary, target);
}

async function atomicWriteJson(target: string, payload: unknown) {
  await atomicWrite(target, `${JSON.stringify(payload, null, 2)}\n`);
}

async function atomicAppendJsonl(target: string, payload: unknown, idField?: string) {
  const existing = await fs.readFile(target, "utf-8").catch(() => "");
  if (idField && isRecord(payload)) {
    const expected = String(payload[idField] ?? "");
    const duplicate = existing.split(/\r?\n/).some((line) => {
      if (!line.trim()) return false;
      try {
        const parsed = JSON.parse(line) as JsonRecord;
        return String(parsed[idField] ?? "") === expected;
      } catch {
        return false;
      }
    });
    if (duplicate) return false;
  }
  await atomicWrite(target, `${existing}${JSON.stringify(payload)}\n`);
  return true;
}

function sha256(value: string | Uint8Array) {
  return createHash("sha256").update(value).digest("hex");
}

async function latestFile(directory: string, matcher: RegExp) {
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && matcher.test(entry.name))
    .map(async (entry) => {
      const absolute = path.join(directory, entry.name);
      const stat = await fs.stat(absolute);
      return { absolute, mtimeMs: stat.mtimeMs };
    }));
  return candidates.sort((left, right) => right.mtimeMs - left.mtimeMs)[0]?.absolute ?? null;
}

async function readJsonObject(target: string) {
  const text = await fs.readFile(target, "utf-8");
  const payload = JSON.parse(text.replace(/^\uFEFF/, "")) as unknown;
  if (!isRecord(payload)) throw new Error("Literature manifest must contain a JSON object.");
  return { payload, text };
}

async function latestValidLiteratureManifest(directory: string, taskId: string) {
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isFile() && /^context_.*\.json$/i.test(entry.name))
    .map(async (entry): Promise<LiteratureManifestCandidate | null> => {
      const absolutePath = path.join(directory, entry.name);
      try {
        const [stat, manifest] = await Promise.all([
          fs.stat(absolutePath),
          readJsonObject(absolutePath)
        ]);
        if (!stat.isFile()) return null;
        return {
          absolutePath,
          expectedTaskId: taskId,
          mtimeMs: stat.mtimeMs,
          payload: manifest.payload
        };
      } catch {
        return null;
      }
    }));
  return selectLatestValidLiteratureManifest(
    candidates.filter((candidate): candidate is LiteratureManifestCandidate => candidate !== null)
  );
}

async function loadManifest(taskId: string, metadata: Record<string, unknown>) {
  const supplied = typeof metadata.manifest_path === "string" ? metadata.manifest_path : "";
  const fallback = supplied
    ? null
    : await latestValidLiteratureManifest(resolveWorkspacePath(`workspace/tasks/${taskId}/rag`), taskId);
  if (!supplied && !fallback) throw new Error("No literature manifest exists for the selected task. Run a real literature search first.");
  const candidate = supplied || toPosix(path.relative(resolveWorkspacePath(""), fallback!.absolutePath));
  const resolved = safeTaskRelativePath(taskId, candidate, ["rag/"]);
  if (!/^context_.*\.json$/i.test(path.basename(resolved.relative))) {
    throw new Error("manifest_path must reference a task RAG context manifest.");
  }
  const stat = await fs.stat(resolved.absolute).catch(() => null);
  if (!stat?.isFile()) throw new Error("Literature manifest does not exist.");
  const { payload, text } = await readJsonObject(resolved.absolute);
  const manifestTaskId = validTaskId(String(payload.task_id ?? ""));
  if (manifestTaskId !== taskId) throw new Error("Literature manifest belongs to a different task.");
  if (!Array.isArray(payload.papers)) throw new Error("Literature manifest has no papers array.");
  const contextCandidate = typeof payload.context_path === "string"
    ? payload.context_path
    : typeof metadata.context_path === "string"
      ? metadata.context_path
      : "";
  if (!contextCandidate) throw new Error("Literature manifest has no context artifact.");
  const context = safeTaskRelativePath(taskId, contextCandidate, ["rag/"]);
  const contextStat = await fs.stat(context.absolute).catch(() => null);
  if (!contextStat?.isFile()) throw new Error("Literature context artifact does not exist.");
  return {
    payload,
    manifestPath: resolved.relative,
    manifestAbsolute: resolved.absolute,
    manifestChecksum: sha256(text),
    contextPath: context.relative
  };
}

function paperList(manifest: JsonRecord) {
  return (Array.isArray(manifest.papers) ? manifest.papers : []).filter(isRecord);
}

function compactPaper(paper: JsonRecord) {
  const provenance = isRecord(paper.provenance) ? paper.provenance : {};
  return {
    id: String(paper.id ?? ""),
    title: String(paper.title ?? ""),
    source: String(paper.source ?? "unknown"),
    year: String(paper.year ?? ""),
    authors: Array.isArray(paper.authors) ? paper.authors.slice(0, 12) : [],
    abstract: String(paper.abstract ?? "").slice(0, 1800),
    methods: Array.isArray(paper.methods) ? paper.methods.slice(0, 12) : [],
    risks: Array.isArray(paper.risks) ? paper.risks.slice(0, 12) : [],
    doi: typeof paper.doi === "string" ? paper.doi : null,
    url: typeof paper.url === "string" ? paper.url : typeof paper.source_url === "string" ? paper.source_url : null,
    artifact_path: typeof paper.artifact_path === "string" ? paper.artifact_path : null,
    provenance: {
      verified: provenance.verified === true,
      source: String(provenance.source ?? ""),
      retrieved_at: typeof provenance.retrieved_at === "string" ? provenance.retrieved_at : null,
      checksum: typeof provenance.checksum === "string" ? provenance.checksum : null
    }
  };
}

function actionKey(...parts: unknown[]) {
  return sha256(parts.map((part) => typeof part === "string" ? part : JSON.stringify(part)).join("\u0001"));
}

async function buildAgentContext(taskId: string, metadata: Record<string, unknown>) {
  const manifest = await loadManifest(taskId, metadata);
  const papers = paperList(manifest.payload).map(compactPaper);
  if (!papers.length) throw new Error("The selected literature manifest contains no papers.");
  const contextId = `literature_context_${manifest.manifestChecksum.slice(0, 20)}`;
  const relative = `workspace/tasks/${taskId}/rag/agent_contexts/${contextId}.json`;
  const payload = {
    schema: "evomind.multi_agent.literature_context.v1",
    context_id: contextId,
    task_id: taskId,
    created_at: new Date().toISOString(),
    source_manifest: {
      path: manifest.manifestPath,
      sha256: manifest.manifestChecksum,
      query: String(manifest.payload.query ?? ""),
      generated_at: manifest.payload.generated_at ?? null
    },
    source_context_path: manifest.contextPath,
    evidence_contract: {
      search_hit_is_not_claim_support: true,
      citation_requires_independent_review: true,
      official_score_rank_or_medal_claims: "blocked_without_official_response_artifact"
    },
    source_counts: isRecord(manifest.payload.source_counts) ? manifest.payload.source_counts : {},
    integrity: isRecord(manifest.payload.integrity) ? manifest.payload.integrity : {},
    papers,
    retrieval: Array.isArray(manifest.payload.retrieval) ? manifest.payload.retrieval.slice(0, 40) : [],
    strategies: Array.isArray(manifest.payload.strategies) ? manifest.payload.strategies.slice(0, 20) : [],
    provenance: {
      generated_by: "evomind_literature_agent_actions",
      manifest_sha256_verified_at_build: true
    }
  };
  await atomicWriteJson(resolveWorkspacePath(relative), payload);
  return { relative, payload, manifest };
}

async function createHandoff(taskId: string, metadata: Record<string, unknown>, recipient: "research_agent" | "code_agent") {
  const context = await buildAgentContext(taskId, metadata);
  const handoffId = `handoff_${actionKey(taskId, recipient, context.payload.context_id).slice(0, 24)}`;
  const envelope = {
    schema: "evomind.multi_agent.handoff.v1",
    handoff_id: handoffId,
    task_id: taskId,
    sender: "executive_supervisor",
    recipient,
    created_at: new Date().toISOString(),
    status: "queued",
    input_evidence: [{
      kind: "literature_context",
      path: context.relative,
      context_id: context.payload.context_id,
      manifest_path: context.manifest.manifestPath,
      manifest_sha256: context.manifest.manifestChecksum
    }],
    expected_output: recipient === "research_agent"
      ? "Evidence-grounded literature synthesis with explicit unsupported claims and citations."
      : "Read-only implementation context. Any code change must be returned as a gated patch candidate.",
    budget: { max_turns: 8, max_depth: 1, training_allowed: false, official_submission_allowed: false },
    deadline_condition: "Stop when evidence is insufficient and return open requirements.",
    provenance: { parent_summary_used: false, direct_manifest_reference: true },
    idempotency_key: actionKey(taskId, recipient, context.manifest.manifestChecksum)
  };
  const ledgerRelative = `workspace/tasks/${taskId}/agents/handoffs.jsonl`;
  await atomicAppendJsonl(resolveWorkspacePath(ledgerRelative), envelope, "handoff_id");
  let inboxRelative: string;
  if (recipient === "research_agent") {
    inboxRelative = `workspace/tasks/${taskId}/agents/research_agent/inbox.jsonl`;
    await atomicAppendJsonl(resolveWorkspacePath(inboxRelative), envelope, "handoff_id");
  } else {
    inboxRelative = `workspace/tasks/${taskId}/code_agent_context/literature_handoffs.jsonl`;
    await atomicAppendJsonl(resolveWorkspacePath(inboxRelative), envelope, "handoff_id");
    await atomicWriteJson(resolveWorkspacePath(`workspace/tasks/${taskId}/code_agent_context/literature_context.json`), context.payload);
  }
  return { envelope, ledgerRelative, inboxRelative, context };
}

function selectedPaper(manifest: JsonRecord, metadata: Record<string, unknown>) {
  const selectedPaperId = typeof metadata.selected_paper_id === "string" ? metadata.selected_paper_id.trim() : "";
  if (!selectedPaperId) throw new Error("selected_paper_id is required.");
  const paper = paperList(manifest).find((item) => String(item.id ?? "") === selectedPaperId);
  if (!paper) throw new Error("selected_paper_id does not exist in the current task manifest.");
  return paper;
}

function requiredClaim(metadata: Record<string, unknown>) {
  const claim = typeof metadata.claim === "string" ? metadata.claim.trim() : "";
  if (claim.length < 8) throw new Error("claim is required and must contain at least 8 characters.");
  if (claim.length > 2000) throw new Error("claim must be at most 2000 characters.");
  return claim;
}

async function createClaimBinding(taskId: string, metadata: Record<string, unknown>) {
  const manifest = await loadManifest(taskId, metadata);
  const paper = selectedPaper(manifest.payload, metadata);
  const claim = requiredClaim(metadata);
  const bindingId = `claim_binding_${actionKey(taskId, manifest.manifestChecksum, paper.id, claim).slice(0, 24)}`;
  const relative = `workspace/tasks/${taskId}/rag/claim_bindings/${bindingId}.json`;
  const payload = {
    schema: "evomind.literature.claim_binding.v1",
    binding_id: bindingId,
    task_id: taskId,
    created_at: new Date().toISOString(),
    claim,
    paper: compactPaper(paper),
    evidence_reference: {
      manifest_path: manifest.manifestPath,
      manifest_sha256: manifest.manifestChecksum,
      context_path: manifest.contextPath
    },
    review_status: "pending_independent_review",
    report_gate: "blocked_until_citation_audit_passes"
  };
  await atomicWriteJson(resolveWorkspacePath(relative), payload);
  return { relative, payload, manifest };
}

const ENGLISH_RELEVANCE_STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "had", "has", "have",
  "in", "into", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "these", "this", "those",
  "to", "using", "was", "were", "with", "within", "without"
]);

function relevanceTokens(value: string) {
  const normalized = value.toLowerCase().replace(/https?:\/\/\S+/g, " ");
  const tokens = new Set(
    (normalized.match(/[a-z0-9][a-z0-9._-]{1,}|[\u3400-\u9fff]+/g) ?? [])
      .filter((token) => /[\u3400-\u9fff]/.test(token) || !ENGLISH_RELEVANCE_STOP_WORDS.has(token))
  );
  for (const sequence of normalized.match(/[\u3400-\u9fff]{2,}/g) ?? []) {
    for (let index = 0; index < sequence.length - 1; index += 1) tokens.add(sequence.slice(index, index + 2));
  }
  return tokens;
}

function claimRelevance(claim: string, paper: JsonRecord) {
  const claimTokens = relevanceTokens(claim);
  const evidenceTokens = relevanceTokens([
    paper.title,
    paper.abstract,
    Array.isArray(paper.methods) ? paper.methods.join(" ") : "",
    Array.isArray(paper.authors) ? paper.authors.join(" ") : ""
  ].map((value) => String(value ?? "")).join(" "));
  const matched = [...claimTokens].filter((token) => evidenceTokens.has(token));
  return {
    score: claimTokens.size ? Number((matched.length / claimTokens.size).toFixed(4)) : 0,
    matched_terms: matched.slice(0, 24)
  };
}

async function verifyPaperProvenance(taskId: string, paper: JsonRecord) {
  const source = String(paper.source ?? "");
  const provenance = isRecord(paper.provenance) ? paper.provenance : {};
  const verified = provenance.verified === true && Boolean(String(provenance.source ?? ""));
  const hasExternalLocator = Boolean(paper.doi || paper.url || paper.source_url);
  let checksumStatus: "not_applicable" | "verified" | "missing" | "mismatch" = "not_applicable";
  if (source === "imported") {
    const expected = typeof provenance.checksum === "string" ? provenance.checksum : "";
    const artifactPath = typeof paper.artifact_path === "string" ? paper.artifact_path : "";
    if (!expected || !artifactPath) {
      checksumStatus = "missing";
    } else {
      const resolved = safeTaskRelativePath(taskId, artifactPath, ["literature/imports/"]);
      const bytes = await fs.readFile(resolved.absolute).catch(() => null);
      checksumStatus = bytes ? sha256(bytes) === expected ? "verified" : "mismatch" : "missing";
    }
  }
  const complete = verified
    && (["arxiv", "openalex", "crossref"].includes(source) ? hasExternalLocator : true)
    && (source !== "imported" || checksumStatus === "verified");
  return { complete, source, verified, has_external_locator: hasExternalLocator, checksum_status: checksumStatus };
}

function isOverclaim(claim: string) {
  const englishOverclaim = /\b(?:guarantees?|proves?|causes?|state[- ]of[- ]the[- ]art|sota|official\s+(?:rank|score|medal)|kaggle\s+(?:rank|score|medal))\b/i;
  const quantifiedOverclaim = /(?:significantly\s+improves?\s*\d|improves?\s*\d+(?:\.\d+)?%|显著提高\s*\d|提升\s*\d+(?:\.\d+)?%)/i;
  const chineseOverclaim = /(?:保证|证明了?|必然|因果|官方(?:排名|分数|奖牌)|金牌|银牌|铜牌)/;
  return englishOverclaim.test(claim) || quantifiedOverclaim.test(claim) || chineseOverclaim.test(claim);
}

async function latestBinding(taskId: string) {
  const absolute = await latestFile(resolveWorkspacePath(`workspace/tasks/${taskId}/rag/claim_bindings`), /^claim_binding_.*\.json$/i);
  if (!absolute) return null;
  const { payload } = await readJsonObject(absolute);
  return { payload, relative: toPosix(path.relative(resolveWorkspacePath(""), absolute)) };
}

async function runCitationAudit(taskId: string, metadata: Record<string, unknown>) {
  const explicitBinding = metadata.selected_paper_id || metadata.claim
    ? await createClaimBinding(taskId, metadata)
    : null;
  const storedBinding = explicitBinding ? null : await latestBinding(taskId);
  const binding = explicitBinding?.payload ?? storedBinding?.payload;
  const bindingPath = explicitBinding?.relative ?? storedBinding?.relative;
  if (!binding || !bindingPath) throw new Error("No claim binding exists for the selected task.");
  if (validTaskId(String(binding.task_id ?? "")) !== taskId) throw new Error("Claim binding belongs to a different task.");
  const paper = isRecord(binding.paper) ? binding.paper : null;
  const claim = String(binding.claim ?? "").trim();
  if (!paper || !claim) throw new Error("Claim binding is incomplete.");
  const evidenceReference = isRecord(binding.evidence_reference) ? binding.evidence_reference : {};
  const manifest = await loadManifest(taskId, {
    manifest_path: evidenceReference.manifest_path,
    context_path: evidenceReference.context_path
  });
  const expectedManifestChecksum = String(evidenceReference.manifest_sha256 ?? "");
  const manifestChecksumMatches = expectedManifestChecksum === manifest.manifestChecksum;
  const directPaper = paperList(manifest.payload).find((item) => String(item.id ?? "") === String(paper.id ?? ""));
  const provenance = directPaper ? await verifyPaperProvenance(taskId, directPaper) : {
    complete: false,
    source: String(paper.source ?? ""),
    verified: false,
    has_external_locator: false,
    checksum_status: "missing" as const
  };
  const relevance = directPaper ? claimRelevance(claim, directPaper) : { score: 0, matched_terms: [] as string[] };
  const overclaim = isOverclaim(claim);
  const independentlyVerifiable = ["arxiv", "openalex", "crossref", "imported"].includes(provenance.source);
  const blockers: string[] = [];
  if (!manifestChecksumMatches) blockers.push("manifest_checksum_mismatch");
  if (!directPaper) blockers.push("paper_missing_from_manifest");
  if (!provenance.complete) blockers.push("incomplete_or_invalid_provenance");
  if (overclaim) blockers.push("overclaim_requires_stronger_evidence");
  const openRequirements: string[] = [];
  if (!independentlyVerifiable) openRequirements.push("replace_internal_context_with_external_or_imported_source");
  if (relevance.score < 0.2 || relevance.matched_terms.length < 2) {
    openRequirements.push("provide_claim_specific_supporting_passage");
  }
  const status = blockers.length ? "blocked" : openRequirements.length ? "needs_evidence" : "passed";
  const auditId = `citation_audit_${actionKey(binding.binding_id, manifest.manifestChecksum, "reviewer_v1").slice(0, 24)}`;
  const relative = `workspace/tasks/${taskId}/rag/citation_audits/${auditId}.json`;
  const payload = {
    schema: "evomind.independent_reviewer.citation_audit.v1",
    audit_id: auditId,
    task_id: taskId,
    reviewer: "independent_reviewer",
    reviewer_mode: "read_only_direct_evidence",
    created_at: new Date().toISOString(),
    status,
    gate: status === "passed" ? "citation_gate_passed" : "citation_gate_blocked",
    claim,
    paper_id: String(paper.id ?? ""),
    binding_path: bindingPath,
    evidence_checked: {
      manifest_path: manifest.manifestPath,
      manifest_sha256: manifest.manifestChecksum,
      manifest_checksum_matches: manifestChecksumMatches,
      paper_loaded_directly_from_manifest: Boolean(directPaper),
      provenance,
      relevance,
      overclaim
    },
    blockers,
    open_requirements: openRequirements,
    conclusion: status === "passed"
      ? "The claim is eligible for report use with this citation and its current wording."
      : "The claim is not eligible for an evidence-backed report until the listed requirements are resolved.",
    validated_memory_eligible: status === "passed"
  };
  await atomicWriteJson(resolveWorkspacePath(relative), payload);
  await atomicAppendJsonl(
    resolveWorkspacePath(`workspace/tasks/${taskId}/agents/reviewer_agent/reviews.jsonl`),
    payload,
    "audit_id"
  );
  return { relative, payload, bindingPath };
}

export function isLiteratureAgentAction(action: string) {
  return AGENT_ACTIONS.has(action);
}

export async function executeLiteratureAgentAction(input: LiteratureActionInput): Promise<LiteratureActionResult> {
  const taskId = validTaskId(input.taskId);
  const metadata = input.metadata ?? {};
  if (input.action === "rag_build_agent_context") {
    const context = await buildAgentContext(taskId, metadata);
    return {
      artifactPath: context.relative,
      message: `Versioned literature context built from ${context.payload.papers.length} papers.`,
      metadata: { manifest_path: context.manifest.manifestPath, manifest_sha256: context.manifest.manifestChecksum },
      result: { context_id: context.payload.context_id, context_path: context.relative, paper_count: context.payload.papers.length }
    };
  }
  if (input.action === "rag_send_research_agent" || input.action === "rag_send_code_agent") {
    const recipient = input.action === "rag_send_research_agent" ? "research_agent" : "code_agent";
    const handoff = await createHandoff(taskId, metadata, recipient);
    return {
      artifactPath: handoff.inboxRelative,
      message: `Literature evidence queued for ${recipient}; no training or code execution was started.`,
      metadata: { handoff_id: handoff.envelope.handoff_id, handoff_ledger: handoff.ledgerRelative, context_path: handoff.context.relative },
      result: { handoff: handoff.envelope, handoff_path: handoff.inboxRelative, context_path: handoff.context.relative }
    };
  }
  if (input.action === "rag_bind_report_claim") {
    const binding = await createClaimBinding(taskId, metadata);
    return {
      artifactPath: binding.relative,
      message: "Report claim bound to the selected paper and queued for independent citation review.",
      metadata: { binding_id: binding.payload.binding_id, paper_id: binding.payload.paper && (binding.payload.paper as JsonRecord).id, review_status: binding.payload.review_status },
      result: { claim_binding: binding.payload, claim_binding_path: binding.relative }
    };
  }
  if (input.action === "rag_request_citation_audit") {
    const audit = await runCitationAudit(taskId, metadata);
    return {
      artifactPath: audit.relative,
      message: `Independent citation review completed: ${audit.payload.status}.`,
      metadata: { audit_id: audit.payload.audit_id, status: audit.payload.status, gate: audit.payload.gate, binding_path: audit.bindingPath },
      result: { citation_audit: audit.payload, citation_audit_path: audit.relative, reviewer_status: audit.payload.status }
    };
  }
  throw new Error(`Unsupported literature agent action: ${input.action}`);
}

export async function loadLiteratureStateByTask(taskIdInput: string) {
  const taskId = validTaskId(taskIdInput);
  const taskRoot = resolveWorkspacePath(`workspace/tasks/${taskId}`);
  const manifest = await latestValidLiteratureManifest(path.join(taskRoot, "rag"), taskId);
  if (!manifest) {
    return { present: false, task_id: taskId, papers: [], claim_audit: [], citation_audits: [] };
  }
  const manifestAbsolute = manifest.absolutePath;
  const payload = manifest.payload as JsonRecord;
  if (validTaskId(String(payload.task_id ?? "")) !== taskId) {
    return { present: false, task_id: taskId, error: "manifest_task_mismatch", papers: [], claim_audit: [], citation_audits: [] };
  }
  const agentContextAbsolute = await latestFile(path.join(taskRoot, "rag", "agent_contexts"), /^literature_context_.*\.json$/i);
  const bindingAbsolute = await latestFile(path.join(taskRoot, "rag", "claim_bindings"), /^claim_binding_.*\.json$/i);
  const auditAbsolute = await latestFile(path.join(taskRoot, "rag", "citation_audits"), /^citation_audit_.*\.json$/i);
  const handoffText = await fs.readFile(path.join(taskRoot, "agents", "handoffs.jsonl"), "utf-8").catch(() => "");
  const handoffs = handoffText.split(/\r?\n/).filter(Boolean).flatMap((line) => {
    try {
      const row = JSON.parse(line) as unknown;
      return isRecord(row) && row.task_id === taskId ? [row] : [];
    } catch {
      return [];
    }
  });
  const readOptional = async (absolute: string | null) => absolute ? (await readJsonObject(absolute)).payload : null;
  const [agentContext, claimBinding, citationAudit] = await Promise.all([
    readOptional(agentContextAbsolute),
    readOptional(bindingAbsolute),
    readOptional(auditAbsolute)
  ]);
  const manifestPath = toPosix(path.relative(resolveWorkspacePath(""), manifestAbsolute));
  return {
    ...payload,
    present: true,
    task_id: taskId,
    manifest_path: manifestPath,
    papers: paperList(payload).slice(0, 40),
    claim_audit: Array.isArray(payload.claim_audit) ? payload.claim_audit.slice(0, 40) : [],
    agent_context: agentContext,
    agent_context_path: agentContextAbsolute ? toPosix(path.relative(resolveWorkspacePath(""), agentContextAbsolute)) : null,
    handoffs: handoffs.slice(-20),
    latest_handoff: handoffs.at(-1) ?? null,
    claim_binding: claimBinding,
    claim_binding_path: bindingAbsolute ? toPosix(path.relative(resolveWorkspacePath(""), bindingAbsolute)) : null,
    citation_audits: citationAudit ? [citationAudit] : [],
    latest_citation_audit: citationAudit,
    citation_audit_path: auditAbsolute ? toPosix(path.relative(resolveWorkspacePath(""), auditAbsolute)) : null
  };
}

export async function loadLiteratureStateForAllTasks() {
  const tasksRoot = resolveWorkspacePath("workspace/tasks");
  const entries = await fs.readdir(tasksRoot, { withFileTypes: true }).catch(() => []);
  const rows = await Promise.all(entries
    .filter((entry) => entry.isDirectory() && /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(entry.name))
    .map(async (entry) => [entry.name, await loadLiteratureStateByTask(entry.name)] as const));
  return Object.fromEntries(rows.filter(([, value]) => value.present));
}

export async function writeLiteratureActionReceipt(taskIdInput: string, action: string, metadata: Record<string, unknown>) {
  const taskId = validTaskId(taskIdInput);
  const relative = `workspace/runtime/literature_action_${stamp()}.json`;
  await atomicWriteJson(resolveWorkspacePath(relative), {
    schema: "evomind.literature_action.v1",
    action,
    task_id: taskId,
    metadata,
    created_at: new Date().toISOString(),
    claim_boundary: "literature_context_only_not_official_score_rank_or_medal"
  });
  return relative;
}
