import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId, resolveWorkspacePath, stamp, toRelativePath, workspaceRoot, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";
import type { LiteratureChunk, LiteratureClaimAudit, LiteraturePaper, LiteratureSearchResponse, LiteratureStrategy } from "@/lib/api/types";

export const dynamic = "force-dynamic";

const SAFE_SEARCH_DIRS = ["docs", "reports", "prompts", "configs", "references", "examples", "workspace/tasks", "workspace/workstation_runs"] as const;
const TEXT_EXTENSIONS = new Set([".md", ".txt", ".json", ".yaml", ".yml", ".csv"]);
const MAX_FILE_BYTES = 512_000;
const MAX_LOCAL_FILES = 220;
const CLAIM_BOUNDARY_TEXT = "文献检索结果只能作为研究上下文与策略候选，不能直接声称官方 Kaggle 提分、排名或奖牌。";
const CLAIM_BOUNDARY_SENTINEL = "literature_context_only_not_official_score_rank_or_medal";

const seedPapers: LiteraturePaper[] = [
  {
    id: "seed-mlevolve",
    title: "MLEvolve: Self-Evolving Machine Learning Engineering Agents",
    type: "paper",
    year: "2026",
    venue: "arXiv",
    score: 0.74,
    task: "MLE-Bench / Kaggle",
    exp: "Search Controller",
    status: "seed",
    source: "seed",
    abstract: "Progressive search controller, retrospective memory, and adaptive code generation for MLE tasks.",
    methods: ["progressive search", "retrospective memory", "adaptive code generation", "ensemble"],
    risks: ["benchmark overclaim", "public leaderboard overfit"],
    url: null
  },
  {
    id: "seed-xcientist",
    title: "XCIENTIST-style Research Harness",
    type: "paper",
    year: "2026",
    venue: "Research Harness",
    score: 0.7,
    task: "Validation / Audit",
    exp: "Claim Audit",
    status: "seed",
    source: "seed",
    abstract: "Hypothesis, implementation contract, risk checklist, ablation plan, and claim drift audit for auditable research agents.",
    methods: ["validation contract", "claim audit", "ablation", "risk checklist"],
    risks: ["claim drift", "insufficient evidence"],
    url: null
  },
  {
    id: "seed-lightgbm",
    title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
    type: "paper",
    year: "2017",
    venue: "NeurIPS",
    score: 0.68,
    task: "Tabular Kaggle",
    exp: "Model Family",
    status: "seed",
    source: "seed",
    abstract: "Histogram-based gradient boosting with leaf-wise growth and categorical/feature handling useful for tabular machine learning.",
    methods: ["gradient boosting", "feature engineering", "categorical handling", "early stopping"],
    risks: ["overfit", "cv public mismatch"],
    url: null
  }
];

function cleanText(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function hasMojibakeRiskText(value: string) {
  return value.includes("\uFFFD") || /[\u00c3\u00c2\u00e2]/.test(value);
}

function tokenize(value: string) {
  return Array.from(new Set(cleanText(value).toLowerCase().match(/[\p{L}\p{N}_+-]{2,}/gu) ?? []));
}

function xmlText(entry: string, tag: string) {
  const match = entry.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return match ? decodeXml(match[1]).trim() : "";
}

function decodeXml(value: string) {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'");
}

function inferMethods(text: string) {
  const lower = text.toLowerCase();
  const tags = [
    ["lightgbm", "LightGBM"],
    ["xgboost", "XGBoost"],
    ["catboost", "CatBoost"],
    ["ensemble", "Ensemble"],
    ["stacking", "Stacking"],
    ["blend", "Blending"],
    ["feature", "Feature Engineering"],
    ["cross validation", "Cross Validation"],
    ["oof", "OOF"],
    ["transformer", "Transformer"],
    ["attention", "Attention"],
    ["time series", "Time Series"],
    ["tabular", "Tabular"],
    ["neural", "Neural Network"],
    ["ablation", "Ablation"],
    ["claim", "Claim Audit"]
  ] as const;
  return tags.filter(([needle]) => lower.includes(needle)).map(([, label]) => label).slice(0, 6);
}

function inferRisks(text: string) {
  const lower = text.toLowerCase();
  const tags = [
    ["leak", "data leakage"],
    ["overfit", "overfitting"],
    ["public", "CV-public gap"],
    ["private", "leaderboard gap"],
    ["drift", "claim drift"],
    ["timeout", "training timeout"],
    ["insufficient", "insufficient evidence"],
    ["schema", "submission schema"],
    ["missing", "missing evidence"]
  ] as const;
  return tags.filter(([needle]) => lower.includes(needle)).map(([, label]) => label).slice(0, 5);
}

function scoreText(queryTokens: string[], text: string, title = "") {
  const lower = `${title} ${text}`.toLowerCase();
  if (!queryTokens.length) return 0.1;
  let score = 0;
  for (const token of queryTokens) {
    if (lower.includes(token)) score += title.toLowerCase().includes(token) ? 3 : 1;
  }
  const methodBoost = inferMethods(lower).length * 0.45;
  const riskBoost = inferRisks(lower).length * 0.2;
  return Math.min(0.99, (score / Math.max(3, queryTokens.length * 2)) + methodBoost / 10 + riskBoost / 20);
}

function splitChunks(text: string, maxLen = 760) {
  const cleaned = cleanText(text);
  if (cleaned.length <= maxLen) return [cleaned];
  const chunks: string[] = [];
  for (let i = 0; i < cleaned.length; i += maxLen) {
    chunks.push(cleaned.slice(i, i + maxLen));
    if (chunks.length >= 10) break;
  }
  return chunks;
}

async function walkSafeFiles(root: string, relativeRoot: string, files: string[]) {
  if (files.length >= MAX_LOCAL_FILES) return;
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (files.length >= MAX_LOCAL_FILES) return;
    if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === ".next") continue;
    const absolute = path.join(root, entry.name);
    const relative = path.join(relativeRoot, entry.name);
    if (entry.isDirectory()) {
      await walkSafeFiles(absolute, relative, files);
      continue;
    }
    const ext = path.extname(entry.name).toLowerCase();
    if (!TEXT_EXTENSIONS.has(ext)) continue;
    const stat = await fs.stat(absolute).catch(() => null);
    if (!stat?.isFile() || stat.size > MAX_FILE_BYTES) continue;
    files.push(relative.replaceAll("\\", "/"));
  }
}

async function collectLocalDocuments() {
  const files: string[] = [];
  for (const dir of SAFE_SEARCH_DIRS) {
    const absolute = resolveWorkspacePath(dir);
    await walkSafeFiles(absolute, dir, files);
    if (files.length >= MAX_LOCAL_FILES) break;
  }
  return Promise.all(
    files.map(async (relativePath) => {
      const absolute = resolveWorkspacePath(relativePath);
      const text = await fs.readFile(absolute, "utf-8").catch(() => "");
      return { relativePath, text };
    })
  );
}

async function fetchArxiv(query: string, maxResults: number): Promise<LiteraturePaper[]> {
  const params = new URLSearchParams({
    search_query: `all:${query}`,
    start: "0",
    max_results: String(Math.min(maxResults, 8)),
    sortBy: "relevance",
    sortOrder: "descending"
  });
  const response = await fetch(`https://export.arxiv.org/api/query?${params.toString()}`, {
    headers: { "User-Agent": "research-agent-workstation/0.1 literature-rag" },
    signal: AbortSignal.timeout(10_000)
  }).catch(() => null);
  if (!response?.ok) return [];
  const xml = await response.text();
  const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];
  return entries.map((entry, index) => {
    const idUrl = xmlText(entry, "id");
    const title = cleanText(xmlText(entry, "title"));
    const summary = cleanText(xmlText(entry, "summary"));
    const published = xmlText(entry, "published");
    const authors = Array.from(entry.matchAll(/<author>[\s\S]*?<name>([\s\S]*?)<\/name>[\s\S]*?<\/author>/g)).map((match) => decodeXml(match[1]).trim()).slice(0, 6);
    const year = published ? published.slice(0, 4) : "";
    const arxivId = idUrl.split("/abs/").pop() ?? `arxiv-${index + 1}`;
    return {
      id: `arxiv-${arxivId.replace(/[^\w.-]/g, "_")}`,
      title: title || `arXiv result ${index + 1}`,
      type: "paper",
      year,
      venue: "arXiv",
      score: 0.5,
      task: "dynamic search",
      exp: "RAG",
      status: "retrieved",
      source: "arxiv" as const,
      url: idUrl,
      artifact_path: idUrl,
      abstract: summary,
      methods: inferMethods(`${title} ${summary}`),
      risks: inferRisks(`${title} ${summary}`),
      authors
    };
  });
}

function papersFromLocalDocs(docs: Array<{ relativePath: string; text: string }>, queryTokens: string[], taskId: string): LiteraturePaper[] {
  return docs
    .map((doc) => {
      const heading = doc.text.match(/^#\s+(.+)$/m)?.[1]?.trim();
      const title = heading || path.basename(doc.relativePath);
      const score = scoreText(queryTokens, doc.text.slice(0, 6000), title);
      return {
        id: `local-${Buffer.from(doc.relativePath).toString("base64url").slice(0, 18)}`,
        title,
        type: doc.relativePath.includes("report") ? "report" : doc.relativePath.includes("prompt") ? "prompt" : "local_doc",
        year: "local",
        venue: doc.relativePath.split("/")[0] ?? "workspace",
        score,
        task: taskId,
        exp: doc.relativePath.includes("mle") ? "MLE-Bench" : "Research OS",
        status: score > 0.45 ? "matched" : "indexed",
        source: "local" as const,
        url: null,
        artifact_path: doc.relativePath,
        abstract: cleanText(doc.text.slice(0, 620)),
        methods: inferMethods(doc.text),
        risks: inferRisks(doc.text)
      };
    })
    .filter((paper) => paper.score > 0.12)
    .sort((a, b) => b.score - a.score)
    .slice(0, 24);
}

function buildChunks(papers: LiteraturePaper[], queryTokens: string[]): LiteratureChunk[] {
  const chunks: LiteratureChunk[] = [];
  for (const paper of papers) {
    const body = paper.abstract || paper.title;
    for (const [index, chunk] of splitChunks(body).entries()) {
      chunks.push({
        id: `${paper.id}-chunk-${index + 1}`,
        rank: 0,
        chunk: chunk.slice(0, 260),
        score: Number(scoreText(queryTokens, chunk, paper.title).toFixed(3)),
        source: `${paper.title}${paper.year ? ` (${paper.year})` : ""}`,
        page: paper.source === "arxiv" ? "abstract" : "local",
        artifact: paper.artifact_path || paper.url || paper.id,
        used: paper.score > 0.5 ? "accepted" : paper.score > 0.28 ? "review" : "candidate",
        paper_id: paper.id,
        method_tags: paper.methods ?? [],
        risk_tags: paper.risks ?? []
      });
    }
  }
  return chunks
    .sort((a, b) => b.score - a.score)
    .slice(0, 32)
    .map((chunk, index) => ({ ...chunk, rank: index + 1 }));
}

function buildStrategies(papers: LiteraturePaper[]): LiteratureStrategy[] {
  const strategies = new Map<string, LiteratureStrategy>();
  for (const paper of papers) {
    for (const method of paper.methods ?? []) {
      if (strategies.has(method)) continue;
      strategies.set(method, {
        strategy: method,
        paper_id: paper.id,
        family: method.includes("LightGBM") || method.includes("XGBoost") || method.includes("CatBoost") ? "Tabular Boosting" : method.includes("Transformer") ? "Sequence Modeling" : "Research OS",
        exp: paper.exp,
        benefit: method.includes("Validation") || method.includes("OOF") ? "提升可靠性" : "提升搜索效率/候选质量",
        risk: (paper.risks ?? [])[0] ?? "需消融验证"
      });
    }
  }
  return Array.from(strategies.values()).slice(0, 8);
}

function buildClaimAudit(papers: LiteraturePaper[], contextPath: string): LiteratureClaimAudit[] {
  const rows = papers.slice(0, 6).map((paper) => ({
    claim: paper.methods?.[0] ? `${paper.methods[0]} may inform next experiment` : "literature relevance supports research context",
    paper: paper.id,
    exp: paper.exp,
    artifact: paper.artifact_path || contextPath,
    status: paper.risks?.length ? "needs audit" : "supported"
  }));
  rows.push({
    claim: "文献命中不等于 Kaggle 官方提分或奖牌",
    paper: "-",
    exp: "claim boundary",
    artifact: contextPath,
    status: "blocked overclaim"
  });
  return rows;
}

function buildContextMarkdown(input: {
  taskId: string;
  query: string;
  papers: LiteraturePaper[];
  chunks: LiteratureChunk[];
  strategies: LiteratureStrategy[];
  claimAudit: LiteratureClaimAudit[];
}) {
  return [
    `# RAG Context for ${input.taskId}`,
    "",
    `- query: ${input.query}`,
    `- generated_at: ${new Date().toISOString()}`,
    `- boundary: ${CLAIM_BOUNDARY_TEXT}`,
    `- boundary_sentinel: ${CLAIM_BOUNDARY_SENTINEL}`,
    "",
    "## Top Papers",
    "| score | source | title | methods | risks | artifact |",
    "| --- | --- | --- | --- | --- | --- |",
    ...input.papers.slice(0, 12).map((paper) => `| ${paper.score.toFixed(3)} | ${paper.source} | ${paper.title.replaceAll("|", " ")} | ${(paper.methods ?? []).join(", ")} | ${(paper.risks ?? []).join(", ")} | ${paper.artifact_path ?? paper.url ?? "-"} |`),
    "",
    "## Retrieval Chunks",
    "| rank | score | source | chunk | used |",
    "| --- | --- | --- | --- | --- |",
    ...input.chunks.slice(0, 12).map((chunk) => `| ${chunk.rank} | ${chunk.score.toFixed(3)} | ${chunk.source.replaceAll("|", " ")} | ${chunk.chunk.replaceAll("|", " ")} | ${chunk.used} |`),
    "",
    "## Reusable Strategies",
    "| strategy | paper | family | benefit | risk |",
    "| --- | --- | --- | --- | --- |",
    ...input.strategies.map((strategy) => `| ${strategy.strategy} | ${strategy.paper_id} | ${strategy.family} | ${strategy.benefit} | ${strategy.risk} |`),
    "",
    "## Claim Audit",
    "| claim | paper | artifact | status |",
    "| --- | --- | --- | --- |",
    ...input.claimAudit.map((row) => `| ${row.claim.replaceAll("|", " ")} | ${row.paper} | ${row.artifact} | ${row.status} |`)
  ].join("\n");
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const taskId = normalizeTaskId(String(body.task_id ?? body.taskId ?? "playground_series_s6e6"));
  const task = await prisma.task.findUnique({ where: { id: taskId } }).catch(() => null);
  const query = cleanText(String(body.query ?? `${task?.name ?? taskId} ${task?.taskType ?? ""} ${task?.metric ?? ""} kaggle modeling validation ensemble`));
  const includeArxiv = body.include_arxiv !== false;
  const maxResults = Number.isFinite(Number(body.max_results)) ? Math.max(5, Math.min(40, Number(body.max_results))) : 18;
  const queryTokens = tokenize(`${query} ${task?.taskType ?? ""} ${task?.metric ?? ""}`);

  const [localDocs, arxivPapers] = await Promise.all([
    collectLocalDocuments(),
    includeArxiv ? fetchArxiv(query, Math.min(8, maxResults)).catch(() => []) : Promise.resolve([])
  ]);
  const localPapers = papersFromLocalDocs(localDocs, queryTokens, taskId);
  const scoredArxiv = arxivPapers.map((paper) => ({
    ...paper,
    score: scoreText(queryTokens, `${paper.title} ${paper.abstract ?? ""}`, paper.title)
  }));
  const scoredSeeds = seedPapers.map((paper) => ({
    ...paper,
    score: Math.max(paper.score, scoreText(queryTokens, `${paper.title} ${paper.abstract ?? ""}`, paper.title))
  }));
  const papers = [...localPapers, ...scoredArxiv, ...scoredSeeds]
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults);
  const retrieval = buildChunks(papers, queryTokens);
  const strategies = buildStrategies(papers).map((strategy) => ({
    ...strategy,
    benefit: strategy.strategy.includes("Validation") || strategy.strategy.includes("OOF")
      ? "提升验证可靠性与证据质量"
      : "提升搜索效率与候选策略质量",
    risk: hasMojibakeRiskText(strategy.risk) ? "\u9700\u8981\u6d88\u878d\u9a8c\u8bc1" : strategy.risk
  }));
  const contextTokens = Math.min(8192, Math.round((papers.map((paper) => paper.abstract ?? paper.title).join(" ").length + retrieval.map((chunk) => chunk.chunk).join(" ").length) / 4));
  const stampId = stamp();
  const contextPath = `workspace/tasks/${taskId}/rag/context_${stampId}.md`;
  const manifestPath = `workspace/tasks/${taskId}/rag/context_${stampId}.json`;
  const claimAudit = buildClaimAudit(papers, contextPath).map((row) => row.status === "blocked overclaim"
    ? { ...row, claim: "文献命中不等于 Kaggle 官方提分、排名或奖牌" }
    : row);
  const contextMarkdown = buildContextMarkdown({ taskId, query, papers, chunks: retrieval, strategies, claimAudit })
    .replace(
      /^- boundary: .*$/m,
      `- boundary: ${CLAIM_BOUNDARY_TEXT}`
    );
  await writeTextArtifact(contextPath, contextMarkdown);
  const manifest: LiteratureSearchResponse = {
    ok: true,
    task_id: taskId,
    query,
    generated_at: new Date().toISOString(),
    source_counts: {
      local: localPapers.length,
      arxiv: scoredArxiv.length,
      seed: scoredSeeds.length
    },
    metrics: {
      paper_count: papers.length,
      chunk_count: retrieval.length,
      citation_confidence: papers.length ? Math.round((retrieval.filter((chunk) => chunk.used === "accepted").length / Math.max(1, retrieval.length)) * 100) : 0,
      context_tokens: contextTokens,
      max_tokens: 8192,
      local_documents_indexed: localDocs.length,
      arxiv_results: scoredArxiv.length
    },
    papers,
    retrieval,
    strategies,
    claim_audit: claimAudit,
    context_markdown: contextMarkdown,
    context_path: contextPath,
    manifest_path: manifestPath,
    used_fallback: !scoredArxiv.length
  };
  await writeJsonArtifact(manifestPath, manifest);
  await prisma.evidence.create({
    data: {
      id: `rag_${stampId}_${Math.random().toString(36).slice(2, 7)}`,
      taskId,
      label: "RAG literature context",
      artifactPath: contextPath,
      source: "LiteratureKnowledge",
      claimBinding: "research_context"
    }
  }).catch(() => undefined);
  await logAction({
    action: "literature_search",
    taskId,
    message: `RAG literature search completed: ${papers.length} papers, ${retrieval.length} chunks.`,
    artifactPath: contextPath,
    metadata: {
      query,
      manifest_path: manifestPath,
      source_counts: manifest.source_counts,
      citation_confidence: manifest.metrics.citation_confidence
    }
  });

  return NextResponse.json(manifest);
}
