import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId, resolveWorkspacePath, stamp, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";
import type { LiteratureChunk, LiteratureClaimAudit, LiteraturePaper, LiteratureSearchResponse, LiteratureStrategy } from "@/lib/api/types";

export const dynamic = "force-dynamic";

const SAFE_SEARCH_DIRS = ["docs", "reports", "prompts", "configs", "references", "examples", "workspace/tasks", "workspace/workstation_runs"] as const;
const TEXT_EXTENSIONS = new Set([".md", ".txt", ".json", ".yaml", ".yml", ".csv"]);
const MAX_FILE_BYTES = 512_000;
const MAX_LOCAL_FILES = 220;
const CLAIM_BOUNDARY_TEXT = "文献检索结果只能作为研究上下文与策略候选，不能直接声称官方 Kaggle 提分、排名或奖牌。";
const CLAIM_BOUNDARY_SENTINEL = "literature_context_only_not_official_score_rank_or_medal";
const EXTERNAL_USER_AGENT = "EvoMind/1.0 (research-agent-workstation; literature provenance)";

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
    source: "internal",
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
    source: "internal",
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
    source: "internal",
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

type SearchVariant = {
  query: string;
  methodTerms: string[];
  domainTokens: string[];
  foundation: boolean;
};

const METHOD_SEARCH_SPECS = [
  { term: "catboost", label: "CatBoost", foundation: "CatBoost unbiased boosting categorical features" },
  { term: "lightgbm", label: "LightGBM", foundation: "LightGBM highly efficient gradient boosting decision tree" },
  { term: "xgboost", label: "XGBoost", foundation: "XGBoost scalable tree boosting system" },
  { term: "transformer", label: "Transformer", foundation: "Transformer attention is all you need" }
] as const;

const GENERIC_RELEVANCE_TOKENS = new Set([
  "paper", "papers", "literature", "study", "studies", "research", "method", "methods",
  "model", "models", "machine", "learning", "prediction", "predicting", "classification",
  "regression", "forecasting", "analysis", "dataset", "data", "using", "based", "comparison",
  "kaggle", "validation", "experiment"
]);

function normalizeAcademicQuery(value: string, taskId: string) {
  const escapedTask = taskId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/[-_]+/g, "[-_ ]");
  const taskPrefix = new RegExp(`^(?:请|帮我|我想|我要|麻烦|请你)?\\s*为\\s*${escapedTask}\\s*`, "i");
  let query = cleanText(value).replace(taskPrefix, "")
    .replace(/(?:房价|房屋价格|住宅价格|房地产价格)(?:预测|估值)?/gi, "house price prediction")
    .replace(/乳腺癌(?:诊断|预测|分类)?/gi, "breast cancer classification")
    .replace(/时间序列(?:预测|建模)?/gi, "time series forecasting")
    .replace(/图像(?:识别|分类)?/gi, "image classification")
    .replace(/文本(?:识别|分类)?/gi, "text classification")
    .replace(/表格(?:数据|建模)?/gi, "tabular machine learning")
    .replace(/生存分析/gi, "survival analysis")
    .replace(/异常检测/gi, "anomaly detection")
    .replace(/推荐系统/gi, "recommender systems")
    .replace(/\bhouse(?:[-_]prices| prices)\b/gi, "house price prediction")
    .replace(/(?:，|,|。|；|;)?\s*(?:请)?(?:给出|提供|输出|生成|总结|说明|并给出|然后给出).*$/gi, "")
    .replace(/(?:，|,|。|；|;)?\s*(?:不|不要|无需|无须|禁止)\s*(?:启动|开始|执行|进行)?\s*(?:任何)?(?:训练|实验|提交).*$/gi, "")
    .replace(/^(?:请|帮我|我想|我要|麻烦|请你|能否|可以)?\s*(?:为\s*)?/gi, "")
    .replace(/(?:检索|搜索|查找|查询|找出|找|核验|验证|交叉核验|实际检索|真实检索|参考)(?:并|和|与)?/gi, " ")
    .replace(/(?:真实|相关|学术)?\s*(?:参考文献|论文|文献)(?:列表|资料|证据)?/gi, " ")
    .replace(/^(?:关于|有关|针对)\s*/gi, "")
    .replace(/(?<=[A-Za-z0-9])\s*(?:和|与|以及|及)\s*(?=[A-Za-z0-9])/g, " ")
    .replace(/[：:，,。.!！?？；;、]+/g, " ");
  query = cleanText(query);
  if (normalizeTaskId(taskId) === "siim-isic-melanoma-classification") {
    const siimDefault = "SIIM ISIC melanoma classification dermoscopy patient level cross validation";
    if (/^(?:ai|ai\s*science|ai科学|人工智能|科研|科学)$/i.test(query)) {
      query = siimDefault;
    } else if (!/\b(?:siim|isic|melanoma|dermoscopy|skin)\b/i.test(query)) {
      query = cleanText(query + " " + siimDefault);
    }
  }
  if (normalizeTaskId(taskId) === "house_prices" && !/\bhouse\s+price\b/i.test(query)) {
    query = cleanText(`${query} house price prediction`);
  }
  return query || cleanText(value);
}

function buildSearchVariants(query: string): SearchVariant[] {
  const lower = query.toLowerCase();
  const methods = METHOD_SEARCH_SPECS.filter((spec) => lower.includes(spec.term));
  const methodTerms = new Set<string>(methods.map((spec) => spec.term));
  const domainTokens = tokenize(query).filter((token) => !methodTerms.has(token) && !GENERIC_RELEVANCE_TOKENS.has(token));
  const domainPhrase = tokenize(query).filter((token) => !methodTerms.has(token)).join(" ");
  const variants: SearchVariant[] = [];

  if (methods.length && domainPhrase) {
    for (const method of methods.slice(0, 3)) {
      variants.push({
        query: cleanText(`${method.label} ${domainPhrase}`),
        methodTerms: [method.term],
        domainTokens,
        foundation: false
      });
    }
  } else {
    variants.push({ query, methodTerms: methods.map((method) => method.term), domainTokens, foundation: false });
  }

  for (const method of methods) {
    if (variants.length >= 4) break;
    variants.push({
      query: method.foundation,
      methodTerms: [method.term],
      domainTokens: tokenize(method.foundation).filter((token) => token !== method.term && !GENERIC_RELEVANCE_TOKENS.has(token)),
      foundation: true
    });
  }

  const seen = new Set<string>();
  return variants.filter((variant) => {
    const key = variant.query.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, 4);
}

function paperMatchesVariant(paper: LiteraturePaper, variant: SearchVariant) {
  const text = searchablePaperText(paper).toLowerCase();
  const title = paper.title.toLowerCase();
  const methodMatch = !variant.methodTerms.length || variant.methodTerms.some((term) => text.includes(term));
  if (!methodMatch) return false;

  const coreDomain = variant.domainTokens.filter((token) => !GENERIC_RELEVANCE_TOKENS.has(token));
  if (variant.foundation) {
    const normalizedTitle = title.replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const canonicalTitleMatch = variant.methodTerms.some((term) => normalizedTitle.startsWith(term))
      || (variant.methodTerms.includes("transformer") && normalizedTitle.includes("attention is all you need"));
    const anchorHits = coreDomain.filter((token) => title.includes(token)).length;
    const requiredAnchorHits = Math.max(1, Math.ceil(coreDomain.length * 0.75));
    return canonicalTitleMatch && anchorHits >= requiredAnchorHits;
  }
  if (!coreDomain.length) return methodMatch;
  const strongDomainTokens = coreDomain.filter((token) =>
    ["siim", "isic", "melanoma", "dermoscopy", "skin"].includes(token)
  );
  if (strongDomainTokens.length && !strongDomainTokens.some((token) => text.includes(token))) {
    return false;
  }
  const hits = coreDomain.filter((token) => text.includes(token)).length;
  return hits >= Math.min(2, coreDomain.length);
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

async function fetchArxiv(query: string, maxResults: number): Promise<SourceFetchResult> {
  const params = new URLSearchParams({
    search_query: tokenize(query).length >= 3 ? `ti:"${query.replace(/["\\]/g, " ")}"` : `all:${query}`,
    start: "0",
    max_results: String(Math.min(maxResults, 8)),
    sortBy: "relevance",
    sortOrder: "descending"
  });
  let response: Response | null = null;
  let requestError: unknown = null;
  try {
    response = await fetch(`https://export.arxiv.org/api/query?${params.toString()}`, {
    headers: { "User-Agent": "research-agent-workstation/0.1 literature-rag" },
    signal: AbortSignal.timeout(10_000)
    });
  } catch (error) {
    requestError = error;
  }
  if (!response?.ok) return { papers: [], error: { source: "arxiv", error: requestError instanceof Error ? requestError.message : response ? `HTTP ${response.status}` : "request failed", retryable: true } };
  const xml = await response.text();
  const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];
  return { papers: entries.map((entry, index) => {
    const idUrl = xmlText(entry, "id");
    const title = cleanText(xmlText(entry, "title"));
    const summary = cleanText(xmlText(entry, "summary"));
    const published = xmlText(entry, "published");
    const authors = Array.from(entry.matchAll(/<author>[\s\S]*?<name>([\s\S]*?)<\/name>[\s\S]*?<\/author>/g)).map((match) => decodeXml(match[1]).trim()).slice(0, 6);
    const year = published ? published.slice(0, 4) : "";
    const arxivId = idUrl.split("/abs/").pop() ?? `arxiv-${index + 1}`;
    const canonicalArxivId = arxivId.replace(/v\d+$/i, "");
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
      url: `https://arxiv.org/abs/${arxivId}`,
      source_url: `https://arxiv.org/abs/${arxivId}`,
      doi: `10.48550/arXiv.${canonicalArxivId}`,
      artifact_path: `https://arxiv.org/abs/${arxivId}`,
      abstract: summary,
      methods: inferMethods(`${title} ${summary}`),
      risks: inferRisks(`${title} ${summary}`),
      authors,
      provenance: { verified: true, source: "arXiv", retrieved_at: new Date().toISOString() }
    };
  }) };
}

function searchablePaperText(paper: LiteraturePaper) {
  return [paper.title, paper.abstract, paper.authors?.join(" "), paper.year, paper.venue, paper.doi].filter(Boolean).join(" ");
}

type SourceFetchResult = {
  papers: LiteraturePaper[];
  error?: { source: string; error: string; retryable: boolean };
};

async function fetchJson(url: string, source: string): Promise<{ data: Record<string, unknown> | null; error?: SourceFetchResult["error"] }> {
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json", "User-Agent": EXTERNAL_USER_AGENT },
      signal: AbortSignal.timeout(12_000)
    });
    if (!response.ok) {
      return { data: null, error: { source, error: `HTTP ${response.status}`, retryable: response.status >= 500 || response.status === 429 } };
    }
    return { data: await response.json() as Record<string, unknown> };
  } catch (error) {
    return { data: null, error: { source, error: error instanceof Error ? error.message : "request failed", retryable: true } };
  }
}

function openAlexAbstract(invertedIndex: unknown) {
  if (!invertedIndex || typeof invertedIndex !== "object" || Array.isArray(invertedIndex)) return "";
  const rows: Array<{ position: number; word: string }> = [];
  for (const [word, positions] of Object.entries(invertedIndex as Record<string, unknown>)) {
    if (!Array.isArray(positions)) continue;
    for (const position of positions) if (typeof position === "number") rows.push({ position, word });
  }
  return rows.sort((a, b) => a.position - b.position).map((row) => row.word).join(" ").slice(0, 4000);
}

function stripMarkup(value: string) {
  return cleanText(value.replace(/<[^>]+>/g, " "));
}

async function fetchOpenAlex(query: string, maxResults: number): Promise<SourceFetchResult> {
  const params = new URLSearchParams({ search: query, "per-page": String(Math.min(maxResults, 12)), select: "id,doi,title,publication_year,primary_location,authorships,abstract_inverted_index,open_access" });
  const result = await fetchJson(`https://api.openalex.org/works?${params.toString()}`, "openalex");
  if (!result.data) return { papers: [], error: result.error };
  const rows = Array.isArray(result.data.results) ? result.data.results : [];
  return {
    papers: rows.map((row, index) => {
      const item = row as Record<string, unknown>;
      const location = item.primary_location as Record<string, unknown> | null;
      const source = location?.source as Record<string, unknown> | null;
      const authorships = Array.isArray(item.authorships) ? item.authorships : [];
      const authors = authorships.map((entry) => (entry as Record<string, unknown>).author as Record<string, unknown> | null)
        .map((author) => typeof author?.display_name === "string" ? author.display_name : "")
        .filter(Boolean).slice(0, 8);
      const doi = typeof item.doi === "string" ? item.doi.replace(/^https?:\/\/doi.org\//i, "") : null;
      const title = typeof item.title === "string" ? cleanText(item.title) : `OpenAlex work ${index + 1}`;
      const abstract = openAlexAbstract(item.abstract_inverted_index);
      const url = typeof location?.landing_page_url === "string" ? location.landing_page_url : doi ? `https://doi.org/${doi}` : typeof item.id === "string" ? item.id : null;
      return {
        id: `openalex-${String(item.id ?? index + 1).split("/").pop()?.replace(/[^\w.-]/g, "_") ?? index + 1}`,
        title,
        type: "paper",
        year: typeof item.publication_year === "number" ? String(item.publication_year) : "",
        venue: typeof source?.display_name === "string" ? source.display_name : "OpenAlex",
        score: 0.5,
        task: "dynamic search",
        exp: "RAG",
        status: "retrieved",
        source: "openalex" as const,
        url,
        source_url: url,
        doi,
        artifact_path: url,
        abstract,
        methods: inferMethods(`${title} ${abstract}`),
        risks: inferRisks(`${title} ${abstract}`),
        authors,
        provenance: { verified: true, source: "OpenAlex", retrieved_at: new Date().toISOString() }
      };
    })
  };
}

async function fetchCrossref(query: string, maxResults: number): Promise<SourceFetchResult> {
  const params = new URLSearchParams({ "query.bibliographic": query, rows: String(Math.min(maxResults, 12)), select: "DOI,title,author,published,container-title,URL,abstract,type" });
  const result = await fetchJson(`https://api.crossref.org/works?${params.toString()}`, "crossref");
  if (!result.data) return { papers: [], error: result.error };
  const message = result.data.message as Record<string, unknown> | null;
  const rows = Array.isArray(message?.items) ? message.items : [];
  return {
    papers: rows.map((row, index) => {
      const item = row as Record<string, unknown>;
      const title = Array.isArray(item.title) && typeof item.title[0] === "string" ? cleanText(item.title[0]) : `Crossref work ${index + 1}`;
      const authors = Array.isArray(item.author) ? item.author.map((entry) => {
        const author = entry as Record<string, unknown>;
        return [author.given, author.family].filter((part): part is string => typeof part === "string" && Boolean(part)).join(" ");
      }).filter(Boolean).slice(0, 8) : [];
      const published = item.published as Record<string, unknown> | null;
      const dateParts = Array.isArray(published?.["date-parts"]) ? published?.["date-parts"]?.[0] : [];
      const doi = typeof item.DOI === "string" ? item.DOI : null;
      const url = typeof item.URL === "string" ? item.URL : doi ? `https://doi.org/${doi}` : null;
      const abstract = typeof item.abstract === "string" ? stripMarkup(item.abstract) : "";
      return {
        id: `crossref-${(doi ?? String(index + 1)).replace(/[^\w.-]/g, "_")}`,
        title,
        type: typeof item.type === "string" ? item.type : "paper",
        year: Array.isArray(dateParts) && typeof dateParts[0] === "number" ? String(dateParts[0]) : "",
        venue: Array.isArray(item["container-title"]) && typeof item["container-title"][0] === "string" ? item["container-title"][0] : "Crossref",
        score: 0.45,
        task: "dynamic search",
        exp: "RAG",
        status: "retrieved",
        source: "crossref" as const,
        url,
        source_url: url,
        doi,
        artifact_path: url,
        abstract,
        methods: inferMethods(`${title} ${abstract}`),
        risks: inferRisks(`${title} ${abstract}`),
        authors,
        provenance: { verified: true, source: "Crossref", retrieved_at: new Date().toISOString() }
      };
    })
  };
}

async function loadImportedPapers(taskId: string): Promise<LiteraturePaper[]> {
  const root = resolveWorkspacePath(`workspace/tasks/${taskId}/literature/imports`);
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const records = await Promise.all(entries.filter((entry) => entry.isFile() && entry.name.endsWith(".json")).map(async (entry) => {
    const payload = await fs.readFile(path.join(root, entry.name), "utf-8").then((text) => JSON.parse(text) as Record<string, unknown>).catch(() => null);
    return payload;
  }));
  return records.filter((record): record is Record<string, unknown> => Boolean(record?.paper && typeof record.paper === "object"))
    .map((record) => record.paper as LiteraturePaper);
}

function dedupePapers(papers: LiteraturePaper[]) {
  const seen = new Set<string>();
  return papers.filter((paper) => {
    const normalizedTitle = cleanText(paper.title).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const key = paper.source === "imported"
      ? `imported:${paper.id}`
      : normalizedTitle.length >= 12
        ? `title:${normalizedTitle}`
        : (paper.doi || paper.url || paper.title).toLowerCase().replace(/\s+/g, " ");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
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
        risks: inferRisks(doc.text),
        provenance: { verified: true, source: "EvoMind workspace", retrieved_at: new Date().toISOString() }
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
    "| score | source | title | authors | DOI | methods | risks | artifact |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ...input.papers.slice(0, 12).map((paper) => `| ${paper.score.toFixed(3)} | ${paper.source} | ${paper.title.replaceAll("|", " ")} | ${(paper.authors ?? []).join(", ").replaceAll("|", " ")} | ${paper.doi ?? "-"} | ${(paper.methods ?? []).join(", ")} | ${(paper.risks ?? []).join(", ")} | ${paper.artifact_path ?? paper.url ?? "-"} |`),
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
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(taskId)) {
    return NextResponse.json({ ok: false, error: "invalid task_id" }, { status: 400 });
  }
  const task = await prisma.task.findUnique({ where: { id: taskId } }).catch(() => null);
  const originalQuery = cleanText(String(body.original_query ?? body.query ?? `${task?.name ?? taskId} ${task?.taskType ?? ""} ${task?.metric ?? ""} kaggle modeling validation ensemble`));
  const query = normalizeAcademicQuery(String(body.query ?? originalQuery), taskId);
  const includeArxiv = body.include_arxiv !== false;
  const includeOpenAlex = body.include_openalex !== false;
  const includeCrossref = body.include_crossref !== false;
  const includeInternal = body.include_internal === true;
  const maxResults = Number.isFinite(Number(body.max_results)) ? Math.max(5, Math.min(40, Number(body.max_results))) : 18;
  const queryTokens = tokenize(`${query} ${task?.taskType ?? ""} ${task?.metric ?? ""}`);
  const searchVariants = buildSearchVariants(query);

  const [localDocs, importedPapers, sourceRuns] = await Promise.all([
    collectLocalDocuments(),
    loadImportedPapers(taskId),
    Promise.all(searchVariants.map(async (variant) => {
      const [arxiv, openalex, crossref] = await Promise.all([
        includeArxiv ? fetchArxiv(variant.query, Math.min(6, maxResults)) : Promise.resolve({ papers: [] } as SourceFetchResult),
        includeOpenAlex ? fetchOpenAlex(variant.query, Math.min(6, maxResults)) : Promise.resolve({ papers: [] } as SourceFetchResult),
        includeCrossref ? fetchCrossref(variant.query, Math.min(6, maxResults)) : Promise.resolve({ papers: [] } as SourceFetchResult)
      ]);
      return { variant, arxiv, openalex, crossref };
    }))
  ]);
  const localPapers = papersFromLocalDocs(localDocs, queryTokens, taskId).filter((paper) => !paper.artifact_path?.includes("/literature/imports/"));
  const scoredImported = importedPapers.map((paper) => ({ ...paper, score: scoreText(queryTokens, searchablePaperText(paper), paper.title) }));
  const rawExternal: LiteraturePaper[] = [];
  const acceptedArxiv: LiteraturePaper[] = [];
  const acceptedOpenAlex: LiteraturePaper[] = [];
  const acceptedCrossref: LiteraturePaper[] = [];
  const sourceErrors: Array<{ source: string; error: string; retryable: boolean }> = [];
  for (const run of sourceRuns) {
    const variantTokens = tokenize(run.variant.query);
    for (const [source, result, accepted] of [
      ["arxiv", run.arxiv, acceptedArxiv],
      ["openalex", run.openalex, acceptedOpenAlex],
      ["crossref", run.crossref, acceptedCrossref]
    ] as const) {
      rawExternal.push(...result.papers);
      for (const paper of result.papers) {
        if (!paperMatchesVariant(paper, run.variant)) continue;
        accepted.push({
          ...paper,
          score: Math.max(
            scoreText(queryTokens, searchablePaperText(paper), paper.title),
            scoreText(variantTokens, searchablePaperText(paper), paper.title)
          ),
          status: "relevance_verified"
        });
      }
      if (result.error) {
        sourceErrors.push({
          source,
          error: `${result.error.error} [query=${run.variant.query}]`,
          retryable: result.error.retryable
        });
      }
    }
  }
  const scoredArxiv = dedupePapers(acceptedArxiv);
  const scoredOpenAlex = dedupePapers(acceptedOpenAlex);
  const scoredCrossref = dedupePapers(acceptedCrossref);
  const scoredInternal = seedPapers.map((paper) => ({
    ...paper,
    score: Math.max(paper.score, scoreText(queryTokens, `${paper.title} ${paper.abstract ?? ""}`, paper.title))
  })).filter(() => includeInternal);
  const sourceTrust: Record<string, number> = {
    imported: 5,
    arxiv: 5,
    openalex: 5,
    crossref: 5,
    local: 2,
    internal: 1
  };
  const relevantPapers = dedupePapers(
    [...scoredArxiv, ...scoredOpenAlex, ...scoredCrossref, ...(includeInternal ? localPapers : []), ...scoredInternal]
      .sort((a, b) => b.score - a.score || (sourceTrust[b.source] ?? 0) - (sourceTrust[a.source] ?? 0))
  ).slice(0, maxResults);
  const pinnedImports = dedupePapers(scoredImported.sort((a, b) => b.score - a.score));
  const papers = dedupePapers([...relevantPapers, ...pinnedImports]);
  const acceptedExternalCount = relevantPapers.filter((paper) => ["arxiv", "openalex", "crossref"].includes(paper.source)).length;
  const rawExternalCount = rawExternal.length;
  const filteredExternalCount = Math.max(0, rawExternalCount - acceptedExternalCount);
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
  const relevanceStatus = acceptedExternalCount > 0
    ? "passed"
    : rawExternalCount > 0
      ? "relevance_filtered"
      : "no_external_results";
  const manifest: LiteratureSearchResponse = {
    ok: acceptedExternalCount > 0,
    task_id: taskId,
    query,
    original_query: originalQuery,
    search_queries: searchVariants.map((variant) => variant.query),
    generated_at: new Date().toISOString(),
    source_counts: {
      local: localPapers.length,
      imported: scoredImported.length,
      arxiv: scoredArxiv.length,
      openalex: scoredOpenAlex.length,
      crossref: scoredCrossref.length,
      internal: scoredInternal.length
    },
    metrics: {
      paper_count: papers.length,
      chunk_count: retrieval.length,
      citation_confidence: papers.length ? Math.round((retrieval.filter((chunk) => chunk.used === "accepted").length / Math.max(1, retrieval.length)) * 100) : 0,
      context_tokens: contextTokens,
      max_tokens: 8192,
      local_documents_indexed: localDocs.length,
      arxiv_results: scoredArxiv.length,
      openalex_results: scoredOpenAlex.length,
      crossref_results: scoredCrossref.length,
      imported_documents: scoredImported.length,
      raw_external_results: rawExternalCount,
      relevance_filtered_results: filteredExternalCount
    },
    papers,
    retrieval,
    strategies,
    claim_audit: claimAudit,
    context_markdown: contextMarkdown,
    context_path: contextPath,
    manifest_path: manifestPath,
    used_fallback: acceptedExternalCount === 0,
    source_errors: sourceErrors,
    relevance: {
      status: relevanceStatus,
      raw_external: rawExternalCount,
      accepted_external: acceptedExternalCount,
      filtered_external: filteredExternalCount
    },
    integrity: {
      external_verified: acceptedExternalCount,
      imported: scoredImported.length,
      internal_context: scoredInternal.length,
      fabricated: 0
    },
    error: acceptedExternalCount > 0
      ? undefined
      : sourceErrors.length && rawExternalCount === 0
        ? "external_sources_unavailable"
        : relevanceStatus
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
      original_query: originalQuery,
      search_queries: manifest.search_queries,
      manifest_path: manifestPath,
      source_counts: manifest.source_counts,
      citation_confidence: manifest.metrics.citation_confidence,
      relevance: manifest.relevance
    }
  });

  return NextResponse.json(manifest);
}
