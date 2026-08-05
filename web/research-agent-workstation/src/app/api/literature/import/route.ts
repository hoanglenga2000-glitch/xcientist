import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId, resolveWorkspacePath, stamp, writeJsonArtifact } from "@/lib/server/paths";
import type { LiteratureImportResponse, LiteraturePaper } from "@/lib/api/types";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const MAX_BYTES = 20 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([".pdf", ".md", ".markdown", ".txt", ".json"]);

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || (process.platform === "win32" ? "C:\\codex-python\\python.exe" : "python3");
}

async function atomicWrite(target: string, data: string | Uint8Array) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(temporary, data);
  await fs.rename(temporary, target);
}

async function extractPdf(filePath: string) {
  const script = [
    "import json, sys",
    "from pypdf import PdfReader",
    "reader = PdfReader(sys.argv[1])",
    "pages = [(page.extract_text() or '') for page in reader.pages]",
    "meta = reader.metadata or {}",
    "print(json.dumps({'pages': len(pages), 'text': '\\n\\n'.join(pages), 'title': str(meta.get('/Title') or '')}, ensure_ascii=False))"
  ].join("; ");
  const result = await execFileAsync(pythonExecutable(), ["-c", script, filePath], {
    timeout: 60_000,
    maxBuffer: 8 * 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" }
  });
  return JSON.parse(result.stdout) as { pages: number; text: string; title: string };
}

function cleanText(value: string) {
  return value.split(String.fromCharCode(0)).join("").replace(/\r/g, "").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function firstHeading(text: string) {
  return text.split(/\n+/).map((line) => line.trim()).find((line) => line.length >= 8 && line.length <= 240 && !/^(abstract|introduction|keywords?)$/i.test(line)) ?? "";
}

function extractDoi(text: string) {
  return text.match(/\b10\.\d{4,9}\/[-._;()/:A-Z0-9]+/i)?.[0]?.replace(/[.,;]+$/, "") ?? null;
}

export async function POST(request: Request) {
  const form = await request.formData().catch(() => null);
  const file = form?.get("file");
  if (!(file instanceof File)) return NextResponse.json({ ok: false, error: "file is required" }, { status: 400 });
  if (file.size <= 0 || file.size > MAX_BYTES) return NextResponse.json({ ok: false, error: `file must be between 1 byte and ${MAX_BYTES} bytes` }, { status: 413 });

  const taskId = normalizeTaskId(String(form?.get("task_id") ?? "playground_series_s6e6"));
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(taskId)) {
    return NextResponse.json({ ok: false, error: "invalid task_id" }, { status: 400 });
  }
  const extension = path.extname(file.name).toLowerCase();
  if (!ALLOWED_EXTENSIONS.has(extension)) return NextResponse.json({ ok: false, error: "supported files: PDF, Markdown, TXT, JSON" }, { status: 415 });

  const bytes = Buffer.from(await file.arrayBuffer());
  const checksum = createHash("sha256").update(bytes).digest("hex");
  const importId = `${stamp()}_${checksum.slice(0, 12)}`;
  const importRoot = resolveWorkspacePath(`workspace/tasks/${taskId}/literature/imports`);
  const sourceName = `${importId}${extension === ".markdown" ? ".md" : extension}`;
  const sourcePath = path.join(importRoot, sourceName);
  await atomicWrite(sourcePath, bytes);

  let extractedText = "";
  let pageCount: number | null = null;
  let pdfTitle = "";
  try {
    if (extension === ".pdf") {
      const extracted = await extractPdf(sourcePath);
      extractedText = cleanText(extracted.text).slice(0, 1_500_000);
      pageCount = extracted.pages;
      pdfTitle = cleanText(extracted.title);
    } else {
      extractedText = cleanText(bytes.toString("utf-8")).slice(0, 1_500_000);
    }
  } catch (error) {
    await fs.rm(sourcePath, { force: true }).catch(() => undefined);
    return NextResponse.json({ ok: false, error: `document extraction failed: ${error instanceof Error ? error.message : "unknown error"}` }, { status: 422 });
  }

  if (!extractedText) return NextResponse.json({ ok: false, error: "document contains no extractable text" }, { status: 422 });
  const suppliedTitle = String(form?.get("title") ?? "").trim();
  const title = suppliedTitle || pdfTitle || firstHeading(extractedText) || path.basename(file.name, extension);
  const suppliedDoi = String(form?.get("doi") ?? "").trim();
  const doi = suppliedDoi || extractDoi(`${title}\n${extractedText}`);
  const sourceUrl = String(form?.get("source_url") ?? "").trim() || (doi ? `https://doi.org/${doi}` : null);
  const authors = String(form?.get("authors") ?? "").split(/[;,]/).map((item) => item.trim()).filter(Boolean).slice(0, 12);
  const year = String(form?.get("year") ?? "").trim() || extractedText.match(/\b(19|20)\d{2}\b/)?.[0] || "imported";
  const textRelative = `workspace/tasks/${taskId}/literature/imports/${importId}.txt`;
  const sourceRelative = `workspace/tasks/${taskId}/literature/imports/${sourceName}`;
  const manifestRelative = `workspace/tasks/${taskId}/literature/imports/${importId}.json`;
  await atomicWrite(resolveWorkspacePath(textRelative), extractedText);
  const paper: LiteraturePaper = {
    id: `imported-${checksum.slice(0, 16)}`,
    title,
    type: "paper",
    year,
    venue: "User import",
    score: 1,
    task: taskId,
    exp: "Imported literature",
    status: "imported_indexed",
    source: "imported",
    url: sourceUrl,
    source_url: sourceUrl,
    doi,
    artifact_path: sourceRelative,
    abstract: extractedText.slice(0, 900),
    methods: [],
    risks: [],
    authors,
    provenance: { verified: true, source: "user_import", retrieved_at: new Date().toISOString(), checksum }
  };
  const manifest = {
    schema: "evomind.literature_import.v1",
    task_id: taskId,
    imported_at: new Date().toISOString(),
    original_name: file.name,
    content_type: file.type || null,
    bytes: file.size,
    sha256: checksum,
    pages: pageCount,
    source_artifact: sourceRelative,
    text_artifact: textRelative,
    paper,
    provenance: { source: "user_import", extraction: extension === ".pdf" ? "pypdf" : "utf8", checksum }
  };
  await writeJsonArtifact(manifestRelative, manifest);
  await prisma.task.upsert({ where: { id: taskId }, update: {}, create: { id: taskId, name: taskId.replaceAll("_", " "), taskType: "research_task", status: "runtime_ready", taskDir: `workspace/tasks/${taskId}` } }).catch(() => undefined);
  await prisma.evidence.create({
    data: {
      id: `literature_import_${checksum.slice(0, 20)}`,
      taskId,
      label: `Imported literature: ${title}`,
      artifactPath: sourceRelative,
      hash: checksum,
      source: "UserImport",
      claimBinding: "literature_source"
    }
  }).catch(() => undefined);
  await logAction({ action: "literature_import", taskId, message: `Literature imported and indexed: ${title}`, artifactPath: manifestRelative, metadata: { source_artifact: sourceRelative, text_artifact: textRelative, sha256: checksum, pages: pageCount, doi } });
  const response: LiteratureImportResponse = { ok: true, task_id: taskId, paper, source_artifact: sourceRelative, text_artifact: textRelative, manifest_path: manifestRelative, indexed: true };
  return NextResponse.json(response);
}
