import { NextResponse } from "next/server";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { latestExperimentPathWithAnyArtifacts, normalizeTaskId, resolveWorkspacePath } from "@/lib/server/paths";
import { serializeEvidence } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

const FALLBACK_ARTIFACTS = [
  ["metrics.json", "Metrics"],
  ["oof_predictions.csv", "OOF predictions"],
  ["submission.csv", "Submission"],
  ["agent_trace.json", "Agent trace"],
  ["artifact_manifest.json", "Artifact manifest"],
  ["orchestrator_run.json", "Orchestrator run"],
  ["gate_engine.json", "Gate engine"],
  ["experiment_graph.json", "Experiment graph"],
  ["experiment_memory.json", "Experiment memory"],
  ["report.md", "Research report"],
  ["workstation_report.md", "Workstation report"],
] as const;

async function fileHash(filePath: string) {
  const payload = await fs.readFile(filePath);
  return createHash("sha256").update(payload).digest("hex");
}

async function filesystemEvidence(taskId: string) {
  const latest = await latestExperimentPathWithAnyArtifacts(taskId, FALLBACK_ARTIFACTS.map(([name]) => name));
  if (!latest) return [];
  const root = resolveWorkspacePath(latest);
  const items = await Promise.all(FALLBACK_ARTIFACTS.map(async ([name, label]) => {
    const artifactPath = path.join(latest, name).replaceAll("\\", "/");
    const absolutePath = path.join(root, name);
    const stat = await fs.stat(absolutePath).catch(() => null);
    if (!stat?.isFile()) return null;
    return {
      id: `${taskId}_filesystem_${name.replace(/[^a-zA-Z0-9]+/g, "_")}`,
      task_id: taskId,
      run_id: null,
      label,
      artifact_path: artifactPath,
      hash: await fileHash(absolutePath),
      source: "filesystem_fallback",
      claim_binding: "artifact_based_workflow",
      created_at: stat.mtime.toISOString()
    };
  }));
  return items.filter((item): item is NonNullable<typeof item> => Boolean(item));
}

export async function GET(_request: Request, { params }: { params: { taskId: string } }) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(params.taskId);
  const evidence = await prisma.evidence.findMany({ where: { taskId }, orderBy: { createdAt: "desc" } });
  const serialized = evidence.map(serializeEvidence);
  return NextResponse.json({
    ok: true,
    task_id: taskId,
    evidence: serialized.length ? serialized : await filesystemEvidence(taskId)
  });
}
