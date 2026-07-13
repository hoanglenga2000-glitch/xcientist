import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { normalizeTaskId, resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const figureNames = [
  "validation_curve.svg",
  "metric_comparison.svg",
  "feature_importance.svg",
  "missing_values.svg",
  "target_distribution.svg",
  "experiment_lineage.svg"
];

async function listFigures(taskId: string) {
  const dir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "figures"));
  const files = await fs.readdir(dir).catch(() => []);
  return files
    .filter((file) => /\.(png|svg)$/i.test(file))
    .map((file) => ({
      name: file,
      path: toRelativePath(path.join(dir, file)),
      type: file.endsWith(".png") ? "png" : "svg"
    }));
}

export async function GET(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  return NextResponse.json({ ok: true, task_id: taskId, figures: await listFigures(taskId), expected: figureNames });
}
