import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const workspaceRoot = path.resolve(process.cwd(), "..", "..");

export async function POST(request: Request, { params }: { params: { taskId: string } }) {
  const taskId = normalizeTaskId(params.taskId);
  const body = await request.json().catch(() => ({}));
  const patchText = String(body.patch_diff ?? body.patch ?? "");
  const sourceAgent = String(body.source_agent ?? "external");

  if (!patchText.trim()) {
    return NextResponse.json({ ok: false, error: "patch_diff is required" }, { status: 400 });
  }

  const patchDir = path.join(workspaceRoot, "workspace", "tasks", taskId, "code", "patches");
  await fs.mkdir(patchDir, { recursive: true });
  const files = await fs.readdir(patchDir).catch(() => []);
  const patchId = `patch_${String(files.filter((name) => name.endsWith(".diff")).length + 1).padStart(4, "0")}`;
  const patchPath = path.join(patchDir, `${patchId}.diff`);
  await fs.writeFile(patchPath, patchText, "utf-8");
  await fs.writeFile(
    path.join(patchDir, `${patchId}.json`),
    JSON.stringify(
      {
        patch_id: patchId,
        source_agent: sourceAgent,
        review_status: "pending",
        applied_at: null,
        rollback_path: null
      },
      null,
      2
    ),
    "utf-8"
  );

  const patch_path = path.relative(workspaceRoot, patchPath);
  await logAction({
    action: "import_agent_patch",
    taskId,
    message: `Patch imported: ${patch_path}`,
    artifactPath: patch_path,
    metadata: { patch_id: patchId, source_agent: sourceAgent }
  });

  return NextResponse.json({
    ok: true,
    task_id: taskId,
    patch_id: patchId,
    patch_path
  });
}
