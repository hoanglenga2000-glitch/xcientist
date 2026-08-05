import { promises as fs } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import path from "node:path";
import { NextResponse } from "next/server";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId, stamp, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

async function atomicWrite(target: string, content: string) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = path.join(path.dirname(target), `.${path.basename(target)}.${randomUUID()}.tmp`);
  await fs.writeFile(temporary, content, "utf-8");
  await fs.rename(temporary, target).catch(async (error) => {
    await fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  });
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(taskId) || taskId === "." || taskId === "..") {
    return NextResponse.json({ ok: false, error: "Invalid task_id." }, { status: 400 });
  }
  const body = await request.json().catch(() => ({}));
  const patchText = String(body.patch_diff ?? body.patch ?? "");
  const sourceAgent = String(body.source_agent ?? "external");

  if (!patchText.trim()) {
    return NextResponse.json({ ok: false, error: "patch_diff is required" }, { status: 400 });
  }
  if (Buffer.byteLength(patchText, "utf-8") > 5 * 1024 * 1024) {
    return NextResponse.json({ ok: false, error: "patch_diff exceeds the 5 MiB local limit" }, { status: 413 });
  }

  const patchDir = path.join(workspaceRoot, "workspace", "tasks", taskId, "code", "patches");
  await fs.mkdir(patchDir, { recursive: true });
  const patchId = `patch_${stamp()}_${randomUUID().replaceAll("-", "").slice(0, 8)}`;
  const patchPath = path.join(patchDir, `${patchId}.diff`);
  await atomicWrite(patchPath, patchText);
  const patchSha256 = createHash("sha256").update(patchText).digest("hex");
  await atomicWrite(
    path.join(patchDir, `${patchId}.json`),
    JSON.stringify(
      {
        patch_id: patchId,
        source_agent: sourceAgent,
        review_status: "pending",
        applied_at: null,
        rollback_path: null,
        patch_sha256: patchSha256
      },
      null,
      2
    )
  );

  const patch_path = path.relative(workspaceRoot, patchPath);
  await logAction({
    action: "import_agent_patch",
    taskId,
    message: `Patch imported: ${patch_path}`,
    artifactPath: patch_path,
    metadata: { patch_id: patchId, source_agent: sourceAgent, patch_sha256: patchSha256 }
  });

  return NextResponse.json({
    ok: true,
    task_id: taskId,
    patch_id: patchId,
    patch_path,
    patch_sha256: patchSha256
  });
}
