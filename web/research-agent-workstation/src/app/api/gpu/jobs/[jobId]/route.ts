import { NextResponse } from "next/server";
import { readGpuJob } from "@/lib/server/gpu-ssh-gateway";
import { legacyGpuAccess, legacyGpuDenied } from "@/lib/server/legacy-gpu-access";

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: Promise<{ jobId: string }> }) {
  const access = legacyGpuAccess(request);
  if (!access.ok) return legacyGpuDenied(access);
  const { jobId } = await params;
  const record = await readGpuJob(jobId);
  if (!record) return NextResponse.json({ ok: false, error: "GPU job not found" }, { status: 404 });
  return NextResponse.json(record);
}
