import { NextResponse } from "next/server";
import { readGpuJob } from "@/lib/server/gpu-ssh-gateway";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  const record = await readGpuJob(jobId);
  if (!record) return NextResponse.json({ ok: false, error: "GPU job not found" }, { status: 404 });
  return NextResponse.json(record);
}
