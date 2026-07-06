import { NextResponse } from "next/server";
import { readGpuJob } from "@/lib/server/gpu-ssh-gateway";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { jobId: string } }) {
  const record = await readGpuJob(params.jobId);
  if (!record) return NextResponse.json({ ok: false, error: "GPU job not found" }, { status: 404 });
  return NextResponse.json(record);
}
