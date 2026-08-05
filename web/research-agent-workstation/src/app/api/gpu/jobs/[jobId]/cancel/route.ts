import { NextResponse } from "next/server";
import { cancelGpuJob } from "@/lib/server/gpu-ssh-gateway";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return NextResponse.json(await cancelGpuJob(jobId));
}
