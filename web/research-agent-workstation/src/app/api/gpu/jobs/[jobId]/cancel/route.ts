import { NextResponse } from "next/server";
import { cancelGpuJob } from "@/lib/server/gpu-ssh-gateway";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: { jobId: string } }) {
  return NextResponse.json(await cancelGpuJob(params.jobId));
}
