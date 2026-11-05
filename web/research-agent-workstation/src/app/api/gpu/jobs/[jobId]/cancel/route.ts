import { NextResponse } from "next/server";
import { cancelGpuJob } from "@/lib/server/gpu-ssh-gateway";
import { legacyGpuAccess, legacyGpuDenied } from "@/lib/server/legacy-gpu-access";

export const dynamic = "force-dynamic";

export async function POST(request: Request, { params }: { params: Promise<{ jobId: string }> }) {
  const access = legacyGpuAccess(request);
  if (!access.ok) return legacyGpuDenied(access);
  const { jobId } = await params;
  return NextResponse.json(await cancelGpuJob(jobId));
}
