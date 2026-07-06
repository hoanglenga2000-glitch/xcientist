import { NextResponse } from "next/server";
import { testGpuConnection } from "@/lib/server/gpu-ssh-gateway";

export const dynamic = "force-dynamic";

export async function POST() {
  return NextResponse.json(await testGpuConnection());
}
