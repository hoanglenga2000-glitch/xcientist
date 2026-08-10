import { NextResponse } from "next/server";
import { listHpcJobLineage } from "@/lib/server/hpc-job-lineage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const jobs = await listHpcJobLineage();
  return NextResponse.json({
    schema: "evomind.hpc_job_lineage.v1",
    jobs,
    active_job_count: jobs.filter((job) => job.identity_kind === "execution_job" && !["COMPLETED", "FAILED", "RETIRED"].includes(String(job.status))).length,
    generated_at: new Date().toISOString(),
  });
}
