import os from "node:os";
import { promisify } from "node:util";
import { execFile } from "node:child_process";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);

function toNumber(value: string | undefined) {
  const parsed = Number(value?.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

export async function GET() {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      [
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
      ],
      { timeout: 10_000, windowsHide: true, maxBuffer: 64 * 1024 }
    );
    const fields = stdout.trim().split(/\r?\n/, 1)[0]?.split(",").map((item) => item.trim()) ?? [];
    if (fields.length < 6) throw new Error("nvidia-smi returned an incomplete row");
    const gpu = {
      name: fields[0],
      memory_total_mib: toNumber(fields[1]),
      memory_used_mib: toNumber(fields[2]),
      memory_free_mib: toNumber(fields[3]),
      utilization_percent: toNumber(fields[4]),
      temperature_c: toNumber(fields[5])
    };
    return NextResponse.json({
      ok: true,
      status: gpu.name.includes("RTX 4060") ? "ready" : "unexpected_gpu",
      gpu,
      system_memory: {
        available_gib: Number((os.freemem() / 1024 ** 3).toFixed(2)),
        total_gib: Number((os.totalmem() / 1024 ** 3).toFixed(2))
      },
      compute_backend: "local_gpu",
      remote_compute_used: false,
      official_submission: "disabled",
      sampled_at: new Date().toISOString()
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        status: "unavailable",
        error: error instanceof Error ? error.message : "Local GPU telemetry unavailable",
        compute_backend: "local_gpu",
        remote_compute_used: false,
        official_submission: "disabled",
        sampled_at: new Date().toISOString()
      },
      { status: 503 }
    );
  }
}
