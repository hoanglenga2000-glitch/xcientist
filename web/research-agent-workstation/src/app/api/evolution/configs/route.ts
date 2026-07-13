import { NextResponse } from "next/server";
import { listEvolutionConfigs } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

/**
 * List the real evolution task configs (configs/evolution/*.json) so the UI task
 * picker matches the engine's actual task set. Read-only; no training, no secrets.
 */
export async function GET() {
  try {
    const configs = await listEvolutionConfigs();
    return NextResponse.json({
      ok: true,
      count: configs.length,
      configs,
      config_dir: "configs/evolution"
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution configs listing failed";
    return NextResponse.json({ ok: false, count: 0, configs: [], config_dir: "configs/evolution", error: message }, { status: 400 });
  }
}
