import { NextResponse } from "next/server";
import { handleWorkstationAction, type WorkstationActionPayload } from "@/lib/server/workstation-actions";
import { HpcExecutionContractError } from "@/lib/server/hpc-execution-contract";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as WorkstationActionPayload;
  const action = body.action ?? "unknown";

  try {
    return NextResponse.json(await handleWorkstationAction(body));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown action error";
    if (error instanceof HpcExecutionContractError) {
      return NextResponse.json({
        ok: false,
        action,
        code: error.code,
        error: message,
        missing_fields: error.missingFields,
      }, { status: error.statusCode });
    }
    return NextResponse.json({ ok: false, action, error: message }, { status: 500 });
  }
}
