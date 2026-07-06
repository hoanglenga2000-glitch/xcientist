import { NextResponse } from "next/server";
import { handleWorkstationAction, type WorkstationActionPayload } from "@/lib/server/workstation-actions";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as WorkstationActionPayload;
  const action = body.action ?? "unknown";

  try {
    return NextResponse.json(await handleWorkstationAction(body));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown action error";
    return NextResponse.json({ ok: false, action, error: message }, { status: 500 });
  }
}
