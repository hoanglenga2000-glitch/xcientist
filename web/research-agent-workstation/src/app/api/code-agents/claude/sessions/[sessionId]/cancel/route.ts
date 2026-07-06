import { NextResponse } from "next/server";
import { cancelClaudeSession } from "@/lib/server/claude-agent-sessions";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: { sessionId: string } }) {
  const record = await cancelClaudeSession(params.sessionId);
  if (!record) return NextResponse.json({ ok: false, error: "Claude session not found" }, { status: 404 });
  return NextResponse.json(record);
}
