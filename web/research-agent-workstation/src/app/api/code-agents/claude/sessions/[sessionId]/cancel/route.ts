import { NextResponse } from "next/server";
import { cancelClaudeSession } from "@/lib/server/claude-agent-sessions";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  const record = await cancelClaudeSession(sessionId);
  if (!record) return NextResponse.json({ ok: false, error: "Claude session not found" }, { status: 404 });
  return NextResponse.json(record);
}
