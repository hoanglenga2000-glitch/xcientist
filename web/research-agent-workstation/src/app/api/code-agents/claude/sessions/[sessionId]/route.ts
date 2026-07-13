import { NextResponse } from "next/server";
import { readClaudeSession } from "@/lib/server/claude-agent-sessions";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  const record = await readClaudeSession(sessionId);
  if (!record) return NextResponse.json({ ok: false, error: "Claude session not found" }, { status: 404 });
  return NextResponse.json(record);
}
