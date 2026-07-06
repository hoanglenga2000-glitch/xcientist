import { NextResponse } from "next/server";
import { runDeepSeekSmoke } from "@/lib/server/deepseek-provider";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const prompt = typeof body.prompt === "string" ? body.prompt : undefined;
  const result = await runDeepSeekSmoke(prompt);
  return NextResponse.json(result, { status: result.ok || result.status === "not_configured" ? 200 : 502 });
}
