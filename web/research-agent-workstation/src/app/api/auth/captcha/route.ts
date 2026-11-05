import { NextResponse } from "next/server";

import { issueLoginCaptcha } from "@/lib/server/login-captcha";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(issueLoginCaptcha(), { headers: { "Cache-Control": "no-store" } });
}
