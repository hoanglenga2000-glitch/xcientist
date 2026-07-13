import { NextRequest, NextResponse } from "next/server";

import {
  isAllowedMutationSource,
  isLoopbackHostHeader,
  normalizeTaskId
} from "@/lib/security/request-boundary";

const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function errorResponse(status: number, error: string) {
  return NextResponse.json({ ok: false, error }, { status });
}

export function middleware(request: NextRequest) {
  const host = request.headers.get("host");
  if (!isLoopbackHostHeader(host)) {
    return errorResponse(403, "The workstation API is available on loopback hosts only");
  }

  if (
    MUTATING_METHODS.has(request.method)
    && !isAllowedMutationSource(
      request.headers.get("origin"),
      host,
      request.headers.get("sec-fetch-site")
    )
  ) {
    return errorResponse(403, "Cross-origin workstation mutations are not allowed");
  }

  const taskPrefix = "/api/tasks/";
  if (request.nextUrl.pathname.startsWith(taskPrefix)) {
    const taskSegment = request.nextUrl.pathname.slice(taskPrefix.length).split("/", 1)[0];
    try {
      normalizeTaskId(taskSegment);
    } catch {
      return errorResponse(400, "Invalid task ID");
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: "/api/:path*"
};
