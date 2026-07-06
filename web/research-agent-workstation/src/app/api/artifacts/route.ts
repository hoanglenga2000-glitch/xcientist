import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const allowedFigurePath = /^workspace[\\/]+tasks[\\/]+[^\\/]+[\\/]+reports[\\/]+figures[\\/]+[^\\/]+\.(svg|png|jpe?g|webp)$/i;

const contentTypes: Record<string, string> = {
  ".svg": "image/svg+xml; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp"
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const relativePath = url.searchParams.get("path") ?? "";
  const normalizedPath = relativePath.replaceAll("/", path.sep).replaceAll("\\", path.sep);

  if (!allowedFigurePath.test(normalizedPath)) {
    return NextResponse.json({ ok: false, error: "artifact path is not allowed" }, { status: 403 });
  }

  const target = path.resolve(resolveWorkspacePath(normalizedPath));
  const root = path.resolve(workspaceRoot);
  if (!target.startsWith(root + path.sep)) {
    return NextResponse.json({ ok: false, error: "artifact path escapes workspace" }, { status: 403 });
  }

  const extension = path.extname(target).toLowerCase();
  const body = await fs.readFile(target).catch(() => null);
  if (!body) {
    return NextResponse.json({ ok: false, error: "artifact not found" }, { status: 404 });
  }

  return new NextResponse(body, {
    headers: {
      "Content-Type": contentTypes[extension] ?? "application/octet-stream",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    }
  });
}
