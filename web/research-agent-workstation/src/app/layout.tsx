import type { Metadata } from "next";
import { headers } from "next/headers";
import { LocalSessionBootstrap } from "@/components/workstation/LocalSessionBootstrap";
import { ThemeProvider } from "@/components/workstation/theme/ThemeProvider";
import "./globals.css";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "EvoMind Research Workstation",
  description: "Auditable scientific research and machine-learning operations workstation"
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const nonce = (await headers()).get("x-nonce") ?? undefined;
  const enableFigmaCapture = process.env.NODE_ENV !== "production" || process.env.NEXT_PUBLIC_FIGMA_CAPTURE === "true";
  return (
    <html lang="zh-CN" data-theme="dark" data-theme-mode="dark" className="dark" suppressHydrationWarning>
      <body>
        <LocalSessionBootstrap><ThemeProvider>{children}</ThemeProvider></LocalSessionBootstrap>
        {enableFigmaCapture ? <script nonce={nonce} src="https://mcp.figma.com/mcp/html-to-design/capture.js" async /> : null}
      </body>
    </html>
  );
}
