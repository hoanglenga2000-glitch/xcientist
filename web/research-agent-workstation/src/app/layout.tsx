import type { Metadata } from "next";
import { headers } from "next/headers";
import { LocalSessionBootstrap } from "@/components/workstation/LocalSessionBootstrap";
import { ThemeProvider, THEME_BOOT_SCRIPT } from "@/components/workstation/theme/ThemeProvider";
import { configuredPublicOrigin } from "@/lib/security/request-boundary";
import "./globals.css";
import "@/components/workstation/deepevo-brand.css";

export const dynamic = "force-dynamic";

// t237 C6-brand: favicon / PWA / og from the Iris kit in public/. No src/app/icon.* — it would win over these links.
const DESCRIPTION = "可审计的科研与机器学习工作台";

export function generateMetadata(): Metadata {
  const origin = configuredPublicOrigin();
  return {
    title: "DeepEvo",
    description: DESCRIPTION,
    ...(origin ? { metadataBase: new URL(origin) } : {}),
    icons: {
      icon: [{ url: "/favicon.ico", sizes: "16x16 32x32 48x48" }, { url: "/favicon.svg", type: "image/svg+xml" }],
      apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
    },
    manifest: "/manifest.webmanifest",
    openGraph: { title: "DeepEvo", description: DESCRIPTION, images: [{ url: "/og-image.png", width: 1200, height: 630, alt: "DeepEvo 深度进化" }] },
  };
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const nonce = (await headers()).get("x-nonce") ?? undefined;
  const enableFigmaCapture = !process.env.EVOMIND_TEST_FIXTURE_ROOT && (process.env.NODE_ENV !== "production" || process.env.NEXT_PUBLIC_FIGMA_CAPTURE === "true");
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <script nonce={nonce} dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body>
        <LocalSessionBootstrap><ThemeProvider>{children}</ThemeProvider></LocalSessionBootstrap>
        {enableFigmaCapture ? <script nonce={nonce} src="https://mcp.figma.com/mcp/html-to-design/capture.js" async /> : null}
      </body>
    </html>
  );
}
