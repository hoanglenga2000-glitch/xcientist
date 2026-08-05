import { readFileSync } from "node:fs";

const packageVersion = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")).version;

const commonSecurityHeaders = [
  { key: "Referrer-Policy", value: "no-referrer" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=()" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  env: {
    NEXT_PUBLIC_EVOMIND_VERSION: packageVersion,
  },
  // Do not pin generateBuildId. Next emits a unique build identity so cached
  // HTML can never point at another release's content-addressed chunks.
  async headers() {
    return [
      {
        source: "/:path((?!_next/static|_next/image|api/artifacts$|api/public-scientific-report/artifact$).*)",
        headers: [
          ...commonSecurityHeaders,
          { key: "Cache-Control", value: "private, no-store, max-age=0, must-revalidate" },
        ],
      },
      {
        source: "/_next/static/:path*",
        headers: [
          ...commonSecurityHeaders,
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      // These download routes intentionally emit a stricter, media-specific
      // CSP. Do not replace it with the application-shell CSP.
      ...["/api/artifacts", "/api/public-scientific-report/artifact"].map((source) => ({
        source,
        headers: [
          ...commonSecurityHeaders.filter((header) => header.key !== "X-Frame-Options"),
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "Cache-Control", value: "private, no-store, max-age=0, must-revalidate" },
        ],
      })),
    ];
  },
};

export default nextConfig;
