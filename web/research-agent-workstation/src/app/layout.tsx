import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Research Agent Workstation",
  description: "AI Data Scientist Lab research agent workstation prototype"
};

const cssRescueScript = `
(function () {
  function isCssMissing() {
    var bodyStyle = window.getComputedStyle(document.body);
    var shell = document.querySelector(".workstation-chrome");
    var shellStyle = shell ? window.getComputedStyle(shell) : null;
    var bodyLooksDefault = bodyStyle.backgroundColor === "rgba(0, 0, 0, 0)" || bodyStyle.backgroundColor === "rgb(255, 255, 255)";
    var shellLooksDefault = shellStyle ? shellStyle.backgroundColor === "rgba(0, 0, 0, 0)" : true;
    var defaultFont = /Times New Roman/i.test(bodyStyle.fontFamily);
    return bodyLooksDefault && (shellLooksDefault || defaultFont);
  }

  function reloadStylesheet() {
    var existing = document.querySelector('link[rel="stylesheet"][href*="/_next/static/css/"]');
    if (!existing || document.querySelector('link[data-workstation-css-rescue="true"]')) return;
    var nextHref = existing.getAttribute("href");
    if (!nextHref) return;
    var separator = nextHref.indexOf("?") === -1 ? "?" : "&";
    var rescued = document.createElement("link");
    rescued.rel = "stylesheet";
    rescued.href = nextHref + separator + "workstation_css_rescue=" + Date.now();
    rescued.setAttribute("data-workstation-css-rescue", "true");
    document.head.appendChild(rescued);
  }

  window.addEventListener("load", function () {
    window.setTimeout(function () {
      if (isCssMissing()) reloadStylesheet();
    }, 250);
  });
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        {children}
        <script dangerouslySetInnerHTML={{ __html: cssRescueScript }} />
        <script src="https://mcp.figma.com/mcp/html-to-design/capture.js" async />
      </body>
    </html>
  );
}
