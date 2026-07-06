from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def workspace_path(raw_path: str | None) -> Path:
    if not raw_path:
        return ROOT
    return ROOT / raw_path.replace("\\", "/")


def request_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_bytes(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read(), response.headers.get("content-type", "")


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify report figure generation and browser-readable artifact previews.")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--task-id", default="house_prices")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    payload = request_json(f"{base}/api/tasks/{args.task_id}/generate-figures", method="POST")
    figures = payload.get("figures") or []
    if len(figures) < 6:
        fail("expected at least six report figures", {"payload": payload})

    manifest = workspace_path(str(payload.get("manifest_path", "")))
    if not manifest.exists():
        fail("figure manifest was not written", {"manifest_path": payload.get("manifest_path")})

    missing_figures = []
    for figure in figures:
        figure_path = workspace_path(str(figure.get("path", "")))
        if not figure_path.exists() or figure_path.stat().st_size <= 0:
            missing_figures.append(figure)
    if missing_figures:
        fail("one or more generated figure files are missing or empty", {"missing_figures": missing_figures})

    preview_figure = next((figure for figure in figures if str(figure.get("path", "")).lower().endswith(".svg")), figures[0])
    artifact_url = f"{base}/api/artifacts?path={urllib.parse.quote(str(preview_figure['path']))}"
    body, content_type = request_bytes(artifact_url)
    if b"<svg" not in body[:500] or "image/svg+xml" not in content_type:
        fail("artifact preview route did not return an SVG image", {"content_type": content_type, "artifact_url": artifact_url})

    report_payload = post_json(
        f"{base}/api/tasks/{args.task_id}/generate-report-draft",
        {"language": "zh-CN", "style": "publication"},
    )
    html_path_raw = report_payload.get("html_path")
    if not html_path_raw:
        fail("report draft did not return an HTML path", {"payload": report_payload})
    html_path = workspace_path(str(html_path_raw))
    if not html_path.exists():
        fail("report HTML draft was not written", {"html_path": html_path_raw})
    html = html_path.read_text(encoding="utf-8", errors="replace")
    figure_count = html.count("<figure><img")
    if figure_count < 1 or "/api/artifacts?path=" not in html:
        fail(
            "report HTML draft does not render generated figures",
            {"html_path": html_path_raw, "figure_count": figure_count},
        )

    print(json.dumps({
        "status": "passed",
        "task_id": args.task_id,
        "figures": [figure["name"] for figure in figures],
        "manifest_path": payload.get("manifest_path"),
        "artifact_preview_url": artifact_url,
        "report_html_path": html_path_raw,
        "report_html_figure_count": figure_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
