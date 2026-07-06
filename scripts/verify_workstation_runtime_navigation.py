from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
SRC = WEB / "src"
NAVIGATION_TS = SRC / "components" / "workstation" / "navigation.ts"
PAGE_TSX = SRC / "app" / "page.tsx"
CLIENT_TS = SRC / "lib" / "api" / "client.ts"
OUT_JSON = ROOT / "workspace" / "workstation_runtime_navigation_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_RUNTIME_NAVIGATION_20260630.md"


BASE_API_PATHS = [
    "/api/workstation-summary",
    "/api/tasks",
    "/api/settings",
    "/api/gpu/jobs",
    "/api/paper-evidence-bundle",
]

PAGE_ALIASES = {
    "mission": "overview",
    "evidence-detail": "evidence",
    "design": "settings",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_nav_ids() -> list[str]:
    text = read(NAVIGATION_TS)
    return re.findall(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', text)


def extract_page_ids_array() -> list[str]:
    text = read(PAGE_TSX)
    match = re.search(r"const\s+pageIds\s*=\s*\[([\s\S]*?)\]\s+as\s+const", text)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []


def extract_rendered_page_ids() -> list[str]:
    text = read(PAGE_TSX)
    return re.findall(r'activePage\s*===\s*"([^"]+)"', text)


def extract_client_static_get_paths() -> list[str]:
    text = read(CLIENT_TS)
    candidates = set(re.findall(r'fetch\(\s*["`](/api/[^"`$]+)["`]\s*\)', text))
    return sorted(path for path in candidates if "${" not in path)


def request_get(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        request = Request(url, headers={"Accept": "text/html,application/json"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read(1024)
            content_type = response.headers.get("content-type", "")
            return {
                "target": path,
                "url": url,
                "status": response.status,
                "content_type": content_type,
                "body_size_sample": len(body),
                "ok": response.status == 200 and bool(body),
            }
    except HTTPError as exc:
        return {
            "target": path,
            "url": url,
            "status": exc.code,
            "ok": False,
            "error": exc.reason,
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "target": path,
            "url": url,
            "status": "error",
            "ok": False,
            "error": str(exc),
        }


def build_report(base_url: str, timeout: int) -> dict[str, Any]:
    nav_ids = extract_nav_ids()
    page_ids_array = extract_page_ids_array()
    rendered_ids = extract_rendered_page_ids()

    static_client_paths = extract_client_static_get_paths()
    api_paths = sorted(set(BASE_API_PATHS + static_client_paths))
    page_targets = sorted(set(nav_ids + list(PAGE_ALIASES) + ["unknown-runtime-smoke"]))

    page_smoke = [request_get(base_url, f"/?page={page}", timeout) for page in page_targets]
    api_smoke = [request_get(base_url, path, timeout) for path in api_paths]

    missing_from_parser = [page for page in nav_ids if page not in page_ids_array]
    missing_from_render = [page for page in nav_ids if page not in rendered_ids]
    rendered_not_nav = [page for page in rendered_ids if page not in nav_ids and page != "design"]
    alias_checks = [
        {
            "alias": alias,
            "expected_page": expected,
            "declared_in_parser": f'normalized === "{alias}"' in read(PAGE_TSX),
            "http_ok": next((item["ok"] for item in page_smoke if item["target"] == f"/?page={alias}"), False),
        }
        for alias, expected in PAGE_ALIASES.items()
    ]

    failed_pages = [item for item in page_smoke if not item["ok"]]
    failed_apis = [item for item in api_smoke if not item["ok"]]
    contract_failures = missing_from_parser or missing_from_render or rendered_not_nav

    status = "passed" if not failed_pages and not failed_apis and not contract_failures else "failed"
    return {
        "schema": "academic_research_os.workstation_runtime_navigation.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "status": status,
        "nav_page_count": len(nav_ids),
        "parser_page_count": len(page_ids_array),
        "rendered_page_count": len(rendered_ids),
        "api_path_count": len(api_paths),
        "nav_ids": nav_ids,
        "page_ids_array": page_ids_array,
        "rendered_ids": rendered_ids,
        "aliases": alias_checks,
        "missing_from_parser": missing_from_parser,
        "missing_from_render": missing_from_render,
        "rendered_not_in_navigation": rendered_not_nav,
        "page_smoke": page_smoke,
        "api_smoke": api_smoke,
        "failed_pages": failed_pages,
        "failed_apis": failed_apis,
        "claim_boundary": (
            "This runtime check proves that declared workstation pages, supported URL aliases, "
            "unknown-page fallback, and read-only API endpoints respond over the running frontend. "
            "It does not trigger training, GPU jobs, Kaggle submission, or Figma edits."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站运行时导航与 API 矩阵检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 导航页面数：`{report['nav_page_count']}`",
        f"- URL parser 页面数：`{report['parser_page_count']}`",
        f"- 实际渲染页面数：`{report['rendered_page_count']}`",
        f"- API 检查数：`{report['api_path_count']}`",
        "",
        "## 页面覆盖",
        "",
        f"- parser 缺失：`{', '.join(report['missing_from_parser']) or 'none'}`",
        f"- render 缺失：`{', '.join(report['missing_from_render']) or 'none'}`",
        f"- render 中未在导航声明：`{', '.join(report['rendered_not_in_navigation']) or 'none'}`",
        "",
        "## 别名入口",
        "",
        "| alias | expected page | parser declared | HTTP ok |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["aliases"]:
        lines.append(
            f"| `{item['alias']}` | `{item['expected_page']}` | `{item['declared_in_parser']}` | `{item['http_ok']}` |"
        )

    lines.extend([
        "",
        "## 页面 Smoke",
        "",
        "| target | status | ok |",
        "| --- | --- | --- |",
    ])
    for item in report["page_smoke"]:
        lines.append(f"| `{item['target']}` | `{item['status']}` | `{item['ok']}` |")

    lines.extend([
        "",
        "## API Smoke",
        "",
        "| target | status | ok |",
        "| --- | --- | --- |",
    ])
    for item in report["api_smoke"]:
        lines.append(f"| `{item['target']}` | `{item['status']}` | `{item['ok']}` |")

    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify runtime page navigation and read-only API smoke coverage.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args.base_url, args.timeout)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "nav_page_count": report["nav_page_count"],
        "api_path_count": report["api_path_count"],
        "failed_page_count": len(report["failed_pages"]),
        "failed_api_count": len(report["failed_apis"]),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
