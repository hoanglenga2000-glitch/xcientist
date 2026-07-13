from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
SRC = WEB / "src"
CLIENT_TS = SRC / "lib" / "api" / "client.ts"
NAVIGATION_TS = SRC / "components" / "workstation" / "navigation.ts"
PAGE_TSX = SRC / "app" / "home-client.tsx"
OUT_JSON = ROOT / "workspace" / "workstation_frontend_api_contract_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_FRONTEND_API_CONTRACT_20260630.md"


METHOD_PATTERN = re.compile(r"method:\s*[\"'](?P<method>GET|POST|PATCH|PUT|DELETE)[\"']", re.IGNORECASE)
DYNAMIC_SEGMENT_PATTERN = re.compile(r"\$\{\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^}]*\}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def endpoint_to_route(endpoint: str) -> tuple[Path | None, str | None]:
    if not endpoint.startswith("/api/"):
        return None, "not_api_endpoint"
    endpoint = endpoint.split("?", 1)[0]
    route_parts: list[str] = []
    for part in endpoint.strip("/").split("/")[1:]:
        dynamic_match = DYNAMIC_SEGMENT_PATTERN.fullmatch(part)
        if dynamic_match:
            route_parts.append(f"[{dynamic_match.group('name')}]")
        else:
            route_parts.append(part)
    return SRC / "app" / "api" / Path(*route_parts) / "route.ts", None


def iter_fetch_calls(text: str) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    cursor = 0
    while True:
        start = text.find("fetch(", cursor)
        if start == -1:
            break
        depth = 0
        quote: str | None = None
        escaped = False
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = None
                    continue
                continue
            if char in {'"', "'", "`"}:
                quote = char
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        calls.append((start, text[start:end]))
        cursor = end
    return calls


def extract_endpoint(call_text: str) -> str | None:
    match = re.search(r"fetch\(\s*([`\"])(?P<endpoint>.*?)(?:\1)", call_text, re.DOTALL)
    return match.group("endpoint") if match else None


def extract_fetch_contracts() -> list[dict[str, Any]]:
    text = read(CLIENT_TS)
    contracts: list[dict[str, Any]] = []
    for start, call_text in iter_fetch_calls(text):
        endpoint = extract_endpoint(call_text)
        if not endpoint:
            continue
        method_match = METHOD_PATTERN.search(call_text)
        method = method_match.group("method").upper() if method_match else "GET"
        line = text[:start].count("\n") + 1
        route_path, skip_reason = endpoint_to_route(endpoint)
        exists = bool(route_path and route_path.exists())
        route_text = read(route_path) if exists and route_path else ""
        method_exported = bool(re.search(rf"export\s+async\s+function\s+{method}\b", route_text))
        contracts.append({
            "endpoint": endpoint,
            "method": method,
            "client_file": str(CLIENT_TS.relative_to(ROOT)).replace("\\", "/"),
            "client_line": line,
            "route_file": str(route_path.relative_to(ROOT)).replace("\\", "/") if route_path else None,
            "route_exists": exists,
            "method_exported": method_exported,
            "skip_reason": skip_reason,
            "ok": bool(route_path) and exists and method_exported,
        })
    return contracts


def extract_quoted_ids(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text)


def build_navigation_contract() -> dict[str, Any]:
    navigation_text = read(NAVIGATION_TS)
    page_text = read(PAGE_TSX)
    page_type_block = re.search(r"export\s+type\s+PageId\s*=([\s\S]*?);", navigation_text)
    page_type_ids = re.findall(r'"([^"]+)"', page_type_block.group(1)) if page_type_block else []
    nav_ids = extract_quoted_ids(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', navigation_text)
    rendered_ids = extract_quoted_ids(r'activePage\s*===\s*"([^"]+)"', page_text)
    page_ids_array_match = re.search(r"const\s+pageIds\s*=\s*\[([\s\S]*?)\]\s+as\s+const", page_text)
    page_ids_array = re.findall(r'"([^"]+)"', page_ids_array_match.group(1)) if page_ids_array_match else []
    aliases = {
        "mission": "overview" if 'normalized === "mission"' in page_text else None,
        "evidence-detail": "evidence" if 'normalized === "evidence-detail"' in page_text else None,
        "design": "settings" if 'normalized === "design"' in page_text else None,
    }
    missing_from_type = [page for page in nav_ids if page not in page_type_ids]
    missing_from_render = [page for page in nav_ids if page not in rendered_ids]
    missing_from_page_parser = [page for page in nav_ids if page not in page_ids_array]
    rendered_not_in_nav = [page for page in rendered_ids if page not in nav_ids and page != "design"]
    return {
        "page_type_ids": page_type_ids,
        "nav_ids": nav_ids,
        "rendered_ids": rendered_ids,
        "page_ids_array": page_ids_array,
        "aliases": aliases,
        "missing_from_type": missing_from_type,
        "missing_from_render": missing_from_render,
        "missing_from_page_parser": missing_from_page_parser,
        "rendered_not_in_nav": rendered_not_in_nav,
        "ok": not (missing_from_type or missing_from_render or missing_from_page_parser or rendered_not_in_nav),
    }


def build_report() -> dict[str, Any]:
    fetch_contracts = extract_fetch_contracts()
    navigation_contract = build_navigation_contract()
    failed_contracts = [item for item in fetch_contracts if not item["ok"]]
    return {
        "schema": "academic_research_os.frontend_api_contract.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not failed_contracts and navigation_contract["ok"] else "failed",
        "fetch_contract_count": len(fetch_contracts),
        "failed_fetch_contracts": failed_contracts,
        "navigation_contract": navigation_contract,
        "fetch_contracts": fetch_contracts,
        "claim_boundary": "This check proves frontend API client route/method bindings and navigation-page coverage. It does not prove business-side training success or Kaggle scores.",
    }


def write_markdown(report: dict[str, Any]) -> None:
    nav = report["navigation_contract"]
    lines = [
        "# 工作站前端 API 与导航契约检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 总状态：`{report['status']}`",
        f"- API client fetch 数量：`{report['fetch_contract_count']}`",
        f"- 失败绑定数量：`{len(report['failed_fetch_contracts'])}`",
        "",
        "## 导航与页面覆盖",
        "",
        f"- PageId 类型数量：`{len(nav['page_type_ids'])}`",
        f"- 侧边栏导航数量：`{len(nav['nav_ids'])}`",
        f"- 实际渲染页面数量：`{len(nav['rendered_ids'])}`",
        f"- URL parser 页面数量：`{len(nav['page_ids_array'])}`",
        f"- missing from PageId：`{', '.join(nav['missing_from_type']) or 'none'}`",
        f"- missing from render：`{', '.join(nav['missing_from_render']) or 'none'}`",
        f"- missing from URL parser：`{', '.join(nav['missing_from_page_parser']) or 'none'}`",
        "",
        "## API Client 绑定",
        "",
        "| method | endpoint | route | exists | method exported |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["fetch_contracts"]:
        lines.append(
            f"| `{item['method']}` | `{item['endpoint']}` | `{item['route_file']}` | `{item['route_exists']}` | `{item['method_exported']}` |"
        )
    if report["failed_fetch_contracts"]:
        lines.extend(["", "## 失败项", ""])
        for item in report["failed_fetch_contracts"]:
            lines.append(f"- `{item['method']} {item['endpoint']}` -> `{item['route_file']}`")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"]])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frontend API client route/method bindings and navigation coverage.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)
    print(json.dumps({
        "status": report["status"],
        "fetch_contract_count": report["fetch_contract_count"],
        "failed_fetch_contract_count": len(report["failed_fetch_contracts"]),
        "navigation_ok": report["navigation_contract"]["ok"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
