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
COMPONENT_DIR = SRC / "components" / "workstation"
PAGE_TSX = SRC / "app" / "page.tsx"
APP_SHELL = COMPONENT_DIR / "AppShell.tsx"
WORKSTATION_ACTIONS = SRC / "lib" / "server" / "workstation-actions.ts"
OUT_JSON = ROOT / "workspace" / "workstation_ui_action_contract_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_UI_ACTION_CONTRACT_20260630.md"


SAFE_LIVE_ACTIONS = [
    {
        "action": "ui_component_click",
        "task_id": "playground_series_s6e6",
        "metadata": {
            "page": "overview",
            "component_type": "button",
            "action_id": "contract_smoke_click",
            "label": "contract smoke click",
            "disabled": False,
        },
    },
    {
        "action": "quick_open_workspace",
        "task_id": "playground_series_s6e6",
        "metadata": {"page": "overview", "source": "contract_smoke"},
    },
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def source_files() -> list[Path]:
    files = sorted(COMPONENT_DIR.glob("*.tsx"))
    return [PAGE_TSX, *files]


def line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def extract_data_ui_actions() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    literal_pattern = re.compile(r"data-ui-action=(?:\"([^\"]+)\"|'([^']+)'|\{`([^`]+)`\})")
    for path in source_files():
        text = read(path)
        for match in literal_pattern.finditer(text):
            value = next(group for group in match.groups() if group is not None)
            dynamic = "${" in value
            actions.append({
                "action_id": value,
                "dynamic": dynamic,
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "line": line_number(text, match.start()),
            })
    return actions


def extract_frontend_invoked_actions() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    patterns = [
        re.compile(r"onAction\?\.\(\s*\"([^\"]+)\""),
        re.compile(r"onAction\(\s*\"([^\"]+)\""),
        re.compile(r"runWorkstationAction\(\s*\"([^\"]+)\""),
        re.compile(r"api\.runWorkstationAction\(\s*\"([^\"]+)\""),
    ]
    for path in source_files():
        text = read(path)
        for pattern in patterns:
            for match in pattern.finditer(text):
                actions.append({
                    "action": match.group(1),
                    "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "line": line_number(text, match.start()),
                })
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in actions:
        unique[(item["action"], item["file"], item["line"])] = item
    return sorted(unique.values(), key=lambda item: (item["action"], item["file"], item["line"]))


def extract_global_ui_action_routes() -> dict[str, dict[str, Any]]:
    text = read(APP_SHELL)
    match = re.search(r"const\s+uiActionRoutes\s*:[\s\S]+?=\s*\{(?P<body>[\s\S]+?)\n\};", text)
    if not match:
        return {}
    routes: dict[str, dict[str, Any]] = {}
    for route_match in re.finditer(r"^\s*(?P<key>[a-zA-Z0-9_]+)\s*:\s*\{(?P<body>[^\n]+)\}", match.group("body"), re.MULTILINE):
        body = route_match.group("body")
        page_match = re.search(r'page:\s*"([^"]+)"', body)
        action_match = re.search(r'action:\s*"([^"]+)"', body)
        routes[route_match.group("key")] = {
            "page": page_match.group(1) if page_match else None,
            "action": action_match.group(1) if action_match else None,
        }
    return routes


def extract_global_ui_action_route_patterns() -> list[dict[str, Any]]:
    text = read(APP_SHELL)
    match = re.search(r"const\s+uiActionRoutePatterns\s*:[\s\S]+?=\s*\[(?P<body>[\s\S]+?)\n\];", text)
    if not match:
        return []
    patterns: list[dict[str, Any]] = []
    for route_match in re.finditer(r'prefix:\s*"(?P<prefix>[^"]+)"[\s\S]*?route:\s*\{(?P<body>[^}]+)\}', match.group("body")):
        body = route_match.group("body")
        page_match = re.search(r'page:\s*"([^"]+)"', body)
        action_match = re.search(r'action:\s*"([^"]+)"', body)
        patterns.append({
            "prefix": route_match.group("prefix"),
            "page": page_match.group(1) if page_match else None,
            "action": action_match.group(1) if action_match else None,
        })
    return patterns


def extract_backend_cases() -> list[str]:
    text = read(WORKSTATION_ACTIONS)
    return sorted(set(re.findall(r'case\s+"([^"]+)"\s*:', text)))


def has_backend_default_handler() -> bool:
    text = read(WORKSTATION_ACTIONS)
    return (
        "default:" in text
        and "writeJsonArtifact(`workspace/runtime/${action}_${stamp()}.json`" in text
        and "logAction({" in text
    )


def has_global_click_delegate() -> bool:
    text = read(APP_SHELL)
    return (
        "onClickCapture={handleUiClick}" in text
        and "closest(interactiveSelector)" in text
        and 'onAction("ui_component_click"' in text
        and "data-ui-action" in text
    )


def live_post(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    origin = base_url.rstrip("/")
    url = f"{origin}/api/workstation-actions"
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": origin,
            "Referer": f"{origin}/",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(2048).decode("utf-8", errors="replace")
            parsed = json.loads(body)
            return {
                "action": payload["action"],
                "status": response.status,
                "ok": response.status == 200 and parsed.get("ok") is not False,
                "response_keys": sorted(parsed.keys()),
                "artifact": parsed.get("artifact"),
                "message": parsed.get("message"),
            }
    except HTTPError as exc:
        return {
            "action": payload["action"],
            "status": exc.code,
            "ok": False,
            "error": exc.read(1024).decode("utf-8", errors="replace") or exc.reason,
        }
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {
            "action": payload["action"],
            "status": "error",
            "ok": False,
            "error": str(exc),
        }


def build_report(base_url: str, live_post_safe: bool, timeout: int) -> dict[str, Any]:
    data_ui_actions = extract_data_ui_actions()
    frontend_actions = extract_frontend_invoked_actions()
    global_ui_routes = extract_global_ui_action_routes()
    global_ui_route_patterns = extract_global_ui_action_route_patterns()
    backend_cases = extract_backend_cases()
    backend_case_set = set(backend_cases)
    backend_default = has_backend_default_handler()
    global_delegate = has_global_click_delegate()

    direct_action_contracts = []
    for item in frontend_actions:
        explicit = item["action"] in backend_case_set
        direct_action_contracts.append({
            **item,
            "backend_explicit_case": explicit,
            "backend_default_fallback": backend_default and not explicit,
            "ok": explicit or backend_default,
        })

    data_ui_contracts = []
    for item in data_ui_actions:
        action_id = item["action_id"]
        explicit = action_id in backend_case_set
        directly_invoked = any(action_id == action["action"] for action in frontend_actions)
        global_route = global_ui_routes.get(action_id)
        if not global_route:
            global_route = next((pattern for pattern in global_ui_route_patterns if action_id.startswith(pattern["prefix"])), None)
        routed_action = global_route.get("action") if global_route else None
        routed_explicit = bool(routed_action and routed_action in backend_case_set)
        routed_default = bool(routed_action and backend_default and routed_action not in backend_case_set)
        data_ui_contracts.append({
            **item,
            "backend_explicit_case": explicit,
            "direct_frontend_invocation": directly_invoked,
            "global_route": global_route,
            "global_route_action_explicit_case": routed_explicit,
            "global_route_action_default_fallback": routed_default,
            "global_click_audit": global_delegate,
            "ok": global_delegate or explicit or directly_invoked or bool(global_route),
        })

    live_results = [live_post(base_url, payload, timeout) for payload in SAFE_LIVE_ACTIONS] if live_post_safe else []

    failed_direct = [item for item in direct_action_contracts if not item["ok"]]
    failed_data_ui = [item for item in data_ui_contracts if not item["ok"]]
    failed_live = [item for item in live_results if not item["ok"]]

    telemetry_only_count = sum(
        1
        for item in data_ui_contracts
        if (
            item["global_click_audit"]
            and not item["backend_explicit_case"]
            and not item["direct_frontend_invocation"]
            and not item["global_route"]
        )
    )
    global_route_count = sum(1 for item in data_ui_contracts if item["global_route"])
    global_route_pattern_count = len(global_ui_route_patterns)

    status = "passed" if global_delegate and backend_default and not failed_direct and not failed_data_ui and not failed_live else "failed"
    return {
        "schema": "academic_research_os.workstation_ui_action_contract.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "status": status,
        "global_click_delegate": global_delegate,
        "backend_default_handler": backend_default,
        "backend_case_count": len(backend_cases),
        "data_ui_action_count": len(data_ui_contracts),
        "direct_frontend_action_count": len(direct_action_contracts),
        "global_ui_route_count": global_route_count,
        "global_ui_route_pattern_count": global_route_pattern_count,
        "telemetry_only_data_ui_action_count": telemetry_only_count,
        "failed_direct_actions": failed_direct,
        "failed_data_ui_actions": failed_data_ui,
        "live_post_safe_enabled": live_post_safe,
        "live_post_results": live_results,
        "failed_live_posts": failed_live,
        "backend_cases": backend_cases,
        "direct_action_contracts": direct_action_contracts,
        "data_ui_action_contracts": data_ui_contracts,
        "claim_boundary": (
            "This check proves UI action wiring and safe backend handling. It separates business actions "
            "from telemetry-only clicks. It does not prove that every telemetry-only click starts a domain workflow, "
            "and it does not trigger training, GPU jobs, or Kaggle submission."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站 UI Action 合约检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 全局点击审计：`{report['global_click_delegate']}`",
        f"- 后端 default 安全处理器：`{report['backend_default_handler']}`",
        f"- 后端显式 action case 数：`{report['backend_case_count']}`",
        f"- data-ui-action 数：`{report['data_ui_action_count']}`",
        f"- 前端直接调用 action 数：`{report['direct_frontend_action_count']}`",
        f"- 仅审计记录的 UI action 数：`{report['telemetry_only_data_ui_action_count']}`",
        "",
        "## 结论",
        "",
    ]
    if report["status"] == "passed":
        lines.append("页面交互具备统一点击审计，前端直接调用的工作站 action 均能被后端显式 case 或 default 安全处理器接住。")
    else:
        lines.append("仍存在未接线的 UI action 或安全后端处理器缺失，不能声明全部页面组件可用。")

    lines.extend([
        "",
        "## Safe Live POST",
        "",
        f"- 是否执行：`{report['live_post_safe_enabled']}`",
        "",
        "| action | status | ok | artifact |",
        "| --- | --- | --- | --- |",
    ])
    for item in report["live_post_results"]:
        lines.append(f"| `{item['action']}` | `{item['status']}` | `{item['ok']}` | `{item.get('artifact')}` |")
    if not report["live_post_results"]:
        lines.append("| none | skipped | true | none |")

    lines.extend([
        "",
        "## 前端直接调用 Action",
        "",
        "| action | explicit backend case | default fallback | source |",
        "| --- | --- | --- | --- |",
    ])
    for item in report["direct_action_contracts"]:
        lines.append(
            f"| `{item['action']}` | `{item['backend_explicit_case']}` | `{item['backend_default_fallback']}` | `{item['file']}:{item['line']}` |"
        )

    if report["failed_direct_actions"] or report["failed_data_ui_actions"] or report["failed_live_posts"]:
        lines.extend(["", "## 失败项", ""])
        for item in report["failed_direct_actions"]:
            lines.append(f"- direct action `{item['action']}` at `{item['file']}:{item['line']}`")
        for item in report["failed_data_ui_actions"]:
            lines.append(f"- data-ui-action `{item['action_id']}` at `{item['file']}:{item['line']}`")
        for item in report["failed_live_posts"]:
            lines.append(f"- live POST `{item['action']}` failed: `{item.get('error') or item.get('status')}`")

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify workstation UI action wiring against backend workstation actions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--live-post-safe", action="store_true", help="POST two safe non-training actions to /api/workstation-actions.")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args.base_url, args.live_post_safe, args.timeout)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "global_click_delegate": report["global_click_delegate"],
        "backend_default_handler": report["backend_default_handler"],
        "backend_case_count": report["backend_case_count"],
        "data_ui_action_count": report["data_ui_action_count"],
        "direct_frontend_action_count": report["direct_frontend_action_count"],
        "global_ui_route_count": report["global_ui_route_count"],
        "global_ui_route_pattern_count": report["global_ui_route_pattern_count"],
        "telemetry_only_data_ui_action_count": report["telemetry_only_data_ui_action_count"],
        "failed_direct_action_count": len(report["failed_direct_actions"]),
        "failed_data_ui_action_count": len(report["failed_data_ui_actions"]),
        "failed_live_post_count": len(report["failed_live_posts"]),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
