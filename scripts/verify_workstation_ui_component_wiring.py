from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
SRC = WEB / "src"
HOME_CLIENT = SRC / "app" / "home-client.tsx"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def http_ok(base_url: str, path: str, accepted_statuses: set[int] | None = None) -> dict[str, object]:
    accepted = accepted_statuses or {200}
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            body = response.read(256).decode("utf-8", errors="replace")
            return {
                "target": path,
                "status": response.status,
                "ok": response.status in accepted and bool(body),
            }
    except HTTPError as exc:
        body = exc.read(256).decode("utf-8", errors="replace")
        return {
            "target": path,
            "status": exc.code,
            "ok": exc.code in accepted and bool(body),
            "expected_empty_state": exc.code in accepted,
        }
    except Exception as exc:  # pragma: no cover - smoke utility
        return {"target": path, "status": "error", "ok": False, "error": str(exc)}


def extract_nav_ids() -> list[str]:
    text = read(SRC / "components" / "workstation" / "navigation.ts")
    return re.findall(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', text)


def extract_rendered_page_ids() -> list[str]:
    text = read(HOME_CLIENT)
    return re.findall(r'activePage\s*===\s*"([^"]+)"', text)


def audit_data_ui_actions() -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    shell = SRC / "components" / "workstation" / "AppShell.tsx"
    shell_text = read(shell) if shell.exists() else ""
    has_global_delegate = (
        "onClickCapture={handleUiClick}" in shell_text
        and "closest(interactiveSelector)" in shell_text
        and "ui_component_click" in shell_text
    )
    if has_global_delegate:
        return findings

    files = [
        SRC / "components" / "workstation" / "Screens.tsx",
        SRC / "components" / "workstation" / "OverviewBoardEnhanced.tsx",
        SRC / "components" / "workstation" / "AiControlConsole.tsx",
        SRC / "components" / "workstation" / "AppShell.tsx",
        SRC / "components" / "workstation" / "Sidebar.tsx",
    ]
    tag_pattern = re.compile(r"<(?:Button|button)\b(?=[^>]*data-ui-action=)([^>]*)>", re.DOTALL)
    action_pattern = re.compile(r'data-ui-action=(?:"([^"]+)"|`([^`]+)`|\{`([^`]+)`\})')
    for path in files:
        if not path.exists():
            continue
        text = read(path)
        for match in tag_pattern.finditer(text):
            tag = match.group(0)
            if "onClick=" in tag:
                continue
            line_no = text[:match.start()].count("\n") + 1
            action_match = action_pattern.search(tag)
            action = next((group for group in (action_match.groups() if action_match else []) if group), "unknown")
            findings.append({
                "file": str(path.relative_to(ROOT)),
                "line": line_no,
                "action": action,
                "reason": "clickable control has data-ui-action but no onClick handler on the element",
            })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify workstation UI component wiring and live route reachability.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    nav_ids = extract_nav_ids()
    rendered_ids = extract_rendered_page_ids()
    missing_render = [page for page in nav_ids if page not in rendered_ids]

    page_results = [
        http_ok(args.base_url, f"/?page={page}")
        for page in sorted(set(nav_ids + ["mission", "evidence-detail"]))
    ]
    api_results = [
        http_ok(args.base_url, "/api/workstation-summary"),
        http_ok(args.base_url, "/api/tasks"),
        http_ok(args.base_url, "/api/settings"),
        http_ok(args.base_url, "/api/paper-evidence-bundle", {200, 404}),
    ]
    action_findings = audit_data_ui_actions()
    shell_text = read(SRC / "components" / "workstation" / "AppShell.tsx")
    global_event_delegate = (
        "onClickCapture={handleUiClick}" in shell_text
        and "closest(interactiveSelector)" in shell_text
        and "ui_component_click" in shell_text
    )

    failed_pages = [result for result in page_results if not result["ok"]]
    failed_apis = [result for result in api_results if not result["ok"]]
    status = "passed" if not (missing_render or failed_pages or failed_apis or action_findings) else "failed"
    payload = {
        "base_url": args.base_url,
        "status": status,
        "nav_page_count": len(nav_ids),
        "rendered_page_count": len(rendered_ids),
        "missing_rendered_pages": missing_render,
        "page_smoke": page_results,
        "api_smoke": api_results,
        "global_event_delegate": global_event_delegate,
        "unwired_clickable_actions": action_findings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
