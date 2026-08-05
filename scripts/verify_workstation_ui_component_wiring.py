from __future__ import annotations

import http.cookiejar
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
SRC = WEB / "src"
BASE_URL = os.environ.get("WORKSTATION_TEST_BASE_URL", "http://127.0.0.1:8088").rstrip("/")
HTTP = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
AUTH_MODE = "none"


def establish_local_session() -> str:
    """Exchange an optional one-time token without printing or persisting it."""

    global AUTH_MODE
    token = os.environ.get("WORKSTATION_TEST_BOOTSTRAP_TOKEN", "").strip()
    if not token:
        AUTH_MODE = "missing_bootstrap_token"
        return AUTH_MODE
    request = urllib.request.Request(
        f"{BASE_URL}/api/session/bootstrap",
        data=json.dumps({"token": token}, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Origin": BASE_URL},
    )
    try:
        with HTTP.open(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status != 200 or payload.get("ok") is not True:
                raise RuntimeError("local session bootstrap did not return ready")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("local session bootstrap failed") from exc
    AUTH_MODE = "one_time_fragment_exchange"
    return AUTH_MODE


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def http_ok(path: str) -> dict[str, object]:
    url = f"{BASE_URL}{path}"
    try:
        with HTTP.open(url, timeout=12) as response:
            body = response.read(256).decode("utf-8", errors="replace")
            return {
                "target": path,
                "status": response.status,
                "ok": response.status == 200 and bool(body),
            }
    except Exception as exc:  # pragma: no cover - smoke utility
        return {"target": path, "status": "error", "ok": False, "error": str(exc)}


def extract_nav_ids() -> list[str]:
    """Extract only routable leaf page ids from `navItems`, not the IA section
    headers in `navSections` (those are grouping labels, not navigable pages)."""
    text = read(SRC / "components" / "workstation" / "navigation.ts")
    nav_items_match = re.search(r"export const navItems\s*=\s*\[(.*?)\]\s*as const", text, re.DOTALL)
    scope = nav_items_match.group(1) if nav_items_match else text
    return re.findall(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', scope)


def extract_rendered_page_ids() -> list[str]:
    text = read(SRC / "app" / "page.tsx")
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
    auth_mode = establish_local_session()
    nav_ids = extract_nav_ids()
    rendered_ids = extract_rendered_page_ids()
    missing_render = [page for page in nav_ids if page not in rendered_ids]

    page_results = [http_ok(f"/?page={page}") for page in sorted(set(nav_ids + ["mission", "evidence-detail"]))]
    api_results = [http_ok(path) for path in [
        "/api/workstation-summary",
        "/api/tasks",
        "/api/settings",
        "/api/gpu/jobs",
        "/api/paper-evidence-bundle",
        "/api/runtime/health",
    ]]
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
        "status": status,
        "auth_mode": auth_mode,
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
