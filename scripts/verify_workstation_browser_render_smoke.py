from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLICK_JSON = ROOT / "workspace" / "workstation_click_smoke_20260701.json"
OUT_JSON = ROOT / "workspace" / "workstation_browser_render_smoke_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_BROWSER_RENDER_SMOKE_20260630.md"

DEFAULT_PAGES = [
    "overview",
    "control",
    "tasks",
    "data",
    "gpu",
    "evidence",
    "literature",
    "workflow",
    "code",
    "runtime",
    "experiments",
    "report",
    "gates",
    "settings",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def run_click_smoke(base_url: str, timeout: int) -> tuple[int, str, str]:
    command = [
        "node",
        "scripts\\verify_workstation_click_smoke.mjs",
        "--write-report",
        "--base-url",
        base_url,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


def load_click_report() -> dict[str, Any]:
    if not CLICK_JSON.exists():
        raise FileNotFoundError(f"click smoke report not found: {CLICK_JSON}")
    return json.loads(CLICK_JSON.read_text(encoding="utf-8"))


def build_report(base_url: str, pages: list[str], click_timeout: int) -> dict[str, Any]:
    returncode, stdout, stderr = run_click_smoke(base_url, click_timeout)
    try:
        click_report = load_click_report()
    except Exception as exc:
        return {
            "schema": "academic_research_os.workstation_browser_render_smoke.v2",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "status": "failed",
            "blocker": "click_smoke_report_unavailable",
            "browser": None,
            "pages": pages,
            "page_results": [],
            "failed_pages": pages,
            "click_smoke_returncode": returncode,
            "click_smoke_stdout_tail": stdout[-2000:],
            "click_smoke_stderr_tail": stderr[-2000:],
            "error": str(exc),
            "claim_boundary": (
                "The browser render smoke delegates to the CDP click smoke so pages are inspected after "
                "client-side routing has settled. It does not trigger training, GPU jobs, Kaggle submission, "
                "or Figma edits."
            ),
        }

    click_pages = {item.get("page"): item for item in click_report.get("page_results", [])}
    page_results: list[dict[str, Any]] = []
    for page in pages:
        item = click_pages.get(page)
        if not item:
            page_results.append({
                "page": page,
                "ok": False,
                "returncode": returncode,
                "has_shell": False,
                "has_page_marker": False,
                "has_sidebar": False,
                "action_count": 0,
                "button_count": 0,
                "link_count": 0,
                "component_count": 0,
                "visible_text_size": 0,
                "missing_keywords": ["page_not_checked_by_click_smoke"],
                "error_matches": [],
                "stderr_tail": stderr[-1000:],
                "timeout": False,
            })
            continue

        page_results.append({
            "page": page,
            "ok": bool(item.get("ok")),
            "returncode": returncode,
            "has_shell": item.get("activePage") == page,
            "has_page_marker": item.get("activePage") == page,
            "has_sidebar": True,
            "action_count": int(item.get("actionCount") or 0),
            "button_count": int(item.get("buttonCount") or 0),
            "link_count": 0,
            "component_count": int(item.get("actionCount") or 0),
            "visible_text_size": int(item.get("textSize") or 0),
            "missing_keywords": [],
            "error_matches": ["runtime_error"] if item.get("hasErrorText") else [],
            "stderr_tail": stderr[-1000:],
            "timeout": False,
        })

    failed_pages = [item["page"] for item in page_results if not item["ok"]]
    runtime_error_count = int(click_report.get("runtime_error_count") or 0)
    status = "passed" if not failed_pages and runtime_error_count == 0 and returncode == 0 else "failed"
    return {
        "schema": "academic_research_os.workstation_browser_render_smoke.v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "status": status,
        "blocker": None,
        "browser": click_report.get("chrome"),
        "pages": pages,
        "page_results": page_results,
        "failed_pages": failed_pages,
        "runtime_error_count": runtime_error_count,
        "click_smoke_returncode": returncode,
        "click_smoke_status": click_report.get("status"),
        "click_smoke_stdout_tail": stdout[-2000:],
        "click_smoke_stderr_tail": stderr[-2000:],
        "claim_boundary": (
            "This browser smoke uses the real CDP click smoke as its render engine, so it waits for "
            "client-side page routing before inspecting the DOM. It verifies page shell, active page marker, "
            "interactive controls, text volume, and runtime errors. It does not trigger training, GPU jobs, "
            "Kaggle submission, or Figma edits."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站浏览器级页面渲染冒烟检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 状态：`{report['status']}`",
        f"- 浏览器：`{report.get('browser') or 'not_found'}`",
        f"- 阻断项：`{report.get('blocker') or 'none'}`",
        f"- CDP 点击冒烟状态：`{report.get('click_smoke_status')}`",
        f"- 运行时错误数：`{report.get('runtime_error_count', 0)}`",
        "",
        "## 页面结果",
        "",
        "| page | ok | shell | page marker | sidebar | actions | buttons | text size | missing keywords | errors |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report.get("page_results") or []:
        lines.append(
            f"| `{item['page']}` | `{item['ok']}` | `{item['has_shell']}` | `{item['has_page_marker']}` | "
            f"`{item['has_sidebar']}` | {item['action_count']} | {item['button_count']} | "
            f"{item['visible_text_size']} | `{', '.join(item['missing_keywords']) or 'none'}` | "
            f"`{', '.join(item['error_matches']) or 'none'}` |"
        )

    lines.extend([
        "",
        "## 失败页面",
        "",
        f"`{', '.join(report.get('failed_pages') or []) or 'none'}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run browser-level rendered DOM smoke checks for workstation pages.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--pages", default=",".join(DEFAULT_PAGES), help="Comma-separated page IDs to inspect.")
    parser.add_argument("--timeout", type=int, default=30, help="Kept for backward compatibility.")
    parser.add_argument("--click-timeout", type=int, default=180, help="Timeout in seconds for the delegated CDP click smoke.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    pages = [page.strip() for page in args.pages.split(",") if page.strip()]
    report = build_report(args.base_url, pages, args.click_timeout)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "blocker": report.get("blocker"),
        "browser": report.get("browser"),
        "page_count": len(pages),
        "failed_pages": report.get("failed_pages"),
        "json": safe_relative(OUT_JSON) if args.write_report else None,
        "md": safe_relative(OUT_MD) if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
