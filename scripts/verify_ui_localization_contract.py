from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SOURCE_FILES = [
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "page.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Common.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "IntegrationStatus.tsx",
]

SOURCE_TERMS = [
    "任务队列",
    "运行任务",
    "系统动作日志",
    "接入状态",
    "Claude Code",
    "GPU SSH 网关",
    "待配置",
    "最终待批准",
    "查看全部",
    "人工 Gate",
    "报告大纲",
    "在线报告工作区",
    "AI 起草",
    "生成 AI 草稿",
    "专业图表",
    "工作站名称",
    "默认语言",
    "设置已从 SQLite 加载",
    "来源追踪",
    "批准提交",
    "审计导出快照",
    "updateReportLanguage",
    'setLocale?.(nextLanguage)',
    'source: "report_studio"',
]

LIVE_TERMS = [
    "任务总控台",
    "任务队列",
    "运行任务",
    "系统动作日志",
    "待配置",
]

SUMMARY_TERMS = ["Code Agent", "GPU", "Kaggle"]


def fail(message: str, evidence: dict) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence}, ensure_ascii=False, indent=2))


def read_sources() -> str:
    missing = [str(path.relative_to(ROOT)) for path in SOURCE_FILES if not path.exists()]
    if missing:
        fail("UI source files missing", {"missing_files": missing})
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_FILES)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Chinese UI localization and non-dead-entry labels for the workstation.")
    parser.add_argument("--url", default=None, help="Optional running dashboard URL, for example http://127.0.0.1:8088")
    args = parser.parse_args()

    source_text = read_sources()
    missing_source_terms = [term for term in SOURCE_TERMS if term not in source_text]
    if missing_source_terms:
        fail("required Chinese UI source terms are missing", {"missing_source_terms": missing_source_terms})

    live_result = None
    if args.url:
        base = args.url.rstrip("/")
        html = fetch(base)
        missing_live_terms = [term for term in LIVE_TERMS if term not in html]
        if missing_live_terms:
            fail("running dashboard is missing required visible UI terms", {"url": args.url, "missing_live_terms": missing_live_terms})
        summary_text = fetch(f"{base}/api/workstation-summary")
        missing_summary_terms = [term for term in SUMMARY_TERMS if term not in summary_text]
        if missing_summary_terms:
            fail("workstation summary is missing required integration terms", {"url": f"{base}/api/workstation-summary", "missing_summary_terms": missing_summary_terms})
        if "Not Configured" not in summary_text and "Ready" not in summary_text and "fully_ready" not in summary_text:
            fail("workstation summary is missing an integration readiness state", {"url": f"{base}/api/workstation-summary"})
        live_result = {"url": args.url, "checked_terms": LIVE_TERMS, "summary_terms": SUMMARY_TERMS}

    print(json.dumps({
        "status": "passed",
        "source_terms": SOURCE_TERMS,
        "live": live_result,
        "conclusion": "Chinese UI labels, run entry, integration status, and system action log labels are protected by a localization contract.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
