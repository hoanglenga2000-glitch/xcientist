from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "Chrome浏览器验收状态-20260610.md"

REQUIRED_TERMS = [
    "http://127.0.0.1:8088",
    "Codex Chrome Extension",
    "Profile 1",
    "Browser is not available: extension",
    "verify_ui_localization_contract.py",
    "run_full_acceptance.py",
    "报告编辑页",
    "不伪造成功",
    "2026-06-11 更新",
]


def main() -> None:
    if not DOC.exists():
        raise SystemExit(json.dumps({"status": "failed", "missing": str(DOC.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    text = DOC.read_text(encoding="utf-8")
    missing = [term for term in REQUIRED_TERMS if term not in text]
    if missing:
        raise SystemExit(json.dumps({"status": "failed", "missing_terms": missing}, ensure_ascii=False, indent=2))
    print(json.dumps({
        "status": "passed",
        "doc": str(DOC.relative_to(ROOT)),
        "checked_terms": REQUIRED_TERMS,
        "conclusion": "Chrome acceptance status and fallback validation path are documented.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
