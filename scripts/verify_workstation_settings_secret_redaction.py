from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "workspace" / "workstation_settings_secret_redaction_20260701.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_SETTINGS_SECRET_REDACTION_20260701.md"

SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"KGAT_[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|cookie|credential|private[_-]?key|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
ALLOWED_REDACTED_VALUES = {
    "",
    "hidden",
    "hidden_configured",
    "configured",
    "not_configured",
    "configured_dpapi",
    "ready",
    "blocked",
}


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as response:
      return json.loads(response.read().decode("utf-8"))


def inspect_value(value: Any, path: str, findings: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            if SENSITIVE_KEY_PATTERN.search(key):
                if isinstance(item, str) and item not in ALLOWED_REDACTED_VALUES:
                    findings.append({"path": child_path, "reason": "sensitive_key_has_unredacted_string_value"})
                elif item not in (None, False) and not isinstance(item, str):
                    findings.append({"path": child_path, "reason": "sensitive_key_has_non_status_value"})
            inspect_value(item, child_path, findings)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            inspect_value(item, f"{path}[{index}]", findings)
        return
    if isinstance(value, str):
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                findings.append({"path": path, "reason": "secret_value_pattern_detected"})
                break


def build_report(url: str) -> dict[str, Any]:
    payload = fetch_json(url)
    findings: list[dict[str, str]] = []
    inspect_value(payload.get("settings", payload), "settings", findings)
    return {
        "schema": "academic_research_os.settings_secret_redaction.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not findings else "failed",
        "url": url,
        "finding_count": len(findings),
        "findings": findings,
        "claim_boundary": "This check proves the public settings API response does not expose obvious secret values. It does not inspect the private server-side secret store.",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站 Settings API 脱敏检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 状态：`{report['status']}`",
        f"- 检查地址：`{report['url']}`",
        f"- 发现数量：`{report['finding_count']}`",
        "",
        "## 发现",
        "",
    ]
    if report["findings"]:
        for item in report["findings"]:
            lines.append(f"- `{item['path']}`：{item['reason']}")
    else:
        lines.append("- 未发现公开 settings 响应中的明文密钥模式。")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"]])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify /api/settings redacts secret-like values.")
    parser.add_argument("--url", default="http://127.0.0.1:8088/api/settings")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report(args.url)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)
    print(json.dumps({
        "status": report["status"],
        "finding_count": report["finding_count"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
