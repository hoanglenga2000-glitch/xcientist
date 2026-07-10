from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "docs" / "verified_workstation_launch_audit.json"
AUDIT_MD = ROOT / "docs" / "verified_workstation_launch_audit.md"

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bKGAT_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:DEEPSEEK_API_KEY|ANTHROPIC_API_KEY|KAGGLE_KEY|KAGGLE_API_TOKEN)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{12,}"),
]


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def require(condition: bool, message: str, evidence: dict | None = None) -> None:
    if not condition:
        fail(message, evidence)


def scan_for_secret_text(path: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append({"file": str(path.relative_to(ROOT)), "line": line_no, "pattern": pattern.pattern})
    return findings


def is_local_dashboard_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port is not None


def main() -> None:
    require(AUDIT_JSON.is_file(), "verified workstation audit JSON is missing", {"path": str(AUDIT_JSON.relative_to(ROOT))})
    require(AUDIT_MD.is_file(), "verified workstation audit markdown is missing", {"path": str(AUDIT_MD.relative_to(ROOT))})
    report = json.loads(AUDIT_JSON.read_text(encoding="utf-8-sig"))
    require(report.get("status") == "passed", "verified launcher audit did not pass", {"status": report.get("status")})
    require(is_local_dashboard_url(report.get("dashboard_url")), "verified launcher audit points to unexpected dashboard", {"dashboard_url": report.get("dashboard_url")})

    dpapi = report.get("dpapi_loaded") or {}
    require(dpapi.get("deepseek") is True, "DeepSeek DPAPI key should be loaded for the verified local launch", {"dpapi_loaded": dpapi})
    for key in ("claude", "kaggle", "hpc_ssh"):
        require(isinstance(dpapi.get(key), bool), f"{key} DPAPI audit flag must be boolean", {"dpapi_loaded": dpapi})
    require(
        isinstance(report.get("secret_policy"), str) and "No secret values" in report.get("secret_policy", ""),
        "verified launcher audit must document the no-secret policy",
        {"secret_policy": report.get("secret_policy")},
    )
    require(
        report.get("allow_real_external") is False,
        "verified local launch audit should not invoke real external resources by default",
        {"allow_real_external": report.get("allow_real_external")},
    )

    labels = {item.get("label"): item for item in report.get("result_summaries") or []}
    required_labels = {
        "backend_resource_status",
        "deepseek_smoke",
        "external_gateway_smoke",
        "kaggle_secret_smoke",
        "plaintext_secret_scan",
    }
    missing = sorted(required_labels - set(labels))
    require(not missing, "verified launcher audit is missing required smoke labels", {"missing": missing, "labels": sorted(labels)})
    failed = {label: item for label, item in labels.items() if not item.get("ok")}
    require(not failed, "one or more verified launcher smoke steps failed", {"failed": failed})

    external_excerpt = str(labels["external_gateway_smoke"].get("output_excerpt") or "")
    if dpapi.get("claude") or dpapi.get("hpc_ssh"):
        require(
            "configured_not_invoked" in external_excerpt,
            "configured external resources must remain not-invoked unless the launch explicitly allows real external smoke",
            {"external_gateway_smoke": labels["external_gateway_smoke"]},
        )
    kaggle_excerpt = str(labels["kaggle_secret_smoke"].get("output_excerpt") or "")
    if dpapi.get("kaggle"):
        require(
            "configured_not_invoked" in kaggle_excerpt and "human_gate_required_for_submission" in kaggle_excerpt,
            "Kaggle configured state must preserve non-invocation and human submission gate",
            {"kaggle_secret_smoke": labels["kaggle_secret_smoke"]},
        )

    secret_findings = scan_for_secret_text(AUDIT_JSON) + scan_for_secret_text(AUDIT_MD)
    require(not secret_findings, "verified launcher audit contains possible plaintext secrets", {"findings": secret_findings[:20]})

    print(json.dumps({
        "status": "passed",
        "audit_json": str(AUDIT_JSON.relative_to(ROOT)),
        "audit_markdown": str(AUDIT_MD.relative_to(ROOT)),
        "dpapi_loaded": dpapi,
        "smoke_labels": sorted(labels),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
