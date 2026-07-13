from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

try:
    from scripts.manage_workstation_dashboard import source_tree_digest
except ModuleNotFoundError:  # Direct execution from scripts/.
    from manage_workstation_dashboard import source_tree_digest

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
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append({"file": str(display_path), "line": line_no, "pattern": pattern.pattern})
    return findings


def is_local_dashboard_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port is not None


def parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def process_is_running(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def verify_runtime_binding(report: dict, root: Path) -> dict:
    runtime = report.get("dashboard_runtime") or {}
    require(isinstance(runtime, dict), "dashboard runtime identity is missing")
    parsed_url = urlparse(str(report.get("dashboard_url") or ""))
    require(runtime.get("mode") == "start", "verified launch must use Next.js production mode", {"runtime": runtime})
    require(runtime.get("build_requested") is True, "verified launch must build the current source tree", {"runtime": runtime})
    require(runtime.get("port") == parsed_url.port, "dashboard runtime port does not match audit URL", {"runtime": runtime})
    require(process_is_running(runtime.get("pid")), "audited dashboard process is not running", {"pid": runtime.get("pid")})

    app_dir = root / "web" / "research-agent-workstation"
    build_id_path = app_dir / ".next" / "BUILD_ID"
    state_path = app_dir / ".runtime-logs" / "dashboard.state.json"
    pid_path = app_dir / ".runtime-logs" / "dashboard.pid"
    require(build_id_path.is_file(), "audited production BUILD_ID is missing")
    require(state_path.is_file(), "dashboard runtime state is missing")
    require(pid_path.is_file(), "dashboard PID file is missing")
    build_id = build_id_path.read_text(encoding="utf-8").strip()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        pid_file = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        fail("dashboard PID file is invalid")
    for key in ("pid", "port", "mode", "build_id", "source_digest", "build_requested"):
        require(runtime.get(key) == state.get(key), "dashboard runtime state changed after audit", {"key": key})
    require(runtime.get("pid") == pid_file, "dashboard PID changed after audit")
    require(runtime.get("build_id") == build_id, "dashboard BUILD_ID changed after audit")
    current_digest = source_tree_digest(app_dir)
    require(runtime.get("source_digest") == current_digest, "dashboard source changed after the production build")
    return {"pid": pid_file, "build_id": build_id, "source_digest": current_digest, "mode": runtime.get("mode")}


def main() -> None:
    require(AUDIT_JSON.is_file(), "verified workstation audit JSON is missing", {"path": str(AUDIT_JSON.relative_to(ROOT))})
    require(AUDIT_MD.is_file(), "verified workstation audit markdown is missing", {"path": str(AUDIT_MD.relative_to(ROOT))})
    report = json.loads(AUDIT_JSON.read_text(encoding="utf-8-sig"))
    require(
        report.get("status") in {"passed", "local_ready_external_unverified"},
        "verified launcher audit did not complete",
        {"status": report.get("status")},
    )
    require(is_local_dashboard_url(report.get("dashboard_url")), "verified launcher audit points to unexpected dashboard", {"dashboard_url": report.get("dashboard_url")})
    require(re.fullmatch(r"[a-f0-9]{32}", str(report.get("run_id") or "")) is not None, "verified launcher run ID is invalid")
    generated_at = parse_utc_timestamp(report.get("generated_at"))
    require(generated_at is not None, "verified launcher audit timestamp is invalid")
    age_seconds = (datetime.now(timezone.utc) - generated_at).total_seconds()
    require(-120 <= age_seconds <= 900, "verified launcher audit is stale or future-dated", {"age_seconds": age_seconds})
    runtime_binding = verify_runtime_binding(report, ROOT)

    dpapi = report.get("dpapi_loaded") or {}
    for key in ("deepseek", "claude", "kaggle", "hpc_ssh"):
        require(isinstance(dpapi.get(key), bool), f"{key} DPAPI audit flag must be boolean", {"dpapi_loaded": dpapi})
    require(
        dpapi.get("deepseek") is True or dpapi.get("claude") is True,
        "At least one protected LLM provider must be loaded for the verified local launch",
        {"dpapi_loaded": dpapi},
    )
    require(
        isinstance(report.get("secret_policy"), str)
        and "No secret values" in report.get("secret_policy", "")
        and "raw command output" in report.get("secret_policy", ""),
        "verified launcher audit must document the no-secret policy",
        {"secret_policy": report.get("secret_policy")},
    )
    require(
        isinstance(report.get("allow_real_external"), bool),
        "verified launch audit must record whether real external calls were allowed",
        {"allow_real_external": report.get("allow_real_external")},
    )

    labels = {item.get("label"): item for item in report.get("result_summaries") or []}
    required_labels = {
        "backend_resource_status",
        "external_gateway_smoke",
        "kaggle_secret_smoke",
        "plaintext_secret_scan",
    }
    if dpapi.get("deepseek") is True:
        required_labels.add("deepseek_smoke")
    missing = sorted(required_labels - set(labels))
    require(not missing, "verified launcher audit is missing required smoke labels", {"missing": missing, "labels": sorted(labels)})
    failed = {label: item for label, item in labels.items() if not item.get("ok")}
    require(not failed, "one or more verified launcher smoke steps failed", {"failed": failed})
    require(
        all("output_excerpt" not in item for item in labels.values()),
        "verified launcher audit must not persist raw command output",
    )

    external_signals = labels["external_gateway_smoke"].get("signals") or {}
    deepseek_verified = bool(dpapi.get("deepseek") and labels.get("deepseek_smoke", {}).get("ok"))
    code_agent_verified = bool(dpapi.get("claude") and external_signals.get("code_agent_smoke_tested") is True)
    provider_verified = deepseek_verified or code_agent_verified
    require(
        report.get("external_provider_runtime_verified") is provider_verified,
        "external provider verification flag does not match allowlisted runtime evidence",
    )
    if report.get("status") == "passed":
        require(provider_verified, "passed status requires a live external LLM provider smoke")
    else:
        require(not provider_verified, "external-unverified status contradicts provider evidence")
        require(
            "no external LLM provider was invoked successfully" in str(report.get("claim_boundary") or ""),
            "external-unverified launch must preserve its claim boundary",
        )
    if report.get("allow_real_external") is True:
        require(provider_verified, "real external launch mode requires a successful live LLM provider smoke")

    if dpapi.get("claude"):
        require(
            external_signals.get("code_agent_configured_not_invoked") is True
            or external_signals.get("code_agent_smoke_tested") is True,
            "configured code agent must be explicitly not-invoked or live-smoke-tested",
            {"external_gateway_smoke": labels["external_gateway_smoke"]},
        )
    if dpapi.get("hpc_ssh"):
        require(
            external_signals.get("gpu_configured_not_invoked") is True
            or external_signals.get("gpu_smoke_tested") is True
            or (report.get("allow_resource_blockers") is True and external_signals.get("gpu_resource_blocked") is True),
            "configured GPU gateway must preserve an explicit runtime state",
            {"external_gateway_smoke": labels["external_gateway_smoke"]},
        )
    kaggle_signals = labels["kaggle_secret_smoke"].get("signals") or {}
    if dpapi.get("kaggle"):
        require(
            (
                kaggle_signals.get("kaggle_configured_not_invoked") is True
                or kaggle_signals.get("kaggle_authenticated_real_api") is True
            )
            and kaggle_signals.get("human_gate_required_for_submission") is True,
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
        "dashboard_runtime": runtime_binding,
        "external_provider_runtime_verified": provider_verified,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
