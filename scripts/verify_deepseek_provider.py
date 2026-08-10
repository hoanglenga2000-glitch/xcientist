from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from workstation_local_auth import authenticated_headers  # noqa: E402


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    parsed = urllib.parse.urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=authenticated_headers(origin, {"Content-Type": "application/json", "Origin": origin}),
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "status": "provider_unavailable",
            "http_status": exc.code,
            "configured": True,
        }


def get_json(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(url, headers=authenticated_headers(origin))
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DeepSeek provider status and optional real smoke route.")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--require-configured", action="store_true")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    summary = get_json(f"{base}/api/workstation-summary")
    connector = (summary.get("connector_status") or {}).get("deepseek") or {}
    env_keys = (summary.get("connector_status") or {}).get("env_keys") or {}
    if not connector:
        fail("DeepSeek connector is missing from workstation summary")
    model = env_keys.get("DEEPSEEK_MODEL") or connector.get("model")
    if model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        fail("DeepSeek model contract is not current", {"model": model})

    configured = bool(connector.get("configured"))
    if args.require_configured and not configured:
        fail("DeepSeek is required but not configured", {"connector": connector})
    if configured and "ready" not in str(connector.get("state", "")).lower():
        fail("DeepSeek is configured but not reporting ready", {"connector": connector})

    smoke = post_json(f"{base}/api/llm/deepseek/smoke", {"prompt": "Return exactly: deepseek-ok"})
    optional_degraded = False
    if configured:
        if smoke.get("status") != "passed" or smoke.get("content") != "deepseek-ok":
            openai_report_path = ROOT / "workspace" / "llm" / "openai_gateway_smoke_current.json"
            try:
                openai_report = json.loads(openai_report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                openai_report = {}
            openai_primary_ready = (
                openai_report.get("status") == "passed"
                and openai_report.get("ok") is True
                and openai_report.get("requested_model") == "gpt-5.6-sol"
                and all((openai_report.get(key) or {}).get("ok") is True for key in ("models", "nonstream", "stream", "tool_call"))
            )
            if not openai_primary_ready:
                fail("DeepSeek real smoke did not pass and the primary OpenAI gateway is not proven", {"smoke": smoke})
            optional_degraded = True
        if not optional_degraded and not smoke.get("artifact_path"):
            fail("DeepSeek smoke did not write an audit artifact", {"smoke": smoke})
    else:
        if smoke.get("status") != "not_configured" or smoke.get("configured"):
            fail("DeepSeek unconfigured path is not explicit", {"smoke": smoke})

    print(json.dumps({
        "status": "passed",
        "dashboard_url": base,
        "connector": {
            "configured": configured,
            "state": connector.get("state"),
            "model": connector.get("model"),
        },
        "smoke_status": smoke.get("status"),
        "optional_provider_degraded": optional_degraded,
        "primary_openai_gateway_proven": optional_degraded,
        "artifact_path": smoke.get("artifact_path"),
        "local_env_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
