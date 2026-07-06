from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Any


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
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
    if env_keys.get("DEEPSEEK_MODEL") not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        fail("DeepSeek model contract is not current", {"env_keys": env_keys})

    configured = bool(connector.get("configured"))
    if args.require_configured and not configured:
        fail("DeepSeek is required but not configured", {"connector": connector})
    if configured and "ready" not in str(connector.get("state", "")).lower():
        fail("DeepSeek is configured but not reporting ready", {"connector": connector})

    smoke = post_json(f"{base}/api/llm/deepseek/smoke", {"prompt": "Return exactly: deepseek-ok"})
    if configured:
        if smoke.get("status") != "passed" or smoke.get("content") != "deepseek-ok":
            fail("DeepSeek real smoke did not pass", {"smoke": smoke})
        if not smoke.get("artifact_path"):
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
        "artifact_path": smoke.get("artifact_path"),
        "local_env_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
