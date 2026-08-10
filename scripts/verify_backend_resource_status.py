from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from workstation_local_auth import authenticated_headers  # noqa: E402

REQUIRED_LOCAL_CONNECTORS = {
    "llm": "rule_based",
    "python_runner": "local",
    "storage": "local_workspace",
}

EXTERNAL_CONNECTORS = {
    "code_agent": {
        "name": "Code Agent",
        "required_when_configured": ["ANTHROPIC_API_KEY or DEEPSEEK_API_KEY"],
    },
    "gpu": {
        "name": "GPU SSH Gateway",
        "required_when_configured": [
            "GPU_SSH_HOST",
            "GPU_SSH_USER",
            "GPU_SSH_KEY_PATH or GPU_SSH_PASSWORD",
            "GPU_REMOTE_WORKSPACE",
        ],
    },
}

OPTIONAL_EXTERNAL_CONNECTORS = {
    "openai": {
        "name": "OpenAI-compatible Agent LLM",
        "required_when_configured": ["OPENAI_API_KEY"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "required_when_configured": ["DEEPSEEK_API_KEY"],
    },
    "kaggle": {
        "name": "Kaggle",
        "required_when_configured": ["KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY"],
    },
}


def get_json(url: str) -> dict[str, Any]:
    base_url = url.split("/api/", 1)[0]
    request = urllib.request.Request(url, headers=authenticated_headers(base_url))
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {"status": "failed", "message": message, "evidence": evidence or {}},
            ensure_ascii=False,
            indent=2,
        )
    )


def configured_state_is_acceptable(key: str, state: str) -> bool:
    normalized = state.lower()
    if key == "kaggle":
        return normalized == "configured_unverified" or any(
            marker in normalized for marker in ("authenticated", "ready", "verified")
        )
    if key == "gpu" and ("auth pending" in normalized or "external ssh pending" in normalized):
        return False
    if key == "gpu" and normalized == "configured_unverified":
        return False
    if key == "gpu" and "blocked" in normalized:
        return False
    return "ready" in normalized or "verified" in normalized


def local_connector_ready(item: dict[str, Any], expected_raw_state: str) -> bool:
    """Validate both the canonical registry state and its source projection."""

    return bool(
        item.get("configured") is True
        and item.get("state") == "READY"
        and item.get("raw_state") == expected_raw_state
        and item.get("source") == "connector_health_service"
    )


def gpu_current_gate_ready(item: dict[str, Any]) -> bool:
    authoritative_gate = item.get("current_gate_ready")
    if isinstance(authoritative_gate, bool):
        return authoritative_gate

    evidence = item.get("evidence") or {}
    dependency_gate = evidence.get("latest_s6e6_dependency_gate") or {}
    latest_ssh = evidence.get("latest_ssh_connection") or {}
    if latest_ssh.get("present") and latest_ssh.get("passed") is False:
        return False
    if dependency_gate.get("status") in {"blocked_resource_gateway", "blocked_dependency", "failed", "not_configured"}:
        return False
    return dependency_gate.get("status") == "passed" or latest_ssh.get("passed") is True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify backend-authoritative connector readiness from /api/workstation-summary.")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--require-external-configured", action="store_true")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    summary = get_json(f"{base}/api/workstation-summary")
    connectors = summary.get("connector_status") or {}

    local_results = {}
    for key, expected_state in REQUIRED_LOCAL_CONNECTORS.items():
        item = connectors.get(key) or {}
        local_results[key] = {
            "configured": bool(item.get("configured")),
            "state": item.get("state"),
            "expected_state": "READY",
            "raw_state": item.get("raw_state"),
            "expected_raw_state": expected_state,
            "source": item.get("source"),
        }
        if not local_connector_ready(item, expected_state):
            fail("required local connector is not ready", {"connector": key, "status": local_results[key]})

    external_results = {}
    missing_external = []
    for key, spec in {**EXTERNAL_CONNECTORS, **OPTIONAL_EXTERNAL_CONNECTORS}.items():
        item = connectors.get(key) or {}
        configured = bool(item.get("configured"))
        current_resource_gate_ready = True
        external_results[key] = {
            "name": item.get("name") or spec["name"],
            "configured": configured,
            "state": item.get("state"),
            "notes": item.get("notes"),
            "required_when_configured": spec["required_when_configured"],
            "optional": key in OPTIONAL_EXTERNAL_CONNECTORS,
        }
        if key == "gpu":
            current_resource_gate_ready = gpu_current_gate_ready(item)
            external_results[key]["current_gate_ready"] = current_resource_gate_ready
            external_results[key]["latest_s6e6_dependency_gate"] = (item.get("evidence") or {}).get("latest_s6e6_dependency_gate")
            if not current_resource_gate_ready:
                if key in EXTERNAL_CONNECTORS:
                    missing_external.append("gpu_current_resource_gate")
        if key == "gpu" and "auth pending" in str(item.get("state", "")).lower():
            missing_external.append("gpu_ssh_auth")
        elif not configured and key in EXTERNAL_CONNECTORS:
            missing_external.append(key)
        if configured and not configured_state_is_acceptable(key, str(item.get("state", ""))):
            if key == "gpu" and not current_resource_gate_ready:
                continue
            if key == "gpu" and "auth pending" in str(item.get("state", "")).lower():
                continue
            fail("configured external connector is not reporting ready", {"connector": key, "status": external_results[key]})

    if args.require_external_configured and missing_external:
        fail(
            "external resources are not configured in backend connector status",
            {"missing_external": missing_external, "external_results": external_results},
        )

    # The authenticated summary intentionally omits raw environment-contract
    # fields in its lightweight response.  Validate the public connector
    # projection and use the dedicated readiness artifacts for secret-backed
    # resources instead of requiring the API to expose internal env metadata.
    env_keys = connectors.get("env_keys") or {}
    env_contract = {
        "public_summary_env_keys_exposed": bool(env_keys),
        "code_agent_configured": bool((connectors.get("code_agent") or {}).get("configured")),
        "openai_configured": bool((connectors.get("openai") or {}).get("configured")),
        "deepseek_configured": bool((connectors.get("deepseek") or {}).get("configured")),
    }
    if not env_contract["code_agent_configured"]:
        fail("backend code-agent connector is not configured", {"connector": connectors.get("code_agent")})
    if not (env_contract["openai_configured"] or env_contract["deepseek_configured"]):
        fail("backend external LLM connector is not configured", {"connectors": ["openai", "deepseek"]})

    kaggle = connectors.get("kaggle") or {}
    kaggle_report_path = ROOT / "docs" / "kaggle_dpapi_readiness.json"
    try:
        kaggle_report = json.loads(kaggle_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        kaggle_report = {}
    kaggle_toolchain_ready = bool(
        kaggle.get("toolchain_ready")
        or (kaggle_report.get("tool_status") or {}).get("python_package_installed")
    )
    kaggle_human_gate = bool(
        kaggle.get("human_gate_required_for_submission")
        or kaggle_report.get("human_gate_required_for_submission")
    )
    if not kaggle_toolchain_ready:
        fail("Kaggle readiness artifact must prove toolchain readiness", {"report": str(kaggle_report_path.relative_to(ROOT))})
    if not kaggle_human_gate:
        fail("Kaggle readiness artifact must keep leaderboard submission behind Human Gate", {"report": str(kaggle_report_path.relative_to(ROOT))})

    print(
        json.dumps(
            {
                "status": "passed",
                "dashboard_url": base,
                "local_connectors": local_results,
                "external_connectors": external_results,
                "missing_external": missing_external,
                "env_contract": env_contract,
                "kaggle_readiness_artifact": str(kaggle_report_path.relative_to(ROOT)),
                "ready_mode": "fully_ready" if not missing_external else "ready_for_external_resources",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
