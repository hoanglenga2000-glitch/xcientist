from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


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
    with urllib.request.urlopen(url, timeout=20) as response:
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
    if key == "gpu" and ("auth pending" in normalized or "external ssh pending" in normalized):
        return False
    if key == "gpu" and "blocked" in normalized:
        return False
    return "ready" in normalized or "verified" in normalized


def gpu_current_gate_ready(item: dict[str, Any]) -> bool:
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
            "expected_state": expected_state,
        }
        if not item.get("configured") or item.get("state") != expected_state:
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

    env_keys = connectors.get("env_keys") or {}
    expected_env_contract = {
        "CODE_AGENT_PROVIDER": ["claude_agent_sdk", "deepseek_code_agent"],
        "GPU_PROVIDER": "ssh_gateway",
        "DATABASE_PROVIDER": "sqlite",
    }
    env_contract = {key: env_keys.get(key) for key in expected_env_contract}
    for key, expected in expected_env_contract.items():
        actual = env_keys.get(key)
        if isinstance(expected, list):
            ok = actual in expected
        else:
            ok = actual == expected
        if not ok:
            fail("backend env contract is not ready", {"key": key, "expected": expected, "actual": env_keys.get(key)})
    if "DEEPSEEK_API_KEY_STATUS" not in env_keys or "DEEPSEEK_MODEL" not in env_keys:
        fail("backend DeepSeek env contract is missing", {"env_keys": env_keys})
    if env_keys.get("KAGGLE_TOOLCHAIN_STATUS") != "ready" or env_keys.get("KAGGLE_TOKEN_STATUS") not in {"not_configured", "configured_dpapi"}:
        fail("backend Kaggle DPAPI/toolchain contract is missing", {"env_keys": env_keys})

    kaggle = connectors.get("kaggle") or {}
    if not kaggle.get("toolchain_ready"):
        fail("backend Kaggle connector must expose toolchain readiness", {"kaggle": kaggle})
    if not kaggle.get("human_gate_required_for_submission"):
        fail("backend Kaggle connector must keep leaderboard submission behind Human Gate", {"kaggle": kaggle})

    print(
        json.dumps(
            {
                "status": "passed",
                "dashboard_url": base,
                "local_connectors": local_results,
                "external_connectors": external_results,
                "missing_external": missing_external,
                "env_contract": env_contract,
                "ready_mode": "fully_ready" if not missing_external else "ready_for_external_resources",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
