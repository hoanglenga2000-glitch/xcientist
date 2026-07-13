from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "external_resources.yaml"


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {"status": "failed", "message": message, "evidence": evidence or {}},
            ensure_ascii=False,
            indent=2,
        )
    )


def require(condition: bool, message: str, evidence: dict[str, Any] | None = None) -> None:
    if not condition:
        fail(message, evidence)


def main() -> None:
    require(MANIFEST.is_file(), "external resources manifest is missing")
    text = MANIFEST.read_text(encoding="utf-8-sig")
    manifest = yaml.safe_load(text) or {}
    require(manifest.get("schema") == "evomind.external_resources.v2", "manifest schema is not release-safe v2")

    policy = manifest.get("policy") or {}
    require(
        policy.get("secrets_storage") == "protected_user_store_or_secret_file_only",
        "secret storage policy must require protected storage",
        {"policy": policy},
    )
    require(policy.get("no_plaintext_secrets_in_repo") is True, "plaintext secret policy is missing")
    require(policy.get("gpu_shell_policy") == "whitelist_templates_only", "GPU shell policy must remain whitelist-only")
    require(policy.get("local_training_fallback") == "disabled", "local training fallback must remain disabled")

    resources = manifest.get("resources") or {}
    required_resources = {"deepseek", "hpc_gpu_ssh", "claude_code", "kaggle"}
    require(required_resources == set(resources), "manifest resource set is incomplete", {"resources": sorted(resources)})

    deepseek = resources["deepseek"]
    require(
        deepseek.get("status") == "implementation_ready_runtime_not_configured",
        "release source must not claim a live DeepSeek credential",
        {"deepseek": deepseek},
    )
    require(deepseek.get("base_url") == "https://api.deepseek.com", "DeepSeek base URL is not official")
    require(
        {"deepseek-chat", "deepseek-reasoner"}.issubset(set(deepseek.get("supported_models") or [])),
        "DeepSeek release model names are incomplete",
    )
    require("Not Configured" in str(deepseek.get("current_runtime_note")), "DeepSeek boundary is not explicit")

    gpu = resources["hpc_gpu_ssh"]
    require(gpu.get("status") == "not_configured", "release source must not claim a live HPC allocation")
    bridge = gpu.get("local_bridge") or {}
    require(bridge.get("listen") == "127.0.0.1:7890", "HPC bridge must remain loopback-only")
    for key, expected in {
        "script": "scripts/hpc_socks_bridge.py",
        "launcher": "scripts/start_hpc_socks_bridge.py",
        "manager": "scripts/manage_hpc_proxy_bridge.ps1",
    }.items():
        require(bridge.get(key) == expected and (ROOT / expected).is_file(), "HPC bridge contract is incomplete", {key: bridge.get(key)})
    required_runtime = set(gpu.get("required_runtime") or [])
    require(
        {
            "EVOMIND_HPC_HOST",
            "EVOMIND_HPC_PORT",
            "EVOMIND_HPC_USER",
            "EVOMIND_HPC_PASSWORD",
            "EVOMIND_HPC_REMOTE_WORKSPACE",
        }.issubset(required_runtime),
        "HPC runtime contract is incomplete",
        {"required_runtime": sorted(required_runtime)},
    )
    require("Not Configured" in str(gpu.get("current_blocker")), "HPC release blocker is not explicit")
    for forbidden_key in (
        "ssh_host",
        "ssh_port",
        "ssh_user",
        "proxy",
        "verified_gpu_environment",
        "automated_job_credential_status",
        "ssh_config",
    ):
        require(forbidden_key not in gpu, "release manifest contains machine-specific HPC state", {"key": forbidden_key})

    claude = resources["claude_code"]
    require(claude.get("status") == "awaiting_api_key", "release source must not claim a live code-agent provider")
    require("Manual Gate" in str(claude.get("gate_policy")), "code-agent patch policy must require a Manual Gate")

    kaggle = resources["kaggle"]
    require(kaggle.get("status") == "awaiting_token", "release source must not claim a live Kaggle token")
    require("Human Gate" in str(kaggle.get("gate_policy")), "Kaggle submission policy must require a Human Gate")

    forbidden_fragments = (
        "C:/" + "Users/",
        "D:/",
        "aimslab" + "-",
        "job_id:",
        "ncat_path:",
    )
    leaked_fragments = [fragment for fragment in forbidden_fragments if fragment.casefold() in text.casefold()]
    require(not leaked_fragments, "release manifest contains machine-specific fragments", {"fragments": leaked_fragments})

    non_loopback_ips: list[str] = []
    for token in re.findall(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text):
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            continue
        if not address.is_loopback:
            non_loopback_ips.append(token)
    require(not non_loopback_ips, "release manifest contains a concrete remote IP", {"addresses": sorted(set(non_loopback_ips))})

    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "resources": sorted(resources),
                "release_default": "not_configured",
                "machine_specific_remote_ips": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
