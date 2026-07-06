from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "external_resources.yaml"
DOC = ROOT / "docs" / "DeepSeek与HPC资源接入验收-20260613.md"


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def require(condition: bool, message: str, evidence: dict[str, Any] | None = None) -> None:
    if not condition:
        fail(message, evidence)


def main() -> None:
    require(MANIFEST.is_file(), "external resources manifest is missing", {"path": str(MANIFEST.relative_to(ROOT))})
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    resources = manifest.get("resources") or {}
    required = {"deepseek", "hpc_gpu_ssh", "claude_code", "kaggle"}
    missing = sorted(required - set(resources))
    require(not missing, "external resources manifest is missing required resource entries", {"missing": missing})

    policy = manifest.get("policy") or {}
    require(policy.get("secrets_storage") == "env_or_secret_file_only", "secret storage policy must forbid committed plaintext secrets", {"policy": policy})
    require(policy.get("no_plaintext_secrets_in_repo") is True, "plaintext-secret repository policy must be explicit", {"policy": policy})
    require(policy.get("gpu_shell_policy") == "whitelist_templates_only", "GPU shell policy must remain whitelist-only", {"policy": policy})
    require(policy.get("local_training_fallback") == "disabled", "local training fallback must stay disabled; Kaggle/MLE-Bench training must use workstation-gated HPC/GPU resources", {"policy": policy})

    deepseek = resources["deepseek"]
    require(
        deepseek.get("status") in {"ready", "implementation_ready_runtime_not_configured"},
        "DeepSeek status must distinguish implementation readiness from runtime credentials",
        {"deepseek": deepseek},
    )
    require(deepseek.get("base_url") == "https://api.deepseek.com", "DeepSeek base URL must match official API endpoint", {"deepseek": deepseek})
    require(deepseek.get("default_model") in {"deepseek-v4-flash", "deepseek-v4-pro"}, "DeepSeek default model is not current", {"deepseek": deepseek})
    require({"deepseek-v4-flash", "deepseek-v4-pro"}.issubset(set(deepseek.get("supported_models") or [])), "DeepSeek supported models are incomplete", {"deepseek": deepseek})
    require("DEEPSEEK_API_KEY" in (deepseek.get("required_env") or []), "DeepSeek required env is missing", {"deepseek": deepseek})
    if deepseek.get("status") == "implementation_ready_runtime_not_configured":
        require("Not Configured" in str(deepseek.get("current_runtime_note")), "DeepSeek runtime credential boundary must be explicit", {"deepseek": deepseek})

    gpu = resources["hpc_gpu_ssh"]
    require(
        gpu.get("status") in {"ready", "network_ready_auth_pending", "gpu_verified_job_gateway_credentials_pending", "ssh_ready_gpu_allocation_missing", "ssh_ready_gpu_busy", "configured_channels_closed"},
        "HPC/GPU status must distinguish current SSH readiness from older pending states",
        {"hpc_gpu_ssh": gpu},
    )
    bridge = gpu.get("local_bridge") or {}
    require(bridge.get("listen") == "127.0.0.1:7890", "HPC local bridge must use the PDF-compatible 7890 endpoint", {"local_bridge": bridge})
    require("hpc_socks_bridge.py" in str(bridge.get("script")) and "manage_hpc_proxy_bridge.ps1" in str(bridge.get("manager")), "HPC bridge scripts must be recorded", {"local_bridge": bridge})
    require("open_hpc_browser.ps1" in str(bridge.get("browser_launcher")), "HPC browser launcher must be recorded", {"local_bridge": bridge})
    require(bridge.get("ncat_status") == "installed_verified_user_path", "PDF ncat path must be installed and verified", {"local_bridge": bridge})
    require("ncat.exe" in str(bridge.get("ncat_path")) and str(bridge.get("ncat_version")), "ncat executable path and version must be recorded", {"local_bridge": bridge})
    require("ncat --proxy 127.0.0.1:7890" in str(bridge.get("ncat_pdf_command")), "PDF ncat command must be recorded", {"local_bridge": bridge})
    require(gpu.get("ssh_host") == "100.85.169.63" and int(gpu.get("ssh_port")) == 1235, "HPC SSH endpoint is not recorded", {"hpc_gpu_ssh": gpu})
    proxy = gpu.get("proxy") or {}
    require(proxy.get("type") == "socks5" and proxy.get("auth") == "username_password", "HPC SOCKS5 proxy auth contract is missing", {"proxy": proxy})
    required_gpu_env = {"GPU_SSH_HOST", "GPU_SSH_PORT", "GPU_SSH_USER", "GPU_REMOTE_WORKSPACE"}
    required_env_values = set(gpu.get("required_env") or [])
    require(required_gpu_env.issubset(required_env_values), "GPU required env contract is incomplete", {"hpc_gpu_ssh": gpu})
    require(
        "GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH" in required_env_values or "GPU_SSH_KEY_PATH" in required_env_values,
        "GPU required env must allow a password or private-key credential",
        {"hpc_gpu_ssh": gpu},
    )
    if gpu.get("status") == "ready":
        require(gpu.get("current_blocker") in {"", None}, "Ready GPU status must not keep an active blocker", {"current_blocker": gpu.get("current_blocker")})
        require("DPAPI" in str(gpu.get("current_runtime_note")), "Ready GPU status must document DPAPI credential loading", {"current_runtime_note": gpu.get("current_runtime_note")})
        job_credentials = gpu.get("automated_job_credential_status") or {}
        require(job_credentials.get("password_credential_status") == "ready_windows_dpapi", "GPU password credential status must record Windows DPAPI readiness", {"automated_job_credential_status": job_credentials})
        require(job_credentials.get("public_key_install_status") == "optional_not_required_for_current_password_flow", "GPU public-key status must be optional for the current password flow", {"automated_job_credential_status": job_credentials})
    elif gpu.get("status") == "ssh_ready_gpu_allocation_missing":
        blocker = str(gpu.get("current_blocker"))
        require("cuda_device_count=0" in blocker and "/dev/nvidia" in blocker, "GPU allocation blocker must record missing CUDA devices", {"current_blocker": gpu.get("current_blocker")})
        require(gpu.get("platform_environment_status") == "ssh_verified_current_allocation_has_no_cuda_devices", "GPU platform status must record current no-CUDA allocation", {"status": gpu.get("platform_environment_status")})
        require("DPAPI" in str(gpu.get("current_runtime_note")) and "torch.cuda.device_count > 0" in str(gpu.get("current_runtime_note")), "Current GPU runtime note must document DPAPI and strict CUDA readiness", {"current_runtime_note": gpu.get("current_runtime_note")})
        verified = gpu.get("verified_gpu_environment") or {}
        require(int(verified.get("gpu_count") or 0) == 0 and "none_visible" in str(verified.get("gpu_type")), "Current GPU environment must record zero visible GPUs", {"verified_gpu_environment": verified})
        require(str(verified.get("evidence_file", "")).startswith("workspace/gpu/connection_test_"), "Current GPU blocker evidence must be a connection-test artifact", {"verified_gpu_environment": verified})
        job_credentials = gpu.get("automated_job_credential_status") or {}
        require(job_credentials.get("password_credential_status") == "ready_windows_dpapi", "Current SSH credential must remain DPAPI-backed", {"automated_job_credential_status": job_credentials})
        require("rotating per allocation" in str(job_credentials.get("portal_key_binding_observation")), "Credential rotation requirement must be documented", {"automated_job_credential_status": job_credentials})
    elif gpu.get("status") == "ssh_ready_gpu_busy":
        blocker = str(gpu.get("current_blocker"))
        require("100%" in blocker and "free" in blocker, "Busy GPU blocker must record utilization and free-memory evidence", {"current_blocker": gpu.get("current_blocker")})
        require(gpu.get("platform_environment_status") == "ssh_ready_gpu_busy_current_1x_a800", "GPU platform status must record current busy A800 allocation", {"status": gpu.get("platform_environment_status")})
        require("DPAPI" in str(gpu.get("current_runtime_note")) and "resource-blocked" in str(gpu.get("current_runtime_note")), "Busy GPU note must document DPAPI and resource-blocked state", {"current_runtime_note": gpu.get("current_runtime_note")})
        verified = gpu.get("verified_gpu_environment") or {}
        require(int(verified.get("gpu_count") or 0) == 1 and "A800" in str(verified.get("gpu_type")), "Busy GPU environment must record the visible A800", {"verified_gpu_environment": verified})
        require(str(verified.get("evidence_file", "")).startswith("workspace/gpu/connection_test_"), "Busy GPU evidence must be a connection-test artifact", {"verified_gpu_environment": verified})
        job_credentials = gpu.get("automated_job_credential_status") or {}
        require(job_credentials.get("password_credential_status") == "ready_windows_dpapi", "Busy GPU SSH credential must remain DPAPI-backed", {"automated_job_credential_status": job_credentials})
    elif gpu.get("status") == "configured_channels_closed":
        blocker = str(gpu.get("current_blocker"))
        require("channels" in blocker and "closed" in blocker, "Closed-channel GPU blocker must record that local channels are intentionally closed", {"current_blocker": gpu.get("current_blocker")})
        require(gpu.get("platform_environment_status") == "channels_closed_allowed_allocations_only", "GPU platform status must record closed local channels and allowed allocations", {"status": gpu.get("platform_environment_status")})
        require("87384" in str(gpu.get("current_runtime_note")) and "87318" in str(gpu.get("current_runtime_note")), "Closed-channel GPU note must list the only allowed allocations", {"current_runtime_note": gpu.get("current_runtime_note")})
        require("research_agent_workstation" in str(gpu.get("current_runtime_note")), "Closed-channel GPU note must keep the dedicated workspace policy", {"current_runtime_note": gpu.get("current_runtime_note")})
        verified = gpu.get("verified_gpu_environment") or {}
        require(verified.get("ssh_user") == "aimslab-TTA-A800-1GPU", "Closed-channel default allocation must be the user-approved 87384 account", {"verified_gpu_environment": verified})
        require(str(verified.get("evidence_file", "")).startswith("workspace/gpu/local_hpc_channels_closed_"), "Closed-channel evidence must be recorded", {"verified_gpu_environment": verified})
        job_credentials = gpu.get("automated_job_credential_status") or {}
        require(job_credentials.get("password_credential_status") == "ready_windows_dpapi", "Closed-channel SSH credential must remain DPAPI-backed", {"automated_job_credential_status": job_credentials})
    elif gpu.get("status") == "gpu_verified_job_gateway_credentials_pending":
        current_blocker = str(gpu.get("current_blocker"))
        normalized_blocker = current_blocker.lower().replace("public key", "publickey")
        require(
            "partial success" in normalized_blocker and "publickey" in normalized_blocker,
            "GPU blocker must be narrowed to remote publickey authorization after nvidia-smi proof",
            {"current_blocker": gpu.get("current_blocker")},
        )
        require(gpu.get("platform_environment_status") == "gpu_verified_via_login_node_web_terminal", "GPU platform status must show verified login-node/Web Terminal evidence", {"status": gpu.get("platform_environment_status")})
        verified = gpu.get("verified_gpu_environment") or {}
        require(verified.get("evidence_file") == "workspace/hpc/web_terminal_probe.txt", "verified GPU evidence file must be recorded", {"verified_gpu_environment": verified})
        require(int(verified.get("gpu_count") or 0) == 4 and "A800" in str(verified.get("gpu_type")), "verified GPU environment must record 4 x A800", {"verified_gpu_environment": verified})
        require(verified.get("auth_mode_evidence_file") == "workspace/hpc/ssh_auth_mode_20260613.json", "GPU SSH auth-mode evidence file must be recorded", {"verified_gpu_environment": verified})
        require(verified.get("portal_key_binding_probe_file") == "workspace/hpc/portal_key_binding_probe_20260613.json", "GPU portal key-binding probe file must be recorded", {"verified_gpu_environment": verified})
        job_credentials = gpu.get("automated_job_credential_status") or {}
        require(job_credentials.get("public_key_install_status") == "blocked_remote_requires_preauthorized_publickey", "GPU job credential blocker must record remote publickey authorization requirement", {"automated_job_credential_status": job_credentials})
        require(
            "authorized public key" in str(job_credentials.get("portal_key_binding_observation")) or "no self-service SSH public-key binding endpoint" in str(job_credentials.get("portal_key_binding_observation")),
            "GPU job credential blocker must record the public-key binding requirement",
            {"automated_job_credential_status": job_credentials},
        )
    else:
        require("Permission denied (password,publickey)" in str(gpu.get("current_blocker")), "Current HPC blocker must be explicit and non-fake", {"current_blocker": gpu.get("current_blocker")})
        require(gpu.get("platform_environment_status") == "gpu_environment_created_web_terminal_ready_ssh_external_pending", "GPU platform status must remain Web Terminal ready and external SSH pending until nvidia-smi evidence exists", {"status": gpu.get("platform_environment_status")})
    require("10.120.x" in str(gpu.get("development_environment_note")) and ("not reachable" in str(gpu.get("development_environment_note")) or "timed out" in str(gpu.get("development_environment_note"))), "Development environment inner-IP boundary must be explicit", {"note": gpu.get("development_environment_note")})
    ssh_config = gpu.get("ssh_config") or {}
    require(ssh_config.get("host_alias") == "hpc-hkust-gz" and "ncat --proxy 127.0.0.1:7890" in str(ssh_config.get("proxy_command")), "SSH config alias must record the ncat ProxyCommand", {"ssh_config": ssh_config})
    strongest = gpu.get("strongest_visible_resource") or {}
    require(
        any(gpu_name in str(strongest.get("gpu_type")) for gpu_name in ["A40", "A800"]),
        "Strongest visible GPU selection must be recorded",
        {"strongest_visible_resource": strongest},
    )
    required_post_auth_checks = {"whoami", "hostname", "pwd", "python --version", "nvidia-smi", "df -hT", "free -h"}
    missing_post_auth_checks = sorted(required_post_auth_checks - set(gpu.get("post_authorization_checks") or []))
    require(
        not missing_post_auth_checks,
        "GPU post-authorization/Web Terminal proof checklist is incomplete",
        {"missing": missing_post_auth_checks, "hpc_gpu_ssh": gpu},
    )

    claude = resources["claude_code"]
    require(claude.get("status") in {"ready_via_deepseek_fallback", "awaiting_api_key"}, "Code Agent status must record DeepSeek fallback readiness or an explicit missing-key state", {"claude_code": claude})
    require(
        any("DEEPSEEK_API_KEY" in item or "ANTHROPIC_API_KEY" in item for item in (claude.get("required_env") or [])),
        "Code Agent required env must accept DeepSeek or Anthropic credentials",
        {"claude_code": claude},
    )
    if claude.get("status") == "ready_via_deepseek_fallback":
        require("DeepSeek" in str(claude.get("current_runtime_note")), "Code Agent DeepSeek fallback must be documented", {"claude_code": claude})
    require("Manual Gate" in str(claude.get("gate_policy")), "Code Agent gate policy must require manual approval", {"claude_code": claude})

    kaggle = resources["kaggle"]
    require(kaggle.get("status") in {"awaiting_token", "ready"}, "Kaggle must distinguish awaiting token from ready state", {"kaggle": kaggle})
    required_kaggle_env = set(kaggle.get("required_env") or [])
    require(
        "KAGGLE_API_TOKEN" in required_kaggle_env or {"KAGGLE_USERNAME", "KAGGLE_KEY"}.issubset(required_kaggle_env),
        "Kaggle required env is missing access-token or legacy credentials",
        {"kaggle": kaggle},
    )
    require("Human Gate" in str(kaggle.get("gate_policy")), "Kaggle official submission must require human gate", {"kaggle": kaggle})

    require(DOC.is_file(), "DeepSeek/HPC acceptance document is missing", {"path": str(DOC.relative_to(ROOT))})
    doc_text = DOC.read_text(encoding="utf-8")
    doc_terms = [
        "DeepSeek",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "SSH-2.0-SSHPiper",
        "Ncat: Version 7.92",
        "4 x A800",
        "nvidia-smi",
    ]
    missing_doc_terms = [term for term in doc_terms if term not in doc_text]
    require(not missing_doc_terms, "DeepSeek/HPC acceptance document is missing required evidence terms", {"missing_terms": missing_doc_terms})

    print(json.dumps({
        "status": "passed",
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "resources": sorted(resources),
        "deepseek_status": deepseek.get("status"),
        "hpc_gpu_status": gpu.get("status"),
        "hpc_current_blocker": gpu.get("current_blocker"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
