from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def require_contains(path: Path, terms: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [term for term in terms if term not in text]


def main() -> None:
    files = {
        "capabilities": ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "capabilities.ts",
        "claude_sessions": ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "claude-agent-sessions.ts",
        "gpu_gateway": ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "gpu-ssh-gateway.ts",
        "deepseek_provider": ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "deepseek-provider.ts",
        "deepseek_route": ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "llm" / "deepseek" / "smoke" / "route.ts",
        "docker_compose": ROOT / "docker-compose.yml",
        "root_env_example": ROOT / ".env.example",
        "web_env_example": ROOT / "web" / "research-agent-workstation" / ".env.example",
    }

    missing_files = [name for name, path in files.items() if not path.exists()]
    if missing_files:
        fail("required launch integration files are missing", {"missing_files": missing_files})

    checks = {
        "secret_file_support": {
            "file": "capabilities",
            "terms": [
                "WORKSTATION_SECRET_DIR",
                "ANTHROPIC_API_KEY_FILE",
                "DEEPSEEK_API_KEY_FILE",
                "deepSeekApiKeyValue",
                "claudeApiKeyValue",
                "GPU_SSH_KNOWN_HOSTS_PATH",
                "GPU_SSH_SOCKS_HOST",
                "GPU_SSH_SOCKS_PASSWORD",
                "GPU_REMOTE_WORKSPACE",
            ],
        },
        "claude_sdk_secret_injection": {
            "file": "claude_sessions",
            "terms": [
                "claudeApiKeyValue",
                "process.env.ANTHROPIC_API_KEY",
                "@anthropic-ai/claude-agent-sdk",
                "disallowedTools",
                "permissionMode: \"dontAsk\"",
            ],
        },
        "gpu_host_key_and_whitelist": {
            "file": "gpu_gateway",
            "terms": [
                "StrictHostKeyChecking=yes",
                "UserKnownHostsFile",
                "host_key_policy",
                "custom_known_hosts_required",
                "system_known_hosts_required",
                "ProxyCommand",
                "hpc_socks_proxy.py",
                "proxy_policy",
                "allowedTemplates",
                "gpu_job_template_rejected",
                "telco_churn_seed_sweep",
                "all_tasks_seed_sweep",
            ],
        },
        "deepseek_provider_smoke": {
            "file": "deepseek_provider",
            "terms": [
                "DEEPSEEK_API_KEY",
                "deepseek-v4-flash",
                "chat/completions",
                "deepseek_smoke_passed",
                "writeSmokeArtifact",
            ],
        },
        "deepseek_api_route": {
            "file": "deepseek_route",
            "terms": [
                "runDeepSeekSmoke",
                "force-dynamic",
                "POST",
            ],
        },
        "docker_secure_local_default": {
            "file": "docker_compose",
            "terms": [
                "LLM_PROVIDER: rule_based",
                "DEEPSEEK_MODEL",
                "GPU_SSH_HOST",
                "GPU_SSH_USER",
                "GPU_SSH_KEY_PATH",
                "GPU_REMOTE_WORKSPACE",
                "GPU_SSH_KNOWN_HOSTS_PATH",
                "GPU_SSH_SOCKS_HOST",
                "127.0.0.1:3090:3090",
                "127.0.0.1:8088:3090",
            ],
        },
        "env_template_completeness": {
            "file": "root_env_example",
            "terms": [
                "ANTHROPIC_API_KEY_FILE",
                "DEEPSEEK_API_KEY_FILE",
                "OPENAI_API_KEY_FILE",
                "DEEPSEEK_MODEL",
                "GPU_SSH_KEY_PATH_FILE",
                "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
                "GPU_SSH_SOCKS_HOST",
                "GPU_SSH_SOCKS_PASSWORD_FILE",
                "WORKSTATION_SECRET_DIR",
                "Do not persist database credentials",
            ],
        },
        "web_env_template_completeness": {
            "file": "web_env_example",
            "terms": [
                "DATABASE_URL=\"file:./workstation.db\"",
                "ANTHROPIC_API_KEY_FILE",
                "DEEPSEEK_API_KEY_FILE",
                "DEEPSEEK_MODEL",
                "GPU_REMOTE_WORKSPACE",
                "GPU_SSH_SOCKS_HOST",
                "GPU_SSH_SOCKS_PASSWORD_FILE",
                "WORKSTATION_SECRET_DIR",
                "Do not persist database credentials",
            ],
        },
    }

    failures = {}
    for check_name, spec in checks.items():
        missing = require_contains(files[spec["file"]], spec["terms"])
        if missing:
            failures[check_name] = {"file": str(files[spec["file"]].relative_to(ROOT)), "missing_terms": missing}

    direct_secret_names = {
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "GPU_SSH_PASSWORD",
        "GPU_SSH_SOCKS_PASSWORD",
        "HPC_SOCKS_PASSWORD",
    }
    assignment = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
    for file_key in ("root_env_example", "web_env_example"):
        names = set(assignment.findall(files[file_key].read_text(encoding="utf-8-sig")))
        direct = sorted(names.intersection(direct_secret_names))
        if direct:
            failures[f"{file_key}_direct_secrets"] = {
                "file": str(files[file_key].relative_to(ROOT)),
                "variables": direct,
            }

    compose = yaml.safe_load(files["docker_compose"].read_text(encoding="utf-8-sig"))
    service = compose["services"]["research-agent-workstation"]
    compose_env = service.get("environment", {})
    if isinstance(compose_env, list):
        compose_env = {str(item).partition("=")[0]: str(item).partition("=")[2] for item in compose_env}
    direct = sorted(set(compose_env).intersection(direct_secret_names))
    if direct:
        failures["docker_direct_secret_interpolation"] = {
            "file": "docker-compose.yml",
            "variables": direct,
        }
    unsafe_mounts = [
        str(volume)
        for volume in service.get("volumes", [])
        if "workspace/secrets" in str(volume).replace("\\", "/")
        or "/run/secrets/workstation" in str(volume).replace("\\", "/")
    ]
    if unsafe_mounts:
        failures["docker_implicit_project_secret_mount"] = {
            "file": "docker-compose.yml",
            "mounts": unsafe_mounts,
        }

    if failures:
        fail("launch integration hardening checks failed", failures)

    result = {
        "status": "passed",
        "checks": sorted(checks),
        "conclusion": (
            "Claude Code, GPU SSH, and secret-file configuration are wired without project secret templates; "
            "the default Docker profile is loopback-only and does not interpolate or mount project secrets."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
