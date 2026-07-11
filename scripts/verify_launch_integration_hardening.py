from __future__ import annotations

import json
from pathlib import Path

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
                "StrictHostKeyChecking=accept-new",
                "UserKnownHostsFile",
                "host_key_policy",
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
        "docker_external_resource_env": {
            "file": "docker_compose",
            "terms": [
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_MODEL",
                "GPU_SSH_HOST",
                "GPU_SSH_USER",
                "GPU_SSH_KEY_PATH",
                "GPU_REMOTE_WORKSPACE",
                "GPU_SSH_KNOWN_HOSTS_PATH",
                "GPU_SSH_SOCKS_HOST",
                "GPU_SSH_SOCKS_PASSWORD",
                "WORKSTATION_SECRET_DIR",
                "/run/secrets/workstation",
            ],
        },
        "env_template_completeness": {
            "file": "root_env_example",
            "terms": [
                "ANTHROPIC_API_KEY_FILE",
                "DEEPSEEK_API_KEY_FILE",
                "DEEPSEEK_MODEL",
                "GPU_SSH_KEY_PATH_FILE",
                "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
                "GPU_SSH_SOCKS_HOST",
                "GPU_SSH_SOCKS_PASSWORD_FILE",
                "WORKSTATION_SECRET_DIR",
                "DATABASE_URL=postgresql://",
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
                "DATABASE_URL=\"postgresql://",
            ],
        },
    }

    failures = {}
    for check_name, spec in checks.items():
        missing = require_contains(files[spec["file"]], spec["terms"])
        if missing:
            failures[check_name] = {"file": str(files[spec["file"]].relative_to(ROOT)), "missing_terms": missing}

    if failures:
        fail("launch integration hardening checks failed", failures)

    result = {
        "status": "passed",
        "checks": sorted(checks),
        "conclusion": (
            "Claude Code, GPU SSH, secret-file configuration, Docker env propagation, "
            "and production database templates are wired for real-resource smoke tests without storing secrets in SQLite."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
