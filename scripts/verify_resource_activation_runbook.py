from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "正式上线资源接入运行手册.md"
SMOKE = ROOT / "scripts" / "run_real_resource_smoke.py"
VERIFIED_LAUNCHER = ROOT / "scripts" / "start_verified_workstation.ps1"
VERIFIED_AUDIT = ROOT / "scripts" / "verify_verified_workstation_launch_audit.py"

RUNBOOK_TERMS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_FILE",
    "GPU_SSH_HOST",
    "GPU_REMOTE_WORKSPACE",
    "GPU_SSH_KNOWN_HOSTS_PATH",
    "WORKSTATION_SECRET_DIR",
    "run_real_resource_smoke.py",
    "manage_hpc_proxy_bridge.ps1",
    "start_verified_workstation.ps1",
    "manage_deepseek_secret.ps1",
    "manage_claude_secret.ps1",
    "manage_kaggle_secret.ps1",
    "%APPDATA%\\ResearchAgentWorkstation",
    "verify_hpc_web_terminal_probe.py",
    "127.0.0.1:7890",
    "ncat --proxy 127.0.0.1:7890",
    "SSH-2.0-SSHPiper",
    "nvidia-smi",
    "free -h",
    "GPU Environment Created / Web Terminal Ready / External SSH Pending",
    "--require-configured",
    "不伪造成功",
    "白名单训练模板",
]

SMOKE_TERMS = [
    "verify_external_resource_gateways.py",
    "--allow-real-external",
    "run_full_acceptance.py",
    "blocked_missing_external_resources",
    "ANTHROPIC_API_KEY_FILE",
    "GPU_REMOTE_WORKSPACE",
]

VERIFIED_LAUNCHER_TERMS = [
    "Enable-InstalledDpapiSecrets",
    "deepseek_api_key.xml",
    "anthropic_api_key.xml",
    "kaggle_api_token.xml",
    "verify_deepseek_provider.py",
    "verify_external_resource_gateways.py",
    "manage_kaggle_secret.ps1",
    "verify_no_plaintext_secrets.py",
    "run_full_acceptance.py",
    "verified_workstation_launch_audit.json",
    "verified_workstation_launch_audit.md",
]

VERIFIED_AUDIT_TERMS = [
    "verified_workstation_launch_audit.json",
    "verified_workstation_launch_audit.md",
    "backend_resource_status",
    "deepseek_smoke",
    "external_gateway_smoke",
    "kaggle_secret_smoke",
    "plaintext_secret_scan",
]


def fail(message: str, evidence: dict) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence}, ensure_ascii=False, indent=2))


def require_terms(path: Path, terms: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [term for term in terms if term not in text]


def main() -> None:
    missing_files = [str(path.relative_to(ROOT)) for path in [RUNBOOK, SMOKE, VERIFIED_LAUNCHER, VERIFIED_AUDIT] if not path.exists()]
    if missing_files:
        fail("resource activation files missing", {"missing_files": missing_files})

    failures = {
        "runbook": require_terms(RUNBOOK, RUNBOOK_TERMS),
        "smoke_script": require_terms(SMOKE, SMOKE_TERMS),
        "verified_launcher": require_terms(VERIFIED_LAUNCHER, VERIFIED_LAUNCHER_TERMS),
        "verified_audit": require_terms(VERIFIED_AUDIT, VERIFIED_AUDIT_TERMS),
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        fail("resource activation runbook checks failed", failures)

    print(json.dumps({
        "status": "passed",
        "runbook": str(RUNBOOK.relative_to(ROOT)),
        "smoke_script": str(SMOKE.relative_to(ROOT)),
        "verified_launcher": str(VERIFIED_LAUNCHER.relative_to(ROOT)),
        "verified_audit": str(VERIFIED_AUDIT.relative_to(ROOT)),
        "conclusion": "Resource activation runbook and real-resource smoke command are ready.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
