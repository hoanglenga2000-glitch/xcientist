from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

try:
    from scripts.verify_no_plaintext_secrets import discover_candidate_files
except ModuleNotFoundError:  # Direct script execution adds scripts/ to sys.path.
    from verify_no_plaintext_secrets import discover_candidate_files

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PREFIXES = ("scripts/_quarantine/", "tests/")
PORTABILITY_EXEMPT_PATHS = {
    "scripts/build_reproducible_submission_package.py",
    "scripts/verify_external_resources_manifest.py",
    "scripts/verify_security_invariants.py",
}
PORTABILITY_PATTERNS = {
    "personal_hpc_mount": re.compile(r"/hpc2(?:hdd|ssd)/", re.IGNORECASE),
    "personal_hpc_account": re.compile(r"\b" + "aims" + r"lab\b", re.IGNORECASE),
    "personal_workspace_name": re.compile(r"(?:~[/\\])?" + "jing" + "hw", re.IGNORECASE),
    "windows_user_home": re.compile(r"C:\\Users\\[^\\/\r\n]+", re.IGNORECASE),
    "release_validation_path": re.compile("EvoMind-" + "release-validation", re.IGNORECASE),
}
FORBIDDEN_SSH_TOKENS = {
    "strict_host_key_checking_disabled": "StrictHostKeyChecking=" + "no",
    "strict_host_key_checking_tofu": "StrictHostKeyChecking=" + "accept-new",
    "known_hosts_disabled": "UserKnownHostsFile=" + "/dev/null",
    "paramiko_auto_add": "AutoAdd" + "Policy(",
    "paramiko_warning_policy": "Warning" + "Policy(",
    "paramiko_raw_transport": "paramiko." + "Transport(",
}
FORBIDDEN_MANAGER_ARGUMENTS = ("-ApiKey", "-ApiToken", "-Key", "-Password", "-ProxyPassword")
MANAGER_SCRIPT_NAMES = (
    "manage_deepseek_secret.ps1",
    "manage_kaggle_secret.ps1",
    "manage_hpc_ssh_secret.ps1",
    "manage_hpc_proxy_bridge.ps1",
)
FORBIDDEN_SECRET_FILE_GUIDANCE = (
    "edit .env and add " + "DEEPSEEK_API_KEY",
    "edit .env and add " + "ANTHROPIC_API_KEY",
    "edit .env and add " + "KAGGLE_API_TOKEN",
)
HPC_RUNTIME_SCRIPTS = (
    "scripts/gpu_monitor.py",
)
HPC_PROXY_RUNTIME_SCRIPTS = (
    "scripts/manage_hpc_proxy_bridge.ps1",
    "scripts/hpc_socks_bridge.py",
    "scripts/start_hpc_socks_bridge.py",
    "scripts/verify_hpc_socks_gateway.py",
)
DIRECT_SECRET_ENV_NAMES = {
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
MACHINE_SPECIFIC_RELEASE_TOKENS = {
    "historical_hpc_gateway": "100." + "85.169.63",
    "historical_hpc_inner_endpoint": "10." + "120.18.240",
    "historical_hpc_proxy": "8." + "163.52.223",
    "historical_hpc_account": "aimslab" + "-",
    "personal_windows_project_root": "D:/" + "桌面/codex/科研港科技",
}


def repository_files(root: Path = ROOT) -> tuple[list[Path], list[dict[str, object]]]:
    return discover_candidate_files(root)


def main() -> int:
    paths, discovery_findings = repository_files()
    findings: list[dict[str, object]] = [
        {
            "file": str(item.get("file", ".")),
            "rule": str(item.get("pattern", "inventory_discovery_error")),
            **({"error": item["error_type"]} if "error_type" in item else {}),
        }
        for item in discovery_findings
    ]
    scanned = 0
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            continue
        if relative.startswith(EXCLUDED_PREFIXES) or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            findings.append({"file": relative, "rule": "text_read_failed", "error": type(exc).__name__})
            continue
        scanned += 1
        for rule, token in FORBIDDEN_SSH_TOKENS.items():
            if token in text:
                findings.append({"file": relative, "rule": rule})
        for rule, token in MACHINE_SPECIFIC_RELEASE_TOKENS.items():
            if token in text:
                findings.append({"file": relative, "rule": rule})
        if relative not in PORTABILITY_EXEMPT_PATHS:
            for rule, pattern in PORTABILITY_PATTERNS.items():
                if pattern.search(text):
                    findings.append({"file": relative, "rule": rule})
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(name in line for name in MANAGER_SCRIPT_NAMES) and any(
                argument in line for argument in FORBIDDEN_MANAGER_ARGUMENTS
            ):
                findings.append(
                    {"file": relative, "line": line_number, "rule": "plaintext_secret_command_argument"}
                )
            if any(guidance in line for guidance in FORBIDDEN_SECRET_FILE_GUIDANCE):
                findings.append(
                    {"file": relative, "line": line_number, "rule": "plaintext_secret_file_guidance"}
                )
            if ("_safe_" + "input") in line and ("API " + "key") in line:
                findings.append(
                    {"file": relative, "line": line_number, "rule": "echoed_api_key_prompt"}
                )
            if re.search(r"\b(?:password|api_key|api_token|secret)\s*=.*\bsys\.argv\b", line, re.IGNORECASE):
                findings.append(
                    {"file": relative, "line": line_number, "rule": "secret_read_from_process_argv"}
                )

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["research-agent-workstation"]
    required_ports = {"127.0.0.1:3090:3090", "127.0.0.1:8088:3090"}
    if set(service.get("ports", [])) != required_ports or service.get("network_mode") == "host":
        findings.append({"file": "docker-compose.yml", "rule": "workstation_not_loopback_only"})
    compose_environment = service.get("environment", {})
    if isinstance(compose_environment, list):
        compose_environment = {
            str(item).partition("=")[0]: str(item).partition("=")[2] for item in compose_environment
        }
    for name in sorted(DIRECT_SECRET_ENV_NAMES.intersection(compose_environment)):
        findings.append(
            {"file": "docker-compose.yml", "rule": "direct_secret_environment_interpolation", "name": name}
        )
    for volume in service.get("volumes", []):
        volume_text = str(volume).replace("\\", "/")
        if "workspace/secrets" in volume_text or "/run/secrets/workstation" in volume_text:
            findings.append({"file": "docker-compose.yml", "rule": "implicit_project_secret_mount"})

    assignment = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
    for relative in (".env.example", "web/research-agent-workstation/.env.example"):
        names = set(assignment.findall((ROOT / relative).read_text(encoding="utf-8-sig")))
        for name in sorted(DIRECT_SECRET_ENV_NAMES.intersection(names)):
            findings.append({"file": relative, "rule": "direct_secret_template_variable", "name": name})

    for relative in HPC_RUNTIME_SCRIPTS:
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        if "Import-Clixml" in text or "hpc_ssh_credential.xml" in text:
            findings.append({"file": relative, "rule": "legacy_hpc_dpapi_runtime_load"})
        for variable in (
            "EVOMIND_HPC_HOST",
            "EVOMIND_HPC_PORT",
            "EVOMIND_HPC_USER",
            "EVOMIND_HPC_PASSWORD",
            "EVOMIND_HPC_REMOTE_WORKSPACE",
        ):
            if variable not in text:
                findings.append(
                    {"file": relative, "rule": "missing_hpc_runtime_variable", "variable": variable}
                )

    for relative in HPC_PROXY_RUNTIME_SCRIPTS:
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        if any(
            MACHINE_SPECIFIC_RELEASE_TOKENS[name] in text
            for name in ("historical_hpc_proxy", "historical_hpc_gateway")
        ):
            findings.append({"file": relative, "rule": "implicit_hpc_proxy_endpoint"})

    project_env_rules = {
        "install.ps1": ("Copy-Item (Join-Path $Root \".env.example\")",),
        "scripts/quick_setup.ps1": ("Copy-Item (Join-Path $Root \".env.example\")",),
        "scripts/quick_setup.sh": ('cp "$ROOT/.env.example" "$ROOT/.env"',),
        "scripts/run_evolution.py": ("load_dotenv", 'ROOT / ".env"'),
    }
    for relative, tokens in project_env_rules.items():
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        for token in tokens:
            if token in text:
                findings.append({"file": relative, "rule": "project_secret_file_runtime", "token": token})

    required_markers = {
        ".github/workflows/ci.yml": [
            "fetch-depth: 0",
            "name: history_secret_scan",
            "python scripts/verify_no_plaintext_secrets.py --history",
        ],
        "scripts/verify_no_plaintext_secrets.py": [
            "def scan_history_repository(",
            '"git", "-C", str(root), "rev-list", "--all"',
            '"git", "-C", str(root), "cat-file", "--batch"',
        ],
        "web/research-agent-workstation/src/middleware.ts": [
            "isLoopbackHostHeader",
            "isAllowedMutationSource",
            'matcher: "/api/:path*"',
        ],
        "web/research-agent-workstation/src/lib/server/paths.ts": [
            "path.relative(workspaceRoot, target)",
            "Workspace path escapes the workstation root",
        ],
        "web/research-agent-workstation/src/lib/server/gpu-ssh-gateway.ts": [
            "StrictHostKeyChecking=yes",
            "requireNetworkHost",
            "system_known_hosts_required",
        ],
        "scripts/dpapi_credential_store.ps1": [
            "Enter-EvoMindCredentialStoreLock",
            "Commit-EvoMindCredentialFiles",
            "Protect-EvoMindCredentialPath",
            "New-EvoMindHpcCredentialGeneration",
            "Read-EvoMindHpcCurrentGeneration",
            "hpc_ssh_current.json",
            "Invoke-EvoMindHpcBeforePointerTestHook",
        ],
        "scripts/start_verified_workstation.ps1": [
            "Resolve-EvoMindHpcCredentialGeneration",
            "EVOMIND_HPC_HOST",
        ],
        "scripts/manage_hpc_proxy_bridge.ps1": [
            "Read-EvoMindSecureInput",
            "Write-EvoMindCredentialTemp",
            "Protect-EvoMindCredentialPath",
            "HPC_SOCKS_PASSWORD",
            "start_hpc_socks_bridge.py",
            "HPC SOCKS upstream",
        ],
        "scripts/start_hpc_socks_bridge.py": [
            "subprocess.DETACHED_PROCESS",
            "close_fds=True",
            "subprocess.DEVNULL",
            "_SENSITIVE_ENV",
        ],
        "scripts/hpc_socks_bridge.py": [
            "listen host must be loopback",
            "upstream host and port must be configured explicitly",
        ],
        "src/xsci/config.py": [
            "xcientist.windows_dpapi_secrets.v1",
            "CryptProtectData",
            "_protect_windows_acl",
            "_scrub_legacy_secret_file",
            "secret_values='[redacted]'",
        ],
        "src/xsci/kaggle.py": [
            'getpass.getpass("  Anthropic API key (hidden)> ")',
            'getpass.getpass("  DeepSeek API key (hidden)> ")',
            'getpass.getpass("  OpenAI API key (hidden)> ")',
            'or "gpu").strip().lower()',
            "require_hpc_compute(effective_compute)",
            'set_global("compute", "backend", "gpu")',
        ],
        "src/xsci/login.py": [
            "command-line secret values are disabled",
            "protected user storage",
        ],
        "src/xsci/__main__.py": [
            'log.add_argument("--api-key", help=argparse.SUPPRESS)',
            'log.add_argument("--kaggle-key", help=argparse.SUPPRESS)',
            "secret configuration is never displayed",
            '_PUBLIC_COMPUTE_CHOICES = ["gpu"]',
            'default="gpu"',
        ],
        "src/xsci/project.py": [
            'compute: str = "gpu"',
            "require_hpc_compute(compute)",
        ],
        "src/xsci/agent.py": [
            "require_hpc_compute(plan.compute)",
            'compute = "remote HPC/GPU"',
        ],
        "src/xsci/terminal_agent.py": [
            "require_hpc_compute(effective_compute)",
            'emitter.emit("Selecting compute", "compute=gpu"',
        ],
        "src/research_os/evolution_loop.py": [
            "blocked_local_training_disabled: LocalSubprocessRunner is unavailable",
        ],
        "src/research_agent_workstation/tabular_pipeline.py": [
            "blocked_local_training_disabled: local tabular training is disabled",
            '"training_started": False',
        ],
        "scripts/run_local_sklearn_ensemble.py": [
            '"status": "blocked_local_training_disabled"',
            '"training_started": False',
        ],
        "scripts/quick_setup.sh": [
            "No project secret file created",
            "xsci login",
        ],
        "scripts/quick_setup.ps1": [
            "No project secret file created",
            "evomind setup",
        ],
        "scripts/run_evolution.py": [
            "inject_engine_env",
            "load_config(project_root=ROOT)",
            "override=False",
        ],
        "install.ps1": [
            "created non-secret web .env",
            "Legacy root .env detected",
            "evomind setup",
        ],
        "scripts/hpc_runtime_contract.py": [
            "EVOMIND_HPC_HOST",
            "EVOMIND_HPC_PORT",
            "EVOMIND_HPC_USER",
            "EVOMIND_HPC_PASSWORD",
            "EVOMIND_HPC_REMOTE_WORKSPACE",
            "configure both or neither",
        ],
        "configs/external_resources.yaml": [
            'schema: "evomind.external_resources.v2"',
            'status: "not_configured"',
            "No host, account, allocation, private path, or historical readiness evidence",
        ],
        "src/research_agent_workstation/server/strategy/harness_optimizer.py": [
            "Local harness training is disabled by policy",
            'ROOT / "tasks" / task_id / "data"',
        ],
        "src/research_os/hpc_policy.py": [
            "EVOMIND_HPC_REMOTE_WORKSPACE must be configured explicitly",
            "Local training is disabled by release policy",
            "remote workspace must be a dedicated project directory",
        ],
        "scripts/verify_launch_resource_readiness.py": [
            '"overall_status": "blocked_hpc_runtime_verification_required"',
            '"legacy_local_evidence": "not_release_readiness"',
        ],
        "scripts/verify_training_optimization_readiness.py": [
            '"overall_status": "blocked_hpc_runtime_verification_required"',
            '"legacy_local_evidence": "not_release_readiness"',
        ],
        "scripts/verify_final_two_resource_blockers.py": [
            '"overall_status": "blocked_hpc_runtime_verification_required"',
            '"legacy_local_evidence": "not_release_readiness"',
        ],
        "web/research-agent-workstation/src/lib/server/runs.ts": [
            "async function localTrainingFallbackDisabled()",
            "return true;",
            "blocked_local_training_disabled",
        ],
    }
    for relative, markers in required_markers.items():
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        for marker in markers:
            if marker not in text:
                findings.append({"file": relative, "rule": "missing_security_marker", "marker": marker})

    payload = {
        "status": "passed" if not findings else "failed",
        "scanned_text_files": scanned,
        "findings": findings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
