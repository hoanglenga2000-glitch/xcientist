from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CODE_AGENT_KEYS = ["DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "CLAUDE_API_KEY", "CLAUDE_API_KEY_FILE"]
GPU_VALUE_KEYS = ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_REMOTE_WORKSPACE"]
GPU_REQUIRED_LABELS = ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH", "GPU_REMOTE_WORKSPACE"]


def read_file_if_present(file_path: str | None) -> str:
    if not file_path:
        return ""
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def secret_dir_file(names: list[str]) -> Path | None:
    secret_dir = os.environ.get("WORKSTATION_SECRET_DIR")
    if not secret_dir:
        return None
    for name in names:
        candidate = Path(secret_dir) / name
        if candidate.exists():
            return candidate
    return None


def secret_value(key: str, aliases: list[str] | None = None) -> str:
    keys = [key, *(aliases or [])]
    for candidate in keys:
        direct = os.environ.get(candidate)
        if direct:
            return direct
        file_value = read_file_if_present(os.environ.get(f"{candidate}_FILE"))
        if file_value:
            return file_value
    dir_file = secret_dir_file(keys)
    return read_file_if_present(str(dir_file) if dir_file else None)


def secret_path(key: str, names: list[str] | None = None) -> str:
    candidates = [key, *(names or [])]
    direct = os.environ.get(key) or os.environ.get(f"{key}_FILE")
    if direct:
        return direct
    dir_file = secret_dir_file(candidates)
    return str(dir_file) if dir_file else ""


def code_agent_configured() -> bool:
    return bool(secret_value("DEEPSEEK_API_KEY") or secret_value("ANTHROPIC_API_KEY", ["CLAUDE_API_KEY"]))


def missing_gpu_keys() -> list[str]:
    missing = [key for key in GPU_VALUE_KEYS if not secret_value(key)]
    if not (secret_value("GPU_SSH_PASSWORD", ["HPC_SSH_PASSWORD"]) or secret_path("GPU_SSH_KEY_PATH", ["GPU_SSH_PRIVATE_KEY", "gpu_ssh_private_key", "id_rsa"])):
        missing.append("GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH")
    return missing


def run(command: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_json_output(result: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return None


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def connector_ready(summary: dict[str, Any], key: str) -> bool:
    connector = (summary.get("connector_status") or {}).get(key) or {}
    return bool(connector.get("configured"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real external-resource smoke tests once Code Agent and GPU SSH resources are configured."
    )
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8088")
    parser.add_argument("--container-name", default="research-agent-workstation")
    parser.add_argument("--require-configured", action="store_true", help="Fail if Claude or GPU resources are missing.")
    parser.add_argument("--skip-full-acceptance", action="store_true", help="Only run external gateway smoke tests.")
    args = parser.parse_args()

    base_url = args.dashboard_url.rstrip("/")
    summary = get_json(f"{base_url}/api/workstation-summary")
    backend_code_agent_ready = connector_ready(summary, "code_agent")
    backend_gpu_ready = connector_ready(summary, "gpu")
    local_code_agent_ready = code_agent_configured()
    gpu_missing = missing_gpu_keys()
    local_gpu_ready = not gpu_missing
    code_agent_ready = backend_code_agent_ready
    gpu_ready = backend_gpu_ready
    resource_status = {
        "code_agent": {
            "ready": code_agent_ready,
            "backend_ready": backend_code_agent_ready,
            "local_secret_probe_ready": local_code_agent_ready,
            "accepted_inputs": CODE_AGENT_KEYS,
            "missing": [] if code_agent_ready else ["DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, CLAUDE_API_KEY, *_FILE, or WORKSTATION_SECRET_DIR"],
            "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
        },
        "gpu": {
            "ready": gpu_ready,
            "backend_ready": backend_gpu_ready,
            "local_secret_probe_ready": local_gpu_ready,
            "required_inputs": GPU_REQUIRED_LABELS,
            "missing": [] if gpu_ready else gpu_missing,
            "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
            "optional": ["GPU_SSH_PORT", "GPU_SSH_KNOWN_HOSTS_PATH", "GPU_SSH_KNOWN_HOSTS_PATH_FILE"],
        },
    }

    if args.require_configured and (not code_agent_ready or not gpu_ready):
        raise SystemExit(json.dumps({
            "status": "blocked_missing_external_resources",
            "resource_status": resource_status,
        }, ensure_ascii=False, indent=2))

    commands = [
        [
            sys.executable,
            "scripts/verify_external_resource_gateways.py",
            "--url",
            base_url,
            "--container-name",
            args.container_name,
            "--allow-real-external",
        ]
    ]
    if not args.skip_full_acceptance:
        commands.append([
            sys.executable,
            "scripts/run_full_acceptance.py",
            "--dashboard-url",
            base_url,
            "--container-name",
            args.container_name,
        ])

    results = []
    for command in commands:
        result = run(command)
        results.append({**result, "json": parse_json_output(result)})
        if result["returncode"] != 0:
            raise SystemExit(json.dumps({
                "status": "failed",
                "resource_status": resource_status,
                "failed_command": result,
                "results": results,
            }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "status": "passed",
        "resource_status": resource_status,
        "real_external_invocation": {
            "code_agent_invoked_if_configured": code_agent_ready,
            "gpu_invoked_if_configured": gpu_ready,
            "not_configured_paths_remain_audited": not (code_agent_ready and gpu_ready),
        },
        "results": results,
        "next_step": (
            "All external resources are configured; review generated Code Agent/GPU artifacts and promote through manual Gate."
            if code_agent_ready and gpu_ready
            else "Configure the missing external resources, then rerun with --require-configured."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
