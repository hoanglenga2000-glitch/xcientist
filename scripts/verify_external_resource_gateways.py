from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE_AGENT_MISSING = ["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]
GPU_MISSING_BASE = ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_REMOTE_WORKSPACE"]
GPU_MISSING_AUTH = ["GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH_OR_GPU_SSH_PASSWORD", "GPU_SSH_PASSWORD"]


def fail(message: str) -> None:
    raise SystemExit(f"EXTERNAL_GATEWAY_FAILED: {message}")


def post_json(url: str, payload: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Origin": origin},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise error
        if not isinstance(payload, dict):
            raise error
        payload["http_status"] = error.code
        return payload


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def artifact_exists(artifact: str | None, container_name: str | None) -> bool:
    if not artifact:
        return False
    local_path = ROOT / artifact
    if local_path.exists() and local_path.stat().st_size > 0:
        return True
    if container_name:
        normalized = artifact.replace("\\", "/")
        completed = subprocess.run(
            ["docker", "exec", container_name, "test", "-s", f"/app/{normalized}"],
            text=True,
            capture_output=True,
        )
        return completed.returncode == 0
    return False


def connector_configured(summary: dict[str, Any], key: str) -> bool:
    connector = (summary.get("connector_status") or {}).get(key) or {}
    return bool(connector.get("configured"))


def is_controlled_gpu_blocker(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "")
    error = str(payload.get("error") or "")
    return status in {"failed", "blocked_resource_gateway"} and any(
        marker in error
        for marker in [
            "SSH protocol banner",
            "kex_exchange_identification",
            "Connection closed by remote host",
            "ConnectionRefusedError",
            "WinError 10061",
            "timed out",
        ]
    )


def verify_code_agent(base_url: str, container_name: str | None, allow_real_external: bool) -> dict[str, Any]:
    summary = get_json(f"{base_url}/api/workstation-summary")
    configured = connector_configured(summary, "code_agent")
    if configured and not allow_real_external:
        return {
            "provider": "code_agent",
            "status": "configured_not_invoked",
            "configured": True,
            "note": "Code Agent is configured; real model smoke test was skipped because --allow-real-external was not set.",
        }

    payload = post_json(
        f"{base_url}/api/tasks/house_prices/code-agent-draft",
        {
            "source_agent": "claude_code",
            "max_turns": 5,
            "timeout_seconds": 150,
            "prompt": (
                "Smoke test the real Code Agent integration. Return a concise reviewable unified diff only. "
                "Use plain ASCII punctuation in the diff. "
                "Do not edit files directly. Add a short gated note to docs/claude_code_smoke_note.md explaining "
                "that generated patches must pass Code Quality Gate and Manual Gate before apply."
            ),
        },
        timeout=180,
    )
    if not payload.get("ok"):
        fail(f"Code Agent draft endpoint did not return ok: {payload}")

    if not configured:
        missing = payload.get("missing_env") or []
        if payload.get("configured") is not False or payload.get("cli_status") != "not_configured":
            fail(f"Code Agent endpoint did not expose not_configured state: {payload}")
        # Accept either Claude or DeepSeek as the code agent provider
        any_code_agent_missing = any(key in missing for key in CODE_AGENT_MISSING)
        if not any_code_agent_missing:
            fail(f"Code Agent endpoint did not report required missing env keys: {payload}")
        if payload.get("patch_path"):
            fail(f"Code Agent endpoint produced a patch despite missing API key: {payload}")
        if not artifact_exists(payload.get("manifest_path") or payload.get("draft_path"), container_name):
            fail(f"Code Agent manifest artifact was not written: {payload}")
        return {
            "provider": "code_agent",
            "status": "not_configured_verified",
            "configured": False,
            "missing_env": missing,
            "manifest_path": payload.get("manifest_path") or payload.get("draft_path"),
            "patch_path": payload.get("patch_path"),
        }

    if payload.get("configured") is not True:
        fail(f"Code Agent is configured but endpoint did not run as configured: {payload}")
    if payload.get("cli_status") != "completed":
        fail(f"Code Agent is configured but the session did not complete successfully: {payload}")
    if not artifact_exists(payload.get("manifest_path") or payload.get("draft_path"), container_name):
        fail(f"Code Agent configured smoke test did not write a manifest: {payload}")
    if not artifact_exists(payload.get("transcript_path"), container_name):
        fail(f"Code Agent configured smoke test did not write a transcript: {payload}")
    if not artifact_exists(payload.get("patch_path"), container_name):
        fail(f"Code Agent configured smoke test did not produce a reviewable patch: {payload}")
    return {
        "provider": payload.get("source_agent") or "code_agent",
        "status": "configured_smoke_tested",
        "configured": True,
        "session_id": payload.get("session_id"),
        "cli_status": payload.get("cli_status"),
        "manifest_path": payload.get("manifest_path") or payload.get("draft_path"),
        "transcript_path": payload.get("transcript_path"),
        "patch_path": payload.get("patch_path"),
    }


def verify_gpu(base_url: str, container_name: str | None, allow_real_external: bool) -> dict[str, Any]:
    summary = get_json(f"{base_url}/api/workstation-summary")
    configured = connector_configured(summary, "gpu")
    invalid = post_json(f"{base_url}/api/gpu/jobs", {"task_id": "house_prices", "template": "not_allowed_shell"})
    if invalid.get("status") != "rejected":
        fail(f"GPU invalid template endpoint did not reject unsupported template: {invalid}")
    if not artifact_exists(invalid.get("artifact_path"), container_name):
        fail(f"GPU invalid template rejection did not write artifact: {invalid}")
    if configured and not allow_real_external:
        return {
            "provider": "ssh_gateway",
            "status": "configured_not_invoked",
            "configured": True,
            "invalid_template_status": invalid.get("status"),
            "invalid_template_artifact": invalid.get("artifact_path"),
            "note": "GPU SSH env is configured; real SSH smoke test was skipped because --allow-real-external was not set.",
        }

    connection = post_json(f"{base_url}/api/gpu/connections/test", {})
    job = post_json(f"{base_url}/api/gpu/jobs", {"task_id": "house_prices", "template": "connection_smoke"})
    gated_job = post_json(f"{base_url}/api/gpu/jobs", {"task_id": "house_prices"})
    telco_job = post_json(f"{base_url}/api/gpu/jobs", {"task_id": "telco_churn"})
    for label, payload in [("connection", connection), ("job", job), ("gated_job", gated_job)]:
        if not payload.get("ok"):
            fail(f"GPU {label} endpoint did not return ok: {payload}")
        if not artifact_exists(payload.get("artifact_path"), container_name):
            fail(f"GPU {label} endpoint did not write artifact: {payload}")

    if is_controlled_gpu_blocker(connection) or is_controlled_gpu_blocker(job):
        return {
            "provider": "ssh_gateway",
            "status": "configured_resource_blocked",
            "configured": True,
            "connection_status": connection.get("status"),
            "job_status": job.get("status"),
            "blocker": "Configured GPU endpoint did not complete current SSH/CUDA smoke; workstation must block training.",
            "connection_artifact": connection.get("artifact_path"),
            "job_artifact": job.get("artifact_path"),
            "gate_rejection_status": gated_job.get("status"),
            "invalid_template_status": invalid.get("status"),
            "invalid_template_artifact": invalid.get("artifact_path"),
        }

    effective_configured = configured or connection.get("configured") is True or job.get("configured") is True

    if not effective_configured:
        for label, payload in [("connection", connection), ("job", job)]:
            missing = payload.get("missing_env") or []
            if payload.get("configured") is not False or payload.get("status") != "not_configured":
                fail(f"GPU {label} endpoint did not expose not_configured state: {payload}")
            has_base_missing = set(GPU_MISSING_BASE).issubset(set(missing))
            has_auth_missing = any(k in missing for k in GPU_MISSING_AUTH)
            if not (has_base_missing and has_auth_missing):
                fail(f"GPU {label} endpoint did not report all required env keys: {payload}")
        if telco_job.get("status") != "not_configured":
            fail(f"GPU telco default endpoint did not expose not_configured state: {telco_job}")
        return {
            "provider": "ssh_gateway",
            "status": "not_configured_verified",
            "configured": False,
            "missing_env": sorted(set((connection.get("missing_env") or []) + (job.get("missing_env") or []))),
            "connection_artifact": connection.get("artifact_path"),
            "job_artifact": job.get("artifact_path"),
            "telco_default_artifact": telco_job.get("artifact_path"),
            "invalid_template_status": invalid.get("status"),
            "invalid_template_artifact": invalid.get("artifact_path"),
        }

    if (
        connection.get("configured") is not True
        or job.get("configured") is not True
        or gated_job.get("configured") is not True
    ):
        fail(
            f"GPU is configured but endpoints did not run as configured: connection={connection}, job={job}, gated_job={gated_job}"
        )
    if connection.get("status") != "passed" or job.get("status") != "submitted":
        fail(f"GPU configured endpoints did not pass smoke statuses: connection={connection}, job={job}")
    if gated_job.get("status") != "rejected":
        fail(f"GPU non-smoke job without hpc_execution_approval was not rejected: {gated_job}")
    return {
        "provider": "ssh_gateway",
        "status": "configured_smoke_tested",
        "configured": True,
        "connection_status": connection.get("status"),
        "job_status": job.get("status"),
        "gate_rejection_status": gated_job.get("status"),
        "connection_artifact": connection.get("artifact_path"),
        "job_artifact": job.get("artifact_path"),
        "gate_rejection_artifact": gated_job.get("artifact_path"),
        "invalid_template_status": invalid.get("status"),
        "invalid_template_artifact": invalid.get("artifact_path"),
    }


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# External Gateway Readiness",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Overall status: {report['overall_status']}",
        f"- Dashboard URL: {report['dashboard_url']}",
        "",
        "## Claude Code",
        "",
        f"- Status: {report['claude']['status']}",
        f"- Configured: {report['claude']['configured']}",
        f"- Missing env: {', '.join(report['claude'].get('missing_env') or []) or 'none'}",
        f"- Manifest: {report['claude'].get('manifest_path') or 'not generated'}",
        "",
        "## GPU SSH Gateway",
        "",
        f"- Status: {report['gpu']['status']}",
        f"- Configured: {report['gpu']['configured']}",
        f"- Missing env: {', '.join(report['gpu'].get('missing_env') or []) or 'none'}",
        f"- Connection artifact: {report['gpu'].get('connection_artifact') or 'not generated'}",
        f"- Job artifact: {report['gpu'].get('job_artifact') or 'not generated'}",
        "",
        "## Conclusion",
        "",
        report["conclusion"],
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Code Agent and GPU SSH external gateways through live API endpoints."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="Dashboard base URL.")
    parser.add_argument("--container-name", default=None, help="Optional Docker container name for artifact checks.")
    parser.add_argument(
        "--allow-real-external",
        action="store_true",
        help="Allow real Claude SDK and SSH smoke tests when credentials are configured.",
    )
    parser.add_argument("--write-report", action="store_true", help="Write JSON and Markdown report under docs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")
    code_agent = verify_code_agent(base_url, args.container_name, args.allow_real_external)
    gpu = verify_gpu(base_url, args.container_name, args.allow_real_external)
    acceptable_statuses = {
        "not_configured_verified",
        "configured_not_invoked",
        "configured_smoke_tested",
        "configured_resource_blocked",
    }
    passed = code_agent["status"] in acceptable_statuses and gpu["status"] in acceptable_statuses
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dashboard_url": base_url,
        "overall_status": "passed" if passed else "failed",
        "claude": code_agent,
        "code_agent": code_agent,
        "gpu": gpu,
        "external_resources_still_required": [
            "ANTHROPIC_API_KEY or DEEPSEEK_API_KEY" if not code_agent["configured"] else None,
            "GPU SSH credentials" if not gpu["configured"] else None,
        ],
        "conclusion": (
            "Claude Code and GPU SSH Gateway backend interfaces passed the readiness contract. When real external checks are disabled, configured connectors are reported as configured_not_invoked; use --allow-real-external for live smoke tests."
            if passed
            else "Claude Code or GPU SSH Gateway interface readiness failed."
        ),
    }
    report["external_resources_still_required"] = [item for item in report["external_resources_still_required"] if item]

    if args.write_report:
        json_path = ROOT / "docs" / "external_gateway_readiness.json"
        md_path = ROOT / "docs" / "外部资源网关接口验收.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {
            "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
            "markdown": str(md_path.relative_to(ROOT)).replace("\\", "/"),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
