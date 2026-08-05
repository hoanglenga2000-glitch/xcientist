#!/usr/bin/env python3
"""Build immutable execution-environment evidence for an MLE Lite A/B campaign.

This script is deliberately pre-execution only: it performs an authenticated
metadata request against the loopback model gateway and never asks for a chat
completion, starts a task-run, runs a grader, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in os.sys.path:
        os.sys.path.insert(0, str(import_root))

from research_os.mle_ab_campaign import ARMS, canonical_bytes, load_campaign, sha256_file  # noqa: E402
from scripts.verify_mle_ab_campaign_readiness import (  # noqa: E402
    AUTH_NO_COMPLETION_PROBE_SCHEMA,
    EXECUTION_ENVIRONMENT_SCHEMA,
    HOST_FINGERPRINT_SCHEMA,
    SAME_HARDWARE_POLICY_SCHEMA,
    SAME_HARDWARE_RECEIPT_SCHEMA,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_gateway_key(config_path: Path | None = None) -> tuple[str, str | None]:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "environment:OPENAI_API_KEY"
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(config_path.expanduser())
    env_config = os.environ.get("EVOMIND_LOCAL_GATEWAY_CONFIG", "").strip()
    if env_config:
        candidates.append(Path(env_config).expanduser())
    candidates.append(Path.home() / ".antigravity_cockpit" / "codex_local_access_sidecar" / "config.json")
    for candidate in candidates:
        try:
            if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size <= 1024 * 1024:
                payload = _read_json(candidate)
                keys = payload.get("api-keys")
                if isinstance(keys, list):
                    key = next((str(item).strip() for item in keys if str(item).strip()), "")
                    if key:
                        return key, str(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("loopback gateway API key not found")


def authenticated_no_completion_probe(base_url: str, api_key: str, *, timeout: float = 10.0) -> dict[str, Any]:
    clean = str(base_url or "").rstrip("/")
    if not clean:
        raise ValueError("base_url is required")
    # Standard OpenAI-compatible metadata endpoint.  Do not call chat/completions.
    url = clean + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    started = utc_now()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback/explicit URL
            body = response.read(1024 * 1024)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read(1024 * 1024)
        status = int(exc.code)
    payload_sha = hashlib.sha256(body).hexdigest()
    return {
        "schema": AUTH_NO_COMPLETION_PROBE_SCHEMA,
        "request_kind": "authenticated_metadata",
        "endpoint": "/models",
        "http_status": status,
        "authentication_succeeded": 200 <= status < 300,
        "completion_requested": False,
        "response_sha256": payload_sha,
        "observed_at": started,
    }


def build_host_fingerprint(*, salt: str = "evomind-mle-lite-r4-host-v1") -> dict[str, Any]:
    raw_machine = "|".join([
        platform.node(),
        platform.system(),
        platform.machine(),
        str(uuid.getnode()),
        socket.gethostname(),
    ])
    return {
        "schema": HOST_FINGERPRINT_SCHEMA,
        "machine_id_sha256": sha256_text(salt + ":" + raw_machine),
        "os_family": platform.system() or os.name,
        "cpu_architecture": platform.machine() or platform.processor() or "unknown",
    }


def build_execution_environment(
    campaign_dir: str | Path,
    *,
    base_url: str = "http://127.0.0.1:65068/v1",
    gateway_config: str | Path | None = None,
    probe_timeout: float = 10.0,
) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, _schedule = load_campaign(campaign)
    lock_sha256 = sha256_file(campaign / "campaign-lock.json")
    api_key, key_source = resolve_gateway_key(Path(gateway_config) if gateway_config else None)
    probe = authenticated_no_completion_probe(base_url, api_key, timeout=probe_timeout)
    fingerprint = build_host_fingerprint()
    fingerprint_sha256 = canonical_sha256(fingerprint)
    policy = {
        "schema": SAME_HARDWARE_POLICY_SCHEMA,
        "receipts_required_for_every_run": True,
        "receipt_schema": SAME_HARDWARE_RECEIPT_SCHEMA,
        "pair_key_fields": ["task_id", "seed"],
        "paired_arms": list(ARMS),
        "host_fingerprint_sha256": fingerprint_sha256,
    }
    payload = {
        "schema": EXECUTION_ENVIRONMENT_SCHEMA,
        "schema_version": 2,
        "created_at": utc_now(),
        "namespace": prereg["namespace"],
        "campaign_lock_sha256": lock_sha256,
        "provider": prereg["provider"],
        "model": prereg["model"],
        "authenticated_no_completion_probe": probe,
        "host_fingerprint": fingerprint,
        "host_fingerprint_sha256": fingerprint_sha256,
        "same_hardware_policy": policy,
        "completion_requested": False,
        "task_runs_started": 0,
        "grader_calls": 0,
        "kaggle_submissions": 0,
        "credential_source_sha256": sha256_text(str(key_source or "unknown")),
        "claim_boundary": "pre-execution environment binding only; no completions, grading, task runs, or submissions",
    }
    # Validate the no-completion invariant locally without retaining secrets.
    if not probe["authentication_succeeded"]:
        payload["status"] = "failed_closed"
    else:
        payload["status"] = "bound_pre_execution"
    return payload


def write_json_atomic(path: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"))
    parser.add_argument("--gateway-config", type=Path)
    parser.add_argument("--probe-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_execution_environment(
        args.campaign,
        base_url=args.base_url,
        gateway_config=args.gateway_config,
        probe_timeout=args.probe_timeout,
    )
    output = write_json_atomic(args.output, payload, overwrite=args.overwrite)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "completion_requested": payload["completion_requested"],
        "probe_http_status": payload["authenticated_no_completion_probe"]["http_status"],
        "host_fingerprint_sha256": payload["host_fingerprint_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "bound_pre_execution" else 2


if __name__ == "__main__":
    raise SystemExit(main())
