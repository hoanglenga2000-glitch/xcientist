"""Shared fail-closed CLI contract for current EvoMind HPC operations."""

from __future__ import annotations

import argparse
import os


def env_port(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not 1 <= value <= 65535:
        raise RuntimeError(f"{name} must be between 1 and 65535")
    return value


def add_hpc_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_remote_root: bool = True,
) -> None:
    parser.add_argument("--host", default=os.environ.get("EVOMIND_HPC_HOST", "").strip())
    parser.add_argument("--port", type=int, default=env_port("EVOMIND_HPC_PORT"))
    parser.add_argument("--user", default=os.environ.get("EVOMIND_HPC_USER", "").strip())
    parser.add_argument(
        "--proxy-host", default=os.environ.get("EVOMIND_HPC_SOCKS_HOST", "").strip()
    )
    parser.add_argument("--proxy-port", type=int, default=env_port("EVOMIND_HPC_SOCKS_PORT"))
    parser.add_argument("--password-env", default="EVOMIND_HPC_PASSWORD")
    if include_remote_root:
        parser.add_argument(
            "--remote-root",
            default=os.environ.get("EVOMIND_HPC_REMOTE_WORKSPACE", "").strip(),
        )


def validate_hpc_runtime_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    require_remote_root: bool = True,
) -> None:
    missing: list[str] = []
    if not args.host:
        missing.append("EVOMIND_HPC_HOST/--host")
    if not 1 <= int(args.port or 0) <= 65535:
        missing.append("EVOMIND_HPC_PORT/--port")
    if not args.user:
        missing.append("EVOMIND_HPC_USER/--user")
    if not os.environ.get(args.password_env, ""):
        missing.append(f"password environment variable {args.password_env}")
    if require_remote_root and not getattr(args, "remote_root", ""):
        missing.append("EVOMIND_HPC_REMOTE_WORKSPACE/--remote-root")
    if bool(args.proxy_host) != bool(args.proxy_port):
        missing.append("EVOMIND_HPC_SOCKS_HOST/PORT (configure both or neither)")
    if args.proxy_port and not 1 <= int(args.proxy_port) <= 65535:
        missing.append("EVOMIND_HPC_SOCKS_PORT/--proxy-port")
    if missing:
        parser.error("missing or invalid explicit HPC runtime configuration: " + ", ".join(missing))
