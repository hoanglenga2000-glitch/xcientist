from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.hpc_runtime_contract import (
    add_hpc_runtime_arguments,
    env_port,
    validate_hpc_runtime_arguments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_hpc_runtime_arguments(parser)
    return parser


def test_hpc_runtime_contract_fails_closed_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "EVOMIND_HPC_HOST",
        "EVOMIND_HPC_PORT",
        "EVOMIND_HPC_USER",
        "EVOMIND_HPC_PASSWORD",
        "EVOMIND_HPC_REMOTE_WORKSPACE",
        "EVOMIND_HPC_SOCKS_HOST",
        "EVOMIND_HPC_SOCKS_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    parser = _parser()
    args = parser.parse_args([])

    with pytest.raises(SystemExit) as exc:
        validate_hpc_runtime_arguments(parser, args)

    assert exc.value.code == 2


def test_hpc_runtime_contract_accepts_explicit_direct_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVOMIND_HPC_HOST", "hpc.example.invalid")
    monkeypatch.setenv("EVOMIND_HPC_PORT", "2222")
    monkeypatch.setenv("EVOMIND_HPC_USER", "fixture-user")
    monkeypatch.setenv("EVOMIND_HPC_PASSWORD", "fixture-password")
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", "/srv/evomind")
    monkeypatch.delenv("EVOMIND_HPC_SOCKS_HOST", raising=False)
    monkeypatch.delenv("EVOMIND_HPC_SOCKS_PORT", raising=False)
    parser = _parser()
    args = parser.parse_args([])

    validate_hpc_runtime_arguments(parser, args)

    assert (args.host, args.port, args.user) == ("hpc.example.invalid", 2222, "fixture-user")
    assert args.remote_root == "/srv/evomind"
    assert args.password_env == "EVOMIND_HPC_PASSWORD"


@pytest.mark.parametrize("value", ["0", "65536", "abc"])
def test_hpc_runtime_contract_rejects_invalid_ports(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EVOMIND_HPC_PORT", value)
    with pytest.raises(RuntimeError, match="EVOMIND_HPC_PORT"):
        env_port("EVOMIND_HPC_PORT")


def test_current_hpc_cli_sources_have_no_historical_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "scripts").glob("run_hpc_*.py")),
        root / "scripts" / "submit_hpc_kaggle_submission.py",
        root / "scripts" / "gpu_monitor.py",
    ]
    offenders: dict[str, list[str]] = {}
    forbidden = (
        "100.85.169.63",
        "10.120.18.240",
        "aims" + "lab-",
        'default="GPU_SSH_PASSWORD"',
    )
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        hits = [token for token in forbidden if token in text]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}
