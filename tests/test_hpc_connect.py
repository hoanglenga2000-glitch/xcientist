from __future__ import annotations

from pathlib import Path

import pytest

from scripts import hpc_connect


class FakeSSHClient:
    def __init__(self) -> None:
        self.system_keys_loaded = False
        self.host_key_paths: list[str] = []
        self.policy = None
        self.closed = False

    def load_system_host_keys(self) -> None:
        self.system_keys_loaded = True

    def load_host_keys(self, path: str) -> None:
        self.host_key_paths.append(path)

    def set_missing_host_key_policy(self, policy: object) -> None:
        self.policy = policy

    def close(self) -> None:
        self.closed = True


def test_secure_client_loads_system_and_custom_known_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("fixture host key\n", encoding="utf-8")
    monkeypatch.setenv("GPU_SSH_KNOWN_HOSTS_PATH", str(known_hosts))

    fake_client = FakeSSHClient()
    policy = object()
    monkeypatch.setattr(hpc_connect.paramiko, "SSHClient", lambda: fake_client)
    monkeypatch.setattr(hpc_connect.paramiko, "RejectPolicy", lambda: policy)

    result = hpc_connect.secure_ssh_client()

    assert result is fake_client
    assert fake_client.system_keys_loaded is True
    assert fake_client.host_key_paths == [str(known_hosts)]
    assert fake_client.policy is policy
    assert fake_client.closed is False


def test_missing_custom_known_hosts_closes_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_SSH_KNOWN_HOSTS_PATH", str(tmp_path / "missing_known_hosts"))
    fake_client = FakeSSHClient()
    monkeypatch.setattr(hpc_connect.paramiko, "SSHClient", lambda: fake_client)

    with pytest.raises(RuntimeError, match="path hidden"):
        hpc_connect.secure_ssh_client()

    assert fake_client.closed is True
    assert fake_client.policy is None


def test_hpc_helper_has_no_historical_endpoint_registry() -> None:
    source = Path(hpc_connect.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "HPC_HOST =",
        "PROFILES =",
        "JOBS =",
        "GPU_SSH_PASSWORD",
        "100.85.169.63",
        "aims" + "lab-",
    ):
        assert forbidden not in source


def test_scripts_do_not_trust_or_attach_unverified_transports() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    forbidden = ("AutoAddPolicy", "WarningPolicy", "._transport =", "paramiko." + "Transport(")
    offenders: list[str] = []
    for path in scripts_dir.rglob("*.py"):
        if "_quarantine" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in forbidden):
            offenders.append(str(path.relative_to(scripts_dir)))
    assert offenders == []


def test_scripts_do_not_read_passwords_from_process_argv() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    for path in scripts_dir.rglob("*.py"):
        if "_quarantine" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "password = sys.argv" in text or "secret = sys.argv" in text:
            offenders.append(str(path.relative_to(scripts_dir)))
    assert offenders == []
