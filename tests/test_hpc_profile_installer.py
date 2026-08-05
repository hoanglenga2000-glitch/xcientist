from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts" / "install_hpc_ssh_credential_from_stdin.ps1"


@pytest.mark.skipif(os.name != "nt", reason="DPAPI CLIXML is Windows-specific")
def test_installer_creates_an_isolated_named_profile(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["APPDATA"] = str(tmp_path)
    # The host process can be PowerShell 7, whose module path is incompatible
    # with powershell.exe 5.1.  Pin the child to the Windows PowerShell modules
    # so this test exercises the installer rather than host-shell contamination.
    system_root = Path(env.get("SystemRoot", r"C:\Windows"))
    env["PSModulePath"] = str(system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(INSTALLER),
            "-User",
            "fixture-user",
            "-Profile",
            "job90353",
            "-HostName",
            "127.0.0.1",
            "-Port",
            "2222",
            "-AllocationBindingId",
            "allocation-fixture-90353",
            "-AllocationGeneration",
            "1",
            "-FromStdin",
        ],
        input="fixture-value\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=True,
        timeout=30,
    )
    result = json.loads(completed.stdout.lstrip("\ufeff"))
    profile = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    metadata = json.loads((profile / "hpc_ssh_metadata.json").read_text(encoding="utf-8-sig"))

    assert result["status"] == "installed"
    assert result["credential_profile"] == "job90353"
    assert result["profile_state"] == "provisioning"
    assert result["allocation_generation"] == 1
    assert (profile / "hpc_ssh_credential.xml").is_file()
    assert metadata["schema"] == "evomind.hpc.dpapi_profile.v2"
    assert metadata["credential_profile"] == "job90353"
    assert metadata["job_id"] == 90353
    assert metadata["profile_state"] == "provisioning"
    assert metadata["allocation_binding_id"] == "allocation-fixture-90353"
    assert metadata["allocation_generation"] == 1
    assert metadata["lifecycle_revision"] == 1
    assert metadata["profile_instance_id"] == result["profile_instance_id"]
    assert metadata["known_hosts_path"] == "known_hosts"
    assert not (tmp_path / "ResearchAgentWorkstation" / "hpc_ssh_credential.xml").exists()


@pytest.mark.skipif(os.name != "nt", reason="DPAPI CLIXML is Windows-specific")
def test_installer_refuses_to_overwrite_an_existing_named_profile(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["APPDATA"] = str(tmp_path)
    system_root = Path(env.get("SystemRoot", r"C:\Windows"))
    env["PSModulePath"] = str(
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    )
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(INSTALLER),
        "-User",
        "fixture-user",
        "-Profile",
        "job90353",
        "-HostName",
        "127.0.0.1",
        "-Port",
        "2222",
        "-AllocationBindingId",
        "allocation-fixture-90353",
        "-AllocationGeneration",
        "1",
        "-FromStdin",
    ]
    first = subprocess.run(
        command,
        input="fixture-value\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=True,
        timeout=30,
    )
    profile = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    metadata_before = (profile / "hpc_ssh_metadata.json").read_bytes()
    credential_before = (profile / "hpc_ssh_credential.xml").read_bytes()

    second = subprocess.run(
        command,
        input="different-fixture-value\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
        timeout=30,
    )

    assert json.loads(first.stdout.lstrip("\ufeff"))["status"] == "installed"
    assert second.returncode != 0
    assert (profile / "hpc_ssh_metadata.json").read_bytes() == metadata_before
    assert (profile / "hpc_ssh_credential.xml").read_bytes() == credential_before
