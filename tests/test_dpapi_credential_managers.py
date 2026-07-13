from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None,
    reason="PowerShell and Windows DPAPI are required",
)


def _manager_env(state_dir: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["EVOMIND_DPAPI_STATE_DIR"] = str(state_dir)
    env["EVOMIND_ALLOW_TEST_STATE_DIR"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def _manager_command(script_name: str, *arguments: str) -> list[str]:
    return [
        str(POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "scripts" / script_name),
        *arguments,
    ]


def _invoke_manager(
    state_dir: Path,
    script_name: str,
    *arguments: str,
    input_value: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _manager_command(script_name, *arguments),
        cwd=ROOT,
        env=_manager_env(state_dir, extra_env),
        input=(input_value + "\n") if input_value is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )


def _run_manager(
    state_dir: Path,
    script_name: str,
    *arguments: str,
    input_value: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    completed = _invoke_manager(
        state_dir,
        script_name,
        *arguments,
        input_value=input_value,
        extra_env=extra_env,
    )
    if input_value is not None:
        assert input_value not in completed.stdout
        assert input_value not in completed.stderr
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def _run_manager_failure(
    state_dir: Path,
    script_name: str,
    *arguments: str,
    input_value: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    completed = _invoke_manager(
        state_dir,
        script_name,
        *arguments,
        input_value=input_value,
        extra_env=extra_env,
    )
    if input_value is not None:
        assert input_value not in completed.stdout
        assert input_value not in completed.stderr
    assert completed.returncode != 0
    assert not completed.stderr.strip()
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "credential_operation_failed"
    return payload


def _start_manager(
    state_dir: Path,
    script_name: str,
    *arguments: str,
    input_value: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        _manager_command(script_name, *arguments),
        cwd=ROOT,
        env=_manager_env(state_dir, extra_env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if input_value is not None:
        assert process.stdin is not None
        process.stdin.write(input_value + "\n")
        process.stdin.close()
    return process


def _wait_for_path(path: Path, process: subprocess.Popen[str], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        returncode = process.poll()
        if returncode is not None:
            stdout = process.stdout.read() if process.stdout is not None else ""
            stderr = process.stderr.read() if process.stderr is not None else ""
            pytest.fail(f"manager exited before test hook ({returncode}): {stdout}{stderr}")
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for manager test hook: {path}")


def _kill_manager(process: subprocess.Popen[str], secret: str | None = None) -> None:
    try:
        process.kill()
        assert process.wait(timeout=15) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
    stdout = process.stdout.read() if process.stdout is not None else ""
    stderr = process.stderr.read() if process.stderr is not None else ""
    if secret is not None:
        assert secret not in stdout
        assert secret not in stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_generation(state_dir: Path, generation_id: str) -> dict:
    assert re.fullmatch(r"[a-f0-9]{32}", generation_id)
    generation_dir = state_dir / "hpc_ssh_generations" / generation_id
    credential_path = generation_dir / "credential.xml"
    metadata_path = generation_dir / "metadata.json"
    assert generation_dir.is_dir()
    assert credential_path.is_file()
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    credential_sha256 = _sha256(credential_path)
    metadata_sha256 = _sha256(metadata_path)
    assert metadata["schema_version"] == 1
    assert metadata["generation_id"] == generation_id
    assert metadata["credential_sha256"] == credential_sha256
    return {
        "generation_id": generation_id,
        "generation_dir": generation_dir,
        "credential_path": credential_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "credential_sha256": credential_sha256,
        "metadata_sha256": metadata_sha256,
    }


def _read_current_generation(state_dir: Path) -> dict:
    pointer_path = state_dir / "hpc_ssh_current.json"
    assert pointer_path.is_file()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer["schema_version"] == 1
    generation = _read_generation(state_dir, pointer["generation_id"])
    assert pointer["credential_sha256"] == generation["credential_sha256"]
    assert pointer["metadata_sha256"] == generation["metadata_sha256"]
    generation.update({"pointer_path": pointer_path, "pointer": pointer})
    return generation


def _write_legacy_hpc_pair(
    state_dir: Path,
    *,
    username: str,
    secret: str,
    host: str = "legacy.example",
) -> tuple[Path, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    credential_path = state_dir / "hpc_ssh_credential.xml"
    metadata_path = state_dir / "hpc_ssh_metadata.json"
    env = os.environ.copy()
    env["EVOMIND_TEST_CREDENTIAL_PATH"] = str(credential_path)
    env["EVOMIND_TEST_CREDENTIAL_USER"] = username
    script = (
        "$value=[Console]::In.ReadLine();"
        "$secure=ConvertTo-SecureString $value -AsPlainText -Force;"
        "try {[System.Management.Automation.PSCredential]::new("
        "$env:EVOMIND_TEST_CREDENTIAL_USER,$secure)|Export-Clixml -LiteralPath "
        "$env:EVOMIND_TEST_CREDENTIAL_PATH -Depth 3}"
        "finally {$value=$null;$secure.Dispose()}"
    )
    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", script],
        env=env,
        input=secret + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert secret not in completed.stdout + completed.stderr
    metadata_path.write_text(
        json.dumps(
            {
                "host": host,
                "port": 2222,
                "remote_workspace": "/tmp/legacy-evomind",
                "socks_host": "127.0.0.1",
                "socks_port": 7890,
            }
        ),
        encoding="utf-8",
    )
    return credential_path, metadata_path


def _hpc_install_arguments(username: str, host: str) -> list[str]:
    return [
        "install-credential",
        "-User",
        username,
        "-SecretFromStdin",
        "-HostName",
        host,
        "-Port",
        "2222",
        "-RemoteWorkspace",
        "/tmp/evomind",
        "-SocksHost",
        "127.0.0.1",
        "-SocksPort",
        "7890",
    ]


def _acl_summary(path: Path) -> dict:
    script = (
        "$acl=Get-Acl -LiteralPath $env:EVOMIND_TEST_ACL_PATH;"
        "$current=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
        "$system=[System.Security.Principal.SecurityIdentifier]::new("
        "[System.Security.Principal.WellKnownSidType]::LocalSystemSid,$null).Value;"
        "$owner=([System.Security.Principal.NTAccount]::new([string]$acl.Owner)).Translate("
        "[System.Security.Principal.SecurityIdentifier]).Value;"
        "$sids=@($acl.Access|ForEach-Object{$_.IdentityReference.Translate("
        "[System.Security.Principal.SecurityIdentifier]).Value}|Sort-Object -Unique);"
        "@{protected=$acl.AreAccessRulesProtected;current=$current;system=$system;owner=$owner;sids=$sids}"
        "|ConvertTo-Json -Compress"
    )
    env = os.environ.copy()
    env["EVOMIND_TEST_ACL_PATH"] = str(path)
    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", script],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def _assert_restricted_acl(path: Path) -> None:
    acl = _acl_summary(path)
    assert acl["protected"] is True
    assert acl["owner"] in {acl["current"], acl["system"]}
    assert set(acl["sids"]) == {acl["current"], acl["system"]}


def _assert_secret_not_present(path: Path, value: str) -> None:
    payload = path.read_bytes()
    assert value.encode("utf-8") not in payload
    assert value.encode("utf-16-le") not in payload


def _assert_dpapi_credential_matches(path: Path, username: str, secret: str) -> None:
    env = os.environ.copy()
    env["EVOMIND_TEST_CREDENTIAL_PATH"] = str(path)
    env["EVOMIND_TEST_CREDENTIAL_USER"] = username
    script = (
        "$expected=[Console]::In.ReadLine();"
        "$credential=Import-Clixml -LiteralPath $env:EVOMIND_TEST_CREDENTIAL_PATH;"
        "$actual=$credential.GetNetworkCredential().Password;"
        "@{user_match=($credential.UserName -ceq $env:EVOMIND_TEST_CREDENTIAL_USER);"
        "secret_match=($actual -ceq $expected)}|ConvertTo-Json -Compress;"
        "$actual=$null;$expected=$null"
    )
    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", script],
        env=env,
        input=secret + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert secret not in completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {"secret_match": True, "user_match": True}


def test_deepseek_manager_round_trips_dpapi_without_plaintext(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-deepseek"
    unicode_fragment = "".join(chr(value) for value in (0x5BC6, 0x7801, 0x30C6, 0x30B9, 0x30C8, 0x1F510))
    value = "runtime-" + unicode_fragment + "-" + secrets.token_urlsafe(24)

    installed = _run_manager(
        state_dir,
        "manage_deepseek_secret.ps1",
        "install-key",
        "-SecretFromStdin",
        input_value=value,
    )
    assert installed["status"] == "configured"
    assert installed["credential_installed"] is True

    credential_path = state_dir / "deepseek_api_key.xml"
    assert credential_path.is_file()
    _assert_secret_not_present(credential_path, value)
    for protected_path in (state_dir, credential_path):
        acl = _acl_summary(protected_path)
        assert acl["protected"] is True
        assert acl["owner"] in {acl["current"], acl["system"]}
        assert set(acl["sids"]) == {acl["current"], acl["system"]}

    status = _run_manager(state_dir, "manage_deepseek_secret.ps1", "status")
    assert status["status"] == "configured"
    assert value not in json.dumps(status)

    removed = _run_manager(state_dir, "manage_deepseek_secret.ps1", "remove")
    assert removed["credential_installed"] is False
    assert not credential_path.exists()


@pytest.mark.parametrize("credential_kind", ["access_token", "legacy_username_key"])
def test_kaggle_manager_round_trips_both_dpapi_formats(
    tmp_path: Path,
    credential_kind: str,
) -> None:
    state_dir = tmp_path / f"evomind-dpapi-test-{credential_kind}"
    value = "runtime-" + secrets.token_urlsafe(24)
    arguments = ["install-token"]
    if credential_kind == "access_token":
        arguments.append("-SecretFromStdin")
        expected_secret = "KGAT_" + value
    else:
        arguments.extend(["-Username", "ci-user", "-SecretFromStdin"])
        expected_secret = value

    installed = _run_manager(
        state_dir,
        "manage_kaggle_secret.ps1",
        *arguments,
        input_value=expected_secret,
    )
    assert installed["credential_installed"] is True
    assert installed["token_type"] == credential_kind

    credential_path = state_dir / "kaggle_api_token.xml"
    assert credential_path.is_file()
    _assert_secret_not_present(credential_path, expected_secret)

    status = _run_manager(state_dir, "manage_kaggle_secret.ps1", "status")
    assert status["token_type"] == credential_kind
    assert expected_secret not in json.dumps(status)

    removed = _run_manager(state_dir, "manage_kaggle_secret.ps1", "remove")
    assert removed["credential_installed"] is False
    assert not credential_path.exists()


def test_kaggle_real_smoke_tolerates_nonfatal_native_stderr(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-kaggle-native-warning"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "python.cmd").write_text(
        "@echo off\n"
        "echo %* | findstr /C:\"importlib.metadata\" >nul\n"
        "if not errorlevel 1 (echo 2.2.2& exit /b 0)\n"
        ">&2 echo nonfatal kaggle client warning\n"
        "echo 20\n"
        "exit /b 0\n",
        encoding="ascii",
    )
    (shim_dir / "kaggle.cmd").write_text("@echo off\nexit /b 0\n", encoding="ascii")
    path = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
    token = "KGAT_" + secrets.token_urlsafe(24)

    _run_manager(
        state_dir,
        "manage_kaggle_secret.ps1",
        "install-token",
        "-SecretFromStdin",
        input_value=token,
        extra_env={"PATH": path},
    )
    smoke = _run_manager(
        state_dir,
        "manage_kaggle_secret.ps1",
        "smoke",
        "-AllowRealExternal",
        extra_env={"PATH": path},
    )

    assert smoke["status"] == "passed"
    assert smoke["real_external_called"] is True
    assert smoke["competition_count"] == 20


def test_kaggle_local_smoke_stays_configured_unverified(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-kaggle-local-only"
    shim_dir = tmp_path / "shim-local-only"
    shim_dir.mkdir()
    (shim_dir / "python.cmd").write_text("@echo off\necho 2.2.2\nexit /b 0\n", encoding="ascii")
    (shim_dir / "kaggle.cmd").write_text("@echo off\nexit /b 0\n", encoding="ascii")
    path = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
    token = "KGAT_" + secrets.token_urlsafe(24)

    _run_manager(
        state_dir,
        "manage_kaggle_secret.ps1",
        "install-token",
        "-SecretFromStdin",
        input_value=token,
        extra_env={"PATH": path},
    )
    smoke = _run_manager(
        state_dir,
        "manage_kaggle_secret.ps1",
        "smoke",
        extra_env={"PATH": path},
    )

    assert smoke["status"] == "configured_unverified"
    assert smoke["credential_status"] == "configured_unverified"
    assert smoke["verification_state"] == "configured_not_invoked"
    assert smoke["real_external_called"] is False
    assert smoke["human_gate_required_for_submission"] is True


def test_hpc_manager_round_trips_credential_and_metadata(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc"
    value = "runtime-" + secrets.token_urlsafe(24)

    installed = _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        "install-credential",
        "-User",
        "ci-user",
        "-SecretFromStdin",
        "-HostName",
        "127.0.0.1",
        "-Port",
        "2222",
        "-RemoteWorkspace",
        "/tmp/evomind",
        "-SocksHost",
        "127.0.0.1",
        "-SocksPort",
        "7890",
        input_value=value,
    )
    assert installed["credential_installed"] is True
    assert installed["host"] == "127.0.0.1"
    assert installed["port"] == 2222
    current = _read_current_generation(state_dir)
    assert installed["generation_id"] == current["generation_id"]
    assert Path(installed["credential_path"]) == current["credential_path"]
    assert Path(installed["metadata_path"]) == current["metadata_path"]
    assert current["metadata"]["host"] == "127.0.0.1"
    assert current["metadata"]["port"] == 2222
    _assert_secret_not_present(current["credential_path"], value)
    _assert_secret_not_present(current["metadata_path"], value)
    _assert_dpapi_credential_matches(current["credential_path"], "ci-user", value)

    for protected_path in (
        state_dir,
        state_dir / "hpc_ssh_generations",
        current["generation_dir"],
        current["credential_path"],
        current["metadata_path"],
        current["pointer_path"],
    ):
        _assert_restricted_acl(protected_path)

    status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")
    assert status["credential_installed"] is True
    assert status["generation_id"] == current["generation_id"]
    assert status["remote_workspace"] == "/tmp/evomind"
    assert value not in json.dumps(status)

    removed = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "remove")
    assert removed["credential_installed"] is False
    assert removed["status"] == "not_configured"
    assert not (state_dir / "hpc_ssh_current.json").exists()
    assert not (state_dir / "hpc_ssh_generations").exists()
    assert not (state_dir / "hpc_ssh_credential.xml").exists()
    assert not (state_dir / "hpc_ssh_metadata.json").exists()


def test_hpc_install_without_explicit_connection_metadata_fails_closed(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-no-default-endpoint"
    payload = _run_manager_failure(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        "install-credential",
        "-User",
        "ci-user",
        "-SecretFromStdin",
        input_value="fixture-password-value",
    )

    assert payload["status"] == "failed"
    assert payload["credential_installed"] is False
    assert not (state_dir / "hpc_ssh_current.json").exists()
    assert not (state_dir / "hpc_ssh_generations").exists()


def test_hpc_install_allows_explicit_direct_connection_without_socks(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-direct"
    installed = _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        "install-credential",
        "-User",
        "ci-user",
        "-SecretFromStdin",
        "-HostName",
        "direct.example",
        "-Port",
        "22",
        "-RemoteWorkspace",
        "/tmp/evomind-direct",
        input_value="fixture-password-value",
    )

    assert installed["status"] == "configured"
    assert installed["host"] == "direct.example"
    assert installed["socks_host"] == ""
    assert installed["socks_port"] == 0


def test_hpc_proxy_manager_uses_stdin_dpapi_and_restricted_acl(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-proxy"
    value = "runtime-" + secrets.token_urlsafe(24)
    installed = _run_manager(
        state_dir,
        "manage_hpc_proxy_bridge.ps1",
        "install-credential",
        "-ProxyUser",
        "proxy-user",
        "-SecretFromStdin",
        input_value=value,
    )

    credential_path = state_dir / "hpc_socks_credential.xml"
    assert installed["status"] == "installed"
    assert Path(installed["credential_path"]) == credential_path
    _assert_secret_not_present(credential_path, value)
    _assert_dpapi_credential_matches(credential_path, "proxy-user", value)
    _assert_restricted_acl(credential_path)
    status = _run_manager(state_dir, "manage_hpc_proxy_bridge.ps1", "status")
    assert status["credential_installed"] is True


def test_hpc_proxy_manager_source_has_no_plaintext_password_argument() -> None:
    source = (ROOT / "scripts" / "manage_hpc_proxy_bridge.ps1").read_text(encoding="utf-8")
    assert "[string]$ProxyPassword" not in source
    assert "-ProxyPassword" not in source
    assert "ConvertTo-SecureString" not in source
    assert "Read-EvoMindSecureInput" in source


def test_hpc_proxy_start_requires_explicit_upstream_and_stays_loopback(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-proxy-start"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_hpc_proxy_bridge.ps1",
        "install-credential",
        "-ProxyUser",
        "proxy-user",
        "-SecretFromStdin",
        input_value=value,
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        listen_port = probe.getsockname()[1]

    missing = _invoke_manager(
        state_dir,
        "manage_hpc_proxy_bridge.ps1",
        "start",
        "-ListenPort",
        str(listen_port),
    )
    assert missing.returncode != 0
    assert value not in missing.stdout + missing.stderr

    try:
        started = _run_manager(
            state_dir,
            "manage_hpc_proxy_bridge.ps1",
            "start",
            "-ListenPort",
            str(listen_port),
            "-UpstreamHost",
            "127.0.0.1",
            "-UpstreamPort",
            "9",
        )
        assert started["status"] == "started"
        assert started["listen_port"] == listen_port
        status = _run_manager(
            state_dir,
            "manage_hpc_proxy_bridge.ps1",
            "status",
            "-ListenPort",
            str(listen_port),
        )
        assert status["status"] == "running"
    finally:
        _invoke_manager(
            state_dir,
            "manage_hpc_proxy_bridge.ps1",
            "stop",
            "-ListenPort",
            str(listen_port),
        )


def test_manager_rejects_directory_collision_without_nested_secret(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-collision"
    collision = state_dir / "deepseek_api_key.xml"
    collision.mkdir(parents=True)
    value = "runtime-" + secrets.token_urlsafe(24)

    payload = _run_manager_failure(
        state_dir,
        "manage_deepseek_secret.ps1",
        "install-key",
        "-SecretFromStdin",
        input_value=value,
    )

    assert payload["credential_installed"] is False
    assert collision.is_dir()
    assert list(collision.iterdir()) == []


def test_status_repairs_untrusted_explicit_file_acl(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-acl-repair"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_deepseek_secret.ps1",
        "install-key",
        "-SecretFromStdin",
        input_value=value,
    )
    credential_path = state_dir / "deepseek_api_key.xml"
    grant = subprocess.run(
        ["icacls.exe", str(credential_path), "/grant", "*S-1-1-0:(F)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert grant.returncode == 0, grant.stdout + grant.stderr
    assert "S-1-1-0" in set(_acl_summary(credential_path)["sids"])

    status = _run_manager(state_dir, "manage_deepseek_secret.ps1", "status")

    assert status["status"] == "configured"
    acl = _acl_summary(credential_path)
    assert set(acl["sids"]) == {acl["current"], acl["system"]}


def test_manager_repairs_administrators_owned_state_directory(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-administrator-owner"
    state_dir.mkdir()
    env = os.environ.copy()
    env["EVOMIND_TEST_ACL_PATH"] = str(state_dir)
    script = (
        "$identity=[System.Security.Principal.WindowsIdentity]::GetCurrent();"
        "$principal=[System.Security.Principal.WindowsPrincipal]::new($identity);"
        "$admin=[System.Security.Principal.SecurityIdentifier]::new("
        "[System.Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid,$null);"
        "$member=$principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)-or"
        "(@($identity.Groups|ForEach-Object{$_.Value})-contains $admin.Value);"
        "if(-not $member){exit 77};"
        "$acl=Get-Acl -LiteralPath $env:EVOMIND_TEST_ACL_PATH;"
        "$acl.SetOwner($admin);"
        "Set-Acl -LiteralPath $env:EVOMIND_TEST_ACL_PATH -AclObject $acl"
    )
    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", script],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("administrator owner repair requires an administrator token")
    assert completed.returncode == 0, completed.stdout + completed.stderr

    owner_before = _acl_summary(state_dir)
    assert owner_before["owner"] not in {owner_before["current"], owner_before["system"]}
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_deepseek_secret.ps1",
        "install-key",
        "-SecretFromStdin",
        input_value=value,
    )

    _assert_restricted_acl(state_dir)
    _assert_restricted_acl(state_dir / "deepseek_api_key.xml")


def test_hpc_invalid_metadata_does_not_replace_existing_credential(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-rollback"
    first_value = "runtime-" + secrets.token_urlsafe(24)
    second_value = "runtime-" + secrets.token_urlsafe(24)
    common_arguments = [
        "install-credential",
        "-User",
        "ci-user",
        "-SecretFromStdin",
        "-HostName",
        "127.0.0.1",
        "-Port",
        "2222",
        "-SocksHost",
        "127.0.0.1",
        "-SocksPort",
        "7890",
    ]
    _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *common_arguments,
        "-RemoteWorkspace",
        "/tmp/evomind",
        input_value=first_value,
    )
    before = _read_current_generation(state_dir)
    before_pointer = before["pointer_path"].read_bytes()
    before_credential = before["credential_path"].read_bytes()
    before_metadata = before["metadata_path"].read_bytes()
    before_generation_ids = {path.name for path in (state_dir / "hpc_ssh_generations").iterdir()}

    _run_manager_failure(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *common_arguments,
        "-RemoteWorkspace",
        "relative/path",
        input_value=second_value,
    )

    after = _read_current_generation(state_dir)
    assert after["generation_id"] == before["generation_id"]
    assert after["pointer_path"].read_bytes() == before_pointer
    assert after["credential_path"].read_bytes() == before_credential
    assert after["metadata_path"].read_bytes() == before_metadata
    assert {path.name for path in (state_dir / "hpc_ssh_generations").iterdir()} == before_generation_ids
    _assert_secret_not_present(after["credential_path"], second_value)


def test_hpc_set_metadata_publishes_new_complete_generation(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-set-metadata"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("metadata-user", "before.example"),
        input_value=value,
    )
    before = _read_current_generation(state_dir)

    updated = _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        "set-metadata",
        "-HostName",
        "after.example",
        "-Port",
        "2200",
        "-RemoteWorkspace",
        "/tmp/updated-evomind",
        "-SocksHost",
        "proxy.example",
        "-SocksPort",
        "8899",
    )

    after = _read_current_generation(state_dir)
    assert after["generation_id"] != before["generation_id"]
    assert updated["generation_id"] == after["generation_id"]
    assert updated["user"] == "metadata-user"
    assert updated["host"] == "after.example"
    assert after["metadata"]["port"] == 2200
    assert after["metadata"]["remote_workspace"] == "/tmp/updated-evomind"
    assert after["metadata"]["socks_host"] == "proxy.example"
    assert after["metadata"]["socks_port"] == 8899
    _assert_dpapi_credential_matches(after["credential_path"], "metadata-user", value)
    _assert_secret_not_present(after["credential_path"], value)


def test_hpc_legacy_pair_migrates_to_current_generation(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-legacy"
    value = "runtime-" + secrets.token_urlsafe(24)
    legacy_credential, legacy_metadata = _write_legacy_hpc_pair(
        state_dir,
        username="legacy-user",
        secret=value,
    )
    assert not (state_dir / "hpc_ssh_current.json").exists()

    status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")

    current = _read_current_generation(state_dir)
    assert status["status"] == "configured"
    assert status["generation_id"] == current["generation_id"]
    assert status["user"] == "legacy-user"
    assert status["host"] == "legacy.example"
    assert status["remote_workspace"] == "/tmp/legacy-evomind"
    _assert_dpapi_credential_matches(current["credential_path"], "legacy-user", value)
    _assert_secret_not_present(current["credential_path"], value)
    assert not legacy_credential.exists()
    assert not legacy_metadata.exists()


@pytest.mark.parametrize("missing", ["credential", "metadata"])
def test_hpc_incomplete_legacy_pair_fails_closed(tmp_path: Path, missing: str) -> None:
    state_dir = tmp_path / f"evomind-dpapi-test-hpc-legacy-missing-{missing}"
    value = "runtime-" + secrets.token_urlsafe(24)
    credential_path, metadata_path = _write_legacy_hpc_pair(
        state_dir,
        username="legacy-user",
        secret=value,
    )
    {"credential": credential_path, "metadata": metadata_path}[missing].unlink()

    payload = _run_manager_failure(state_dir, "manage_hpc_ssh_secret.ps1", "status")

    assert payload["credential_installed"] is False
    assert "host" not in payload
    assert not (state_dir / "hpc_ssh_current.json").exists()
    assert not (state_dir / "hpc_ssh_generations").exists()


def test_workstation_launcher_rejects_legacy_hpc_credential_without_metadata(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-launcher-incomplete"
    value = "runtime-" + secrets.token_urlsafe(24)
    _, metadata_path = _write_legacy_hpc_pair(
        state_dir,
        username="legacy-user",
        secret=value,
    )
    metadata_path.unlink()

    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "start_verified_workstation.ps1"),
            "status",
        ],
        cwd=ROOT,
        env=_manager_env(state_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert value not in output
    assert '"hpc_ssh": true' not in output.casefold()
    assert '"status": "ok"' not in output.casefold()
    assert not (state_dir / "hpc_ssh_current.json").exists()


def test_hpc_orphan_generation_is_not_selected_without_current_pointer(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-orphan"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("orphan-user", "orphan.example"),
        input_value=value,
    )
    current = _read_current_generation(state_dir)
    current["pointer_path"].unlink()

    status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")

    assert status["status"] == "not_configured"
    assert status["credential_installed"] is False
    assert status["generation_id"] is None
    assert status["host"] is None
    assert _read_generation(state_dir, current["generation_id"])["metadata"]["host"] == "orphan.example"


@pytest.mark.parametrize(
    "mutation",
    ["missing_credential", "missing_metadata", "credential_hash", "metadata_hash", "pointer_generation"],
)
def test_hpc_incomplete_or_tampered_current_generation_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    state_dir = tmp_path / f"evomind-dpapi-test-hpc-tamper-{mutation}"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("tamper-user", "tamper.example"),
        input_value=value,
    )
    current = _read_current_generation(state_dir)

    if mutation == "missing_credential":
        current["credential_path"].unlink()
    elif mutation == "missing_metadata":
        current["metadata_path"].unlink()
    elif mutation == "credential_hash":
        current["credential_path"].write_bytes(current["credential_path"].read_bytes() + b"\n")
    elif mutation == "metadata_hash":
        current["metadata_path"].write_bytes(current["metadata_path"].read_bytes() + b"\n")
    else:
        pointer = current["pointer"]
        pointer["generation_id"] = "../legacy"
        current["pointer_path"].write_text(json.dumps(pointer), encoding="utf-8")

    payload = _run_manager_failure(state_dir, "manage_hpc_ssh_secret.ps1", "status")

    assert payload["credential_installed"] is False
    assert "host" not in payload
    assert value not in json.dumps(payload)


def test_hpc_forced_termination_before_first_pointer_publish_leaves_unconfigured_state(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-kill-first"
    value = "runtime-" + secrets.token_urlsafe(24)
    marker = state_dir / ".hpc-before-pointer-publish.ready"
    process = _start_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("first-kill-user", "first-kill.example"),
        input_value=value,
        extra_env={"EVOMIND_HPC_TEST_PAUSE_BEFORE_POINTER_MS": "120000"},
    )
    try:
        _wait_for_path(marker, process)
    finally:
        if process.poll() is None:
            _kill_manager(process, value)

    assert not (state_dir / "hpc_ssh_current.json").exists()
    generation_ids = [path.name for path in (state_dir / "hpc_ssh_generations").iterdir() if path.is_dir()]
    assert len(generation_ids) == 1
    orphan = _read_generation(state_dir, generation_ids[0])
    assert orphan["metadata"]["host"] == "first-kill.example"
    status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")
    assert status["status"] == "not_configured"
    assert status["credential_installed"] is False
    assert status["host"] is None


def test_hpc_forced_termination_before_pointer_replacement_preserves_previous_generation(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-kill-update"
    first_value = "runtime-" + secrets.token_urlsafe(24)
    second_value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("stable-user", "stable.example"),
        input_value=first_value,
    )
    before = _read_current_generation(state_dir)
    before_pointer = before["pointer_path"].read_bytes()
    marker = state_dir / ".hpc-before-pointer-publish.ready"
    process = _start_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("interrupted-user", "interrupted.example"),
        input_value=second_value,
        extra_env={"EVOMIND_HPC_TEST_PAUSE_BEFORE_POINTER_MS": "120000"},
    )
    try:
        _wait_for_path(marker, process)
    finally:
        if process.poll() is None:
            _kill_manager(process, second_value)

    assert before["pointer_path"].read_bytes() == before_pointer
    after = _read_current_generation(state_dir)
    assert after["generation_id"] == before["generation_id"]
    status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")
    assert (status["user"], status["host"]) == ("stable-user", "stable.example")
    _assert_dpapi_credential_matches(after["credential_path"], "stable-user", first_value)

    recovered_value = "runtime-" + secrets.token_urlsafe(24)
    recovered = _run_manager(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        *_hpc_install_arguments("recovered-user", "recovered.example"),
        input_value=recovered_value,
    )
    assert (recovered["user"], recovered["host"]) == ("recovered-user", "recovered.example")
    assert recovered["generation_id"] != before["generation_id"]


def test_hpc_manager_rejects_shell_metacharacters_in_socks_host(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-hpc-host"
    value = "runtime-" + secrets.token_urlsafe(24)

    _run_manager_failure(
        state_dir,
        "manage_hpc_ssh_secret.ps1",
        "install-credential",
        "-User",
        "ci-user",
        "-SecretFromStdin",
        "-SocksHost",
        "127.0.0.1&whoami",
        input_value=value,
    )

    assert not (state_dir / "hpc_ssh_current.json").exists()
    assert not (state_dir / "hpc_ssh_generations").exists()


def test_workstation_launcher_reports_python_exported_to_web_runtime(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-workstation-python"
    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "start_verified_workstation.ps1"),
            "status",
        ],
        cwd=ROOT,
        env=_manager_env(state_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    selected = Path(payload["workstation_python"])
    assert selected.is_file()


def test_workstation_launcher_isolates_invalid_optional_llm_credential(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-invalid-optional-llm"
    state_dir.mkdir()
    credential_path = state_dir / "deepseek_api_key.xml"
    env = _manager_env(state_dir)
    env["EVOMIND_TEST_CREDENTIAL_PATH"] = str(credential_path)
    script = (
        "$secure=ConvertTo-SecureString 'placeholder-secret' -AsPlainText -Force;"
        "try {[System.Management.Automation.PSCredential]::new('wrong-marker',$secure)"
        "|Export-Clixml -LiteralPath $env:EVOMIND_TEST_CREDENTIAL_PATH -Depth 3}"
        "finally {$secure.Dispose()}"
    )
    created = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", script],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stdout + created.stderr

    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "start_verified_workstation.ps1"),
            "status",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dpapi_loaded"]["deepseek"] is False
    assert payload["dpapi_loaded"]["credential_errors"]["deepseek"] == "invalid_or_unreadable"


def test_workstation_launcher_writes_audit_only_after_full_acceptance() -> None:
    source = (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8")
    smoke_marker = "$smokeResults = Invoke-SmokeSuite"
    final_audit_marker = "$auditPaths = Write-VerifiedAuditReport"
    full_acceptance_marker = "$smokeResults += Invoke-JsonCommand -Label \"full_acceptance\""
    self_check_marker = "Invoke-JsonCommand -Label \"verified_launch_audit\""

    smoke_index = source.index(smoke_marker)
    full_acceptance_index = source.index(full_acceptance_marker, smoke_index)
    final_audit_index = source.index(final_audit_marker, full_acceptance_index)
    self_check_index = source.index(self_check_marker, final_audit_index)

    assert smoke_index < full_acceptance_index < final_audit_index < self_check_index
    assert source.count(final_audit_marker) == 1
    assert '"--skip-verified-launch-audit"' in source
    assert "[switch]$RunFullAcceptance" in source
    assert "if ($ShouldRunFullAcceptance)" in source


def test_locked_credential_removal_fails_closed(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-locked-remove"
    value = "runtime-" + secrets.token_urlsafe(24)
    _run_manager(
        state_dir,
        "manage_deepseek_secret.ps1",
        "install-key",
        "-SecretFromStdin",
        input_value=value,
    )
    credential_path = state_dir / "deepseek_api_key.xml"
    env = os.environ.copy()
    env.update(
        {
            "EVOMIND_DPAPI_STATE_DIR": str(state_dir),
            "EVOMIND_ALLOW_TEST_STATE_DIR": "1",
            "EVOMIND_TEST_CREDENTIAL_PATH": str(credential_path),
            "EVOMIND_TEST_MANAGER_PATH": str(ROOT / "scripts" / "manage_deepseek_secret.ps1"),
        }
    )
    lock_and_remove = (
        "$stream=[System.IO.File]::Open($env:EVOMIND_TEST_CREDENTIAL_PATH,"
        "[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::Read);"
        "try { & powershell -NoProfile -ExecutionPolicy Bypass -File "
        "$env:EVOMIND_TEST_MANAGER_PATH remove; exit $LASTEXITCODE } "
        "finally { $stream.Dispose() }"
    )

    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", lock_and_remove],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert credential_path.is_file()


def test_state_override_without_test_opt_in_returns_clean_json_error(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["EVOMIND_DPAPI_STATE_DIR"] = str(tmp_path / "evomind-dpapi-test-no-opt-in")
    env.pop("EVOMIND_ALLOW_TEST_STATE_DIR", None)

    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "manage_deepseek_secret.ps1"),
            "status",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert not completed.stderr.strip()
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "credential_operation_failed"


def test_hpc_concurrent_installs_preserve_credential_metadata_pairing(tmp_path: Path) -> None:
    for round_index in range(2):
        state_dir = tmp_path / f"evomind-dpapi-test-concurrent-{round_index}"
        specs = [("usera", "a.example"), ("userb", "b.example"), ("userc", "c.example")]
        processes: list[tuple[subprocess.Popen[str], str, str, str]] = []
        for username, host in specs:
            value = "runtime-" + secrets.token_urlsafe(24)
            process = _start_manager(
                state_dir,
                "manage_hpc_ssh_secret.ps1",
                *_hpc_install_arguments(username, host),
                input_value=value,
            )
            processes.append((process, username, host, value))
        for process, username, host, value in processes:
            assert process.wait(timeout=60) == 0
            assert process.stdout is not None
            assert process.stderr is not None
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            assert value not in stdout + stderr
            payload = json.loads(stdout)
            assert not stderr.strip()
            assert (payload["user"], payload["host"]) == (username, host)
            assert re.fullmatch(r"[a-f0-9]{32}", payload["generation_id"])

        status = _run_manager(state_dir, "manage_hpc_ssh_secret.ps1", "status")
        assert (status["user"], status["host"]) in set(specs)
        current = _read_current_generation(state_dir)
        assert status["generation_id"] == current["generation_id"]
        assert current["metadata"]["host"] == status["host"]
        expected_secret = {username: value for _, username, _, value in processes}[status["user"]]
        _assert_dpapi_credential_matches(current["credential_path"], status["user"], expected_secret)


def test_secure_parameter_output_is_capturable_by_installer(tmp_path: Path) -> None:
    state_dir = tmp_path / "evomind-dpapi-test-installer-capture"
    value = "runtime-" + secrets.token_urlsafe(24)
    env = os.environ.copy()
    env.update(
        {
            "EVOMIND_DPAPI_STATE_DIR": str(state_dir),
            "EVOMIND_ALLOW_TEST_STATE_DIR": "1",
            "EVOMIND_TEST_MANAGER_PATH": str(ROOT / "scripts" / "manage_deepseek_secret.ps1"),
            "EVOMIND_TEST_SECRET_VALUE": value,
        }
    )
    wrapper = (
        "$secure=ConvertTo-SecureString $env:EVOMIND_TEST_SECRET_VALUE -AsPlainText -Force;"
        "try { $captured=& $env:EVOMIND_TEST_MANAGER_PATH install-key -SecureApiKey $secure;"
        "$code=$LASTEXITCODE;$payload=$captured|ConvertFrom-Json;"
        "@{manager_exit=$code;manager_status=$payload.status;captured=($null -ne $captured)}"
        "|ConvertTo-Json -Compress } finally { $secure.Dispose() }"
    )

    completed = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", wrapper],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert value not in completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"captured": True, "manager_exit": 0, "manager_status": "configured"}
