from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_pid_detection_and_port_specific_runtime_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("launchfix_dashboard_manager", "scripts/manage_workstation_dashboard.py")
    assert manager.pid_running(os.getpid()) is True
    assert manager.pid_running(999_999_999) is False
    default_pid, _, default_out, _ = manager.runtime_paths(8088)
    custom_pid, _, custom_out, _ = manager.runtime_paths(18192)
    assert default_pid.name == "dashboard.pid"
    assert default_out.name == "dashboard.out.log"
    assert custom_pid.name == "dashboard.18192.pid"
    assert custom_out.name == "dashboard.18192.out.log"

    monkeypatch.delenv("WORKSTATION_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(manager, "bundle_mode", lambda: True)
    monkeypatch.setattr(manager, "data_root", lambda: tmp_path)
    bundled_pid, bundled_state, _, _ = manager.runtime_paths(8088)
    assert bundled_pid == tmp_path / "logs/dashboard.pid"
    assert bundled_state == tmp_path / "logs/dashboard.process.json"


def test_bundle_entrypoints_honor_the_managed_environment_port() -> None:
    for relative in ("install.ps1", "start.ps1", "stop.ps1", "status.ps1"):
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert '$PSBoundParameters.ContainsKey("Port")' in source
        assert "[int]::TryParse($env:WORKSTATION_PORT, [ref]$resolvedPort)" in source
        assert "$Port = $resolvedPort" in source
        assert '$env:PYTHONDONTWRITEBYTECODE = "1"' in source


def test_runtime_environment_isolates_browser_secrets_and_binds_source_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = load("release_runtime_environment", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "bundle_mode", lambda: False)
    source = {
        "WORKSTATION_SESSION_SECRET": "test-dashboard-only",
        "WORKSTATION_BOOTSTRAP_TOKEN_HASH": "test-bootstrap-only",
        "WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH": "test-automation-only",
        "HTTP_COOKIE": "cookie",
        "COOKIE": "cookie2",
        "EVOMIND_SESSION_COOKIE": "cookie3",
        "EVOMIND_CSRF_TOKEN": "csrf",
        "PYTHONPATH": "untrusted",
        "WORKSTATION_DATA_DIR": "data",
    }
    isolated = manager.runtime_env(source, "n" * 43, 18765)
    for name in (
        "WORKSTATION_SESSION_SECRET", "WORKSTATION_BOOTSTRAP_TOKEN_HASH",
        "WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH", "HTTP_COOKIE",
        "COOKIE", "EVOMIND_SESSION_COOKIE", "EVOMIND_CSRF_TOKEN",
    ):
        assert name not in isolated
    assert isolated["PYTHONPATH"] == str((manager.ROOT / "src").resolve())
    assert isolated["WORKSTATION_RELEASE_NONCE"] == "n" * 43
    assert isolated["EVOMIND_RUNTIME_PORT"] == "18765"
    assert source["WORKSTATION_SESSION_SECRET"] == "test-dashboard-only"


def test_dashboard_runtime_health_uses_file_token_without_exposing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("launchfix_dashboard_runtime_health", "scripts/manage_workstation_dashboard.py")
    token = "runtime-secret-token-for-test"
    token_path = tmp_path / "workspace" / "runtime" / "runtime.token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(token, encoding="ascii")
    monkeypatch.setattr(manager, "data_root", lambda: tmp_path)
    monkeypatch.setenv("EVOMIND_RUNTIME_PORT", "18765")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"status":"ready","version":"0.3.0"}'

    def fake_urlopen(request, *, timeout):
        assert request.full_url == "http://127.0.0.1:18765/v1/health"
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert timeout == 1.5
        return Response()

    monkeypatch.setattr(manager.urllib.request, "urlopen", fake_urlopen)
    health = manager.fetch_runtime_status(timeout=1.5)

    assert health == {
        "reachable": True,
        "http_status": 200,
        "status": "ready",
        "version": "0.3.0",
        "port": 18765,
    }
    assert token not in json.dumps(health)


def test_runtime_launch_uses_data_root_without_duplicate_workspace_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("release_runtime_workspace_root", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "data_root", lambda: tmp_path)
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "bundle_mode", lambda: False)
    monkeypatch.setenv("WORKSTATION_PYTHON", sys.executable)

    command, cwd = manager.runtime_launch_command("n" * 43, 18765)

    assert Path(command[3]) == tmp_path.resolve()
    assert Path(command[3]) / "workspace" / "runtime" == tmp_path / "workspace" / "runtime"
    assert "workspace/workspace" not in Path(command[3]).as_posix()
    assert cwd == tmp_path.resolve()


def test_runtime_launch_uses_console_python_in_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "nt":
        pytest.skip("CI console Python selection is Windows-specific")
    manager = load("release_runtime_ci_python", "scripts/manage_workstation_dashboard.py")
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    monkeypatch.setattr(manager, "data_root", lambda: tmp_path)
    monkeypatch.setenv("WORKSTATION_PYTHON", str(python))
    monkeypatch.setenv("CI", "true")

    command, _cwd = manager.runtime_launch_command("n" * 43, 18765)

    assert Path(command[0]) == python.resolve()


def test_runtime_process_falls_back_to_console_python_when_pythonw_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "nt":
        pytest.skip("pythonw fallback is Windows-specific")
    manager = load("release_runtime_pythonw_fallback", "scripts/manage_workstation_dashboard.py")
    pythonw = tmp_path / "pythonw.exe"
    python = tmp_path / "python.exe"
    pythonw.write_bytes(b"")
    python.write_bytes(b"")
    calls: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            raise PermissionError(5, "access denied", str(pythonw))
        return object()

    monkeypatch.setattr(manager.subprocess, "Popen", fake_popen)
    result = manager.launch_process_with_windows_fallback(
        [str(pythonw), "-c", "pass"],
        cwd=tmp_path,
        env={},
        stdout=None,
        stderr=None,
        creationflags=0,
    )

    assert result is not None
    assert calls == [[str(pythonw), "-c", "pass"], [str(python), "-c", "pass"]]


def test_managed_process_retries_without_breakaway_when_job_forbids_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "nt":
        pytest.skip("Windows job-object fallback is Windows-specific")
    manager = load("release_runtime_job_fallback", "scripts/manage_workstation_dashboard.py")
    breakaway = getattr(manager.subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    if not breakaway:
        pytest.skip("CREATE_BREAKAWAY_FROM_JOB is unavailable")
    calls: list[tuple[list[str], int]] = []

    def fake_popen(command, **kwargs):
        calls.append((list(command), kwargs["creationflags"]))
        if len(calls) == 1:
            raise PermissionError(5, "access denied", command[0])
        return object()

    monkeypatch.setattr(manager.subprocess, "Popen", fake_popen)
    command = [str(tmp_path / "runtime.exe"), "--serve"]
    creationflags = manager.subprocess.CREATE_NO_WINDOW | breakaway
    result = manager.launch_process_with_windows_fallback(
        command,
        cwd=tmp_path,
        env={},
        stdout=None,
        stderr=None,
        creationflags=creationflags,
    )

    assert result is not None
    assert calls == [
        (command, creationflags),
        (command, creationflags & ~breakaway),
    ]


def test_dashboard_bootstrap_fragment_is_written_to_private_one_time_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("launchfix_dashboard_bootstrap", "scripts/manage_workstation_dashboard.py")
    token = "one-time-bootstrap-token-for-test"
    monkeypatch.setattr(manager, "runtime_dir", lambda: tmp_path)

    path = manager.write_bootstrap_url("127.0.0.1", 18088, token)
    safe_result = {
        "url": manager.control_url_for("127.0.0.1", 18088),
        "bootstrap_url_file": str(path),
        "bootstrap_token_exposed": False,
    }

    assert path == tmp_path / "dashboard.18088.bootstrap.once"
    assert not path.is_symlink()
    assert path.read_text(encoding="utf-8") == (
        f"http://127.0.0.1:18088/?page=assistant#bootstrap={token}"
    )
    assert token not in json.dumps(safe_result)
    assert "#bootstrap=" not in safe_result["url"]


def test_dashboard_bootstrap_and_automation_tokens_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = load("launchfix_dashboard_independent_auth", "scripts/manage_workstation_dashboard.py")
    issued = iter(("bootstrap-token-value-aaaaaaaa", "automation-token-value-bbbbbbbb"))
    monkeypatch.setattr(manager.secrets, "token_urlsafe", lambda _size: next(issued))
    env: dict[str, str] = {}

    bootstrap_token, automation_token = manager.bind_local_auth_tokens(env)

    assert bootstrap_token != automation_token
    assert env["WORKSTATION_BOOTSTRAP_TOKEN_HASH"] != env["WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH"]
    assert bootstrap_token not in json.dumps(env)
    assert automation_token not in json.dumps(env)


def test_automation_token_survives_bootstrap_claim_and_rotates_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = load("launchfix_dashboard_automation_lifecycle", "scripts/manage_workstation_dashboard.py")
    auth = load("launchfix_local_automation_reader", "scripts/workstation_local_auth.py")
    monkeypatch.setattr(manager, "runtime_dir", lambda: tmp_path)
    monkeypatch.setenv("WORKSTATION_RUNTIME_DIR", str(tmp_path))
    first = "automation-token-first-aaaaaaaa"
    second = "automation-token-second-bbbbbbb"

    bootstrap = manager.write_bootstrap_url("127.0.0.1", 18088, "bootstrap-token-cccccccccccc")
    token_path = manager.write_automation_token(18088, first)
    bootstrap.unlink()

    assert auth.automation_token("http://127.0.0.1:18088") == first
    manager.write_automation_token(18088, second)
    assert token_path.read_text(encoding="ascii") == second
    assert auth.authenticated_headers("http://127.0.0.1:18088")[auth.AUTOMATION_HEADER] == second
    assert capsys.readouterr().out == ""


def test_verified_stop_cleanup_removes_all_local_auth_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("launchfix_dashboard_auth_cleanup", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "runtime_dir", lambda: tmp_path)
    bootstrap = manager.write_bootstrap_url("127.0.0.1", 18088, "bootstrap-token-dddddddddddd")
    automation = manager.write_automation_token(18088, "automation-token-eeeeeeeeeeee")

    manager.remove_local_auth_files(18088)

    assert not bootstrap.exists()
    assert not automation.exists()


def test_gateway_rejects_non_loopback_or_wrong_port() -> None:
    gateway = load("launchfix_gateway", "scripts/manage_local_gateway.py")
    assert gateway.validate_base_url("http://127.0.0.1:65068/v1") == "http://127.0.0.1:65068/v1"
    with pytest.raises(SystemExit):
        gateway.validate_base_url("https://example.invalid/v1")
    with pytest.raises(SystemExit):
        gateway.validate_base_url("http://127.0.0.1:8080/v1")


def test_gateway_publishes_degraded_ui_connector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = load("launchfix_gateway_publish", "scripts/manage_local_gateway.py")
    monkeypatch.setenv("WORKSTATION_ROOT", str(tmp_path))
    report = gateway.publish_status(
        {"base_url": gateway.DEFAULT_BASE_URL, "tcp_reachable": False, "http_status": None, "models_endpoint_ok": False, "credential_present": False},
        "status_only",
        None,
    )
    assert report["status"] == "degraded_unavailable"
    assert report["local_fallback_available"] is True
    summary = json.loads((tmp_path / "workspace" / "workstation_summary.json").read_text(encoding="utf-8"))
    connector = summary["connector_status"]["local_gateway"]
    assert connector["state"] == "Local Deterministic Fallback"
    assert connector["configured"] is False


def test_release_paths_reject_traversal_and_user_data() -> None:
    lifecycle = load("launchfix_lifecycle", "scripts/workstation_lifecycle.py")
    assert lifecycle.safe_relative("app/server.js").as_posix() == "app/server.js"
    with pytest.raises(ValueError):
        lifecycle.safe_relative("../escape.txt")
    with pytest.raises(ValueError):
        lifecycle.safe_relative("user-data/workspace.db")


def test_release_database_migrator_rejects_destructive_sql() -> None:
    migrator = load("release_db_migrate_additive", "scripts/release_db_migrate.py")
    migrator.validate_additive_migration("add_column", 'ALTER TABLE "tasks" ADD COLUMN "note" TEXT;')
    with pytest.raises(RuntimeError, match="not additive: DROP"):
        migrator.validate_additive_migration("drop_tasks", 'DROP TABLE "tasks";')


def test_release_lifecycle_registration_uses_central_data_layout(tmp_path: Path) -> None:
    lifecycle = load("release_lifecycle_central_data", "scripts/workstation_lifecycle.py")
    package = tmp_path / "package"
    package.mkdir()
    payload = package / "start.ps1"
    payload.write_text("Write-Output ok\n", encoding="utf-8")
    manifest = {
        "version": "0.3.0",
        "release_id": "fixture-release",
        "files": [
            {
                "path": "start.ps1",
                "sha256": lifecycle.file_sha256(payload),
            }
        ],
    }
    (package / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    install_root = tmp_path / "versions" / "0.3.0"
    data_root = tmp_path / "data"
    backups_root = tmp_path / "backups"
    marker = lifecycle.initialize(install_root, package, data_root, backups_root)
    assert marker["data_root"] == str(data_root.resolve())
    assert marker["backups_root"] == str(backups_root.resolve())
    assert data_root.is_dir()
    assert backups_root.is_dir()
    assert not (install_root / "user-data").exists()


@pytest.mark.parametrize(
    "value",
    ["C:drive-relative.txt", "C:/absolute.txt", "app/file.txt:ads", "app/CON", "app/name. ", "../escape"],
)
def test_release_paths_reject_windows_aliases(value: str) -> None:
    lifecycle = load(f"release_path_{abs(hash(value))}", "scripts/workstation_lifecycle.py")
    with pytest.raises(ValueError):
        lifecycle.safe_relative(value)


def test_release_manifest_rejects_casefold_collision(tmp_path: Path) -> None:
    lifecycle = load("release_casefold_collision", "scripts/workstation_lifecycle.py")
    package = tmp_path / "package"
    (package / "app").mkdir(parents=True)
    payload = package / "app" / "A.txt"
    payload.write_text("x", encoding="utf-8")
    digest = lifecycle.file_sha256(payload)
    (package / "release-manifest.json").write_text(json.dumps({
        "version": "0.3.0",
        "files": [
            {"path": "app/A.txt", "sha256": digest},
            {"path": "app/a.txt", "sha256": digest},
        ],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate release path"):
        lifecycle.load_release(package)


@pytest.mark.parametrize(
    ("name", "sql", "label"),
    [
        ("alter_drop", 'ALTER TABLE "tasks" DROP COLUMN "name";', "ALTER_DROP"),
        ("trigger_dml", 'CREATE TRIGGER t AFTER INSERT ON tasks BEGIN DELETE FROM tasks; END;', "TRIGGER"),
        ("cte_delete", 'WITH doomed AS (SELECT id FROM tasks) DELETE FROM tasks WHERE id IN doomed;', "CTE_DML"),
        ("attach", "ATTACH DATABASE 'other.db' AS other;", "ATTACH"),
        ("pragma", "PRAGMA writable_schema=ON;", "PRAGMA"),
        ("transaction", "BEGIN IMMEDIATE; CREATE TABLE x(id INTEGER); COMMIT;", "TRANSACTION_CONTROL"),
    ],
)
def test_release_database_migrator_strict_statement_allowlist(name: str, sql: str, label: str) -> None:
    migrator = load(f"release_migrator_{name}", "scripts/release_db_migrate.py")
    with pytest.raises(RuntimeError, match=label):
        migrator.validate_additive_migration(name, sql)


def test_release_database_migrator_authorizer_and_schema_fingerprint(tmp_path: Path) -> None:
    migrator = load("release_migrator_fingerprint", "scripts/release_db_migrate.py")
    migrations = tmp_path / "migrations"
    first = migrations / "001_create"
    second = migrations / "002_add"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "migration.sql").write_text(
        'CREATE TABLE "tasks" ("id" TEXT PRIMARY KEY, "name" TEXT NOT NULL);', encoding="utf-8"
    )
    (second / "migration.sql").write_text(
        'ALTER TABLE "tasks" ADD COLUMN "note" TEXT;\nCREATE INDEX "tasks_name_idx" ON "tasks"("name");',
        encoding="utf-8",
    )
    database = tmp_path / "local" / "EvoMind" / "data" / "prisma" / "workstation.db"
    backup = tmp_path / "local" / "EvoMind" / "backups" / "database"
    result = migrator.migrate(database, migrations, backup)
    assert result["ok"] is True
    assert len(result["schema_before_sha256"]) == 64
    assert len(result["schema_after_sha256"]) == 64
    assert result["schema_before_sha256"] != result["schema_after_sha256"]
    with sqlite3.connect(database) as connection:
        assert [row[1] for row in connection.execute('PRAGMA table_info("tasks")')] == ["id", "name", "note"]
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_release_database_baseline_adoption_rejects_same_names_with_wrong_schema(tmp_path: Path) -> None:
    migrator = load("release_migrator_baseline_adoption", "scripts/release_db_migrate.py")
    migrations = tmp_path / "migrations"
    baseline = migrations / "001_baseline"
    baseline.mkdir(parents=True)
    canonical = "\n".join(
        f'CREATE TABLE "{table}" ("id" TEXT PRIMARY KEY NOT NULL, "value" TEXT NOT NULL);'
        for table in sorted(migrator.BASELINE_TABLES)
    )
    (baseline / "migration.sql").write_text(canonical, encoding="utf-8")
    database = tmp_path / "data" / "workstation.db"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    try:
        for table in sorted(migrator.BASELINE_TABLES):
            connection.execute(f'CREATE TABLE "{table}" ("shell" INTEGER)')
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RuntimeError, match="baseline adoption rejected"):
        migrator.migrate(database, migrations, tmp_path / "backups")


def test_release_zip_preflight_rejects_bomb_alias_and_link_before_extract(tmp_path: Path) -> None:
    verifier = load("release_zip_preflight", "scripts/verify_release_bundle.py")

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("EvoMind/payload.bin", b"0" * 1024 * 1024)
    bomb_target = tmp_path / "bomb-out"
    with pytest.raises(RuntimeError, match="compression-ratio"):
        verifier.safe_extract(bomb, bomb_target, max_compression_ratio=10.0)
    assert not bomb_target.exists()

    alias = tmp_path / "alias.zip"
    with zipfile.ZipFile(alias, "w") as archive:
        archive.writestr("EvoMind/CON", b"x")
    with pytest.raises(RuntimeError, match="reserved Windows"):
        verifier.safe_extract(alias, tmp_path / "alias-out")

    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("EvoMind/link")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target")
    with pytest.raises(RuntimeError, match="link or special-file"):
        verifier.safe_extract(link, tmp_path / "link-out")


def test_official_release_verifier_authenticates_before_opening_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verifier = load("release_auth_before_zip", "scripts/verify_release_bundle.py")
    archive = tmp_path / "untrusted.zip"
    manifest = tmp_path / "untrusted.json"
    cli_tgz = tmp_path / "evomind-cli.tgz"
    archive.write_bytes(b"not a zip")
    manifest.write_text("{}", encoding="utf-8")
    cli_tgz.write_bytes(b"npm fixture")
    calls: list[str] = []

    def install_cli(*_args, **_kwargs):
        calls.append("install-cli")
        cli = tmp_path / "cli-prefix" / "node_modules" / "@evomind-ai" / "cli" / "bin" / "evomind.mjs"
        cli.parent.mkdir(parents=True)
        cli.write_text("", encoding="utf-8")
        return cli, {"name": "@evomind-ai/cli", "version": "0.3.0"}

    def fail_auth(*_args, **_kwargs):
        calls.append("authenticate")
        raise RuntimeError("signature rejected")

    def opened_zip(*_args, **_kwargs):
        calls.append("extract")
        raise AssertionError("ZIP must not be opened before package-pinned authentication")

    monkeypatch.setattr(verifier, "authenticate_release", fail_auth)
    monkeypatch.setattr(verifier, "safe_extract", opened_zip)
    monkeypatch.setattr(verifier, "install_cli_from_tgz", install_cli)
    monkeypatch.setattr(verifier, "cleanup_cli", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(verifier, "port_open", lambda _port: False)
    monkeypatch.setattr(sys, "argv", [
        "verify_release_bundle.py", "--zip", str(archive), "--manifest", str(manifest),
        "--cli-tgz", str(cli_tgz),
        "--port", "18192", "--runtime-port", "18765", "--temp-root", str(tmp_path / "temp"),
    ])
    assert verifier.main() == 1
    assert calls == ["install-cli", "authenticate"]
    assert "signature rejected" in capsys.readouterr().out


def test_release_backup_is_hash_bound_before_restore(tmp_path: Path) -> None:
    lifecycle = load("release_backup_hash_binding", "scripts/workstation_lifecycle.py")

    def package(version: str, body: str) -> Path:
        root = tmp_path / f"package-{version}"
        root.mkdir()
        payload = root / "start.ps1"
        payload.write_text(body, encoding="utf-8")
        (root / "release-manifest.json").write_text(json.dumps({
            "version": version,
            "release_id": f"release-{version}",
            "files": [{"path": "start.ps1", "sha256": lifecycle.file_sha256(payload)}],
        }), encoding="utf-8")
        return root

    root = tmp_path / "install"
    data = tmp_path / "data"
    backups = tmp_path / "backups"
    lifecycle.initialize(root, package("0.3.0", "v1"), data, backups)
    upgraded = lifecycle.upgrade(root, package("0.4.0", "v2"), backups)
    backup = Path(upgraded["backup"])
    (backup / "start.ps1").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        lifecycle.rollback(root, backup, backups)
    assert (root / "start.ps1").read_text(encoding="utf-8") == "v2"


def test_process_identity_mismatch_is_conflict_without_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = load("release_process_identity", "scripts/manage_workstation_dashboard.py")
    nonce = "n" * 43
    install = manager.normalize_path(manager.ROOT)
    command = f"{manager.normalize_path(sys.executable)} {install}/app/server.js evomind-{nonce}"
    actual = {
        "pid": 8123,
        "creation_time": "100",
        "creation_token": "100-token",
        "executable": manager.normalize_path(sys.executable),
        "command_line": manager.normalize_command_line(command),
        "command_line_raw": command,
        "cwd": install,
    }
    record = {
        "schema": "evomind.process_identity.v1",
        "role": "dashboard",
        "port": 18088,
        "pid": 8123,
        "creation_time": "99",
        "creation_token": "99-token",
        "executable": actual["executable"],
        "command_line": actual["command_line"],
        "cwd": install,
        "install_dir": install,
        "release_nonce": nonce,
    }
    monkeypatch.setattr(manager, "pid_running", lambda _pid: True)
    monkeypatch.setattr(manager, "process_identity", lambda _pid: actual)
    monkeypatch.setattr(manager, "pids_on_port", lambda _port: [8123])
    monkeypatch.setattr(manager.subprocess, "run", lambda *_a, **_k: pytest.fail("signal command must not run"))
    result = manager.stop_process_record(record)
    assert result["conflict"] is True
    assert "creation_token mismatch" in result["failures"]


def test_real_process_identity_binds_chinese_cwd_nonce_and_listener() -> None:
    manager = load("release_real_process_identity", "scripts/manage_workstation_dashboard.py")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    nonce = "r" * 43
    code = (
        "import socket,sys,time; s=socket.socket(); "
        "s.bind(('127.0.0.1',int(sys.argv[1]))); s.listen(); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(port), f"evomind-{nonce}", str(manager.ROOT.resolve())],
        cwd=manager.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    record = None
    launcher_record = None
    try:
        deadline = manager.time.monotonic() + 10
        while manager.time.monotonic() < deadline and not manager.pids_on_port(port):
            manager.time.sleep(0.1)
        listeners = manager.pids_on_port(port)
        assert len(listeners) == 1
        launcher_record = manager.make_process_record(
            process.pid, role="dashboard_launcher", port=port, cwd=manager.ROOT, release_nonce=nonce
        )
        record = manager.make_process_record(
            listeners[0], role="dashboard", port=port, cwd=manager.ROOT, release_nonce=nonce
        )
        matched, failures = manager.verify_process_record(record)
        assert matched, failures
        stopped = manager.stop_process_record(record, timeout=10)
        assert stopped["stopped"] is True, stopped
        if launcher_record["pid"] != record["pid"]:
            launcher_stopped = manager.stop_process_record(launcher_record, timeout=10, require_listener=False)
            assert launcher_stopped["stopped"] is True, launcher_stopped
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            if record:
                manager.stop_process_record(record, timeout=5, require_listener=False)
                if launcher_record and launcher_record["pid"] != record["pid"]:
                    manager.stop_process_record(launcher_record, timeout=5, require_listener=False)
            else:
                process.kill()


def test_legacy_pid_and_unknown_listener_are_never_signaled(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = load("release_unknown_listener", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "read_pid", lambda _port: 9991)
    monkeypatch.setattr(manager, "read_process_state", lambda _port: {})
    monkeypatch.setattr(manager, "fetch_status", lambda *_a, **_k: {"reachable": True})
    monkeypatch.setattr(manager, "runtime_service_port", lambda: 18765)
    monkeypatch.setattr(manager, "pids_on_port", lambda port: [9991] if port == 18088 else [])
    monkeypatch.setattr(manager, "pid_running", lambda _pid: True)
    monkeypatch.setattr(manager, "stop_process_record", lambda *_a, **_k: pytest.fail("unknown PID must not be signaled"))
    stopped, clean, conflicts = manager.stop_managed("127.0.0.1", 18088, 0.0, include_listener=True)
    assert stopped == []
    assert clean is False
    assert conflicts


def test_stale_reused_launcher_pid_does_not_block_verified_state_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("release_stale_launcher_cleanup", "scripts/manage_workstation_dashboard.py")
    nonce = "n" * 43
    state_file = tmp_path / "dashboard.process.json"
    runtime_state_file = tmp_path / "runtime.process.json"
    for path in (state_file, runtime_state_file, tmp_path / "dashboard.pid", tmp_path / "runtime.pid"):
        path.write_text("{}", encoding="utf-8")
    launcher = {
        "schema": "evomind.process_identity.v1",
        "role": "runtime_launcher",
        "port": 18765,
        "pid": 4242,
        "creation_time": "old",
        "creation_token": "old-token",
        "executable": manager.normalize_path(sys.executable),
        "command_line": "old launcher",
        "cwd": manager.normalize_path(manager.ROOT),
        "install_dir": manager.normalize_path(manager.ROOT),
        "release_nonce": nonce,
    }
    monkeypatch.setattr(manager, "read_pid", lambda _port: None)
    monkeypatch.setattr(manager, "read_process_state", lambda _port: {
        "runtime_port": 18765, "processes": {"runtime_launcher": launcher}
    })
    monkeypatch.setattr(manager, "fetch_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(manager, "pids_on_port", lambda _port: [])
    monkeypatch.setattr(manager, "pid_running", lambda pid: pid == 4242)
    monkeypatch.setattr(manager, "process_identity", lambda _pid: None)
    monkeypatch.setattr(
        manager, "runtime_paths",
        lambda _port: (tmp_path / "dashboard.pid", state_file, tmp_path / "out", tmp_path / "err"),
    )
    monkeypatch.setattr(
        manager, "runtime_service_paths",
        lambda _port: (tmp_path / "runtime.pid", runtime_state_file, tmp_path / "rout", tmp_path / "rerr"),
    )
    monkeypatch.setattr(manager, "bootstrap_url_path", lambda _port: tmp_path / "bootstrap.once")

    stopped, clean, conflicts = manager.stop_managed("127.0.0.1", 18088, 0.0, include_listener=True)

    assert stopped == []
    assert clean is True
    assert conflicts == []
    assert not state_file.exists()
    assert not runtime_state_file.exists()


def test_legacy_v1_stop_requires_full_identity_before_same_handle_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = load("release_legacy_v1_stop", "scripts/manage_workstation_dashboard.py")
    standalone = tmp_path / "web" / ".next" / "standalone" / "server.js"
    standalone.parent.mkdir(parents=True)
    standalone.write_text("server", encoding="utf-8")
    pid_file = tmp_path / "dashboard.pid"
    state_file = tmp_path / "dashboard.process.json"
    pid_file.write_text("4242", encoding="utf-8")
    created = "2026-08-03T00:00:00.0000000Z"
    state_file.write_text(json.dumps({
        "launcher_pid": 4242,
        "listener_pids": [4242],
        "managed_pid": 4242,
        "host": "127.0.0.1",
        "port": 18088,
        "mode": "source-standalone",
        "started_at": 1785715200.0,
    }), encoding="utf-8")
    stopped = False
    signaled: list[dict] = []
    executable = manager.normalize_path(sys.executable)
    identity = {
        "pid": 4242,
        "creation_time": created,
        "creation_token": "123456789",
        "executable": executable,
        "command_line_raw": f'"{sys.executable}" "{standalone}" --unexpected',
        "cwd": manager.normalize_path(standalone.parent),
    }
    monkeypatch.setattr(manager, "SOURCE_STANDALONE_SERVER", standalone)
    monkeypatch.setattr(manager, "bundle_mode", lambda: False)
    monkeypatch.setattr(manager, "runtime_paths", lambda _port: (pid_file, state_file, tmp_path / "out", tmp_path / "err"))
    monkeypatch.setattr(manager, "bootstrap_url_path", lambda _port: tmp_path / "bootstrap")
    monkeypatch.setattr(manager, "pid_running", lambda _pid: not stopped)
    monkeypatch.setattr(manager, "pids_on_port", lambda _port: [] if stopped else [4242])
    monkeypatch.setattr(manager, "process_identity", lambda _pid: identity)
    monkeypatch.setattr(manager, "fetch_status", lambda *_args, **_kwargs: {
        "http_status": 200, "service": "evomind-workstation", "version": "0.3.0",
    })
    monkeypatch.setattr(manager, "application_version", lambda: "0.3.0")

    def terminate(record, timeout):
        nonlocal stopped
        signaled.append(record)
        stopped = True
        return True, []

    monkeypatch.setattr(manager, "_windows_terminate_verified", terminate)
    args = type("Args", (), {"host": "127.0.0.1", "port": 18088, "timeout": 5.0})()
    with pytest.raises(SystemExit, match="argv, or cwd binding mismatch"):
        manager.stop_legacy_v1(args)
    assert signaled == []
    identity["command_line_raw"] = f'"{sys.executable}" "{standalone}"'
    manager.stop_legacy_v1(args)
    assert len(signaled) == 1
    assert signaled[0]["creation_token"] == "123456789"
    assert not pid_file.exists() and not state_file.exists()
    assert json.loads(capsys.readouterr().out)["same_handle_termination"] is True


def test_release_wrappers_remove_unsigned_upgrade_and_verifier_uses_official_lifecycle() -> None:
    upgrade = (ROOT / "upgrade.ps1").read_text(encoding="utf-8")
    manager = (ROOT / "scripts" / "manage_workstation_lifecycle.ps1").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_release_bundle.py").read_text(encoding="utf-8")
    assert "evomind" in upgrade and "--manifest" in upgrade
    assert "Expand-Archive" not in manager
    assert "direct PackagePath upgrades were removed" in manager
    for command in ('"install"', '"status"', '"doctor"', '"stop"', '"start"', '"uninstall"'):
        assert command in verifier


def test_bundle_install_phase_matches_the_durable_cli_transaction_order() -> None:
    installer = (ROOT / "install.ps1").read_text(encoding="utf-8")
    cli = (ROOT / "packages" / "evomind-cli" / "src" / "core.mjs").read_text(encoding="utf-8")

    phase = 'writeTransactionPhase(paths, journal, "python_env_prepared")'
    install = 'runBundleScript(destination, "install.ps1"'
    assert '$Transaction.phase -ne "python_env_prepared"' in installer
    assert phase in cli
    assert install in cli
    assert cli.index(phase) < cli.index(install)
    assert 'transactionId.replaceAll("-", "").slice(0, 16).toLowerCase()' in cli
    assert '$transactionId.Replace(\'-\', \'\').Substring(0, 16).ToLowerInvariant()' in installer
    assert 'runtime\\python-env\\.stage-' in installer
    env_writer = installer.split("function Set-ManagedEnvValues", 1)[1].split("Write-Host", 1)[0]
    assert "New-Item -ItemType Directory -Force -Path $parent" in env_writer
    assert env_writer.index("New-Item -ItemType Directory") < env_writer.index("[IO.File]::WriteAllLines")
    assert '$env:PYTHONDONTWRITEBYTECODE = "1"' in installer
    assert "$env:PYTHONPYCACHEPREFIX = $null" in installer
    assert 'Join-Path $DataDir "tmp\\pycache"' not in installer
    assert '"-m", "py_compile"' not in installer
    assert "compile(pathlib.Path(value).read_text" in installer


def test_runtime_build_manifest_binds_source_backend_frontend_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("release_runtime_build_manifest", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SOURCE_APP_DIR", tmp_path / "web")
    monkeypatch.setattr(manager, "DEFAULT_DATABASE_PATH", tmp_path / "workstation.db")
    monkeypatch.setattr(manager, "source_tree_digest", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr(manager, "git_source_identity", lambda: ("b" * 40, True))
    monkeypatch.setattr(manager, "application_version", lambda: "0.3.0")
    monkeypatch.setattr(manager, "backend_version", lambda: "0.3.0")
    monkeypatch.setattr(
        manager,
        "database_schema_identity",
        lambda *_args, **_kwargs: {
            "version": "20260728171000_performance_indexes",
            "sha256": "c" * 64,
        },
    )

    manifest = manager.runtime_build_manifest("build-123")

    assert manifest == {
        "schema": "evomind.runtime_build.v1",
        "commit_hash": "b" * 40,
        "source_dirty": True,
        "source_tree_sha256": "a" * 64,
        "build_id": "build-123",
        "build_time": manifest["build_time"],
        "backend_version": "0.3.0",
        "frontend_version": "0.3.0",
        "database_schema_version": "20260728171000_performance_indexes",
        "database_schema_sha256": "c" * 64,
    }
    assert manifest["build_time"].endswith("Z")


def test_source_build_stale_uses_exact_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("release_exact_source_build_identity", "scripts/manage_workstation_dashboard.py")
    app = tmp_path / "web"
    active = app / ".next"
    active.mkdir(parents=True)
    (active / "BUILD_ID").write_text("build-123", encoding="utf-8")
    (active / "runtime-build-manifest.json").write_text(
        json.dumps({
            "schema": "evomind.runtime_build.v1",
            "source_tree_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "SOURCE_APP_DIR", app)
    monkeypatch.setattr(manager, "source_tree_digest", lambda *_args, **_kwargs: "a" * 64)
    assert manager.source_build_stale() is False

    monkeypatch.setattr(manager, "source_tree_digest", lambda *_args, **_kwargs: "d" * 64)
    assert manager.source_build_stale() is True


def test_source_build_registration_creates_managed_install_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load("release_source_install_marker", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    manifest = {
        "schema": "evomind.runtime_build.v1",
        "commit_hash": "b" * 40,
        "source_tree_sha256": "a" * 64,
        "build_id": "build-123",
        "build_time": "2026-08-09T00:00:00Z",
        "backend_version": "0.3.0",
        "frontend_version": "0.3.0",
        "database_schema_version": "20260728171000_performance_indexes",
        "database_schema_sha256": "c" * 64,
        "source_dirty": True,
    }

    marker_path = manager.write_source_install_marker(manifest)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert marker_path == tmp_path / ".workstation-install.json"
    assert marker["format_version"] == 1
    assert marker["product"] == "research-workstation"
    assert marker["layout"] == "source_tree"
    assert marker["data_root"] == str(tmp_path.resolve())
    assert marker["runtime_build"]["build_id"] == "build-123"
    assert marker["managed_files"] == []


def test_source_lifecycle_status_uses_the_same_data_root_as_source_launch() -> None:
    script = (ROOT / "scripts" / "manage_workstation_lifecycle.ps1").read_text(encoding="utf-8")
    assert 'elseif ($env:WORKSTATION_DATA_DIR)' in script
    assert 'else { $Root }' in script
