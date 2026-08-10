from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = ROOT / "scripts" / "manage_workstation_dashboard.py"


def load_manager(name: str):
    spec = importlib.util.spec_from_file_location(name, MANAGER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source_tree(root: Path) -> Path:
    app = root / "web-app"
    (app / "src").mkdir(parents=True)
    (app / "src" / "page.tsx").write_text("export default function Page() { return null; }", encoding="utf-8")
    (app / "public").mkdir()
    (app / "public" / "favicon.ico").write_bytes(b"icon")
    migration = app / "prisma" / "migrations" / "001_init" / "migration.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text(
        "CREATE TABLE runtime_build_test (id INTEGER PRIMARY KEY);\n",
        encoding="utf-8",
    )
    (app / "prisma" / "schema.prisma").write_text("generator client { provider = \"prisma-client-js\" }", encoding="utf-8")
    database = app / "prisma" / "workstation.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE runtime_build_test (id INTEGER PRIMARY KEY)")
    (app / ".env").write_text("SECRET=must-not-enter-staging", encoding="utf-8")
    for name, value in {
        "package.json": "{}",
        "package-lock.json": "{}",
        "next.config.mjs": "export default { output: 'standalone' };",
        "tsconfig.json": "{}",
        "next-env.d.ts": "",
    }.items():
        (app / name).write_text(value, encoding="utf-8")
    active = app / ".next"
    (active / "standalone").mkdir(parents=True)
    (active / "BUILD_ID").write_text("known-good", encoding="utf-8")
    (active / "standalone" / "server.js").write_text("old", encoding="utf-8")
    return app


def write_candidate(cwd: Path, build_id: str = "candidate-good") -> None:
    candidate = cwd / ".next"
    (candidate / "standalone").mkdir(parents=True)
    (candidate / "static").mkdir()
    (candidate / "BUILD_ID").write_text(build_id, encoding="utf-8")
    (candidate / "standalone" / "server.js").write_text("new", encoding="utf-8")


def configure_manager(manager, app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "SOURCE_APP_DIR", app)
    monkeypatch.setattr(manager, "SOURCE_STANDALONE_SERVER", app / ".next" / "standalone" / "server.js")
    monkeypatch.setattr(manager, "DEFAULT_DATABASE_PATH", app / "prisma" / "workstation.db")
    monkeypatch.setattr(manager, "node_command", lambda: "node")
    monkeypatch.setattr(manager, "next_cli_path", lambda: "next-cli")


def test_dashboard_env_binds_release_version_without_npm_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_version_env")
    app = tmp_path / "source-app"
    app.mkdir()
    (app / "package.json").write_text('{"name":"evomind","version":"9.8.7"}', encoding="utf-8")
    monkeypatch.setattr(manager, "SOURCE_APP_DIR", app)
    monkeypatch.setattr(manager, "STANDALONE_SERVER", tmp_path / "missing-bundle" / "server.js")
    monkeypatch.setenv("npm_package_version", "stale-parent-version")

    env = manager.dashboard_env("127.0.0.1", 18088)

    assert env["npm_package_version"] == "9.8.7"


def test_dashboard_env_binds_loopback_gateway_to_its_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_local_gateway_key")
    gateway = tmp_path / "gateway.json"
    gateway.write_text(json.dumps({"api-keys": ["local-gateway-key"]}), encoding="utf-8")
    monkeypatch.setenv("EVOMIND_LOCAL_GATEWAY_CONFIG", str(gateway))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-parent-key")

    env = manager.dashboard_env("127.0.0.1", 18088)

    assert env["OPENAI_API_KEY"] == "local-gateway-key"
    assert env["EVOLUTION_PRIMARY_PROVIDER"] == "openai"
    assert env["EVOLUTION_PROVIDER_STRICT"] == "true"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:65068/v1"
    assert env["OPENAI_MODEL"] == "gpt-5.6-sol"


def test_dashboard_env_overrides_unrelated_parent_model_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_overrides_parent_model_route")
    gateway = tmp_path / "gateway.json"
    gateway.write_text(json.dumps({"api-keys": ["local-gateway-key"]}), encoding="utf-8")
    monkeypatch.setenv("EVOMIND_LOCAL_GATEWAY_CONFIG", str(gateway))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "unrelated-model")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "unrelated-provider")
    monkeypatch.delenv("EVOLUTION_PROVIDER_STRICT", raising=False)

    env = manager.dashboard_env("127.0.0.1", 18088)

    assert env["OPENAI_API_KEY"] == "local-gateway-key"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:65068/v1"
    assert env["OPENAI_MODEL"] == "gpt-5.6-sol"
    assert env["EVOLUTION_PRIMARY_PROVIDER"] == "openai"
    assert env["EVOLUTION_PROVIDER_STRICT"] == "true"


def test_dashboard_env_uses_the_installed_interactive_gateway_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_interactive_gateway_profile")
    gateway = tmp_path / "gateway.json"
    gateway.write_text(json.dumps({"api-keys": ["local-gateway-key"]}), encoding="utf-8")
    profile = tmp_path / "openai_gateway_metadata.json"
    profile.write_text(
        json.dumps({
            "interactive_reasoning_effort": "low",
            "research_reasoning_effort": "high",
            "service_tier": "priority",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOMIND_LOCAL_GATEWAY_CONFIG", str(gateway))
    monkeypatch.setenv("EVOMIND_OPENAI_GATEWAY_METADATA", str(profile))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "flex")

    env = manager.dashboard_env("127.0.0.1", 18088)

    assert env["OPENAI_REASONING_EFFORT"] == "low"
    assert env["OPENAI_SERVICE_TIER"] == "priority"


def test_dashboard_env_drops_unrelated_key_when_loopback_binding_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_missing_local_gateway_key")
    monkeypatch.setenv("EVOMIND_LOCAL_GATEWAY_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-parent-key")

    env = manager.dashboard_env("127.0.0.1", 18088)

    assert "OPENAI_API_KEY" not in env


def test_build_source_activates_validated_staging_and_retains_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_transaction_success")
    app = make_source_tree(tmp_path)
    configure_manager(manager, app, monkeypatch)

    def fake_run(command, *, cwd, **kwargs):
        staging = Path(cwd)
        assert staging != app
        assert (app / ".next" / "BUILD_ID").read_text(encoding="utf-8") == "known-good"
        assert not (staging / ".env").exists()
        assert not (staging / "prisma" / "workstation.db").exists()
        write_candidate(staging)
        return subprocess.CompletedProcess(command, 0, stdout="compiled", stderr="")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    metadata = manager.build_source({"WORKSTATION_BUILD_TIMEOUT_SECONDS": "60"})

    assert metadata["status"] == "built"
    assert metadata["build_id"] == "candidate-good"
    assert (app / ".next" / "BUILD_ID").read_text(encoding="utf-8") == "candidate-good"
    pointer = json.loads((app / ".next-rollback.json").read_text(encoding="utf-8"))
    assert pointer["active_build_id"] == "candidate-good"
    assert pointer["rollback_build_id"] == "known-good"
    assert pointer["rollback_dir"].startswith(".next-rollback-")
    rollback = app / pointer["rollback_dir"]
    assert (rollback / "BUILD_ID").read_text(encoding="utf-8") == "known-good"
    assert not (app / ".next-build-staging").exists()


def test_build_source_failure_never_mutates_active_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = load_manager("dashboard_transaction_failure")
    app = make_source_tree(tmp_path)
    configure_manager(manager, app, monkeypatch)

    def fake_run(command, *, cwd, **kwargs):
        partial = Path(cwd) / ".next"
        partial.mkdir()
        (partial / "BUILD_ID").write_text("partial", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, stdout="compile failed", stderr="synthetic failure")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        manager.build_source({})

    payload = json.loads(str(exc_info.value))
    assert payload["stage"] == "build"
    assert payload["active_build_preserved"] is True
    assert (app / ".next" / "BUILD_ID").read_text(encoding="utf-8") == "known-good"
    assert not (app / ".next-rollback.json").exists()
    assert not (app / ".next-build-staging").exists()


def test_activation_error_restores_exact_previous_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = load_manager("dashboard_transaction_switch_failure")
    app = make_source_tree(tmp_path)
    configure_manager(manager, app, monkeypatch)

    def fake_run(command, *, cwd, **kwargs):
        write_candidate(Path(cwd))
        return subprocess.CompletedProcess(command, 0, stdout="compiled", stderr="")

    real_replace = manager.os.replace

    def fail_candidate_switch(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == app / ".next" and ".next-build-staging" in source_path.parts:
            raise OSError("synthetic atomic-switch failure")
        return real_replace(source, destination)

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    monkeypatch.setattr(manager.os, "replace", fail_candidate_switch)
    with pytest.raises(OSError, match="synthetic atomic-switch failure"):
        manager.build_source({})

    assert (app / ".next" / "BUILD_ID").read_text(encoding="utf-8") == "known-good"
    assert not list(app.glob(".next-rollback-*"))
    assert not (app / ".next-rollback.json").exists()
    assert not (app / ".next-build-staging").exists()


def test_activation_retries_one_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_transaction_transient_permission")
    app = make_source_tree(tmp_path)
    configure_manager(manager, app, monkeypatch)

    def fake_run(command, *, cwd, **kwargs):
        write_candidate(Path(cwd))
        return subprocess.CompletedProcess(command, 0, stdout="compiled", stderr="")

    real_replace = manager.os.replace
    candidate_attempts = 0

    def transient_candidate_switch(source, destination):
        nonlocal candidate_attempts
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == app / ".next" and ".next-build-staging" in source_path.parts:
            candidate_attempts += 1
            if candidate_attempts == 1:
                raise PermissionError("synthetic transient sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    monkeypatch.setattr(manager.os, "replace", transient_candidate_switch)
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)

    metadata = manager.build_source({})

    assert metadata["status"] == "built"
    assert candidate_attempts == 2
    assert (app / ".next" / "BUILD_ID").read_text(encoding="utf-8") == "candidate-good"


def test_nested_standalone_entry_is_normalized_to_stable_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_manager("dashboard_nested_standalone")
    app = make_source_tree(tmp_path)
    configure_manager(manager, app, monkeypatch)

    def fake_run(command, *, cwd, **kwargs):
        candidate = Path(cwd) / ".next"
        nested = candidate / "standalone" / ".next-build-staging" / "candidate-test"
        (nested / ".next").mkdir(parents=True)
        (candidate / "standalone").mkdir(exist_ok=True)
        (candidate / "static").mkdir()
        (candidate / "BUILD_ID").write_text("nested-candidate", encoding="utf-8")
        (candidate / "standalone" / "package.json").write_text('{"type":"commonjs"}', encoding="utf-8")
        (nested / "package.json").write_text('{"name":"research-agent-workstation"}', encoding="utf-8")
        (nested / "server.js").write_text("nested-server", encoding="utf-8")
        (nested / ".next" / "required-server-file.js").write_text("compiled", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="compiled", stderr="")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    manager.build_source({})

    standalone = app / ".next" / "standalone"
    assert (standalone / "server.js").read_text(encoding="utf-8") == "nested-server"
    assert json.loads((standalone / "package.json").read_text(encoding="utf-8"))["name"] == "research-agent-workstation"
    assert (standalone / ".next" / "required-server-file.js").is_file()
    assert not (standalone / ".next-build-staging").exists()
