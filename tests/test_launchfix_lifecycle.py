from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"launchfix_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_ready_and_degraded_statuses_are_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = load_script("manage_local_gateway.py")
    monkeypatch.setenv("WORKSTATION_ROOT", str(tmp_path))
    ready = gateway.publish_status(
        {
            "base_url": gateway.DEFAULT_BASE_URL,
            "tcp_reachable": True,
            "http_status": 200,
            "models_endpoint_ok": True,
            "credential_present": False,
        },
        "status_only",
        None,
    )
    assert ready["status"] == "ready"
    assert ready["ready"] is True
    degraded = gateway.publish_status(
        {
            "base_url": gateway.DEFAULT_BASE_URL,
            "tcp_reachable": False,
            "http_status": None,
            "models_endpoint_ok": False,
            "credential_present": False,
        },
        "runtime_manifest_missing",
        None,
    )
    assert degraded["status"] == "degraded_unavailable"
    assert degraded["local_fallback_available"] is True
    summary = json.loads((tmp_path / "workspace" / "workstation_summary.json").read_text(encoding="utf-8"))
    assert summary["connector_status"]["local_gateway"]["state"] == "Local Deterministic Fallback"


def test_gateway_rejects_non_loopback_endpoint() -> None:
    gateway = load_script("manage_local_gateway.py")
    with pytest.raises(SystemExit):
        gateway.validate_base_url("https://example.invalid/v1")


def test_release_paths_and_migrations_are_non_destructive() -> None:
    lifecycle = load_script("workstation_lifecycle.py")
    migrations = load_script("apply_workstation_migrations.py")
    assert lifecycle.safe_relative("app/server.js").as_posix() == "app/server.js"
    with pytest.raises(ValueError):
        lifecycle.safe_relative("../outside.txt")
    with pytest.raises(ValueError):
        lifecycle.safe_relative("user-data/private.db")
    migrations.validate_additive_migration("add_index", 'CREATE INDEX "x" ON "tasks"("updated_at");')
    with pytest.raises(RuntimeError):
        migrations.validate_additive_migration("destructive", 'DROP TABLE "tasks";')

