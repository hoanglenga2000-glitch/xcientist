from __future__ import annotations

from pathlib import Path

import xsci.dashboard as dashboard


def _workstation_root(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "manage_workstation_dashboard.py").write_text("# manager\n", encoding="utf-8")
    web = root / "web" / "research-agent-workstation"
    web.mkdir(parents=True)
    (web / "package.json").write_text("{}\n", encoding="utf-8")
    return root


def test_dashboard_resolves_explicit_workstation_root(tmp_path, monkeypatch):
    root = _workstation_root(tmp_path / "workstation")
    monkeypatch.setenv("EVOMIND_WORKSTATION_ROOT", str(root))
    monkeypatch.setattr(dashboard, "ROOT_POINTER", tmp_path / "missing-pointer")
    monkeypatch.setattr(dashboard, "SOURCE_ROOT", tmp_path / "missing-source")

    assert dashboard.resolve_workstation_root() == root.resolve()


def test_dashboard_resolves_installer_pointer_with_utf8_bom(tmp_path, monkeypatch):
    root = _workstation_root(tmp_path / "workstation")
    pointer = tmp_path / "workstation-root.txt"
    pointer.write_text(f"\ufeff{root}\n", encoding="utf-8")
    monkeypatch.delenv("EVOMIND_WORKSTATION_ROOT", raising=False)
    monkeypatch.setattr(dashboard, "ROOT_POINTER", pointer)
    monkeypatch.setattr(dashboard, "SOURCE_ROOT", tmp_path / "missing-source")

    assert dashboard.resolve_workstation_root() == root.resolve()


def test_dashboard_fails_closed_without_full_source(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("EVOMIND_WORKSTATION_ROOT", raising=False)
    monkeypatch.setattr(dashboard, "ROOT_POINTER", tmp_path / "missing-pointer")
    monkeypatch.setattr(dashboard, "SOURCE_ROOT", tmp_path / "missing-source")

    assert dashboard.run_dashboard("status") == 1
    assert "full EvoMind workstation source bundle" in capsys.readouterr().out
