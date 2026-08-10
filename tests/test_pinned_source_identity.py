from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_taxi_source_audited_plan import portable_freshness_value
from scripts.pinned_source_identity import (
    PinnedPathError,
    bind_legacy_record_to_project_anchor,
    resolve_pinned_project_path,
)


def test_relative_identity_is_authoritative_across_checkouts(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "runner.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    resolved = resolve_pinned_project_path(
        {
            "path": "D:/old-machine/checkout/scripts/runner.py",
            "relative_path": "scripts/runner.py",
        },
        tmp_path,
    )

    assert resolved == source.resolve()


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.py", "/absolute.py", "C:/outside.py", "scripts/../outside.py"],
)
def test_relative_identity_rejects_escape(relative_path: str, tmp_path: Path) -> None:
    with pytest.raises(PinnedPathError):
        resolve_pinned_project_path({"relative_path": relative_path}, tmp_path)


def test_legacy_absolute_identity_must_already_be_confined(tmp_path: Path) -> None:
    source = tmp_path / "legacy.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    assert resolve_pinned_project_path({"path": str(source)}, tmp_path) == source.resolve()

    with pytest.raises(PinnedPathError):
        resolve_pinned_project_path({"path": str(tmp_path.parent / "outside.py")}, tmp_path)


def test_legacy_anchor_discards_only_the_machine_specific_prefix(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "hpc" / "job123" / "collected" / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    record = bind_legacy_record_to_project_anchor(
        {"path": r"D:\old-checkout\workspace\hpc\job123\collected\manifest.json"},
        "workspace/hpc/job123",
    )
    assert record["relative_path"] == "workspace/hpc/job123/collected/manifest.json"
    assert resolve_pinned_project_path(record, tmp_path) == source.resolve()

    with pytest.raises(PinnedPathError):
        bind_legacy_record_to_project_anchor(
            {"path": r"D:\old-checkout\workspace\hpc\other\manifest.json"},
            "workspace/hpc/job123",
        )


def test_portable_freshness_ignores_only_projected_absolute_paths() -> None:
    first = {
        "path": r"D:\first\scripts\runner.py",
        "relative_path": "scripts/runner.py",
        "sha256": "a" * 64,
    }
    second = {
        "path": r"E:\second\scripts\runner.py",
        "relative_path": "scripts/runner.py",
        "sha256": "a" * 64,
    }
    assert portable_freshness_value(first) == portable_freshness_value(second)
    assert portable_freshness_value({"path": "remote-only"}) == {"path": "remote-only"}
