from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import merge_siim_sidecar_when_complete as merger


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_fixture(tmp_path: Path) -> dict[str, Path]:
    inventory_path = tmp_path / "inventory.json"
    sidecar = tmp_path / "sidecar"
    destination = tmp_path / "canonical"
    report_path = tmp_path / "sidecar_report.json"
    output = tmp_path / "public_staging_report.json"
    entries = [
        {"path": "train.csv", "size": 3},
        {"path": "jpeg/train/ISIC_1.jpg", "size": 4},
        {"path": "jpeg/test/ISIC_2.jpg", "size": 5},
    ]
    inventory = {
        "schema": merger.INVENTORY_SCHEMA,
        "competition_id": merger.COMPETITION,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "manifest_sha256": "fixture-inventory-hash",
        "entries": entries,
    }
    write_json(inventory_path, inventory)
    contents = {
        "train.csv": b"abc",
        "jpeg/train/ISIC_1.jpg": b"1234",
        "jpeg/test/ISIC_2.jpg": b"56789",
    }
    for name, content in contents.items():
        path = sidecar / Path(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(
        report_path,
        {
            "schema": merger.REPORT_SCHEMA,
            "status": "size_verified_complete",
            "competition_id": merger.COMPETITION,
            "completed_files": inventory["file_count"],
            "completed_bytes": inventory["total_bytes"],
            "inventory_manifest_sha256": inventory["manifest_sha256"],
            "errors": [],
            "private_paths_requested": False,
            "process_signals_sent": 0,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    return {
        "inventory": inventory_path,
        "sidecar": sidecar,
        "report": report_path,
        "destination": destination,
        "output": output,
    }


def test_merge_links_only_missing_files_and_emits_complete_report(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    destination = fixture["destination"]
    destination.mkdir()
    existing = destination / "train.csv"
    existing.write_bytes(b"abc")

    payload = merger.merge_sidecar(
        inventory_path=fixture["inventory"],
        sidecar=fixture["sidecar"],
        sidecar_report_path=fixture["report"],
        destination=destination,
        output=fixture["output"],
    )

    assert payload["status"] == "size_verified_complete"
    assert payload["completed_files"] == 3
    assert payload["completed_bytes"] == 12
    assert payload["already_present_files"] == 1
    assert payload["linked_files"] == 2
    assert payload["linked_bytes"] == 9
    assert payload["merge_races_verified"] == 0
    assert existing.read_bytes() == b"abc"
    assert os.path.samefile(
        fixture["sidecar"] / "jpeg" / "train" / "ISIC_1.jpg",
        destination / "jpeg" / "train" / "ISIC_1.jpg",
    )
    saved = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["process_signals_sent"] == 0
    assert saved["official_grader_executed"] is False
    assert saved["kaggle_submission_executed"] is False


def test_merge_rejects_wrong_sized_existing_canonical_file(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    destination = fixture["destination"]
    destination.mkdir()
    (destination / "train.csv").write_bytes(b"wrong-size")

    with pytest.raises(RuntimeError, match="Canonical SIIM file has wrong size"):
        merger.merge_sidecar(
            inventory_path=fixture["inventory"],
            sidecar=fixture["sidecar"],
            sidecar_report_path=fixture["report"],
            destination=destination,
            output=fixture["output"],
        )

    assert not fixture["output"].exists()
    assert not (destination / "jpeg" / "train" / "ISIC_1.jpg").exists()


def test_merge_rejects_wrong_sized_sidecar_file_before_linking(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    corrupt = fixture["sidecar"] / "jpeg" / "train" / "ISIC_1.jpg"
    corrupt.write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="sidecar tree differs"):
        merger.merge_sidecar(
            inventory_path=fixture["inventory"],
            sidecar=fixture["sidecar"],
            sidecar_report_path=fixture["report"],
            destination=fixture["destination"],
            output=fixture["output"],
        )

    assert not fixture["output"].exists()
    assert not fixture["destination"].exists()
