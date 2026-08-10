from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import build_mle_prepared_contract as contract

FIXTURE_COMMIT = "a" * 40
FIXTURE_VERSION = "1.0.0"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_id = "spooky-author-identification"
    competition = tmp_path / "data" / task_id
    public = competition / "prepared" / "public"
    private = competition / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "train.csv").write_text("id,text,author\n1,x,EAP\n", encoding="utf-8")
    (public / "test.csv").write_text("id,text\n2,y\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text("id,EAP,HPL,MWS\n", encoding="utf-8")
    (public / "nested").mkdir()
    (public / "nested" / "evidence.bin").write_bytes(b"recursive-stat-evidence")
    (private / "test.csv").write_text("id,author\n2,HPL\n", encoding="utf-8")
    zip_path = competition / f"{task_id}.zip"
    zip_path.write_bytes(b"official-kaggle-zip")

    upstream = tmp_path / "upstream"
    source = upstream / "mlebench" / "competitions" / task_id
    source.mkdir(parents=True)
    (source / "prepare.py").write_text("def prepare(raw, public, private):\n    pass\n", encoding="utf-8")
    (source / "config.yaml").write_text(f"id: {task_id}\n", encoding="utf-8")
    checksums = {
        "zip": contract.md5_file(zip_path),
        "public": {
            path.name: contract.md5_file(path)
            for path in sorted(public.glob("*.csv"))
        },
        "private": {"test.csv": contract.md5_file(private / "test.csv")},
    }
    (source / "checksums.yaml").write_text(yaml.safe_dump(checksums, sort_keys=True), encoding="utf-8")
    return competition, upstream


def test_prepared_contract_contains_pinned_sources_official_md5_and_recursive_totals(tmp_path: Path) -> None:
    competition, upstream = _fixture(tmp_path)
    payload = contract.build_prepared_contract(
        competition,
        upstream,
        upstream_commit=FIXTURE_COMMIT,
        upstream_version=FIXTURE_VERSION,
        pinned_upstream_commit=FIXTURE_COMMIT,
        pinned_upstream_version=FIXTURE_VERSION,
    )
    assert payload["status"] == "verified"
    assert payload["upstream"]["repository_commit"] == FIXTURE_COMMIT
    assert payload["upstream"]["package_version"] == FIXTURE_VERSION
    assert set(payload["upstream"]["source_sha256"]) == {
        "prepare.py",
        "config.yaml",
        "checksums.yaml",
    }
    assert all(len(value) == 64 for value in payload["upstream"]["source_sha256"].values())
    assert payload["zip"]["expected_md5"] == payload["zip"]["actual_md5"]
    assert payload["public"]["expected_md5"] == payload["public"]["actual_md5"]
    assert payload["private"]["expected_md5"] == payload["private"]["actual_md5"]
    assert payload["public"]["recursive_file_count"] == 4
    assert payload["public"]["recursive_bytes"] == sum(
        path.stat().st_size for path in (competition / "prepared" / "public").rglob("*") if path.is_file()
    )

    contract_path = competition / "prepared-contract.json"
    _write_json(contract_path, payload)
    verified = contract.verify_prepared_contract(
        contract_path,
        competition,
        upstream,
        expected_upstream_commit=FIXTURE_COMMIT,
        expected_upstream_version=FIXTURE_VERSION,
        upstream_commit_override=FIXTURE_COMMIT,
        upstream_version_override=FIXTURE_VERSION,
    )
    assert verified["status"] == "verified"
    assert verified["contract_matches_recomputed"] is True


def test_prepared_contract_verification_fails_after_official_file_or_source_drift(tmp_path: Path) -> None:
    competition, upstream = _fixture(tmp_path)
    payload = contract.build_prepared_contract(
        competition,
        upstream,
        upstream_commit=FIXTURE_COMMIT,
        upstream_version=FIXTURE_VERSION,
        pinned_upstream_commit=FIXTURE_COMMIT,
        pinned_upstream_version=FIXTURE_VERSION,
    )
    contract_path = competition / "prepared-contract.json"
    _write_json(contract_path, payload)
    (competition / "prepared" / "public" / "train.csv").write_text("tampered\n", encoding="utf-8")
    (upstream / "mlebench" / "competitions" / competition.name / "prepare.py").write_text(
        "def prepare(raw, public, private):\n    raise RuntimeError\n",
        encoding="utf-8",
    )
    verified = contract.verify_prepared_contract(
        contract_path,
        competition,
        upstream,
        expected_upstream_commit=FIXTURE_COMMIT,
        expected_upstream_version=FIXTURE_VERSION,
        upstream_commit_override=FIXTURE_COMMIT,
        upstream_version_override=FIXTURE_VERSION,
    )
    assert verified["status"] == "failed_closed"
    assert verified["contract_matches_recomputed"] is False
    assert verified["recomputed"]["checks"]["official_public_md5_matches"] is False


def test_siim_inventory_scan_reports_missing_wrong_size_and_part_counts(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "train.csv").write_bytes(b"abc")
    image = public / "jpeg" / "test" / "ISIC_1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"xx")
    (public / "download.jpg.part").write_bytes(b"partial")
    expected = {
        "manifest_sha256": "f" * 64,
        "file_count": 3,
        "total_bytes": 12,
        "train_jpeg_count": 0,
        "test_jpeg_count": 2,
    }
    inventory = tmp_path / "public_staging_inventory.json"
    _write_json(
        inventory,
        {
            "schema": contract.SIIM_INVENTORY_SCHEMA,
            "competition_id": contract.SIIM_TASK,
            "manifest_sha256": expected["manifest_sha256"],
            "file_count": 3,
            "total_bytes": 12,
            "train_jpeg_count": 0,
            "test_jpeg_count": 2,
            "entries": [
                {"path": "train.csv", "size": 3},
                {"path": "jpeg/test/ISIC_1.jpg", "size": 4},
                {"path": "jpeg/test/ISIC_2.jpg", "size": 5},
            ],
        },
    )
    scan = contract.scan_siim_inventory(inventory, public, expected=expected)
    assert scan["status"] == "failed_closed"
    assert scan["completed_files"] == 1
    assert scan["completed_bytes"] == 3
    assert scan["missing_files"] == 1
    assert scan["wrong_size_files"] == 1
    assert scan["part_files"] == 1
