from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.release_finalize import (
    BUILD_RECEIPT_SCHEMA,
    build_receipt,
    build_runtime_manifest,
    normalize_generated_json,
    validate_wheelhouse,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wheelhouse_fixture(tmp_path: Path) -> dict[str, Path | str]:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    dependency_name = "demo_dep-1.2.3-py3-none-any.whl"
    dependency_bytes = b"locked dependency wheel"
    dependency_hash = _sha256(dependency_bytes)
    (wheelhouse / dependency_name).write_bytes(dependency_bytes)
    (wheelhouse / "xcientist-0.3.0-py3-none-any.whl").write_bytes(b"first party wheel")

    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text(
        "\n".join(
            [
                "version = 1",
                "revision = 3",
                "requires-python = \">=3.12\"",
                "",
                "[[package]]",
                'name = "demo-dep"',
                'version = "1.2.3"',
                "wheels = [",
                "    { url = \"https://files.example.invalid/"
                f"{dependency_name}\", hash = \"sha256:{dependency_hash}\", size = {len(dependency_bytes)} }},",
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(f"demo-dep==1.2.3 --hash=sha256:{dependency_hash}\n", encoding="utf-8")
    report = tmp_path / "pip-report.json"
    _write_json(
        report,
        {
            "version": "1",
            "install": [
                {
                    "download_info": {
                        "url": f"file:///C:/wheelhouse/{dependency_name}",
                        "archive_info": {"hashes": {"sha256": dependency_hash}},
                    },
                    "metadata": {"name": "demo-dep", "version": "1.2.3"},
                }
            ],
        },
    )
    return {
        "wheelhouse": wheelhouse,
        "uv_lock": uv_lock,
        "requirements": requirements,
        "report": report,
        "dependency_name": dependency_name,
    }


def test_normalize_generated_json_is_deterministic(tmp_path: Path) -> None:
    route_payload = {"/z/route": "/z", "/a/route": "/a"}
    _write_json(tmp_path / "app/.next/app-path-routes-manifest.json", route_payload)
    _write_json(tmp_path / "app/.next/server/app-paths-manifest.json", route_payload)
    _write_json(
        tmp_path / "metadata/node-sbom.cdx.json",
        {
            "metadata": {"timestamp": "2026-07-28T15:00:31.790Z"},
            "serialNumber": "urn:uuid:random",
            "bomFormat": "CycloneDX",
        },
    )
    commit = "47b8ee06e629c55b57916e96ce07e8dda57ccbed"
    epoch = 1_783_745_275

    normalize_generated_json(tmp_path, commit=commit, source_date_epoch=epoch)
    first = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }
    normalize_generated_json(tmp_path, commit=commit, source_date_epoch=epoch)
    second = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }

    assert first == second
    assert first["app/.next/app-path-routes-manifest.json"].index(b'"/a/route"') < first[
        "app/.next/app-path-routes-manifest.json"
    ].index(b'"/z/route"')
    sbom = json.loads(first["metadata/node-sbom.cdx.json"])
    assert sbom["serialNumber"] == f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, commit)}"
    assert sbom["metadata"]["timestamp"] == datetime.fromtimestamp(
        epoch, timezone.utc
    ).isoformat().replace("+00:00", "Z")


def test_normalize_generated_json_merges_reproducible_builder_properties(tmp_path: Path) -> None:
    route_payload = {"/route": "/route"}
    _write_json(tmp_path / "app/.next/app-path-routes-manifest.json", route_payload)
    _write_json(tmp_path / "app/.next/server/app-paths-manifest.json", route_payload)
    _write_json(
        tmp_path / "metadata/node-sbom.cdx.json",
        {"metadata": {"properties": [{"name": "kept", "value": "yes"}]}, "bomFormat": "CycloneDX"},
    )
    receipt = {
        "source_digest": "a" * 64,
        "build_id": "next-build-id",
        "git_commit": "b" * 40,
        "release_contract_sha256": "c" * 64,
        "source_date_epoch": 1_783_745_275,
        "toolchain_contract_sha256": "d" * 64,
        "wheelhouse_manifest_sha256": "e" * 64,
    }
    _write_json(tmp_path / "metadata/build-receipt.json", receipt)

    normalize_generated_json(
        tmp_path,
        commit=receipt["git_commit"],
        source_date_epoch=receipt["source_date_epoch"],
        receipt=receipt,
    )

    sbom = json.loads((tmp_path / "metadata/node-sbom.cdx.json").read_text(encoding="utf-8"))
    properties = {item["name"]: item["value"] for item in sbom["metadata"]["properties"]}
    assert properties["kept"] == "yes"
    assert properties["evomind:source-digest"] == receipt["source_digest"]
    assert properties["evomind:build-id"] == receipt["build_id"]
    assert properties["evomind:build-receipt-sha256"] == _sha256(
        (tmp_path / "metadata/build-receipt.json").read_bytes()
    )
    serial_seed = f"evomind:{receipt['git_commit']}:{receipt['source_digest']}"
    assert sbom["serialNumber"] == f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, serial_seed)}"


def test_validate_wheelhouse_binds_exact_set_to_uv_lock(tmp_path: Path) -> None:
    fixture = _wheelhouse_fixture(tmp_path)
    manifest = validate_wheelhouse(
        fixture["wheelhouse"],
        uv_lock=fixture["uv_lock"],
        locked_requirements=fixture["requirements"],
        pip_report=fixture["report"],
        project_name="xcientist",
        project_version="0.3.0",
    )

    assert manifest["wheel_count"] == 2
    assert {wheel["origin"] for wheel in manifest["wheels"]} == {
        "first-party-source-build",
        "uv.lock",
    }
    assert manifest["uv_lock_sha256"] == _sha256(fixture["uv_lock"].read_bytes())


def test_validate_wheelhouse_rejects_extra_wheel(tmp_path: Path) -> None:
    fixture = _wheelhouse_fixture(tmp_path)
    (fixture["wheelhouse"] / "unlocked-9.9.9-py3-none-any.whl").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="exact-set mismatch"):
        validate_wheelhouse(
            fixture["wheelhouse"],
            uv_lock=fixture["uv_lock"],
            locked_requirements=fixture["requirements"],
            pip_report=fixture["report"],
            project_name="xcientist",
            project_version="0.3.0",
        )


def test_validate_wheelhouse_rejects_missing_wheel(tmp_path: Path) -> None:
    fixture = _wheelhouse_fixture(tmp_path)
    (fixture["wheelhouse"] / fixture["dependency_name"]).unlink()

    with pytest.raises(RuntimeError, match="exact-set mismatch"):
        validate_wheelhouse(
            fixture["wheelhouse"],
            uv_lock=fixture["uv_lock"],
            locked_requirements=fixture["requirements"],
            pip_report=fixture["report"],
            project_name="xcientist",
            project_version="0.3.0",
        )


def test_validate_wheelhouse_rejects_uv_lock_hash_drift(tmp_path: Path) -> None:
    fixture = _wheelhouse_fixture(tmp_path)
    lock_text = fixture["uv_lock"].read_text(encoding="utf-8")
    fixture["uv_lock"].write_text(lock_text.replace(_sha256(b"locked dependency wheel"), "f" * 64), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not bound to uv.lock"):
        validate_wheelhouse(
            fixture["wheelhouse"],
            uv_lock=fixture["uv_lock"],
            locked_requirements=fixture["requirements"],
            pip_report=fixture["report"],
            project_name="xcientist",
            project_version="0.3.0",
        )


def test_build_receipt_keeps_publication_time_outside_reproducible_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "wheelhouse-manifest.json"
    _write_json(manifest_path, {"schema": "test", "wheels": []})
    metadata = {"schema": "builder", "source_digest": "a" * 64}

    receipt = build_receipt(metadata, manifest_path)

    assert receipt["schema"] == BUILD_RECEIPT_SCHEMA
    assert "published_utc" not in receipt
    assert receipt["wheelhouse_manifest_sha256"] == _sha256(manifest_path.read_bytes())


def test_runtime_manifest_is_deterministic_and_bound_to_staged_migrations(tmp_path: Path) -> None:
    migration = tmp_path / "app/prisma/migrations/20260810000000_baseline/migration.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text(
        'CREATE TABLE "tasks" ("id" TEXT PRIMARY KEY NOT NULL);\n',
        encoding="utf-8",
    )
    receipt = {
        "git_commit": "b" * 40,
        "source_dirty": False,
        "source_digest": "a" * 64,
        "build_id": "next-build-id",
    }
    epoch = 1_786_297_200

    first = build_runtime_manifest(
        tmp_path,
        receipt=receipt,
        version="0.3.0",
        source_date_epoch=epoch,
    )
    second = build_runtime_manifest(
        tmp_path,
        receipt=receipt,
        version="0.3.0",
        source_date_epoch=epoch,
    )

    assert first == second
    assert first == {
        "schema": "evomind.runtime_build.v1",
        "commit_hash": "b" * 40,
        "source_dirty": False,
        "source_tree_sha256": "a" * 64,
        "build_id": "next-build-id",
        "build_time": datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend_version": "0.3.0",
        "frontend_version": "0.3.0",
        "database_schema_version": "20260810000000_baseline",
        "database_schema_sha256": first["database_schema_sha256"],
    }
    assert len(first["database_schema_sha256"]) == 64
    assert set(first["database_schema_sha256"]) <= set("0123456789abcdef")
    assert not list((tmp_path / "app/prisma").glob("workstation.db*"))
