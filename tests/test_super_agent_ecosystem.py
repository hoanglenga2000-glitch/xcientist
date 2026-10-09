from __future__ import annotations

import json
from pathlib import Path

import pytest

from evomind_runtime.ecosystem import (
    CapabilityCatalog,
    ExternalExecutionDisabled,
    ManifestError,
    SkillScanPolicy,
    ToolPackage,
    parse_cli_manifest,
    parse_mcp_manifest,
    parse_openapi_manifest,
    scan_skill_manifests,
)


def test_catalog_discovers_from_declared_metadata_without_domain_keyword_table() -> None:
    manifest = {
        "serverInfo": {"name": "storage-node", "version": "2.1.0"},
        "health_evidence": {"status": "healthy", "checked_by": "contract"},
        "tools": [
            {
                "name": "mirror_objects",
                "description": "Synchronize verified objects into an authorized mount",
                "inputSchema": {
                    "type": "object",
                    "properties": {"mount_id": {"type": "string"}},
                    "required": ["mount_id"],
                },
                "annotations": {"readOnlyHint": False, "idempotentHint": True},
                "tags": ["同步", "授权目录"],
            },
            {
                "name": "inspect_queue",
                "description": "Read pending work without modification",
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": True},
            },
        ],
    }
    descriptors = parse_mcp_manifest(manifest)
    catalog = CapabilityCatalog(descriptors)

    match = catalog.discover("把对象同步到授权目录", required_operations=["mirror_objects"])

    assert [item.operations[0] for item in match] == ["mirror_objects"]
    assert match[0].provider_id == "storage-node"
    assert match[0].version == "2.1.0"
    assert match[0].risk_class == "write"
    assert match[0].idempotency == "idempotent"
    assert match[0].healthy is True
    assert len(match[0].tool_package_sha256) == 64


def test_mcp_parser_honors_protocol_risk_annotations_and_schema() -> None:
    tools = parse_mcp_manifest({
        "name": "files",
        "tools": [
            {
                "name": "purge",
                "description": "Remove a staged object",
                "inputSchema": {
                    "type": "object",
                    "properties": {"object_id": {"type": "string"}},
                    "required": ["object_id"],
                },
                "annotations": {"destructiveHint": True, "idempotentHint": True},
            }
        ],
    })

    assert tools[0].risk_class == "destructive"
    assert tools[0].input_schema["required"] == ["object_id"]
    assert tools[0].source_kind == "mcp"


def test_openapi_parser_builds_typed_operations_without_contacting_server() -> None:
    spec = {
        "openapi": "3.1.0",
        "info": {"title": "Archive Service", "version": "4.0.0"},
        "paths": {
            "/archives/{archive_id}": {
                "parameters": [
                    {"name": "archive_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "get": {
                    "operationId": "readArchive",
                    "summary": "Read archive metadata",
                    "tags": ["archive"],
                },
                "delete": {
                    "operationId": "deleteArchive",
                    "summary": "Delete an archive",
                },
            },
            "/archives": {
                "post": {
                    "operationId": "createArchive",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}},
                    },
                }
            },
        },
    }

    descriptors = {item.operations[0]: item for item in parse_openapi_manifest(spec)}

    assert descriptors["readarchive"].risk_class == "observe"
    assert descriptors["readarchive"].idempotency == "idempotent"
    assert descriptors["readarchive"].input_schema["required"] == ["archive_id"]
    assert descriptors["deletearchive"].risk_class == "destructive"
    assert descriptors["createarchive"].risk_class == "write"
    assert descriptors["createarchive"].input_schema["required"] == ["body"]


def test_cli_bridge_only_parses_declared_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("manifest parsing must never spawn a process")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    descriptors = parse_cli_manifest({
        "name": "bounded-cli",
        "version": "1.2.3",
        "health_evidence": {"status": "ready"},
        "commands": [
            {
                "name": "hash-file",
                "description": "Hash a selected file",
                "risk_class": "observe",
                "idempotency": "idempotent",
                "arguments": [{"name": "relative_path", "type": "string", "required": True}],
            }
        ],
    })

    assert descriptors[0].input_schema["required"] == ["relative_path"]
    assert descriptors[0].source_kind == "cli"
    assert descriptors[0].healthy is True


def test_skill_scan_is_bounded_skips_symlink_and_redacts_front_matter(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    good = root / "safe-skill"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\n"
        "name: safe-reader\n"
        "description: 'Inspect metadata; token=do-not-leak'\n"
        "version: 1.0.0\n"
        "operations: [inspect_metadata]\n"
        "---\n"
        "Untrusted body instructions must never become executable metadata.\n",
        encoding="utf-8",
    )
    oversized = root / "oversized"
    oversized.mkdir()
    (oversized / "SKILL.md").write_text("---\nname: huge\n---\n" + "x" * 1_000, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: escaped\n---\n", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        link = None

    report = scan_skill_manifests(
        root,
        policy=SkillScanPolicy(max_files=8, max_file_bytes=512, max_total_bytes=1_024, max_depth=3),
    )

    assert report.scanned_files == 1
    assert report.descriptors[0].operations == ("inspect_metadata",)
    assert "do-not-leak" not in report.descriptors[0].description
    assert "<redacted>" in report.descriptors[0].description
    assert report.descriptors[0].source_ref == "safe-skill/SKILL.md"
    assert report.skipped_files >= 1
    if link is not None:
        assert all("linked" not in item.source_ref for item in report.descriptors)


def _package_manifest(*, run_id: str = "run-a") -> dict[str, object]:
    return {
        "schema": "evomind.tool_package.v1",
        "package_id": "mount-auditor",
        "run_id": run_id,
        "version": "1.0.0",
        "provider_id": "run-sandbox",
        "capabilities": [
            {
                "name": "audit_mount",
                "description": "Audit an authorized mount",
                "operations": ["audit_mount"],
                "input_schema": {
                    "type": "object",
                    "properties": {"directory_id": {"type": "string"}},
                    "required": ["directory_id"],
                },
                "risk_class": "observe",
                "idempotency": "idempotent",
            }
        ],
        "permissions": {"directories": ["declared-only"]},
        "tests": [{"id": "contract", "expected": "pass"}],
        "canary": {"scope": "isolated-fixture"},
        "rollback": {"action": "discard-run-package"},
    }


def test_tool_package_is_run_scoped_and_requires_tests_and_canary_before_promotion() -> None:
    package = ToolPackage.from_manifest(_package_manifest(), run_id="run-a")
    catalog = CapabilityCatalog()
    catalog.stage_tool_package(package)

    with pytest.raises(ManifestError, match="tests"):
        catalog.promote_tool_package("run-a", "mount-auditor")
    assert catalog.discover("audit authorized mount", run_id="run-a") == []

    package.record_tests(passed=True, evidence={"status": "passed", "suite_sha256": "a" * 64})
    package.record_canary(passed=True, evidence={"status": "passed", "receipt_sha256": "b" * 64})
    promoted = catalog.promote_tool_package("run-a", "mount-auditor")

    assert promoted.promotion_status == "promoted"
    assert catalog.discover("audit authorized mount") == []
    assert catalog.discover("audit authorized mount", run_id="run-a")[0].tool_package_sha256 == package.manifest_sha256
    assert catalog.discover("audit authorized mount", run_id="run-b") == []
    assert catalog.tool_package("run-b", "mount-auditor") is None
    with pytest.raises(ExternalExecutionDisabled):
        package.install()
    with pytest.raises(ExternalExecutionDisabled):
        package.execute()


def test_tool_package_rejects_cross_run_and_secret_bearing_manifest() -> None:
    with pytest.raises(ManifestError, match="different run"):
        ToolPackage.from_manifest(_package_manifest(run_id="run-a"), run_id="run-b")

    manifest = _package_manifest()
    manifest["api_key"] = "plaintext-value"
    with pytest.raises(ManifestError, match="secret material"):
        ToolPackage.from_manifest(manifest)


def test_json_manifest_size_and_shape_are_bounded() -> None:
    with pytest.raises(ManifestError, match="root"):
        parse_mcp_manifest(json.dumps([{"name": "not-an-object"}]))
    with pytest.raises(ManifestError, match="size"):
        parse_cli_manifest(b"{" + b" " * 1_048_576 + b"}")
