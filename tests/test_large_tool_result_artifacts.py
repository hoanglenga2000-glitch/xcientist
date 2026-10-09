from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evomind_runtime import tools
from evomind_runtime.models import Session, ToolResult, ToolSpec
from evomind_runtime.store import RuntimeStore


FIXTURE_METADATA = json.loads(
    (Path(__file__).parent / "fixtures" / "ariel_large_tool_result_metadata.json").read_text(
        encoding="utf-8"
    )
)
ARIEL_RECEIPT = FIXTURE_METADATA["verified_receipt_projection"]


def _context(tmp_path: Path, store: RuntimeStore, session_id: str = "run_ariel_fixture") -> tools.ToolContext:
    if store.get_session(session_id) is None:
        store.create_session(Session(id=session_id, workspace_root=str(tmp_path)))
    return tools.ToolContext(
        session_id=session_id,
        workspace_root=tmp_path,
        project_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        artifact_root=tmp_path / "runtime" / "artifacts",
        store=store,
        metadata={},
    )


def _large_ariel_result() -> dict:
    smoke = {
        f"file_{index:05d}": {"ok": True, "detail": "verified-" + "x" * 128}
        for index in range(ARIEL_RECEIPT["loader_smoke_count"])
    }
    return {
        "schema": "evomind.competition_data_tool_result.v1",
        "action": "status",
        "competition": "ariel_2025",
        "status": "FULL_DATA_READY",
        "receipt": {
            "schema": "evomind.competition_data_receipt.v1",
            "competition": "ariel_2025",
            "status": "FULL_DATA_READY",
            "files": ARIEL_RECEIPT["files"],
            "bytes": ARIEL_RECEIPT["bytes"],
            "manifest_sha256": ARIEL_RECEIPT["manifest_sha256"],
            "archive_sha256": ARIEL_RECEIPT["archive_sha256"],
            "worker_alive": False,
            "worker_exit_code": 0,
            "failure_code": "",
            "loader_smoke": smoke,
            "secret_values_logged": False,
        },
        "hpc_identity_gate": "passed_5_of_5",
        "local_fallback_used": False,
        "secret_values_logged": False,
        "hpc_identity": {
            "credential_profile": "must-not-appear-in-preview",
            "host_uuid": "must-not-appear-in-preview",
            "remote_root": "must-not-appear-in-preview",
        },
    }


def test_large_ariel_result_uses_valid_projection_and_sha_bound_generic_artifact(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    context = _context(tmp_path, store)
    registry = tools.ToolRegistry()
    full_result = _large_ariel_result()
    full_bytes = json.dumps(full_result, ensure_ascii=False, default=str).encode("utf-8")
    assert len(full_bytes) > 2_000_000
    assert FIXTURE_METADATA["observed_tool_result_artifact"]["bytes"] > 2_000_000
    assert len(FIXTURE_METADATA["observed_tool_result_artifact"]["sha256"]) == 64
    registry.register(
        ToolSpec(
            name="large_ariel_fixture",
            description="fixture",
            input_schema={"type": "object", "additionalProperties": False},
            capability="artifact.read",
            max_result_bytes=64_000,
        ),
        lambda _args, _context: ToolResult("", True, full_result, "fixture"),
    )

    result = registry.invoke("large_ariel_fixture", {}, context)

    assert result.ok is True
    assert result.content["truncated"] is True
    assert result.content["preview_complete"] is False
    assert result.content["preview_encoding"] == "structured_json_projection"
    assert result.content["content_bytes"] == len(full_bytes)
    assert result.content["content_sha256"] == hashlib.sha256(full_bytes).hexdigest()
    assert "path" not in result.content["artifact"]
    projection = json.loads(result.content["preview"])
    assert projection["competition"] == "ariel_2025"
    assert projection["status"] == "FULL_DATA_READY"
    assert projection["receipt"]["files"] == ARIEL_RECEIPT["files"]
    assert projection["receipt"]["loader_smoke_count"] == ARIEL_RECEIPT["loader_smoke_count"]
    assert projection["receipt"]["loader_smoke_all_ok"] is True
    assert projection["receipt"]["file_sha_manifest_bound"] is True
    assert "must-not-appear-in-preview" not in result.content["preview"]

    artifact_id = result.content["artifact"]["id"]
    stored = store.get_artifact(artifact_id)
    assert stored is not None
    assert Path(stored["path"]).read_bytes() == full_bytes
    preview = tools._artifact_preview({"artifact_id": artifact_id}, context)
    assert preview.ok is True
    assert preview.content["artifact_hash_verified"] is True
    assert preview.content["json_valid"] is True
    assert preview.content["source_content_sha256"] == hashlib.sha256(full_bytes).hexdigest()
    rehydrated_projection = json.loads(preview.content["preview"])
    assert rehydrated_projection["receipt"]["loader_smoke_count"] == ARIEL_RECEIPT["loader_smoke_count"]
    assert rehydrated_projection["receipt"]["loader_smoke_all_ok"] is True
    store.close()


def test_generic_tool_result_artifact_is_run_bound_and_tamper_evident(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    owner = _context(tmp_path, store)
    other = _context(tmp_path, store, session_id="run_other")
    source = json.dumps(_large_ariel_result(), separators=(",", ":")).encode("utf-8")
    artifact = store.add_artifact(
        owner.session_id,
        source,
        ".json",
        "application/json",
        owner.artifact_root,
    )

    cross_run = tools._artifact_preview({"artifact_id": artifact["id"]}, other)
    assert cross_run.ok is False
    assert cross_run.error == "artifact_not_found"

    Path(artifact["path"]).write_bytes(source + b"\n")
    tampered = tools._artifact_preview({"artifact_id": artifact["id"]}, owner)
    assert tampered.ok is False
    assert tampered.error == "artifact_integrity"
    store.close()


def test_exactly_approved_artifact_import_is_same_tenant_sha_bound_and_non_destructive(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    task_root = tmp_path / "assistant_tasks"
    source_id = "run_" + "a" * 32
    target_id = "run_" + "b" * 32
    foreign_id = "run_" + "c" * 32
    tenant = {"managed_hpc_identity": {"tenant_id": "tenant_fixture"}}
    for run_id, metadata in ((source_id, tenant), (target_id, tenant), (foreign_id, {"managed_hpc_identity": {"tenant_id": "tenant_other"}})):
        root = task_root / run_id
        (root / "outputs").mkdir(parents=True)
        store.create_session(Session(id=run_id, workspace_root=str(root), metadata=metadata))

    source_path = task_root / source_id / "outputs" / "split.json"
    source_bytes = b'{"holdout_sha256":"' + b"1" * 64 + b'"}'
    source_path.write_bytes(source_bytes)
    source_artifact_id = "artifact_" + "d" * 32
    store.publish_deliverable({
        "id": source_artifact_id,
        "run_id": source_id,
        "session_id": source_id,
        "name": "split.json",
        "path": str(source_path),
        "media_type": "application/json",
        "bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_tool_call": "fixture",
        "preview_kind": "structured_text",
        "verified_at": "2026-09-04T00:00:00+00:00",
    })

    target = tools.ToolContext(
        session_id=target_id,
        workspace_root=task_root / target_id,
        project_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        artifact_root=tmp_path / "runtime" / "artifacts",
        store=store,
        metadata=tenant,
        approval_verified=True,
    )
    arguments = {"source_run_id": source_id, "artifact_ids": [source_artifact_id]}
    imported = tools._artifact_import(arguments, target)

    assert imported.ok is True
    assert imported.content["same_tenant_verified"] is True
    assert imported.content["exact_approval_verified"] is True
    assert imported.content["count"] == 1
    assert imported.content["source_manifest"][0]["source_artifact_id"] == source_artifact_id
    imported_row = store.get_deliverable(imported.content["artifacts"][0]["id"])
    assert imported_row is not None and imported_row["run_id"] == target_id
    assert Path(imported_row["path"]).read_bytes() == source_bytes
    assert source_path.read_bytes() == source_bytes

    target.approval_verified = False
    denied = tools._artifact_import(arguments, target)
    assert denied.ok is False and denied.error == "exact_approval_required"
    target.approval_verified = True
    foreign_arguments = {"source_run_id": foreign_id, "artifact_ids": [source_artifact_id]}
    foreign = tools._artifact_import(foreign_arguments, target)
    assert foreign.ok is False and foreign.error == "artifact_import_tenant_mismatch"
    store.close()


def test_ariel_fixture_does_not_change_weather_failure_evidence() -> None:
    weather = FIXTURE_METADATA["weather4cast_unchanged"]

    assert weather == {
        "status": "DOWNLOAD_FAILED",
        "worker_alive": False,
        "worker_exit_code": 124,
        "failure_code": "SFTP_TIMEOUT",
    }
