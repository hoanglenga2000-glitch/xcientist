from __future__ import annotations

import base64
from typing import Any

from .capabilities import ApprovalRequiredError, CapabilityError, CapabilityNotFoundError
from .models import ToolResult
from .tool_package_runtime import ToolPackageApprovalRequired, ToolPackageStateError


def _runtime(context: Any):
    runtime = getattr(context, "super_agent", None)
    if runtime is None or getattr(runtime, "mode", None) is None:
        raise RuntimeError("super agent runtime is unavailable")
    return runtime


def _authorize(context: Any, *directory_ids: str) -> None:
    runtime = _runtime(context)
    workspace_root = getattr(context, "workspace_root", runtime.workspace_root)
    metadata = getattr(context, "metadata", {})
    for directory_id in dict.fromkeys(str(value) for value in directory_ids):
        runtime.authorize_directory(
            directory_id,
            session_id=context.session_id,
            workspace_root=workspace_root,
            metadata=metadata,
        )


def _result(operation: str, invoke) -> ToolResult:
    try:
        receipt = invoke()
    except ApprovalRequiredError as exc:
        return ToolResult("", False, {}, "exact approval is required", error=f"ApprovalRequiredError: {exc}")
    except CapabilityError as exc:
        return ToolResult("", False, {}, f"{operation} was rejected by the capability boundary", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return ToolResult("", False, {}, f"{operation} failed", error=f"{type(exc).__name__}: {exc}")
    payload = receipt.to_dict()
    return ToolResult("", bool(receipt.ok), {"receipt": payload}, f"{operation} completed", error=receipt.error_class)


def _exact_gate(*, reason: str, required_action: str, resume_point: str, error: str) -> ToolResult:
    return ToolResult(
        "",
        False,
        {
            "exact_gate": {
                "schema": "evomind.exact_gate.v1",
                "reason": reason,
                "required_action": required_action,
                "resume_point": resume_point,
            }
        },
        "one exact external Gate is required",
        error=error,
    )


def capability_discover(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    objective = str(args.get("objective") or "")
    operations = tuple(str(item) for item in (args.get("required_operations") or []))
    rows = runtime.discover(
        objective,
        required_operations=operations,
        healthy_only=bool(args.get("healthy_only", True)),
        run_id=context.session_id,
        workspace_root=getattr(context, "workspace_root", runtime.workspace_root),
        metadata=getattr(context, "metadata", {}),
        limit=max(1, min(int(args.get("limit", 20)), 100)),
    )
    return ToolResult(
        "",
        True,
        {
            "schema": "evomind.capability_discovery.v1",
            "matches": rows,
            "count": len(rows),
            "mode": runtime.mode.value,
        },
        f"discovered {len(rows)} capability match(es)",
    )


def super_agent_status(_args: dict[str, Any], context: Any) -> ToolResult:
    return ToolResult("", True, _runtime(context).status(), "super agent runtime status")


def ecosystem_refresh(_args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    result = runtime.refresh_installed_skills()
    return ToolResult(
        "",
        True,
        {"schema": "evomind.ecosystem_refresh.v1", **result, "capability_count": len(runtime.catalog.all())},
        f"refreshed {result['scanned_files']} bounded skill manifest(s)",
    )


def connector_health(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "connector_health",
        lambda: runtime.broker.connector_health(str(args["directory_id"])),
    )


def directory_list(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "directory_list",
        lambda: runtime.broker.list(
            str(args["directory_id"]),
            str(args.get("relative_path") or ""),
            recursive=bool(args.get("recursive", False)),
            limit=max(1, min(int(args.get("limit", 5_000)), 100_000)),
        ),
    )


def directory_stat(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "directory_stat",
        lambda: runtime.broker.stat(str(args["directory_id"]), str(args.get("relative_path") or "")),
    )


def directory_read(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    maximum = max(1, min(int(args.get("max_bytes", 131_072)), 2 * 1024 * 1024))
    try:
        receipt = runtime.broker.read(
            str(args["directory_id"]),
            str(args["relative_path"]),
            max_bytes=maximum,
        )
    except Exception as exc:
        return ToolResult("", False, {}, "directory_read failed", error=f"{type(exc).__name__}: {exc}")
    data = receipt.data or b""
    content: dict[str, Any] = {"receipt": receipt.to_dict(), "binary": b"\x00" in data[:4096]}
    if content["binary"]:
        content["base64_preview"] = base64.b64encode(data[:4096]).decode("ascii")
    else:
        content["text"] = data.decode(str(args.get("encoding") or "utf-8"), errors="replace")
    return ToolResult("", receipt.ok, content, "directory_read completed", error=receipt.error_class)


def directory_hash(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "directory_hash",
        lambda: runtime.broker.hash(str(args["directory_id"]), str(args["relative_path"])),
    )


def directory_mkdir(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "directory_mkdir",
        lambda: runtime.broker.mkdir(
            str(args["directory_id"]),
            str(args["relative_path"]),
            parents=bool(args.get("parents", True)),
            exist_ok=bool(args.get("exist_ok", True)),
        ),
    )


def directory_write_atomic(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    text_present = isinstance(args.get("content"), str)
    encoded_present = isinstance(args.get("base64_content"), str)
    if text_present == encoded_present:
        return ToolResult("", False, {}, "provide exactly one content encoding", error="content_encoding_invalid")
    try:
        data = (
            str(args["content"]).encode(str(args.get("encoding") or "utf-8"))
            if text_present
            else base64.b64decode(str(args["base64_content"]), validate=True)
        )
    except (UnicodeError, ValueError) as exc:
        return ToolResult("", False, {}, "content decoding failed", error=type(exc).__name__)
    if len(data) > 2 * 1024 * 1024:
        return ToolResult("", False, {"bytes": len(data)}, "model write exceeds the bounded size", error="write_size_exceeded")
    return _result(
        "directory_write_atomic",
        lambda: runtime.broker.write_atomic(
            str(args["directory_id"]),
            str(args["relative_path"]),
            data,
            create_parents=bool(args.get("create_parents", True)),
            expected_sha256=str(args.get("expected_sha256") or ""),
        ),
    )


def directory_copy(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(
        context,
        str(args["source_directory_id"]),
        str(args["destination_directory_id"]),
    )
    return _result(
        "directory_copy",
        lambda: runtime.broker.copy(
            str(args["source_directory_id"]),
            str(args["source_relative_path"]),
            str(args["destination_directory_id"]),
            str(args["destination_relative_path"]),
        ),
    )


def directory_sync(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(
        context,
        str(args["source_directory_id"]),
        str(args["destination_directory_id"]),
    )
    delete_extraneous = bool(args.get("delete_extraneous", False))
    return _result(
        "directory_sync",
        lambda: runtime.broker.sync(
            str(args["source_directory_id"]),
            str(args.get("source_relative_path") or ""),
            str(args["destination_directory_id"]),
            str(args.get("destination_relative_path") or ""),
            delete_extraneous=delete_extraneous,
            approved_delete=bool(getattr(context, "approval_verified", False)),
        ),
    )


def directory_delete(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    _authorize(context, str(args["directory_id"]))
    return _result(
        "directory_delete",
        lambda: runtime.broker.delete(
            str(args["directory_id"]),
            str(args["relative_path"]),
            recursive=bool(args.get("recursive", False)),
            approved=bool(getattr(context, "approval_verified", False)),
        ),
    )


def tool_synthesize(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    manifest = args.get("manifest")
    if not isinstance(manifest, dict):
        return ToolResult("", False, {}, "tool manifest must be an object", error="manifest_invalid")
    try:
        snapshot = runtime.tool_packages(context.session_id).stage(manifest)
        package = runtime.tool_packages(context.session_id).get(str(snapshot["package_id"]))
        if package:
            from .ecosystem import ToolPackage

            runtime.catalog.stage_tool_package(ToolPackage.from_manifest(manifest, run_id=context.session_id))
    except Exception as exc:
        return ToolResult("", False, {}, "tool package staging failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", True, {"package": snapshot}, "run-scoped tool package staged")


def _evidence(context: Any, artifact_id: str, expected_sha256: str) -> dict[str, Any]:
    row = context.store.get_deliverable(artifact_id)
    if not row or row.get("run_id") != context.session_id:
        raise ValueError("verification evidence is not bound to this run")
    digest = str(row.get("sha256") or "").casefold()
    if digest != str(expected_sha256 or "").casefold():
        raise ValueError("verification evidence hash mismatch")
    return {"artifact_id": artifact_id, "evidence_sha256": digest}


def tool_test(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        evidence = _evidence(context, str(args["evidence_artifact_id"]), str(args["evidence_sha256"]))
        snapshot = runtime.tool_packages(context.session_id).record_tests(
            str(args["package_id"]), passed=bool(args["passed"]), evidence=evidence
        )
        catalog_package = runtime.catalog.tool_package(context.session_id, str(args["package_id"]))
        if catalog_package is not None:
            catalog_package.record_tests(passed=bool(args["passed"]), evidence=evidence)
    except Exception as exc:
        return ToolResult("", False, {}, "tool package test evidence was rejected", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", True, {"package": snapshot}, "tool package test evidence recorded")


def tool_canary(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        evidence = _evidence(context, str(args["evidence_artifact_id"]), str(args["evidence_sha256"]))
        snapshot = runtime.tool_packages(context.session_id).record_canary(
            str(args["package_id"]), passed=bool(args["passed"]), evidence=evidence
        )
        catalog_package = runtime.catalog.tool_package(context.session_id, str(args["package_id"]))
        if catalog_package is not None:
            catalog_package.record_canary(passed=bool(args["passed"]), evidence=evidence)
    except Exception as exc:
        return ToolResult("", False, {}, "tool package canary evidence was rejected", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", True, {"package": snapshot}, "tool package canary evidence recorded")


def tool_promote(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        snapshot = runtime.tool_packages(context.session_id).promote(str(args["package_id"]))
        package = runtime.tool_packages(context.session_id).get(str(args["package_id"]))
        if package:
            runtime.catalog.promote_tool_package(context.session_id, str(args["package_id"]))
    except ToolPackageApprovalRequired as exc:
        return ToolResult("", False, {"exact_gate": True}, "exact approval is required", error=f"ToolPackageApprovalRequired: {exc}")
    except ToolPackageStateError as exc:
        return ToolResult("", False, {}, "tool package promotion was rejected", error=f"ToolPackageStateError: {exc}")
    return ToolResult("", True, {"package": snapshot}, "tool package metadata promoted for this run")


def tool_reject(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    if not bool(getattr(context, "approval_verified", False)):
        return _exact_gate(
            reason="tool package rejection is terminal and irreversible",
            required_action="approve rejection of the exact run-scoped package",
            resume_point="tool_reject",
            error="tool_package_reject_approval_required",
        )
    try:
        snapshot = runtime.tool_packages(context.session_id).reject(str(args["package_id"]))
    except Exception as exc:
        return ToolResult("", False, {}, "tool package rejection failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", True, {"package": snapshot}, "tool package rejected")


def transfer_fetch(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        _authorize(context, str(args["directory_id"]))
        receipt = runtime.transfer_fetch(
            source_connector_id=str(args["source_connector_id"]),
            source_url=str(args["source_url"]),
            directory_id=str(args["directory_id"]),
            relative_path=str(args["relative_path"]),
            resume=bool(args.get("resume", True)),
            expected_sha256=str(args.get("expected_sha256") or ""),
        )
    except (KeyError, CapabilityNotFoundError):
        return _exact_gate(
            reason="the requested source connector is not configured",
            required_action="authorize and health-check the exact source connector",
            resume_point="transfer_fetch",
            error="source_connector_not_configured",
        )
    except Exception as exc:
        return ToolResult("", False, {}, "transfer_fetch failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", receipt.ok, {"receipt": receipt.to_dict()}, "transfer_fetch completed", error=receipt.error_class)


def job_execute(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        _authorize(context, str(args["directory_id"]))
        connector, capability = runtime.execution_connector(str(args["directory_id"]))
        receipt = connector.execute(capability, str(args.get("relative_path") or ""), dict(args.get("job_spec") or {}))
    except (KeyError, CapabilityNotFoundError):
        return _exact_gate(
            reason="the requested execution connector is not configured",
            required_action="authorize and bind one exact execution connector capability",
            resume_point="job_execute",
            error="execution_connector_not_configured",
        )
    except Exception as exc:
        return ToolResult("", False, {}, "job_execute failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", receipt.ok, {"receipt": receipt.to_dict()}, "job_execute completed", error=receipt.error_class)


def job_status(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        _authorize(context, str(args["directory_id"]))
        connector, capability = runtime.execution_connector(str(args["directory_id"]))
        receipt = connector.status(capability, str(args["job_ref"]))
    except (KeyError, CapabilityNotFoundError):
        return _exact_gate(
            reason="the requested execution connector is not configured",
            required_action="authorize and bind one exact execution connector capability",
            resume_point="job_status",
            error="execution_connector_not_configured",
        )
    except Exception as exc:
        return ToolResult("", False, {}, "job_status failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", receipt.ok, {"receipt": receipt.to_dict()}, "job_status completed", error=receipt.error_class)


def job_cancel(args: dict[str, Any], context: Any) -> ToolResult:
    runtime = _runtime(context)
    try:
        _authorize(context, str(args["directory_id"]))
        connector, capability = runtime.execution_connector(str(args["directory_id"]))
        receipt = connector.cancel(
            capability,
            str(args["job_ref"]),
            approved=bool(getattr(context, "approval_verified", False)),
        )
    except (KeyError, CapabilityNotFoundError):
        return _exact_gate(
            reason="the requested execution connector is not configured",
            required_action="authorize and bind one exact execution connector capability",
            resume_point="job_cancel",
            error="execution_connector_not_configured",
        )
    except Exception as exc:
        return ToolResult("", False, {}, "job_cancel failed", error=f"{type(exc).__name__}: {exc}")
    return ToolResult("", receipt.ok, {"receipt": receipt.to_dict()}, "job_cancel completed", error=receipt.error_class)


__all__ = [
    "capability_discover",
    "connector_health",
    "directory_copy",
    "directory_delete",
    "directory_hash",
    "directory_list",
    "directory_mkdir",
    "directory_read",
    "directory_stat",
    "directory_sync",
    "directory_write_atomic",
    "ecosystem_refresh",
    "super_agent_status",
    "transfer_fetch",
    "job_cancel",
    "job_execute",
    "job_status",
    "tool_canary",
    "tool_promote",
    "tool_reject",
    "tool_synthesize",
    "tool_test",
]
