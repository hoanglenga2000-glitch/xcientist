from __future__ import annotations

from types import SimpleNamespace

from evomind_runtime import super_agent_tools
from evomind_runtime.super_agent_runtime import SuperAgentRuntime


def _context(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = SuperAgentRuntime(workspace_root=workspace, runtime_root=tmp_path / "runtime")
    return runtime, SimpleNamespace(super_agent=runtime, session_id="run_tools")


def test_generic_directory_tools_use_capability_ids_and_never_project_root(tmp_path):
    runtime, context = _context(tmp_path)
    discovered = super_agent_tools.capability_discover(
        {"objective": "write data", "required_operations": ["write"]}, context
    )
    assert discovered.ok is True
    assert discovered.content["matches"][0]["descriptor"]["capability_id"] == "workspace"
    written = super_agent_tools.directory_write_atomic(
        {"directory_id": "workspace", "relative_path": "data/a.txt", "content": "hello"}, context
    )
    assert written.ok is True
    assert "opaque_root_ref" not in str(written.content)
    read = super_agent_tools.directory_read(
        {"directory_id": "workspace", "relative_path": "data/a.txt"}, context
    )
    assert read.content["text"] == "hello"
    assert read.content["receipt"]["sha256"]
    runtime.close()


def test_generic_tools_reject_path_escape_and_require_bounded_content(tmp_path):
    runtime, context = _context(tmp_path)
    escaped = super_agent_tools.directory_read(
        {"directory_id": "workspace", "relative_path": "../outside.txt"}, context
    )
    assert escaped.ok is False
    invalid = super_agent_tools.directory_write_atomic(
        {
            "directory_id": "workspace",
            "relative_path": "bad.txt",
            "content": "one",
            "base64_content": "dHdv",
        },
        context,
    )
    assert invalid.error == "content_encoding_invalid"
    unapproved_delete = super_agent_tools.directory_delete(
        {"directory_id": "workspace", "relative_path": "bad.txt"}, context
    )
    assert unapproved_delete.ok is False
    assert "ApprovalRequiredError" in unapproved_delete.error
    runtime.close()
