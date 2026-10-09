from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest


def harness():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_invitation_server_candidate.py"
    spec = importlib.util.spec_from_file_location("invitation_server_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venv_child_binding_accepts_verified_launcher_lineage():
    module = harness()
    binding = {"chain": [{"pid": 222, "parent_pid": 111, "image": "python.exe", "command_matches": True},
                         {"pid": 111, "parent_pid": 99, "image": "python.exe", "command_matches": True}],
               "listener_pids": [222]}
    assert module.binding_valid(binding, 222, 111, "backend")
    assert not module.binding_valid(binding, 222, 333, "backend")
    binding["listener_pids"] = [333]
    assert not module.binding_valid(binding, 222, 111, "backend")


def test_venv_binding_rejects_wrong_command_and_image():
    module = harness()
    binding = {"chain": [{"pid": 222, "image": "python.exe", "command_matches": False}], "listener_pids": [222]}
    assert not module.binding_valid(binding, 222, 222, "backend")
    binding["chain"][0].update(command_matches=True, image="node.exe")
    assert not module.binding_valid(binding, 222, 222, "backend")


def test_sse_keepalive_is_not_a_persisted_event():
    result = harness().read_sse_frames(io.BytesIO(b": keepalive 123\n\n"))
    assert result == {"frames": [], "keepalive_seen": True}


def test_sse_parses_sequenced_terminal_events():
    result = harness().read_sse_frames(io.BytesIO(b'id: 42\nevent: run_completed\ndata: {"status":"completed"}\n\n'))
    assert result["frames"] == [{"id": 42, "event": "run_completed", "data": {"status": "completed"}}]


def test_sse_unsequenced_error_is_not_run_history():
    result = harness().read_sse_frames(io.BytesIO(b'event: run_failed\ndata: {"status":"failed"}\n\n'))
    assert result["frames"][0]["id"] is None


def test_sse_incomplete_frame_is_rejected():
    module = harness()
    with pytest.raises(module.AcceptanceError, match="sse_truncated_frame"):
        module.read_sse_frames(io.BytesIO(b'id: 4\ndata: {"status":"running"}\n'))
