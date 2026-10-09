from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime.competition_data import persistent_root
from evomind_runtime.models import ToolResult
from evomind_runtime.policy import PolicyEngine, argument_fingerprint
from evomind_runtime.tools import (
    _competition_data_progress,
    _competition_data_progress_remote_source,
    _progress_child_pids,
    build_default_registry,
)


def _managed_identity() -> dict[str, object]:
    tenant = "tenant_aaaaaaaaaaaaaaaaaaaaaaaa"
    return {
        "tenant_id": tenant,
        "owner_principal_id": "admin",
        "job_id": 92257,
        "credential_profile": f"{tenant}_job92257_g21",
        "allocation_generation": 21,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job92257-fixture",
    }


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="run_fixture",
        metadata={"managed_hpc_identity": _managed_identity()},
    )


def _gate() -> dict[str, object]:
    return {
        "designated_proxy_path_verified": True,
        "pinned_gateway_host_key_verified": True,
        "allocation_role_authenticated": True,
        "expected_host_uuid_match": True,
        "expected_gpu_uuid_match": True,
        "expected_gpu_model_and_memory_match": True,
        "allowed_remote_root_match": True,
        "job_container_verified": True,
        "identity_consistent": True,
        "samples_requested": 5,
        "samples_passed": 5,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _remote_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "evomind.competition_data_progress.remote.v1",
        "sampled_at_utc": "2026-08-27T00:00:00+00:00",
        "regular_files": 17,
        "nonempty_files": 16,
        "total_bytes": 123456,
        "newest_mtime_utc": "2026-08-27T00:00:00+00:00",
        "partial_count": 1,
        "partial_bytes": 4096,
        "largest_open_data_file_bytes": 8192,
        "worker_alive": True,
        "worker_elapsed_seconds": 123.5,
        "worker_status": "RUNNING",
        "worker_phase": "downloading",
        "failure_code": "",
        "worker_exit_code": None,
        "worker_binding_valid": True,
        "process_tree_count": 3,
        "transfer_backend": "lftp",
        "adapter_transfer_mode": "lftp_parallel",
        "adapter_requested_parallelism": 8,
        "adapter_listing_status": "not_run",
        "adapter_listing_entries": 0,
        "adapter_listing_files": 0,
        "adapter_listing_directories": 0,
        "adapter_listing_archive_files": 0,
        "adapter_listing_scientific_files": 0,
        "adapter_listing_metadata_files": 0,
        "adapter_listing_other_files": 0,
        "adapter_listing_max_depth": 0,
        "adapter_listing_raw_lines": 0,
        "adapter_listing_control_lines": 0,
        "adapter_listing_rejected_lines": 0,
        "adapter_listing_skipped_directories": 0,
        "adapter_listing_timeout_exhaustions": 0,
        "adapter_parallel_failures": 0,
        "adapter_retry_rounds": 0,
        "adapter_pending_batches": 0,
        "adapter_fallback_reason": "",
        "adapter_sftp_buffer_bytes": 0,
        "adapter_sftp_num_requests": 0,
        "adapter_tuning_requested": False,
        "adapter_tuning_argv_present": False,
        "transport_process_count": 2,
        "tcp_socket_count": 4,
        "established_socket_count": 4,
        "tcp_send_queue_bytes": 0,
        "tcp_receive_queue_bytes": 0,
        "cpu_ticks": 91,
        "cpu_seconds": 0.91,
        "read_bytes": 2048,
        "write_bytes": 4096,
        "io_readable": True,
        "open_data_files": 2,
        "metadata_last_activity_utc": "2026-08-27T00:00:00+00:00",
        "log_available": False,
        "read_only": True,
        "remote_writes": 0,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    payload.update(overrides)
    return payload


def test_progress_collects_children_created_by_nonleader_linux_threads(tmp_path: Path) -> None:
    process_root = tmp_path / "4242" / "task"
    (process_root / "4242").mkdir(parents=True)
    (process_root / "4243").mkdir()
    (process_root / "not-a-task").mkdir()
    (process_root / "4242" / "children").write_text("", encoding="utf-8")
    (process_root / "4243" / "children").write_text("5001 5002 5001\n", encoding="utf-8")
    (process_root / "not-a-task" / "children").write_text("9999\n", encoding="utf-8")

    assert _progress_child_pids(tmp_path, 4242) == [5001, 5002]
    assert _progress_child_pids(tmp_path, 9999) == []


class _Channel:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def recv_exit_status(self) -> int:
        return self.exit_code


class _Stream(io.BytesIO):
    def __init__(self, value: bytes, *, exit_code: int = 0) -> None:
        super().__init__(value)
        self.channel = _Channel(exit_code)


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.commands: list[tuple[str, int]] = []
        self.closed = False

    def exec_command(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        encoded = (json.dumps(self.payload, separators=(",", ":")) + "\n").encode()
        return io.BytesIO(), _Stream(encoded), _Stream(b"")

    def close(self) -> None:
        self.closed = True


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object] | None = None,
) -> _FakeClient:
    client = _FakeClient(payload or _remote_payload())
    monkeypatch.setattr(
        "evomind_runtime.tools._hpc_verify",
        lambda _args, _context: ToolResult("", True, _gate(), "verified"),
    )
    monkeypatch.setattr("evomind_runtime.tools._load_bound_hpc_config", lambda *_args: object())
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.connect_ssh",
        lambda _config, timeout: client,
    )
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.verify_job_container_identity",
        lambda _client, _config, *, expected_job_id: {
            **{key: value for key, value in _gate().items() if key != "identity_consistent"},
            "job_id": expected_job_id,
        },
    )
    return client


def test_progress_tool_schema_accepts_only_fixed_competition_enum() -> None:
    spec = build_default_registry().get("competition_data_progress")
    assert spec is not None
    assert spec.read_only is True
    assert spec.input_schema["additionalProperties"] is False
    assert spec.input_schema["required"] == ["competition"]
    properties = spec.input_schema["properties"]
    assert set(properties) == {"competition"}
    assert properties["competition"]["enum"] == [
        "cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer",
    ]
    serialized = json.dumps(spec.input_schema, sort_keys=True).casefold()
    for forbidden in ("job_id", "credential_profile", "path", "command", "password", "token", "secret"):
        assert forbidden not in serialized

    decision = PolicyEngine().evaluate(
        tool_name="competition_data_progress",
        arguments={"competition": "weather4cast"},
        permission_level="observe",
        workspace_root=".",
    )
    assert decision.allowed is True
    assert decision.requires_approval is False
    assert decision.reason == "read-only capability"


def test_acceleration_tool_is_weather_only_and_requires_exact_approval(tmp_path) -> None:
    registry = build_default_registry()
    spec = registry.get("competition_data_accelerate")
    assert spec is not None
    assert spec.read_only is False
    assert set(spec.input_schema["properties"]) == {"competition", "timeout_seconds"}
    assert spec.input_schema["properties"]["competition"]["enum"] == ["weather4cast"]
    arguments = {"competition": "weather4cast", "timeout_seconds": 1800}
    pending = PolicyEngine().evaluate(
        tool_name="competition_data_accelerate",
        arguments=arguments,
        permission_level="full-auto",
        workspace_root=tmp_path,
    )
    assert pending.allowed is False
    assert pending.requires_approval is True
    assert pending.reason == "exact approval required"
    approved = PolicyEngine().evaluate(
        tool_name="competition_data_accelerate",
        arguments=arguments,
        permission_level="workspace-write",
        workspace_root=tmp_path,
        approved_fingerprint=argument_fingerprint("competition_data_accelerate", arguments),
    )
    assert approved.allowed is True
    assert approved.requires_approval is False


def test_progress_source_has_no_remote_write_or_signal_primitive() -> None:
    source = _competition_data_progress_remote_source("weather4cast")
    compile(source, "competition-progress-remote.py", "exec")
    forbidden_patterns = (
        r"\b(?:mkdir|unlink|remove|rename|replace|chmod|chown|kill|pkill)\s*\(",
        r"\.write_(?:text|bytes)\s*\(",
        r"\bos\.open\s*\(",
        r"\bopen\s*\([^\n]+,[^\n]+[\"'](?:w|a|x|\+)",
        r"\bsubprocess\b",
        r"\bsignal\b",
        r"open_sftp\s*\(",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, source, re.IGNORECASE) is None, pattern
    assert "os.scandir" in source
    assert 'entry.name != ".evomind"' in source
    assert "/proc/" in source
    assert "__EVOMIND_PROGRESS_CHILD_PIDS__" not in source
    assert '_progress_child_pids(pathlib.Path("/proc"), current)' in source
    assert 'task_root = proc_root / str(int(pid)) / "task"' in source
    assert "task_root.iterdir()" in source
    assert "environ" not in source
    for field in (
        "adapter_listing_archive_files",
        "adapter_listing_scientific_files",
        "adapter_listing_metadata_files",
        "adapter_listing_other_files",
        "adapter_listing_max_depth",
        "adapter_listing_skipped_directories",
        "adapter_listing_timeout_exhaustions",
    ):
        assert f'"{field}": adapter_count("{field}")' in source


def test_progress_blocks_before_any_remote_command_when_five_of_five_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _gate()
    evidence["expected_gpu_uuid_match"] = False
    monkeypatch.setattr(
        "evomind_runtime.tools._hpc_verify",
        lambda _args, _context: ToolResult("", True, evidence, "incomplete"),
    )
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.connect_ssh",
        lambda *_args, **_kwargs: pytest.fail("remote connection occurred before the complete gate"),
    )

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is False
    assert result.error == "hpc_identity_evidence_incomplete"
    assert result.content["remote_commands_attempted"] == 0
    assert result.content["remote_writes"] == 0
    assert result.content["signals_sent"] == 0


def test_progress_uses_fixed_root_and_parses_fake_ssh_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_success_fakes(monkeypatch)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["competition"] == "weather4cast"
    assert result.content["hpc_identity_gate"] == "passed_5_of_5"
    assert result.content["same_connection_identity_verified"] is True
    assert result.content["regular_files"] == 17
    assert result.content["total_bytes"] == 123456
    assert result.content["write_bytes"] == 4096
    assert result.content["transfer_backend"] == "lftp"
    assert result.content["adapter_transfer_mode"] == "lftp_parallel"
    assert result.content["adapter_requested_parallelism"] == 8
    assert result.content["adapter_listing_status"] == "not_run"
    assert result.content["adapter_listing_archive_files"] == 0
    assert result.content["adapter_listing_scientific_files"] == 0
    assert result.content["adapter_listing_metadata_files"] == 0
    assert result.content["adapter_listing_other_files"] == 0
    assert result.content["adapter_listing_max_depth"] == 0
    assert result.content["adapter_listing_skipped_directories"] == 0
    assert result.content["adapter_listing_timeout_exhaustions"] == 0
    assert result.content["adapter_fallback_reason"] == ""
    assert result.content["adapter_tuning_requested"] is False
    assert result.content["adapter_tuning_argv_present"] is False
    assert result.content["established_socket_count"] == 4
    assert result.content["read_only"] is True
    assert result.content["remote_writes"] == 0
    assert result.content["signals_sent"] == 0
    assert result.content["other_processes_modified"] is False
    assert result.content["remote_commands_attempted"] == 1
    assert len(client.commands) == 1
    command, timeout = client.commands[0]
    assert timeout < 55
    assert persistent_root("weather4cast") in command
    assert client.closed is True


def test_progress_accepts_r114_openssh_recursive_get_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(
        transfer_backend="openssh_sftp",
        adapter_transfer_mode="openssh_recursive_get",
        adapter_requested_parallelism=1,
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["transfer_backend"] == "openssh_sftp"
    assert result.content["adapter_transfer_mode"] == "openssh_recursive_get"
    assert result.content["remote_writes"] == 0
    assert result.content["signals_sent"] == 0
    assert result.content["other_processes_modified"] is False


def test_progress_fails_closed_when_run_binding_drifts_after_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()

    def mutate_binding(_args, _context):
        context.metadata["managed_hpc_identity"]["allocation_binding_id"] = "changed-binding"
        return ToolResult("", True, _gate(), "verified")

    monkeypatch.setattr("evomind_runtime.tools._hpc_verify", mutate_binding)
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.connect_ssh",
        lambda *_args, **_kwargs: pytest.fail("connection occurred after binding drift"),
    )

    result = _competition_data_progress({"competition": "weather4cast"}, context)

    assert result.ok is False
    assert result.error == "progress_probe:ValueError"
    assert result.content["remote_commands_attempted"] == 0


def test_progress_blocks_aggregation_when_same_connection_identity_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(_remote_payload())
    monkeypatch.setattr(
        "evomind_runtime.tools._hpc_verify",
        lambda _args, _context: ToolResult("", True, _gate(), "verified"),
    )
    monkeypatch.setattr("evomind_runtime.tools._load_bound_hpc_config", lambda *_args: object())
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.connect_ssh",
        lambda _config, timeout: client,
    )
    incomplete = _gate()
    incomplete["allowed_remote_root_match"] = False
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.verify_job_container_identity",
        lambda *_args, **_kwargs: incomplete,
    )

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is False
    assert result.error == "progress_probe:ValueError"
    assert result.content["remote_commands_attempted"] == 0
    assert client.commands == []
    assert client.closed is True


@pytest.mark.parametrize(
    "payload",
    [
        _remote_payload(
            worker_alive=False,
            worker_elapsed_seconds=900.0,
            worker_status="FULL_DATA_READY",
            worker_phase="complete",
            worker_exit_code=0,
            process_tree_count=0,
            cpu_ticks=0,
            cpu_seconds=0.0,
            read_bytes=0,
            write_bytes=0,
            io_readable=False,
            open_data_files=0,
            largest_open_data_file_bytes=0,
        ),
        _remote_payload(
            worker_alive=False,
            worker_elapsed_seconds=None,
            worker_status="",
            worker_phase="",
            worker_binding_valid=False,
            process_tree_count=0,
            cpu_ticks=0,
            cpu_seconds=0.0,
            read_bytes=0,
            write_bytes=0,
            io_readable=False,
            open_data_files=0,
            largest_open_data_file_bytes=0,
        ),
    ],
)
def test_progress_accepts_completed_or_missing_worker_observations(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    _install_success_fakes(monkeypatch, payload)
    result = _competition_data_progress({"competition": "weather4cast"}, _context())
    assert result.ok is True
    assert result.content["worker_alive"] is False
    assert result.content["signals_sent"] == 0
    assert result.content["remote_writes"] == 0


def test_progress_rejects_unexpected_remote_fields_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload()
    payload["path"] = "/forbidden/internal/path"
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is False
    assert result.error == "progress_probe:ValueError"
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "/forbidden/internal/path" not in serialized
    assert "credential_profile" not in serialized
    assert "job92257" not in serialized


@pytest.mark.parametrize(
    "overrides",
    [
        {"adapter_transfer_mode": "untrusted"},
        {"adapter_listing_status": "untrusted"},
        {"adapter_fallback_reason": "untrusted"},
        {"adapter_requested_parallelism": 17},
        {"adapter_listing_entries": 2, "adapter_listing_files": 1, "adapter_listing_directories": 0},
        {"adapter_listing_archive_files": 10_001},
        {"adapter_listing_scientific_files": 10_001},
        {"adapter_listing_metadata_files": 10_001},
        {"adapter_listing_other_files": 10_001},
        {"adapter_listing_max_depth": 10_001},
        {"adapter_listing_skipped_directories": 10_001},
        {"adapter_listing_skipped_directories": True},
        {"adapter_listing_timeout_exhaustions": 10_001},
        {"adapter_listing_timeout_exhaustions": True},
        {"adapter_listing_timeout_exhaustions": 1},
        {
            "adapter_listing_entries": 1,
            "adapter_listing_directories": 1,
            "adapter_listing_skipped_directories": 2,
        },
        {"adapter_listing_files": 1, "adapter_listing_archive_files": 2},
        {"adapter_listing_raw_lines": 1, "adapter_listing_control_lines": 1, "adapter_listing_rejected_lines": 1},
        {"adapter_parallel_failures": 17},
        {"adapter_retry_rounds": 5},
        {"adapter_pending_batches": 17},
        {"adapter_sftp_buffer_bytes": 1},
        {"adapter_sftp_num_requests": 1},
        {"adapter_tuning_requested": True},
        {"regular_files": True},
        {"cpu_seconds": float("nan")},
        {"cpu_seconds": float("inf")},
        {"worker_elapsed_seconds": True},
        {"worker_exit_code": False},
    ],
)
def test_progress_rejects_invalid_adapter_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    _install_success_fakes(monkeypatch, _remote_payload(**overrides))
    result = _competition_data_progress({"competition": "weather4cast"}, _context())
    assert result.ok is False
    assert result.error == "progress_probe:ValueError"


def test_progress_accepts_bounded_deep_listing_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _remote_payload(
        adapter_listing_entries=5_000,
        adapter_listing_files=108,
        adapter_listing_directories=4_892,
        adapter_listing_archive_files=50,
        adapter_listing_scientific_files=40,
        adapter_listing_metadata_files=10,
        adapter_listing_other_files=8,
        adapter_listing_max_depth=16,
        adapter_listing_skipped_directories=7,
        adapter_listing_timeout_exhaustions=7,
        adapter_listing_status="ok_with_skips",
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["adapter_listing_entries"] == 5_000
    assert result.content["adapter_listing_archive_files"] == 50
    assert result.content["adapter_listing_scientific_files"] == 40
    assert result.content["adapter_listing_metadata_files"] == 10
    assert result.content["adapter_listing_other_files"] == 8
    assert result.content["adapter_listing_max_depth"] == 16
    assert result.content["adapter_listing_skipped_directories"] == 7
    assert result.content["adapter_listing_timeout_exhaustions"] == 7
    assert result.content["adapter_listing_status"] == "ok_with_skips"


def test_progress_reports_incomplete_listing_as_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(
        worker_alive=False,
        worker_status="DOWNLOAD_FAILED",
        worker_phase="failed",
        failure_code="WEATHER_LISTING_INCOMPLETE",
        worker_exit_code=18,
        adapter_listing_status="ok_with_skips",
        adapter_listing_entries=5,
        adapter_listing_directories=5,
        adapter_listing_skipped_directories=1,
        adapter_listing_timeout_exhaustions=1,
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["failure_code"] == "WEATHER_LISTING_INCOMPLETE"
    assert result.content["adapter_listing_skipped_directories"] == 1
    assert result.content["adapter_listing_timeout_exhaustions"] == 1
    serialized = json.dumps(result.to_dict(), sort_keys=True).casefold()
    assert "current_directory" not in serialized
    assert "skipped_directory_paths" not in serialized


def test_progress_accepts_pre_profile_legacy_listing_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(
        adapter_listing_entries=108,
        adapter_listing_files=108,
        adapter_listing_directories=0,
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["adapter_listing_archive_files"] == 0
    assert result.content["adapter_listing_scientific_files"] == 0
    assert result.content["adapter_listing_metadata_files"] == 0
    assert result.content["adapter_listing_other_files"] == 0


@pytest.mark.parametrize(
    "reason",
    [
        "duplicate_directory",
        "depth",
        "invalid_relative",
        "duplicate_file",
        "entry_cap",
    ],
)
def test_progress_accepts_safe_listing_structure_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    payload = _remote_payload(
        adapter_listing_status="parse_rejected",
        adapter_listing_rejected_lines=1,
        adapter_listing_raw_lines=1,
        adapter_fallback_reason=f"listing_structure_{reason}",
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["adapter_fallback_reason"] == f"listing_structure_{reason}"


def test_progress_rejects_listing_telemetry_above_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _remote_payload(
        adapter_listing_entries=10_001,
        adapter_listing_files=108,
        adapter_listing_directories=9_893,
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is False
    assert result.error == "progress_probe:ValueError"
    assert result.content["remote_writes"] == 0
    assert result.content["signals_sent"] == 0


@pytest.mark.parametrize(
    "reject_reason",
    [
        "total_like_format",
        "field_count",
        "symlink_type",
        "special_type",
        "unknown_type",
        "path",
        "leading_dash",
        "colon",
        "bracket",
        "percent",
        "ampersand",
        "hash",
        "other_leaf_charset",
        "leaf_length",
        "apostrophe",
        "double_quote",
        "semicolon",
        "exclamation",
        "tilde",
        "dollar",
        "backtick",
        "brace",
        "asterisk",
        "question",
        "control_character",
        "unicode_character",
        "other_ascii",
        "duplicate",
    ],
)
def test_progress_accepts_safe_listing_reject_reason_enums(
    monkeypatch: pytest.MonkeyPatch,
    reject_reason: str,
) -> None:
    payload = _remote_payload(
        worker_alive=False,
        worker_status="DOWNLOAD_FAILED",
        worker_phase="failed",
        failure_code="SFTP_LISTING_REJECTED",
        worker_exit_code=15,
        adapter_listing_status="parse_rejected",
        adapter_listing_raw_lines=1,
        adapter_listing_rejected_lines=1,
        adapter_fallback_reason=f"listing_parse_rejected_{reject_reason}",
    )
    _install_success_fakes(monkeypatch, payload)

    result = _competition_data_progress({"competition": "weather4cast"}, _context())

    assert result.ok is True
    assert result.content["adapter_fallback_reason"] == (
        f"listing_parse_rejected_{reject_reason}"
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "/" not in result.content["adapter_fallback_reason"]
    assert "private" not in serialized.casefold()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_status", "RUNNING /internal/private/path"),
        ("worker_phase", "password=fake-sensitive-marker"),
    ],
)
def test_progress_rejects_untrusted_status_text_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    _install_success_fakes(monkeypatch, _remote_payload(**{field: value}))
    result = _competition_data_progress({"competition": "weather4cast"}, _context())
    assert result.ok is False
    assert result.error == "progress_probe:ValueError"
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert value not in serialized
    assert "fake-sensitive-marker" not in serialized
