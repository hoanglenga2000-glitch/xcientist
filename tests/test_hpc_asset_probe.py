from __future__ import annotations

import hashlib
import inspect
import io
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evomind_runtime.models import ToolResult
from evomind_runtime.policy import PolicyEngine
from evomind_runtime.tools import _hpc_asset_probe, _hpc_asset_probe_remote_source, build_default_registry
from research_agent_workstation.server.core.gpu_credentials import ALLOWED_GPU_REMOTE_ROOT


def _identity() -> dict[str, object]:
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
    return SimpleNamespace(session_id="run_fixture", metadata={"managed_hpc_identity": _identity()})


def _gate(*, passed: int = 5) -> dict[str, object]:
    identity = _identity()
    return {
        "ok": True,
        "status": "job_container_verified",
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
        "samples_passed": passed,
        "samples": [{"sample_index": index + 1, "complete": index < passed} for index in range(5)],
        "job_id": identity["job_id"],
        "credential_profile": identity["credential_profile"],
        "host_uuid": "host-uuid-fixture",
        "gpu_uuids": ["GPU-fixture"],
        "gpu_name": "NVIDIA A800-SXM4-80GB",
        "gpu_memory_total_mib": 81920,
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "read_only": True,
        "signals_sent": 0,
        "remote_writes": 0,
        "other_processes_modified": False,
        "training_started": False,
        "worker_control": 0,
    }


def _payload(**overrides: object) -> dict[str, object]:
    files = [
        {"path": "config.json", "bytes": 2, "mtime_ns": 10, "regular_file": True, "content_read": False}
    ]
    closure = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value: dict[str, object] = {
        "schema": "evomind.g21_hpc_asset_probe.v1",
        "scope": "cure_mindgames",
        "packages": [
            {"name": name, "available": name in {"textarena", "trueskill"}, "version": "0.7.4" if name == "textarena" else "0.4.5" if name == "trueskill" else None, "origin": [f"competition_data/.runtime/mindgames/site-packages/{name}"] if name in {"textarena", "trueskill"} else []}
            for name in ("textarena", "trueskill", "transformers", "torch")
        ],
        "models": [
            {
                "path": "competition_data/.runtime/mindgames/models/Qwen3-8B",
                "file_count": 1,
                "total_bytes": 2,
                "metadata_closure_sha256": closure,
                "metadata_closure_complete": True,
                "content_sha256_computed": False,
                "file_details_included": False,
            }
        ],
        "skipped_optional_paths": 0,
        "manifest_candidates": [
            {"path": "competition_data/.runtime/mindgames/formal-protocol.json", "bytes": 100, "mtime_ns": 11, "regular_file": True, "content_read": False, "role": "formal"}
        ],
        "allowed_remote_root_match": True,
        "read_only": True,
        "network_access": False,
        "file_contents_read": False,
        "test_labels_used": False,
        "secret_values_logged": False,
        "signals_sent": 0,
        "remote_writes": 0,
        "other_processes_modified": False,
        "training_started": False,
        "worker_control": 0,
    }
    value.update(overrides)
    return value


class _Channel:
    def __init__(self, code: int = 0) -> None:
        self.code = code

    def recv_exit_status(self) -> int:
        return self.code


class _Stream(io.BytesIO):
    def __init__(self, value: bytes, code: int = 0) -> None:
        super().__init__(value)
        self.channel = _Channel(code)


class _Client:
    def __init__(self, payload: object, *, error: Exception | None = None, stderr: bytes = b"") -> None:
        self.payload = payload
        self.error = error
        self.stderr = stderr
        self.commands: list[tuple[str, int]] = []
        self.closed = False

    def exec_command(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        if self.error:
            raise self.error
        raw = self.payload if isinstance(self.payload, bytes) else (json.dumps(self.payload, separators=(",", ":")) + "\n").encode()
        return io.BytesIO(), _Stream(raw), _Stream(self.stderr)

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, payload: object | None = None, *, client_error: Exception | None = None, gate: dict[str, object] | None = None, root: str = ALLOWED_GPU_REMOTE_ROOT, stderr: bytes = b"") -> _Client:
    client = _Client(payload or _payload(), error=client_error, stderr=stderr)
    monkeypatch.setattr("evomind_runtime.tools._hpc_verify", lambda _args, _context: ToolResult("", True, gate or _gate(), "verified"))
    identity = _identity()
    monkeypatch.setattr(
        "evomind_runtime.tools._load_bound_hpc_config",
        lambda *_args: SimpleNamespace(
            remote_workspace=root,
            expected_host_uuid="host-uuid-fixture",
            expected_gpu_uuid="GPU-fixture",
            job_id=identity["job_id"],
            credential_profile=identity["credential_profile"],
        ),
    )
    monkeypatch.setattr("research_agent_workstation.server.core.gpu_credentials.connect_ssh", lambda _config, timeout: client)
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.verify_job_container_identity",
        lambda _client, _config, *, expected_job_id: {**_gate(), "job_id": expected_job_id},
    )
    return client


def test_tool_schema_is_read_only_and_cannot_accept_command_path_job_or_profile(tmp_path) -> None:
    registry = build_default_registry()
    spec = registry.get("hpc_asset_probe")
    assert spec is not None and spec.read_only is True
    assert spec.input_schema["additionalProperties"] is False
    assert spec.input_schema["required"] == []
    assert spec.input_schema["properties"] == {"scope": {"type": "string", "enum": ["cure_mindgames"]}}
    serialized = json.dumps(spec.input_schema, sort_keys=True)
    for forbidden in ("command", "path", "job_id", "credential_profile", "remote_root"):
        assert forbidden not in serialized
    for bad in ({"command": "id"}, {"path": "/tmp"}, {"job_id": 1}, {"scope": "other"}):
        result = registry.invoke("hpc_asset_probe", bad, SimpleNamespace())
        assert result.ok is False
        assert result.summary == "invalid tool arguments"
    decision = PolicyEngine().evaluate(
        tool_name="hpc_asset_probe",
        arguments={},
        permission_level="observe",
        workspace_root=tmp_path,
    )
    assert decision.allowed is True and decision.requires_approval is False


def test_probe_runs_one_fixed_same_connection_command_after_complete_5_of_5(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch)
    result = _hpc_asset_probe({}, _context())
    assert result.ok is True
    assert result.content["hpc_identity_gate"] == "passed_5_of_5"
    assert result.content["same_connection_identity_verified"] is True
    assert result.content["remote_commands_attempted"] == 1
    assert result.content["remote_writes"] == 0
    assert result.content["signals_sent"] == 0
    assert result.content["other_processes_modified"] is False
    assert result.content["training_started"] is False
    assert result.content["worker_control"] == 0
    assert result.content["binding_unchanged"] is True
    assert result.content["stdout_bytes"] > 0
    assert result.content["stderr_present"] is False
    assert result.content["stderr_sha256"] == hashlib.sha256(b"").hexdigest()
    assert len(client.commands) == 1 and client.closed is True
    command, timeout = client.commands[0]
    parts = shlex.split(command)
    assert parts[:4] == ["python3", "-I", "-S", "-c"]
    assert parts[-1] == ALLOWED_GPU_REMOTE_ROOT
    assert timeout == 45
    assert "|" not in command and ">" not in command and "<" not in command


def test_4_of_5_and_retarget_arguments_fail_before_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    connected: list[bool] = []
    monkeypatch.setattr("evomind_runtime.tools._hpc_verify", lambda _args, _context: ToolResult("", True, _gate(passed=4), "incomplete"))
    monkeypatch.setattr("research_agent_workstation.server.core.gpu_credentials.connect_ssh", lambda *_a, **_k: connected.append(True))
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False and result.error == "hpc_identity_evidence_incomplete"
    assert result.content["remote_commands_attempted"] == 0 and connected == []
    result = _hpc_asset_probe({"job_id": 92257}, _context())
    assert result.ok is False and result.error == "asset_probe_arguments_invalid"
    assert connected == []


def test_count_only_identity_without_five_complete_samples_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _gate()
    evidence.pop("samples")
    client = _install(monkeypatch, gate=evidence)
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False and result.error == "hpc_identity_evidence_incomplete"
    assert result.content["remote_commands_attempted"] == 0
    assert client.commands == []


def test_binding_change_and_remote_root_escape_fail_before_remote_command(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context()
    client = _install(monkeypatch)

    def mutate(_args, _context):
        context.metadata["managed_hpc_identity"]["allocation_binding_id"] = "changed-binding-fixture"
        return ToolResult("", True, _gate(), "verified")

    monkeypatch.setattr("evomind_runtime.tools._hpc_verify", mutate)
    changed = _hpc_asset_probe({}, context)
    assert changed.ok is False and changed.content["remote_commands_attempted"] == 0
    assert client.commands == []


def test_same_connection_failure_reports_exact_stage_and_no_command(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch)
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.verify_job_container_identity",
        lambda *_args, **_kwargs: {**_gate(), "job_container_verified": False},
    )
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False
    assert result.content["binding_unchanged"] is True
    assert result.content["same_connection_identity_verified"] is False
    assert result.content["remote_commands_attempted"] == 0
    assert result.content["training_started"] is False
    assert result.content["worker_control"] == 0
    assert client.commands == [] and client.closed is True

    context = _context()
    client = _install(monkeypatch, root=ALLOWED_GPU_REMOTE_ROOT + "/../escape")
    escaped = _hpc_asset_probe({}, context)
    assert escaped.ok is False and escaped.content["remote_commands_attempted"] == 0
    assert client.commands == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", 1),
        ("credential_profile", "wrong-profile"),
        ("host_uuid", "wrong-host"),
        ("gpu_uuids", ["GPU-wrong"]),
        ("gpu_name", "wrong-gpu"),
        ("gpu_memory_total_mib", 80000),
        ("remote_root", "/wrong/root"),
    ],
)
def test_same_connection_identity_values_must_equal_preflight_and_config(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    client = _install(monkeypatch)
    monkeypatch.setattr(
        "research_agent_workstation.server.core.gpu_credentials.verify_job_container_identity",
        lambda *_args, **_kwargs: {**_gate(), field: value},
    )
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False
    assert result.content["binding_unchanged"] is True
    assert result.content["same_connection_identity_verified"] is False
    assert result.content["remote_commands_attempted"] == 0
    assert client.commands == [] and client.closed is True


@pytest.mark.parametrize(
    "payload",
    [
        {**_payload(), "unexpected": True},
        _payload(signals_sent=1),
        _payload(models=[{**_payload()["models"][0], "path": "/escape"}]),
        _payload(models=[{**_payload()["models"][0], "metadata_closure_sha256": "0" * 63}]),
        _payload(models=[{**_payload()["models"][0], "metadata_closure_complete": False}]),
        _payload(models=[{**_payload()["models"][0], "file_details_included": True}]),
    ],
)
def test_output_anomalies_fail_closed(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    client = _install(monkeypatch, payload)
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False
    assert result.error == "asset_probe:ValueError"
    assert result.content["remote_commands_attempted"] == 1
    assert client.closed is True


def test_timeout_is_reported_fail_closed_with_actual_command_count(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch, client_error=TimeoutError("fixture timeout"))
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False and result.error == "asset_probe:TimeoutError"
    assert result.content["remote_commands_attempted"] == 1
    assert result.content["remote_writes"] == 0
    assert len(client.commands) == 1 and client.closed is True


def test_nonempty_stderr_is_rejected_without_returning_stderr_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch, stderr=b"warning fixture")
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False and result.error == "asset_probe:ValueError"
    assert "warning fixture" not in json.dumps(result.content)
    assert result.content["remote_commands_attempted"] == 1
    assert result.content["failure_stage"] == "remote_output_validation"
    assert result.content["failure_code"] == "remote_output_invalid"
    assert result.content["stderr_present"] is True
    assert result.content["remote_exit_code"] == 0
    assert client.closed is True


def test_multiline_stdout_is_rejected_instead_of_accepting_last_json(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"unexpected-prefix\n" + (json.dumps(_payload(), separators=(",", ":")) + "\n").encode()
    client = _install(monkeypatch, raw)
    result = _hpc_asset_probe({}, _context())
    assert result.ok is False and result.error == "asset_probe:ValueError"
    assert result.content["remote_commands_attempted"] == 1
    assert result.content["file_contents_read"] is False
    assert result.content["secret_values_logged"] is False
    assert result.content["local_fallback_used"] is False
    assert client.closed is True


def test_probe_has_no_hpc_runtime_sftp_network_or_content_read_path() -> None:
    handler = inspect.getsource(_hpc_asset_probe)
    remote = _hpc_asset_probe_remote_source()
    for forbidden in ("HpcRuntime", "open_sftp", "sftp", "execute_solution", "cancel("):
        assert forbidden not in handler
    for forbidden in (
        "socket",
        "urllib",
        "requests",
        "subprocess",
        "open(",
        ".read(",
        "unlink",
        "remove(",
        "kill(",
        "importlib.metadata",
    ):
        assert forbidden not in remote


def test_remote_probe_uses_exact_persistent_competition_data_layout() -> None:
    remote = _hpc_asset_probe_remote_source()
    assert 'os.path.join(root, "competition_data", ".runtime", "mindgames")' in remote
    assert '"competition_data/.runtime/mindgames/models/Qwen3-8B"' in remote
    assert '"competition_data/cure_bench/managed_runtime/models/Qwen3-8B"' in remote
    assert '"competition_data/.runtime/mindgames/formal-protocol.json"' in remote
    assert '"competition_data/cure_bench/.evomind/cure-bench/frozen-encoder.json"' in remote
    assert '"competition_data/cure_bench/managed_runtime/cure-bench-primary-baseline.json"' in remote
    assert '"competition_data/cure_bench/.evomind/cure-bench/primary-baseline-manifest.json"' in remote
    assert '"competition_data/cure_bench/managed_runtime/primary-baseline.json"' not in remote
    assert "competition-data" not in remote
    assert 'os.path.join(root, ".runtime", "mindgames")' not in remote
    assert '\n            ".runtime/mindgames/' not in remote


def _remote_probe_root(tmp_path):
    root = tmp_path / "allowed"
    runtime = root / "competition_data/.runtime/mindgames"
    site = runtime / "lib/python3.12/site-packages"
    for name, version in (("textarena", "0.7.4"), ("trueskill", "0.4.5")):
        package = site / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# fixture\n", encoding="utf-8")
        (site / f"{name}-{version}.dist-info").mkdir()
    model = runtime / "models/Qwen3-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return root, model


def test_remote_probe_allows_tokenizer_filename_without_reading_content(tmp_path) -> None:
    root, _model = _remote_probe_root(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _hpc_asset_probe_remote_source(), str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    packages = {row["name"]: row for row in payload["packages"]}
    assert packages["textarena"]["version"] == "0.7.4"
    assert packages["trueskill"]["version"] == "0.4.5"
    qwen = next(row for row in payload["models"] if row["path"].endswith("Qwen3-8B"))
    metadata_rows = [
        {
            "path": path.relative_to(_model).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "regular_file": True,
            "content_read": False,
        }
        for path in sorted(_model.rglob("*"), key=lambda value: value.as_posix())
        if path.is_file()
    ]
    expected_closure = hashlib.sha256(
        json.dumps(metadata_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert qwen["file_count"] == 2
    assert qwen["total_bytes"] == sum(row["bytes"] for row in metadata_rows)
    assert qwen["metadata_closure_sha256"] == expected_closure
    assert qwen["metadata_closure_complete"] is True
    assert qwen["file_details_included"] is False
    assert "files" not in qwen


def test_remote_probe_large_model_tree_stays_inside_bounded_stdout(tmp_path) -> None:
    root, model = _remote_probe_root(tmp_path)
    for index in range(4096):
        (model / f"shard-{index:05d}.meta").write_bytes(b"x")
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _hpc_asset_probe_remote_source(), str(root)],
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert len(completed.stdout) < 262_144
    payload = json.loads(completed.stdout)
    qwen = next(row for row in payload["models"] if row["path"].endswith("Qwen3-8B"))
    assert qwen["file_count"] == 4098
    assert qwen["file_details_included"] is False


@pytest.mark.parametrize(
    "filename",
    ["secret.key", "secrets.json", "token.txt", "tokens.json", "cookie.txt", "cookies.json", "password.txt", "passwords.json", "credentials.json"],
)
def test_remote_probe_rejects_anchored_credential_filename(tmp_path, filename: str) -> None:
    root, model = _remote_probe_root(tmp_path)
    (model / filename).write_text("fixture\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _hpc_asset_probe_remote_source(), str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 25
    assert completed.stdout == ""
