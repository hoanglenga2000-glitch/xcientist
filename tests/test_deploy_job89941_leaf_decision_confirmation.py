from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_job89941_leaf_decision_confirmation.py"
PLAN = (
    ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_decision_logit_confirmation_job89941_frozen_plan_v3_20260727.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_job89941_leaf_confirmation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_python(command: str) -> str:
    marker = "base64.b64decode('"
    encoded = command.split(marker, 1)[1].split("')", 1)[0]
    return base64.b64decode(encoded).decode("utf-8")


def test_current_frozen_plan_is_complete_and_hash_verified() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)

    assert plan.payload["job_id"] == 89941
    assert plan.payload["resource_mode"] == "cpu_only"
    assert plan.payload["run_id"] == "job89941_leaf_decision_s434445_v3_20260727"
    assert plan.payload["remote_base"].endswith(
        "/job89941_leaf_decision_s434445_v3_20260727"
    )
    revision = plan.payload["operational_revision"]
    assert revision["predecessor_status"] == "failed_before_first_model_fit"
    assert revision["fold_assignment_implementation_changed"] is True
    assert revision["every_validation_fold_requires_all_99_classes"] is True
    assert revision["scoreability_gate_relaxed"] is False
    assert revision["predecessor_state_reused"] is False
    assert plan.payload["contract"] == module.runner.fixed_contract()
    assert len(plan.payload["sources"]) == 7
    assert len(plan.payload["inputs"]) == 6
    assert {value["relative_path"] for value in plan.payload["sources"]} == (
        module.EXPECTED_SOURCE_PATHS
    )
    assert {value["name"] for value in plan.payload["inputs"]} == (
        module.EXPECTED_INPUT_NAMES
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("job_id", 89942, "job_id"),
        ("resource_mode", "gpu", "resource_mode"),
        ("remote_root", "/tmp/outside", "remote_root"),
    ],
)
def test_plan_rejects_execution_contract_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    module = load_module()
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.LeafDeploymentError, match=message):
        module.validate_plan(path)


def test_plan_rejects_source_hash_drift(tmp_path: Path) -> None:
    module = load_module()
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["sources"][0]["sha256"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.LeafDeploymentError, match="hash drift"):
        module.validate_plan(path)


def test_argv_environment_and_stage_paths_are_cpu_only_and_confined() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    environment = module.runtime_environment(plan)
    runner_argv = module.runner_argv(plan)
    verifier_argv = module.verifier_argv(plan)
    root = PurePosixPath(module.hpc.ALLOWED_GPU_REMOTE_ROOT)

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "1"
    assert module.hpc.REMOTE_UNIFIED_SITE_PACKAGES in environment["PYTHONPATH"]
    assert "--reference-image-manifest" in runner_argv
    assert "--embedding-cache-dir" in runner_argv
    assert "--run-dir" in verifier_argv
    assert len(module.staged_records(plan)) == 14
    for record in module.staged_records(plan):
        PurePosixPath(record["remote_path"]).relative_to(root)


def test_readonly_preflight_hashes_public_files_without_writes() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    source = decode_python(module.render_readonly_preflight(plan))

    for record in plan.payload["public_files"]:
        assert record["sha256"] in source
    assert "resolve(strict=True)" in source
    assert "root_writable" in source
    assert ".mkdir(" not in source
    assert "write_text(" not in source


def test_staged_smoke_loads_override_manifest_and_both_caches() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    source = decode_python(module.render_staged_smoke(plan))

    assert "validate_reference_manifest" in source
    assert "override_path=" in source
    assert "resolve_cache_records" in source
    assert "expected_rows=len(train)+len(test)" in source


def test_launch_is_exclusive_detached_and_contains_no_signal_api() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    source = decode_python(module.render_launch(plan))
    lowered = source.lower()

    assert "os.o_excl" in lowered
    assert "start_new_session=true" in lowered
    assert "stdin=subprocess.devnull" in lowered
    assert "duplicate_processes_started':0" in lowered
    assert "os.kill" not in lowered
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "taskkill" not in lowered


def test_resumable_prefix_hash_matches_exact_local_prefix(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "cache.npy"
    path.write_bytes(b"abcdefghij")

    assert module._local_prefix_sha256(path, 6) == hashlib.sha256(b"abcdef").hexdigest()


def test_all_remote_python_payloads_compile() -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    commands = (
        module.render_readonly_preflight(plan),
        module.render_staged_smoke(plan),
        module.render_launch(plan),
        module.render_status(plan),
    )
    for index, command in enumerate(commands):
        compile(decode_python(command), f"<leaf-remote-{index}>", "exec")


class _FakeRemoteFile:
    def __init__(
        self,
        payload: bytes,
        *,
        fail_after_bytes: int | None = None,
        seek_offsets: list[int] | None = None,
    ) -> None:
        self.payload = payload
        self.fail_after_bytes = fail_after_bytes
        self.seek_offsets = seek_offsets if seek_offsets is not None else []
        self.position = 0
        self.failed = False

    def __enter__(self) -> _FakeRemoteFile:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def seek(self, offset: int) -> None:
        self.position = offset
        self.seek_offsets.append(offset)

    def read(self, size: int) -> bytes:
        if (
            self.fail_after_bytes is not None
            and not self.failed
            and self.position >= self.fail_after_bytes
        ):
            self.failed = True
            raise ConnectionResetError("fixture transport dropped")
        end = min(len(self.payload), self.position + size)
        if self.fail_after_bytes is not None and not self.failed:
            end = min(end, self.fail_after_bytes)
        block = self.payload[self.position : end]
        self.position = end
        return block


class _FakeSFTP:
    def __init__(
        self,
        payload: bytes,
        *,
        fail_after_bytes: int | None = None,
        seek_offsets: list[int] | None = None,
    ) -> None:
        self.payload = payload
        self.fail_after_bytes = fail_after_bytes
        self.seek_offsets = seek_offsets if seek_offsets is not None else []

    def __enter__(self) -> _FakeSFTP:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def lstat(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=len(self.payload))

    def open(self, _path: str, mode: str) -> _FakeRemoteFile:
        assert mode == "rb"
        return _FakeRemoteFile(
            self.payload,
            fail_after_bytes=self.fail_after_bytes,
            seek_offsets=self.seek_offsets,
        )


class _FakeClient:
    def __init__(self, sftp: _FakeSFTP) -> None:
        self.sftp = sftp
        self.closed = False

    def open_sftp(self) -> _FakeSFTP:
        return self.sftp

    def close(self) -> None:
        self.closed = True


def _collection_record(payload: bytes, relative_path: str = "artifact.bin") -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_collection_reuses_existing_exact_file_without_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    payload = b"verified immutable artifact"
    destination = tmp_path / "collected"
    destination.mkdir()
    (destination / "artifact.bin").write_bytes(payload)
    monkeypatch.setattr(
        module.hpc,
        "connect_job89941",
        lambda: pytest.fail("verified local reuse must not reconnect"),
    )

    result = module._collect_remote_file_resumable(
        plan=plan,
        record=_collection_record(payload),
        destination=destination,
    )

    assert result == {
        "relative_path": "artifact.bin",
        "status": "reused_verified",
        "attempts": 0,
        "resumed_from_bytes": len(payload),
    }


def test_collection_rejects_existing_drifted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    payload = b"expected artifact"
    destination = tmp_path / "collected"
    destination.mkdir()
    (destination / "artifact.bin").write_bytes(b"drifted artifact")
    monkeypatch.setattr(
        module.hpc,
        "connect_job89941",
        lambda: pytest.fail("drift must fail before reconnect"),
    )

    with pytest.raises(module.LeafDeploymentError, match="Existing collected.*drift"):
        module._collect_remote_file_resumable(
            plan=plan,
            record=_collection_record(payload),
            destination=destination,
        )


def test_collection_resumes_existing_partial_from_exact_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    payload = b"abcdefghij"
    destination = tmp_path / "collected"
    destination.mkdir()
    partial = destination / "artifact.bin.part"
    partial.write_bytes(payload[:4])
    seek_offsets: list[int] = []
    client = _FakeClient(_FakeSFTP(payload, seek_offsets=seek_offsets))
    monkeypatch.setattr(module.hpc, "connect_job89941", lambda: client)

    result = module._collect_remote_file_resumable(
        plan=plan,
        record=_collection_record(payload),
        destination=destination,
        chunk_bytes=64 * 1024,
    )

    assert result["status"] == "downloaded_verified"
    assert result["attempts"] == 1
    assert result["resumed_from_bytes"] == 4
    assert seek_offsets == [4]
    assert (destination / "artifact.bin").read_bytes() == payload
    assert not partial.exists()
    assert client.closed is True


def test_collection_reconnects_after_transport_drop_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    payload = b"0123456789"
    destination = tmp_path / "collected"
    seek_offsets: list[int] = []
    clients = [
        _FakeClient(
            _FakeSFTP(payload, fail_after_bytes=4, seek_offsets=seek_offsets)
        ),
        _FakeClient(_FakeSFTP(payload, seek_offsets=seek_offsets)),
    ]
    connections: list[_FakeClient] = []

    def connect() -> _FakeClient:
        client = clients[len(connections)]
        connections.append(client)
        return client

    monkeypatch.setattr(module.hpc, "connect_job89941", connect)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._collect_remote_file_resumable(
        plan=plan,
        record=_collection_record(payload),
        destination=destination,
        max_attempts=2,
        chunk_bytes=64 * 1024,
    )

    assert result["status"] == "downloaded_verified"
    assert result["attempts"] == 2
    assert result["resumed_from_bytes"] == 0
    assert seek_offsets == [0, 4]
    assert (destination / "artifact.bin").read_bytes() == payload
    assert all(client.closed for client in connections)


def test_collection_rejects_oversized_partial_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    payload = b"abc"
    destination = tmp_path / "collected"
    destination.mkdir()
    (destination / "artifact.bin.part").write_bytes(payload + b"x")
    monkeypatch.setattr(
        module.hpc,
        "connect_job89941",
        lambda: pytest.fail("oversized partial must fail before reconnect"),
    )

    with pytest.raises(module.LeafDeploymentError, match="exceeds frozen size"):
        module._collect_remote_file_resumable(
            plan=plan,
            record=_collection_record(payload),
            destination=destination,
        )


def test_collection_rejects_final_sha_mismatch_without_atomic_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    plan = module.validate_plan(PLAN)
    expected = b"expected"
    remote_payload = b"tampered"
    assert len(expected) == len(remote_payload)
    destination = tmp_path / "collected"
    client = _FakeClient(_FakeSFTP(remote_payload))
    monkeypatch.setattr(module.hpc, "connect_job89941", lambda: client)

    with pytest.raises(module.LeafDeploymentError, match="hash differs"):
        module._collect_remote_file_resumable(
            plan=plan,
            record=_collection_record(expected),
            destination=destination,
            chunk_bytes=64 * 1024,
        )

    assert not (destination / "artifact.bin").exists()
    assert (destination / "artifact.bin.part").read_bytes() == remote_payload
    assert client.closed is True
