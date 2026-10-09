from __future__ import annotations

import hashlib
from pathlib import Path

import evomind_runtime.tools as runtime_tools
import pytest
from evomind_runtime.hpc_runtime_overlay import HpcRuntime as OverlayHpcRuntime
from evomind_runtime.models import ToolResult
from evomind_runtime.runtime import AgentRuntime
from research_os.hpc_runtime import HpcJobResult
from research_os.hpc_runtime import HpcRuntime


class _Stream:
    def __init__(self, payload: bytes = b"", exit_code: int = 0) -> None:
        self.payload = payload
        self.exit_code = exit_code
        self.channel = self

    def read(self) -> bytes:
        return self.payload

    def recv_exit_status(self) -> int:
        return self.exit_code

    def recv_ready(self) -> bool:
        return bool(self.payload)

    def recv(self, size: int) -> bytes:
        payload, self.payload = self.payload[:size], self.payload[size:]
        return payload

    def recv_stderr_ready(self) -> bool:
        return False

    def exit_status_ready(self) -> bool:
        return True


class _Sftp:
    def close(self) -> None:
        pass


class _Client:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []
        self.closed = False

    def exec_command(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        return None, _Stream(b"training complete\n"), _Stream()

    def open_sftp(self) -> _Sftp:
        return _Sftp()

    def close(self) -> None:
        self.closed = True


def _dependency_command(command: str):
    import base64
    import json
    import re
    import shlex

    argv = shlex.split(command)
    assert argv[:2] == ["python3", "-c"]
    assert len(argv) == 5
    encoded = re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", argv[2])
    assert encoded is not None
    return argv[3], json.loads(argv[4]), base64.b64decode(encoded.group(1)).decode("utf-8")


class _DependencyClient(_Client):
    def exec_command(self, command: str, timeout: int):
        import json

        self.commands.append((command, timeout))
        root, pins, _source = _dependency_command(command)
        receipt = {
            "schema": "evomind.dependency_install.v1", "status": "ready",
            "target": root + "/site-packages", "requested_pins": pins,
            "hashes_before_install": True, "lock_sha256": "a" * 64, "reused": False,
        }
        return None, _Stream(json.dumps(receipt).encode()), _Stream()


class _RemoteWriter:
    def __init__(self, files: dict[str, bytes], path: str) -> None:
        self.files = files
        self.path = path
        self.payload = bytearray()

    def write(self, payload: bytes) -> None:
        self.payload.extend(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.files[self.path] = bytes(self.payload)


class _ManagedDataSftp:
    def __init__(self, receipt: dict[str, object]) -> None:
        self.receipt = receipt
        self.files: dict[str, bytes] = {}
        self.modes: dict[str, int] = {}
        self.removed: list[str] = []

    def stat(self, _path: str):
        class _Stat:
            st_mode = 0o040700
        return _Stat()

    def mkdir(self, _path: str) -> None:
        pass

    def file(self, path: str, _mode: str) -> _RemoteWriter:
        return _RemoteWriter(self.files, path)

    def chmod(self, path: str, mode: int) -> None:
        self.modes[path] = mode

    def get(self, remote_path: str, local_path: str) -> None:
        if not remote_path.endswith("receipt.json"):
            raise OSError("fixture missing")
        Path(local_path).write_text(__import__("json").dumps(self.receipt), encoding="utf-8")

    def remove(self, path: str) -> None:
        self.removed.append(path)
        self.files.pop(path, None)

    def close(self) -> None:
        pass


class _ManagedDataClient(_Client):
    def __init__(self, receipt: dict[str, object]) -> None:
        super().__init__()
        self.sftp = _ManagedDataSftp(receipt)

    def open_sftp(self) -> _ManagedDataSftp:
        return self.sftp


def _artifact(path: Path, relative_path: str) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


@pytest.mark.parametrize("runtime_class", [HpcRuntime, OverlayHpcRuntime])
def test_execute_solution_stages_arbitrary_run_inputs_and_collects_all_outputs(tmp_path: Path, runtime_class) -> None:
    task_root = tmp_path / "assistant_tasks" / "run_workspace_contract"
    script = task_root / "work" / "train_gpu.py"
    inputs = task_root / "inputs"
    dataset = inputs / "nonlinear-binary-classification-30000x20.csv"
    script.parent.mkdir(parents=True)
    inputs.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    dataset.write_text("feature,target\n0,0\n1,1\n", encoding="utf-8")

    client = _Client()
    runtime = runtime_class(
        run_id="run_workspace_contract",
        local_run_dir=task_root / "work" / "hpc",
        connector=lambda: client,
    )
    staged: dict[str, object] = {}
    progress_events: list[dict[str, object]] = []
    if isinstance(runtime, OverlayHpcRuntime):
        runtime.progress_callback = lambda **event: progress_events.append(event)

    def stage_bundle(files, *, remote_subdir):
        staged["files"] = {name: Path(path) for name, path in files.items()}
        staged["remote_subdir"] = remote_subdir
        return {
            "remote_dir": f"{runtime.remote_run_dir}/{remote_subdir}",
            "combined_sha256": "a" * 64,
            "files": [],
        }

    def download_tree(_sftp, remote_root, local_root, **_kwargs):
        assert remote_root.endswith("/output")
        local_root.mkdir(parents=True, exist_ok=True)
        metrics = local_root / "metrics.json"
        model = local_root / "models" / "best-model.joblib"
        model.parent.mkdir(parents=True)
        metrics.write_text('{"status":"complete"}\n', encoding="utf-8")
        model.write_bytes(b"fixture-model")
        return [_artifact(metrics, "metrics.json"), _artifact(model, "models/best-model.joblib")]

    runtime.stage_bundle = stage_bundle  # type: ignore[method-assign]
    if isinstance(runtime, OverlayHpcRuntime):
        runtime.prepare_solution_environment = lambda _path: {"status": "ready", "packages": {}}  # type: ignore[method-assign]
        runtime._download_solution_outputs = download_tree  # type: ignore[method-assign]
    else:
        runtime._download_tree = download_tree  # type: ignore[method-assign]

    result = runtime.execute_solution(
        solution_id="nonlinear_binary_30000_v1",
        script_path=script,
        data_dir=inputs,
    )

    assert result.status == "completed", result.error
    assert result.exit_code == 0
    assert staged["files"] == {
        "inputs/nonlinear-binary-classification-30000x20.csv": dataset,
        "work/train_gpu.py": script,
    }
    assert str(staged["remote_subdir"]).startswith("solutions/nonlinear_binary_30000_v1/bundle_")
    command, timeout = client.commands[0]
    remote_solution = f"{runtime.remote_run_dir}/solutions/nonlinear_binary_30000_v1"
    assert "EVOMIND_TASK_ROOT=" in command
    assert "EVOMIND_INPUT_DIR=" in command
    assert "EVOMIND_OUTPUT_DIR=" in command
    assert "EVOMIND_WORK_DIR=" in command
    if isinstance(runtime, OverlayHpcRuntime):
        import re
        attempt = re.search(re.escape(remote_solution) + r"/attempts/[a-f0-9]{32}", command)
        assert attempt is not None
        remote_solution = attempt.group(0)
        assert f"HOME={remote_solution}/work/home" in command
        assert f"PYTHONUSERBASE={remote_solution}/work/python-userbase" in command
        assert f"PIP_CACHE_DIR={remote_solution}/work/pip-cache" in command
        assert f"XDG_CACHE_HOME={remote_solution}/work/xdg-cache" in command
    assert "work/train_gpu.py" in command
    assert "--data-dir" in command and "/inputs" in command
    assert "--out-dir" in command and "/output" in command
    remote_compat_output = f"{remote_solution}/work/outputs" if isinstance(runtime, OverlayHpcRuntime) else f"{runtime.remote_run_dir}/{staged['remote_subdir']}/outputs"
    cleanup = f"rm -rf {remote_solution}/output {remote_solution}/work {remote_compat_output}"
    compat_copy = (
        f"if test -d {remote_compat_output}; then "
        f"cp -a {remote_compat_output}/. {remote_solution}/output/; fi;"
    )
    if isinstance(runtime, OverlayHpcRuntime):
        assert "rm -rf" not in command
        assert "flock -n -E 75" in command
    else:
        assert cleanup in command
    assert compat_copy in command
    assert command.index("rc=$?;") < command.index(compat_copy) < command.index("exit $rc")
    assert timeout == runtime.timeout_seconds + 60
    assert {Path(item["path"]).name for item in result.local_artifacts} == {
        "training.log",
        "metrics.json",
        "best-model.joblib",
    }
    assert client.closed is True
    if isinstance(runtime, OverlayHpcRuntime):
        assert [event["phase"] for event in progress_events] == ["staging_solution", "executing_solution"]
        assert all(event["worker_state"] == "unverified" for event in progress_events)
        assert all(event["work_kind"] != "training" for event in progress_events)


def test_dependency_failure_does_not_report_solution_execution(tmp_path: Path) -> None:
    script = tmp_path / "solution.py"
    script.write_text("print('fixture')\n", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "data.csv").write_text("x,y\n0,1\n", encoding="utf-8")
    calls = []
    events = []
    runtime = OverlayHpcRuntime(run_id="dependency_failure", local_run_dir=tmp_path / "run", connector=lambda: calls.append(True))
    runtime.progress_callback = lambda **event: events.append(event)

    def failure(_path):
        raise ValueError("dependency_integrity_failed")

    runtime.prepare_solution_environment = failure
    result = runtime.execute_solution(solution_id="dependency_failure", script_path=script, data_dir=inputs)
    assert result.status == "failed"
    assert calls == []
    assert events == []


@pytest.mark.parametrize("runtime_class", [HpcRuntime, OverlayHpcRuntime])
def test_execute_solution_returns_structured_input_failure_before_connecting(tmp_path: Path, runtime_class) -> None:
    script = tmp_path / "work" / "train_gpu.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    connector_calls: list[bool] = []
    runtime = runtime_class(
        run_id="run_missing_inputs",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: connector_calls.append(True),
    )

    result = runtime.execute_solution(
        solution_id="missing_inputs",
        script_path=script,
        data_dir=tmp_path / "inputs",
    )

    assert result.status == "failed"
    assert result.exit_code == -1
    assert result.failure_type == "input"
    assert "data directory" in result.error
    assert connector_calls == []


def test_hpc_tool_resolves_run_relative_paths_and_preserves_nested_artifacts(tmp_path: Path, monkeypatch) -> None:
    task_root = tmp_path / "assistant_tasks" / "run_tool_paths"
    script = task_root / "work" / "train_gpu.py"
    inputs = task_root / "inputs"
    script.parent.mkdir(parents=True)
    inputs.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    (inputs / "dataset.csv").write_text("feature,target\n0,0\n", encoding="utf-8")

    runtime = AgentRuntime(tmp_path / "service", runtime_root=tmp_path / "runtime")
    runtime.create_session(
        session_id="run_tool_paths",
        objective="fixture",
        workspace_root=str(task_root),
    )
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "91699")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job91699")
    monkeypatch.setattr(
        runtime_tools,
        "_hpc_verify",
        lambda _args, _context: ToolResult("", True, {"samples_passed": 5}, "verified"),
    )
    monkeypatch.setattr(runtime_tools, "_load_bound_hpc_config", lambda *_args: object())
    observed: dict[str, Path] = {}

    def execute_solution(self, *, solution_id, script_path, data_dir):
        observed["script_path"] = Path(script_path)
        observed["data_dir"] = Path(data_dir)
        local_output = self.local_run_dir / "solutions" / solution_id / "output"
        model = local_output / "models" / "best-model.joblib"
        metrics = local_output / "metrics.json"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fixture-model")
        metrics.write_text('{"status":"complete"}\n', encoding="utf-8")
        return HpcJobResult(
            status="completed",
            run_id=self.run_id,
            solution_id=solution_id,
            remote_dir=f"{self.remote_run_dir}/solutions/{solution_id}",
            exit_code=0,
            local_artifacts=[
                _artifact(model, "models/best-model.joblib"),
                _artifact(metrics, "metrics.json"),
            ],
        )

    monkeypatch.setattr(OverlayHpcRuntime, "execute_solution", execute_solution)
    outcome = runtime.invoke_tool(
        "run_tool_paths",
        "hpc_execute_solution",
        {
            "solution_id": "workspace_nested",
            "script_path": "work/train_gpu.py",
            "data_dir": "inputs",
            "timeout_seconds": 300,
        },
    )

    assert outcome["status"] == "completed"
    assert observed == {"script_path": script.resolve(), "data_dir": inputs.resolve()}
    published = outcome["result"]["content"]["published_artifacts"]
    assert all(Path(item["path"]).is_relative_to(task_root / "outputs" / ".a") for item in published)
    assert (task_root / "outputs" / "hpc" / "workspace_nested" / "metrics.json").read_text(encoding="utf-8") == '{"status":"complete"}\n'
    assert (task_root / "outputs" / "hpc" / "workspace_nested" / "models" / "best-model.joblib").read_bytes() == b"fixture-model"
    assert {item["name"] for item in published} == {"metrics.json", "best-model.joblib"}
    assert all(Path(item["path"]).is_file() for item in published)
    runtime.close()


def test_hpc_tool_executes_full_managed_competition_without_local_data_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_root = tmp_path / "assistant_tasks" / "run_managed_competition"
    script = task_root / "work" / "train_gpu.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    persistent = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/cure_bench"
    manifest_sha256 = "c" * 64

    runtime = AgentRuntime(tmp_path / "service", runtime_root=tmp_path / "runtime")
    runtime.create_session(
        session_id="run_managed_competition",
        objective="fixture",
        workspace_root=str(task_root),
    )
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "91699")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job91699")
    monkeypatch.setattr(
        runtime_tools,
        "_managed_competition_training_source",
        lambda _competition, _args, _context: {
            "competition": "cure_bench",
            "source_kind": "managed_competition_data",
            "data_status": "FULL_DATA_READY",
            "persistent_root": persistent,
            "manifest_sha256": manifest_sha256,
            "sha256": manifest_sha256,
            "files": 3,
            "bytes": 2_631_105,
            "local_copy_created": False,
        },
    )
    monkeypatch.setattr(
        runtime_tools,
        "_hpc_verify",
        lambda *_args: pytest.fail("competition binding already performs the current 5/5 status verification"),
    )
    monkeypatch.setattr(runtime_tools, "_load_bound_hpc_config", lambda *_args: object())
    observed: dict[str, object] = {}

    def execute_solution(self, **kwargs):
        observed.update(kwargs)
        local_output = self.local_run_dir / "solutions" / kwargs["solution_id"] / "output"
        metrics = local_output / "metrics.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text('{"status":"complete"}\n', encoding="utf-8")
        return HpcJobResult(
            status="completed",
            run_id=self.run_id,
            solution_id=kwargs["solution_id"],
            remote_dir=f"{self.remote_run_dir}/solutions/{kwargs['solution_id']}",
            exit_code=0,
            local_artifacts=[_artifact(metrics, "metrics.json")],
        )

    monkeypatch.setattr(OverlayHpcRuntime, "execute_solution", execute_solution)
    outcome = runtime.invoke_tool(
        "run_managed_competition",
        "hpc_execute_solution",
        {
            "solution_id": "cure_bench_v1",
            "script_path": "work/train_gpu.py",
            "competition": "cure_bench",
            "timeout_seconds": 300,
        },
    )

    assert outcome["status"] == "completed"
    assert observed["script_path"] == script.resolve()
    assert observed["persistent_data_root"] == persistent
    assert observed["expected_manifest_sha256"] == manifest_sha256
    assert "data_dir" not in observed
    binding = outcome["result"]["content"]["competition_data_binding"]
    assert binding["competition"] == "cure_bench"
    assert binding["local_copy_created"] is False
    runtime.close()


@pytest.mark.parametrize(
    ("script_name", "populate_data", "message_fragment"),
    [
        ("train_gpu.sh", True, ".py file"),
        ("train_gpu.py", False, "at least one regular non-symlink input file"),
    ],
)
def test_hpc_tool_rejects_invalid_ordinary_inputs_before_identity_verification(
    tmp_path: Path,
    monkeypatch,
    script_name: str,
    populate_data: bool,
    message_fragment: str,
) -> None:
    script = tmp_path / "work" / script_name
    inputs = tmp_path / "inputs"
    script.parent.mkdir(parents=True)
    inputs.mkdir()
    script.write_text("print('fixture')\n", encoding="utf-8")
    if populate_data:
        (inputs / "dataset.csv").write_text("feature,target\n0,0\n", encoding="utf-8")

    def unexpected_verify(_args, _context):
        pytest.fail("invalid local inputs must fail before _hpc_verify")

    monkeypatch.setattr(runtime_tools, "_hpc_verify", unexpected_verify)
    context = type("InputContext", (), {"workspace_root": tmp_path, "metadata": {}})()

    result = runtime_tools._hpc_execute_solution(
        {
            "solution_id": "invalid_inputs",
            "script_path": str(script),
            "data_dir": str(inputs),
        },
        context,
    )

    assert result.ok is False
    assert result.error == "invalid_hpc_solution_inputs"
    assert message_fragment in result.summary


def test_hpc_tool_fails_closed_when_exit_zero_has_only_runtime_log(tmp_path: Path, monkeypatch) -> None:
    task_root = tmp_path / "assistant_tasks" / "run_missing_output"
    script = task_root / "work" / "train_gpu.py"
    inputs = task_root / "inputs"
    script.parent.mkdir(parents=True)
    inputs.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    (inputs / "dataset.csv").write_text("feature,target\n0,0\n", encoding="utf-8")

    runtime = AgentRuntime(tmp_path / "service", runtime_root=tmp_path / "runtime")
    runtime.create_session(
        session_id="run_missing_output",
        objective="fixture",
        workspace_root=str(task_root),
    )
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "91699")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job91699")
    monkeypatch.setattr(
        runtime_tools,
        "_hpc_verify",
        lambda _args, _context: ToolResult("", True, {"samples_passed": 5}, "verified"),
    )
    monkeypatch.setattr(runtime_tools, "_load_bound_hpc_config", lambda *_args: object())

    def execute_solution(self, *, solution_id, script_path, data_dir):
        local_output = self.local_run_dir / "solutions" / solution_id / "output"
        local_output.mkdir(parents=True)
        training_log = local_output / "training.log"
        training_log.write_text("process exited 0\n", encoding="utf-8")
        return HpcJobResult(
            status="completed",
            run_id=self.run_id,
            solution_id=solution_id,
            remote_dir=f"{self.remote_run_dir}/solutions/{solution_id}",
            exit_code=0,
            local_artifacts=[_artifact(training_log, "training.log")],
        )

    monkeypatch.setattr(OverlayHpcRuntime, "execute_solution", execute_solution)
    outcome = runtime.invoke_tool(
        "run_missing_output",
        "hpc_execute_solution",
        {
            "solution_id": "missing_output",
            "script_path": "work/train_gpu.py",
            "data_dir": "inputs",
            "timeout_seconds": 300,
        },
    )

    assert outcome["status"] == "failed"
    assert outcome["result"]["error"] == "hpc_output_artifact_missing"
    content = outcome["result"]["content"]
    assert content["status"] == "failed"
    assert content["runtime_status"] == "completed"
    assert content["published_output_artifact_count"] == 0
    assert {item["name"] for item in content["published_artifacts"]} == {"training.log"}
    runtime.close()


def test_hpc_execute_solution_description_states_python_io_and_artifact_contract() -> None:
    spec = runtime_tools.build_default_registry().get("hpc_execute_solution")

    assert spec is not None
    for required_text in (
        "Python .py",
        "--data-dir",
        "--out-dir",
        "at least one regular non-symlink input file",
        "FULL_DATA_READY",
        "at least one output artifact",
    ):
        assert required_text in spec.description


def test_overlay_uses_manifest_bound_persistent_competition_data_without_staging_it(tmp_path: Path) -> None:
    task_root = tmp_path / "assistant_tasks" / "run_persistent_inputs"
    script = task_root / "work" / "train_gpu.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")
    persistent = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/cure_bench"
    manifest_sha256 = "d" * 64
    client = _Client()
    runtime = OverlayHpcRuntime(
        run_id="run_persistent_inputs",
        local_run_dir=task_root / "work" / "hpc",
        connector=lambda: client,
    )
    staged: dict[str, object] = {}

    def stage_bundle(files, *, remote_subdir):
        staged["files"] = {name: Path(path) for name, path in files.items()}
        staged["remote_subdir"] = remote_subdir
        return {
            "remote_dir": f"{runtime.remote_run_dir}/{remote_subdir}",
            "combined_sha256": "a" * 64,
            "files": [],
        }

    def download_tree(_sftp, _remote_root, local_root):
        local_root.mkdir(parents=True, exist_ok=True)
        metrics = local_root / "metrics.json"
        metrics.write_text('{"status":"complete"}\n', encoding="utf-8")
        return [_artifact(metrics, "metrics.json")]

    runtime.stage_bundle = stage_bundle  # type: ignore[method-assign]
    runtime.prepare_solution_environment = lambda _path: {"status": "ready", "packages": {}}  # type: ignore[method-assign]
    runtime._download_solution_outputs = download_tree  # type: ignore[method-assign]

    result = runtime.execute_solution(
        solution_id="cure_bench_persistent_v1",
        script_path=script,
        persistent_data_root=persistent,
        expected_manifest_sha256=manifest_sha256,
    )

    assert result.status == "completed"
    assert staged["files"] == {"work/train_gpu.py": script}
    command, _timeout = client.commands[0]
    assert f"--data-dir {persistent}" in command
    assert f"EVOMIND_INPUT_DIR={persistent}" in command
    assert manifest_sha256 in command
    assert f"{persistent}/.evomind/data-manifest.json" in command
    assert "managed competition manifest mismatch" in command
    assert not any(name.startswith("inputs/") for name in staged["files"])


def test_overlay_prepares_allowlisted_xgboost_in_run_scoped_directory(tmp_path: Path) -> None:
    script = tmp_path / "train_gpu.py"
    script.write_text("import numpy\nimport xgboost as xgb\n", encoding="utf-8")
    client = _DependencyClient()
    runtime = OverlayHpcRuntime(
        run_id="run_dependency_contract",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: client,
    )

    result = runtime.prepare_solution_environment(script)

    assert result["status"] == "ready"
    assert result["packages"] == {"xgboost": "2.1.3"}
    command, timeout = client.commands[0]
    locked_root, pins, source = _dependency_command(command)
    assert runtime.remote_solution_deps == locked_root + "/site-packages"
    assert pins == {"xgboost": "2.1.3"}
    assert '"--require-hashes"' in source
    assert '"--no-index"' in source
    assert '"--dry-run"' in source
    assert '"--only-binary=:all:"' in source
    assert timeout == 2800
    assert result["lock_sha256"] == "a" * 64
    assert client.closed is True


def test_overlay_prepares_allowlisted_lightgbm_without_mutating_user_site(tmp_path: Path) -> None:
    script = tmp_path / "train_gpu.py"
    script.write_text("import lightgbm as lgb\nimport xgboost as xgb\n", encoding="utf-8")
    client = _DependencyClient()
    runtime = OverlayHpcRuntime(
        run_id="run_lightgbm_dependency_contract",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: client,
    )

    result = runtime.prepare_solution_environment(script)

    assert result["status"] == "ready"
    assert result["packages"] == {"lightgbm": "4.6.0", "xgboost": "2.1.3"}
    command, _timeout = client.commands[0]
    _locked_root, pins, source = _dependency_command(command)
    assert pins == {"lightgbm": "4.6.0", "xgboost": "2.1.3"}
    assert '"--target"' in source
    assert '"PYTHONNOUSERSITE": "1"' in source
    assert '"--user"' not in source
    assert '"--require-hashes"' in source


def test_overlay_creates_fresh_remote_run_root_before_dependency_preparation(tmp_path: Path) -> None:
    script = tmp_path / "train_gpu.py"
    script.write_text("import xgboost as xgb\n", encoding="utf-8")
    client = _DependencyClient()
    runtime = OverlayHpcRuntime(
        run_id="run_fresh_dependency_contract",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: client,
    )

    runtime.prepare_solution_environment(script)

    command, _timeout = client.commands[0]
    locked_root, _pins, source = _dependency_command(command)
    assert locked_root.startswith(runtime.remote_runtime_dir + "/solution_deps/")
    assert "root.mkdir(parents=True, exist_ok=True)" in source
    assert source.index("root.mkdir(parents=True, exist_ok=True)") < source.index('command("install", "--dry-run"')
    assert "rm -rf" not in command
    runtime.prepare_solution_environment(script)
    second_root, _pins, _source = _dependency_command(client.commands[1][0])
    assert second_root == locked_root


def test_overlay_does_not_connect_when_script_needs_no_managed_dependency(tmp_path: Path) -> None:
    script = tmp_path / "train_gpu.py"
    script.write_text("import json\n", encoding="utf-8")
    calls: list[bool] = []
    runtime = OverlayHpcRuntime(
        run_id="run_no_dependency",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: calls.append(True),
    )

    result = runtime.prepare_solution_environment(script)

    assert result == {"status": "not_required", "packages": {}}
    assert calls == []


def test_managed_data_task_keeps_secret_values_out_of_command_and_cleans_ephemeral_file(tmp_path: Path) -> None:
    script = tmp_path / "adapter.sh"
    script.write_text("#!/usr/bin/env bash\nset -eu\n", encoding="utf-8")
    secret = b'{"username":"fixture-user","password":"fixture-secret"}'
    client = _ManagedDataClient({"competition": "weather4cast", "status": "RUNNING"})
    runtime = OverlayHpcRuntime(
        run_id="run_weather_secret_contract",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: client,
    )

    result = runtime.execute_managed_data_task(
        task_id="weather4cast",
        script_path=script,
        persistent_data_root="/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/weather4cast",
        secret_files={"EVOMIND_SECRET_WEATHER4CAST_FILE": secret},
        timeout_seconds=60,
    )

    command, _timeout = client.commands[0]
    assert result["status"] == "completed"
    assert "fixture-user" not in command and "fixture-secret" not in command
    assert "EVOMIND_SECRET_WEATHER4CAST_FILE=" in command
    secret_paths = [path for path in client.sftp.modes if "/.secrets/" in path]
    assert len(secret_paths) == 1
    assert client.sftp.modes[secret_paths[0]] == 0o600
    assert secret_paths[0] in client.sftp.removed
    assert secret not in client.sftp.files.values()
    assert result["stdout_tail"] == "suppressed_secret_bearing_task_output"
    assert result["secret_values_logged"] is False


def test_managed_data_task_keeps_prepare_and_status_evidence_separate_from_receipt_identity(tmp_path: Path) -> None:
    script = tmp_path / "adapter.sh"
    script.write_text("#!/usr/bin/env bash\nset -eu\n", encoding="utf-8")
    client = _ManagedDataClient({"competition": "mindgames", "status": "FULL_DATA_READY"})
    runtime = OverlayHpcRuntime(
        run_id="run_mindgames_evidence_contract",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: client,
    )

    result = runtime.execute_managed_data_task(
        task_id="mindgames-prepare",
        receipt_competition="mindgames",
        script_path=script,
        persistent_data_root="/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/mindgames",
        timeout_seconds=60,
    )

    assert result["status"] == "completed"
    assert result["task_id"] == "mindgames-prepare"
    assert "/managed_data/mindgames-prepare" in result["remote_task_root"]
    assert Path(result["local_log"]).parent.name == "mindgames-prepare"


def test_managed_data_task_rejects_persistent_root_outside_competition_directory_before_connecting(tmp_path: Path) -> None:
    script = tmp_path / "adapter.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    connected: list[bool] = []
    runtime = OverlayHpcRuntime(
        run_id="run_bad_persistent_root",
        local_run_dir=tmp_path / "hpc",
        connector=lambda: connected.append(True),
    )

    with pytest.raises(ValueError, match="managed competition directory"):
        runtime.execute_managed_data_task(
            task_id="mindgames",
            script_path=script,
            persistent_data_root="/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/other",
        )
    assert connected == []
