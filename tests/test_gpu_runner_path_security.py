from __future__ import annotations

import io

import pytest

from research_os.gpu_runner import GPURunner, GPURunnerConfig, _shell_arg
from research_os.hpc_policy import HPCPolicyError, join_remote_workspace


class _Stream(io.BytesIO):
    def __init__(self, payload: bytes = b"", exit_code: int = 0) -> None:
        super().__init__(payload)
        self.channel = self
        self._exit_code = exit_code

    def recv_exit_status(self) -> int:
        return self._exit_code


class _SFTP:
    class _Handle(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()
            return False

    def file(self, *_args, **_kwargs):
        return self._Handle()

    def close(self) -> None:
        pass


class _SSH:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def exec_command(self, command: str, timeout: int | None = None):
        self.commands.append(command)
        if "solution.py" in command and "timeout" in command:
            return None, _Stream(b"CV_SCORE=0.75\n"), _Stream()
        if command.startswith("ls"):
            return None, _Stream(b"metrics.json\nsubmission.csv\n"), _Stream()
        return None, _Stream(), _Stream()

    def open_sftp(self) -> _SFTP:
        return _SFTP()

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "unsafe",
    [
        "/tmp/unconstrained",
        "../escape",
        "safe/../escape",
        "data;touch-pwned",
        "data$(id)",
        "data with space",
        "",
        ".",
        "safe//empty",
        "safe/./dot",
    ],
)
def test_gpu_runner_rejects_unsafe_data_dir_before_connecting(unsafe: str) -> None:
    connected = False

    def connect():
        nonlocal connected
        connected = True
        return _SSH()

    runner = GPURunner(
        "essay-scoring",
        config=GPURunnerConfig(remote_root="/srv/evomind"),
        connect=connect,
    )

    with pytest.raises(HPCPolicyError):
        runner.run("print('x')", data_dir=unsafe, out_dir="unused-local-output", exp_id="EXP001")

    assert connected is False


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("evolution_subdir", "/tmp/runs"),
        ("evolution_subdir", "runs;id"),
        ("data_root", "../datasets"),
        ("data_root", "datasets$(id)"),
        ("remote_python", "python3;id"),
        ("timeout", "60;id"),
    ],
)
def test_gpu_runner_rejects_unsafe_config_paths(field: str, unsafe: str) -> None:
    kwargs = {field: unsafe, "remote_root": "/srv/evomind"}
    with pytest.raises(HPCPolicyError):
        GPURunnerConfig(**kwargs)


@pytest.mark.parametrize("unsafe", ["/srv/evomind;id", "/srv/evomind$(id)", "/srv/evomind space"])
def test_gpu_runner_rejects_unsafe_remote_workspace(unsafe: str) -> None:
    with pytest.raises(HPCPolicyError):
        GPURunnerConfig(remote_root=unsafe)


@pytest.mark.parametrize("unsafe", ["/absolute-task", "../task", "task;id", "task$(id)", "task name"])
def test_gpu_runner_rejects_unsafe_task_path(unsafe: str) -> None:
    with pytest.raises(HPCPolicyError):
        GPURunner(unsafe, config=GPURunnerConfig(remote_root="/srv/evomind"))


@pytest.mark.parametrize("unsafe", ["/EXP001", "../EXP001", "EXP;id", "EXP$(id)", "EXP 001"])
def test_gpu_runner_rejects_unsafe_experiment_path_before_connecting(unsafe: str) -> None:
    runner = GPURunner(
        "essay-scoring",
        config=GPURunnerConfig(remote_root="/srv/evomind"),
        connect=lambda: pytest.fail("invalid exp_id reached SSH"),
    )
    with pytest.raises(HPCPolicyError):
        runner.run("print('x')", data_dir="essay-scoring", out_dir="unused", exp_id=unsafe)


def test_join_remote_workspace_keeps_safe_relative_path_inside_root() -> None:
    path = join_remote_workspace("/srv/evomind", "datasets/raw", "essay-scoring_v2")
    assert path == "/srv/evomind/datasets/raw/essay-scoring_v2"


def test_remote_shell_arguments_use_posix_shell_quoting() -> None:
    assert _shell_arg("value with space;$(id)") == "'value with space;$(id)'"


def test_gpu_runner_executes_only_workspace_contained_safe_paths() -> None:
    ssh = _SSH()
    runner = GPURunner(
        "essay-scoring_v2",
        config=GPURunnerConfig(
            remote_root="/srv/evomind",
            evolution_subdir="runs/evolution",
            data_root="datasets/raw",
            remote_python="/opt/evomind/bin/python3",
            timeout=60,
        ),
        connect=lambda: ssh,
    )

    result = runner.run(
        "print('x')",
        data_dir="essay-scoring_v2",
        out_dir="ignored-local-output",
        exp_id="EXP_001",
    )

    assert result.success is True
    assert result.out_dir == "/srv/evomind/runs/evolution/essay-scoring_v2/EXP_001/out"
    assert ssh.closed is True
    assert ssh.commands
    assert all("/srv/evomind" in command for command in ssh.commands)
    assert all("../" not in command for command in ssh.commands)
    run_command = next(command for command in ssh.commands if "timeout" in command)
    assert "/srv/evomind/datasets/raw/essay-scoring_v2" in run_command
    assert "/opt/evomind/bin/python3" in run_command
