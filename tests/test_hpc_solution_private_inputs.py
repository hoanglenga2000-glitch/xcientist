import hashlib
from pathlib import Path
from types import SimpleNamespace

from evomind_runtime.hpc_runtime_overlay import HpcRuntime


class Writer:
    def __init__(self, storage, path): self.storage, self.path, self.body = storage, path, bytearray()
    def write(self, value): self.body.extend(value)
    def flush(self): pass
    def close(self): self.storage[self.path] = bytes(self.body)


class Sftp:
    def __init__(self, fail_cleanup=False):
        self.files, self.dirs, self.modes, self.removed = {}, set(), {}, []
        self.fail_cleanup = fail_cleanup
    def lstat(self, path):
        if path not in self.dirs: raise FileNotFoundError(path)
        return SimpleNamespace(st_mode=0o040700)
    def mkdir(self, path): self.dirs.add(path)
    def chmod(self, path, mode): self.modes[path] = mode
    def file(self, path, mode): return Writer(self.files, path)
    def remove(self, path):
        if self.fail_cleanup: raise PermissionError("fixture cleanup failure")
        self.removed.append(path); self.files.pop(path, None)
    def close(self): pass


class Client:
    def __init__(self, fail_cleanup=False): self.sftp, self.closed = Sftp(fail_cleanup), False
    def open_sftp(self): return self.sftp
    def close(self): self.closed = True


def exercise(tmp_path, fail_cleanup=False):
    script = tmp_path / "adapter.py"
    script.write_text("import json\n")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "source.json").write_text("{}")
    client = Client(fail_cleanup)
    runtime = HpcRuntime(run_id="run_private_inputs", local_run_dir=tmp_path / "run", connector=lambda: client)
    runtime.prepare_solution_environment = lambda _path: {}
    staged, commands = {}, []
    def stage(files, *, remote_subdir):
        staged.update(files)
        return {"remote_dir": runtime.remote_run_dir + "/" + remote_subdir}
    runtime.stage_bundle = stage
    def execute(_client, command, timeout):
        commands.append(command)
        return 0, "https://us.aws.cdn.hf.co/object?Signature=fixture-secret", "fixture-secret"
    runtime._exec_observed = execute
    def collect(_sftp, remote, local):
        path = local / "model-manifest.json"
        path.write_text('{"fixture":true}')
        return [{"path": str(path), "relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}]
    runtime._download_solution_outputs = collect
    result = runtime.execute_solution(solution_id="model_prepare", script_path=script, data_dir=inputs,
                                      secret_files={"EVOMIND_SECRET_MODEL_CDN_FILE": b"fixture-secret"})
    return result, runtime, client, staged, commands


def test_private_values_not_bundled_logged_or_passed_on_command_line(tmp_path):
    result, runtime, client, staged, commands = exercise(tmp_path)
    assert result.status == "completed" and result.exit_code == 0
    assert set(staged) == {"work/train_gpu.py", "inputs/source.json"}
    assert "fixture-secret" not in commands[0]
    assert "EVOMIND_SECRET_MODEL_CDN_FILE=" in commands[0]
    assert "trap " in commands[0]
    paths = [path for path, mode in client.sftp.modes.items() if mode == 0o600]
    assert len(paths) == 1 and paths[0] in client.sftp.removed
    assert not client.sftp.files
    assert runtime.last_solution_secret_cleanup_ok
    assert "fixture-secret" not in str(result)
    log = next(item for item in result.local_artifacts if item["relative_path"] == "training.log")
    assert "fixture-secret" not in Path(log["path"]).read_text()


def test_cleanup_failure_prevents_success_and_retains_artifact_evidence(tmp_path):
    result, runtime, client, _, _ = exercise(tmp_path, fail_cleanup=True)
    assert result.status == "failed" and result.exit_code == -1
    assert "managed_secret_cleanup_failed" in result.error
    assert result.local_artifacts
    assert not runtime.last_solution_secret_cleanup_ok
    assert client.closed


def test_bad_private_input_rejected_before_connection(tmp_path):
    calls = []
    runtime = HpcRuntime(run_id="run_invalid_private", local_run_dir=tmp_path, connector=lambda: calls.append(True))
    result = runtime.execute_solution(solution_id="invalid", script_path=tmp_path / "missing.py", data_dir=tmp_path,
                                      secret_files={"INJECT_OTHER_ENV": b"fixture-secret"})
    assert result.status == "failed" and calls == []
    assert "fixture-secret" not in str(result)
