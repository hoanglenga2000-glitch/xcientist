"""Bundle-first HPC runtime overlay for durable assistant Run execution.

Production starts Python with the managed bundle before the tracked release
source tree.  This class deliberately inherits the stable release runtime and
overrides only the generic assistant solution contract, so the hotfix can stay
inside the sealed bundle without dirtying the tracked release checkout.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import posixpath
import re
import shlex
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from research_os.hpc_runtime import (
    HpcJobResult,
    HpcRuntime as _ReleaseHpcRuntime,
    _sha256,
    _validate_component,
    _validate_relative_path,
    _validate_remote,
)
from research_agent_workstation.server.core.gpu_credentials import ALLOWED_GPU_REMOTE_ROOT

_MAX_SOLUTION_INPUT_FILES = 256
_MAX_SOLUTION_INPUT_BYTES = 256 * 1024 * 1024
_MAX_SOLUTION_OUTPUT_FILES = 512
_MAX_SOLUTION_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
_SOLUTION_PACKAGE_PINS = {
    "lightgbm": "4.6.0",
    "xgboost": "2.1.3",
    "torch": "2.8.0",
    "transformers": "4.55.4",
    "safetensors": "0.6.2",
}
_PERSISTENT_COMPETITION_DATA_ROOT = _validate_remote(
    posixpath.join(ALLOWED_GPU_REMOTE_ROOT, "competition_data")
)
_SECRET_ENV_NAME = re.compile(r"^EVOMIND_SECRET_[A-Z0-9_]{1,64}_FILE$")


class HpcRuntime(_ReleaseHpcRuntime):
    """Patch the release runtime's Kaggle-only solution staging contract."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.progress_callback = None
        self.trusted_progress = False
        self.managed_work_kind = "downloading"
        self.deadline = None
        self.solution_environment_manifest = ""
        self.remote_solution_deps = _validate_remote(posixpath.join(self.remote_runtime_dir, "solution_deps"))
        self._solution_deps_base = self.remote_solution_deps

    def remaining_seconds(self, maximum: int, *, margin: int = 0) -> int:
        if self.deadline is None:
            return maximum
        remaining = int(self.deadline - time.monotonic()) - margin
        if remaining < 1:
            raise TimeoutError("managed_execution_budget_deadline")
        return min(maximum, remaining)

    def _exec_observed(self, client, command: str, *, timeout: int) -> tuple[int, str, str]:
        if self.progress_callback is None:
            return self._exec(client, command, timeout=timeout)
        from .execution_progress import safe_text

        _stdin, stdout, _stderr = client.exec_command(command, timeout=timeout)
        channel = stdout.channel
        output, errors = bytearray(), bytearray()
        line_buffer = ""
        deadline = time.monotonic() + timeout
        while True:
            if channel.recv_ready():
                block = channel.recv(65536)
                output.extend(block)
                if len(output) > 8 * 1024 * 1024:
                    del output[:-8 * 1024 * 1024]
                line_buffer += block.decode("utf-8", "replace")
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    if self.trusted_progress and line.startswith("EVOMIND_PROGRESS "):
                        try:
                            value = json.loads(line[len("EVOMIND_PROGRESS "):])
                            fields = {key: value[key] for key in ("completed_units", "total_units", "unit", "phase", "failure_class", "detail") if key in value}
                            self.progress_callback(source="managed_adapter", work_kind=self.managed_work_kind, worker_state="observed", **fields)
                        except (ValueError, TypeError):
                            pass
                line_buffer = line_buffer[-65536:]
            if channel.recv_stderr_ready():
                errors.extend(channel.recv_stderr(65536))
                if len(errors) > 2 * 1024 * 1024:
                    del errors[:-2 * 1024 * 1024]
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if time.monotonic() > deadline:
                # Never send a process signal or create a replacement worker here.
                self.progress_callback(phase="transport_deadline", worker_state="unverified", detail="Remote settlement requires identity-bound reconciliation")
                raise TimeoutError("remote_settlement_unconfirmed")
            time.sleep(0.05)
        out = safe_text(output.decode("utf-8", "replace"), 8 * 1024 * 1024)
        err = safe_text(errors.decode("utf-8", "replace"), 2 * 1024 * 1024)
        return channel.recv_exit_status(), out, err

    @staticmethod
    def classify_failure(*, exit_code: int | None = None, error: str = "") -> str:
        low = error.lower()
        if any(term in low for term in ("filenotfounderror", "file not found", "data directory", "input directory")):
            return "input"
        return _ReleaseHpcRuntime.classify_failure(exit_code=exit_code, error=error)

    def _download_solution_outputs(self, sftp, remote_root: str, local_root: Path) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        downloaded_bytes = 0
        local_root.mkdir(parents=True, exist_ok=True)

        def walk(remote_dir: str, local_dir: Path) -> None:
            nonlocal downloaded_bytes
            local_dir.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(remote_dir):
                remote_path = _validate_remote(posixpath.join(remote_dir, entry.filename))
                local_path = local_dir / entry.filename
                file_type = entry.st_mode & 0o170000
                if file_type == 0o040000:
                    walk(remote_path, local_path)
                    continue
                if file_type != 0o100000:
                    continue
                size = int(getattr(entry, "st_size", 0) or 0)
                if len(artifacts) >= _MAX_SOLUTION_OUTPUT_FILES:
                    raise RuntimeError("remote output exceeds the bounded artifact file count")
                if downloaded_bytes + size > _MAX_SOLUTION_OUTPUT_BYTES:
                    raise RuntimeError("remote output exceeds the bounded artifact byte limit")
                sftp.get(remote_path, str(local_path))
                downloaded_bytes += local_path.stat().st_size
                artifacts.append(
                    {
                        "path": str(local_path),
                        "relative_path": local_path.relative_to(local_root).as_posix(),
                        "sha256": _sha256(local_path),
                        "bytes": local_path.stat().st_size,
                    }
                )

        walk(remote_root, local_root)
        return artifacts

    def prepare_solution_environment(self, script_path: str | Path) -> dict[str, Any]:
        source = Path(script_path).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source, filename=str(script_path))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        required = {name: version for name, version in _SOLUTION_PACKAGE_PINS.items() if name in imported}
        if "torch" in required or "transformers" in required:
            required.update(numpy="2.2.6", safetensors="0.6.2")
        if not required:
            return {"status": "not_required", "packages": {}}

        from . import dependency_lock
        pin_key = hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest()[:16]
        locked_root = _validate_remote(posixpath.join(self._solution_deps_base, pin_key))
        source = Path(dependency_lock.__file__).read_bytes()
        encoded = base64.b64encode(source).decode("ascii")
        program = "import base64;exec(compile(base64.b64decode(" + repr(encoded) + "),'managed_dependency_lock.py','exec'))"
        client = self._connector()
        try:
            if self.progress_callback:
                self.progress_callback(source="managed_adapter", work_kind="installing", phase="hash_locked_dependencies", worker_state="observed", detail="Resolving exact official wheels before installation")
            limit = self.remaining_seconds(2800, margin=60)
            prefix = f"timeout --kill-after=10 {limit} " if self.deadline is not None else ""
            command = prefix + "python3 -c " + shlex.quote(program) + " " + shlex.quote(locked_root) + " " + shlex.quote(json.dumps(required))
            rc, out, err = self._exec(client, command, timeout=limit + 20 if self.deadline is not None else limit)
            if rc != 0:
                raise RuntimeError(f"managed dependency lock preparation failed: exit={rc}")
            expected_target = _validate_remote(posixpath.join(locked_root, "site-packages"))
            receipt = dependency_lock.validate_install_receipt(out, expected_target, required)
            self.remote_solution_deps = expected_target
            self.solution_environment_manifest = _validate_remote(posixpath.join(locked_root, "dependency-lock.json"))
            return {"status": "ready", "packages": required, "remote_root": expected_target, "lock_sha256": receipt["lock_sha256"]}
        finally:
            client.close()

    def cancel(self, *, solution_id: str) -> bool:
        from . import managed_cancel

        solution_id = _validate_component(solution_id, "solution_id")
        remote_solution = _validate_remote(posixpath.join(self.remote_run_dir, "solutions", solution_id))
        encoded = base64.b64encode(Path(managed_cancel.__file__).read_bytes()).decode("ascii")
        program = "import base64;exec(compile(base64.b64decode(" + repr(encoded) + "),'managed_cancel.py','exec'))"
        client = self._connector()
        try:
            command = "python3 -c " + shlex.quote(program) + " " + shlex.quote(remote_solution) + " cancel"
            rc, out, _err = self._exec(client, command, timeout=30)
            if rc != 0:
                raise ValueError("managed_cancel_transport_failed")
            self.last_cancel_receipt = managed_cancel.confirmed_receipt(out, remote_solution)
            return True
        finally:
            client.close()

    @staticmethod
    def _write_remote_bytes(sftp, remote_path: str, payload: bytes, *, mode: int) -> None:
        handle = sftp.file(remote_path, "wb")
        try:
            handle.write(payload)
            if hasattr(handle, "flush"):
                handle.flush()
        finally:
            handle.close()
        sftp.chmod(remote_path, mode)

    @staticmethod
    def _persistent_competition_root(value: str) -> str:
        normalized = _validate_remote(value)
        if normalized != _PERSISTENT_COMPETITION_DATA_ROOT and not normalized.startswith(
            _PERSISTENT_COMPETITION_DATA_ROOT + "/"
        ):
            raise ValueError("persistent data root is outside the managed competition directory")
        return normalized

    def execute_managed_data_task(
        self,
        *,
        task_id: str,
        receipt_competition: str | None = None,
        script_path: str | Path,
        persistent_data_root: str,
        secret_files: dict[str, bytes] | None = None,
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        """Run one trusted competition-data adapter against the persistent HPC root.

        Secret values are uploaded only as mode-0600 ephemeral files.  The
        command receives file paths, never secret bytes, and every exit path
        attempts removal through SFTP.  Large datasets stay on the shared HPC
        filesystem; only the bounded JSON receipt is copied back to the Run.
        """

        task_id = _validate_component(task_id, "task_id")
        script_path = Path(script_path)
        persistent_root = self._persistent_competition_root(str(persistent_data_root))
        if script_path.is_symlink() or not script_path.is_file():
            raise FileNotFoundError(f"managed data script not found: {script_path}")
        script_bytes = script_path.read_bytes()
        if not script_bytes or len(script_bytes) > 2 * 1024 * 1024:
            raise ValueError("managed data script is empty or exceeds the bounded size")

        supplied_secrets = dict(secret_files or {})
        for name, value in supplied_secrets.items():
            if not _SECRET_ENV_NAME.fullmatch(name):
                raise ValueError("invalid managed secret environment name")
            if not isinstance(value, bytes) or not value or len(value) > 64 * 1024:
                raise ValueError("managed secret payload is empty or exceeds the bounded size")

        remote_task = _validate_remote(posixpath.join(self.remote_run_dir, "managed_data", task_id))
        remote_script = _validate_remote(posixpath.join(remote_task, "adapter.sh"))
        remote_receipt = _validate_remote(posixpath.join(remote_task, "receipt.json"))
        remote_secret_root = _validate_remote(posixpath.join(remote_task, ".secrets"))
        local_root = self.local_run_dir / "managed_data" / task_id
        local_root.mkdir(parents=True, exist_ok=True)
        local_receipt = local_root / "receipt.json"
        local_log = local_root / "execution.log"
        client = self._connector()
        sftp = None
        remote_secret_paths: list[str] = []
        try:
            sftp = client.open_sftp()
            self._mkdirs(sftp, remote_task)
            self._mkdirs(sftp, remote_secret_root)
            self._write_remote_bytes(sftp, remote_script, script_bytes, mode=0o700)
            secret_environment: list[str] = []
            for index, (name, payload) in enumerate(sorted(supplied_secrets.items())):
                remote_path = _validate_remote(posixpath.join(remote_secret_root, f"secret-{index:02d}"))
                self._write_remote_bytes(sftp, remote_path, payload, mode=0o600)
                remote_secret_paths.append(remote_path)
                secret_environment.append(f"{name}={shlex.quote(remote_path)}")

            q_task = shlex.quote(remote_task)
            q_script = shlex.quote(remote_script)
            q_root = shlex.quote(persistent_root)
            q_receipt = shlex.quote(remote_receipt)
            environment = " ".join(secret_environment)
            command = (
                f"mkdir -p {q_task} {q_root} && rm -f {q_receipt} && "
                f"timeout {max(30, int(timeout_seconds))} env EVOMIND_COMPETITION_DATA_ROOT={q_root} "
                f"EVOMIND_COMPETITION_RECEIPT={q_receipt} EVOMIND_RUN_ID={shlex.quote(self.run_id)} "
                f"{environment + ' ' if environment else ''}bash {q_script}"
            )
            rc, out, err = self._exec(client, command, timeout=max(30, int(timeout_seconds)) + 60)
            if supplied_secrets:
                local_log.write_text(
                    json.dumps(
                        {
                            "schema": "evomind.managed_data_execution_log.v1",
                            "exit_code": rc,
                            "output_suppressed": True,
                            "reason": "secret_bearing_task",
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                local_log.write_text(out + ("\n[stderr]\n" + err if err else ""), encoding="utf-8")

            receipt: dict[str, Any] = {}
            try:
                sftp.get(remote_receipt, str(local_receipt))
                if local_receipt.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("managed data receipt exceeds the bounded size")
                loaded = json.loads(local_receipt.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("managed data receipt must be a JSON object")
                receipt = loaded
            except OSError:
                pass
            expected_competition = str(receipt_competition or task_id)
            ok = rc == 0 and bool(receipt) and str(receipt.get("competition") or "") == expected_competition
            return {
                "schema": "evomind.managed_competition_data_execution.v1",
                "status": "completed" if ok else "failed",
                "task_id": task_id,
                "remote_task_root": remote_task,
                "persistent_data_root": persistent_root,
                "exit_code": rc,
                "receipt": receipt,
                "local_receipt": str(local_receipt) if local_receipt.is_file() else "",
                "local_log": str(local_log),
                "secret_files_used": sorted(supplied_secrets),
                "secret_values_logged": False,
                "stdout_tail": "suppressed_secret_bearing_task_output" if supplied_secrets else out[-1200:],
                "stderr_tail": "suppressed_secret_bearing_task_output" if supplied_secrets else err[-1200:],
                "error": "" if ok else (str(receipt.get("error") or "managed_data_task_failed") if receipt else "managed_data_receipt_missing"),
            }
        finally:
            if sftp is not None:
                for remote_path in remote_secret_paths:
                    try:
                        sftp.remove(remote_path)
                    except OSError:
                        pass
                try:
                    sftp.close()
                except Exception:
                    pass
            client.close()

    def execute_solution(
        self,
        *,
        solution_id: str,
        script_path: str | Path,
        data_dir: str | Path | None = None,
        persistent_data_root: str | None = None,
        expected_manifest_sha256: str = "",
        secret_files: dict[str, bytes] | None = None,
    ) -> HpcJobResult:
        solution_id = _validate_component(solution_id, "solution_id")
        script_path = Path(script_path)
        local_data_dir = Path(data_dir) if data_dir is not None else None
        persistent_root = self._persistent_competition_root(persistent_data_root) if persistent_data_root else ""
        manifest_sha256 = str(expected_manifest_sha256 or "").casefold()
        remote_solution = _validate_remote(posixpath.join(self.remote_run_dir, "solutions", solution_id))
        attempt_id = uuid.uuid4().hex
        remote_attempt = _validate_remote(posixpath.join(remote_solution, "attempts", attempt_id))
        remote_output = _validate_remote(posixpath.join(remote_attempt, "output"))
        remote_work = _validate_remote(posixpath.join(remote_attempt, "work"))
        local_output = self.local_run_dir / "solutions" / solution_id / "attempts" / attempt_id / "output"
        client = None
        secret_sftp = None
        remote_secret_paths: list[str] = []
        supplied_secrets = dict(secret_files or {})
        job_result = None
        self.last_solution_secret_cleanup_ok = not bool(supplied_secrets)
        try:
            if len(supplied_secrets) > 4 or any(not _SECRET_ENV_NAME.fullmatch(name) or not isinstance(payload, bytes) or not payload or len(payload) > 64 * 1024 for name, payload in supplied_secrets.items()):
                raise ValueError("private_solution_input_invalid")
            if script_path.is_symlink() or not script_path.is_file():
                raise FileNotFoundError(f"solution script file not found: {script_path}")
            if bool(local_data_dir) == bool(persistent_root):
                raise ValueError("exactly one local or persistent competition data source is required")
            if persistent_root:
                if re.fullmatch(r"[a-f0-9]{64}", manifest_sha256) is None:
                    raise ValueError("persistent competition data requires a valid manifest SHA-256")
            elif local_data_dir is None or local_data_dir.is_symlink() or not local_data_dir.is_dir():
                raise FileNotFoundError(f"data directory not found: {local_data_dir}")

            bundle_files: dict[str, Path] = {"work/train_gpu.py": script_path}
            input_count = 0
            input_bytes = 0
            if local_data_dir is not None:
                for path in sorted(local_data_dir.rglob("*"), key=lambda value: value.as_posix().casefold()):
                    if path.is_symlink():
                        raise ValueError(f"input directory contains a symbolic link: {path.name}")
                    if not path.is_file():
                        continue
                    relative = _validate_relative_path(path.relative_to(local_data_dir).as_posix())
                    input_count += 1
                    input_bytes += path.stat().st_size
                    if input_count > _MAX_SOLUTION_INPUT_FILES:
                        raise ValueError("input directory exceeds the bounded file count")
                    if input_bytes > _MAX_SOLUTION_INPUT_BYTES:
                        raise ValueError("input directory exceeds the bounded byte limit")
                    bundle_files[f"inputs/{relative}"] = path
                if input_count == 0:
                    raise FileNotFoundError(f"input directory contains no regular files: {local_data_dir}")

            self.prepare_solution_environment(script_path)
            if self.progress_callback and not self.trusted_progress:
                self.progress_callback(
                    source="executor", work_kind="preparing", phase="staging_solution",
                    worker_state="unverified", detail="Dependencies verified; staging the bounded solution",
                )

            fingerprint_source = "".join(
                f"{relative}\0{_sha256(path)}\n"
                for relative, path in sorted(bundle_files.items())
            )
            if persistent_root:
                fingerprint_source += f"persistent_data_root\0{persistent_root}\nmanifest_sha256\0{manifest_sha256}\n"
            for name, payload in sorted(supplied_secrets.items()):
                fingerprint_source += f"private_input\0{name}\0{hashlib.sha256(payload).hexdigest()}\n"
            bundle_fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
            remote_subdir = f"solutions/{solution_id}/bundle_{bundle_fingerprint[:16]}"
            stage = self.stage_bundle(bundle_files, remote_subdir=remote_subdir)
            remote_bundle = _validate_remote(str(stage["remote_dir"]))
            remote_inputs = persistent_root or _validate_remote(posixpath.join(remote_bundle, "inputs"))
            remote_script = _validate_remote(posixpath.join(remote_bundle, "work", "train_gpu.py"))
            remote_compat_output = _validate_remote(posixpath.join(remote_work, "outputs"))
            local_output.mkdir(parents=True, exist_ok=True)
            client = self._connector()
            secret_environment = ""
            secret_trap = ""
            if supplied_secrets:
                import stat
                secret_sftp = client.open_sftp()
                secret_root = _validate_remote(posixpath.join(remote_attempt, ".secrets"))
                current = ALLOWED_GPU_REMOTE_ROOT
                for component in posixpath.relpath(secret_root, ALLOWED_GPU_REMOTE_ROOT).split("/"):
                    current = _validate_remote(posixpath.join(current, component))
                    try:
                        attributes = secret_sftp.lstat(current)
                    except FileNotFoundError:
                        secret_sftp.mkdir(current)
                        secret_sftp.chmod(current, 0o700)
                        attributes = secret_sftp.lstat(current)
                    if not stat.S_ISDIR(attributes.st_mode) or stat.S_ISLNK(attributes.st_mode):
                        raise ValueError("private_solution_directory_rejected")
                secret_sftp.chmod(secret_root, 0o700)
                for index, (name, payload) in enumerate(sorted(supplied_secrets.items())):
                    remote_path = _validate_remote(posixpath.join(secret_root, f"secret-{index:02d}"))
                    remote_secret_paths.append(remote_path)
                    self._write_remote_bytes(secret_sftp, remote_path, payload, mode=0o600)
                    secret_environment += f"{name}={shlex.quote(remote_path)} "
                cleanup = "rm -f -- " + " ".join(shlex.quote(path) for path in remote_secret_paths)
                secret_trap = "trap " + shlex.quote(cleanup) + " EXIT; "
            q_bundle = shlex.quote(remote_bundle)
            q_output = shlex.quote(remote_output)
            q_work = shlex.quote(remote_work)
            q_home = shlex.quote(_validate_remote(posixpath.join(remote_work, "home")))
            q_userbase = shlex.quote(_validate_remote(posixpath.join(remote_work, "python-userbase")))
            q_pip_cache = shlex.quote(_validate_remote(posixpath.join(remote_work, "pip-cache")))
            q_xdg_cache = shlex.quote(_validate_remote(posixpath.join(remote_work, "xdg-cache")))
            q_inputs = shlex.quote(remote_inputs)
            q_script = shlex.quote(remote_script)
            q_compat_output = shlex.quote(remote_compat_output)
            manifest_guard = ""
            if persistent_root:
                remote_manifest = _validate_remote(posixpath.join(persistent_root, ".evomind", "data-manifest.json"))
                manifest_guard = (
                    "python3 -c 'import hashlib,pathlib,sys;"
                    "p=pathlib.Path(sys.argv[1]);"
                    "assert p.is_file(),\"managed competition manifest missing\";"
                    "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2],\"managed competition manifest mismatch\"' "
                    f"{shlex.quote(remote_manifest)} {shlex.quote(manifest_sha256)} && "
                )
            slot_lock = shlex.quote(_validate_remote(posixpath.join(ALLOWED_GPU_REMOTE_ROOT, ".evomind-gpu-slot.lock")))
            environment_copy = f"cp {shlex.quote(self.solution_environment_manifest)} {q_output}/environment.lock.json && " if self.solution_environment_manifest else ""
            execution_seconds = self.remaining_seconds(self.timeout_seconds, margin=60)
            command = (
                f"{secret_trap}{manifest_guard}"
                f"mkdir -p {q_output} {q_work} {q_home} {q_userbase} {q_pip_cache} {q_xdg_cache} && "
                f"{environment_copy}cd {q_work} && flock -n -E 75 {slot_lock} timeout --kill-after=10 {execution_seconds} env "
                f"PYTHONPATH={shlex.quote(self.remote_solution_deps)}:{shlex.quote(self.remote_python_deps)} "
                f"HOME={q_home} PYTHONUSERBASE={q_userbase} PIP_CACHE_DIR={q_pip_cache} XDG_CACHE_HOME={q_xdg_cache} "
                "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
                f"EVOMIND_RUN_ID={shlex.quote(self.run_id)} "
                f"EVOMIND_INPUT_MANIFEST_SHA256={shlex.quote(manifest_sha256)} "
                f"EVOMIND_TASK_ROOT={q_bundle} EVOMIND_INPUT_DIR={q_inputs} "
                f"EVOMIND_OUTPUT_DIR={q_output} EVOMIND_WORK_DIR={q_work} "
                f"{secret_environment}CUDA_VISIBLE_DEVICES=0 python3 -u {q_script} --data-dir {q_inputs} --out-dir {q_output}; "
                "rc=$?; "
                f"if test -d {q_compat_output}; then cp -a {q_compat_output}/. {q_output}/; fi; "
                f"if test -f {q_work}/execution-summary.json; then cp {q_work}/execution-summary.json {q_output}/execution-summary.json; fi; "
                "exit $rc"
            )
            observation_timeout = self.remaining_seconds(self.timeout_seconds + 60)
            try:
                if self.progress_callback and not self.trusted_progress:
                    self.progress_callback(
                        source="executor", work_kind="executing", phase="executing_solution",
                        worker_state="unverified", detail="Executing the bounded solution; model results are not yet verified",
                    )
                rc, out, err = self._exec_observed(client, command, timeout=observation_timeout)
                if rc < 0:
                    raise RuntimeError("remote_exit_status_missing")
            except Exception as exc:
                raise TimeoutError("remote_settlement_unconfirmed") from exc
            training_log = local_output / "training.log"
            log_text = "suppressed_secret_bearing_task_output\n" if supplied_secrets else out + ("\n[stderr]\n" + err if err else "")
            training_log.write_text(log_text, encoding="utf-8")
            artifacts = [
                {
                    "path": str(training_log),
                    "relative_path": "training.log",
                    "sha256": _sha256(training_log),
                    "bytes": training_log.stat().st_size,
                }
            ]
            sftp = client.open_sftp()
            try:
                try:
                    artifacts.extend(self._download_solution_outputs(sftp, remote_output, local_output))
                except OSError:
                    pass
            finally:
                sftp.close()
            output_count = len(artifacts) - 1
            success = rc == 0 and output_count > 0
            evidence_error = "" if success else "solution produced no downloadable output artifacts"
            primary_error = evidence_error if rc == 0 else (err or out or evidence_error)
            job_result = HpcJobResult(
                status="completed" if success else "failed",
                run_id=self.run_id,
                solution_id=solution_id,
                remote_dir=remote_solution,
                exit_code=rc,
                stdout_tail="suppressed_secret_bearing_task_output" if supplied_secrets else out[-1200:],
                stderr_tail="suppressed_secret_bearing_task_output" if supplied_secrets else err[-1200:],
                local_artifacts=artifacts,
                failure_type="" if success else self.classify_failure(exit_code=rc, error=primary_error),
                error=evidence_error or ("secret_bearing_solution_failed" if supplied_secrets and rc else err[-1200:] if rc else ""),
            )
            return job_result
        except Exception as exc:
            detail = "remote_settlement_unconfirmed" if "remote_settlement_unconfirmed" in str(exc) else "private_solution_failed"
            message = f"{type(exc).__name__}: {detail if supplied_secrets else exc}"
            job_result = HpcJobResult(
                status="failed",
                run_id=self.run_id,
                solution_id=solution_id,
                remote_dir=remote_solution,
                exit_code=-1,
                failure_type=self.classify_failure(error=message),
                error=message,
            )
            return job_result
        finally:
            if secret_sftp is not None:
                cleanup_ok = True
                for remote_path in remote_secret_paths:
                    try:
                        secret_sftp.remove(remote_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        cleanup_ok = False
                self.last_solution_secret_cleanup_ok = cleanup_ok
                secret_sftp.close()
                if not cleanup_ok and job_result is not None:
                    job_result.status = "failed"
                    job_result.error = (job_result.error + ";managed_secret_cleanup_failed").lstrip(";")
                    job_result.failure_type = "private_input_cleanup"
                    if job_result.exit_code == 0:
                        job_result.exit_code = -1
            if client is not None:
                client.close()


__all__ = ["HpcJobResult", "HpcRuntime"]
