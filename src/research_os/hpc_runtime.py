"""Controlled HPC runtime shared by EvoMind agents.

All remote state is confined to the dedicated allocation directory. This class
does not reason or select experiments; it probes, stages, executes, cancels, and
collects artifacts for the Supervisor.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent_workstation.server.core.gpu_credentials import (
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
)

_PYTHON_DEV_DEB = "libpython3.10-dev_3.10.12-1~22.04.16_amd64.deb"
_PYTHON_DEV_URL = (
    "http://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/"
    + _PYTHON_DEV_DEB
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError(f"invalid {label}")
    return value


def _validate_remote(path: str) -> str:
    normalized = posixpath.normpath(path)
    if normalized != ALLOWED_GPU_REMOTE_ROOT and not normalized.startswith(ALLOWED_GPU_REMOTE_ROOT + "/"):
        raise ValueError("remote path escapes the dedicated EvoMind root")
    return normalized


def _validate_relative_path(value: str) -> str:
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if not normalized or normalized in {".", ".."} or normalized.startswith("../") or posixpath.isabs(normalized):
        raise ValueError("invalid relative bundle path")
    return normalized


@dataclass
class HpcProbe:
    status: str
    generated_at: str
    user: str = ""
    hostname: str = ""
    pwd: str = ""
    python_version: str = ""
    gpu_inventory: list[dict[str, str]] = field(default_factory=list)
    torch: dict[str, Any] = field(default_factory=dict)
    packages: dict[str, Any] = field(default_factory=dict)
    failure_type: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HpcJobResult:
    status: str
    run_id: str
    solution_id: str
    remote_dir: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    local_artifacts: list[dict[str, Any]] = field(default_factory=list)
    failure_type: str = ""
    error: str = ""


class HpcRuntime:
    def __init__(self, *, run_id: str, local_run_dir: str | Path, timeout_seconds: int = 1200, connector=None) -> None:
        self.run_id = _validate_component(run_id, "run_id")
        self.local_run_dir = Path(local_run_dir)
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.remote_run_dir = _validate_remote(posixpath.join(ALLOWED_GPU_REMOTE_ROOT, "evomind_runs", self.run_id))
        self.remote_runtime_dir = _validate_remote(posixpath.join(self.remote_run_dir, "runtime"))
        self.remote_runtime_home = _validate_remote(posixpath.join(self.remote_runtime_dir, "home"))
        self.remote_python_deps = _validate_remote(posixpath.join(self.remote_run_dir, "runtime", "python_deps"))
        self.remote_python_headers = _validate_remote(posixpath.join(self.remote_runtime_dir, "python-dev"))
        self.remote_venv = _validate_remote(posixpath.join(self.remote_run_dir, "runtime", "llm_venv"))
        self.remote_cache = _validate_remote(posixpath.join(self.remote_run_dir, "cache"))
        self.remote_triton_cache = _validate_remote(posixpath.join(self.remote_cache, "triton"))
        self.remote_torch_extensions = _validate_remote(posixpath.join(self.remote_cache, "torch_extensions"))
        self._connector = connector or connect_ssh
        self._stage_lock = threading.RLock()
        self._staged_data_hash = ""
        self._llm_poll_interval_seconds = 5.0

    @staticmethod
    def _exec(client, command: str, *, timeout: int) -> tuple[int, str, str]:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    @staticmethod
    def classify_failure(*, exit_code: int | None = None, error: str = "") -> str:
        low = error.lower()
        if any(term in low for term in ("authentication", "auth fail", "permission denied")):
            return "connection_auth"
        if any(term in low for term in ("socks", "connect", "ssh", "banner", "socket", "network", "eoferror")):
            return "connection"
        if exit_code in {124, 143} or "timeout" in low or "timed out" in low:
            return "timeout"
        if exit_code in {137, 9} or "out of memory" in low or "cuda oom" in low:
            return "oom"
        if "no module named" in low or "modulenotfounderror" in low:
            return "dependency"
        if "cuda" in low or "gpu" in low or "nvidia-smi" in low:
            return "resource"
        if "artifact" in low or "metrics.json" in low or "oof" in low:
            return "evidence"
        return "code" if exit_code not in {None, 0} else "unknown"

    def _llm_environment(self, python_deps: str) -> str:
        include_root = posixpath.join(self.remote_python_headers, "usr", "include")
        return " ".join([
            f"HOME={shlex.quote(self.remote_runtime_home)}",
            f"XDG_CACHE_HOME={shlex.quote(self.remote_cache)}",
            f"HF_HOME={shlex.quote(posixpath.join(self.remote_cache, 'huggingface'))}",
            f"HF_DATASETS_CACHE={shlex.quote(posixpath.join(self.remote_cache, 'datasets'))}",
            f"TRANSFORMERS_CACHE={shlex.quote(posixpath.join(self.remote_cache, 'huggingface'))}",
            f"PIP_CACHE_DIR={shlex.quote(posixpath.join(self.remote_cache, 'pip'))}",
            f"TRITON_CACHE_DIR={shlex.quote(self.remote_triton_cache)}",
            f"TORCH_EXTENSIONS_DIR={shlex.quote(self.remote_torch_extensions)}",
            f"CPATH={shlex.quote(include_root)}:{shlex.quote(posixpath.join(include_root, 'python3.10'))}",
            f"PYTHONPATH={python_deps}:$HOST_USER_SITE",
            "TOKENIZERS_PARALLELISM=false",
            "CUDA_VISIBLE_DEVICES=0",
        ])

    def _llm_python_prefix(self, python_deps: str) -> str:
        return (
            "HOST_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())') && "
            f"env {self._llm_environment(python_deps)} python3"
        )

    def probe(self, *, use_run_environment: bool = False) -> HpcProbe:
        client = None
        try:
            client = self._connector()
            if use_run_environment:
                python_command = self._llm_python_prefix(shlex.quote(self.remote_python_deps))
                python_env = ""
            else:
                python_command = "python3"
                python_env = ""
            command = " && ".join([
                "whoami",
                "hostname",
                "pwd",
                f"test -d {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)}",
                f"test -w {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)}",
                "python3 --version",
                "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits",
                f"{python_env}{python_command} - <<'PY'\nimport importlib\nimport json\npayload={{}}\npackages={{}}\nfor name in ('dill','joblib','sklearn'):\n try:\n  module=importlib.import_module(name)\n  packages[name]={{'ok':True,'version':getattr(module,'__version__','')}}\n except Exception as exc:\n  packages[name]={{'ok':False,'error':type(exc).__name__+': '+str(exc)}}\ntry:\n import torch\n payload={{'version':torch.__version__,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}}\nexcept Exception as exc:\n payload={{'error':type(exc).__name__+': '+str(exc)}}\nprint('EVOMIND_RUNTIME='+json.dumps({{'torch':payload,'packages':packages}}))\nPY",
            ])
            rc, out, err = self._exec(client, command, timeout=90)
            if rc != 0:
                raise RuntimeError(f"probe exit={rc}: {err[-600:] or out[-600:]}")
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            runtime_line = next((line for line in lines if line.startswith("EVOMIND_RUNTIME=")), "")
            runtime_payload = json.loads(runtime_line.split("=", 1)[1]) if runtime_line else {}
            torch_payload = runtime_payload.get("torch") if isinstance(runtime_payload.get("torch"), dict) else {}
            packages = runtime_payload.get("packages") if isinstance(runtime_payload.get("packages"), dict) else {}
            prefix = lines[:3]
            python_line = next((line for line in lines if line.startswith("Python ")), "")
            gpu_lines = [line for line in lines if line.count(",") >= 3 and not line.startswith("EVOMIND_")]
            inventory = []
            for line in gpu_lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    inventory.append({"index": parts[0], "name": parts[1], "memory_total_mb": parts[2], "driver_version": parts[3]})
            packages_ready = all(bool((packages.get(name) or {}).get("ok")) for name in ("dill", "joblib", "sklearn"))
            ready = bool(inventory) and bool(torch_payload.get("cuda_available")) and packages_ready
            return HpcProbe(
                status="passed" if ready else "blocked",
                generated_at=_now(),
                user=prefix[0] if len(prefix) > 0 else "",
                hostname=prefix[1] if len(prefix) > 1 else "",
                pwd=prefix[2] if len(prefix) > 2 else "",
                python_version=python_line,
                gpu_inventory=inventory,
                torch=torch_payload,
                packages=packages,
                failure_type="" if ready else "resource",
                error="" if ready else "CUDA inventory, torch.cuda, or required runtime packages are not ready",
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return HpcProbe(status="blocked", generated_at=_now(), failure_type=self.classify_failure(error=message), error=message)
        finally:
            if client is not None:
                client.close()

    def prepare_python_environment(self) -> dict[str, Any]:
        """Repair Python dependencies inside this run only, never in user/system paths."""
        client = self._connector()
        try:
            deps = shlex.quote(self.remote_python_deps)
            runtime_dir = shlex.quote(posixpath.dirname(self.remote_python_deps))
            marker = shlex.quote(posixpath.join(posixpath.dirname(self.remote_python_deps), "ready.json"))
            verify = (
                f"PYTHONPATH={deps} python3 -c \"import dill,joblib,sklearn,sympy,torch; "
                "assert hasattr(dill,'extend'); assert torch.cuda.is_available(); "
                "m=torch.nn.Linear(2,1); torch.optim.AdamW(m.parameters(),lr=0.01)\""
            )
            install = (
                f"mkdir -p {runtime_dir} {deps} && "
                f"python3 -m pip install --disable-pip-version-check --no-input --no-deps --target {deps} "
                "dill==0.3.9 joblib==1.4.2 scikit-learn==1.6.1 "
                "mpmath==1.3.0 sympy==1.13.3 && "
                f"{verify} && printf '%s\\n' "
                f"'{{\"schema\":\"evomind.hpc_runtime.v1\",\"status\":\"ready\"}}' > {marker}"
            )
            command = f"cd {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)} && ({verify} || ({install}))"
            rc, out, err = self._exec(client, command, timeout=300)
            if rc != 0:
                raise RuntimeError(f"runtime dependency repair exit={rc}: {(err or out)[-1200:]}")
            return {
                "schema": "evomind.hpc_runtime_environment.v1",
                "status": "ready",
                "remote_python_deps": self.remote_python_deps,
                "packages": {
                    "dill": "0.3.9",
                    "joblib": "1.4.2",
                    "scikit-learn": "1.6.1",
                    "mpmath": "1.3.0",
                    "sympy": "1.13.3",
                },
            }
        finally:
            client.close()

    def prepare_llm_environment(self) -> dict[str, Any]:
        """Create pinned QLoRA dependencies under the run without requiring venv."""
        client = self._connector()
        versions = {
            "dill": "0.3.9",
            "joblib": "1.4.2",
            "transformers": "4.46.3",
            "peft": "0.13.2",
            "bitsandbytes": "0.49.2",
            "accelerate": "1.1.1",
            "safetensors": "0.4.5",
            "tokenizers": "0.20.3",
            "huggingface-hub": "0.26.2",
            "filelock": "3.16.1",
            "fsspec": "2024.9.0",
            "packaging": "24.2",
            "PyYAML": "6.0.2",
            "regex": "2024.11.6",
            "requests": "2.32.3",
            "tqdm": "4.67.1",
            "psutil": "6.1.0",
            "urllib3": "2.2.3",
            "certifi": "2024.8.30",
            "charset-normalizer": "3.4.0",
            "idna": "3.10",
            "typing-extensions": "4.12.2",
            "Jinja2": "3.1.4",
            "MarkupSafe": "3.0.2",
            "sympy": "1.13.3",
            "mpmath": "1.3.0",
            "numpy": "1.26.4",
        }
        try:
            q_deps = shlex.quote(self.remote_python_deps)
            staging_deps = shlex.quote(self.remote_python_deps + ".next")
            q_cache = shlex.quote(posixpath.join(self.remote_cache, "pip"))
            q_runtime = shlex.quote(self.remote_runtime_dir)
            q_home = shlex.quote(self.remote_runtime_home)
            q_triton_cache = shlex.quote(self.remote_triton_cache)
            q_torch_extensions = shlex.quote(self.remote_torch_extensions)
            q_headers = shlex.quote(self.remote_python_headers)
            q_packages = shlex.quote(posixpath.join(self.remote_runtime_dir, "packages"))
            remote_deb = posixpath.join(self.remote_runtime_dir, "packages", _PYTHON_DEV_DEB)
            q_deb = shlex.quote(remote_deb)
            q_marker = shlex.quote(posixpath.join(posixpath.dirname(self.remote_python_deps), "llm_ready.json"))
            critical_versions = {
                name: versions[name]
                for name in ("accelerate", "bitsandbytes", "dill", "numpy", "peft", "sympy", "transformers")
            }

            def verification_command(target: str) -> str:
                return (
                    f"{self._llm_python_prefix(target)} -c \"import importlib.metadata as md; "
                    "import accelerate,bitsandbytes,numpy,peft,sympy,torch,transformers; "
                    f"expected={critical_versions!r}; "
                    "actual={name:md.version(name) for name in expected}; "
                    "assert actual==expected,(actual,expected); assert torch.cuda.is_available(); "
                    "print(torch.__version__, transformers.__version__, peft.__version__, numpy.__version__)\""
                )

            verify = verification_command(q_deps)
            pinned = " ".join(f"{name}=={version}" for name, version in versions.items())
            staging_verify = verification_command(staging_deps)
            headers_ready = (
                f"test -s {q_headers}/usr/include/python3.10/Python.h && "
                f"test -s {q_headers}/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h"
            )
            install_headers = (
                f"mkdir -p {q_packages} {q_headers} && "
                f"test -s {q_deb} || curl -fL --retry 3 --connect-timeout 20 "
                f"-o {q_deb} {shlex.quote(_PYTHON_DEV_URL)} && "
                f"rm -rf {q_headers} && mkdir -p {q_headers} && dpkg-deb -x {q_deb} {q_headers}"
            )
            install = (
                f"mkdir -p {q_runtime} {q_home} {q_cache} {q_triton_cache} {q_torch_extensions} "
                f"{shlex.quote(self.remote_cache)} && "
                f"rm -rf {staging_deps} && mkdir -p {staging_deps} && "
                f"PIP_CACHE_DIR={q_cache} python3 -m pip install --disable-pip-version-check --no-input "
                f"--no-deps --upgrade --target {staging_deps} {pinned} && "
                f"{staging_verify} && rm -rf {q_deps} && mv {staging_deps} {q_deps} && "
                f"{verify} && printf '%s\\n' "
                f"'{{\"schema\":\"evomind.llm_runtime.v1\",\"status\":\"ready\"}}' > {q_marker}"
            )
            command = (
                f"cd {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)} && "
                f"mkdir -p {q_runtime} {q_home} {q_triton_cache} {q_torch_extensions} && "
                f"({headers_ready} || ({install_headers})) && ({verify} || ({install}))"
            )
            rc, out, err = self._exec(client, command, timeout=900)
            if rc != 0:
                combined = (out + ("\n[stderr]\n" + err if err else "")).strip()
                raise RuntimeError(f"LLM runtime dependency repair exit={rc}: {combined[-2400:]}")
            return {
                "schema": "evomind.llm_runtime_environment.v1",
                "status": "ready",
                "python": "python3",
                "remote_python_deps": self.remote_python_deps,
                "remote_python_headers": self.remote_python_headers,
                "remote_runtime_home": self.remote_runtime_home,
                "remote_cache": self.remote_cache,
                "packages": versions,
                "declared_not_required_by_training_script": {
                    "trl": "0.12.2",
                    "datasets": "3.1.0",
                },
                "verification_tail": out[-800:],
            }
        finally:
            client.close()

    def _mkdirs(self, sftp, path: str) -> None:
        current = "/"
        for component in path.strip("/").split("/"):
            current = posixpath.join(current, component)
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)

    def stage_data(self, data_dir: str | Path) -> dict[str, Any]:
        data_dir = Path(data_dir)
        required = [data_dir / "train.csv", data_dir / "test.csv", data_dir / "sample_submission.csv"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        combined = hashlib.sha256("".join(_sha256(path) for path in required).encode("ascii")).hexdigest()
        with self._stage_lock:
            if self._staged_data_hash == combined:
                return {"remote_data_dir": f"{self.remote_run_dir}/data", "data_hash": combined, "reused": True}
            client = self._connector()
            try:
                sftp = client.open_sftp()
                try:
                    remote_data = _validate_remote(posixpath.join(self.remote_run_dir, "data"))
                    self._mkdirs(sftp, remote_data)
                    for path in required:
                        sftp.put(str(path), posixpath.join(remote_data, path.name))
                finally:
                    sftp.close()
            finally:
                client.close()
            self._staged_data_hash = combined
            return {"remote_data_dir": f"{self.remote_run_dir}/data", "data_hash": combined, "reused": False}

    def stage_bundle(self, files: dict[str, str | Path], *, remote_subdir: str = "bundle") -> dict[str, Any]:
        """Stage arbitrary files while preserving validated relative paths."""
        safe_subdir = _validate_relative_path(remote_subdir)
        normalized: list[tuple[str, Path]] = []
        for relative, local_value in sorted(files.items()):
            safe_relative = _validate_relative_path(relative)
            local_path = Path(local_value)
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            normalized.append((safe_relative, local_path))
        if not normalized:
            raise ValueError("bundle must contain at least one file")
        hashes = {relative: _sha256(path) for relative, path in normalized}
        combined = hashlib.sha256(
            "".join(f"{relative}\0{hashes[relative]}\n" for relative, _path in normalized).encode("utf-8")
        ).hexdigest()
        remote_root = _validate_remote(posixpath.join(self.remote_run_dir, safe_subdir))
        client = self._connector()
        try:
            sftp = client.open_sftp()
            try:
                self._mkdirs(sftp, remote_root)
                for relative, local_path in normalized:
                    remote_path = _validate_remote(posixpath.join(remote_root, relative))
                    self._mkdirs(sftp, posixpath.dirname(remote_path))
                    sftp.put(str(local_path), remote_path)
            finally:
                sftp.close()
        finally:
            client.close()
        return {
            "schema": "evomind.hpc_bundle.v1",
            "remote_dir": remote_root,
            "combined_sha256": combined,
            "files": [{"path": relative, "sha256": hashes[relative]} for relative, _path in normalized],
        }

    def _download_tree(
        self,
        sftp,
        remote_root: str,
        local_root: Path,
        *,
        skip_directories: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        skipped = skip_directories or set()
        local_root.mkdir(parents=True, exist_ok=True)

        def walk(remote_dir: str, local_dir: Path) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(remote_dir):
                remote_path = _validate_remote(posixpath.join(remote_dir, entry.filename))
                local_path = local_dir / entry.filename
                if entry.st_mode & 0o170000 == 0o040000:
                    if entry.filename in skipped:
                        continue
                    walk(remote_path, local_path)
                    continue
                sftp.get(remote_path, str(local_path))
                artifacts.append({
                    "path": str(local_path),
                    "sha256": _sha256(local_path),
                    "bytes": local_path.stat().st_size,
                })

        walk(remote_root, local_root)
        return artifacts

    def _connect_resilient(self, *, attempts: int = 3):
        last_error: Exception | None = None
        for attempt in range(max(1, attempts)):
            try:
                return self._connector()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        assert last_error is not None
        raise last_error

    def _launch_or_attach_llm_job(
        self,
        client,
        *,
        remote_bundle: str,
        remote_output: str,
        base_model: str,
        execution_id: str | None = None,
    ) -> tuple[str, str, str]:
        job_dir = _validate_remote(
            posixpath.join(
                self.remote_runtime_dir,
                "llm_job" if execution_id is None else f"llm_jobs/{_validate_component(execution_id, 'execution_id')}",
            )
        )
        pid_path = posixpath.join(job_dir, "pid")
        exit_path = posixpath.join(job_dir, "exit_code")
        log_path = posixpath.join(job_dir, "training.log")
        script_path = posixpath.join(job_dir, "run.sh")
        python_environment = self._llm_environment(shlex.quote(self.remote_python_deps))
        worker_script = "\n".join([
            "#!/usr/bin/env bash",
            "set +e",
            "HOST_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())')",
            f"cd {shlex.quote(remote_bundle)}",
            f"timeout {self.timeout_seconds} env {python_environment} python3 -u code/train_qlora.py "
            f"--data-dir data --config config/qlora_config.json --out-dir {shlex.quote(remote_output)} "
            f"--base-model {shlex.quote(base_model)} --run-id {shlex.quote(self.run_id)}",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}.tmp",
            f"mv {shlex.quote(exit_path)}.tmp {shlex.quote(exit_path)}",
            "exit \"$rc\"",
            "",
        ])
        launch_command = "\n".join([
            "set -eu",
            f"mkdir -p {shlex.quote(job_dir)} {shlex.quote(self.remote_runtime_home)} "
            f"{shlex.quote(self.remote_cache)} {shlex.quote(self.remote_triton_cache)} "
            f"{shlex.quote(self.remote_torch_extensions)}",
            f"if test -s {shlex.quote(exit_path)} && test \"$(cat {shlex.quote(exit_path)})\" = 0; then",
            "  echo EVOMIND_LAUNCH_STATE=finished",
            "  exit 0",
            "fi",
            f"if test -s {shlex.quote(pid_path)} && kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; then",
            "  echo EVOMIND_LAUNCH_STATE=attached",
            "  exit 0",
            "fi",
            f"if test -s {shlex.quote(exit_path)}; then",
            "  stamp=$(date -u +%Y%m%dT%H%M%SZ)",
            f"  test ! -f {shlex.quote(log_path)} || mv {shlex.quote(log_path)} {shlex.quote(log_path)}.failed.$stamp",
            "fi",
            f"rm -f {shlex.quote(pid_path)} {shlex.quote(exit_path)}",
            f"rm -rf {shlex.quote(remote_output)} && mkdir -p {shlex.quote(remote_output)}",
            f"cat > {shlex.quote(script_path)} <<'EVOMIND_LLM_JOB'",
            worker_script.rstrip("\n"),
            "EVOMIND_LLM_JOB",
            f"chmod 700 {shlex.quote(script_path)}",
            f"nohup setsid bash {shlex.quote(script_path)} > {shlex.quote(log_path)} 2>&1 < /dev/null &",
            "job_pid=$!",
            f"printf '%s\\n' \"$job_pid\" > {shlex.quote(pid_path)}.tmp",
            f"mv {shlex.quote(pid_path)}.tmp {shlex.quote(pid_path)}",
            "echo EVOMIND_LAUNCH_STATE=started",
        ])
        rc, out, err = self._exec(client, launch_command, timeout=60)
        if rc != 0:
            raise RuntimeError(f"remote LLM launch failed: {(err or out)[-1200:]}")
        state_line = next((line for line in out.splitlines() if line.startswith("EVOMIND_LAUNCH_STATE=")), "")
        state = state_line.split("=", 1)[1].strip() if state_line else "unknown"
        if state not in {"started", "attached", "finished"}:
            raise RuntimeError(f"remote LLM launch returned an invalid state: {out[-600:]}")
        return state, job_dir, log_path

    def _poll_llm_job(self, client, *, job_dir: str) -> tuple[Any, int, str, str]:
        pid_path = posixpath.join(job_dir, "pid")
        exit_path = posixpath.join(job_dir, "exit_code")
        log_path = posixpath.join(job_dir, "training.log")
        poll_command = "\n".join([
            f"if test -s {shlex.quote(exit_path)}; then",
            "  echo EVOMIND_JOB_STATE=finished",
            f"  echo EVOMIND_EXIT_CODE=$(cat {shlex.quote(exit_path)})",
            f"elif test -s {shlex.quote(pid_path)} && kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; then",
            "  echo EVOMIND_JOB_STATE=running",
            "else",
            "  echo EVOMIND_JOB_STATE=lost",
            "fi",
            "echo EVOMIND_LOG_TAIL_BEGIN",
            f"tail -n 40 {shlex.quote(log_path)} 2>/dev/null || true",
            "echo EVOMIND_LOG_TAIL_END",
        ])
        deadline = time.monotonic() + self.timeout_seconds + 120
        latest_tail = ""
        active_client = client
        while time.monotonic() < deadline:
            try:
                if active_client is None:
                    active_client = self._connect_resilient()
                rc, out, err = self._exec(active_client, poll_command, timeout=45)
                if rc != 0:
                    raise RuntimeError(err or out or f"poll exit={rc}")
                state_line = next((line for line in out.splitlines() if line.startswith("EVOMIND_JOB_STATE=")), "")
                state = state_line.split("=", 1)[1].strip() if state_line else "unknown"
                if "EVOMIND_LOG_TAIL_BEGIN\n" in out and "\nEVOMIND_LOG_TAIL_END" in out:
                    latest_tail = out.split("EVOMIND_LOG_TAIL_BEGIN\n", 1)[1].split("\nEVOMIND_LOG_TAIL_END", 1)[0]
                if state == "finished":
                    exit_line = next((line for line in out.splitlines() if line.startswith("EVOMIND_EXIT_CODE=")), "")
                    exit_code = int(exit_line.split("=", 1)[1].strip()) if exit_line else 125
                    return active_client, exit_code, latest_tail, ""
                if state == "lost":
                    return active_client, 125, latest_tail, "remote LLM process exited without an atomic exit marker"
                if state != "running":
                    raise RuntimeError(f"invalid remote LLM poll state: {out[-600:]}")
            except Exception:
                if active_client is not None:
                    try:
                        active_client.close()
                    except Exception:
                        pass
                active_client = None
            time.sleep(self._llm_poll_interval_seconds)
        return active_client, 124, latest_tail, "remote LLM job exceeded its reconnectable execution deadline"

    def _collect_llm_output(
        self,
        client,
        *,
        remote_output: str,
        remote_log: str,
        local_output: Path,
    ) -> tuple[Any, list[dict[str, Any]]]:
        last_error: Exception | None = None
        active_client = client
        for attempt in range(3):
            try:
                if active_client is None:
                    active_client = self._connect_resilient()
                sftp = active_client.open_sftp()
                try:
                    artifacts: list[dict[str, Any]] = []
                    local_log = local_output / "training.log"
                    try:
                        sftp.get(remote_log, str(local_log))
                        artifacts.append({"path": str(local_log), "sha256": _sha256(local_log), "bytes": local_log.stat().st_size})
                    except OSError:
                        pass
                    try:
                        sftp.stat(remote_output)
                        # Intermediate Trainer checkpoints remain inside the dedicated
                        # remote run directory.  The promoted adapter, tokenizer,
                        # telemetry and evaluation evidence are the release artifacts;
                        # copying every optimizer checkpoint adds gigabytes of transfer
                        # latency without strengthening the review contract.
                        artifacts.extend(
                            self._download_tree(
                                sftp,
                                remote_output,
                                local_output,
                                skip_directories={"checkpoints"},
                            )
                        )
                    except OSError:
                        pass
                    return active_client, artifacts
                finally:
                    sftp.close()
            except Exception as exc:
                last_error = exc
                if active_client is not None:
                    try:
                        active_client.close()
                    except Exception:
                        pass
                active_client = None
                if attempt < 2:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        assert last_error is not None
        raise last_error

    def execute_llm_finetune(
        self,
        *,
        script_path: str | Path,
        data_files: dict[str, str | Path],
        config_path: str | Path,
        base_model: str,
        execution_id: str | None = None,
    ) -> HpcJobResult:
        """Run one real QLoRA job and retrieve its complete evidence tree."""
        safe_execution_id = _validate_component(execution_id, "execution_id") if execution_id else None
        bundle_files = {"code/train_qlora.py": Path(script_path), "config/qlora_config.json": Path(config_path)}
        bundle_files.update({f"data/{_validate_relative_path(name)}": Path(path) for name, path in data_files.items()})
        remote_subdir = "llm_finetune" if safe_execution_id is None else f"llm_finetune/{safe_execution_id}"
        stage = self.stage_bundle(bundle_files, remote_subdir=remote_subdir)
        remote_bundle = str(stage["remote_dir"])
        remote_output = _validate_remote(
            posixpath.join(
                self.remote_run_dir,
                "llm_output" if safe_execution_id is None else f"llm_attempts/{safe_execution_id}/llm_output",
            )
        )
        local_output = self.local_run_dir / "llm_output"
        if local_output.exists():
            shutil.rmtree(local_output)
        local_output.mkdir(parents=True, exist_ok=True)
        client = None
        try:
            client = self._connect_resilient()
            _launch_state, job_dir, remote_log = self._launch_or_attach_llm_job(
                client,
                remote_bundle=remote_bundle,
                remote_output=remote_output,
                base_model=base_model,
                execution_id=safe_execution_id,
            )
            client, rc, log_tail, poll_error = self._poll_llm_job(client, job_dir=job_dir)
            client, artifacts = self._collect_llm_output(
                client,
                remote_output=remote_output,
                remote_log=remote_log,
                local_output=local_output,
            )
            required = {
                "metrics.json",
                "evaluation.json",
                "environment.json",
                "telemetry.jsonl",
                "adapter_reload.json",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
            }
            present = {
                Path(item["path"]).relative_to(local_output).as_posix()
                for item in artifacts
                if Path(item["path"]).is_relative_to(local_output)
            }
            missing = sorted(required - present)
            success = rc == 0 and not missing
            evidence_error = "" if success else f"missing LLM artifacts: {missing}"
            primary_error = (poll_error or log_tail[-1600:] or evidence_error) if rc != 0 else evidence_error
            return HpcJobResult(
                status="completed" if success else "failed",
                run_id=self.run_id,
                solution_id="qwen7b_qlora" if safe_execution_id is None else f"qwen7b_qlora_{safe_execution_id}",
                remote_dir=remote_output,
                exit_code=rc,
                stdout_tail=log_tail[-1600:],
                stderr_tail="",
                local_artifacts=artifacts,
                failure_type="" if success else self.classify_failure(exit_code=rc, error=primary_error),
                error=primary_error,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return HpcJobResult(
                status="failed",
                run_id=self.run_id,
                solution_id="qwen7b_qlora" if safe_execution_id is None else f"qwen7b_qlora_{safe_execution_id}",
                remote_dir=remote_output,
                exit_code=-1,
                failure_type=self.classify_failure(error=message),
                error=message,
            )
        finally:
            if client is not None:
                client.close()

    def execute_solution(self, *, solution_id: str, script_path: str | Path, data_dir: str | Path) -> HpcJobResult:
        solution_id = _validate_component(solution_id, "solution_id")
        script_path = Path(script_path)
        stage = self.stage_data(data_dir)
        remote_solution = _validate_remote(posixpath.join(self.remote_run_dir, "solutions", solution_id))
        remote_output = _validate_remote(posixpath.join(remote_solution, "output"))
        local_output = self.local_run_dir / "solutions" / solution_id / "output"
        local_output.mkdir(parents=True, exist_ok=True)
        client = None
        try:
            client = self._connector()
            sftp = client.open_sftp()
            try:
                self._mkdirs(sftp, remote_solution)
                sftp.put(str(script_path), posixpath.join(remote_solution, "solution.py"))
            finally:
                sftp.close()
            q_solution = shlex.quote(remote_solution)
            q_output = shlex.quote(remote_output)
            q_data = shlex.quote(str(stage["remote_data_dir"]))
            command = (
                f"cd {q_solution} && rm -rf {q_output} && mkdir -p {q_output} && "
                f"timeout {self.timeout_seconds} env PYTHONPATH={shlex.quote(self.remote_python_deps)} "
                f"python3 -u solution.py --data-dir {q_data} --out-dir {q_output}"
            )
            rc, out, err = self._exec(client, command, timeout=self.timeout_seconds + 60)
            (local_output / "training.log").write_text(out + ("\n[stderr]\n" + err if err else ""), encoding="utf-8")
            artifacts = [{
                "path": str(local_output / "training.log"),
                "sha256": _sha256(local_output / "training.log"),
                "bytes": (local_output / "training.log").stat().st_size,
            }]
            if rc == 0:
                sftp = client.open_sftp()
                try:
                    for name in ("metrics.json", "oof_predictions.csv", "submission.csv", "environment.json"):
                        remote_file = posixpath.join(remote_output, name)
                        local_file = local_output / name
                        try:
                            sftp.stat(remote_file)
                            sftp.get(remote_file, str(local_file))
                        except OSError:
                            continue
                        artifacts.append({"path": str(local_file), "sha256": _sha256(local_file), "bytes": local_file.stat().st_size})
                finally:
                    sftp.close()
            required_names = {"metrics.json", "oof_predictions.csv", "submission.csv", "environment.json"}
            present_names = {Path(item["path"]).name for item in artifacts}
            success = rc == 0 and required_names.issubset(present_names)
            evidence_error = "" if success else f"missing artifacts: {sorted(required_names - present_names)}"
            failure_type = "" if success else self.classify_failure(
                exit_code=rc,
                error=evidence_error if rc == 0 else (err or out or evidence_error),
            )
            return HpcJobResult(
                status="completed" if success else "failed",
                run_id=self.run_id,
                solution_id=solution_id,
                remote_dir=remote_solution,
                exit_code=rc,
                stdout_tail=out[-1200:],
                stderr_tail=err[-1200:],
                local_artifacts=artifacts,
                failure_type=failure_type,
                error=evidence_error or (err[-1200:] if rc else ""),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return HpcJobResult(
                status="failed",
                run_id=self.run_id,
                solution_id=solution_id,
                remote_dir=remote_solution,
                exit_code=-1,
                failure_type=self.classify_failure(error=message),
                error=message,
            )
        finally:
            if client is not None:
                client.close()

    def cancel(self, *, solution_id: str) -> bool:
        solution_id = _validate_component(solution_id, "solution_id")
        remote_solution = _validate_remote(posixpath.join(self.remote_run_dir, "solutions", solution_id))
        client = self._connector()
        try:
            marker = shlex.quote(posixpath.join(remote_solution, ".cancelled"))
            rc, _out, _err = self._exec(
                client,
                f"touch {marker} && pkill -f {shlex.quote(remote_solution + '/solution.py')} || true",
                timeout=30,
            )
            return rc == 0
        finally:
            client.close()


__all__ = ["HpcJobResult", "HpcProbe", "HpcRuntime"]
