"""GPU runner: executes generated solutions on the remote A40 via SSH.

Implements the same ``Runner`` protocol as ``LocalSubprocessRunner`` so the
evolution loop is identical whether it runs locally or on the GPU. This is how
the two historical tracks (mock local loop vs. real GPU training) become one.

Policy: every remote file lives under the explicitly configured
``$EVOMIND_HPC_REMOTE_WORKSPACE``. Credentials
come from ``gpu_credentials`` (env / ``*_FILE``); nothing is hardcoded or logged.
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Optional

from .evolution_loop import RunResult
from .hpc_policy import (
    HPCPolicyError,
    join_remote_workspace,
    require_remote_workspace,
    validate_remote_relative_path,
    validate_remote_workspace,
)


def _remote_root() -> str:
    legacy = os.environ.get("GPU_REMOTE_WORKSPACE", "").strip()
    return validate_remote_workspace(legacy) if legacy else require_remote_workspace()


@dataclass
class GPURunnerConfig:
    remote_root: str = ""
    evolution_subdir: str = "evolution"
    remote_python: str = "python3"
    timeout: int = 3600
    data_root: str = "mlebench_raw_data"  # workspace-relative parent for per-task data

    def __post_init__(self) -> None:
        if not self.remote_root:
            self.remote_root = _remote_root()
        else:
            self.remote_root = validate_remote_workspace(self.remote_root)
        if self.remote_root.startswith("~/"):
            raise HPCPolicyError(
                "GPURunner requires an absolute POSIX remote workspace; '~/...' is ambiguous over SFTP"
            )
        self.evolution_subdir = validate_remote_relative_path(
            self.evolution_subdir, field="evolution_subdir"
        )
        self.data_root = validate_remote_relative_path(
            self.data_root, field="data_root"
        )
        self.remote_python = _validate_remote_python(self.remote_python)
        if isinstance(self.timeout, bool):
            raise HPCPolicyError("timeout must be a positive integer")
        try:
            normalized_timeout = int(self.timeout)
        except (TypeError, ValueError) as exc:
            raise HPCPolicyError("timeout must be a positive integer") from exc
        if str(normalized_timeout) != str(self.timeout).strip() or normalized_timeout <= 0:
            raise HPCPolicyError("timeout must be a positive integer")
        self.timeout = normalized_timeout


def _validate_remote_python(value: str) -> str:
    executable = str(value or "")
    if not executable:
        raise HPCPolicyError("remote_python must not be empty")
    if executable.startswith("~/"):
        raise HPCPolicyError("remote_python must not use a '~/' path")
    relative = executable[1:] if executable.startswith("/") else executable
    validate_remote_relative_path(relative, field="remote_python")
    return executable


def _shell_arg(value: object) -> str:
    return shlex.quote(str(value))


class GPURunner:
    """Runs a candidate script on the GPU box over one SSH connection per run."""

    def __init__(self, task_data_dirname: str, *, config: Optional[GPURunnerConfig] = None,
                 connect=None) -> None:
        self.task_data_dirname = validate_remote_relative_path(
            task_data_dirname, field="task_data_dirname"
        )
        self.config = config or GPURunnerConfig()
        # connect is injectable for testing; defaults to the secure credential path.
        self._connect = connect

    def _open(self):
        if self._connect is not None:
            return self._connect()
        # Lazy import so this module stays importable in CI without paramiko/env.
        import sys
        import time
        from pathlib import Path
        src = str(Path(__file__).resolve().parents[1])
        if src not in sys.path:
            sys.path.insert(0, src)
        from research_agent_workstation.server.core.gpu_credentials import connect_ssh
        last_exc = None
        for attempt in range(3):  # transient SOCKS/SSH blips self-heal
            try:
                return connect_ssh()
            except Exception as exc:
                last_exc = exc
                time.sleep(3 * (attempt + 1))
        raise last_exc

    def _exec(self, client, command: str, timeout: int) -> tuple[int, str, str]:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def run(self, code: str, *, data_dir: str, out_dir: str, exp_id: str) -> RunResult:
        # data_dir is remote-relative. out_dir belongs to the local Runner
        # protocol and is deliberately never interpolated into a remote command.
        cfg = self.config
        safe_exp_id = validate_remote_relative_path(exp_id, field="exp_id")
        safe_data_dir = validate_remote_relative_path(data_dir, field="data_dir")
        remote_exp = join_remote_workspace(
            cfg.remote_root, cfg.evolution_subdir, self.task_data_dirname, safe_exp_id
        )
        remote_script = join_remote_workspace(
            cfg.remote_root, cfg.evolution_subdir, self.task_data_dirname, safe_exp_id, "solution.py"
        )
        remote_out = join_remote_workspace(
            cfg.remote_root, cfg.evolution_subdir, self.task_data_dirname, safe_exp_id, "out"
        )
        if "/" in safe_data_dir:
            remote_data = join_remote_workspace(cfg.remote_root, safe_data_dir)
        else:
            remote_data = join_remote_workspace(cfg.remote_root, cfg.data_root, safe_data_dir)

        client = self._open()
        try:
            # Start each run from a CLEAN out dir. The remote path is keyed only by
            # task+exp_id (no run timestamp), so a prior run's out/metrics.json would
            # otherwise survive here. On a kill (timeout/OOM) the current run prints no
            # CV_SCORE, and the fallback below would then read that STALE metrics.json
            # and attribute a phantom score to a failed run -> fabricated result.
            self._exec(
                client,
                (
                    f"mkdir -p -- {_shell_arg(remote_exp)} "
                    f"&& rm -rf -- {_shell_arg(remote_out)} "
                    f"&& mkdir -p -- {_shell_arg(remote_out)}"
                ),
                timeout=60,
            )
            sftp = client.open_sftp()
            try:
                with sftp.file(remote_script, "w") as handle:
                    handle.write(code)
            finally:
                sftp.close()
            cmd = (
                f"cd -- {_shell_arg(remote_exp)} "
                f"&& timeout {_shell_arg(cfg.timeout)} {_shell_arg(cfg.remote_python)} -u "
                f"{_shell_arg(remote_script)} --data-dir {_shell_arg(remote_data)} "
                f"--out-dir {_shell_arg(remote_out)} 2>&1"
            )
            rc, out, err = self._exec(client, cmd, timeout=cfg.timeout + 60)
            score = _parse_remote_score(out)
            # Only trust an on-disk metrics.json when the process exited cleanly. After
            # a non-zero exit (timeout=124/OOM=137/segfault=139) any metrics.json is
            # either stale (survived from a prior run) or half-written, so reading it
            # would fabricate a cv_score for a run that never emitted one. The clean-dir
            # step above already removes stale files; this gate is the belt-and-braces.
            if score is None and rc == 0:
                metrics_path = join_remote_workspace(
                    cfg.remote_root,
                    cfg.evolution_subdir,
                    self.task_data_dirname,
                    safe_exp_id,
                    "out",
                    "metrics.json",
                )
                rc2, mout, _ = self._exec(
                    client, f"cat -- {_shell_arg(metrics_path)} 2>/dev/null", timeout=60
                )
                if rc2 == 0 and mout.strip():
                    try:
                        score = float(json.loads(mout).get("cv_score"))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        score = None
            rc3, listing, _ = self._exec(
                client, f"ls -1 -- {_shell_arg(remote_out)} 2>/dev/null", timeout=60
            )
            artifacts = [f"{remote_out}/{name.strip()}" for name in listing.splitlines() if name.strip()]
            success = rc == 0 and score is not None
            # A remote kill (timeout=124, OOM/SIGKILL=137, segfault=139) leaves NO
            # traceback, only the last normal stdout line. Prepend an explicit
            # diagnostic derived from the exit code so the failure classifier and
            # the repair loop can name what happened instead of guessing.
            error = "" if success else _diagnose_exit(rc, out, timeout_s=cfg.timeout)
            return RunResult(
                success=success, cv_score=score,
                stdout_tail=out[-800:] if success else "",
                error=error, out_dir=remote_out, artifacts=artifacts,
                exit_code=rc,
            )
        finally:
            client.close()


def _parse_remote_score(text: str) -> Optional[float]:
    score = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("CV_SCORE="):
            try:
                score = float(line.split("=", 1)[1].strip())
            except ValueError:
                continue
    return score


# Shell/exit-code -> reusable failure reason. A killed process carries no
# traceback, so the exit code is the ONLY reliable signal of *why* it died.
# 124 = GNU coreutils `timeout` SIGTERM; 137 = 128+9 (SIGKILL, usually the OOM
# killer); 139 = 128+11 (SIGSEGV); 143 = 128+15 (SIGTERM).
def _diagnose_exit(rc: int, out: str, *, timeout_s: int) -> str:
    tail = (out or "").strip()
    last = tail.splitlines()[-1].strip() if tail else ""
    if rc == 124 or rc == 128 + 15 or rc == -15:
        head = (f"RUN_EXIT={rc} TIMEOUT: process exceeded the {timeout_s}s wall "
                f"budget and was killed before emitting CV_SCORE.")
    elif rc == 137 or rc == -9:
        head = (f"RUN_EXIT={rc} OOM_OR_KILLED: process received SIGKILL "
                f"(typically the out-of-memory killer) before emitting CV_SCORE.")
    elif rc == 139 or rc == -11:
        head = f"RUN_EXIT={rc} SEGFAULT: native crash (SIGSEGV) before CV_SCORE."
    elif rc != 0:
        head = f"RUN_EXIT={rc} NONZERO_EXIT: process failed before emitting CV_SCORE."
    else:
        # rc==0 but no score: the script exited cleanly yet never printed
        # CV_SCORE (a genuine contract violation, not a kill).
        return (tail[-1500:] or "no CV_SCORE emitted")
    ctx = f"\nlast stdout line before exit: {last}" if last else ""
    body = ("\n--- captured output tail ---\n" + tail[-1200:]) if tail else ""
    return f"{head}{ctx}{body}"

