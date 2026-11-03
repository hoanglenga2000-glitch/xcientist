"""Isolation helpers for agent-launched child processes.

Agent tools may start allowlisted programs (python, node, git, rg, uv).  The
runtime process holds model API keys, session secrets and DPAPI-backed
credential state, so children get an allowlist-only environment, and each
child runs inside a Windows Job Object with memory / process-count / CPU
limits that kills the whole tree when the handle closes.

On non-Windows hosts the job helpers are no-ops so tests stay portable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

_ALLOWED_NAMES = frozenset(
    {
        "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
        "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
        "PROGRAMW6432", "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
        "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS",
        "LANG", "LC_ALL", "LC_CTYPE", "TZ",
        "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH",
        "VIRTUAL_ENV", "CONDA_PREFIX", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_CACHE_DIR",
        "UV_CACHE_DIR", "UV_INDEX_URL", "NODE_PATH", "NPM_CONFIG_CACHE",
        "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "CUDA_VISIBLE_DEVICES", "WORKSTATION_PYTHON",
    }
)
_SECRET_HINT = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|COOKIE|CREDENTIAL|DPAPI|AUTH)", re.I)
_URL_CREDENTIAL = re.compile(r"://[^/\s@]+:[^/\s@]+@")


def child_environment(
    source: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an allowlist-only environment for an agent child process."""

    base = os.environ if source is None else source
    env: dict[str, str] = {}
    for name, value in base.items():
        upper = name.upper()
        if upper not in _ALLOWED_NAMES or _SECRET_HINT.search(upper):
            continue
        if _URL_CREDENTIAL.search(value):
            continue
        env[name] = value
    for name, value in (extra or {}).items():
        if _SECRET_HINT.search(name):
            raise ValueError(f"refusing to pass secret-like variable to child: {name}")
        env[name] = value
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


@dataclass(frozen=True)
class JobLimits:
    memory_bytes: int = 8 * 1024 ** 3
    active_processes: int = 64
    cpu_rate_percent: int = 50

    @classmethod
    def from_env(cls, source: Mapping[str, str] | None = None) -> "JobLimits":
        base = os.environ if source is None else source

        def number(name: str, default: int, low: int, high: int) -> int:
            try:
                value = int(str(base.get(name, "")).strip() or default)
            except ValueError:
                value = default
            return max(low, min(value, high))

        return cls(
            memory_bytes=number("EVOMIND_CHILD_MEMORY_MB", 8192, 256, 262144) * 1024 * 1024,
            active_processes=number("EVOMIND_CHILD_MAX_PROCESSES", 64, 4, 1024),
            cpu_rate_percent=number("EVOMIND_CHILD_CPU_PERCENT", 50, 5, 100),
        )


class JobHandle:
    """Owns a Windows Job Object handle; closing it kills the process tree."""

    def __init__(self, handle: int | None) -> None:
        self._handle = handle

    @property
    def active(self) -> bool:
        return bool(self._handle)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.CloseHandle(handle)

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def _create_job(limits: JobLimits) -> int:
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class CPU_RATE(ctypes.Structure):
        _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    limit_active_process = 0x00000008
    limit_job_memory = 0x00000200
    limit_die_on_unhandled_exception = 0x00000400
    limit_kill_on_job_close = 0x00002000
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = (
        limit_active_process | limit_job_memory | limit_die_on_unhandled_exception | limit_kill_on_job_close
    )
    info.BasicLimitInformation.ActiveProcessLimit = limits.active_processes
    info.JobMemoryLimit = limits.memory_bytes
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise OSError(error, "SetInformationJobObject(extended) failed")

    if limits.cpu_rate_percent < 100:
        rate = CPU_RATE(0x1 | 0x4, limits.cpu_rate_percent * 100)  # ENABLE | HARD_CAP
        # CPU rate control is best effort; some SKUs reject it.
        kernel32.SetInformationJobObject(handle, 15, ctypes.byref(rate), ctypes.sizeof(rate))
    return int(handle)


def attach_job(process: Any, limits: JobLimits | None = None) -> JobHandle:
    """Place ``process`` (a ``subprocess.Popen``) in a fresh limited job.

    Returns an inactive handle on non-Windows hosts. Raises ``OSError`` when
    the job cannot be created or assigned so callers can fail closed.
    """

    if os.name != "nt":
        return JobHandle(None)
    import ctypes
    from ctypes import wintypes

    handle = _create_job(limits or JobLimits.from_env())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    process_handle = int(getattr(process, "_handle"))
    if not kernel32.AssignProcessToJobObject(handle, process_handle):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise OSError(error, "AssignProcessToJobObject failed")
    return JobHandle(handle)


CREATE_SUSPENDED = 0x00000004


def resume_process(process: Any) -> None:
    """Resume a child created with ``CREATE_SUSPENDED`` after job assignment."""

    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = ntdll.NtResumeProcess(int(getattr(process, "_handle")))
    if status != 0:
        raise OSError(status, "NtResumeProcess failed")
