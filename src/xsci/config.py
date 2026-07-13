"""Layered configuration for the xsci terminal agent.

Precedence (low -> high): global (~/.xsci/config.toml) < project (./.xsci/
config.toml) < environment variables < explicit CLI flags. On Windows, secrets
are protected with current-user DPAPI and restrictive ACLs. Other platforms use
the global 0600 secrets file. Secrets are never written to the project dir.
"""
from __future__ import annotations

import base64
import contextlib
import csv
import ctypes
import json
import os
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # py311+ stdlib; fall back for 3.10
    import tomllib as _toml_read
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml_read  # type: ignore

def _windows_profile_dir() -> Optional[Path]:
    """Resolve the real Windows profile dir without trusting a mojibake env var."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

            @classmethod
            def from_uuid(cls, value: uuid.UUID) -> "GUID":
                data = value.bytes_le
                return cls(
                    int.from_bytes(data[0:4], "little"),
                    int.from_bytes(data[4:6], "little"),
                    int.from_bytes(data[6:8], "little"),
                    (wintypes.BYTE * 8).from_buffer_copy(data[8:16]),
                )

        folder_id_profile = GUID.from_uuid(uuid.UUID("5e6c858f-0e22-4760-9afe-ea3317b67173"))
        path_ptr = wintypes.LPWSTR()
        hr = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(folder_id_profile),
            0,
            None,
            ctypes.byref(path_ptr),
        )
        if hr != 0 or not path_ptr.value:
            return None
        path = Path(path_ptr.value)
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        return path
    except Exception:  # noqa: BLE001 - path fallback must never break imports
        return None


def default_global_dir() -> Path:
    configured = os.environ.get("XSCI_HOME")
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        profile = _windows_profile_dir()
        if profile is not None:
            return profile / ".xsci"
    return Path.home() / ".xsci"


GLOBAL_DIR = default_global_dir()
GLOBAL_CONFIG = GLOBAL_DIR / "config.toml"
SECRETS_FILE = GLOBAL_DIR / "secrets.toml"
ONBOARDED_MARKER = GLOBAL_DIR / "onboarded.json"
# Global fallback workspace: where the `kaggle` command stores tasks/experiments
# when invoked outside a project (mirrors how `claude` works from anywhere).
GLOBAL_WORKSPACE = GLOBAL_DIR / "workspace"
PROJECT_DIRNAME = ".xsci"

_DPAPI_SCHEMA = "xcientist.windows_dpapi_secrets.v1"
_DPAPI_ENTROPY = b"xcientist.secrets.v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_CREDENTIAL_STORE_MUTEX = "Local\\EvoMind.Xcientist.SecretStore.v1"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_ulong),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _windows_dpapi_path() -> Path:
    return SECRETS_FILE.with_name("secrets.dpapi.json")


@contextlib.contextmanager
def _windows_secret_store_lock():
    if os.name != "nt":
        yield
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateMutexW(None, False, _CREDENTIAL_STORE_MUTEX)
    if not handle:
        raise ctypes.WinError()
    acquired = False
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 30_000)
        acquired = wait_result in {_WAIT_OBJECT_0, _WAIT_ABANDONED}
        if not acquired:
            raise TimeoutError("timed out waiting for the Windows credential store lock")
        yield
    finally:
        if acquired:
            kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def _data_blob(value: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(value)
    blob = _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buffer


def _dpapi_transform(value: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Windows DPAPI is unavailable on this platform")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_blob, input_buffer = _data_blob(value)
    entropy_blob, entropy_buffer = _data_blob(_DPAPI_ENTROPY)
    output_blob = _DataBlob()
    description = ctypes.c_wchar_p()
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "EvoMind credential",
            ctypes.byref(entropy_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            ctypes.byref(description),
            ctypes.byref(entropy_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
    _ = input_buffer, entropy_buffer
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        if output_blob.pbData:
            kernel32.LocalFree(output_blob.pbData)
        if description:
            kernel32.LocalFree(description)


def _dpapi_protect(value: str) -> str:
    encrypted = _dpapi_transform(value.encode("utf-8"), protect=True)
    return base64.b64encode(encrypted).decode("ascii")


def _dpapi_unprotect(value: str) -> str:
    try:
        encrypted = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("invalid DPAPI credential encoding") from exc
    try:
        return _dpapi_transform(encrypted, protect=False).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid DPAPI credential payload") from exc


def _assert_no_windows_reparse_points(path: Path) -> None:
    if os.name != "nt":
        return
    get_attributes = ctypes.windll.kernel32.GetFileAttributesW
    for candidate in (path, *path.parents):
        if not candidate.exists():
            continue
        attributes = get_attributes(str(candidate))
        if attributes != _INVALID_FILE_ATTRIBUTES and attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(f"credential storage path contains a reparse point: {candidate}")


def _current_windows_sid() -> str:
    result = subprocess.run(
        ["whoami.exe", "/user", "/fo", "csv", "/nh"],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise OSError("could not resolve the current Windows user SID")
    rows = list(csv.reader(result.stdout.splitlines()))
    if not rows or len(rows[0]) < 2 or not rows[0][1].startswith("S-"):
        raise OSError("whoami returned an invalid Windows user SID")
    return rows[0][1]


def _protect_windows_acl(path: Path, *, directory: bool) -> None:
    sid = _current_windows_sid()
    suffix = "(OI)(CI)(F)" if directory else "(F)"
    result = subprocess.run(
        [
            "icacls.exe",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:{suffix}",
            f"*S-1-5-18:{suffix}",
        ],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise PermissionError(f"could not secure credential ACL for {path.name}")


def _read_windows_dpapi_secrets() -> dict[str, str]:
    path = _windows_dpapi_path()
    if not path.exists():
        return {}
    _assert_no_windows_reparse_points(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Windows DPAPI credential store is malformed") from exc
    if payload.get("schema") != _DPAPI_SCHEMA or not isinstance(payload.get("secrets"), dict):
        raise ValueError("Windows DPAPI credential store schema is invalid")
    secrets: dict[str, str] = {}
    for key, encrypted in payload["secrets"].items():
        if not isinstance(key, str) or not key or not isinstance(encrypted, str) or not encrypted:
            raise ValueError("Windows DPAPI credential entry is invalid")
        secrets[key] = _dpapi_unprotect(encrypted)
    return secrets


def _write_windows_dpapi_secrets(secrets: dict[str, str]) -> Path:
    path = _windows_dpapi_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_windows_reparse_points(path.parent)
    _protect_windows_acl(path.parent, directory=True)
    payload = {
        "schema": _DPAPI_SCHEMA,
        "secrets": {key: _dpapi_protect(value) for key, value in sorted(secrets.items())},
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _protect_windows_acl(temporary, directory=False)
        os.replace(temporary, path)
        _protect_windows_acl(path, directory=False)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _scrub_legacy_secret_file(payload: dict[str, Any]) -> None:
    remaining = {key: value for key, value in payload.items() if key != "secrets"}
    if remaining:
        _write_toml_atomic(SECRETS_FILE, remaining)
    else:
        SECRETS_FILE.unlink(missing_ok=True)


def _load_secret_chunk() -> tuple[dict[str, Any], Optional[Path]]:
    legacy = _read_toml(SECRETS_FILE)
    if os.name != "nt":
        return legacy, SECRETS_FILE if legacy else None
    encrypted = _read_windows_dpapi_secrets()
    legacy_secrets = legacy.get("secrets") if isinstance(legacy.get("secrets"), dict) else {}
    if legacy_secrets:
        with _windows_secret_store_lock():
            legacy = _read_toml(SECRETS_FILE)
            legacy_secrets = legacy.get("secrets") if isinstance(legacy.get("secrets"), dict) else {}
            encrypted = _read_windows_dpapi_secrets()
            if legacy_secrets:
                migrated = {**encrypted, **{str(key): str(value) for key, value in legacy_secrets.items()}}
                _write_windows_dpapi_secrets(migrated)
                _scrub_legacy_secret_file(legacy)
                encrypted = migrated
    target = _windows_dpapi_path()
    return ({"secrets": encrypted} if encrypted else {}), target if encrypted else None

# env var name -> dotted config key it overrides
_ENV_MAP = {
    "XSCI_LLM_PROVIDER": "llm.provider",
    "XSCI_LLM_MODEL": "llm.model",
    "XSCI_COMPUTE": "compute.backend",
    "XSCI_DASHBOARD_URL": "workstation.dashboard_url",
    "EVOMIND_DASHBOARD_URL": "workstation.dashboard_url",
    "ANTHROPIC_API_KEY": "secrets.anthropic_api_key",
    "DEEPSEEK_API_KEY": "secrets.deepseek_api_key",
    "OPENAI_API_KEY": "secrets.openai_api_key",
    "KAGGLE_API_TOKEN": "secrets.kaggle_api_token",
    "KAGGLE_USERNAME": "secrets.kaggle_username",
    "KAGGLE_KEY": "secrets.kaggle_key",
    "GPU_SSH_PASSWORD": "secrets.gpu_ssh_password",
    "GPU_SSH_SOCKS_PASSWORD": "secrets.gpu_ssh_socks_password",
    "GPU_SSH_HOST": "gpu_ssh.host",
    "GPU_SSH_PORT": "gpu_ssh.port",
    "GPU_SSH_USER": "gpu_ssh.user",
    "GPU_REMOTE_WORKSPACE": "gpu_ssh.remote_workspace",
    "GPU_SSH_KEY_PATH": "gpu_ssh.key_path",
    "GPU_SSH_SOCKS_HOST": "gpu_ssh.socks_host",
    "GPU_SSH_SOCKS_PORT": "gpu_ssh.socks_port",
}


@dataclass
class Config:
    """Resolved, read-only view of the merged configuration."""

    data: dict[str, Any] = field(default_factory=dict, repr=False)
    sources: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Config(sources={self.sources!r}, secret_values='[redacted]')"

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        val = self.get(dotted)
        if val in (None, ""):
            raise KeyError(f"missing required config: {dotted}")
        return val


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _toml_read.loads(path.read_text(encoding="utf-8-sig"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _apply_env(data: dict[str, Any]) -> dict[str, Any]:
    for env_name, dotted in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if not val:
            continue
        section, _, leaf = dotted.partition(".")
        data.setdefault(section, {})[leaf] = val
    return data


def find_project_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``.xsci`` project marker.

    The global config home (``GLOBAL_DIR``, e.g. ``~/.xsci``) is NOT a project: it
    holds secrets/config. If a user runs ``kaggle`` from their HOME dir, the naive
    walk finds ``~/.xsci`` and mistakes HOME for a project — which put the workspace
    in HOME and scattered tasks into the secret store. So a candidate whose ``.xsci``
    marker *is* the global home is skipped, and we fall back to the global workspace.
    """
    cur = (start or Path.cwd()).resolve()
    try:
        global_marker = GLOBAL_DIR.resolve()
    except OSError:  # resolve can fail on odd/unmapped paths; fall back unresolved
        global_marker = GLOBAL_DIR
    for candidate in [cur, *cur.parents]:
        marker = candidate / PROJECT_DIRNAME
        if not marker.is_dir():
            continue
        try:
            if marker.resolve() == global_marker:
                continue  # the global config home, not a project marker
        except OSError:
            pass
        return candidate
    return None


def global_workspace() -> Path:
    """Return the current global workspace.

    Tests and embedding tools sometimes monkeypatch ``GLOBAL_DIR`` after import,
    so derive this path dynamically instead of freezing the original home path.
    """
    return GLOBAL_DIR / "workspace"


def active_root(start: Optional[Path] = None) -> Path:
    """The root the `kaggle` console operates under.

    If ``start`` (default cwd) is inside a real ``.xsci`` project, use it — so
    running `kaggle` in a project keeps that project's tasks/experiments. Otherwise
    fall back to the GLOBAL workspace (~/.xsci/workspace), scaffolding its ``.xsci/
    tasks`` marker so the existing task helpers treat it as a project. This is what
    lets `kaggle`, like `claude`, "just work" from any directory.
    """
    proj = find_project_dir(start)
    if proj is not None:
        return proj
    root = global_workspace()
    (root / PROJECT_DIRNAME / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    return root


def is_onboarded() -> bool:
    """First-run detection: True once the onboarding wizard has completed, or if
    the user already has any global config/secrets (so upgraders aren't re-prompted)."""
    return (
        ONBOARDED_MARKER.exists()
        or GLOBAL_CONFIG.exists()
        or SECRETS_FILE.exists()
        or (os.name == "nt" and _windows_dpapi_path().exists())
    )


def mark_onboarded() -> Path:
    """Persist the onboarding-complete marker (no secrets — just a timestamp)."""
    from datetime import datetime

    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    import json as _json

    ONBOARDED_MARKER.write_text(
        _json.dumps({"onboarded_at": datetime.now().isoformat(timespec="seconds")}),
        encoding="utf-8",
    )
    return ONBOARDED_MARKER


def load_config(project_root: Optional[Path] = None) -> Config:
    """Merge global < secrets < project < env into one resolved Config."""
    sources: list[str] = []
    data: dict[str, Any] = {}

    global_chunk = _read_toml(GLOBAL_CONFIG)
    if global_chunk:
        data = _deep_merge(data, global_chunk)
        sources.append(f"global:{GLOBAL_CONFIG}")
    secret_chunk, secret_path = _load_secret_chunk()
    if secret_chunk:
        data = _deep_merge(data, secret_chunk)
        sources.append(f"secrets:{secret_path}")

    root = project_root or find_project_dir()
    if root is not None:
        proj = _read_toml(root / PROJECT_DIRNAME / "config.toml")
        if proj:
            data = _deep_merge(data, proj)
            sources.append(f"project:{root / PROJECT_DIRNAME / 'config.toml'}")

    data = _apply_env(data)
    if any(os.environ.get(k) for k in _ENV_MAP):
        sources.append("env")
    return Config(data=data, sources=sources)


def write_secret(key: str, value: str) -> Path:
    """Persist a secret outside the project using the platform secret store.

    Windows uses current-user DPAPI plus a current-user/SYSTEM DACL. Other
    platforms use the global 0600 secrets file. Never writes into a project.
    """
    if not key or not value:
        raise ValueError("secret key and value are required")
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        with _windows_secret_store_lock():
            legacy = _read_toml(SECRETS_FILE)
            legacy_secrets = legacy.get("secrets") if isinstance(legacy.get("secrets"), dict) else {}
            secrets = _read_windows_dpapi_secrets()
            secrets.update({str(name): str(item) for name, item in legacy_secrets.items()})
            secrets[key] = value
            path = _write_windows_dpapi_secrets(secrets)
            if legacy:
                _scrub_legacy_secret_file(legacy)
            return path
    try:
        os.chmod(GLOBAL_DIR, stat.S_IRWXU)  # 0700
    except OSError:
        pass  # best-effort on platforms without POSIX perms (Windows)
    existing = _read_toml(SECRETS_FILE)
    secrets = existing.get("secrets", {}) if isinstance(existing.get("secrets"), dict) else {}
    secrets[key] = value
    existing["secrets"] = secrets
    _write_toml_atomic(SECRETS_FILE, existing)
    try:
        os.chmod(SECRETS_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass
    return SECRETS_FILE


def set_global(section: str, key: str, value: Any) -> Path:
    """Persist a NON-secret setting (provider, base_url, model, compute backend)
    to the global config.toml. Secrets must use ``write_secret`` instead."""
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_toml(GLOBAL_CONFIG)
    body = existing.get(section, {}) if isinstance(existing.get(section), dict) else {}
    body[key] = value
    existing[section] = body
    _write_toml_atomic(GLOBAL_CONFIG, existing)
    return GLOBAL_CONFIG


# secret config key -> env var the research_os engine reads
_SECRET_ENV = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "kaggle_api_token": "KAGGLE_API_TOKEN",
    "kaggle_username": "KAGGLE_USERNAME",
    "kaggle_key": "KAGGLE_KEY",
    "gpu_ssh_password": "GPU_SSH_PASSWORD",
    "gpu_ssh_socks_password": "GPU_SSH_SOCKS_PASSWORD",
}

_PLAIN_ENV = {
    "gpu_ssh.host": "GPU_SSH_HOST",
    "gpu_ssh.port": "GPU_SSH_PORT",
    "gpu_ssh.user": "GPU_SSH_USER",
    "gpu_ssh.remote_workspace": "GPU_REMOTE_WORKSPACE",
    "gpu_ssh.key_path": "GPU_SSH_KEY_PATH",
    "gpu_ssh.socks_host": "GPU_SSH_SOCKS_HOST",
    "gpu_ssh.socks_port": "GPU_SSH_SOCKS_PORT",
}


def inject_engine_env(cfg: "Config", *, override: bool = False) -> list[str]:
    """Export resolved secrets/provider choice into os.environ so the engine
    (research_os.llm_client, kaggle) can read them. Returns the list of env var
    NAMES set (never values, so this is safe to log). Existing env vars are kept
    unless ``override`` is True.
    """
    injected: list[str] = []
    for skey, env_name in _SECRET_ENV.items():
        val = cfg.get(f"secrets.{skey}")
        if val and (override or not os.environ.get(env_name)):
            os.environ[env_name] = str(val)
            injected.append(env_name)
    for dotted, env_name in _PLAIN_ENV.items():
        val = cfg.get(dotted)
        if val not in (None, "") and (override or not os.environ.get(env_name)):
            os.environ[env_name] = str(val)
            injected.append(env_name)
    provider = cfg.get("llm.provider")
    if provider and (override or not os.environ.get("EVOLUTION_PRIMARY_PROVIDER")):
        os.environ["EVOLUTION_PRIMARY_PROVIDER"] = str(provider)
        injected.append("EVOLUTION_PRIMARY_PROVIDER")
    for prov in ("anthropic", "deepseek", "openai"):
        base = cfg.get(f"llm.{prov}_base_url")
        env_name = f"{prov.upper()}_BASE_URL"
        if base and (override or not os.environ.get(env_name)):
            os.environ[env_name] = str(base)
            injected.append(env_name)
    # The user's chosen default model -> the env var the ACTIVE provider reads
    # (Anthropic client + deep agent read CLAUDE_CODE_MODEL; DeepSeek reads
    # DEEPSEEK_MODEL). Only the active provider's var is set, so a stored model
    # never leaks onto the wrong backend.
    model = cfg.get("llm.model")
    if model:
        model_env = {
            "anthropic": "CLAUDE_CODE_MODEL",
            "deepseek": "DEEPSEEK_MODEL",
            "openai": "OPENAI_MODEL",
        }.get(str(provider or "").lower())
        if model_env and (override or not os.environ.get(model_env)):
            os.environ[model_env] = str(model)
            injected.append(model_env)
    return injected


def _write_toml_atomic(path: Path, data: dict[str, Any]) -> None:
    """Minimal TOML writer (str/int/float/bool/nested-table only)."""
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    for key, val in scalars.items():
        lines.append(f"{key} = {_toml_scalar(val)}")
    for table, body in tables.items():
        lines.append(f"\n[{table}]")
        for key, val in body.items():
            lines.append(f"{key} = {_toml_scalar(val)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def _toml_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
