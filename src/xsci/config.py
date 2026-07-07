"""Layered configuration for the xsci terminal agent.

Precedence (low -> high): global (~/.xsci/config.toml) < project (./.xsci/
config.toml) < environment variables < explicit CLI flags. Secrets live ONLY in
the global home dir with 0600 perms and are NEVER written to the project dir
(which may be committed to git).
"""
from __future__ import annotations

import os
import stat
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

# env var name -> dotted config key it overrides
_ENV_MAP = {
    "XSCI_LLM_PROVIDER": "llm.provider",
    "XSCI_LLM_MODEL": "llm.model",
    "XSCI_COMPUTE": "compute.backend",
    "XSCI_DASHBOARD_URL": "workstation.dashboard_url",
    "EVOMIND_DASHBOARD_URL": "workstation.dashboard_url",
    "ANTHROPIC_API_KEY": "secrets.anthropic_api_key",
    "DEEPSEEK_API_KEY": "secrets.deepseek_api_key",
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

    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

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
    with path.open("rb") as fh:
        return _toml_read.load(fh)


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
    return ONBOARDED_MARKER.exists() or GLOBAL_CONFIG.exists() or SECRETS_FILE.exists()


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

    for path, label in [(GLOBAL_CONFIG, "global"), (SECRETS_FILE, "secrets")]:
        chunk = _read_toml(path)
        if chunk:
            data = _deep_merge(data, chunk)
            sources.append(f"{label}:{path}")

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
    """Persist a secret to the GLOBAL secrets file with 0600 perms.

    Never writes into a project dir. Creates ~/.xsci with 0700 if needed.
    """
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
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
    for prov in ("anthropic", "deepseek"):
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
        model_env = {"anthropic": "CLAUDE_CODE_MODEL",
                     "deepseek": "DEEPSEEK_MODEL"}.get(str(provider or "").lower())
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
