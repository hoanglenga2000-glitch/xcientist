from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:  # Bundles declare psutil; retain a diagnostic fallback for source checkouts.
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE_APP_DIR = ROOT / "web" / "research-agent-workstation"
STANDALONE_SERVER = ROOT / "app" / "server.js"
SOURCE_STANDALONE_SERVER = SOURCE_APP_DIR / ".next" / "standalone" / "server.js"
WORKSTATION_SUMMARY_PATH = "/api/workstation-summary"  # Authenticated UI data; lifecycle probes /api/healthz.
LOOPBACK_OPENAI_BASE_URL = "http://127.0.0.1:65068/v1"
DASHBOARD_OPENAI_MODEL = "gpt-5.6-sol"
DATABASE_RUNTIME_SUFFIXES = (".db", ".db-journal", ".db-shm", ".db-wal")
SOURCE_FILES = {
    "package.json",
    "package-lock.json",
    "next.config.mjs",
    "postcss.config.mjs",
    "tailwind.config.ts",
    "tsconfig.json",
}
SOURCE_DIRECTORIES = ("src", "prisma", "public", "scripts")
APP_DIR = SOURCE_APP_DIR
RUNTIME_DIR = SOURCE_APP_DIR / ".runtime-logs"
PID_FILE = RUNTIME_DIR / "dashboard.pid"
STATE_FILE = RUNTIME_DIR / "dashboard.process.json"
DEFAULT_DATABASE_PATH = SOURCE_APP_DIR / "prisma" / "workstation.db"
PRISMA_PUSH_SCRIPT = SOURCE_APP_DIR / "scripts" / "prisma-db-push.mjs"
# Default-port compatibility names: dashboard.pid, dashboard.out.log,
# dashboard.err.log. Non-default test/user ports receive a numeric suffix.


def bundle_mode() -> bool:
    return STANDALONE_SERVER.is_file()


def source_standalone_mode() -> bool:
    return SOURCE_STANDALONE_SERVER.is_file()


def app_dir() -> Path:
    if bundle_mode():
        return STANDALONE_SERVER.parent
    if source_standalone_mode():
        return SOURCE_STANDALONE_SERVER.parent
    return SOURCE_APP_DIR


def source_tree_digest(target_app_dir: Path = SOURCE_APP_DIR) -> str:
    files = [target_app_dir / name for name in SOURCE_FILES if (target_app_dir / name).is_file()]
    for directory in SOURCE_DIRECTORIES:
        root = target_app_dir / directory
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and not path.name.lower().endswith(DATABASE_RUNTIME_SUFFIXES)
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(target_app_dir).as_posix()):
        relative = path.relative_to(target_app_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def data_root() -> Path:
    configured = os.environ.get("WORKSTATION_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT / "user-data").resolve() if bundle_mode() else ROOT.resolve()


def runtime_dir() -> Path:
    configured = os.environ.get("WORKSTATION_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if bundle_mode():
        return data_root() / "logs"
    return RUNTIME_DIR


def runtime_paths(port: int = 8088) -> tuple[Path, Path, Path, Path]:
    directory = runtime_dir()
    if port == 8088:
        return PID_FILE, STATE_FILE, directory / "dashboard.out.log", directory / "dashboard.err.log"
    suffix = "" if port == 8088 else f".{port}"
    return (
        directory / f"dashboard{suffix}.pid",
        directory / f"dashboard{suffix}.process.json",
        directory / f"dashboard{suffix}.out.log",
        directory / f"dashboard{suffix}.err.log",
    )


def runtime_service_paths(port: int = 8765) -> tuple[Path, Path, Path, Path]:
    directory = runtime_dir()
    suffix = "" if port == 8765 else f".{port}"
    return (
        directory / f"runtime{suffix}.pid",
        directory / f"runtime{suffix}.process.json",
        directory / f"runtime{suffix}.out.log",
        directory / f"runtime{suffix}.err.log",
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def bootstrap_url_path(port: int = 8088) -> Path:
    suffix = "" if port == 8088 else f".{port}"
    return runtime_dir() / f"dashboard{suffix}.bootstrap.once"


def automation_token_path(port: int = 8088) -> Path:
    suffix = "" if port == 8088 else f".{port}"
    return runtime_dir() / f"dashboard{suffix}.automation.token"


def remove_local_auth_files(port: int = 8088) -> None:
    """Remove lifecycle-bound local auth material after a verified stop/failure."""

    bootstrap_url_path(port).unlink(missing_ok=True)
    automation_token_path(port).unlink(missing_ok=True)


def node_command() -> str:
    bundled = ROOT / "runtime" / "node" / "node.exe"
    for candidate in (os.environ.get("WORKSTATION_NODE"), str(bundled), shutil.which("node.exe"), shutil.which("node")):
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise SystemExit("DASHBOARD_MANAGER_FAILED: Node.js was not found (WORKSTATION_NODE, runtime/node/node.exe, PATH)")


def npm_command() -> str:
    candidate = shutil.which("npm.cmd") or shutil.which("npm")
    if not candidate:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: npm was not found")
    return candidate


def next_cli_path() -> str:
    candidate = SOURCE_APP_DIR / "node_modules" / "next" / "dist" / "bin" / "next"
    if candidate.is_file():
        return str(candidate)
    raise SystemExit("DASHBOARD_MANAGER_FAILED: Next.js CLI not found under node_modules")


def application_version() -> str:
    """Resolve the immutable app version even when Node is not launched by npm."""

    candidates = (
        [STANDALONE_SERVER.parent / "package.json", ROOT / "package.json"]
        if bundle_mode()
        else [SOURCE_APP_DIR / "package.json"]
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("version") if isinstance(payload, dict) else None
        if isinstance(value, str) and value.strip() and len(value.strip()) <= 64:
            return value.strip()
    return "unknown"


def backend_version() -> str:
    """Read the Python package version without importing the mutable runtime."""

    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'(?m)^version\s*=\s*"([^"\r\n]{1,64})"\s*$', text)
    return match.group(1).strip() if match else "unknown"


def git_source_identity() -> tuple[str, bool]:
    """Return the checked-out commit and whether tracked/build inputs are dirty."""

    def capture(command: list[str], timeout: float) -> tuple[int, str]:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, _stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        return process.returncode, stdout

    try:
        commit_code, commit_output = capture(["git", "rev-parse", "HEAD"], 15)
        status_code, status_output = capture(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], 30
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("source Git identity is unavailable") from error
    commit_hash = commit_output.strip().lower()
    if commit_code != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit_hash):
        raise RuntimeError("source Git commit identity is invalid")
    if status_code != 0:
        raise RuntimeError("source Git dirty-state identity is unavailable")
    return commit_hash, bool(status_output.strip())


def _normalized_schema_sql(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def database_schema_identity(database_path: Path | None = None) -> dict[str, str]:
    """Fingerprint the effective SQLite schema, not mutable database rows."""

    database = (database_path or DEFAULT_DATABASE_PATH).expanduser().resolve()
    if not database.is_file():
        raise RuntimeError(f"database schema is unavailable: {database}")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, COALESCE(sql, '')
            FROM sqlite_master
            WHERE type IN ('table', 'index', 'view', 'trigger')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name
            """
        ).fetchall()
    finally:
        connection.close()
    digest = hashlib.sha256()
    for row in rows:
        canonical = "\0".join(
            (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""), _normalized_schema_sql(row[3]))
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    migrations = SOURCE_APP_DIR / "prisma" / "migrations"
    versions = sorted(
        child.name
        for child in migrations.iterdir()
        if child.is_dir() and (child / "migration.sql").is_file()
    ) if migrations.is_dir() else []
    return {
        "version": versions[-1] if versions else "prisma-push",
        "sha256": digest.hexdigest(),
    }


def runtime_build_manifest(build_id: str) -> dict[str, object]:
    commit_hash, source_dirty = git_source_identity()
    database = database_schema_identity()
    return {
        "schema": "evomind.runtime_build.v1",
        "commit_hash": commit_hash,
        "source_dirty": source_dirty,
        "source_tree_sha256": source_tree_digest(SOURCE_APP_DIR),
        "build_id": build_id,
        "build_time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "backend_version": backend_version(),
        "frontend_version": application_version(),
        "database_schema_version": database["version"],
        "database_schema_sha256": database["sha256"],
    }


def write_runtime_build_manifest(candidate: Path, manifest: dict[str, object]) -> Path:
    destination = candidate / "runtime-build-manifest.json"
    atomic_json(destination, manifest)
    standalone = candidate / "standalone"
    if standalone.is_dir():
        atomic_json(standalone / "runtime-build-manifest.json", manifest)
    return destination


def load_active_runtime_build_manifest() -> dict | None:
    configured = os.environ.get("EVOMIND_RUNTIME_BUILD_MANIFEST", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    if bundle_mode():
        candidates.extend([
            STANDALONE_SERVER.parent / "runtime-build-manifest.json",
            ROOT / "runtime-build-manifest.json",
        ])
    else:
        candidates.extend([
            SOURCE_APP_DIR / ".next" / "runtime-build-manifest.json",
            SOURCE_STANDALONE_SERVER.parent / "runtime-build-manifest.json",
        ])
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "evomind.runtime_build.v1"
            and re.fullmatch(r"[0-9a-fA-F]{40}", str(payload.get("commit_hash") or ""))
            and re.fullmatch(r"[0-9a-fA-F]{64}", str(payload.get("source_tree_sha256") or ""))
            and re.fullmatch(r"[0-9a-fA-F]{64}", str(payload.get("database_schema_sha256") or ""))
        ):
            return payload
    return None


def bind_runtime_build_identity(env: dict[str, str]) -> dict:
    manifest = load_active_runtime_build_manifest()
    if manifest is None:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: runtime build identity manifest is missing or invalid")
    manifest_path = (
        STANDALONE_SERVER.parent / "runtime-build-manifest.json"
        if bundle_mode()
        else SOURCE_APP_DIR / ".next" / "runtime-build-manifest.json"
    )
    env.update({
        "EVOMIND_RUNTIME_BUILD_MANIFEST": str(manifest_path.resolve()),
        "EVOMIND_BUILD_COMMIT_HASH": str(manifest["commit_hash"]),
        "EVOMIND_SOURCE_TREE_SHA256": str(manifest["source_tree_sha256"]),
        "EVOMIND_BUILD_ID": str(manifest.get("build_id") or ""),
        "EVOMIND_BUILD_TIME": str(manifest.get("build_time") or ""),
        "EVOMIND_BACKEND_VERSION": str(manifest.get("backend_version") or "unknown"),
        "EVOMIND_FRONTEND_VERSION": str(manifest.get("frontend_version") or "unknown"),
        "EVOMIND_DATABASE_SCHEMA_VERSION": str(manifest.get("database_schema_version") or "unknown"),
        "EVOMIND_DATABASE_SCHEMA_SHA256": str(manifest["database_schema_sha256"]),
    })
    return manifest


def write_source_install_marker(manifest: dict[str, object]) -> Path:
    marker = {
        "format_version": 1,
        "product": "research-workstation",
        "version": str(manifest.get("frontend_version") or application_version()),
        "layout": "source_tree",
        "managed_files": [],
        "data_root": str(ROOT.resolve()),
        "backups_root": str((ROOT / "backups").resolve()),
        "installed_at": str(manifest.get("build_time") or dt.datetime.now(dt.timezone.utc).isoformat()),
        "runtime_build": manifest,
    }
    destination = ROOT / ".workstation-install.json"
    atomic_json(destination, marker)
    return destination


def source_build_stale() -> bool:
    build_id = SOURCE_APP_DIR / ".next" / "BUILD_ID"
    if not build_id.is_file():
        return True
    try:
        manifest = json.loads(
            (SOURCE_APP_DIR / ".next" / "runtime-build-manifest.json").read_text(encoding="utf-8-sig")
        )
        return not (
            isinstance(manifest, dict)
            and manifest.get("schema") == "evomind.runtime_build.v1"
            and manifest.get("source_tree_sha256") == source_tree_digest(SOURCE_APP_DIR)
        )
    except (OSError, json.JSONDecodeError):
        return True


def _bind_loopback_gateway_credentials(env: dict[str, str]) -> None:
    """Bind the 65068 client to the gateway's own key, never an ambient key."""
    base_url = str(env.get("OPENAI_BASE_URL") or "").rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        is_local_gateway = (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port == 65068
            and parsed.path.rstrip("/") == "/v1"
        )
    except ValueError:
        is_local_gateway = False
    if not is_local_gateway:
        return

    configured = str(env.get("EVOMIND_LOCAL_GATEWAY_CONFIG") or "").strip()
    gateway_config = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".antigravity_cockpit" / "codex_local_access_sidecar" / "config.json"
    )
    key = ""
    try:
        if gateway_config.is_file() and not gateway_config.is_symlink() and gateway_config.stat().st_size <= 1024 * 1024:
            payload = json.loads(gateway_config.read_text(encoding="utf-8"))
            keys = payload.get("api-keys") if isinstance(payload, dict) else []
            if isinstance(keys, list):
                key = next((str(item).strip() for item in keys if str(item).strip()), "")
    except (OSError, ValueError, json.JSONDecodeError):
        key = ""

    if not key:
        appdata_raw = str(env.get("APPDATA") or "").strip()
        credential_candidates: list[Path] = []
        if appdata_raw:
            appdata = Path(appdata_raw)
            credential_candidates.extend([
                appdata / "EvoMind" / "secrets" / "openai_api_key.xml",
                appdata / "ResearchAgentWorkstation" / "openai_api_key.xml",
            ])
        configured_credential = str(env.get("EVOMIND_OPENAI_GATEWAY_CREDENTIAL") or "").strip()
        if configured_credential:
            credential_candidates.insert(0, Path(configured_credential).expanduser())
        credential_path = next(
            (
                candidate
                for candidate in credential_candidates
                if candidate.is_file()
                and not candidate.is_symlink()
                and candidate.stat().st_size <= 1024 * 1024
            ),
            None,
        )
        if credential_path is not None:
            script = (
                "$ErrorActionPreference='Stop';"
                "$c=Import-Clixml -LiteralPath $args[0];"
                "$c.GetNetworkCredential().Password"
            )
            try:
                completed = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", script, str(credential_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                if completed.returncode == 0:
                    key = completed.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                key = ""

    # A parent process may carry an unrelated cloud key. Never send that key to
    # the local gateway. If its own credential is unavailable, omit OpenAI and
    # let the provider layer transparently use another configured transport.
    if key:
        env["OPENAI_API_KEY"] = key
    else:
        env.pop("OPENAI_API_KEY", None)
    env["EVOLUTION_PRIMARY_PROVIDER"] = "openai"
    env["EVOLUTION_PROVIDER_STRICT"] = "true"
    env["OPENAI_BASE_URL"] = LOOPBACK_OPENAI_BASE_URL
    env["OPENAI_MODEL"] = DASHBOARD_OPENAI_MODEL

    profile_path_raw = str(env.get("EVOMIND_OPENAI_GATEWAY_METADATA") or "").strip()
    appdata_raw = str(env.get("APPDATA") or "").strip()
    profile_path = (
        Path(profile_path_raw).expanduser()
        if profile_path_raw
        else Path(appdata_raw) / "ResearchAgentWorkstation" / "openai_gateway_metadata.json"
        if appdata_raw
        else None
    )
    profile: dict[str, object] = {}
    try:
        if (
            profile_path is not None
            and profile_path.is_file()
            and not profile_path.is_symlink()
            and profile_path.stat().st_size <= 1024 * 1024
        ):
            loaded_profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded_profile, dict):
                profile = loaded_profile
    except (OSError, ValueError, json.JSONDecodeError):
        profile = {}

    interactive_effort = str(profile.get("interactive_reasoning_effort") or "low").strip().lower()
    if interactive_effort not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
        interactive_effort = "low"
    service_tier = str(profile.get("service_tier") or "priority").strip().lower()
    if service_tier not in {"auto", "default", "flex", "priority"}:
        service_tier = "priority"
    # The dashboard is the interactive novice-facing surface.  Override any
    # inherited research/high-effort profile while leaving research commands
    # free to select their own profile in a separate process.
    env["OPENAI_REASONING_EFFORT"] = interactive_effort
    env["OPENAI_SERVICE_TIER"] = service_tier


def dashboard_env(host: str = "127.0.0.1", port: int = 8088) -> dict[str, str]:
    env = os.environ.copy()
    root = data_root() if bundle_mode() else Path(env.get("WORKSTATION_ROOT", ROOT)).expanduser().resolve()
    env.setdefault("WORKSTATION_ROOT", str(root))
    env.setdefault("WORKSTATION_DATA_DIR", str(data_root()))
    if bundle_mode():
        database = data_root() / "prisma" / "workstation.db"
    else:
        database = DEFAULT_DATABASE_PATH
    database.parent.mkdir(parents=True, exist_ok=True)
    env["DATABASE_URL"] = f"file:{database.as_posix()}"
    env["WORKSTATION_PYTHON"] = sys.executable
    # The dashboard is permanently bound to the local account-pool gateway.
    # Do not let an unrelated parent-shell OPENAI_BASE_URL bypass credential,
    # strict-provider, model, and latency-profile binding below.
    env["OPENAI_BASE_URL"] = LOOPBACK_OPENAI_BASE_URL
    _bind_loopback_gateway_credentials(env)
    env.setdefault("WORKSTATION_LOCAL_FALLBACK", "1")
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    # Standalone is executed via `node server.js`, not `npm start`, so npm's
    # automatic npm_package_version variable is absent unless we bind it here.
    env["npm_package_version"] = application_version()
    env["HOSTNAME"] = host
    env["PORT"] = str(port)
    return env


def write_llm_route_snapshot(env: dict[str, str]) -> Path:
    """Persist the effective non-secret dashboard LLM route for restart audits."""
    target = runtime_dir() / "dashboard.llm-route.json"
    payload = {
        "schema": "evomind.dashboard_llm_route.v1",
        "provider": str(env.get("EVOLUTION_PRIMARY_PROVIDER") or ""),
        "strict": str(env.get("EVOLUTION_PROVIDER_STRICT") or "").strip().lower()
        in {"1", "true", "yes", "on"},
        "base_url": str(env.get("OPENAI_BASE_URL") or ""),
        "model": str(env.get("OPENAI_MODEL") or ""),
        "reasoning_effort": str(env.get("OPENAI_REASONING_EFFORT") or ""),
        "service_tier": str(env.get("OPENAI_SERVICE_TIER") or ""),
        "has_key": bool(str(env.get("OPENAI_API_KEY") or "").strip()),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def runtime_env(dashboard: dict[str, str], release_nonce: str, port: int) -> dict[str, str]:
    env = dashboard.copy()
    for name in (
        "WORKSTATION_SESSION_SECRET",
        "WORKSTATION_BOOTSTRAP_TOKEN_HASH",
        "WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH",
        "HTTP_COOKIE",
        "COOKIE",
        "EVOMIND_SESSION_COOKIE",
        "EVOMIND_CSRF_TOKEN",
    ):
        env.pop(name, None)
    env["WORKSTATION_RELEASE_NONCE"] = release_nonce
    env["EVOMIND_RUNTIME_PORT"] = str(port)
    if bundle_mode():
        env.pop("PYTHONPATH", None)
    else:
        source_root = (ROOT / "src").resolve()
        if not source_root.is_dir():
            raise RuntimeError("source runtime PYTHONPATH root is missing")
        env["PYTHONPATH"] = str(source_root)
    return env


def url_for(host: str, port: int, path: str = "") -> str:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}{path}"


def control_url_for(host: str, port: int) -> str:
    return url_for(host, port, "/?page=assistant")


def write_bootstrap_url(host: str, port: int, bootstrap_token: str) -> Path:
    """Persist a one-time fragment URL without exposing its token on stdout."""

    destination = bootstrap_url_path(port)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"{control_url_for(host, port)}#bootstrap={bootstrap_token}")
        os.chmod(temporary, 0o600)
        _replace_with_retry(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_automation_token(port: int, automation_token: str) -> Path:
    """Persist the lifecycle-bound verifier token without printing its value."""

    destination = automation_token_path(port)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as handle:
            handle.write(automation_token)
        os.chmod(temporary, 0o600)
        _replace_with_retry(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def bind_local_auth_tokens(env: dict[str, str]) -> tuple[str, str]:
    """Create independent browser-bootstrap and local-automation credentials."""

    bootstrap_token = secrets.token_urlsafe(32)
    automation_token = secrets.token_urlsafe(32)
    while automation_token == bootstrap_token:
        automation_token = secrets.token_urlsafe(32)
    env["WORKSTATION_BOOTSTRAP_TOKEN_HASH"] = hashlib.sha256(
        bootstrap_token.encode("utf-8")
    ).hexdigest()
    env["WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH"] = hashlib.sha256(
        automation_token.encode("utf-8")
    ).hexdigest()
    return bootstrap_token, automation_token


def fetch_status(host: str, port: int, timeout: float = 5.0) -> dict | None:
    try:
        request = urllib.request.Request(url_for(host, port, "/api/healthz"), headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status_code = response.status
        if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("status") != "ready":
            return None
        return {
            "reachable": True,
            "http_status": status_code,
            "service": payload.get("service"),
            "version": payload.get("version"),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def fetch_runtime_status(timeout: float = 3.0, port: int | None = None) -> dict | None:
    """Probe the already-running loopback runtime without starting it.

    The file-backed bearer token is used only in the request header and is
    never returned in lifecycle JSON or logs.
    """

    if port is None:
        try:
            port = runtime_service_port()
        except RuntimeError:
            return None
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        return None
    token_path = data_root() / "workspace" / "runtime" / "runtime.token"
    try:
        if token_path.is_symlink() or not token_path.is_file() or not 1 <= token_path.stat().st_size <= 512:
            return None
        token = token_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not 16 <= len(token) <= 256:
        return None
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/health",
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status_code = response.status
        if not isinstance(payload, dict) or payload.get("status") != "ready":
            return None
        return {
            "reachable": True,
            "http_status": status_code,
            "status": payload.get("status"),
            "version": payload.get("version"),
            "port": port,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def runtime_service_port() -> int:
    raw_port = os.environ.get("EVOMIND_RUNTIME_PORT", "8765").strip()
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError("EVOMIND_RUNTIME_PORT must be an integer") from error
    if not 1024 <= port <= 65535:
        raise RuntimeError("EVOMIND_RUNTIME_PORT is outside the allowed range")
    return port


def read_pid(port: int = 8088) -> int | None:
    pid_file, _, _, _ = runtime_paths(port)
    if not pid_file.exists():
        return None
    try:
        value = int(pid_file.read_text(encoding="utf-8").strip())
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def read_process_state(port: int = 8088) -> dict:
    _, state_file, _, _ = runtime_paths(port)
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _windows_pid_running(pid: int) -> bool:
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _windows_open_identity_handle(pid: int, *, terminate: bool = False) -> tuple[int, str, str] | None:
    process_query_limited_information = 0x1000
    process_terminate = 0x0001
    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    # Keep identity verification, termination and exit waiting bound to this
    # single kernel handle. SYNCHRONIZE is required by WaitForSingleObject;
    # opening a second handle after TerminateProcess would reintroduce a PID
    # reuse window.
    rights = process_query_limited_information | ((process_terminate | synchronize) if terminate else 0)
    handle = kernel32.OpenProcess(rights, False, pid)
    if not handle:
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    creation, exit_time, kernel_time, user_time = FileTime(), FileTime(), FileTime(), FileTime()
    size = ctypes.c_ulong(32768)
    image = ctypes.create_unicode_buffer(size.value)
    if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
        kernel32.CloseHandle(handle)
        return None
    if not kernel32.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
        kernel32.CloseHandle(handle)
        return None
    creation_token = str((creation.high << 32) | creation.low)
    return int(handle), creation_token, normalize_path(image.value)


def _windows_close_handle(handle: int) -> None:
    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(handle))


def command_line_argv(raw: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(raw)
    argc = ctypes.c_int()
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = shell32.CommandLineToArgvW(raw, ctypes.byref(argc))
    if not argv:
        raise RuntimeError("CommandLineToArgvW failed")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(argv)


def _windows_terminate_verified(record: dict, timeout: float = 15.0) -> tuple[bool, list[str]]:
    opened = _windows_open_identity_handle(int(record["pid"]), terminate=True)
    if not opened:
        if not pid_running(int(record["pid"])):
            return True, []
        return False, ["could not open verified termination handle"]
    handle, creation_token, executable = opened
    try:
        failures = []
        if creation_token != str(record.get("creation_token") or ""):
            failures.append("termination-handle creation time mismatch")
        if executable != str(record.get("executable") or ""):
            failures.append("termination-handle executable mismatch")
        if failures:
            return False, failures
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.TerminateProcess(ctypes.c_void_p(handle), 1):
            return False, [f"TerminateProcess failed: {ctypes.get_last_error()}"]
        wait_result = kernel32.WaitForSingleObject(
            ctypes.c_void_p(handle),
            max(0, min(int(timeout * 1000), 0xFFFFFFFE)),
        )
        if wait_result == 0:  # WAIT_OBJECT_0
            return True, []
        if wait_result == 258:  # WAIT_TIMEOUT
            return False, ["process did not exit"]
        return False, [f"WaitForSingleObject failed: {ctypes.get_last_error()}"]
    finally:
        _windows_close_handle(handle)


def _windows_process_cwd(pid: int) -> str:
    """Read the same-bitness process CurrentDirectory from its PEB.

    Dashboard/runtime binaries are x64 and launched by this x64 manager. A
    failed PEB read is an identity failure, never a reason to fall back to a
    recorded or guessed working directory.
    """

    process_query_information = 0x0400
    process_vm_read = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    ntdll = ctypes.WinDLL("ntdll")
    handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
    if not handle:
        return ""

    class ProcessBasicInformation(ctypes.Structure):
        _fields_ = [
            ("Reserved1", ctypes.c_void_p),
            ("PebBaseAddress", ctypes.c_void_p),
            ("Reserved2", ctypes.c_void_p * 2),
            ("UniqueProcessId", ctypes.c_void_p),
            ("Reserved3", ctypes.c_void_p),
        ]

    def read_memory(address: int, size: int) -> bytes:
        buffer = (ctypes.c_ubyte * size)()
        read = ctypes.c_size_t()
        if not kernel32.ReadProcessMemory(handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(read)) or read.value != size:
            raise OSError(ctypes.get_last_error(), "ReadProcessMemory failed")
        return bytes(buffer)

    try:
        basic = ProcessBasicInformation()
        returned = ctypes.c_ulong()
        status = ntdll.NtQueryInformationProcess(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic), ctypes.byref(returned)
        )
        if status != 0 or not basic.PebBaseAddress:
            return ""
        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        process_parameters_offset = 0x20 if pointer_size == 8 else 0x10
        parameters_raw = read_memory(int(basic.PebBaseAddress) + process_parameters_offset, pointer_size)
        parameters = int.from_bytes(parameters_raw, "little")
        if not parameters:
            return ""
        current_directory_offset = 0x38 if pointer_size == 8 else 0x24
        unicode_size = 16 if pointer_size == 8 else 8
        unicode = read_memory(parameters + current_directory_offset, unicode_size)
        length = int.from_bytes(unicode[0:2], "little")
        buffer_offset = 8 if pointer_size == 8 else 4
        buffer_address = int.from_bytes(unicode[buffer_offset:buffer_offset + pointer_size], "little")
        if not buffer_address or length <= 0 or length > 65534 or length % 2:
            return ""
        return read_memory(buffer_address, length).decode("utf-16-le", errors="strict").rstrip("\\/")
    except (OSError, UnicodeError, ValueError):
        return ""
    finally:
        kernel32.CloseHandle(handle)


def pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_running(pid)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def process_command_line(pid: int) -> str:
    identity = process_identity(pid)
    return str(identity.get("command_line_raw", "")) if identity else ""


def process_matches_dashboard(pid: int, port: int) -> bool:
    command = normalize_command_line(process_command_line(pid))
    if not command:
        return False
    try:
        next_cli = normalize_path(next_cli_path())
    except SystemExit:
        next_cli = ""
    return bool(next_cli and next_cli in command and " start " in f" {command} " and f"--port {int(port)}" in command)


def stop_pid(pid: int, timeout: float = 15.0) -> bool:
    if not pid_running(pid):
        return True
    try:
        if os.name == "nt":
            completed = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=max(1.0, timeout), check=False)
            if completed.returncode != 0 and pid_running(pid):
                return False
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and pid_running(pid):
        time.sleep(0.05)
    return not pid_running(pid)


def runtime_state_matches_process(pid: int, port: int, runtime_state: dict | None = None) -> bool:
    state = runtime_state or read_process_state(port)
    processes = state.get("processes") if isinstance(state.get("processes"), dict) else {}
    dashboard = processes.get("dashboard") if isinstance(processes.get("dashboard"), dict) else state
    return int(dashboard.get("pid") or 0) == int(pid) and int(dashboard.get("port") or port) == int(port) and process_matches_workstation(pid)


def read_runtime_state(port: int = 8088) -> dict:
    return read_process_state(port)


def port_processes(port: int, state: dict | None = None) -> tuple[list[int], list[int]]:
    owned: list[int] = []
    unowned: list[int] = []
    for pid in pids_on_port(port):
        (owned if runtime_state_matches_process(pid, port, state) else unowned).append(pid)
    return owned, unowned


def stop_port(port: int, state: dict | None = None) -> None:
    owned, unowned = port_processes(port, state)
    if unowned:
        raise SystemExit(json.dumps({"status": "failed", "stage": "port_ownership", "evidence": {"port": port, "unowned_pids": unowned}}, ensure_ascii=False))
    failed = [pid for pid in owned if not stop_pid(pid)]
    if failed:
        raise SystemExit(json.dumps({"status": "failed", "stage": "port_cleanup", "evidence": {"port": port, "failed_pids": failed}}, ensure_ascii=False))


def ensure_database_schema(environment: dict[str, str]) -> str:
    if not PRISMA_PUSH_SCRIPT.is_file():
        return "not_required"
    push = subprocess.run([node_command(), str(PRISMA_PUSH_SCRIPT), "--skip-generate"], cwd=APP_DIR, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if push.returncode != 0:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: database schema synchronization failed")
    generated = subprocess.run([npm_command(), "run", "db:generate"], cwd=APP_DIR, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if generated.returncode != 0:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: Prisma client generation failed")
    return "synced"


def normalize_command_line(value: str) -> str:
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    try:
        parts = shlex.split(raw, posix=False)
        raw = " ".join(part.strip('"') for part in parts)
    except ValueError:
        pass
    return raw.replace("\\", "/").casefold()


def normalize_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve(strict=False))).replace("\\", "/").casefold()


def process_identity(pid: int) -> dict:
    if not pid_running(pid):
        return {}
    try:
        if os.name == "nt":
            if psutil is not None:
                try:
                    process = psutil.Process(pid)
                    argv = process.cmdline()
                    raw_command = subprocess.list2cmdline(argv)
                    executable = process.exe()
                    cwd = process.cwd()
                    created_at = dt.datetime.fromtimestamp(process.create_time(), tz=dt.timezone.utc).isoformat()
                    handle_identity = _windows_open_identity_handle(pid)
                    if not raw_command or not executable or not cwd or not handle_identity:
                        return {}
                    handle, creation_token, handle_executable = handle_identity
                    _windows_close_handle(handle)
                    if handle_executable != normalize_path(executable):
                        return {}
                    return {
                        "pid": pid,
                        "creation_time": created_at,
                        "creation_token": creation_token,
                        "executable": normalize_path(executable),
                        "command_line": normalize_command_line(raw_command),
                        "command_line_raw": raw_command,
                        "cwd": normalize_path(cwd),
                    }
                except (psutil.Error, OSError, ValueError):
                    pass
            script = (
                "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
                f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId = {int(pid)}\" -ErrorAction SilentlyContinue; "
                "if($p){[ordered]@{pid=[int]$p.ProcessId;creation_time=$p.CreationDate.ToUniversalTime().ToString('o');"
                "executable=[string]$p.ExecutablePath;command_line=[string]$p.CommandLine}|ConvertTo-Json -Compress}"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if result.returncode or not result.stdout.strip():
                return {}
            payload = json.loads(result.stdout)
            raw_command = str(payload.get("command_line") or "")
            executable = str(payload.get("executable") or "")
            cwd = _windows_process_cwd(pid)
            handle_identity = _windows_open_identity_handle(pid)
            if not raw_command or not executable or not cwd or not handle_identity:
                return {}
            handle, creation_token, handle_executable = handle_identity
            _windows_close_handle(handle)
            if handle_executable != normalize_path(executable):
                return {}
            return {
                "pid": int(payload["pid"]),
                "creation_time": str(payload.get("creation_time") or ""),
                "creation_token": creation_token,
                "executable": normalize_path(executable),
                "command_line": normalize_command_line(raw_command),
                "command_line_raw": raw_command,
                "cwd": normalize_path(cwd),
            }
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        stat_tail = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "creation_time": stat_tail[19],
            "creation_token": stat_tail[19],
            "executable": normalize_path(Path(f"/proc/{pid}/exe").resolve()),
            "command_line": normalize_command_line(raw_command),
            "command_line_raw": raw_command,
            "cwd": normalize_path(Path(f"/proc/{pid}/cwd").resolve()),
        }
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError, subprocess.SubprocessError):
        return {}


def make_process_record(pid: int, *, role: str, port: int, cwd: Path, release_nonce: str) -> dict:
    identity: dict = {}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not identity:
        identity = process_identity(pid)
        if not identity:
            time.sleep(0.05)
    if not identity:
        raise RuntimeError(f"could not capture {role} process identity")
    return {
        "schema": "evomind.process_identity.v1",
        "role": role,
        "port": port,
        "pid": identity["pid"],
        "creation_time": identity["creation_time"],
        "creation_token": identity["creation_token"],
        "executable": identity["executable"],
        "command_line": identity["command_line"],
        "cwd": normalize_path(cwd),
        "install_dir": normalize_path(ROOT),
        "release_nonce": release_nonce,
    }


def verify_process_record(record: dict, *, require_listener: bool = True) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not isinstance(record, dict) or record.get("schema") != "evomind.process_identity.v1":
        return False, ["identity record schema mismatch"]
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False, ["identity PID is invalid"]
    actual = process_identity(pid)
    if not actual:
        return False, ["recorded process is not running or cannot be inspected"]
    if not str(record.get("creation_time") or "") or not str(actual.get("creation_time") or ""):
        failures.append("creation_time missing")
    for field in ("creation_token", "executable", "command_line"):
        if str(actual.get(field) or "") != str(record.get(field) or ""):
            failures.append(f"{field} mismatch")
    install_dir = normalize_path(record.get("install_dir", ""))
    cwd = normalize_path(record.get("cwd", ""))
    expected_install = normalize_path(ROOT)
    if install_dir != expected_install:
        failures.append("install directory mismatch")
    if cwd != install_dir and not cwd.startswith(f"{install_dir}/"):
        failures.append("working directory escapes install directory")
    if actual.get("cwd") and actual["cwd"] != cwd:
        failures.append("actual working directory mismatch")
    nonce = str(record.get("release_nonce") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", nonce) or f"evomind-{nonce}".casefold() not in str(actual.get("command_line") or ""):
        failures.append("release nonce mismatch")
    if install_dir not in str(actual.get("command_line") or ""):
        failures.append("command line is not bound to the install directory")
    port = record.get("port")
    if require_listener and (not isinstance(port, int) or pid not in pids_on_port(port)):
        failures.append("PID does not own the recorded listener")
    return not failures, failures


def process_matches_workstation(pid: int) -> bool:
    state = process_identity(pid)
    if not state:
        return False
    command = str(state.get("command_line") or "")
    return normalize_path(ROOT) in command and ("server.js" in command or "next/dist/bin/next" in command)


def pids_on_port(port: int) -> list[int]:
    if os.name == "nt":
        # ``Get-NetTCPConnection`` can block for its full timeout while the
        # CIM provider is busy, which previously made a successful contract
        # fail during identity verification and cleanup.  netstat is a native,
        # read-only snapshot and is sufficient even when the result is empty;
        # use PowerShell only if netstat itself is unavailable or fails.
        try:
            netstat = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"], text=True, capture_output=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            netstat = None
        pids: list[int] = []
        if netstat is not None and netstat.returncode == 0:
            suffix = f":{port}"
            for line in netstat.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(suffix) and parts[3].upper() == "LISTENING" and parts[-1].isdigit():
                    pids.append(int(parts[-1]))
            return sorted(set(pids))

        script = (
            f"@(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty OwningProcess -Unique) -join \"`n\""
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        if pids:
            return sorted(set(pids))
        return sorted(set(pids))

    result = subprocess.run(["sh", "-c", f"ss -ltnp 'sport = :{port}' 2>/dev/null || true"], text=True, capture_output=True, encoding="utf-8", errors="replace")
    pids: list[int] = []
    for token in result.stdout.replace(",", " ").split():
        if token.startswith("pid=") and token.removeprefix("pid=").isdigit():
            pids.append(int(token.removeprefix("pid=")))
    return sorted(set(pids))


def stop_process_record(record: dict, timeout: float = 15.0, *, require_listener: bool = True) -> dict:
    pid = int(record.get("pid") or 0)
    if not pid_running(pid):
        return {"stopped": True, "pid": pid, "already_exited": True, "conflict": False}
    matched, failures = verify_process_record(record, require_listener=require_listener)
    if not matched:
        return {"stopped": False, "pid": pid, "conflict": True, "failures": failures}
    try:
        if os.name == "nt":
            terminated, termination_failures = _windows_terminate_verified(record, timeout=timeout)
            return {
                "stopped": terminated,
                "pid": pid,
                "conflict": bool(termination_failures and any("mismatch" in item for item in termination_failures)),
                "failures": termination_failures,
            }
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        return {"stopped": not pid_running(pid), "pid": pid, "conflict": False, "failures": ["signal failed"]}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_running(pid):
            return {"stopped": True, "pid": pid, "conflict": False}
        time.sleep(0.2)
    return {"stopped": not pid_running(pid), "pid": pid, "conflict": False, "failures": ["process did not exit"]}


def stop_managed(host: str, port: int, timeout: float, include_listener: bool) -> tuple[list[int], bool, list[dict]]:
    pid = read_pid(port)
    state = read_process_state(port)
    health = fetch_status(host, port, timeout=2)
    runtime_port = int(state.get("runtime_port") or runtime_service_port())
    listeners_by_role = {
        "dashboard": set(pids_on_port(port)),
        "runtime": set(pids_on_port(runtime_port)),
    }
    process_records = state.get("processes") if isinstance(state.get("processes"), dict) else {}
    records = [
        (process_records.get("dashboard"), True),
        (process_records.get("runtime"), True),
        (process_records.get("dashboard_launcher"), False),
        (process_records.get("runtime_launcher"), False),
    ]
    stopped: list[int] = []
    conflicts: list[dict] = []
    known_listeners: dict[str, int] = {}
    known_pids: set[int] = set()
    for record, require_listener in records:
        if not isinstance(record, dict):
            continue
        role = str(record.get("role") or "")
        base_role = role.removesuffix("_launcher")
        if base_role not in listeners_by_role:
            conflicts.append({"role": role or "unknown", "pid": record.get("pid"), "failures": ["unknown process role"]})
            continue
        record_pid = int(record.get("pid") or 0)
        if require_listener:
            known_listeners[base_role] = record_pid
        result = stop_process_record(record, timeout=timeout, require_listener=require_listener)
        if result.get("stopped"):
            if not result.get("already_exited"):
                known_pids.add(record_pid)
                stopped.append(record_pid)
        elif not require_listener and result.get("conflict"):
            # Launcher shims are auxiliary and often exit immediately after
            # creating the real interpreter/Node listener. Their PID may be
            # reused before a later stop. Never signal the reused process and
            # do not let a stale, non-listening launcher record block cleanup
            # of the verified service identities.
            continue
        else:
            known_pids.add(record_pid)
            conflicts.append({"role": role, **result})
    if pid and known_listeners.get("dashboard") != pid and pid_running(pid):
        conflicts.append({"role": "dashboard", "pid": pid, "failures": ["pid file is not bound to the signed process identity record"]})
    if include_listener:
        for role, listeners in listeners_by_role.items():
            unknown = sorted(listener for listener in listeners if listener != known_listeners.get(role))
            if unknown:
                conflicts.append({"role": role, "listener_pids": unknown, "failures": ["listener identity is not managed by this lifecycle state"]})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [candidate for candidate in known_pids if candidate > 0 and pid_running(candidate)]
        dashboard_listeners = pids_on_port(port)
        runtime_listeners = pids_on_port(runtime_port)
        if not alive and not dashboard_listeners and not runtime_listeners:
            break
        time.sleep(0.25)
    remaining_dashboard = pids_on_port(port)
    remaining_runtime = pids_on_port(runtime_port)
    clean = not conflicts and not any(pid_running(candidate) for candidate in known_pids) and not remaining_dashboard and not remaining_runtime
    pid_file, state_file, _, _ = runtime_paths(port)
    runtime_pid_file, runtime_state_file, _, _ = runtime_service_paths(runtime_port)
    if clean or (not health and not known_pids and not remaining_dashboard and not remaining_runtime and not conflicts):
        pid_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)
        runtime_pid_file.unlink(missing_ok=True)
        runtime_state_file.unlink(missing_ok=True)
        remove_local_auth_files(port)
    return stopped, clean, conflicts


def build_source(env: dict[str, str]) -> dict[str, str | None]:
    """Build in an isolated tree and activate the result transactionally.

    Next writes directly to ``.next``.  Running it in the live source tree can
    therefore destroy the last known-good build before a compiler failure is
    reported.  The staging tree deliberately lives on the same volume as the
    application so the final directory renames are atomic filesystem
    operations.  This function only manipulates build directories; process
    lifecycle remains exclusively in ``start``/``stop_managed``.
    """

    staging_parent = SOURCE_APP_DIR / ".next-build-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="candidate-", dir=staging_parent))
    try:
        _copy_source_build_inputs(staging_root)
        (staging_root / "workspace").mkdir(parents=True, exist_ok=True)
        build_env = env.copy()
        build_env.update({
            "DATABASE_URL": f"file:{(staging_root / 'prisma' / 'build.db').as_posix()}",
            "WORKSTATION_ROOT": str(staging_root / "workspace"),
            "WORKSTATION_DATA_DIR": str(staging_root / "workspace"),
            "EVOMIND_LOCAL_SESSION_SECRET": hashlib.sha256(os.urandom(32)).hexdigest(),
        })
        result = subprocess.run(
            [node_command(), next_cli_path(), "build", "--webpack"],
            cwd=staging_root,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=build_env,
            timeout=_source_build_timeout(build_env),
        )
        if result.returncode != 0:
            raise SystemExit(
                json.dumps(
                    {
                        "status": "failed",
                        "stage": "build",
                        "active_build_preserved": (SOURCE_APP_DIR / ".next" / "BUILD_ID").is_file(),
                        "stdout": result.stdout[-6000:],
                        "stderr": result.stderr[-6000:],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        candidate = staging_root / ".next"
        _normalize_standalone_entry(candidate)
        candidate_build_id = _validate_source_build(candidate)
        manifest = runtime_build_manifest(candidate_build_id)
        write_runtime_build_manifest(candidate, manifest)
        rollback_dir = _activate_source_build(candidate, candidate_build_id, staging_root)
        if SOURCE_APP_DIR.resolve() == (ROOT / "web" / "research-agent-workstation").resolve():
            write_source_install_marker(manifest)
        return {
            "status": "built",
            "stage": "activate",
            "build_id": candidate_build_id,
            "active_dir": ".next",
            "rollback_dir": rollback_dir.name if rollback_dir else None,
        }
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "build_timeout",
                    "timeout_seconds": exc.timeout,
                    "active_build_preserved": (SOURCE_APP_DIR / ".next" / "BUILD_ID").is_file(),
                },
                ensure_ascii=False,
                indent=2,
            )
        ) from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            staging_parent.rmdir()
        except OSError:
            pass


def register_source_build() -> dict[str, object]:
    """Bind an already-built source tree after installer-controlled build/migration."""

    if bundle_mode():
        raise SystemExit("DASHBOARD_MANAGER_FAILED: source build registration is unavailable in bundle mode")
    active = SOURCE_APP_DIR / ".next"
    build_id = _validate_source_build(active)
    manifest = runtime_build_manifest(build_id)
    write_runtime_build_manifest(active, manifest)
    marker = write_source_install_marker(manifest)
    return {
        "status": "registered",
        "build_id": build_id,
        "manifest": str((active / "runtime-build-manifest.json").resolve()),
        "install_marker": str(marker.resolve()),
    }


def _source_build_timeout(env: dict[str, str]) -> float:
    raw = env.get("WORKSTATION_BUILD_TIMEOUT_SECONDS", "1800")
    try:
        value = float(raw)
    except ValueError:
        value = 1800.0
    return min(max(value, 60.0), 7200.0)


def _copy_source_build_inputs(destination: Path) -> None:
    """Copy only deterministic build inputs; never copy .env or runtime data."""

    source_files = (
        "package.json",
        "package-lock.json",
        "next.config.mjs",
        "tsconfig.json",
        "next-env.d.ts",
        "postcss.config.mjs",
        "tailwind.config.ts",
        ".eslintrc.json",
    )
    for name in source_files:
        source = SOURCE_APP_DIR / name
        if source.is_file():
            shutil.copy2(source, destination / name)

    for name in ("src", "public"):
        source = SOURCE_APP_DIR / name
        if source.is_dir():
            shutil.copytree(source, destination / name)

    prisma_source = SOURCE_APP_DIR / "prisma"
    prisma_destination = destination / "prisma"
    schema = prisma_source / "schema.prisma"
    migrations = prisma_source / "migrations"
    if schema.is_file():
        prisma_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(schema, prisma_destination / schema.name)
    if migrations.is_dir():
        prisma_destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(migrations, prisma_destination / "migrations")


def _read_build_id(build_dir: Path) -> str | None:
    try:
        value = (build_dir / "BUILD_ID").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if value and len(value) <= 256 else None


def _normalize_standalone_entry(candidate: Path) -> None:
    """Flatten Next's monorepo-style standalone app entry when necessary.

    Because the isolated build tree is nested beneath the real application (so
    it can resolve the existing node_modules without copying it), Next traces
    the app entry below ``standalone/<relative staging path>/server.js``.  The
    runtime contract is intentionally stable at ``standalone/server.js``.
    """

    standalone = candidate / "standalone"
    direct_server = standalone / "server.js"
    if direct_server.is_file() or not standalone.is_dir():
        return
    entries = [
        server.parent
        for server in standalone.rglob("server.js")
        if (server.parent / ".next").is_dir() and (server.parent / "package.json").is_file()
    ]
    if len(entries) != 1:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "normalize_standalone",
                    "app_entry_count": len(entries),
                    "active_build_preserved": (SOURCE_APP_DIR / ".next" / "BUILD_ID").is_file(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    nested = entries[0]
    for child in list(nested.iterdir()):
        target = standalone / child.name
        if target.exists():
            # The tracing root also emits a minimal package.json. Replace it
            # with the app package, matching a normal non-staged build layout.
            if child.name == "package.json" and target.is_file():
                target.unlink()
                _move_standalone_child(child, target)
                continue
            raise SystemExit(
                json.dumps(
                    {"status": "failed", "stage": "normalize_standalone", "conflict": child.name},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        _move_standalone_child(child, target)

    parent = nested
    while parent != standalone:
        next_parent = parent.parent
        try:
            parent.rmdir()
        except OSError:
            break
        parent = next_parent


def _validate_source_build(candidate: Path) -> str:
    build_id = _read_build_id(candidate)
    required = [candidate / "BUILD_ID", candidate / "standalone" / "server.js", candidate / "static"]
    missing = [str(path.relative_to(candidate)).replace("\\", "/") for path in required if not path.exists()]
    bundled_databases = [
        str(path.relative_to(candidate)).replace("\\", "/")
        for path in candidate.rglob("*")
        if path.is_file() and (
            path.suffix.casefold() in {".db", ".sqlite", ".sqlite3"}
            or path.name.casefold().endswith((".db-wal", ".db-shm"))
        )
    ]
    if not build_id or missing or bundled_databases:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "validate_candidate",
                    "build_id_present": bool(build_id),
                    "missing": missing,
                    "bundled_database_files": bundled_databases,
                    "active_build_preserved": (SOURCE_APP_DIR / ".next" / "BUILD_ID").is_file(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return build_id


def _replace_with_retry(source: Path, destination: Path, timeout: float = 15.0) -> None:
    """Retry only transient Windows sharing/access failures during directory switch."""

    deadline = time.monotonic() + max(0.0, timeout)
    delay = 0.05
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            retryable = isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def _move_standalone_child(source: Path, destination: Path) -> None:
    """Move one child while flattening a disposable standalone staging tree.

    Windows can deny os.replace for freshly-created traced directories even
    after the build process exits. This helper is deliberately narrower than
    shutil.move: it refuses to overwrite existing paths and only falls back to
    copy-then-remove for directories that are still inside the disposable
    .next-build-staging tree.
    """

    if destination.exists():
        raise FileExistsError(destination)
    try:
        _replace_with_retry(source, destination)
        return
    except PermissionError:
        staging_parent = (SOURCE_APP_DIR / ".next-build-staging").resolve()
        source_resolved = source.resolve()
        destination_parent_resolved = destination.resolve().parent
        if (
            not source.is_dir()
            or not source_resolved.is_relative_to(staging_parent)
            or not destination_parent_resolved.is_relative_to(staging_parent)
        ):
            raise
    shutil.copytree(source, destination, symlinks=True)
    shutil.rmtree(source)


def _activate_source_build(candidate: Path, candidate_build_id: str, staging_root: Path) -> Path | None:
    """Atomically select a validated candidate and retain the previous build."""

    active = SOURCE_APP_DIR / ".next"
    previous_build_id = _read_build_id(active)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rollback = SOURCE_APP_DIR / f".next-rollback-{stamp}-{secrets.token_hex(4)}" if active.exists() else None
    pointer = SOURCE_APP_DIR / ".next-rollback.json"
    pointer_temp = SOURCE_APP_DIR / f".next-rollback.json.tmp-{secrets.token_hex(4)}"
    pointer_payload = {
        "schema": "evomind.next.rollback.v1",
        "active_dir": ".next",
        "active_build_id": candidate_build_id,
        "rollback_dir": rollback.name if rollback else None,
        "rollback_build_id": previous_build_id,
        "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with pointer_temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(pointer_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    old_moved = False
    candidate_activated = False
    failed_candidate = staging_root / ".next-failed-activation"
    try:
        if rollback:
            _replace_with_retry(active, rollback)
            old_moved = True
        _replace_with_retry(candidate, active)
        candidate_activated = True
        _replace_with_retry(pointer_temp, pointer)
        return rollback
    except BaseException:
        # If any switch/pointer step fails, put the candidate back inside the
        # disposable staging tree and restore the exact previous directory.
        if candidate_activated and active.exists():
            try:
                _replace_with_retry(active, failed_candidate)
            except OSError:
                pass
        if old_moved and rollback and rollback.exists() and not active.exists():
            _replace_with_retry(rollback, active)
        raise
    finally:
        pointer_temp.unlink(missing_ok=True)


def sync_source_standalone_assets() -> None:
    """Make Next standalone output self-contained enough for local serving."""

    if not source_standalone_mode():
        return
    static_src = SOURCE_APP_DIR / ".next" / "static"
    static_dst = SOURCE_STANDALONE_SERVER.parent / ".next" / "static"
    if static_src.is_dir():
        if static_dst.exists():
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
    public_src = SOURCE_APP_DIR / "public"
    public_dst = SOURCE_STANDALONE_SERVER.parent / "public"
    if public_src.is_dir():
        if public_dst.exists():
            shutil.rmtree(public_dst)
        shutil.copytree(public_src, public_dst)


def launch_command(args: argparse.Namespace, env: dict[str, str]) -> tuple[list[str], Path, str]:
    if bundle_mode():
        return [node_command(), str(STANDALONE_SERVER)], STANDALONE_SERVER.parent, "standalone"
    stale = source_build_stale()
    if args.build or (stale and not args.no_auto_build):
        build_source(env)
        stale = source_build_stale()
    if stale:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: production build is missing or stale; rerun without --no-auto-build")
    if source_standalone_mode():
        sync_source_standalone_assets()
        return [node_command(), str(SOURCE_STANDALONE_SERVER)], SOURCE_STANDALONE_SERVER.parent, "source-standalone"
    return [node_command(), next_cli_path(), "start", "--hostname", args.host, "--port", str(args.port)], SOURCE_APP_DIR, "production"


def runtime_launch_command(release_nonce: str, port: int) -> tuple[list[str], Path]:
    configured = os.environ.get("WORKSTATION_PYTHON", "").strip()
    python = Path(configured).expanduser().resolve() if configured else Path(sys.executable).resolve()
    if not python.is_file():
        raise RuntimeError("EvoMind runtime Python is missing")
    # On Windows the Hermes/uv ``python.exe`` venv shim launches a second
    # console-subsystem Python process.  A detached shim can therefore still
    # create a conhost/Windows Terminal window for its child.  Use the sibling
    # GUI-subsystem launcher so the complete runtime chain remains windowless.
    # Hosted CI runners have no interactive console to hide and may allow
    # ``pythonw.exe`` to start without ever scheduling its runtime child, so CI
    # deliberately keeps the console executable under CREATE_NO_WINDOW.
    ci = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes", "on"}
    if os.name == "nt" and not ci and python.name.lower() == "python.exe":
        pythonw = python.with_name("pythonw.exe")
        if pythonw.is_file():
            python = pythonw
    # AgentRuntime owns the stable ``workspace/runtime`` suffix.  Pass the
    # data root itself here; passing ``data_root()/workspace`` creates
    # ``workspace/workspace/runtime`` and leaves the lifecycle health probe
    # reading a different token from the running service.
    workspace = data_root().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    code = (
        "from pathlib import Path; from evomind_runtime.http_server import serve; import sys; "
        "serve(Path(sys.argv[1]).resolve(), '127.0.0.1', int(sys.argv[2]))"
    )
    return [
        str(python), "-c", code, str(workspace), str(port),
        f"evomind-{release_nonce}", str(ROOT.resolve()),
    ], ROOT.resolve()


def launch_process_with_windows_fallback(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout,
    stderr,
    creationflags: int,
) -> subprocess.Popen:
    kwargs = {
        "cwd": cwd,
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }

    commands = [command]
    flags = [creationflags]
    if os.name == "nt":
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        without_breakaway = creationflags & ~breakaway
        if without_breakaway != creationflags:
            flags.append(without_breakaway)
        executable = Path(command[0])
        fallback = executable.with_name("python.exe")
        if executable.name.lower() == "pythonw.exe" and fallback.is_file():
            commands.append([str(fallback), *command[1:]])

    last_error: PermissionError | None = None
    for candidate_command in commands:
        for candidate_flags in flags:
            try:
                return subprocess.Popen(candidate_command, creationflags=candidate_flags, **kwargs)
            except PermissionError as exc:
                last_error = exc
    if last_error is None:
        raise AssertionError("managed process launch exhausted without an error")
    raise last_error


def wait_runtime_ready(port: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = fetch_runtime_status(timeout=min(2.0, max(0.5, timeout)), port=port)
        if status:
            return status
        time.sleep(0.25)
    _, _, out_log, err_log = runtime_service_paths(port)
    raise SystemExit(json.dumps({
        "status": "failed",
        "message": f"runtime did not become ready on 127.0.0.1:{port}",
        "stdout_log": str(out_log),
        "stderr_log": str(err_log),
    }, ensure_ascii=False, indent=2))


def wait_ready(host: str, port: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = fetch_status(host, port, timeout=min(3.0, max(0.5, timeout)))
        if status:
            return status
        time.sleep(0.5)
    _, _, out_log, err_log = runtime_paths(port)
    raise SystemExit(json.dumps({"status": "failed", "message": f"dashboard did not become ready on {url_for(host, port)}", "stdout_log": str(out_log), "stderr_log": str(err_log)}, ensure_ascii=False, indent=2))


def start(args: argparse.Namespace) -> None:
    # Compatibility contract retained by the v2 identity-bound cleanup below:
    # except SystemExit as readiness_error:
    # if stop_pid(process.pid):
    # otherwise runtime metadata was preserved for a fail-closed retry.
    if not hasattr(args, "host"):
        runtime_state = read_runtime_state()
        existing_pid = read_pid()
        try:
            fetch_status(args.port, timeout=2)
        except TypeError:
            pass
        owned, unowned = port_processes(args.port, runtime_state)
        if unowned:
            raise SystemExit(json.dumps({"status": "failed", "stage": "port_ownership", "evidence": {"port": args.port, "unowned_pids": unowned}}, ensure_ascii=False))
        if existing_pid and pid_running(existing_pid) and not runtime_state_matches_process(existing_pid, args.port, runtime_state):
            raise SystemExit(json.dumps({"status": "failed", "stage": "pid_ownership", "evidence": {"pid": existing_pid, "port": args.port}}, ensure_ascii=False))
        if owned:
            stop_port(args.port, runtime_state)
        return
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    runtime_port = runtime_service_port()
    existing = fetch_status(args.host, args.port, timeout=2)
    existing_pid = read_pid(args.port)
    if existing and not args.force:
        state = read_process_state(args.port)
        records = state.get("processes") if isinstance(state.get("processes"), dict) else {}
        dashboard_ok, dashboard_failures = verify_process_record(records.get("dashboard") or {})
        runtime_ok, runtime_failures = verify_process_record(records.get("runtime") or {})
        launcher_failures: dict[str, list[str]] = {}
        for launcher_name in ("dashboard_launcher", "runtime_launcher"):
            launcher = records.get(launcher_name)
            if isinstance(launcher, dict) and pid_running(launcher.get("pid")):
                launcher_ok, failures = verify_process_record(launcher, require_listener=False)
                if not launcher_ok:
                    launcher_failures[launcher_name] = failures
        runtime_health = fetch_runtime_status(timeout=2, port=runtime_port)
        if not dashboard_ok or not runtime_ok or not runtime_health or launcher_failures:
            raise SystemExit(json.dumps({
                "status": "failed",
                "stage": "existing_identity_conflict",
                "dashboard_failures": dashboard_failures,
                "runtime_failures": runtime_failures,
                "launcher_failures": launcher_failures,
                "runtime_health": bool(runtime_health),
            }, ensure_ascii=False, indent=2))
        pending_bootstrap = bootstrap_url_path(args.port)
        print(json.dumps({
            "status": "already_running",
            "url": control_url_for(args.host, args.port),
            "bootstrap_url_file": str(pending_bootstrap) if pending_bootstrap.is_file() else None,
            "bootstrap_token_exposed": False,
            "pid": existing_pid,
            "listener_pids": pids_on_port(args.port),
            "runtime_pid": records["runtime"]["pid"],
            "runtime_port": runtime_port,
            **existing,
        }, ensure_ascii=False, indent=2))
        return
    state_present = bool(read_process_state(args.port))
    if existing_pid or existing or state_present:
        _, clean, conflicts = stop_managed(args.host, args.port, args.timeout, include_listener=True)
        if not clean:
            raise SystemExit(json.dumps({"status": "failed", "stage": "pre_start_stop", "pid": existing_pid, "listener_pids": pids_on_port(args.port), "conflicts": conflicts}, ensure_ascii=False, indent=2))
    dashboard_occupants = pids_on_port(args.port)
    runtime_occupants = pids_on_port(runtime_port)
    if dashboard_occupants or runtime_occupants:
        raise SystemExit(json.dumps({
            "status": "failed",
            "stage": "port_in_use",
            "dashboard": {"port": args.port, "listener_pids": dashboard_occupants},
            "runtime": {"port": runtime_port, "listener_pids": runtime_occupants},
            "message": "a required port is occupied by a process outside the verified EvoMind identity state",
        }, ensure_ascii=False, indent=2))

    remove_local_auth_files(args.port)
    env = dashboard_env(args.host, args.port)
    bootstrap_token, automation_token = bind_local_auth_tokens(env)
    release_nonce = secrets.token_urlsafe(32)
    env["WORKSTATION_SESSION_SECRET"] = secrets.token_urlsafe(48)
    env["WORKSTATION_RELEASE_NONCE"] = release_nonce
    env["EVOMIND_RUNTIME_PORT"] = str(runtime_port)
    write_llm_route_snapshot(env)
    environment = env
    database_schema_status = ensure_database_schema(environment)
    if args.build:
        environment["WORKSTATION_DATABASE_SCHEMA_STATUS"] = database_schema_status
    command, cwd, mode = launch_command(args, env)
    bind_runtime_build_identity(env)
    command = [command[0], f"--title=evomind-{release_nonce}", *command[1:]]
    pid_file, state_file, out_log, err_log = runtime_paths(args.port)
    runtime_pid_file, runtime_state_file, runtime_out, runtime_err = runtime_service_paths(runtime_port)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    runtime_command, runtime_cwd = runtime_launch_command(release_nonce, runtime_port)
    isolated_runtime_env = runtime_env(env, release_nonce, runtime_port)
    runtime_stdout = runtime_out.open("ab")
    runtime_stderr = runtime_err.open("ab")
    try:
        runtime_process = launch_process_with_windows_fallback(
            runtime_command,
            cwd=runtime_cwd,
            env=isolated_runtime_env,
            stdout=runtime_stdout,
            stderr=runtime_stderr,
            creationflags=creationflags,
        )
    finally:
        runtime_stdout.close()
        runtime_stderr.close()
    runtime_launcher_record = make_process_record(runtime_process.pid, role="runtime_launcher", port=runtime_port, cwd=runtime_cwd, release_nonce=release_nonce)
    runtime_pid_file.write_text(str(runtime_process.pid), encoding="utf-8")
    process: subprocess.Popen | None = None
    dashboard_record: dict | None = None
    try:
        runtime_ready = wait_runtime_ready(runtime_port, args.timeout)
        runtime_listeners = pids_on_port(runtime_port)
        if len(runtime_listeners) != 1:
            raise RuntimeError(f"runtime listener identity mismatch: {runtime_listeners}")
        runtime_record = make_process_record(runtime_listeners[0], role="runtime", port=runtime_port, cwd=runtime_cwd, release_nonce=release_nonce)
        runtime_matched, runtime_failures = verify_process_record(runtime_record)
        if not runtime_matched:
            raise RuntimeError(f"runtime identity verification failed: {runtime_failures}")

        stdout = out_log.open("ab")
        stderr = err_log.open("ab")
        try:
            process = launch_process_with_windows_fallback(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
        finally:
            stdout.close()
            stderr.close()
        dashboard_launcher_record = make_process_record(process.pid, role="dashboard_launcher", port=args.port, cwd=cwd, release_nonce=release_nonce)
        pid_file.write_text(str(process.pid), encoding="utf-8")
        ready = wait_ready(args.host, args.port, args.timeout)
        listener_pids = pids_on_port(args.port)
        if len(listener_pids) != 1:
            raise RuntimeError(f"dashboard listener identity mismatch: {listener_pids}")
        dashboard_record = make_process_record(listener_pids[0], role="dashboard", port=args.port, cwd=cwd, release_nonce=release_nonce)
        dashboard_matched, dashboard_failures = verify_process_record(dashboard_record)
        if not dashboard_matched:
            raise RuntimeError(f"dashboard identity verification failed: {dashboard_failures}")
        write_automation_token(args.port, automation_token)
        bootstrap_file = write_bootstrap_url(args.host, args.port, bootstrap_token)
    except BaseException:
        if dashboard_record:
            stop_process_record(dashboard_record, timeout=10)
        elif process is not None:
            for listener in pids_on_port(args.port):
                try:
                    candidate = make_process_record(listener, role="dashboard", port=args.port, cwd=cwd, release_nonce=release_nonce)
                    stop_process_record(candidate, timeout=10)
                except Exception:
                    pass
            try:
                provisional = make_process_record(process.pid, role="dashboard", port=args.port, cwd=cwd, release_nonce=release_nonce)
                stop_process_record(provisional, timeout=10, require_listener=False)
            except Exception:
                pass
        if process is not None and 'dashboard_launcher_record' in locals() and dashboard_launcher_record["pid"] != (dashboard_record or {}).get("pid"):
            stop_process_record(dashboard_launcher_record, timeout=10, require_listener=False)
        if 'runtime_record' in locals():
            stop_process_record(runtime_record, timeout=10)
        else:
            for listener in pids_on_port(runtime_port):
                try:
                    candidate = make_process_record(listener, role="runtime", port=runtime_port, cwd=runtime_cwd, release_nonce=release_nonce)
                    stop_process_record(candidate, timeout=10)
                except Exception:
                    pass
        if runtime_launcher_record["pid"] != (locals().get("runtime_record") or {}).get("pid"):
            stop_process_record(runtime_launcher_record, timeout=10, require_listener=False)
        pid_file.unlink(missing_ok=True)
        runtime_pid_file.unlink(missing_ok=True)
        remove_local_auth_files(args.port)
        raise
    pid_file.write_text(str(dashboard_record["pid"]), encoding="utf-8")
    runtime_pid_file.write_text(str(runtime_record["pid"]), encoding="utf-8")
    processes = {"dashboard": dashboard_record, "runtime": runtime_record}
    if dashboard_launcher_record["pid"] != dashboard_record["pid"]:
        launcher_ok, _ = verify_process_record(dashboard_launcher_record, require_listener=False)
        if launcher_ok:
            processes["dashboard_launcher"] = dashboard_launcher_record
    if runtime_launcher_record["pid"] != runtime_record["pid"]:
        launcher_ok, _ = verify_process_record(runtime_launcher_record, require_listener=False)
        if launcher_ok:
            processes["runtime_launcher"] = runtime_launcher_record
    state = {
        "schema": "evomind.lifecycle_state.v2",
        "host": args.host,
        "port": args.port,
        "runtime_port": runtime_port,
        "mode": mode,
        "started_at": time.time(),
        "release_nonce": release_nonce,
        "processes": processes,
    }
    atomic_json(state_file, state)
    atomic_json(runtime_state_file, {"schema": "evomind.runtime_state.v1", "process": runtime_record, "release_nonce": release_nonce})
    print(json.dumps({
        "status": "started",
        "url": control_url_for(args.host, args.port),
        "bootstrap_url_file": str(bootstrap_file),
        "bootstrap_token_exposed": False,
        "health_url": url_for(args.host, args.port, "/api/healthz"),
        "runtime_health": runtime_ready,
        "pid": dashboard_record["pid"],
        "listener_pids": [dashboard_record["pid"]],
        "runtime_pid": runtime_record["pid"],
        "mode": mode,
        "started_at": state["started_at"],
        **ready,
    }, ensure_ascii=False, indent=2))


def stop(args: argparse.Namespace, emit: bool = True) -> None:
    if not hasattr(args, "host"):
        pid_path = PID_FILE
        state_path = STATE_FILE
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.is_file() else None
        except (OSError, ValueError):
            pid = None
        state = read_runtime_state() if state_path.is_file() else {}
        if pid and pid_running(pid) and not runtime_state_matches_process(pid, args.port, state):
            raise SystemExit(json.dumps({"status": "failed", "stage": "pid_ownership", "evidence": {"pid": pid, "port": args.port}}, ensure_ascii=False))
        if pid and runtime_state_matches_process(pid, args.port, state):
            stop_port(args.port, state)
        pid_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        remove_local_auth_files(args.port)
        if emit:
            print(json.dumps({"status": "stopped"}))
        return
    pid = read_pid(args.port)
    health = fetch_status(args.host, args.port, timeout=2)
    stopped, clean, conflicts = stop_managed(args.host, args.port, args.timeout, include_listener=True)
    status_value = "stopped" if clean and (pid or health or stopped) else "not_running" if clean else "still_running"
    runtime_port = runtime_service_port()
    payload = {
        "status": status_value,
        "pid": pid,
        "stopped_pids": stopped,
        "conflicts": conflicts,
        "listener_pids_after": pids_on_port(args.port),
        "runtime_listener_pids_after": pids_on_port(runtime_port),
        "port_released": not pids_on_port(args.port),
        "runtime_port_released": not pids_on_port(runtime_port),
    }
    if emit:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not clean:
        raise SystemExit(1)


def stop_legacy_v1(args: argparse.Namespace) -> None:
    """One-time, fail-closed retirement of the pre-v2 source dashboard.

    This path never adopts the process into v2 state. It proves the complete
    legacy PID/listener/argv/cwd/time/health tuple, then terminates through the
    same identity-bound Windows handle used by normal v2 lifecycle control.
    """

    if os.name != "nt" or bundle_mode():
        raise SystemExit("LEGACY_STOP_FAILED: legacy source retirement is Windows source-mode only")
    pid_file, state_file, _, _ = runtime_paths(args.port)
    if not pid_file.is_file() or pid_file.is_symlink() or not state_file.is_file() or state_file.is_symlink():
        raise SystemExit("LEGACY_STOP_FAILED: legacy PID/state files are missing or linked")
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"LEGACY_STOP_FAILED: malformed legacy lifecycle state: {error}") from error
    expected_state_keys = {"launcher_pid", "listener_pids", "managed_pid", "host", "port", "mode", "started_at"}
    if (
        not isinstance(state, dict)
        or set(state) != expected_state_keys
        or state.get("launcher_pid") != pid
        or state.get("managed_pid") != pid
        or state.get("listener_pids") != [pid]
        or state.get("host") != "127.0.0.1"
        or state.get("port") != args.port
        or state.get("mode") != "source-standalone"
        or not isinstance(state.get("started_at"), (int, float))
    ):
        raise SystemExit("LEGACY_STOP_FAILED: legacy lifecycle state contract mismatch")
    listeners = pids_on_port(args.port)
    if listeners != [pid] or not pid_running(pid):
        raise SystemExit("LEGACY_STOP_FAILED: legacy PID no longer exclusively owns the dashboard listener")
    identity = process_identity(pid)
    if not identity:
        raise SystemExit("LEGACY_STOP_FAILED: live legacy process identity is unavailable")
    expected_cwd = normalize_path(SOURCE_STANDALONE_SERVER.parent.resolve())
    argv = command_line_argv(str(identity.get("command_line_raw") or ""))
    if (
        len(argv) != 2
        or normalize_path(argv[0]) != identity.get("executable")
        or normalize_path(argv[1]) != normalize_path(SOURCE_STANDALONE_SERVER.resolve())
        or identity.get("cwd") != expected_cwd
    ):
        raise SystemExit("LEGACY_STOP_FAILED: legacy executable, argv, or cwd binding mismatch")
    try:
        created = dt.datetime.fromisoformat(str(identity["creation_time"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("LEGACY_STOP_FAILED: legacy creation time is invalid") from error
    if abs(float(state["started_at"]) - created) > 10.0:
        raise SystemExit("LEGACY_STOP_FAILED: legacy state/process creation time mismatch")
    health = fetch_status(args.host, args.port, timeout=3)
    if (
        not health
        or health.get("http_status") != 200
        or health.get("service") != "evomind-workstation"
        or str(health.get("version") or "") != application_version()
    ):
        raise SystemExit("LEGACY_STOP_FAILED: legacy dashboard health identity mismatch")
    if pids_on_port(args.port) != [pid]:
        raise SystemExit("LEGACY_STOP_FAILED: listener identity changed before termination")
    terminated, failures = _windows_terminate_verified({
        "pid": pid,
        "creation_token": identity["creation_token"],
        "executable": identity["executable"],
    }, timeout=args.timeout)
    if not terminated or failures or pids_on_port(args.port):
        raise SystemExit(f"LEGACY_STOP_FAILED: identity-bound termination failed: {failures}")
    pid_file.unlink(missing_ok=False)
    state_file.unlink(missing_ok=False)
    remove_local_auth_files(args.port)
    print(json.dumps({
        "status": "legacy_v1_stopped",
        "pid": pid,
        "port": args.port,
        "identity_verified": True,
        "same_handle_termination": True,
        "port_released": True,
    }, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    pid = read_pid(args.port)
    health = fetch_status(args.host, args.port, timeout=3)
    listeners = pids_on_port(args.port)
    state = read_process_state(args.port)
    process_records = state.get("processes") if isinstance(state.get("processes"), dict) else {}
    dashboard_record = process_records.get("dashboard") or {}
    runtime_record = process_records.get("runtime") or {}
    dashboard_identity, dashboard_failures = verify_process_record(dashboard_record)
    runtime_identity, runtime_failures = verify_process_record(runtime_record)
    alive = pid_running(pid)
    consistent = dashboard_identity and (pid == dashboard_record.get("pid")) and dashboard_record.get("pid") in listeners
    runtime_port = int(state.get("runtime_port") or runtime_service_port())
    runtime_health = fetch_runtime_status(timeout=2, port=runtime_port)
    if health is not None:
        health = {
            **health,
            "has_runtime": runtime_health is not None and runtime_identity,
            "runtime": runtime_health if runtime_identity else None,
        }
    print(json.dumps({
        "status": "running" if health else "port_conflict" if listeners else "not_reachable",
        "url": control_url_for(args.host, args.port),
        "health_url": url_for(args.host, args.port, "/api/healthz"),
        "pid": pid,
        "pid_running": alive,
        "listener_pids": listeners,
        "process_port_consistent": consistent,
        "identity_contract": "evomind.process_identity.v1",
        "dashboard_identity_verified": dashboard_identity,
        "process_identity_failures": dashboard_failures,
        "runtime_process_consistent": runtime_identity and runtime_health is not None,
        "runtime_identity_verified": runtime_identity,
        "runtime_identity_failures": runtime_failures,
        "runtime_pid": runtime_record.get("pid"),
        "runtime_port": runtime_service_port(),
        "runtime_pid_running": pid_running(runtime_record.get("pid")),
        "runtime_listener_pids": pids_on_port(runtime_service_port()),
        "health": health,
        "layout": "standalone" if bundle_mode() else "source",
        "source_build_stale": source_build_stale() if not bundle_mode() else False,
        "runtime_dir": str(runtime_dir()),
        "bootstrap_pending": bootstrap_url_path(args.port).is_file(),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the loopback-only Research Agent Workstation dashboard.")
    parser.add_argument("command", choices=["start", "stop", "stop-legacy-v1", "restart", "status", "register-source-build"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    if os.environ.get("WORKSTATION_HOST"):
        parser.set_defaults(host=os.environ["WORKSTATION_HOST"])
    if os.environ.get("WORKSTATION_PORT"):
        parser.set_defaults(port=int(os.environ["WORKSTATION_PORT"]))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--build", action="store_true", help="Force a fresh source build before start/restart.")
    parser.add_argument("--no-auto-build", action="store_true", help="Fail instead of rebuilding a missing/stale source build.")
    parser.add_argument("--force", action="store_true", help="Restart a reachable managed workstation instance.")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: local-first mode requires a loopback host")
    if not 1 <= args.port <= 65535:
        raise SystemExit("DASHBOARD_MANAGER_FAILED: invalid port")
    if args.command == "register-source-build":
        print(json.dumps(register_source_build(), ensure_ascii=False, indent=2))
    elif args.command == "start":
        start(args)
    elif args.command == "stop":
        stop(args)
    elif args.command == "stop-legacy-v1":
        stop_legacy_v1(args)
    elif args.command == "restart":
        args.force = True
        stop(args, emit=False)
        start(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
