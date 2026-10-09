from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_HEADER = "x-evomind-local-automation"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{24,256}\Z")
SESSION_COOKIE = "evomind_local_session"
CSRF_HEADER = "x-evomind-csrf"
SESSION_VALUE_PATTERN = re.compile(r"[A-Za-z0-9._-]{20,2048}\Z")
BOOTSTRAP_FRAGMENT = re.compile(r"#bootstrap=([A-Za-z0-9_-]{24,256})\s*\Z")
SAFE_METHODS = {"GET", "HEAD"}


def runtime_dir() -> Path:
    configured = os.environ.get("WORKSTATION_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if (ROOT / "app" / "server.js").is_file():
        data_root = os.environ.get("WORKSTATION_DATA_DIR", "").strip()
        return (Path(data_root).expanduser().resolve() if data_root else (ROOT / "user-data").resolve()) / "logs"
    return ROOT / "web" / "research-agent-workstation" / ".runtime-logs"


def automation_token_path(port: int) -> Path:
    suffix = "" if port == 8088 else f".{port}"
    return runtime_dir() / f"dashboard{suffix}.automation.token"


def _suffix(port: int) -> str:
    return "" if port == 8088 else f".{port}"


def bootstrap_url_path(port: int) -> Path:
    return runtime_dir() / f"dashboard{_suffix(port)}.bootstrap.once"


def test_session_path(port: int) -> Path:
    """Lifecycle-bound browser session obtained through the one-time bootstrap.

    It lives next to the local automation token, is removed by
    ``manage_workstation_dashboard.py stop`` and dies with the server's
    per-start session secret.
    """

    return runtime_dir() / f"dashboard{_suffix(port)}.test-session.json"


def _loopback_port(base_url: str) -> int | None:
    try:
        parsed_base = urlparse(base_url)
        port = parsed_base.port or 80
    except ValueError:
        return None
    if parsed_base.scheme != "http" or parsed_base.hostname not in {"127.0.0.1", "localhost"}:
        return None
    return port


def local_test_session(base_url: str) -> dict[str, str]:
    """Return {cookie, csrf} for a loopback server, or {} when none exists."""

    port = _loopback_port(base_url)
    if port is None:
        return {}
    target = test_session_path(port)
    try:
        if target.is_symlink() or not target.is_file() or target.stat().st_size > 8192:
            return {}
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    cookie = payload.get("cookie") if isinstance(payload, dict) else None
    csrf = payload.get("csrf") if isinstance(payload, dict) else None
    if payload.get("port") != port or not isinstance(cookie, str) or not isinstance(csrf, str):
        return {}
    if not SESSION_VALUE_PATTERN.fullmatch(cookie) or not SESSION_VALUE_PATTERN.fullmatch(csrf):
        return {}
    return {"cookie": cookie, "csrf": csrf}


def establish_local_test_session(base_url: str, timeout: float = 15.0) -> str:
    """Exchange the server's one-time bootstrap token for a real local session.

    This is the same flow a person uses when opening the one-time launch URL:
    the token is single-use, only accepted on a loopback host, and the
    resulting principal is the product's ordinary ``local`` user. Nothing is
    printed; the session is written with owner-only permissions.
    """

    port = _loopback_port(base_url)
    if port is None:
        raise RuntimeError("test_session_requires_loopback_http")
    if local_test_session(base_url):
        return "reused"
    source = bootstrap_url_path(port)
    try:
        match = BOOTSTRAP_FRAGMENT.search(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError("bootstrap_url_unavailable") from exc
    if not match:
        raise RuntimeError("bootstrap_url_invalid")
    origin = base_url.rstrip("/")
    opener = urllib.request.build_opener()
    body = json.dumps({"token": match.group(1)}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{origin}/api/session/bootstrap",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Origin": origin},
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            set_cookies = response.headers.get_all("Set-Cookie") or []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("bootstrap_exchange_failed") from exc
    cookie = ""
    for header in set_cookies:
        name, _, rest = header.partition("=")
        if name.strip() == SESSION_COOKIE:
            cookie = rest.split(";", 1)[0].strip()
    csrf = payload.get("csrf_token") if isinstance(payload, dict) else ""
    if payload.get("ok") is not True or not cookie or not isinstance(csrf, str):
        raise RuntimeError("bootstrap_exchange_rejected")
    target = test_session_path(port)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"port": port, "cookie": cookie, "csrf": csrf}, handle)
    os.replace(temporary, target)
    try:
        source.unlink()
    except OSError:
        pass
    return "bootstrapped"


def remove_local_test_session(base_url: str) -> None:
    port = _loopback_port(base_url)
    if port is not None:
        test_session_path(port).unlink(missing_ok=True)


def automation_token(base_url: str) -> str:
    port = _loopback_port(base_url)
    if port is None:
        return ""
    target = automation_token_path(port)
    try:
        if target.is_symlink() or not target.is_file() or not 24 <= target.stat().st_size <= 256:
            return ""
        token = target.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if not TOKEN_PATTERN.fullmatch(token):
        return ""
    return token


def authenticated_headers(
    base_url: str,
    headers: dict[str, str] | None = None,
    *,
    method: str | None = None,
) -> dict[str, str]:
    """Headers for a release check against a local workstation.

    Read-only checks (``method="GET"``/``"HEAD"``) use the bootstrapped local
    session when one exists, so principal-scoped routes are verified as the
    signed-in user would see them. Everything else keeps the local automation
    header, which deliberately carries no user principal.
    """

    result = dict(headers or {})
    if method is not None and method.upper() in SAFE_METHODS:
        session = local_test_session(base_url)
        if session:
            result["Cookie"] = f"{SESSION_COOKIE}={session['cookie']}"
            return result
    token = automation_token(base_url)
    if token:
        result[AUTOMATION_HEADER] = token
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local release-check session helper")
    parser.add_argument("action", choices=["establish", "remove", "status"])
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "establish":
            print(json.dumps({"ok": True, "session": establish_local_test_session(args.base_url)}))
        elif args.action == "remove":
            remove_local_test_session(args.base_url)
            print(json.dumps({"ok": True, "session": "removed"}))
        else:
            print(json.dumps({"ok": True, "session": "present" if local_test_session(args.base_url) else "absent"}))
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
