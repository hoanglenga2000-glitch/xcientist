from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_HEADER = "x-evomind-local-automation"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{24,256}\Z")


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


def automation_token(base_url: str) -> str:
    try:
        parsed_base = urlparse(base_url)
        port = parsed_base.port or 80
    except ValueError:
        return ""
    if parsed_base.scheme != "http" or parsed_base.hostname not in {"127.0.0.1", "localhost"}:
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


def authenticated_headers(base_url: str, headers: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(headers or {})
    token = automation_token(base_url)
    if token:
        result[AUTOMATION_HEADER] = token
    return result
