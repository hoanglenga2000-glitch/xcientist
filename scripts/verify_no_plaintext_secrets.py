from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Scan the repository inventory returned by Git. Only generated/vendor/cache
# trees and explicit synthetic test-fixture corpora are excluded. A directory
# named ``_quarantine`` is not a security boundary: tracked files there remain
# public Git objects and must be scanned like every other first-party source.
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime-logs",
    ".tox",
    ".venv",
    "__pycache__",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "pip-cache",
    "venv",
}
TEST_FIXTURE_DIRS = {
    "__fixtures__",
    "fixture",
    "fixtures",
    "test-data",
    "test_data",
    "testdata",
}
TEST_FIXTURE_FILES = {"tests/test_credential_scanner.py"}
BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".arrow",
    ".avi",
    ".bin",
    ".bz2",
    ".db",
    ".dll",
    ".doc",
    ".docx",
    ".dylib",
    ".exe",
    ".feather",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".otf",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pyc",
    ".pyd",
    ".pyo",
    ".rar",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".ttf",
    ".wav",
    ".webp",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".xz",
    ".zip",
}

SECRET_PATTERNS = [
    re.compile(r"\b(?P<value>sk-(?:ant|proj|live|test|ya)[A-Za-z0-9_-]{12,})\b"),
    re.compile(r"\b(?P<value>KGAT_[A-Za-z0-9_-]{16,})\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|KAGGLE_KEY|KAGGLE_API_TOKEN)\s*=\s*"
        r"['\"]?(?P<value>[A-Za-z0-9_-]{16,})"
    ),
    re.compile(r"(?i)\bpassword\s*=\s*['\"](?P<value>[^'\"]{6,})['\"]"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|token|secret)\s*[:=]\s*"
        r"['\"](?P<value>[A-Za-z0-9_./+=-]{16,})['\"]"
    ),
]

ALLOWED_PLACEHOLDERS = {
    "<redacted-password>",
    "<required>",
    "<rotated-anthropic-key>",
    "<your-api-key>",
    "[REDACTED]",
}
PLACEHOLDER_VALUE_PATTERN = re.compile(
    r"(?i)^(?:dummy|example|fake|fixture|placeholder|redacted|required|test)"
    r"(?:[-_][A-Za-z0-9]+)*$"
)
TEST_PROVIDER_TOKEN_PATTERN = re.compile(r"(?i)^sk-test(?:[-_][A-Za-z0-9]+)*$")

SECRET_NAME_PATTERN = re.compile(
    r"(?i)(?:password|passwd|passphrase|api[_-]?key|access[_-]?token|secret|private[_-]?key|"
    r"(?:^|_)(?:pass|pwd)(?:$|_))"
)
ENV_REFERENCE_PATTERN = re.compile(
    r"^(?:\$\{?[A-Z][A-Z0-9_]*\}?|%[A-Z][A-Z0-9_]*%|[A-Z][A-Z0-9_]{2,})$"
)
CONFIG_REFERENCE_PATTERN = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*$"
)
CREDENTIAL_METADATA_NAME_PATTERN = re.compile(
    r"(?i)(?:status|policy|storage|username|user_name|path|env|name)$"
)
TEXT_QUOTED_EQUALS_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\s*=\s*"
    r"(?P<value_quote>['\"])(?P<value>[^'\"\r\n]{8,})(?P=value_quote)"
)
TEXT_QUOTED_COLON_PATTERN = re.compile(
    r"(?i)(?:^\s*|[{,]\s*)(?P<key_quote>['\"]?)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_-]*)(?P=key_quote)\s*:\s*"
    r"(?P<value_quote>['\"])(?P<value>[^'\"\r\n]{8,})(?P=value_quote)"
)
REGEX_DEFINITION_PATTERN = re.compile(
    r"(?i)(?:\bre\.compile\s*\(|\b(?:new\s+)?regexp\s*\(|"
    r"\b[A-Za-z0-9_-]*(?:regex|regexp|pattern)[A-Za-z0-9_-]*\s*[:=])"
)
BATCH_LITERAL_ECHO_PATTERN = re.compile(
    r"(?i)^\s*@?echo\s+(?!off\b|on\b|%|<)(?P<value>[^\s]{8,})\s*$"
)
SENSITIVE_POSITIONAL_ARGUMENTS = {
    "auth_password": 1,
    "send_password": 0,
    "sendline": 0,
}

Finding = dict[str, object]


def _relative_name(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _finding(
    path: Path,
    root: Path,
    pattern: str,
    line: int = 0,
    column: int = 0,
    error_type: str | None = None,
) -> Finding:
    finding: Finding = {
        "file": _relative_name(path, root),
        "line": line,
        "column": column,
        "pattern": pattern,
    }
    if error_type:
        finding["error_type"] = error_type
    return finding


def _assignment_name(target: ast.expr) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
        return str(target.slice.value)
    return ""


def _is_placeholder_value(text: str) -> bool:
    value = text.strip()
    return (
        value in ALLOWED_PLACEHOLDERS
        or bool(PLACEHOLDER_VALUE_PATTERN.fullmatch(value))
        or bool(TEST_PROVIDER_TOKEN_PATTERN.fullmatch(value))
        or bool(ENV_REFERENCE_PATTERN.fullmatch(value))
    )


def _is_secret_text(text: str) -> bool:
    value = text.strip()
    return len(value) >= 8 and not _is_placeholder_value(value)


def _is_literal_secret(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and _is_secret_text(value.value)
    )


def _is_credential_name(name: str) -> bool:
    if "/" in name or "\\" in name:
        return False
    return bool(
        SECRET_NAME_PATTERN.search(name)
        and not CREDENTIAL_METADATA_NAME_PATTERN.search(name)
    )


def _python_literal_secret_locations(tree: ast.AST) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_literal_secret(node.value):
            for target in node.targets:
                if _is_credential_name(_assignment_name(target)):
                    locations.append((node.lineno, node.col_offset + 1))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and _is_literal_secret(node.value):
            if _is_credential_name(_assignment_name(node.target)):
                locations.append((node.lineno, node.col_offset + 1))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_credential_name(key.value)
                    and _is_literal_secret(value)
                    and not CONFIG_REFERENCE_PATTERN.fullmatch(value.value)
                ):
                    locations.append((getattr(key, "lineno", node.lineno), getattr(key, "col_offset", 0) + 1))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
                name = str(node.args[0].value)
                if _is_credential_name(name) and _is_literal_secret(node.args[1]):
                    locations.append((node.lineno, node.col_offset + 1))
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and _is_credential_name(keyword.arg) and _is_literal_secret(keyword.value):
                    locations.append((node.lineno, node.col_offset + 1))
            function_name = ""
            if isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                function_name = node.func.id
            secret_index = SENSITIVE_POSITIONAL_ARGUMENTS.get(function_name)
            if secret_index is not None and len(node.args) > secret_index:
                if _is_literal_secret(node.args[secret_index]):
                    locations.append((node.lineno, node.col_offset + 1))
    return sorted(locations)


def python_literal_secret_assignments(path: Path) -> list[int]:
    """Return one line number per Python literal credential assignment."""
    try:
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    return [line for line, _ in _python_literal_secret_locations(tree)]


def _quoted_literal_end(line: str, start: int) -> int:
    quote_index = -1
    quote = ""
    for index in range(start, len(line)):
        if line[index] in {"'", '"'}:
            quote_index = index
            quote = line[index]
            break
    if quote_index < 0:
        return start
    escaped = False
    for index in range(quote_index + 1, len(line)):
        char = line[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return index + 1
    return len(line)


def _regex_definition_ranges(line: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in REGEX_DEFINITION_PATTERN.finditer(line):
        ranges.append((match.start(), _quoted_literal_end(line, match.end())))
    return ranges


def _inside_ranges(offset: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in ranges)


def _text_literal_secret_locations(text: str) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        regex_ranges = _regex_definition_ranges(line)
        for pattern in (TEXT_QUOTED_EQUALS_PATTERN, TEXT_QUOTED_COLON_PATTERN):
            for match in pattern.finditer(line):
                if _inside_ranges(match.start(), regex_ranges):
                    continue
                if _is_credential_name(match.group("name")) and _is_secret_text(match.group("value")):
                    locations.append((line_no, match.start() + 1))
    return sorted(locations)


def text_literal_secret_assignments(path: Path) -> list[int]:
    """Return one line number per non-Python literal credential assignment."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return []
    return [line for line, _ in _text_literal_secret_locations(text)]


def should_skip(path: Path, root: Path = ROOT) -> bool:
    try:
        relative = path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False
    parts = tuple(part.casefold() for part in relative.parts)
    if any(part in SKIP_DIRS or part in TEST_FIXTURE_DIRS for part in parts):
        return True
    if relative.as_posix().casefold() in TEST_FIXTURE_FILES:
        return True
    return path.suffix.casefold() in BINARY_SUFFIXES


def discover_candidate_files(root: Path = ROOT) -> tuple[list[Path], list[Finding]]:
    """Return tracked and non-ignored untracked paths using NUL-safe Git output."""
    command = [
        "git",
        "-C",
        str(root),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--deduplicate",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except OSError as exc:
        if not (root / ".git").exists():
            return discover_filesystem_candidates(root)
        return [], [_finding(root, root, "git_file_discovery_error", error_type=type(exc).__name__)]
    if completed.returncode != 0:
        if not (root / ".git").exists():
            return discover_filesystem_candidates(root)
        return [], [
            _finding(root, root, "git_file_discovery_error", error_type=f"git_exit_{completed.returncode}")
        ]
    try:
        names = [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        return [], [_finding(root, root, "git_path_decode_error", error_type=type(exc).__name__)]

    root_resolved = root.resolve(strict=False)
    candidates: list[Path] = []
    findings: list[Finding] = []
    seen: set[str] = set()
    for name in names:
        path = root.joinpath(*name.split("/"))
        try:
            path.resolve(strict=False).relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            findings.append(_finding(path, root, "git_path_outside_repository", error_type=type(exc).__name__))
            continue
        if "_quarantine" in {part.casefold() for part in name.split("/")}:
            findings.append(_finding(path, root, "tracked_quarantine_path"))
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    candidates.sort(key=lambda item: _relative_name(item, root).casefold())
    return candidates, findings


def discover_filesystem_candidates(root: Path = ROOT) -> tuple[list[Path], list[Finding]]:
    """Discover a source-bundle inventory when Git metadata is intentionally absent."""

    if not root.is_dir():
        return [], [_finding(root, root, "filesystem_root_unavailable", error_type="NotADirectory")]
    root_resolved = root.resolve(strict=False)
    candidates: list[Path] = []
    findings: list[Finding] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in directory_names:
            path = current_path / name
            if name.casefold() in SKIP_DIRS or name.casefold() in TEST_FIXTURE_DIRS:
                continue
            if path.is_symlink():
                findings.append(_finding(path, root, "filesystem_symlink_unsupported"))
                continue
            if name.casefold() == "_quarantine":
                findings.append(_finding(path, root, "source_bundle_quarantine_path"))
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                findings.append(_finding(path, root, "filesystem_symlink_unsupported"))
                continue
            try:
                path.resolve(strict=False).relative_to(root_resolved)
            except (OSError, ValueError) as exc:
                findings.append(_finding(path, root, "filesystem_path_outside_root", error_type=type(exc).__name__))
                continue
            candidates.append(path)
    candidates.sort(key=lambda item: _relative_name(item, root).casefold())
    return candidates, findings


def _read_source(path: Path, root: Path) -> tuple[str | None, Finding | None, bool]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, _finding(path, root, "file_read_error", error_type=type(exc).__name__), False
    try:
        return data.decode("utf-8-sig"), None, True
    except UnicodeDecodeError as exc:
        line = data[: exc.start].count(b"\n") + 1
        return None, _finding(path, root, "unicode_decode_error", line=line, error_type=type(exc).__name__), True


def _raw_secret_findings(path: Path, root: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        regex_ranges = _regex_definition_ranges(line)
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                if _inside_ranges(match.start(), regex_ranges):
                    continue
                value = match.groupdict().get("value")
                if value is not None and _is_placeholder_value(value):
                    continue
                findings.append(
                    _finding(path, root, f"regex:{pattern.pattern}", line=line_no, column=match.start() + 1)
                )
    return findings


def _scan_text(path: Path, root: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if path.suffix.casefold() == ".py":
        try:
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            findings.append(
                _finding(
                    path,
                    root,
                    "python_parse_error",
                    line=getattr(exc, "lineno", 0) or 0,
                    column=getattr(exc, "offset", 0) or 0,
                    error_type=type(exc).__name__,
                )
            )
            for line, column in _text_literal_secret_locations(text):
                findings.append(_finding(path, root, "text_literal_secret_assignment", line, column))
        else:
            for line, column in _python_literal_secret_locations(tree):
                findings.append(_finding(path, root, "python_literal_secret_assignment", line, column))
    else:
        for line, column in _text_literal_secret_locations(text):
            findings.append(_finding(path, root, "text_literal_secret_assignment", line, column))

    if path.suffix.casefold() in {".bat", ".cmd"}:
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = BATCH_LITERAL_ECHO_PATTERN.match(line)
            if match and _is_secret_text(match.group("value")):
                findings.append(
                    _finding(path, root, "batch_literal_echo_secret", line_no, match.start("value") + 1)
                )
    findings.extend(_raw_secret_findings(path, root, text))
    return findings


def scan_file(path: Path, root: Path = ROOT) -> tuple[list[Finding], bool]:
    """Scan one repository path and report whether it was decoded as source text."""
    if should_skip(path, root) or path.is_dir():
        return [], False
    text, read_error, scanned = _read_source(path, root)
    if read_error is not None:
        return [read_error], scanned
    if text is None:
        return [], scanned

    findings = _scan_text(path, root, text)
    return findings, True


def scan_repository(root: Path = ROOT) -> tuple[list[Finding], int]:
    candidates, findings = discover_candidate_files(root)
    scanned_files = 0
    for path in candidates:
        file_findings, scanned = scan_file(path, root)
        findings.extend(file_findings)
        scanned_files += int(scanned)
    return findings, scanned_files


def _history_inventory(root: Path) -> tuple[dict[str, dict[str, str]], int, list[Finding]]:
    """Map reachable blob ids to their historical paths and one containing commit."""
    try:
        revs = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--all"],
            check=False,
            capture_output=True,
            text=True,
            encoding="ascii",
            errors="strict",
        )
    except (OSError, UnicodeError) as exc:
        return {}, 0, [_finding(root, root, "git_history_discovery_error", error_type=type(exc).__name__)]
    if revs.returncode != 0:
        return {}, 0, [
            _finding(root, root, "git_history_discovery_error", error_type=f"git_exit_{revs.returncode}")
        ]

    commits = [line.strip() for line in revs.stdout.splitlines() if line.strip()]
    blobs: dict[str, dict[str, str]] = {}
    for commit in commits:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", commit],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            return {}, len(commits), [
                _finding(root, root, "git_history_tree_error", error_type=f"git_exit_{completed.returncode}")
            ]
        for record in completed.stdout.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_name = record.split(b"\t", 1)
                fields = metadata.split()
                if len(fields) != 3 or fields[1] != b"blob":
                    continue
                object_id = fields[2].decode("ascii")
                name = raw_name.decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                return {}, len(commits), [
                    _finding(root, root, "git_history_path_decode_error", error_type=type(exc).__name__)
                ]
            virtual_path = root.joinpath(*name.split("/"))
            if should_skip(virtual_path, root):
                continue
            blobs.setdefault(object_id, {}).setdefault(name, commit)
    return blobs, len(commits), []


def scan_history_repository(root: Path = ROOT) -> tuple[list[Finding], int, int, int]:
    """Scan every source/config blob reachable from any local Git ref."""
    blobs, commit_count, findings = _history_inventory(root)
    if findings or not blobs:
        return findings, 0, commit_count, len(blobs)

    try:
        process = subprocess.Popen(
            ["git", "-C", str(root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        findings.append(_finding(root, root, "git_history_blob_error", error_type=type(exc).__name__))
        return findings, 0, commit_count, len(blobs)

    scanned_files = 0
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for object_id, path_commits in sorted(blobs.items()):
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            fields = header.split()
            if len(fields) != 3 or fields[0].decode("ascii", errors="ignore") != object_id or fields[1] != b"blob":
                findings.append(
                    _finding(root, root, "git_history_blob_error", error_type="invalid_cat_file_header")
                )
                break
            try:
                size = int(fields[2])
            except ValueError:
                findings.append(
                    _finding(root, root, "git_history_blob_error", error_type="invalid_blob_size")
                )
                break
            data = process.stdout.read(size)
            trailer = process.stdout.read(1)
            if len(data) != size or trailer != b"\n":
                findings.append(
                    _finding(root, root, "git_history_blob_error", error_type="truncated_blob")
                )
                break
            for name, commit in sorted(path_commits.items()):
                path = root.joinpath(*name.split("/"))
                try:
                    text = data.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    line = data[: exc.start].count(b"\n") + 1
                    item = _finding(
                        path,
                        root,
                        "unicode_decode_error",
                        line=line,
                        error_type=type(exc).__name__,
                    )
                    item.update({"object_id": object_id, "commit": commit})
                    findings.append(item)
                    scanned_files += 1
                    continue
                historical = _scan_text(path, root, text)
                for item in historical:
                    item.update({"object_id": object_id, "commit": commit})
                findings.extend(historical)
                scanned_files += 1
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        process.wait(timeout=30)

    if process.returncode != 0:
        findings.append(
            _finding(root, root, "git_history_blob_error", error_type=f"git_exit_{process.returncode}")
        )
    return findings, scanned_files, commit_count, len(blobs)


def main(root: Path = ROOT, *, history: bool = False) -> None:
    findings, scanned_files = scan_repository(root)
    history_scanned_files = 0
    history_commit_count = 0
    history_blob_count = 0
    if history:
        history_findings, history_scanned_files, history_commit_count, history_blob_count = (
            scan_history_repository(root)
        )
        findings.extend(history_findings)
    if findings:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "message": "Potential plaintext secrets or scanner coverage failures found.",
                    "findings": findings[:50],
                    "finding_count": len(findings),
                    "scanned_files": scanned_files,
                    "history_included": history,
                    "history_scanned_files": history_scanned_files,
                    "history_commit_count": history_commit_count,
                    "history_blob_count": history_blob_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "message": (
                    "No plaintext secrets were found in the current repository inventory or reachable Git history."
                    if history
                    else "No plaintext secrets were found in the current source/config inventory."
                ),
                "scanned_files": scanned_files,
                "history_included": history,
                "history_scanned_files": history_scanned_files,
                "history_commit_count": history_commit_count,
                "history_blob_count": history_blob_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fail closed on plaintext credentials in source/config files.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every source/config blob reachable from any local Git ref",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    main(arguments.root, history=arguments.history)
