"""CLI-facing gateway for immutable EvoMind self-upgrade campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .capability_certification import compute_repository_identity
from .scientist_release_evidence import install_capability_certification, read_research_parity_gate
from .scientist_upgrade_controller import (
    DEFAULT_ALLOWED_PREFIXES,
    DEFAULT_PROTECTED_PREFIXES,
    EvaluatorContract,
    UpgradeControllerError,
    promote_upgrade_campaign,
    rollback_upgrade_campaign,
    run_upgrade_campaign,
)

REQUEST_SCHEMA = "evomind.upgrade_campaign_request.v1"
DEFAULT_REQUEST_PATH = Path(".xsci") / "scientist_upgrade_campaign_request.json"


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = path.read_bytes().decode("utf-8-sig")
    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_object(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise UpgradeControllerError("upgrade campaign request is missing or invalid") from exc
    return value


def _git_root(candidate: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=candidate,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return Path(completed.stdout.strip()).resolve()


def resolve_upgrade_repository(workspace_root: Path | str, explicit: Path | str | None = None) -> Path:
    """Select the source Git repository instead of the global EvoMind data root."""

    workspace = Path(workspace_root).resolve()
    candidates: list[Path] = []
    if explicit is not None:
        candidate = Path(explicit).resolve()
        repository = _git_root(candidate) if candidate.is_dir() else None
        if repository is not None:
            return repository
        raise UpgradeControllerError("explicit source repository is not a Git checkout")
    configured = os.environ.get("EVOMIND_SOURCE_REPOSITORY", "").strip()
    if configured:
        candidate = Path(configured).resolve()
        repository = _git_root(candidate) if candidate.is_dir() else None
        if repository is not None:
            return repository
        raise UpgradeControllerError("configured source repository is not a Git checkout")
    candidates.extend((workspace, Path.cwd().resolve(), Path(__file__).resolve().parents[2]))
    visited: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in visited or not candidate.is_dir():
            continue
        visited.add(key)
        repository = _git_root(candidate)
        if repository is not None:
            return repository
    raise UpgradeControllerError(
        "source Git repository was not found; run from a Git checkout or set EVOMIND_SOURCE_REPOSITORY"
    )


def resolve_upgrade_evidence_root(workspace_root: Path | str, explicit: Path | str | None = None) -> Path:
    """Resolve a read-only evidence root without making source archives executable."""

    if explicit is not None:
        candidate = Path(explicit).resolve()
        evidence_root = _upgrade_evidence_root(candidate)
        if evidence_root is not None:
            return evidence_root
        raise UpgradeControllerError("explicit evidence repository is not a valid EvoMind source")
    configured = os.environ.get("EVOMIND_SOURCE_REPOSITORY", "").strip()
    if configured:
        candidate = Path(configured).resolve()
        evidence_root = _upgrade_evidence_root(candidate)
        if evidence_root is not None:
            return evidence_root
        raise UpgradeControllerError("configured evidence repository is not a valid EvoMind source")
    workspace = Path(workspace_root).resolve()
    workspace_evidence_root = _upgrade_evidence_root(workspace)
    if workspace_evidence_root is not None:
        return workspace_evidence_root
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parents[2]]
    visited: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in visited or not candidate.is_dir():
            continue
        visited.add(key)
        evidence_root = _upgrade_evidence_root(candidate)
        if evidence_root is not None:
            return evidence_root
    raise UpgradeControllerError("EvoMind source or Git repository was not found for evidence status")


def _upgrade_evidence_root(candidate: Path) -> Path | None:
    if not candidate.is_dir():
        return None
    repository = _git_root(candidate)
    if repository is not None:
        return repository
    root = candidate
    required = (
        root / "pyproject.toml",
        root / "README.md",
        root / "LICENSE",
        root / "src" / "xsci" / "scientist_release_evidence.py",
    )
    return root if all(path.is_file() for path in required) else None


def initialize_upgrade_repository(source_root: Path | str) -> dict[str, Any]:
    """Create a local baseline repository for an explicitly selected source archive."""

    source = Path(source_root).resolve()
    required = (
        source / "pyproject.toml",
        source / "README.md",
        source / "LICENSE",
        source / "src" / "xsci" / "scientist_upgrade_controller.py",
    )
    if not source.is_dir() or any(not path.is_file() for path in required):
        raise UpgradeControllerError("source archive is missing required EvoMind release files")
    existing = _git_root(source)
    if existing is not None:
        if existing != source:
            raise UpgradeControllerError("source archive is nested inside a different Git repository")
        identity = compute_repository_identity(source)
        return {
            "ok": True,
            "tool": "scientist_upgrade_campaign",
            "action": "init",
            "status": "already_initialized",
            "repository": str(source),
            "repository_identity": identity.to_dict(),
        }
    commands = (
        ("init", "-q"),
        ("config", "user.name", "EvoMind Upgrade Controller"),
        ("config", "user.email", "evomind-upgrade@invalid.local"),
        ("add", "--all"),
        ("commit", "-q", "-m", "Initialize verified EvoMind source archive"),
    )
    for command in commands:
        completed = subprocess.run(
            ["git", *command],
            cwd=source,
            check=False,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise UpgradeControllerError(f"source repository initialization failed at git {command[0]}")
    identity = compute_repository_identity(source)
    if not identity.clean:
        raise UpgradeControllerError("initialized source repository is not clean")
    return {
        "ok": True,
        "tool": "scientist_upgrade_campaign",
        "action": "init",
        "status": "initialized",
        "repository": str(source),
        "repository_identity": identity.to_dict(),
        "claim_boundary": "A local Git baseline enables campaigns but does not inherit an external parity certificate.",
    }


def _resolve(repository: Path, request_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _string_tuple(value: Any, *, field: str, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise UpgradeControllerError(f"{field} must be a list of non-empty strings")
    result = tuple(item.strip() for item in value)
    if len(result) < minimum:
        raise UpgradeControllerError(f"{field} requires at least {minimum} entries")
    return result


def _load_request(repository: Path, request_path: Path) -> dict[str, Any]:
    payload = _json_object(request_path)
    if payload.get("schema") != REQUEST_SCHEMA:
        raise UpgradeControllerError("unsupported upgrade campaign request schema")
    evaluator_data = payload.get("evaluator")
    if not isinstance(evaluator_data, dict):
        raise UpgradeControllerError("upgrade campaign evaluator contract is missing")
    evaluator = EvaluatorContract(
        evaluator_id=str(evaluator_data.get("evaluator_id") or ""),
        evaluator_files=_string_tuple(evaluator_data.get("evaluator_files"), field="evaluator_files", minimum=1),
        commands=_string_tuple(evaluator_data.get("commands"), field="evaluator commands", minimum=1),
        primary_metric=str(evaluator_data.get("primary_metric") or ""),
        direction=str(evaluator_data.get("direction") or "maximize"),
        minimum_delta=float(evaluator_data.get("minimum_delta") or 0.0),
        required_metrics=_string_tuple(evaluator_data.get("required_metrics"), field="required_metrics", minimum=1),
        isolation_level=str(evaluator_data.get("isolation_level") or "external_isolated"),
        provider=str(evaluator_data.get("provider") or ""),
        model=str(evaluator_data.get("model") or ""),
        seed=int(evaluator_data.get("seed") or 0),
        budget=str(evaluator_data.get("budget") or ""),
    )
    evaluator.validate()
    candidate_values = _string_tuple(payload.get("candidate_patches"), field="candidate_patches", minimum=2)
    candidates = tuple(_resolve(repository, request_path, value) for value in candidate_values)
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise UpgradeControllerError("candidate patch is missing")
    activation_commands = payload.get("activation_commands")
    if activation_commands is None:
        activation_commands = []
    if not isinstance(activation_commands, list):
        raise UpgradeControllerError("activation_commands must be a list")
    artifact_root_value = payload.get("artifact_root")
    artifact_root = (
        _resolve(repository, request_path, str(artifact_root_value))
        if isinstance(artifact_root_value, str) and artifact_root_value.strip()
        else None
    )
    allowed = _string_tuple(
        payload.get("allowed_prefixes", list(DEFAULT_ALLOWED_PREFIXES)),
        field="allowed_prefixes",
        minimum=1,
    )
    protected = _string_tuple(
        payload.get("protected_prefixes", list(DEFAULT_PROTECTED_PREFIXES)),
        field="protected_prefixes",
        minimum=1,
    )
    return {
        "path": request_path,
        "payload": payload,
        "evaluator": evaluator,
        "candidate_patches": candidates,
        "activation_commands": activation_commands,
        "artifact_root": artifact_root,
        "allowed_prefixes": allowed,
        "protected_prefixes": protected,
        "minimum_candidates": max(2, int(payload.get("minimum_candidates") or 2)),
    }


def _clean_environment(worktree: Path) -> dict[str, str]:
    names = (
        "ALLUSERSPROFILE",
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    )
    env = {name: os.environ[name] for name in names if name in os.environ}
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(worktree / "src")
    return env


def _command_args(value: Any, *, worktree: Path) -> list[str]:
    if isinstance(value, str):
        args = shlex.split(value, posix=os.name != "nt")
    elif isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        args = list(value)
    else:
        raise UpgradeControllerError("each activation command must be a string or string list")
    replacements = {
        "{python}": sys.executable,
        "{worktree}": str(worktree),
    }
    return [replacements.get(item, item) for item in args]


def build_activation_callback(
    commands: Sequence[Any],
    *,
    timeout_seconds: int = 300,
) -> Callable[[Path, str, str], Mapping[str, Any]]:
    """Build a secret-minimizing detached-worktree runtime canary."""

    if not commands:
        raise UpgradeControllerError("at least one activation canary command is required")

    def activate(repository: Path, commit_sha: str, expected_tree_sha: str) -> Mapping[str, Any]:
        with tempfile.TemporaryDirectory(prefix="evomind-activation-") as temp_dir:
            worktree = Path(temp_dir) / "candidate"
            added = False
            records: list[dict[str, Any]] = []
            try:
                add = subprocess.run(
                    ["git", "worktree", "add", "--detach", "--force", str(worktree), commit_sha],
                    cwd=repository,
                    check=False,
                    capture_output=True,
                    timeout=60,
                )
                if add.returncode != 0:
                    return {"passed": False, "runtime_tree_sha": "", "checks": []}
                added = True
                tree = subprocess.run(
                    ["git", "rev-parse", "HEAD^{tree}"],
                    cwd=worktree,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="ascii",
                    errors="replace",
                    timeout=30,
                ).stdout.strip()
                if tree != expected_tree_sha:
                    return {"passed": False, "runtime_tree_sha": tree, "checks": []}
                for index, command in enumerate(commands, start=1):
                    args = _command_args(command, worktree=worktree)
                    completed = subprocess.run(
                        args,
                        cwd=worktree,
                        env=_clean_environment(worktree),
                        check=False,
                        capture_output=True,
                        timeout=max(1, timeout_seconds),
                    )
                    output_digest = hashlib.sha256(completed.stdout + b"\0" + completed.stderr).hexdigest()
                    records.append(
                        {
                            "index": index,
                            "command_sha256": hashlib.sha256("\0".join(args).encode("utf-8")).hexdigest(),
                            "returncode": completed.returncode,
                            "output_sha256": output_digest,
                        }
                    )
                    if completed.returncode != 0:
                        return {"passed": False, "runtime_tree_sha": tree, "checks": records}
                return {"passed": True, "runtime_tree_sha": tree, "checks": records}
            except (OSError, subprocess.SubprocessError):
                return {"passed": False, "runtime_tree_sha": "", "checks": records}
            finally:
                if added:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(worktree)],
                        cwd=repository,
                        check=False,
                        capture_output=True,
                        timeout=60,
                    )

    return activate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evomind upgrade-campaign")
    parser.add_argument("action", nargs="?", choices=("status", "init", "run", "promote", "rollback"), default="status")
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--human-approved", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    return parser


def run_upgrade_campaign_cli(argv: Sequence[str], repository: Path | str) -> int:
    args = _parser().parse_args(list(argv))
    try:
        if args.action == "init":
            source = args.repository.resolve() if args.repository is not None else Path.cwd().resolve()
            result = initialize_upgrade_repository(source)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        root = (
            resolve_upgrade_evidence_root(repository, args.repository)
            if args.action == "status"
            else resolve_upgrade_repository(repository, args.repository)
        )
        request_path = args.request if args.request.is_absolute() else root / args.request
        if args.action == "status":
            result = read_research_parity_gate(root)
        else:
            request = _load_request(root, request_path.resolve())
            if args.action == "run":
                result = run_upgrade_campaign(
                    root,
                    candidate_patches=request["candidate_patches"],
                    evaluator=request["evaluator"],
                    artifact_root=request["artifact_root"],
                    allowed_prefixes=request["allowed_prefixes"],
                    protected_prefixes=request["protected_prefixes"],
                    minimum_candidates=request["minimum_candidates"],
                )
            else:
                manifest_path = args.manifest
                if manifest_path is None:
                    latest = root / ".xsci" / "scientist_upgrade_campaign.json"
                    latest_payload = _json_object(latest)
                    manifest_value = latest_payload.get("manifest_path")
                    if not isinstance(manifest_value, str) or not manifest_value:
                        raise UpgradeControllerError("campaign manifest path is unavailable")
                    manifest_path = Path(manifest_value)
                if not manifest_path.is_absolute():
                    manifest_path = root / manifest_path
                callback = build_activation_callback(
                    request["activation_commands"],
                    timeout_seconds=max(1, min(1800, args.timeout)),
                )
                if args.action == "promote":
                    result = promote_upgrade_campaign(
                        root,
                        manifest_path,
                        human_approved=args.human_approved,
                        activation_callback=callback,
                    )
                else:
                    result = rollback_upgrade_campaign(root, manifest_path, activation_callback=callback)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.action == "status":
            return 0
        return 0 if result.get("status") in {
            "awaiting_human_promotion",
            "held_no_strict_improvement",
            "active",
            "rolled_back",
        } else 1
    except (UpgradeControllerError, OSError, ValueError) as exc:
        result = {
            "ok": False,
            "tool": "scientist_upgrade_campaign",
            "action": args.action,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


def run_certification_install_cli(argv: Sequence[str], workspace_root: Path | str) -> int:
    parser = argparse.ArgumentParser(prog="evomind certification-install")
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args(list(argv))
    try:
        repository = resolve_upgrade_repository(workspace_root, args.repository)
        result = install_capability_certification(
            repository,
            args.result,
            expected_result_sha256=args.expected_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") is True else 1
    except (OSError, UpgradeControllerError, ValueError) as exc:
        print(json.dumps({
            "ok": False,
            "tool": "capability_certification_install",
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, indent=2))
        return 1


__all__ = [
    "DEFAULT_REQUEST_PATH",
    "REQUEST_SCHEMA",
    "build_activation_callback",
    "initialize_upgrade_repository",
    "resolve_upgrade_evidence_root",
    "resolve_upgrade_repository",
    "run_certification_install_cli",
    "run_upgrade_campaign_cli",
]
