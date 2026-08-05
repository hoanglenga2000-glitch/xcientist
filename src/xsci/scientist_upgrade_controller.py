"""Trusted, hash-bound self-upgrade campaign controller.

Candidate generation is intentionally outside this module.  The controller
accepts reviewable unified diffs, freezes the evaluator before reading them,
creates immutable candidate commits without touching the user's branch, runs a
paired baseline/candidate evaluation, and promotes only a strictly improving
candidate through a compare-and-swap champion ref.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .capability_certification import (
    CAMPAIGN_SCHEMA,
    SOURCE_DIGEST_ALGORITHM,
    compute_campaign_payload_sha256,
    compute_repository_identity,
)

CONTROLLER_SCHEMA = "evomind.self_upgrade_controller.v1"
CONTRACT_SCHEMA = "evomind.self_upgrade_contract.v1"
CHAMPION_REF = "refs/evomind/champion"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
METRIC_RE = re.compile(r"(?m)^EVOMIND_METRIC\s+([A-Za-z0-9_.-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$")

DEFAULT_ALLOWED_PREFIXES = (
    "src/",
    "web/research-agent-workstation/src/",
    "docs/",
)
DEFAULT_PROTECTED_PREFIXES = (
    ".github/",
    "benchmark/",
    "configs/schemas/",
    "scripts/",
    "tests/",
    "src/xsci/scientist_upgrade_controller.py",
    "src/xsci/capability_certification.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-gpu.txt",
    "web/research-agent-workstation/package-lock.json",
)


class UpgradeControllerError(RuntimeError):
    pass


EvaluatorCallback = Callable[[Path, "EvaluatorContract", str], Mapping[str, Any]]
ActivationCallback = Callable[[Path, str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class EvaluatorContract:
    evaluator_id: str
    evaluator_files: tuple[str, ...]
    commands: tuple[str, ...]
    primary_metric: str
    direction: str = "maximize"
    minimum_delta: float = 0.0
    required_metrics: tuple[str, ...] = ()
    isolation_level: str = "host_subprocess"
    provider: str = ""
    model: str = ""
    seed: int = 0
    budget: str = ""

    def validate(self) -> None:
        if not self.evaluator_id.strip():
            raise UpgradeControllerError("evaluator_id is required")
        if not self.evaluator_files:
            raise UpgradeControllerError("at least one evaluator file is required")
        if not self.commands:
            raise UpgradeControllerError("at least one evaluator command identifier is required")
        if not self.primary_metric.strip():
            raise UpgradeControllerError("primary_metric is required")
        if self.direction not in {"maximize", "minimize"}:
            raise UpgradeControllerError("direction must be maximize or minimize")
        if not math.isfinite(float(self.minimum_delta)) or self.minimum_delta < 0:
            raise UpgradeControllerError("minimum_delta must be finite and non-negative")
        if self.isolation_level not in {"external_isolated", "container", "host_subprocess", "test_fixture"}:
            raise UpgradeControllerError("unsupported evaluator isolation_level")


def _utc_seconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    temporary.replace(path)


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
    payload = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("JSON document must be an object")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_object(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise UpgradeControllerError(f"campaign manifest is unreadable: {path}") from exc
    return payload


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 180,
    env: Mapping[str, str] | None = None,
    input_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        input=input_data,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _git(repository: Path, *args: str, timeout: int = 180, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_PREFIX"):
        process_env.pop(key, None)
    if env:
        process_env.update(env)
    return _run(["git", *args], cwd=repository, timeout=timeout, env=process_env)


def _git_ok(repository: Path, *args: str, timeout: int = 180, env: Mapping[str, str] | None = None) -> str:
    result = _git(repository, *args, timeout=timeout, env=env)
    if result.returncode != 0:
        raise UpgradeControllerError(f"git {' '.join(args[:3])} failed")
    return result.stdout.strip()


def _normalize_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or ":" in text:
        raise UpgradeControllerError(f"unsafe repository path: {value!r}")
    return pure.as_posix()


def _matches(path: str, scopes: Sequence[str]) -> bool:
    normalized_path = path.casefold()
    return any(
        normalized_path == scope.rstrip("/").casefold()
        or normalized_path.startswith(scope.rstrip("/").casefold() + "/")
        for scope in scopes
    )


def _changed_paths(repository: Path) -> list[str]:
    result = _git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--no-renames",
    )
    if result.returncode != 0:
        raise UpgradeControllerError("candidate worktree status could not be read")
    changed: set[str] = set()
    for record in result.stdout.split("\0"):
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            raise UpgradeControllerError("candidate worktree returned malformed status")
        changed.add(_normalize_path(record[3:]))
    return sorted(changed)


def _tracked_blob(repository: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{path}"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise UpgradeControllerError(f"evaluator file is not tracked at the locked revision: {path}")
    return result.stdout


def _evaluator_lock(repository: Path, base_commit: str, evaluator: EvaluatorContract) -> dict[str, Any]:
    evaluator.validate()
    files = tuple(dict.fromkeys(_normalize_path(path) for path in evaluator.evaluator_files))
    file_rows = [
        {"path": path, "sha256": hashlib.sha256(_tracked_blob(repository, base_commit, path)).hexdigest()}
        for path in files
    ]
    commands = [str(command).strip() for command in evaluator.commands]
    if any(not command for command in commands):
        raise UpgradeControllerError("evaluator commands must be non-empty")
    files_digest = _digest(file_rows)
    commands_digest = _digest(commands)
    stable_body = {
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_files": file_rows,
        "commands": commands,
        "commands_digest_sha256": commands_digest,
        "files_digest_sha256": files_digest,
        "primary_metric": evaluator.primary_metric,
        "direction": evaluator.direction,
        "minimum_delta": evaluator.minimum_delta,
        "required_metrics": list(evaluator.required_metrics),
        "isolation_level": evaluator.isolation_level,
        "provider": evaluator.provider,
        "model": evaluator.model,
        "seed": evaluator.seed,
        "budget": evaluator.budget,
        "locked_before_candidate_generation": True,
    }
    return {
        **stable_body,
        "evaluator_digest_sha256": _digest(stable_body),
        "locked_at": _utc_seconds(),
    }


def _wait_for_next_utc_second(locked_at: str) -> str:
    while True:
        current = _utc_seconds()
        if current > locked_at:
            return current
        time.sleep(0.02)


class _CampaignLock:
    def __init__(self, path: Path, campaign_id: str) -> None:
        self.path = path
        self.campaign_id = campaign_id
        self.fd: int | None = None

    def __enter__(self) -> "_CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise UpgradeControllerError("another self-upgrade campaign holds the repository lock") from exc
        os.write(self.fd, self.campaign_id.encode("ascii", errors="strict"))
        os.fsync(self.fd)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _record_event(history_path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    previous = "0" * 64
    if history_path.is_file():
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            try:
                previous = str(json.loads(lines[-1]).get("event_sha256") or previous)
            except json.JSONDecodeError:
                raise UpgradeControllerError("campaign event history is malformed")
    payload = {"previous_event_sha256": previous, **dict(event)}
    payload["event_sha256"] = _digest(payload)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return payload


def _clean_environment() -> dict[str, str]:
    keep = {
        "PATH", "Path", "PATHEXT", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR",
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "COMSPEC", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE", "PYTHONUTF8", "PYTHONIOENCODING", "CI",
    }
    result = {key: value for key, value in os.environ.items() if key in keep}
    result["PYTHONUTF8"] = "1"
    result["PYTHONIOENCODING"] = "utf-8"
    result["EVOMIND_EVALUATOR_LOCKED"] = "1"
    return result


def _command_args(command: str) -> list[str]:
    parts = [part.strip('"') for part in shlex.split(command, posix=False)]
    if not parts:
        raise UpgradeControllerError("empty evaluator command")
    first = parts[0].lower()
    if first in {"python", "python.exe"}:
        parts[0] = os.environ.get("WORKSTATION_PYTHON") or os.sys.executable
        if len(parts) < 3 or parts[1:3] not in (["-m", "pytest"], ["-m", "compileall"]):
            raise UpgradeControllerError("host evaluator Python command is not allowlisted")
    elif first in {"npm", "npm.cmd"}:
        if len(parts) < 3 or parts[1] != "run" or parts[2] not in {"typecheck", "build", "test"}:
            raise UpgradeControllerError("host evaluator npm command is not allowlisted")
        if os.name == "nt":
            parts[0] = "npm.cmd"
    elif first == "git":
        if parts[1:] != ["diff", "--check"]:
            raise UpgradeControllerError("host evaluator git command is not allowlisted")
    else:
        raise UpgradeControllerError("host evaluator command is not allowlisted")
    return parts


def _normalize_evaluation(raw: Mapping[str, Any], evaluator: EvaluatorContract, *, label: str) -> dict[str, Any]:
    passed = raw.get("passed") is True
    metrics_raw = raw.get("metrics")
    metrics: dict[str, float] = {}
    if isinstance(metrics_raw, Mapping):
        for key, value in metrics_raw.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                metrics[str(key)] = float(value)
    score_value = raw.get("score", metrics.get(evaluator.primary_metric))
    if not isinstance(score_value, (int, float)) or isinstance(score_value, bool) or not math.isfinite(float(score_value)):
        passed = False
        score = None
    else:
        score = float(score_value)
        metrics.setdefault(evaluator.primary_metric, score)
    missing = [name for name in evaluator.required_metrics if name not in metrics]
    if missing:
        passed = False
    result = {
        "label": label,
        "passed": passed,
        "score": score,
        "metrics": metrics,
        "missing_required_metrics": missing,
        "details": str(raw.get("details") or "")[:2000],
        "evaluator_id": evaluator.evaluator_id,
        "primary_metric": evaluator.primary_metric,
        "direction": evaluator.direction,
    }
    result["evaluation_digest_sha256"] = _digest(result)
    return result


def _evaluate_worktree(
    worktree: Path,
    evaluator: EvaluatorContract,
    *,
    label: str,
    logs_dir: Path,
    callback: EvaluatorCallback | None,
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if callback is not None:
        return _normalize_evaluation(callback(worktree, evaluator, label), evaluator, label=label)
    if evaluator.isolation_level not in {"external_isolated", "container", "host_subprocess"}:
        raise UpgradeControllerError("command evaluator requires a declared isolation level")
    outputs: list[str] = []
    passed = True
    for index, command in enumerate(evaluator.commands, start=1):
        args = _command_args(command)
        try:
            completed = _run(args, cwd=worktree, timeout=900, env=_clean_environment())
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            output = "evaluator command timed out"
            exit_code = 124
        (logs_dir / f"{index:02d}.log").write_text(output[-200_000:], encoding="utf-8", newline="\n")
        outputs.append(output)
        passed = passed and exit_code == 0
    metrics = {name: float(value) for output in outputs for name, value in METRIC_RE.findall(output)}
    return _normalize_evaluation(
        {"passed": passed, "metrics": metrics, "details": "locked evaluator commands completed"},
        evaluator,
        label=label,
    )


def _with_worktree(repository: Path, revision: str, action: Callable[[Path], Any]) -> Any:
    parent = Path(tempfile.mkdtemp(prefix="evomind-upgrade-"))
    worktree = parent / "worktree"
    try:
        _git_ok(repository, "worktree", "add", "--detach", str(worktree), revision, timeout=180)
        return action(worktree)
    finally:
        if worktree.exists():
            _git(repository, "worktree", "remove", "--force", str(worktree), timeout=180)
        _git(repository, "worktree", "prune", timeout=60)
        shutil.rmtree(parent, ignore_errors=True)


def _candidate_from_patch(
    repository: Path,
    *,
    base_commit: str,
    campaign_id: str,
    patch_path: Path,
    allowed_prefixes: Sequence[str],
    protected_prefixes: Sequence[str],
    evaluator_files: Sequence[str],
    artifact_dir: Path,
) -> dict[str, Any]:
    patch = patch_path.resolve(strict=True)
    patch_sha = _sha256_file(patch)
    candidate_id = f"candidate-{patch_sha[:12]}"
    stored_patch = artifact_dir / "candidates" / candidate_id / "candidate.diff"
    stored_patch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(patch, stored_patch)

    def build(worktree: Path) -> dict[str, Any]:
        checked = _git(worktree, "apply", "--recount", "--check", str(stored_patch), timeout=120)
        if checked.returncode != 0:
            return {"candidate_id": candidate_id, "status": "failed_patch_apply", "patch_digest_sha256": patch_sha}
        applied = _git(worktree, "apply", "--recount", str(stored_patch), timeout=120)
        if applied.returncode != 0:
            return {"candidate_id": candidate_id, "status": "failed_patch_apply", "patch_digest_sha256": patch_sha}
        changed = _changed_paths(worktree)
        if not changed:
            return {"candidate_id": candidate_id, "status": "failed_empty_patch", "patch_digest_sha256": patch_sha}
        protected = sorted(path for path in changed if _matches(path, protected_prefixes) or path in evaluator_files)
        outside = sorted(path for path in changed if not _matches(path, allowed_prefixes))
        if protected or outside:
            return {
                "candidate_id": candidate_id,
                "status": "rejected_scope",
                "patch_digest_sha256": patch_sha,
                "changed_paths": changed,
                "protected_paths_modified": protected,
                "outside_allowed_paths": outside,
                "evaluator_files_modified": bool(protected),
            }
        _git_ok(worktree, "add", "--all", "--", *changed)
        tree_sha = _git_ok(worktree, "write-tree")
        if not OID_RE.fullmatch(tree_sha):
            raise UpgradeControllerError("candidate tree identifier is invalid")
        commit_env = {
            "GIT_AUTHOR_NAME": "EvoMind Upgrade Controller",
            "GIT_AUTHOR_EMAIL": "evomind-upgrade@invalid.local",
            "GIT_COMMITTER_NAME": "EvoMind Upgrade Controller",
            "GIT_COMMITTER_EMAIL": "evomind-upgrade@invalid.local",
        }
        message = f"EvoMind upgrade candidate {campaign_id} {candidate_id}"
        commit_sha = _git_ok(worktree, "commit-tree", tree_sha, "-p", base_commit, "-m", message, env=commit_env)
        if not OID_RE.fullmatch(commit_sha):
            raise UpgradeControllerError("candidate commit identifier is invalid")
        candidate_ref = f"refs/evomind/candidates/{campaign_id}/{candidate_id}"
        zeros = "0" * len(commit_sha)
        updated = _git(repository, "update-ref", candidate_ref, commit_sha, zeros)
        if updated.returncode != 0:
            raise UpgradeControllerError("candidate ref compare-and-swap failed")
        return {
            "candidate_id": candidate_id,
            "status": "built",
            "patch_path": str(stored_patch),
            "patch_digest_sha256": patch_sha,
            "changed_paths": changed,
            "protected_paths_modified": [],
            "outside_allowed_paths": [],
            "evaluator_files_modified": False,
            "candidate_ref": candidate_ref,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
        }

    return _with_worktree(repository, base_commit, build)


def _strict_improvement(candidate: Mapping[str, Any], baseline: Mapping[str, Any], evaluator: EvaluatorContract) -> bool:
    if candidate.get("passed") is not True or baseline.get("passed") is not True:
        return False
    candidate_score = candidate.get("score")
    baseline_score = baseline.get("score")
    if (
        not isinstance(candidate_score, (int, float))
        or isinstance(candidate_score, bool)
        or not isinstance(baseline_score, (int, float))
        or isinstance(baseline_score, bool)
    ):
        return False
    improvement = (
        float(candidate_score) - float(baseline_score)
        if evaluator.direction == "maximize"
        else float(baseline_score) - float(candidate_score)
    )
    if not math.isfinite(improvement) or improvement <= 0.0 or improvement < evaluator.minimum_delta:
        return False
    candidate_metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), Mapping) else {}
    baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), Mapping) else {}
    for name in evaluator.required_metrics:
        if name not in candidate_metrics or name not in baseline_metrics:
            return False
        if evaluator.direction == "maximize" and float(candidate_metrics[name]) < float(baseline_metrics[name]):
            return False
        if evaluator.direction == "minimize" and float(candidate_metrics[name]) > float(baseline_metrics[name]):
            return False
    return True


def _manifest_attestation(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("attestation", None)
    return _digest(value)


def _verify_manifest_attestation(manifest: Mapping[str, Any]) -> None:
    attestation = manifest.get("attestation")
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != {"algorithm", "payload_sha256"}
        or attestation.get("algorithm") != "sha256"
        or not isinstance(attestation.get("payload_sha256"), str)
        or SHA256_RE.fullmatch(str(attestation["payload_sha256"])) is None
        or attestation["payload_sha256"] != _manifest_attestation(manifest)
    ):
        raise UpgradeControllerError("campaign manifest attestation is invalid")


def evaluation_attestation_matches(evaluation: Any) -> bool:
    if not isinstance(evaluation, Mapping):
        return False
    payload = dict(evaluation)
    observed = payload.pop("evaluation_digest_sha256", None)
    return (
        isinstance(observed, str)
        and SHA256_RE.fullmatch(observed) is not None
        and observed == _digest(payload)
    )


def run_upgrade_campaign(
    repository: Path | str,
    *,
    candidate_patches: Sequence[Path | str],
    evaluator: EvaluatorContract,
    evaluator_callback: EvaluatorCallback | None = None,
    artifact_root: Path | str | None = None,
    allowed_prefixes: Sequence[str] = DEFAULT_ALLOWED_PREFIXES,
    protected_prefixes: Sequence[str] = DEFAULT_PROTECTED_PREFIXES,
    minimum_candidates: int = 2,
) -> dict[str, Any]:
    """Build and evaluate multiple candidates; stop at the human promotion gate."""

    repository_path = Path(repository).resolve()
    evaluator.validate()
    if len(candidate_patches) < minimum_candidates:
        raise UpgradeControllerError(f"at least {minimum_candidates} candidate patches are required")
    identity = compute_repository_identity(repository_path)
    if not identity.clean:
        raise UpgradeControllerError("self-upgrade campaigns require a clean Git worktree")
    campaign_id = "upgrade-" + uuid.uuid4().hex[:16]
    root = Path(artifact_root).resolve() if artifact_root else repository_path / ".xsci" / "upgrade_campaigns" / campaign_id
    contract_path = root / "contract.json"
    manifest_path = root / "manifest.json"
    history_path = root / "events.jsonl"
    latest_path = repository_path / ".xsci" / "scientist_upgrade_campaign.json"
    lock_path = repository_path / ".xsci" / "scientist_upgrade_campaign.lock"
    normalized_allowed = tuple(dict.fromkeys(_normalize_path(path.rstrip("/")) + ("/" if str(path).endswith("/") else "") for path in allowed_prefixes))
    normalized_protected = tuple(dict.fromkeys(_normalize_path(path.rstrip("/")) + ("/" if str(path).endswith("/") else "") for path in protected_prefixes))

    with _CampaignLock(lock_path, campaign_id):
        base_commit = identity.commit_sha
        base_tree = identity.tree_sha
        evaluator_lock = _evaluator_lock(repository_path, base_commit, evaluator)
        contract = {
            "schema": CONTRACT_SCHEMA,
            "campaign_id": campaign_id,
            "base": identity.to_dict(),
            "evaluator": asdict(evaluator),
            "evaluator_lock": evaluator_lock,
            "allowed_prefixes": list(normalized_allowed),
            "protected_prefixes": list(normalized_protected),
            "minimum_candidates": minimum_candidates,
            "promotion_policy": "strict_improvement_plus_human_approval_plus_runtime_tree_canary",
        }
        contract["contract_sha256"] = _digest(contract)
        _write_json(contract_path, contract)
        _record_event(history_path, {"ts": _utc_seconds(), "phase": "lock", "contract_sha256": contract["contract_sha256"]})
        generation_started_at = _wait_for_next_utc_second(evaluator_lock["locked_at"])

        baseline = _with_worktree(
            repository_path,
            base_commit,
            lambda worktree: _evaluate_worktree(
                worktree,
                evaluator,
                label="baseline",
                logs_dir=root / "baseline" / "logs",
                callback=evaluator_callback,
            ),
        )
        _write_json(root / "baseline" / "evaluation.json", baseline)
        _record_event(history_path, {"ts": _utc_seconds(), "phase": "baseline", "evaluation": baseline["evaluation_digest_sha256"]})

        candidates: list[dict[str, Any]] = []
        for patch_value in candidate_patches:
            current_head = _git_ok(repository_path, "rev-parse", "HEAD")
            if current_head != base_commit:
                raise UpgradeControllerError("repository HEAD changed during candidate generation")
            candidate = _candidate_from_patch(
                repository_path,
                base_commit=base_commit,
                campaign_id=campaign_id,
                patch_path=Path(patch_value),
                allowed_prefixes=normalized_allowed,
                protected_prefixes=normalized_protected,
                evaluator_files=[row["path"] for row in evaluator_lock["evaluator_files"]],
                artifact_dir=root,
            )
            if candidate.get("status") == "built":
                evaluation = _with_worktree(
                    repository_path,
                    candidate["commit_sha"],
                    lambda worktree, candidate_id=candidate["candidate_id"]: _evaluate_worktree(
                        worktree,
                        evaluator,
                        label=candidate_id,
                        logs_dir=root / "candidates" / candidate_id / "logs",
                        callback=evaluator_callback,
                    ),
                )
                candidate["evaluation"] = evaluation
                candidate["evaluation_digest_sha256"] = evaluation["evaluation_digest_sha256"]
                candidate["strictly_improves_baseline"] = _strict_improvement(evaluation, baseline, evaluator)
                candidate["status"] = "passed" if evaluation["passed"] else "failed"
            else:
                candidate.setdefault("evaluation_digest_sha256", "0" * 64)
                candidate["strictly_improves_baseline"] = False
                candidate.setdefault("evaluator_files_modified", bool(candidate.get("protected_paths_modified")))
            _write_json(root / "candidates" / candidate["candidate_id"] / "candidate.json", candidate)
            candidates.append(candidate)
            _record_event(
                history_path,
                {
                    "ts": _utc_seconds(),
                    "phase": "candidate",
                    "candidate_id": candidate["candidate_id"],
                    "status": candidate["status"],
                    "strictly_improves_baseline": candidate["strictly_improves_baseline"],
                },
            )

        improving = [
            item for item in candidates
            if item.get("status") == "passed" and item.get("strictly_improves_baseline") is True
        ]
        reverse = evaluator.direction == "maximize"
        improving.sort(
            key=lambda item: float((item.get("evaluation") or {}).get("score")),
            reverse=reverse,
        )
        selected = improving[0] if improving else None
        status = "awaiting_human_promotion" if selected else "held_no_strict_improvement"
        manifest: dict[str, Any] = {
            "schema": CONTROLLER_SCHEMA,
            "campaign_id": campaign_id,
            "status": status,
            "created_at": _utc_seconds(),
            "candidate_generation_started_at": generation_started_at,
            "repository": identity.to_dict(),
            "base_commit": base_commit,
            "base_tree": base_tree,
            "contract_path": str(contract_path),
            "contract_sha256": contract["contract_sha256"],
            "evaluator": asdict(evaluator),
            "evaluator_lock": evaluator_lock,
            "baseline": baseline,
            "candidates": candidates,
            "selection": {
                "candidate_id": selected.get("candidate_id") if selected else "",
                "decision": "awaiting_human_promotion" if selected else "hold",
                "strictly_improves_baseline": bool(selected),
            },
            "promotion": None,
            "rollback": None,
            "champion_ref": CHAMPION_REF,
            "history_path": str(history_path),
            "manifest_path": str(manifest_path),
            "certification_manifest_path": str(root / "self_upgrade_campaign_evidence.json"),
            "human_gate": "explicit_human_approval_required",
            "main_worktree_modified": False,
            "no_training_started": True,
        }
        manifest["attestation"] = {"algorithm": "sha256", "payload_sha256": _manifest_attestation(manifest)}
        _write_json(manifest_path, manifest)
        _write_json(latest_path, manifest)
        return manifest


def _verify_locked_evaluator(repository: Path, manifest: Mapping[str, Any]) -> EvaluatorContract:
    base = str(manifest.get("base_commit") or "")
    evaluator_data = manifest.get("evaluator")
    locked = manifest.get("evaluator_lock")
    if not isinstance(evaluator_data, Mapping) or not isinstance(locked, Mapping):
        raise UpgradeControllerError("campaign evaluator contract is missing")
    evaluator = EvaluatorContract(
        evaluator_id=str(evaluator_data.get("evaluator_id") or ""),
        evaluator_files=tuple(evaluator_data.get("evaluator_files") or ()),
        commands=tuple(evaluator_data.get("commands") or ()),
        primary_metric=str(evaluator_data.get("primary_metric") or ""),
        direction=str(evaluator_data.get("direction") or "maximize"),
        minimum_delta=float(evaluator_data.get("minimum_delta") or 0.0),
        required_metrics=tuple(evaluator_data.get("required_metrics") or ()),
        isolation_level=str(evaluator_data.get("isolation_level") or "host_subprocess"),
        provider=str(evaluator_data.get("provider") or ""),
        model=str(evaluator_data.get("model") or ""),
        seed=int(evaluator_data.get("seed") or 0),
        budget=str(evaluator_data.get("budget") or ""),
    )
    observed = _evaluator_lock(repository, base, evaluator)
    for key in ("evaluator_digest_sha256", "commands_digest_sha256", "files_digest_sha256"):
        if observed.get(key) != locked.get(key):
            raise UpgradeControllerError("frozen evaluator digest changed after candidate generation")
    return evaluator


def _read_ref(repository: Path, ref: str) -> str:
    result = _git(repository, "rev-parse", "--verify", ref)
    return result.stdout.strip() if result.returncode == 0 else ""


def _cas_ref(repository: Path, ref: str, new: str, old: str) -> None:
    expected = old or ("0" * len(new))
    result = _git(repository, "update-ref", ref, new, expected)
    if result.returncode != 0:
        raise UpgradeControllerError(f"compare-and-swap failed for {ref}")


def _delete_ref(repository: Path, ref: str, old: str) -> None:
    result = _git(repository, "update-ref", "-d", ref, old)
    if result.returncode != 0:
        raise UpgradeControllerError(f"compare-and-swap delete failed for {ref}")


def _rollback_ref_test(repository: Path, *, campaign_id: str, base_commit: str, candidate_commit: str, base_tree: str) -> dict[str, Any]:
    ref = f"refs/evomind/rollback-tests/{campaign_id}"
    zeros = "0" * len(base_commit)
    try:
        if _git(repository, "update-ref", ref, base_commit, zeros).returncode != 0:
            return {"tested": True, "passed": False, "restored_tree_sha": ""}
        _cas_ref(repository, ref, candidate_commit, base_commit)
        _cas_ref(repository, ref, base_commit, candidate_commit)
        restored = _git_ok(repository, "rev-parse", f"{ref}^{{tree}}")
        return {"tested": True, "passed": restored == base_tree, "restored_tree_sha": restored}
    finally:
        current = _read_ref(repository, ref)
        if current:
            _delete_ref(repository, ref, current)


def _certification_manifest(manifest: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    baseline = manifest["baseline"]
    evidence: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "created_at": manifest["created_at"],
        "subject": {
            "commit_sha": selected["commit_sha"],
            "tree_sha": selected["tree_sha"],
            "source_digest_algorithm": SOURCE_DIGEST_ALGORITHM,
            "source_digest_sha256": str(manifest["promotion"]["source_digest_sha256"]),
        },
        "evaluator_lock": {
            "evaluator_id": manifest["evaluator_lock"]["evaluator_id"],
            "evaluator_digest_sha256": manifest["evaluator_lock"]["evaluator_digest_sha256"],
            "commands_digest_sha256": manifest["evaluator_lock"]["commands_digest_sha256"],
            "files_digest_sha256": manifest["evaluator_lock"]["files_digest_sha256"],
            "locked_at": manifest["evaluator_lock"]["locked_at"],
            "locked_before_candidate_generation": True,
        },
        "candidate_generation_started_at": manifest["candidate_generation_started_at"],
        "baseline": {
            "status": "passed" if baseline.get("passed") else "failed",
            "commit_sha": manifest["base_commit"],
            "tree_sha": manifest["base_tree"],
            "metrics_digest_sha256": baseline["evaluation_digest_sha256"],
        },
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "patch_digest_sha256": item["patch_digest_sha256"],
                "evaluation_digest_sha256": item["evaluation_digest_sha256"],
                "status": "passed" if item.get("status") == "passed" else "failed",
                "evaluator_files_modified": bool(item.get("evaluator_files_modified")),
                "strictly_improves_baseline": bool(item.get("strictly_improves_baseline")),
            }
            for item in manifest["candidates"]
        ],
        "selection": {
            "candidate_id": selected["candidate_id"],
            "decision": "promoted",
            "strictly_improves_baseline": True,
        },
        "promotion": {
            "candidate_id": selected["candidate_id"],
            "human_approved": True,
            "verified": True,
            "promoted_commit_sha": selected["commit_sha"],
            "promoted_tree_sha": selected["tree_sha"],
        },
        "rollback": manifest["rollback"],
    }
    evidence["attestation"] = {"algorithm": "sha256", "payload_sha256": compute_campaign_payload_sha256(evidence)}
    return evidence


def promote_upgrade_campaign(
    repository: Path | str,
    manifest_path: Path | str,
    *,
    human_approved: bool,
    activation_callback: ActivationCallback | None,
    champion_ref: str = CHAMPION_REF,
) -> dict[str, Any]:
    """CAS-promote the selected candidate and verify the activated runtime tree."""

    repository_path = Path(repository).resolve()
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    if manifest.get("schema") != CONTROLLER_SCHEMA:
        raise UpgradeControllerError("unsupported self-upgrade controller manifest")
    _verify_manifest_attestation(manifest)
    if not human_approved:
        raise UpgradeControllerError("explicit human approval is required for promotion")
    if activation_callback is None:
        raise UpgradeControllerError("runtime activation/canary callback is required")
    if manifest.get("status") != "awaiting_human_promotion":
        raise UpgradeControllerError("campaign is not awaiting promotion")
    selected_id = str((manifest.get("selection") or {}).get("candidate_id") or "")
    selected = next((item for item in manifest.get("candidates", []) if item.get("candidate_id") == selected_id), None)
    if not isinstance(selected, dict) or selected.get("strictly_improves_baseline") is not True:
        raise UpgradeControllerError("selected candidate is not a strict improvement")
    evaluator = _verify_locked_evaluator(repository_path, manifest)
    baseline_evaluation = manifest.get("baseline")
    candidate_evaluation = selected.get("evaluation")
    evaluation_evidence_ok = (
        evaluation_attestation_matches(baseline_evaluation)
        and evaluation_attestation_matches(candidate_evaluation)
        and isinstance(candidate_evaluation, Mapping)
        and selected.get("evaluation_digest_sha256") == candidate_evaluation.get("evaluation_digest_sha256")
        and _strict_improvement(candidate_evaluation, baseline_evaluation, evaluator)
    )
    if not evaluation_evidence_ok:
        raise UpgradeControllerError("selected candidate strict improvement evidence is invalid")
    candidate_commit = _read_ref(repository_path, selected["candidate_ref"])
    if candidate_commit != selected["commit_sha"]:
        raise UpgradeControllerError("candidate ref no longer matches the evaluated commit")
    if _git_ok(repository_path, "rev-parse", f"{candidate_commit}^{{tree}}") != selected["tree_sha"]:
        raise UpgradeControllerError("candidate tree no longer matches evaluation evidence")

    lock_path = repository_path / ".xsci" / "scientist_upgrade_campaign.lock"
    with _CampaignLock(lock_path, str(manifest["campaign_id"])):
        previous = _read_ref(repository_path, champion_ref)
        previous_target = previous or str(manifest["base_commit"])
        previous_tree = _git_ok(repository_path, "rev-parse", f"{previous_target}^{{tree}}")
        rollback_test = _rollback_ref_test(
            repository_path,
            campaign_id=str(manifest["campaign_id"]),
            base_commit=str(manifest["base_commit"]),
            candidate_commit=candidate_commit,
            base_tree=str(manifest["base_tree"]),
        )
        if rollback_test.get("passed") is not True:
            raise UpgradeControllerError("pre-promotion rollback CAS test failed")
        _cas_ref(repository_path, champion_ref, candidate_commit, previous)
        activation_raw = dict(activation_callback(repository_path, candidate_commit, selected["tree_sha"]))
        activation_passed = (
            activation_raw.get("passed") is True
            and activation_raw.get("runtime_tree_sha") == selected["tree_sha"]
        )
        if not activation_passed:
            if previous:
                _cas_ref(repository_path, champion_ref, previous, candidate_commit)
            else:
                _delete_ref(repository_path, champion_ref, candidate_commit)
            manifest["status"] = "rolled_back_after_failed_activation"
            manifest["promotion"] = {
                "candidate_id": selected_id,
                "human_approved": True,
                "verified": False,
                "previous_champion": previous,
                "activation": activation_raw,
            }
            manifest["rollback"] = {
                "tested": True,
                "passed": _read_ref(repository_path, champion_ref) == previous,
                "restored_tree_sha": previous_tree,
            }
            manifest["attestation"] = {"algorithm": "sha256", "payload_sha256": _manifest_attestation(manifest)}
            _write_json(path, manifest)
            raise UpgradeControllerError("runtime activation canary failed; previous champion restored")

        source_identity = compute_repository_identity(repository_path)
        # The committed source digest is recomputed for the candidate by a detached worktree.
        candidate_source_digest = _with_worktree(
            repository_path,
            candidate_commit,
            lambda worktree: compute_repository_identity(worktree).source_digest_sha256,
        )
        manifest["status"] = "active"
        manifest["champion_ref"] = champion_ref
        manifest["repository_path"] = str(repository_path)
        manifest["promotion"] = {
            "candidate_id": selected_id,
            "human_approved": True,
            "verified": True,
            "previous_champion": previous,
            "previous_tree_sha": previous_tree,
            "promoted_commit_sha": candidate_commit,
            "promoted_tree_sha": selected["tree_sha"],
            "source_digest_sha256": candidate_source_digest,
            "activation": activation_raw,
            "repository_head_at_promotion": source_identity.commit_sha,
        }
        manifest["rollback"] = rollback_test
        manifest["selection"] = {
            "candidate_id": selected_id,
            "decision": "promoted",
            "strictly_improves_baseline": True,
        }
        manifest["attestation"] = {"algorithm": "sha256", "payload_sha256": _manifest_attestation(manifest)}
        evidence = _certification_manifest(manifest, selected)
        evidence_path = Path(str(manifest["certification_manifest_path"]))
        _write_json(evidence_path, evidence)
        _write_json(path, manifest)
        latest = repository_path / ".xsci" / "scientist_upgrade_campaign.json"
        _write_json(latest, manifest)
        _record_event(
            Path(str(manifest["history_path"])),
            {"ts": _utc_seconds(), "phase": "promote", "candidate_id": selected_id, "tree_sha": selected["tree_sha"]},
        )
        return manifest


def rollback_upgrade_campaign(
    repository: Path | str,
    manifest_path: Path | str,
    *,
    activation_callback: ActivationCallback,
    champion_ref: str = CHAMPION_REF,
) -> dict[str, Any]:
    """Rollback an active campaign without overwriting a concurrently advanced ref."""

    repository_path = Path(repository).resolve()
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    _verify_manifest_attestation(manifest)
    if manifest.get("status") == "rolled_back":
        return manifest
    if manifest.get("status") != "active" or not isinstance(manifest.get("promotion"), dict):
        raise UpgradeControllerError("campaign is not active")
    current = str(manifest["promotion"]["promoted_commit_sha"])
    previous = str(manifest["promotion"].get("previous_champion") or manifest["base_commit"])
    previous_tree = _git_ok(repository_path, "rev-parse", f"{previous}^{{tree}}")
    if _read_ref(repository_path, champion_ref) != current:
        raise UpgradeControllerError("champion ref advanced concurrently; rollback refused")
    _cas_ref(repository_path, champion_ref, previous, current)
    activation = dict(activation_callback(repository_path, previous, previous_tree))
    passed = activation.get("passed") is True and activation.get("runtime_tree_sha") == previous_tree
    if not passed:
        raise UpgradeControllerError("rollback ref moved but previous runtime activation did not verify")
    manifest["status"] = "rolled_back"
    manifest["rollback"] = {"tested": True, "passed": True, "restored_tree_sha": previous_tree, "activation": activation}
    manifest["attestation"] = {"algorithm": "sha256", "payload_sha256": _manifest_attestation(manifest)}
    _write_json(path, manifest)
    _write_json(repository_path / ".xsci" / "scientist_upgrade_campaign.json", manifest)
    return manifest


def read_upgrade_campaign_status(repository: Path | str) -> dict[str, Any]:
    path = Path(repository).resolve() / ".xsci" / "scientist_upgrade_campaign.json"
    if not path.is_file():
        return {
            "ok": True,
            "tool": "scientist_upgrade_campaign",
            "status": "not_run",
            "artifact_path": str(path),
            "human_gate": "build_and_evaluate_multiple_candidates_first",
        }
    payload = _read_json(path)
    payload["ok"] = payload.get("status") in {
        "awaiting_human_promotion",
        "held_no_strict_improvement",
        "active",
        "rolled_back",
    }
    payload["tool"] = "scientist_upgrade_campaign"
    payload["artifact_path"] = str(path)
    return payload


__all__ = [
    "CHAMPION_REF",
    "CONTROLLER_SCHEMA",
    "CONTRACT_SCHEMA",
    "DEFAULT_ALLOWED_PREFIXES",
    "DEFAULT_PROTECTED_PREFIXES",
    "EvaluatorContract",
    "UpgradeControllerError",
    "evaluation_attestation_matches",
    "promote_upgrade_campaign",
    "read_upgrade_campaign_status",
    "rollback_upgrade_campaign",
    "run_upgrade_campaign",
]
