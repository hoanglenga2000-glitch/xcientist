#!/usr/bin/env python3
"""Execute the SIIM MLE-Bench private grader exactly once after candidate freeze.

The local half validates the independent review and every frozen artifact, then
dispatches an immutable request through the Run-bound named SSH profile.  The
remote half is the only code path that opens MLE-Bench's private answers.  It
records an append-only claim before grading and an immutable receipt afterward.

This command does not train, tune, resume a workflow, contact Kaggle, or signal
any process.  Its sanitized result is staged as SIIM workflow ingress; raw
grader evidence and private-answer paths stay on the remote host.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import posixpath
import shlex
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from research_os.siim_hpc_binding import binding_from_environment  # noqa: E402

BINDING = binding_from_environment()
HPC_JOB_ID = BINDING.job_id
CREDENTIAL_PROFILE = BINDING.credential_profile
JOB_TAG = BINDING.job_tag
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
COMPETITION_ID = "siim-isic-melanoma-classification"
CANDIDATE_FREEZE_SCHEMA = "evomind.siim.candidate_freeze.v1"
REQUEST_SCHEMA = "evomind.siim.private_grader_request.v1"
CLAIM_SCHEMA = "evomind.siim.private_grader_claim.v1"
RESULT_SCHEMA = "evomind.siim.private_grader.v1"
RECEIPT_SCHEMA = "evomind.siim.private_grader_receipt.v1"
LOCAL_LEDGER_SCHEMA = "evomind.siim.private_grader_once_ledger.v1"
EXPECTED_TEST_ROWS = 4_142
EXPECTED_REVIEW_CHECKS = (
    "patient_group_overlap_zero",
    "content_group_overlap_zero",
    "oof_coverage_exactly_once",
    "submission_schema_and_order",
    "private_labels_unavailable_during_training",
    "private_grader_not_executed",
    "official_submission_not_executed",
)
REMOTE_PRIVATE_GRADER_PARENT_RELATIVE = f"siim_{JOB_TAG}/private_grader"
REMOTE_DATA_ROOT_RELATIVE = "mlebench_official_data"
REMOTE_OFFICIAL_SOURCE_RELATIVE = "mle-bench"
REMOTE_RAW_GRADER_NAME = "private_grader_raw_remote.json"
DEDICATED_REMOTE_ENVIRONMENT_KEYS = (
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "CUPY_CACHE_DIR",
    "JOBLIB_TEMP_FOLDER",
    "PYTHONPYCACHEPREFIX",
    "KAGGLE_CONFIG_DIR",
)
SAFE_RUN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


class PrivateGraderOnceError(RuntimeError):
    """Fail-closed private-grader contract error."""


@dataclass(frozen=True)
class PreparedCandidate:
    project_root: Path
    run_id: str
    run_dir: Path
    freeze_path: Path
    review_path: Path
    submission_path: Path
    candidate_freeze_sha256: str
    review_sha256: str
    submission_sha256: str
    runner_sha256: str
    request: dict[str, Any]


@dataclass(frozen=True)
class RemoteDispatchEvidence:
    result_bytes: bytes
    receipt_bytes: bytes
    claim_sha256: str
    remote_root: str
    idempotent_reuse: bool
    remote_environment_root: str = ""


RemoteDispatcher = Callable[
    [PreparedCandidate, Mapping[str, Any], Path], RemoteDispatchEvidence
]
RemoteGrader = Callable[..., dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalize_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PrivateGraderOnceError(f"{label} must be a SHA-256 hex digest")
    return digest


def safe_run_id(value: str) -> str:
    if not value or len(value) > 160 or any(character not in SAFE_RUN_CHARACTERS for character in value):
        raise PrivateGraderOnceError("invalid SIIM run_id")
    return value


def exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise PrivateGraderOnceError(f"missing {label}: {path.name}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivateGraderOnceError(f"invalid {label}: {path.name}") from exc
    if not isinstance(payload, dict):
        raise PrivateGraderOnceError(f"{label} must contain a JSON object")
    return payload


def write_bytes_exclusive(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PrivateGraderOnceError(f"immutable evidence already exists: {path.name}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def ensure_same_or_write(path: Path, payload: bytes, *, label: str) -> str:
    if path.is_file():
        if path.read_bytes() != payload:
            raise PrivateGraderOnceError(f"conflicting immutable {label}")
        return "reused"
    write_bytes_exclusive(path, payload)
    return "created"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(json_bytes(dict(payload)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def resolve_within(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not relative:
        raise PrivateGraderOnceError(f"unsafe {label} path")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise PrivateGraderOnceError(f"{label} path escaped its root")
    return resolved


def require_remote_within(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise PrivateGraderOnceError(f"{label} escaped the dedicated remote root")
    return path


def validate_dedicated_process_environment(allowed_root: Path) -> dict[str, str]:
    """Fail closed unless every runtime/cache/temp variable stays in the root."""

    allowed_root = allowed_root.resolve()
    validated: dict[str, str] = {}
    for name in DEDICATED_REMOTE_ENVIRONMENT_KEYS:
        value = str(os.environ.get(name) or "").strip()
        if not value:
            raise PrivateGraderOnceError(
                f"remote private-grader environment is missing {name}"
            )
        path = require_remote_within(
            allowed_root,
            Path(value),
            label=f"remote private-grader {name}",
        )
        validated[name] = str(path)
    return validated


def require_regular_non_symlink_file(path: Path, *, label: str) -> Path:
    """Require a non-empty regular file without following a final symlink."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PrivateGraderOnceError(f"missing {label}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PrivateGraderOnceError(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateGraderOnceError(f"{label} must be a regular file")
    if metadata.st_size <= 0:
        raise PrivateGraderOnceError(f"{label} must not be empty")
    return path


def _review_passed(review: Mapping[str, Any]) -> bool:
    return bool(
        review.get("status") in {"passed", "review_passed", "approved", "verified"}
        or review.get("passed") is True
        or review.get("ok") is True
    )


def _validate_submission_csv(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["image_name", "target"]:
                raise PrivateGraderOnceError("frozen submission columns are not image_name,target")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PrivateGraderOnceError("frozen submission is not a readable CSV") from exc
    if len(rows) != EXPECTED_TEST_ROWS:
        raise PrivateGraderOnceError("frozen submission row count is not 4,142")
    identifiers: list[str] = []
    for row in rows:
        image_name = str(row.get("image_name") or "").strip()
        if not image_name:
            raise PrivateGraderOnceError("frozen submission contains a missing image_name")
        identifiers.append(image_name)
        try:
            target = float(row.get("target", ""))
        except (TypeError, ValueError) as exc:
            raise PrivateGraderOnceError("frozen submission contains a nonnumeric target") from exc
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise PrivateGraderOnceError("frozen submission target is outside [0, 1]")
    if len(set(identifiers)) != len(identifiers):
        raise PrivateGraderOnceError("frozen submission contains duplicate image_name values")


def _validate_review(
    review: Mapping[str, Any],
    *,
    run_id: str,
    frozen_hashes: Mapping[str, str],
) -> None:
    if review.get("run_id") != run_id:
        raise PrivateGraderOnceError("independent review run_id mismatch")
    if not _review_passed(review):
        raise PrivateGraderOnceError("independent review is not passed")
    checks = review.get("checks")
    if not isinstance(checks, dict):
        raise PrivateGraderOnceError("independent review checks are missing")
    failed = [name for name in EXPECTED_REVIEW_CHECKS if checks.get(name) is not True]
    if failed:
        raise PrivateGraderOnceError(f"independent review check failed: {failed[0]}")
    reviewed_hashes = review.get("artifact_hashes")
    if not isinstance(reviewed_hashes, dict):
        raise PrivateGraderOnceError("independent review artifact hashes are missing")
    for relative, digest in frozen_hashes.items():
        if normalize_sha256(reviewed_hashes.get(relative), label=f"review hash for {relative}") != digest:
            raise PrivateGraderOnceError(f"independent review hash mismatch: {relative}")


def _request_core(
    *,
    run_id: str,
    candidate_freeze_sha256: str,
    review_sha256: str,
    submission_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA,
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "competition_id": COMPETITION_ID,
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "review_sha256": review_sha256,
        "submission_sha256": submission_sha256,
        "runner_sha256": runner_sha256,
        "execution_index": 1,
        "private_label_scope": "remote_terminal_grader_only",
        "grader_feedback_policy": "terminal_delivery_only_no_tuning",
        "official_submission": "forbidden",
        "kaggle_submission": "forbidden",
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def build_request(**kwargs: str) -> dict[str, Any]:
    core = _request_core(**kwargs)
    return {**core, "request_id": sha256_bytes(canonical_json_bytes(core))}


def prepare_candidate(project_root: Path, run_id: str) -> PreparedCandidate:
    project_root = project_root.expanduser().resolve()
    run_id = safe_run_id(run_id)
    run_dir = (project_root / "workspace" / "evomind_runs" / run_id).resolve()
    expected_runs_root = (project_root / "workspace" / "evomind_runs").resolve()
    if expected_runs_root not in run_dir.parents or not run_dir.is_dir():
        raise PrivateGraderOnceError("SIIM run directory is missing")

    freeze_path = run_dir / "candidate_freeze.json"
    review_path = run_dir / "review.json"
    submission_path = run_dir / "submission.csv"
    freeze = read_json(freeze_path, label="candidate freeze")
    if freeze.get("schema") != CANDIDATE_FREEZE_SCHEMA:
        raise PrivateGraderOnceError("candidate freeze schema is unsupported")
    if freeze.get("run_id") != run_id:
        raise PrivateGraderOnceError("candidate freeze run_id mismatch")
    if freeze.get("status") != "frozen_before_private_grader":
        raise PrivateGraderOnceError("candidate is not frozen before private grading")
    if freeze.get("tuning_closed") is not True:
        raise PrivateGraderOnceError("candidate tuning is not closed")
    if freeze.get("official_submission") != "forbidden":
        raise PrivateGraderOnceError("candidate freeze does not forbid official submission")
    if not exact_int(freeze.get("private_grader_execution_count_before_freeze"), 0):
        raise PrivateGraderOnceError("candidate was graded before freeze")

    records = freeze.get("artifacts")
    if not isinstance(records, list) or not records:
        raise PrivateGraderOnceError("candidate freeze artifact list is empty")
    frozen_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise PrivateGraderOnceError("candidate freeze contains an invalid artifact record")
        relative = str(record.get("path") or "")
        if relative in frozen_hashes:
            raise PrivateGraderOnceError(f"candidate freeze contains duplicate artifact: {relative}")
        artifact = resolve_within(run_dir, relative, label="frozen artifact")
        if not artifact.is_file():
            raise PrivateGraderOnceError(f"frozen artifact is missing: {relative}")
        actual = sha256_file(artifact)
        expected = normalize_sha256(record.get("sha256"), label=f"freeze hash for {relative}")
        if actual != expected:
            raise PrivateGraderOnceError(f"frozen artifact changed: {relative}")
        if record.get("bytes") is not None and not exact_int(
            record.get("bytes"), artifact.stat().st_size
        ):
            raise PrivateGraderOnceError(f"frozen artifact size changed: {relative}")
        frozen_hashes[relative] = actual
    if "submission.csv" not in frozen_hashes:
        raise PrivateGraderOnceError("candidate freeze does not bind submission.csv")
    if submission_path.resolve() != resolve_within(run_dir, "submission.csv", label="submission"):
        raise PrivateGraderOnceError("submission path contract failed")
    _validate_submission_csv(submission_path)

    review = read_json(review_path, label="independent review")
    _validate_review(review, run_id=run_id, frozen_hashes=frozen_hashes)
    candidate_freeze_sha256 = sha256_file(freeze_path)
    review_sha256 = sha256_file(review_path)
    submission_sha256 = sha256_file(submission_path)
    if frozen_hashes["submission.csv"] != submission_sha256:
        raise PrivateGraderOnceError("submission hash differs from the candidate freeze")
    runner_sha256 = sha256_file(Path(__file__).resolve())
    request = build_request(
        run_id=run_id,
        candidate_freeze_sha256=candidate_freeze_sha256,
        review_sha256=review_sha256,
        submission_sha256=submission_sha256,
        runner_sha256=runner_sha256,
    )
    return PreparedCandidate(
        project_root=project_root,
        run_id=run_id,
        run_dir=run_dir,
        freeze_path=freeze_path,
        review_path=review_path,
        submission_path=submission_path,
        candidate_freeze_sha256=candidate_freeze_sha256,
        review_sha256=review_sha256,
        submission_sha256=submission_sha256,
        runner_sha256=runner_sha256,
        request=request,
    )


def validate_request_for_candidate(request: Mapping[str, Any], prepared: PreparedCandidate) -> dict[str, Any]:
    expected = {
        "schema": REQUEST_SCHEMA,
        "run_id": prepared.run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "competition_id": COMPETITION_ID,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "execution_index": 1,
        "private_label_scope": "remote_terminal_grader_only",
        "grader_feedback_policy": "terminal_delivery_only_no_tuning",
        "official_submission": "forbidden",
        "kaggle_submission": "forbidden",
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise PrivateGraderOnceError(f"existing private-grader request conflicts on {key}")
    runner_sha256 = normalize_sha256(request.get("runner_sha256"), label="request runner SHA-256")
    core = dict(request)
    request_id = normalize_sha256(core.pop("request_id", None), label="request id")
    if set(core) != set(_request_core(
        run_id=prepared.run_id,
        candidate_freeze_sha256=prepared.candidate_freeze_sha256,
        review_sha256=prepared.review_sha256,
        submission_sha256=prepared.submission_sha256,
        runner_sha256=runner_sha256,
    )):
        raise PrivateGraderOnceError("existing private-grader request has unsupported fields")
    if sha256_bytes(canonical_json_bytes(core)) != request_id:
        raise PrivateGraderOnceError("existing private-grader request id is invalid")
    return dict(request)


def load_or_create_request(control_dir: Path, prepared: PreparedCandidate) -> dict[str, Any]:
    request_path = control_dir / "request.json"
    if request_path.is_file():
        return validate_request_for_candidate(read_json(request_path, label="private-grader request"), prepared)
    ensure_same_or_write(request_path, json_bytes(prepared.request), label="private-grader request")
    return dict(prepared.request)


def validate_result(
    result: Mapping[str, Any],
    prepared: PreparedCandidate,
    *,
    expected_execution_id: str,
) -> float:
    expected = {
        "schema": RESULT_SCHEMA,
        "run_id": prepared.run_id,
        "competition_id": COMPETITION_ID,
        "status": "passed",
        "execution_id": expected_execution_id,
        "execution_index": 1,
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "executed_after_freeze": True,
        "official_mlebench_grader_executed": True,
        "feedback_used_for_tuning": False,
        "post_grader_tuning": "forbidden",
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "training_artifacts_modified": False,
        "private_labels_exported": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise PrivateGraderOnceError(f"private-grader result contract failed: {key}")
    if not exact_int(result.get("execution_index"), 1) or not exact_int(
        result.get("execution_count"), 1
    ):
        raise PrivateGraderOnceError("private-grader execution counters are not exact integers")
    normalize_sha256(result.get("private_answers_sha256"), label="private answers SHA-256")
    normalize_sha256(result.get("raw_grader_sha256"), label="raw grader evidence SHA-256")
    try:
        score = float(result.get("mle_private_grader_score"))
    except (TypeError, ValueError) as exc:
        raise PrivateGraderOnceError("private-grader score is not numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise PrivateGraderOnceError("private-grader score is outside [0, 1]")
    try:
        score_alias = float(result.get("score"))
    except (TypeError, ValueError) as exc:
        raise PrivateGraderOnceError("private-grader score alias is not numeric") from exc
    if score_alias != score:
        raise PrivateGraderOnceError("private-grader score fields disagree")
    return score


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    prepared: PreparedCandidate,
    request: Mapping[str, Any],
    result_sha256: str,
) -> None:
    expected = {
        "schema": RECEIPT_SCHEMA,
        "run_id": prepared.run_id,
        "competition_id": COMPETITION_ID,
        "status": "passed",
        "execution_id": request["request_id"],
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "submission_sha256": prepared.submission_sha256,
        "result_sha256": result_sha256,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PrivateGraderOnceError(f"private-grader receipt contract failed: {key}")
    if not exact_int(receipt.get("execution_count"), 1):
        raise PrivateGraderOnceError("private-grader receipt execution count is not an exact integer")
    normalize_sha256(receipt.get("claim_sha256"), label="remote claim SHA-256")
    if not str(receipt.get("completed_at") or "").strip():
        raise PrivateGraderOnceError("private-grader receipt completion time is missing")


def _validate_local_ledger(
    ledger: Mapping[str, Any],
    *,
    prepared: PreparedCandidate,
    result_sha256: str,
) -> str:
    expected = {
        "schema": LOCAL_LEDGER_SCHEMA,
        "run_id": prepared.run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "competition_id": COMPETITION_ID,
        "status": "terminal_execution_recorded",
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "result_sha256": result_sha256,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise PrivateGraderOnceError(f"local exactly-once ledger conflict: {key}")
    if not exact_int(ledger.get("execution_count"), 1):
        raise PrivateGraderOnceError("local exactly-once ledger count is not an exact integer")
    return normalize_sha256(ledger.get("execution_id"), label="ledger execution id")


def _fast_local_reuse(prepared: PreparedCandidate, control_dir: Path) -> dict[str, Any] | None:
    result_path = control_dir / "private_grader.json"
    ledger_path = control_dir / "private_grader_once_ledger.json"
    if not result_path.exists() and not ledger_path.exists():
        return None
    if not result_path.is_file() or not ledger_path.is_file():
        return None
    result_bytes = result_path.read_bytes()
    result = read_json(result_path, label="local private-grader result")
    ledger = read_json(ledger_path, label="local private-grader ledger")
    execution_id = _validate_local_ledger(
        ledger,
        prepared=prepared,
        result_sha256=sha256_bytes(result_bytes),
    )
    score = validate_result(result, prepared, expected_execution_id=execution_id)
    ingress = prepared.project_root / "workspace" / "siim_hpc_ingress" / prepared.run_id / "private_grader.json"
    ensure_same_or_write(ingress, result_bytes, label="SIIM private-grader ingress")
    return {
        "schema": "evomind.siim.private_grader_once_run.v1",
        "status": "reused_existing_result",
        "run_id": prepared.run_id,
        "execution_id": execution_id,
        "execution_count": 1,
        "mle_private_grader_score": score,
        "result_path": str(result_path),
        "ledger_path": str(ledger_path),
        "ingress_path": str(ingress),
        "idempotent_reuse": True,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _load_deployment(prepared: PreparedCandidate) -> dict[str, Any]:
    path = (
        prepared.project_root
        / "workspace"
        / "hpc"
        / f"{JOB_TAG}_siim_campaign"
        / prepared.run_id
        / "deployment.json"
    )
    deployment = read_json(path, label=f"{JOB_TAG} deployment")
    if deployment.get("schema") != BINDING.schema("deployment"):
        raise PrivateGraderOnceError(f"{JOB_TAG} deployment schema is unsupported")
    if deployment.get("run_id") != prepared.run_id:
        raise PrivateGraderOnceError(f"{JOB_TAG} deployment run_id mismatch")
    if int(deployment.get("job_id") or 0) != HPC_JOB_ID:
        raise PrivateGraderOnceError(f"{JOB_TAG} deployment job binding is not exact")
    if deployment.get("credential_profile") != CREDENTIAL_PROFILE:
        raise PrivateGraderOnceError(f"{JOB_TAG} named credential profile is not bound")
    if deployment.get("remote_root") != REMOTE_ROOT:
        raise PrivateGraderOnceError(f"{JOB_TAG} deployment remote root is not confined")
    remote = deployment.get("remote")
    if not isinstance(remote, dict) or not remote.get("bundle"):
        raise PrivateGraderOnceError(f"{JOB_TAG} deployment remote bundle is missing")
    return deployment


def _remote_sha256(client: Any, path: str, run_remote: Callable[..., tuple[int, str, str]]) -> str:
    with client.open_sftp() as sftp:
        try:
            metadata = sftp.lstat(path)
        except OSError:
            return ""
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PrivateGraderOnceError(
                "remote private-grader hash target is not a regular non-link file"
            )
    command = f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)} | cut -d' ' -f1; fi"
    code, output, _error = run_remote(client, command)
    if code or not output.strip():
        return ""
    candidate = output.strip().splitlines()[-1]
    try:
        return normalize_sha256(candidate, label="remote file SHA-256")
    except PrivateGraderOnceError:
        return ""


def _upload_remote_exclusive(
    client: Any,
    local: Path,
    remote: str,
    *,
    run_remote: Callable[..., tuple[int, str, str]],
) -> str:
    expected = sha256_file(local)
    existing = _remote_sha256(client, remote, run_remote)
    if existing:
        if existing != expected:
            raise PrivateGraderOnceError(f"conflicting immutable remote input: {PurePosixPath(remote).name}")
        return "reused"
    parent = posixpath.dirname(remote)
    code, _output, _error = run_remote(
        client,
        f"umask 077; mkdir -p -m 700 -- {shlex.quote(parent)}",
    )
    if code:
        raise PrivateGraderOnceError("remote private-grader directory creation failed")
    partial = f"{remote}.part.{os.getpid()}"
    with client.open_sftp() as sftp:
        sftp.put(str(local), partial)
        sftp.chmod(partial, 0o600)
    command = (
        "set -eu; "
        f"if ln -- {shlex.quote(partial)} {shlex.quote(remote)} 2>/dev/null; then "
        f"rm -f -- {shlex.quote(partial)}; echo created; else "
        f"rm -f -- {shlex.quote(partial)}; echo exists; fi"
    )
    code, _output, _error = run_remote(client, command)
    if code or _remote_sha256(client, remote, run_remote) != expected:
        raise PrivateGraderOnceError(f"remote immutable upload failed: {PurePosixPath(remote).name}")
    return "created"


def _download_remote_file(client: Any, remote: str, local: Path) -> bytes:
    temporary = local.with_name(f".{local.name}.{os.getpid()}.download")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with client.open_sftp() as sftp:
            sftp.get(remote, str(temporary))
        payload = temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def dispatch_remote(
    prepared: PreparedCandidate,
    request: Mapping[str, Any],
    control_dir: Path,
) -> RemoteDispatchEvidence:
    """Upload immutable inputs, execute the remote-only worker, and collect sanitized evidence."""

    for entry in (str(prepared.project_root), str(prepared.project_root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from research_agent_workstation.server.core.gpu_credentials import ALLOWED_GPU_REMOTE_ROOT
    from scripts import manage_siim_job89508_campaign as manager
    from scripts import mlebench_remote_ops as remote_ops

    deployment = _load_deployment(prepared)
    remote_deployment = deployment["remote"]
    bundle = manager.ensure_remote(str(remote_deployment["bundle"]))
    remote_root = manager.ensure_remote(
        f"{ALLOWED_GPU_REMOTE_ROOT}/{REMOTE_PRIVATE_GRADER_PARENT_RELATIVE}/{prepared.run_id}"
    )
    environment_root = manager.ensure_remote(f"{remote_root}/runtime_environment")
    cache_root = manager.ensure_remote(f"{environment_root}/cache")
    temporary_root = manager.ensure_remote(
        f"{ALLOWED_GPU_REMOTE_ROOT}/.t/"
        f"{hashlib.sha256(('grader:' + prepared.run_id).encode('utf-8')).hexdigest()[:12]}"
    )
    remote_environment = {
        "HOME": manager.ensure_remote(f"{environment_root}/home"),
        "TMPDIR": temporary_root,
        "TEMP": temporary_root,
        "TMP": temporary_root,
        "XDG_CACHE_HOME": manager.ensure_remote(f"{cache_root}/xdg"),
        "HF_HOME": manager.ensure_remote(f"{cache_root}/huggingface"),
        "TORCH_HOME": manager.ensure_remote(
            f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_model_cache/torch"
        ),
        "PIP_CACHE_DIR": manager.ensure_remote(f"{cache_root}/pip"),
        "MPLCONFIGDIR": manager.ensure_remote(f"{cache_root}/matplotlib"),
        "NUMBA_CACHE_DIR": manager.ensure_remote(f"{cache_root}/numba"),
        "CUDA_CACHE_PATH": manager.ensure_remote(f"{cache_root}/cuda"),
        "TRITON_CACHE_DIR": manager.ensure_remote(f"{cache_root}/triton"),
        "CUPY_CACHE_DIR": manager.ensure_remote(f"{cache_root}/cupy"),
        "JOBLIB_TEMP_FOLDER": temporary_root,
        "PYTHONPYCACHEPREFIX": manager.ensure_remote(f"{cache_root}/python"),
        "KAGGLE_CONFIG_DIR": manager.ensure_remote(
            f"{environment_root}/kaggle_disabled"
        ),
    }
    remote_files = {
        "request": manager.ensure_remote(f"{remote_root}/request.json"),
        "freeze": manager.ensure_remote(f"{remote_root}/candidate_freeze.json"),
        "review": manager.ensure_remote(f"{remote_root}/review.json"),
        "submission": manager.ensure_remote(f"{remote_root}/submission.csv"),
        "runner": manager.ensure_remote(f"{remote_root}/runner.py"),
        "result": manager.ensure_remote(f"{remote_root}/result.json"),
        "receipt": manager.ensure_remote(f"{remote_root}/execution_receipt.json"),
        "claim": manager.ensure_remote(f"{remote_root}/execution_claim.json"),
    }
    request_path = control_dir / "request.json"
    current_runner = Path(__file__).resolve()
    request_runner_sha = normalize_sha256(request.get("runner_sha256"), label="request runner SHA-256")

    client, _config = manager.connect()
    try:
        with client.open_sftp() as sftp:
            manager.verify_remote_path_chain(sftp, remote_root)
            for path in remote_environment.values():
                manager.verify_remote_path_chain(sftp, path)
        code, _output, _error = manager.run_remote(
            client,
            "mkdir -p -m 700 -- "
            + " ".join(
                shlex.quote(path)
                for path in dict.fromkeys((remote_root, *remote_environment.values()))
            ),
        )
        if code:
            raise PrivateGraderOnceError("remote private-grader root creation failed")
        _upload_remote_exclusive(
            client, request_path, remote_files["request"], run_remote=manager.run_remote
        )
        _upload_remote_exclusive(
            client, prepared.freeze_path, remote_files["freeze"], run_remote=manager.run_remote
        )
        _upload_remote_exclusive(
            client, prepared.review_path, remote_files["review"], run_remote=manager.run_remote
        )
        _upload_remote_exclusive(
            client, prepared.submission_path, remote_files["submission"], run_remote=manager.run_remote
        )
        remote_runner_sha = _remote_sha256(client, remote_files["runner"], manager.run_remote)
        if not remote_runner_sha:
            if sha256_file(current_runner) != request_runner_sha:
                raise PrivateGraderOnceError("the requested grader runner is no longer available")
            _upload_remote_exclusive(
                client, current_runner, remote_files["runner"], run_remote=manager.run_remote
            )
        elif remote_runner_sha != request_runner_sha:
            raise PrivateGraderOnceError("conflicting immutable remote grader runner")

        pythonpath = ":".join(
            (
                bundle,
                f"{bundle}/src",
                remote_ops.REMOTE_UNIFIED_SITE_PACKAGES,
                remote_ops.REMOTE_GRADER_SITE_PACKAGES,
            )
        )
        data_root = manager.ensure_remote(f"{ALLOWED_GPU_REMOTE_ROOT}/{REMOTE_DATA_ROOT_RELATIVE}")
        official_source = manager.ensure_remote(
            f"{ALLOWED_GPU_REMOTE_ROOT}/{REMOTE_OFFICIAL_SOURCE_RELATIVE}"
        )
        environment_prefix = " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in remote_environment.items()
        )
        command = (
            f"cd -- {shlex.quote(remote_root)} && umask 077 && "
            "env -u KAGGLE_USERNAME -u KAGGLE_KEY "
            f"{environment_prefix} "
            f"EVOMIND_SIIM_HPC_JOB_ID={HPC_JOB_ID} "
            f"EVOMIND_HPC_CREDENTIAL_PROFILE={shlex.quote(CREDENTIAL_PROFILE)} "
            f"EVOMIND_SIIM_RUN_ID={shlex.quote(prepared.run_id)} "
            "PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "
            f"PYTHONPATH={shlex.quote(pythonpath)} python3 {shlex.quote(remote_files['runner'])} "
            f"--remote-worker --request-dir {shlex.quote(remote_root)} "
            f"--data-root {shlex.quote(data_root)} "
            f"--official-source-root {shlex.quote(official_source)} "
            f"--allowed-root {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)} "
            f"--bundle-root {shlex.quote(bundle)}"
        )
        code, output, _error = manager.run_remote(client, command, timeout=1_800)
        if code:
            raise PrivateGraderOnceError("remote terminal private grader failed closed; inspect remote evidence")
        try:
            worker_status = json.loads(output.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise PrivateGraderOnceError("remote terminal private grader returned invalid status") from exc
        if worker_status.get("status") not in {"passed", "reused_existing_result"}:
            raise PrivateGraderOnceError("remote terminal private grader did not complete")
        result_bytes = _download_remote_file(
            client, remote_files["result"], control_dir / "remote-result.download"
        )
        receipt_bytes = _download_remote_file(
            client, remote_files["receipt"], control_dir / "remote-receipt.download"
        )
        claim_sha256 = _remote_sha256(client, remote_files["claim"], manager.run_remote)
        if not claim_sha256:
            raise PrivateGraderOnceError("remote terminal private-grader claim is missing")
    finally:
        client.close()
    return RemoteDispatchEvidence(
        result_bytes=result_bytes,
        receipt_bytes=receipt_bytes,
        claim_sha256=claim_sha256,
        remote_root=remote_root,
        idempotent_reuse=worker_status.get("status") == "reused_existing_result",
        remote_environment_root=environment_root,
    )


def run_private_grader_once(
    project_root: Path,
    run_id: str,
    *,
    dispatcher: RemoteDispatcher | None = None,
) -> dict[str, Any]:
    prepared = prepare_candidate(project_root, run_id)
    control_dir = (
        prepared.project_root
        / "workspace"
        / "hpc"
        / f"{JOB_TAG}_siim_private_grader"
        / prepared.run_id
    )
    control_dir.mkdir(parents=True, exist_ok=True)
    reused = _fast_local_reuse(prepared, control_dir)
    if reused is not None:
        return reused

    request = load_or_create_request(control_dir, prepared)
    dispatch = (dispatcher or dispatch_remote)(prepared, request, control_dir)
    result_payload = json.loads(dispatch.result_bytes.decode("utf-8-sig"))
    receipt_payload = json.loads(dispatch.receipt_bytes.decode("utf-8-sig"))
    if not isinstance(result_payload, dict) or not isinstance(receipt_payload, dict):
        raise PrivateGraderOnceError("remote private-grader evidence must contain JSON objects")
    score = validate_result(
        result_payload,
        prepared,
        expected_execution_id=normalize_sha256(request.get("request_id"), label="request id"),
    )
    result_sha256 = sha256_bytes(dispatch.result_bytes)
    _validate_receipt(
        receipt_payload,
        prepared=prepared,
        request=request,
        result_sha256=result_sha256,
    )
    if normalize_sha256(receipt_payload.get("claim_sha256"), label="receipt claim SHA-256") != normalize_sha256(
        dispatch.claim_sha256, label="remote claim SHA-256"
    ):
        raise PrivateGraderOnceError("remote claim hash differs from the receipt")

    result_path = control_dir / "private_grader.json"
    receipt_path = control_dir / "remote_execution_receipt.json"
    ensure_same_or_write(result_path, dispatch.result_bytes, label="local private-grader result")
    ensure_same_or_write(receipt_path, dispatch.receipt_bytes, label="remote private-grader receipt")
    ledger = {
        "schema": LOCAL_LEDGER_SCHEMA,
        "run_id": prepared.run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "competition_id": COMPETITION_ID,
        "status": "terminal_execution_recorded",
        "execution_id": request["request_id"],
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "result_sha256": result_sha256,
        "remote_receipt_sha256": sha256_bytes(dispatch.receipt_bytes),
        "remote_claim_sha256": normalize_sha256(dispatch.claim_sha256, label="remote claim SHA-256"),
        "remote_root": dispatch.remote_root,
        "remote_environment_root": dispatch.remote_environment_root,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "recorded_at": receipt_payload.get("completed_at"),
    }
    ledger_path = control_dir / "private_grader_once_ledger.json"
    ensure_same_or_write(ledger_path, json_bytes(ledger), label="local private-grader ledger")
    ingress = (
        prepared.project_root
        / "workspace"
        / "siim_hpc_ingress"
        / prepared.run_id
        / "private_grader.json"
    )
    ensure_same_or_write(ingress, dispatch.result_bytes, label="SIIM private-grader ingress")
    return {
        "schema": "evomind.siim.private_grader_once_run.v1",
        "status": "passed",
        "run_id": prepared.run_id,
        "execution_id": request["request_id"],
        "execution_count": 1,
        "mle_private_grader_score": score,
        "result_path": str(result_path),
        "ledger_path": str(ledger_path),
        "ingress_path": str(ingress),
        "idempotent_reuse": dispatch.idempotent_reuse,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _validate_remote_request(
    request_dir: Path,
    *,
    runner_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = read_json(request_dir / "request.json", label="remote private-grader request")
    if request.get("schema") != REQUEST_SCHEMA:
        raise PrivateGraderOnceError("remote private-grader request schema is unsupported")
    run_id = safe_run_id(str(request.get("run_id") or ""))
    if request_dir.name != run_id:
        raise PrivateGraderOnceError("remote request directory does not match run_id")
    if request.get("competition_id") != COMPETITION_ID:
        raise PrivateGraderOnceError("remote private-grader competition mismatch")
    if int(request.get("job_id") or 0) != HPC_JOB_ID:
        raise PrivateGraderOnceError("remote private-grader job binding mismatch")
    if request.get("credential_profile") != CREDENTIAL_PROFILE:
        raise PrivateGraderOnceError("remote private-grader credential profile mismatch")
    if not exact_int(request.get("execution_index"), 1):
        raise PrivateGraderOnceError("remote private-grader execution index is not one")
    if request.get("private_label_scope") != "remote_terminal_grader_only":
        raise PrivateGraderOnceError("remote private-label scope is invalid")
    if request.get("grader_feedback_policy") != "terminal_delivery_only_no_tuning":
        raise PrivateGraderOnceError("remote grader feedback policy is invalid")
    if request.get("official_submission") != "forbidden" or request.get("kaggle_submission") != "forbidden":
        raise PrivateGraderOnceError("remote request does not forbid official submission")
    if request.get("signals_sent") != 0 or request.get("other_processes_modified") is not False:
        raise PrivateGraderOnceError("remote request process boundary is invalid")
    runner_sha256 = normalize_sha256(request.get("runner_sha256"), label="remote runner SHA-256")
    if sha256_file(runner_path) != runner_sha256:
        raise PrivateGraderOnceError("remote grader runner hash mismatch")
    core = dict(request)
    request_id = normalize_sha256(core.pop("request_id", None), label="remote request id")
    expected_core_keys = set(
        _request_core(
            run_id=run_id,
            candidate_freeze_sha256=normalize_sha256(
                request.get("candidate_freeze_sha256"), label="remote freeze SHA-256"
            ),
            review_sha256=normalize_sha256(
                request.get("review_sha256"), label="remote review SHA-256"
            ),
            submission_sha256=normalize_sha256(
                request.get("submission_sha256"), label="remote submission SHA-256"
            ),
            runner_sha256=runner_sha256,
        )
    )
    if set(core) != expected_core_keys:
        raise PrivateGraderOnceError("remote private-grader request has unsupported fields")
    if sha256_bytes(canonical_json_bytes(core)) != request_id:
        raise PrivateGraderOnceError("remote request id is invalid")

    freeze_path = request_dir / "candidate_freeze.json"
    review_path = request_dir / "review.json"
    submission_path = request_dir / "submission.csv"
    if sha256_file(freeze_path) != normalize_sha256(
        request.get("candidate_freeze_sha256"), label="remote freeze SHA-256"
    ):
        raise PrivateGraderOnceError("remote candidate freeze hash mismatch")
    if sha256_file(review_path) != normalize_sha256(
        request.get("review_sha256"), label="remote review SHA-256"
    ):
        raise PrivateGraderOnceError("remote review hash mismatch")
    if sha256_file(submission_path) != normalize_sha256(
        request.get("submission_sha256"), label="remote submission SHA-256"
    ):
        raise PrivateGraderOnceError("remote submission hash mismatch")
    freeze = read_json(freeze_path, label="remote candidate freeze")
    review = read_json(review_path, label="remote independent review")
    if freeze.get("schema") != CANDIDATE_FREEZE_SCHEMA or freeze.get("run_id") != run_id:
        raise PrivateGraderOnceError("remote candidate freeze contract failed")
    if freeze.get("status") != "frozen_before_private_grader" or freeze.get("tuning_closed") is not True:
        raise PrivateGraderOnceError("remote candidate is not frozen with tuning closed")
    if freeze.get("official_submission") != "forbidden":
        raise PrivateGraderOnceError("remote candidate freeze does not forbid official submission")
    if not exact_int(freeze.get("private_grader_execution_count_before_freeze"), 0):
        raise PrivateGraderOnceError("remote candidate was graded before freeze")
    submission_record = next(
        (
            item
            for item in freeze.get("artifacts") or []
            if isinstance(item, dict) and item.get("path") == "submission.csv"
        ),
        None,
    )
    if not isinstance(submission_record, dict) or normalize_sha256(
        submission_record.get("sha256"), label="remote frozen submission SHA-256"
    ) != request["submission_sha256"]:
        raise PrivateGraderOnceError("remote freeze does not bind the submission")
    _validate_review(
        review,
        run_id=run_id,
        frozen_hashes={
            str(item["path"]): normalize_sha256(
                item.get("sha256"), label=f"remote frozen hash for {item.get('path')}"
            )
            for item in freeze.get("artifacts") or []
            if isinstance(item, dict) and item.get("path")
        },
    )
    _validate_submission_csv(submission_path)
    return request, freeze, review


def _validate_remote_claim(claim: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    expected = {
        "schema": CLAIM_SCHEMA,
        "run_id": request["run_id"],
        "competition_id": COMPETITION_ID,
        "status": "claimed",
        "execution_id": request["request_id"],
        "execution_index": 1,
        "candidate_freeze_sha256": request["candidate_freeze_sha256"],
        "review_sha256": request["review_sha256"],
        "submission_sha256": request["submission_sha256"],
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if claim.get(key) != value:
            raise PrivateGraderOnceError(f"remote exactly-once claim conflict: {key}")
    if not exact_int(claim.get("execution_index"), 1):
        raise PrivateGraderOnceError("remote exactly-once claim index is not an exact integer")


def _remote_receipt_payload(
    *,
    request: Mapping[str, Any],
    claim_sha256: str,
    result_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "run_id": request["run_id"],
        "competition_id": COMPETITION_ID,
        "status": "passed",
        "execution_id": request["request_id"],
        "execution_count": 1,
        "candidate_freeze_sha256": request["candidate_freeze_sha256"],
        "review_sha256": request["review_sha256"],
        "submission_sha256": request["submission_sha256"],
        "claim_sha256": claim_sha256,
        "result_sha256": result_sha256,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "completed_at": utc_now(),
    }


def _reuse_remote_result(
    request_dir: Path,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    claim_path = request_dir / "execution_claim.json"
    result_path = request_dir / "result.json"
    receipt_path = request_dir / "execution_receipt.json"
    if not result_path.exists() and not receipt_path.exists():
        return None
    if not result_path.is_file() or not claim_path.is_file():
        raise PrivateGraderOnceError("remote exactly-once evidence is incomplete")
    claim = read_json(claim_path, label="remote execution claim")
    _validate_remote_claim(claim, request)
    result = read_json(result_path, label="remote private-grader result")
    prepared = _remote_prepared_candidate(request_dir, request)
    validate_result(result, prepared, expected_execution_id=str(request["request_id"]))
    result_sha256 = sha256_file(result_path)
    expected_receipt = _remote_receipt_payload(
        request=request,
        claim_sha256=sha256_file(claim_path),
        result_sha256=result_sha256,
    )
    if receipt_path.is_file():
        receipt = read_json(receipt_path, label="remote execution receipt")
        for key, value in expected_receipt.items():
            if key != "completed_at" and receipt.get(key) != value:
                raise PrivateGraderOnceError(f"remote execution receipt conflict: {key}")
    else:
        write_bytes_exclusive(receipt_path, json_bytes(expected_receipt))
    return {
        "status": "reused_existing_result",
        "run_id": request["run_id"],
        "execution_id": request["request_id"],
        "execution_count": 1,
        "result_sha256": result_sha256,
        "idempotent_reuse": True,
    }


def _remote_prepared_candidate(request_dir: Path, request: Mapping[str, Any]) -> PreparedCandidate:
    return PreparedCandidate(
        project_root=request_dir,
        run_id=str(request["run_id"]),
        run_dir=request_dir,
        freeze_path=request_dir / "candidate_freeze.json",
        review_path=request_dir / "review.json",
        submission_path=request_dir / "submission.csv",
        candidate_freeze_sha256=str(request["candidate_freeze_sha256"]),
        review_sha256=str(request["review_sha256"]),
        submission_sha256=str(request["submission_sha256"]),
        runner_sha256=str(request["runner_sha256"]),
        request=dict(request),
    )


def _read_remote_siim_ids(path: Path, *, label: str) -> list[str]:
    require_regular_non_symlink_file(path, label=label)
    identifiers: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["image_name", "target"]:
                raise PrivateGraderOnceError(
                    f"{label} columns are not image_name,target"
                )
            for row in reader:
                if None in row:
                    raise PrivateGraderOnceError(f"{label} contains a malformed CSV row")
                image_name = str(row.get("image_name") or "").strip()
                if not image_name:
                    raise PrivateGraderOnceError(f"{label} contains a missing image_name")
                try:
                    target = float(row.get("target", ""))
                except (TypeError, ValueError) as exc:
                    raise PrivateGraderOnceError(
                        f"{label} contains a nonnumeric target"
                    ) from exc
                if not math.isfinite(target):
                    raise PrivateGraderOnceError(f"{label} contains a nonfinite target")
                identifiers.append(image_name)
    except PrivateGraderOnceError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise PrivateGraderOnceError(f"{label} is not a readable CSV") from exc
    if len(identifiers) != EXPECTED_TEST_ROWS:
        raise PrivateGraderOnceError(f"{label} row count is not 4,142")
    if len(set(identifiers)) != len(identifiers):
        raise PrivateGraderOnceError(f"{label} contains duplicate image_name values")
    return identifiers


def _load_official_remote_grader(official_source_root: Path) -> RemoteGrader:
    source = str(official_source_root)
    inserted = source not in sys.path
    if inserted:
        sys.path.insert(0, source)
    importlib.invalidate_caches()
    try:
        from research_os.mlebench_phase_a import grade_private_submission

        grade_module = importlib.import_module("mlebench.grade")
        registry_module = importlib.import_module("mlebench.registry")
        for module, label in (
            (grade_module, "official mlebench.grade module"),
            (registry_module, "official mlebench.registry module"),
        ):
            module_file_value = str(getattr(module, "__file__", "") or "").strip()
            if not module_file_value:
                raise PrivateGraderOnceError(f"{label} has no source file")
            module_file = Path(module_file_value)
            require_remote_within(
                official_source_root,
                module_file,
                label=label,
            )
        if not callable(getattr(grade_module, "grade_csv", None)):
            raise PrivateGraderOnceError("official mlebench.grade grade_csv is unavailable")
        if not callable(getattr(registry_module, "Registry", None)):
            raise PrivateGraderOnceError("official mlebench.registry Registry is unavailable")
        if not callable(grade_private_submission):
            raise PrivateGraderOnceError("EvoMind private grader entry point is unavailable")
        return grade_private_submission
    except PrivateGraderOnceError:
        raise
    except Exception as exc:
        raise PrivateGraderOnceError(
            "official MLE-Bench grader modules or dependencies are not importable"
        ) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(source)
            except ValueError:
                pass


def _prepare_remote_grader_readiness(
    request_dir: Path,
    *,
    data_root: Path,
    official_source_root: Path,
    allowed_root: Path,
    bundle_root: Path,
    runner_path: Path,
    grader: RemoteGrader | None,
) -> tuple[RemoteGrader, Path, str]:
    """Complete every non-consuming check before claiming execution."""

    submission_path = request_dir / "submission.csv"
    answers_path = (
        data_root / COMPETITION_ID / "prepared" / "private" / "test.csv"
    )
    bundle_grader_path = bundle_root / "src" / "research_os" / "mlebench_phase_a.py"
    official_grade_path = official_source_root / "mlebench" / "grade.py"
    official_registry_path = official_source_root / "mlebench" / "registry.py"

    require_regular_non_symlink_file(runner_path, label="grader runner")
    require_regular_non_symlink_file(submission_path, label="grader submission input")
    require_regular_non_symlink_file(answers_path, label="private answers input")
    require_regular_non_symlink_file(bundle_grader_path, label="bundle grader module")
    require_regular_non_symlink_file(
        official_grade_path, label="official mlebench.grade source"
    )
    require_regular_non_symlink_file(
        official_registry_path, label="official mlebench.registry source"
    )
    require_remote_within(data_root, answers_path, label="private answers input")
    require_remote_within(bundle_root, bundle_grader_path, label="bundle grader module")
    require_remote_within(
        official_source_root,
        official_grade_path,
        label="official mlebench.grade source",
    )
    require_remote_within(
        official_source_root,
        official_registry_path,
        label="official mlebench.registry source",
    )
    require_remote_within(allowed_root, runner_path, label="grader runner")
    require_remote_within(allowed_root, submission_path, label="grader submission input")

    submission_ids = _read_remote_siim_ids(
        submission_path, label="grader submission input"
    )
    answer_ids = _read_remote_siim_ids(answers_path, label="private answers input")
    if submission_ids != answer_ids:
        raise PrivateGraderOnceError(
            "private answers and grader submission IDs/order do not align"
        )

    if grader is None:
        grader = _load_official_remote_grader(official_source_root)
    elif not callable(grader):
        raise PrivateGraderOnceError("injected private grader is not callable")
    return grader, answers_path.resolve(), sha256_file(answers_path)


def remote_grade_once(
    request_dir: Path,
    *,
    data_root: Path,
    official_source_root: Path,
    allowed_root: Path,
    bundle_root: Path,
    runner_path: Path | None = None,
    grader: RemoteGrader | None = None,
) -> dict[str, Any]:
    """Remote-only exactly-once grader worker; private answers stay on this host."""

    allowed_root = allowed_root.resolve()
    request_dir = require_remote_within(allowed_root, request_dir, label="request directory")
    data_root = require_remote_within(allowed_root, data_root, label="private data root")
    official_source_root = require_remote_within(
        allowed_root, official_source_root, label="official source root"
    )
    bundle_root = require_remote_within(allowed_root, bundle_root, label="bundle root")
    runner_path = require_remote_within(
        allowed_root,
        (runner_path or Path(__file__)),
        label="grader runner",
    )
    expected_request_parent = (
        allowed_root / REMOTE_PRIVATE_GRADER_PARENT_RELATIVE
    ).resolve()
    expected_data_root = (allowed_root / REMOTE_DATA_ROOT_RELATIVE).resolve()
    expected_official_source = (allowed_root / REMOTE_OFFICIAL_SOURCE_RELATIVE).resolve()
    expected_bundle_parent = (allowed_root / f"siim_{JOB_TAG}" / "bundles").resolve()
    if request_dir.parent != expected_request_parent:
        raise PrivateGraderOnceError("request directory is outside the SIIM private-grader root")
    if data_root != expected_data_root:
        raise PrivateGraderOnceError("private grader data root is not the pinned MLE-Bench data root")
    if official_source_root != expected_official_source:
        raise PrivateGraderOnceError("private grader source root is not the pinned MLE-Bench source")
    if bundle_root.parent != expected_bundle_parent:
        raise PrivateGraderOnceError("private grader bundle is outside the content-addressed bundle root")
    normalize_sha256(bundle_root.name, label="content-addressed bundle id")
    if runner_path != request_dir / "runner.py":
        raise PrivateGraderOnceError("private grader runner is not the immutable request runner")
    request, _freeze, _review = _validate_remote_request(request_dir, runner_path=runner_path)
    reused = _reuse_remote_result(request_dir, request)
    if reused is not None:
        return reused

    claim_path = request_dir / "execution_claim.json"
    result_path = request_dir / "result.json"
    receipt_path = request_dir / "execution_receipt.json"
    lock_path = request_dir / ".grader_lock"
    if claim_path.exists() or receipt_path.exists() or lock_path.exists():
        if claim_path.is_file():
            _validate_remote_claim(read_json(claim_path, label="remote execution claim"), request)
        raise PrivateGraderOnceError(
            "remote private-grader execution is incomplete; exactly-once policy blocks retry"
        )
    grader, expected_answers, expected_answers_sha256 = _prepare_remote_grader_readiness(
        request_dir,
        data_root=data_root,
        official_source_root=official_source_root,
        allowed_root=allowed_root,
        bundle_root=bundle_root,
        runner_path=runner_path,
        grader=grader,
    )
    try:
        lock_path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise PrivateGraderOnceError("remote private-grader execution is already in progress") from exc

    claim = {
        "schema": CLAIM_SCHEMA,
        "run_id": request["run_id"],
        "competition_id": COMPETITION_ID,
        "status": "claimed",
        "execution_id": request["request_id"],
        "execution_index": 1,
        "candidate_freeze_sha256": request["candidate_freeze_sha256"],
        "review_sha256": request["review_sha256"],
        "submission_sha256": request["submission_sha256"],
        "private_label_scope": "remote_terminal_grader_only",
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "claimed_at": utc_now(),
    }
    write_bytes_exclusive(claim_path, json_bytes(claim))

    try:
        raw_path = request_dir / REMOTE_RAW_GRADER_NAME
        raw = grader(
            request_dir / "submission.csv",
            COMPETITION_ID,
            data_root,
            official_source_root=official_source_root,
            seed=None,
            budget={"stage": "terminal_private_grader", "execution_count": 1},
            code_paths=[runner_path, bundle_root / "src" / "research_os" / "mlebench_phase_a.py"],
            output_path=raw_path,
        )
        if not raw_path.is_file():
            write_bytes_exclusive(raw_path, json_bytes(raw))
        if raw.get("status") != "passed" or raw.get("official_mlebench_grader_executed") is not True:
            raise PrivateGraderOnceError("official MLE-Bench private grader did not pass")
        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            raise PrivateGraderOnceError("official private-grader provenance is missing")
        if normalize_sha256(
            provenance.get("submission_sha256"), label="grader submission SHA-256"
        ) != request["submission_sha256"]:
            raise PrivateGraderOnceError("official private grader used a different submission")
        private_answers_sha256 = normalize_sha256(
            provenance.get("answers_sha256"), label="private answers SHA-256"
        )
        if private_answers_sha256 != expected_answers_sha256:
            raise PrivateGraderOnceError(
                "official private grader used changed private answers"
            )
        actual_answers = Path(str(provenance.get("answers_path") or "")).resolve()
        if actual_answers != expected_answers:
            raise PrivateGraderOnceError("official private grader used an unexpected answers file")
        try:
            score = float(raw.get("score"))
        except (TypeError, ValueError) as exc:
            raise PrivateGraderOnceError("official private-grader score is not numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise PrivateGraderOnceError("official private-grader score is outside [0, 1]")
        result = {
            "schema": RESULT_SCHEMA,
            "run_id": request["run_id"],
            "competition_id": COMPETITION_ID,
            "status": "passed",
            "execution_id": request["request_id"],
            "execution_index": 1,
            "execution_count": 1,
            "candidate_freeze_sha256": request["candidate_freeze_sha256"],
            "review_sha256": request["review_sha256"],
            "submission_sha256": request["submission_sha256"],
            "executed_after_freeze": True,
            "official_mlebench_grader_executed": True,
            "metric": "roc_auc",
            "direction": "maximize",
            "score": score,
            "mle_private_grader_score": score,
            "private_answers_sha256": private_answers_sha256,
            "raw_grader_sha256": sha256_file(raw_path),
            "mlebench_revision": provenance.get("mlebench_revision"),
            "mlebench_version": provenance.get("mlebench_version"),
            "private_label_scope": "remote_terminal_grader_only",
            "private_labels_exported": False,
            "raw_grader_evidence_location": "remote_only",
            "feedback_used_for_tuning": False,
            "post_grader_tuning": "forbidden",
            "official_submission_executed": False,
            "kaggle_submission_executed": False,
            "training_artifacts_modified": False,
            "signals_sent": 0,
            "other_processes_modified": False,
            "completed_at": utc_now(),
        }
        result_bytes = json_bytes(result)
        write_bytes_exclusive(result_path, result_bytes)
        receipt = _remote_receipt_payload(
            request=request,
            claim_sha256=sha256_file(claim_path),
            result_sha256=sha256_bytes(result_bytes),
        )
        write_bytes_exclusive(receipt_path, json_bytes(receipt))
        shutil.rmtree(lock_path)
    except Exception as exc:
        failure_path = request_dir / "execution_failure.json"
        if not failure_path.exists():
            failure = {
                "schema": "evomind.siim.private_grader_failure.v1",
                "run_id": request["run_id"],
                "status": "failed_closed",
                "execution_id": request["request_id"],
                "execution_count_claimed": 1,
                "error_type": type(exc).__name__,
                "feedback_used_for_tuning": False,
                "official_submission_executed": False,
                "kaggle_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
                "failed_at": utc_now(),
            }
            write_bytes_exclusive(failure_path, json_bytes(failure))
        raise
    return {
        "status": "passed",
        "run_id": request["run_id"],
        "execution_id": request["request_id"],
        "execution_count": 1,
        "result_sha256": sha256_file(result_path),
        "idempotent_reuse": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--remote-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--official-source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--allowed-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bundle-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.remote_worker:
        for name in ("request_dir", "data_root", "official_source_root", "allowed_root", "bundle_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required with --remote-worker")
    elif not args.run_id:
        parser.error("--run-id is required")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.remote_worker:
            if os.environ.get("EVOMIND_SIIM_RUN_ID", "").strip() != args.request_dir.name:
                raise PrivateGraderOnceError("remote worker Run environment binding is missing or changed")
            validate_dedicated_process_environment(args.allowed_root)
            payload = remote_grade_once(
                args.request_dir,
                data_root=args.data_root,
                official_source_root=args.official_source_root,
                allowed_root=args.allowed_root,
                bundle_root=args.bundle_root,
            )
        else:
            payload = run_private_grader_once(args.project_root, args.run_id)
    except PrivateGraderOnceError as exc:
        print(
            json.dumps(
                {
                    "schema": "evomind.siim.private_grader_once_error.v1",
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "official_submission_executed": False,
                    "kaggle_submission_executed": False,
                    "signals_sent": 0,
                    "other_processes_modified": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
