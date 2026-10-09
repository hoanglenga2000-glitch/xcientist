"""Create one local, fail-closed G21 human-baseline study scaffold.

The command writes only an incomplete template package.  It cannot contact
production, HPC, a GPU, organizers, or participants, and it never creates
scores.  The generated manifest deliberately carries ``synthetic_fixture=true``
and ``status=INCOMPLETE_TEMPLATE`` so the production-compatible local validator
must reject it until every placeholder is replaced with real, source-bound
evidence and the package is resealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_g21_human_baseline_study as validator


SCAFFOLD_SCHEMA = "evomind.g21-human-baseline-study-scaffold.v1"
SCAFFOLD_RESULT_SCHEMA = "evomind.g21-human-baseline-study-scaffold-result.v1"
INCOMPLETE_STATUS = "INCOMPLETE_TEMPLATE"
SAFE_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
SAFE_METRIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$")

ROLE_PATHS = {
    "protocol": "protocol.json",
    "task_manifest": "task-manifest.json",
    "scorer": "scorer.json",
    "human_baseline_source": "human-baseline-source.placeholder.txt",
    "holdout_ledger": "holdout-ledger.json",
    "holdout_claim": "holdout-claim.json",
    "participant_metrics": "participant-metrics.json",
    "human_baseline": "human-baseline.json",
}

NO_SIDE_EFFECTS = {
    "production_writes": 0,
    "hpc_accessed": False,
    "gpu_touched": False,
    "remote_writes": 0,
    "signals_sent": 0,
    "training_tool_calls": 0,
}


class ScaffoldError(ValueError):
    """A deterministic, non-secret scaffold creation error."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ScaffoldError(code, detail)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _is_production_path(path: Path) -> bool:
    normalized = "/" + str(path).replace("\\", "/").casefold().strip("/") + "/"
    return "/programdata/evomind/" in normalized


def _resolve_new_output_directory(output_dir: Path) -> Path:
    raw = os.fspath(output_dir)
    _require(bool(raw) and "\x00" not in raw, "OUTPUT_PATH_UNSAFE")
    _require(not raw.startswith(("\\\\", "//")), "NONLOCAL_PATH_FORBIDDEN")

    supplied = output_dir.expanduser()
    _require(all(part not in {".", ".."} for part in supplied.parts), "OUTPUT_PATH_UNSAFE")
    _require(SAFE_OUTPUT_NAME_RE.fullmatch(supplied.name or "") is not None, "OUTPUT_NAME_UNSAFE")
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied

    parent_input = supplied.parent.absolute()
    probe = parent_input
    while True:
        if os.path.lexists(probe):
            _require(not probe.is_symlink(), "OUTPUT_ANCESTOR_SYMLINK_FORBIDDEN")
        if probe.parent == probe:
            break
        probe = probe.parent

    try:
        parent = parent_input.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ScaffoldError("OUTPUT_PARENT_INVALID", type(exc).__name__) from exc
    _require(parent.is_dir() and not parent.is_symlink(), "OUTPUT_PARENT_INVALID")

    target = parent / supplied.name
    _require(not _is_production_path(target), "PRODUCTION_PATH_FORBIDDEN")
    _require(not os.path.lexists(target), "OUTPUT_ALREADY_EXISTS")
    return target


def _identity(schema: str, *, competition: str, study_id: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "status": INCOMPLETE_STATUS,
        "training_authorized": False,
        "run_id": validator.competition_goal.FIXED_RUN_ID,
        "allocation": validator.competition_goal.FIXED_ALLOCATION,
        "competition": competition,
        "study_id": study_id,
    }


def _artifact_id(study_id: str, role: str) -> str:
    suffix = hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:12]
    return f"template_{role}_{suffix}"


def _manifest_entry(root: Path, *, role: str, relative_path: str, study_id: str) -> dict[str, Any]:
    target = root / relative_path
    payload = target.read_bytes()
    return {
        "path": relative_path,
        "role": role,
        "artifact_id": _artifact_id(study_id, role),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "regular_file": True,
        "symlink": False,
    }


def _template_payloads(
    root: Path,
    *,
    competition: str,
    study_id: str,
    metric: str,
    direction: str,
) -> dict[str, bytes]:
    scorer = {
        **_identity(validator.SCORER_SCHEMA, competition=competition, study_id=study_id),
        "scorer_id": "replace_with_frozen_official_scorer_id",
        "metric": metric,
        "direction": direction,
        "primary_aggregation_unit": "human_participant",
        "participant_score_field": "score",
        "aggregation": "arithmetic_mean",
        "placeholder_fields": ["scorer_id", "official_scorer_bytes", "official_scorer_sha256"],
    }
    scorer_bytes = _json_bytes(scorer)
    _write_exclusive(root / ROLE_PATHS["scorer"], scorer_bytes)

    task = {
        **_identity(validator.TASK_SCHEMA, competition=competition, study_id=study_id),
        "task_manifest_id": "replace_with_frozen_blind_task_manifest_id",
        "metric": metric,
        "direction": direction,
        "scorer_sha256": _sha256_bytes(scorer_bytes),
        "participant_blind_to_labels": True,
        "participant_blind_to_candidate_outputs": True,
        "test_labels_exposed": False,
        "placeholder_fields": ["task_manifest_id", "frozen_task_content", "holdout_identity"],
    }
    task_bytes = _json_bytes(task)
    _write_exclusive(root / ROLE_PATHS["task_manifest"], task_bytes)

    ledger = {
        **_identity(validator.HOLDOUT_LEDGER_SCHEMA, competition=competition, study_id=study_id),
        "ledger_id": "replace_with_read_only_holdout_ledger_id",
        "task_manifest_sha256": _sha256_bytes(task_bytes),
        "placeholder_fields": ["ledger_id", "immutable_ledger_receipt", "prior_consumption_history"],
    }
    ledger_bytes = _json_bytes(ledger)
    _write_exclusive(root / ROLE_PATHS["holdout_ledger"], ledger_bytes)

    claim = {
        **_identity(validator.HOLDOUT_CLAIM_SCHEMA, competition=competition, study_id=study_id),
        "claim_id": "replace_with_atomic_holdout_claim_id",
        "ledger_id": ledger["ledger_id"],
        "task_manifest_sha256": _sha256_bytes(task_bytes),
        "ledger_sha256": _sha256_bytes(ledger_bytes),
        "unconsumed_at_claim": None,
        "consumed_once": False,
        "selection_data_scope": None,
        "prior_holdout_overlap": None,
        "test_labels_exposed": None,
        "placeholder_fields": [
            "claim_id",
            "ledger_id",
            "unconsumed_at_claim",
            "consumed_once",
            "selection_data_scope",
            "prior_holdout_overlap",
            "test_labels_exposed",
        ],
    }
    claim_bytes = _json_bytes(claim)
    _write_exclusive(root / ROLE_PATHS["holdout_claim"], claim_bytes)

    source_bytes = (
        "INCOMPLETE_TEMPLATE\n"
        "No human-baseline source evidence is present.\n"
        "Replace this entire file with exact authoritative, non-secret source bytes.\n"
        "This placeholder is not evidence and authorizes no training.\n"
    ).encode("utf-8")
    _write_exclusive(root / ROLE_PATHS["human_baseline_source"], source_bytes)

    protocol_text = (
        "INCOMPLETE_TEMPLATE: replace with the frozen, source-bound, protocol-comparable human study protocol."
    )
    protocol = {
        **_identity(validator.PROTOCOL_SCHEMA, competition=competition, study_id=study_id),
        "protocol_id": "replace_with_frozen_protocol_id",
        "protocol_text": protocol_text,
        "participant_type": "human",
        "participant_id_scheme": "sha256_salted_pseudonym_v1",
        "metric": metric,
        "direction": direction,
        "task_manifest_sha256": _sha256_bytes(task_bytes),
        "scorer_sha256": _sha256_bytes(scorer_bytes),
        "source_sha256": _sha256_bytes(source_bytes),
        "holdout_ledger_sha256": _sha256_bytes(ledger_bytes),
        "holdout_claim_sha256": _sha256_bytes(claim_bytes),
        "bootstrap": {
            "method": validator.BOOTSTRAP_METHOD,
            "seed": validator.BOOTSTRAP_SEED,
            "rounds": validator.BOOTSTRAP_ROUNDS,
            "confidence_level": 0.95,
            "resampling_unit": "human_participant",
        },
        "placeholder_fields": ["protocol_id", "protocol_text", "source_sha256"],
    }
    protocol_bytes = _json_bytes(protocol)
    _write_exclusive(root / ROLE_PATHS["protocol"], protocol_bytes)

    participant_metrics = {
        **_identity(validator.PARTICIPANT_METRICS_SCHEMA, competition=competition, study_id=study_id),
        "metric": metric,
        "direction": direction,
        "participant_count": 0,
        "records": [],
        "contains_synthetic_scores": False,
        "placeholder_fields": ["records", "participant_count"],
    }
    participant_metrics_bytes = _json_bytes(participant_metrics)
    _write_exclusive(root / ROLE_PATHS["participant_metrics"], participant_metrics_bytes)

    source_sha256 = _sha256_bytes(source_bytes)
    human_baseline = {
        **_identity("evomind.human-baseline.v1", competition=competition, study_id=study_id),
        "participant_type": "human",
        "source_url": None,
        "source_authority": None,
        "source_sha256": source_sha256,
        "evidence_artifact_id": _artifact_id(study_id, "human_baseline_source"),
        "evidence_artifact_sha256": source_sha256,
        "title": "INCOMPLETE_TEMPLATE",
        "sample_size": 0,
        "protocol_id": protocol["protocol_id"],
        "protocol": protocol_text,
        "protocol_sha256": hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
        "protocol_comparable": False,
        "metric": metric,
        "direction": direction,
        "mean": None,
        "uncertainty": {
            "lower": None,
            "upper": None,
            "confidence_level": 0.95,
            "method": validator.BOOTSTRAP_METHOD,
            "source_sha256": source_sha256,
        },
        "protocol_file_sha256": _sha256_bytes(protocol_bytes),
        "task_manifest_sha256": _sha256_bytes(task_bytes),
        "scorer_sha256": _sha256_bytes(scorer_bytes),
        "holdout_ledger_sha256": _sha256_bytes(ledger_bytes),
        "holdout_claim_sha256": _sha256_bytes(claim_bytes),
        "placeholder_fields": [
            "source_url_or_doi",
            "source_authority",
            "title",
            "sample_size",
            "protocol",
            "protocol_comparable",
            "mean",
            "uncertainty.lower",
            "uncertainty.upper",
        ],
    }
    human_baseline_bytes = _json_bytes(human_baseline)
    _write_exclusive(root / ROLE_PATHS["human_baseline"], human_baseline_bytes)

    return {
        "scorer": scorer_bytes,
        "task_manifest": task_bytes,
        "holdout_ledger": ledger_bytes,
        "holdout_claim": claim_bytes,
        "human_baseline_source": source_bytes,
        "protocol": protocol_bytes,
        "participant_metrics": participant_metrics_bytes,
        "human_baseline": human_baseline_bytes,
    }


def create_scaffold(
    output_dir: Path,
    *,
    competition: str,
    study_id: str,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    """Create a new closed template directory without overwriting any path."""

    _require(competition in validator.competition_goal.COMPETITIONS, "COMPETITION_INVALID")
    _require(validator.SAFE_ID_RE.fullmatch(study_id) is not None, "STUDY_ID_INVALID")
    _require(SAFE_METRIC_RE.fullmatch(metric) is not None, "METRIC_INVALID")
    _require(direction in validator.competition_goal.VALID_DIRECTIONS, "DIRECTION_INVALID")
    target = _resolve_new_output_directory(output_dir)

    created = False
    try:
        target.mkdir()
        created = True
        _template_payloads(
            target,
            competition=competition,
            study_id=study_id,
            metric=metric,
            direction=direction,
        )
        files = [
            _manifest_entry(target, role=role, relative_path=relative_path, study_id=study_id)
            for role, relative_path in ROLE_PATHS.items()
        ]
        manifest = {
            "schema": validator.MANIFEST_SCHEMA,
            "scaffold_schema": SCAFFOLD_SCHEMA,
            "status": INCOMPLETE_STATUS,
            "training_authorized": False,
            "synthetic_fixture": True,
            "synthetic_scores_present": False,
            "manifest_self_excluded": True,
            "run_id": validator.competition_goal.FIXED_RUN_ID,
            "allocation": validator.competition_goal.FIXED_ALLOCATION,
            "competition": competition,
            "study_id": study_id,
            "file_count": len(files),
            "total_bytes": sum(int(item["bytes"]) for item in files),
            "files": files,
            "completion_requirements": [
                "replace every placeholder with real source-bound evidence",
                f"include at least {validator.MIN_PARTICIPANTS} unique human participants",
                "recompute participant metrics and bootstrap confidence interval",
                "remove the synthetic fixture marker only after evidence replacement",
                "reseal every bytes and sha256 manifest entry",
                "run validate_g21_human_baseline_study.py; its receipt still does not authorize training",
            ],
            "side_effects": dict(NO_SIDE_EFFECTS),
        }
        manifest_bytes = _json_bytes(manifest)
        _write_exclusive(target / validator.MANIFEST_NAME, manifest_bytes)
    except BaseException:
        if created and target.exists():
            shutil.rmtree(target)
        raise

    return {
        "schema": SCAFFOLD_RESULT_SCHEMA,
        "status": INCOMPLETE_STATUS,
        "training_authorized": False,
        "output_dir": str(target),
        "run_id": validator.competition_goal.FIXED_RUN_ID,
        "allocation": validator.competition_goal.FIXED_ALLOCATION,
        "competition": competition,
        "study_id": study_id,
        "file_count": len(ROLE_PATHS),
        "package_manifest_sha256": _sha256_bytes(manifest_bytes),
        "synthetic_scores_present": False,
        "side_effects": dict(NO_SIDE_EFFECTS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--competition", choices=validator.competition_goal.COMPETITIONS, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--direction", choices=sorted(validator.competition_goal.VALID_DIRECTIONS), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = create_scaffold(
            args.output_dir,
            competition=args.competition,
            study_id=args.study_id,
            metric=args.metric,
            direction=args.direction,
        )
    except (OSError, ScaffoldError) as exc:
        if isinstance(exc, ScaffoldError):
            error = exc.to_dict()
        else:
            error = {"code": "LOCAL_IO_ERROR", "detail": type(exc).__name__}
        print(json.dumps({"status": "REJECTED", "error": error}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
