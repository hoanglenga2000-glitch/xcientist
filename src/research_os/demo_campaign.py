"""Deterministic, evidence-producing EvoMind product-demo campaign.

The demo is deliberately isolated from SIIM and MLE-Bench identities.  Every
candidate is still executed in the normal local subprocess runner against a
real synthetic dataset; the deterministic proposal plan exists only to make a
short screen recording reproducible and to guarantee that Draft, Improve,
Debug and Crossover are all exercised.  No score is precomputed or injected.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from .experience_mcgs import CANONICAL_HASH_SCHEMA, canonical_json as _experience_canonical_json
from .variation_generator import TaskContext, VariationProposal

DEMO_SCHEMA_VERSION = 1
DEMO_TASK_ID = "evomind_demo_customer_churn"
DEMO_PLAN_ID = "evomind.demo.verified_sklearn_plan.v1"
DEMO_ITERATIONS = 8
DEMO_MAX_NODES = DEMO_ITERATIONS
DEMO_RUNNER = "local"
DEMO_SEARCH_MODE = "experience_mcgs_v1"
DEMO_MAX_TOKENS = 100_000
DEMO_MAX_WALL_SECONDS = 600.0
DEMO_MAX_COST_USD = 0.01
DEMO_OPERATORS = ("Draft", "Improve", "Debug", "Crossover")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _experience_json_sha256(value: Any) -> str:
    return hashlib.sha256(_experience_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist one evidence link so partial writes never look valid."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_hashed_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or is not a regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _safe_frozen_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("candidate freeze contains an invalid artifact path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("candidate freeze artifact path escapes the run directory")
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate freeze artifact path escapes the run directory") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"candidate freeze artifact is missing: {relative}")
    return resolved


def _validate_candidate_freeze(
    root: Path,
    *,
    summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    freeze, freeze_sha = _read_hashed_json(root / "candidate-freeze.json", label="candidate freeze")
    if freeze.get("schema") != "evomind.demo.candidate_freeze.v1":
        raise ValueError("candidate freeze schema mismatch")
    if freeze.get("task_id") != DEMO_TASK_ID or freeze.get("run_id") != root.name:
        raise ValueError("candidate freeze identity mismatch")
    if freeze.get("status") != "frozen_before_independent_review":
        raise ValueError("candidate freeze status mismatch")
    if freeze.get("official_submission_performed") is not False or freeze.get("private_grader_used") is not False:
        raise ValueError("candidate freeze crossed the demo evidence boundary")

    best_exp_id = freeze.get("best_exp_id")
    if not isinstance(best_exp_id, str) or not best_exp_id:
        raise ValueError("candidate freeze has no best experiment")
    expected_paths = {
        f"{best_exp_id}/out/submission.csv",
        f"{best_exp_id}/out/metrics.json",
        f"{best_exp_id}/out/model.joblib",
    }
    records = freeze.get("files")
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ValueError("candidate freeze artifact set is incomplete")
    observed_paths: set[str] = set()
    observed_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("candidate freeze contains an invalid artifact record")
        relative = record.get("path")
        path = _safe_frozen_path(root, relative)
        relative_text = str(relative)
        if relative_text in observed_paths:
            raise ValueError("candidate freeze contains duplicate artifact paths")
        observed_paths.add(relative_text)
        actual_sha = _sha256(path)
        if record.get("sha256") != actual_sha or record.get("bytes") != path.stat().st_size:
            raise ValueError(f"candidate freeze artifact hash mismatch: {relative_text}")
        observed_hashes[relative_text] = actual_sha
    if observed_paths != expected_paths:
        raise ValueError("candidate freeze artifact set does not match the best experiment")

    if summary is not None:
        if freeze.get("best_exp_id") != summary.get("best_exp_id"):
            raise ValueError("candidate freeze best experiment differs from the execution summary")
        if freeze.get("public_validation_score") != summary.get("best_cv_score"):
            raise ValueError("candidate freeze score differs from the execution summary")
    return freeze, freeze_sha, observed_hashes


def _validate_independent_review(
    root: Path,
    campaign_root: Path,
    *,
    freeze: dict[str, Any],
    freeze_sha: str,
    frozen_hashes: dict[str, str],
) -> tuple[dict[str, Any], str]:
    review, review_sha = _read_hashed_json(root / "independent-review.json", label="independent review")
    if review.get("schema") != "evomind.demo.independent_review.v1":
        raise ValueError("independent review schema mismatch")
    if review.get("status") != "passed" or review.get("reviewer") != "IndependentDemoReviewer":
        raise ValueError("independent review did not pass")
    if review.get("candidate_freeze_sha256") != freeze_sha:
        raise ValueError("independent review is not bound to the current candidate freeze")
    submission_relative = f"{freeze['best_exp_id']}/out/submission.csv"
    if review.get("submission_sha256") != frozen_hashes.get(submission_relative):
        raise ValueError("independent review submission hash differs from the candidate freeze")
    labels_path = campaign_root / "review" / "holdout_labels.csv"
    if not labels_path.is_file() or labels_path.is_symlink():
        raise ValueError("sealed independent-review labels are missing")
    if review.get("sealed_labels_sha256") != _sha256(labels_path):
        raise ValueError("independent review sealed-label hash mismatch")
    if review.get("rows") != len(pd.read_csv(labels_path)):
        raise ValueError("independent review row count differs from the sealed labels")
    score = review.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("independent review score is not finite")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("independent review score is outside the metric range")
    if review.get("model_reload_passed") is not True or review.get("source") != "sealed_synthetic_holdout":
        raise ValueError("independent review smoke/provenance checks failed")
    if review.get("official_submission_performed") is not False:
        raise ValueError("independent review crossed the demo evidence boundary")
    return review, review_sha


def _summary_execution_evidence(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    raw_iterations = summary.get("iterations")
    iterations = raw_iterations if isinstance(raw_iterations, list) else []
    records = [item for item in iterations if isinstance(item, dict)]
    exp_ids = [str(item.get("exp_id")) for item in records if isinstance(item.get("exp_id"), str)]
    unique_exp_ids = list(dict.fromkeys(exp_ids))
    executed_exp_ids = [
        exp_id for exp_id in unique_exp_ids if (root / "runs" / exp_id / "solution.py").is_file()
    ]
    declared = summary.get("n_iterations")
    summary_consistent = (
        summary.get("task") == DEMO_TASK_ID
        and isinstance(declared, int)
        and not isinstance(declared, bool)
        and declared == len(records) == len(exp_ids) == len(unique_exp_ids)
    )
    budget = summary.get("budget") if isinstance(summary.get("budget"), dict) else {}
    budget_consistent = (
        budget.get("nodes") == DEMO_MAX_NODES
        and budget.get("max_nodes") == DEMO_MAX_NODES
        and budget.get("max_total_tokens") == DEMO_MAX_TOKENS
        and budget.get("max_wall_seconds") == DEMO_MAX_WALL_SECONDS
        and budget.get("max_cost_usd") == DEMO_MAX_COST_USD
        and budget.get("terminal_reason") == "node_budget_exhausted"
        and budget.get("can_continue") is False
        and summary.get("terminal_reason") == "node_budget_exhausted"
    )
    complete = (
        summary_consistent
        and len(executed_exp_ids) == DEMO_ITERATIONS
        and set(executed_exp_ids) == set(unique_exp_ids)
        and budget_consistent
    )
    return {
        "declared_n_iterations": declared,
        "recorded_iteration_count": len(records),
        "executed_candidate_count": len(executed_exp_ids),
        "executed_exp_ids": executed_exp_ids,
        "summary_consistent": summary_consistent,
        "budget_consistent": budget_consistent,
        "complete_fixed_budget": complete,
    }


def _search_evidence(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    board_path = root / "experience-board.json"
    if not board_path.is_file() or board_path.is_symlink():
        return {
            "integrity": "failed",
            "error": "experience board is missing",
            "operators": [],
            "crossover_lineages": [],
        }
    try:
        board, board_file_sha = _read_hashed_json(board_path, label="experience board")
        if board.get("schema") != "evomind.experience_mcgs.v1" or board.get("task_id") != DEMO_TASK_ID:
            raise ValueError("experience board identity/schema mismatch")
        if board.get("hash_canonicalization") != CANONICAL_HASH_SCHEMA:
            raise ValueError("experience board canonical hash schema mismatch")
        cards = board.get("cards")
        append_order = board.get("append_order")
        if not isinstance(cards, list) or not isinstance(append_order, list) or len(cards) != len(append_order):
            raise ValueError("experience board card/append order mismatch")

        chain_head = "0" * 64
        card_ids: list[str] = []
        for card in cards:
            if not isinstance(card, dict) or not isinstance(card.get("card_id"), str):
                raise ValueError("experience board contains an invalid card")
            card_without_id = dict(card)
            card_id = str(card_without_id.pop("card_id"))
            card_hash = _experience_json_sha256(card_without_id)
            if card_id != f"exp_{card_hash[:24]}":
                raise ValueError("experience board card content hash mismatch")
            chain_head = _experience_json_sha256({
                "previous": chain_head,
                "card_id": card_id,
                "card_hash": card_hash,
            })
            card_ids.append(card_id)
        if card_ids != append_order or board.get("append_chain_head") != chain_head:
            raise ValueError("experience board append chain mismatch")
        hash_payload = {
            key: board.get(key)
            for key in (
                "schema",
                "hash_canonicalization",
                "task_id",
                "append_order",
                "append_chain_head",
                "cards",
                "task_global_aggregation",
            )
        }
        if board.get("board_hash") != _experience_json_sha256(hash_payload):
            raise ValueError("experience board hash mismatch")

        iterations = summary.get("iterations") if isinstance(summary.get("iterations"), list) else []
        success_by_exp = {
            str(item.get("exp_id")): item.get("success") is True
            for item in iterations
            if isinstance(item, dict) and isinstance(item.get("exp_id"), str)
        }
        cards_by_node = {
            str(card.get("node_id")): card
            for card in cards
            if isinstance(card, dict) and isinstance(card.get("node_id"), str)
        }
        node_coverage = set(cards_by_node) == set(success_by_exp) and len(cards) == len(cards_by_node)
        operators = sorted({str(card.get("operator")) for card in cards if isinstance(card, dict)})
        lineages: list[dict[str, Any]] = []
        for card in cards:
            if not isinstance(card, dict) or card.get("operator") != "Crossover":
                continue
            child = str(card.get("node_id") or "")
            parents = [str(item) for item in card.get("parent_ids", []) if isinstance(item, str)]
            distinct_parents = list(dict.fromkeys(parents))
            parent_cards = [cards_by_node.get(parent) for parent in distinct_parents]
            valid = (
                len(parents) == len(distinct_parents) == 2
                and child in success_by_exp
                and all(parent != child for parent in distinct_parents)
                and all(success_by_exp.get(parent) is True for parent in distinct_parents)
                and all(isinstance(parent_card, dict) and parent_card.get("status") == "success" for parent_card in parent_cards)
            )
            lineages.append({"child_exp_id": child, "parent_exp_ids": distinct_parents, "valid": valid})

        graph_path = root / "search_graph.json"
        graph_sha: str | None = None
        if graph_path.is_file() and not graph_path.is_symlink():
            graph, graph_sha = _read_hashed_json(graph_path, label="search graph")
            graph_nodes = {
                str(item.get("exp_id")): item
                for item in graph.get("nodes", [])
                if isinstance(item, dict) and isinstance(item.get("exp_id"), str)
            }
            for lineage in lineages:
                graph_node = graph_nodes.get(lineage["child_exp_id"])
                graph_parents = (
                    [str(item) for item in graph_node.get("reference_parent_ids", []) if isinstance(item, str)]
                    if isinstance(graph_node, dict)
                    else []
                )
                lineage["graph_bound"] = set(graph_parents) == set(lineage["parent_exp_ids"])
                lineage["valid"] = bool(lineage["valid"] and lineage["graph_bound"])

        return {
            "integrity": "passed",
            "board_sha256": board_file_sha,
            "board_hash": board.get("board_hash"),
            "search_graph_sha256": graph_sha,
            "node_coverage": node_coverage,
            "operators": operators,
            "crossover_lineages": lineages,
        }
    except (OSError, ValueError) as exc:
        return {
            "integrity": "failed",
            "error": str(exc),
            "operators": [],
            "crossover_lineages": [],
        }


def _build_claim_audit(
    root: Path,
    summary: dict[str, Any],
    *,
    freeze_sha: str,
    review: dict[str, Any],
    review_sha: str,
) -> dict[str, Any]:
    execution = _summary_execution_evidence(root, summary)
    search = _search_evidence(root, summary)
    valid_crossovers = [item for item in search["crossover_lineages"] if item.get("valid") is True]
    operator_set_complete = set(search["operators"]) == set(DEMO_OPERATORS)

    executed_count = int(execution["executed_candidate_count"])
    execution_supported = execution["complete_fixed_budget"] is True
    if execution_supported:
        execution_claim = (
            f"{executed_count} real local CPU candidate executions completed under the "
            f"{DEMO_MAX_NODES}-node fixed demo budget."
        )
    else:
        execution_claim = (
            f"{executed_count} candidate execution records are bound to the summary; "
            "completion of the fixed demo budget remains on hold."
        )

    structure_supported = (
        search.get("integrity") == "passed"
        and search.get("node_coverage") is True
        and operator_set_complete
        and bool(valid_crossovers)
    )
    if structure_supported:
        crossover = valid_crossovers[-1]
        structure_claim = (
            f"Observed operators are {', '.join(DEMO_OPERATORS)}; Crossover {crossover['child_exp_id']} "
            f"is bound to distinct parents {' and '.join(crossover['parent_exp_ids'])}."
        )
    else:
        observed = ", ".join(search["operators"]) or "none"
        structure_claim = (
            f"Observed operators are {observed}; a complete operator set with verified two-parent "
            "Crossover lineage remains on hold."
        )

    review_supported = review.get("status") == "passed" and review.get("candidate_freeze_sha256") == freeze_sha
    if review_supported:
        review_claim = (
            f"Independent review passed on {review.get('rows')} sealed synthetic holdout rows "
            f"(roc_auc={float(review['score']):.6f}) and is hash-bound to the frozen candidate."
        )
    else:
        review_claim = "Independent sealed-holdout review evidence remains on hold."

    decisions = [
        {"claim_id": "execution_count", "claim": execution_claim, "decision": "supported" if execution_supported else "hold"},
        {"claim_id": "operators_and_crossover", "claim": structure_claim, "decision": "supported" if structure_supported else "hold"},
        {"claim_id": "independent_review", "claim": review_claim, "decision": "supported" if review_supported else "hold"},
    ]
    supported = [item["claim"] for item in decisions if item["decision"] == "supported"]
    held = [item["claim"] for item in decisions if item["decision"] == "hold"]
    return {
        "schema": "evomind.demo.claim_audit.v1",
        "status": "passed" if not held else "hold",
        "claims": decisions,
        "supported_claims": supported,
        "held_claims": held,
        # Preserve the established consumer field while making its content
        # evidence-derived rather than hard-coded.
        "unsupported_claims": held,
        "evidence": {
            "execution_summary_sha256": _json_sha256(summary),
            "candidate_freeze_sha256": freeze_sha,
            "independent_review_sha256": review_sha,
            "execution": execution,
            "search": search,
        },
        "claim_boundary": "Product-demo evidence only; not MLE-Bench, Kaggle, SIIM private grading or an official medal.",
    }


def _validate_claim_audit(
    root: Path,
    *,
    freeze_sha: str,
    review_sha: str,
    expected: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    claim, claim_sha = _read_hashed_json(root / "claim-audit.json", label="claim audit")
    if claim.get("schema") != "evomind.demo.claim_audit.v1":
        raise ValueError("claim audit schema mismatch")
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("claim audit evidence is missing")
    if evidence.get("candidate_freeze_sha256") != freeze_sha:
        raise ValueError("claim audit is not bound to the current candidate freeze")
    if evidence.get("independent_review_sha256") != review_sha:
        raise ValueError("claim audit is not bound to the current independent review")
    decisions = claim.get("claims")
    if not isinstance(decisions, list) or len(decisions) != 3:
        raise ValueError("claim audit decisions are incomplete")
    if any(
        not isinstance(item, dict)
        or item.get("decision") not in {"supported", "hold"}
        or not isinstance(item.get("claim"), str)
        for item in decisions
    ):
        raise ValueError("claim audit contains an invalid decision")
    supported = [item["claim"] for item in decisions if item["decision"] == "supported"]
    held = [item["claim"] for item in decisions if item["decision"] == "hold"]
    expected_status = "passed" if not held else "hold"
    if (
        claim.get("status") != expected_status
        or claim.get("supported_claims") != supported
        or claim.get("held_claims") != held
        or claim.get("unsupported_claims") != held
    ):
        raise ValueError("claim audit supported/hold projections are inconsistent")
    if expected is not None and claim != expected:
        raise ValueError("claim audit content differs from the current execution evidence")
    return claim, claim_sha


def verify_demo_evidence_chain(
    exp_root: str | Path,
    campaign_root: str | Path,
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify freeze -> independent review -> claim audit, raising on any drift."""

    root = Path(exp_root).resolve()
    campaign = Path(campaign_root).resolve()
    resolved_summary = summary
    summary_path = root / "summary.json"
    if resolved_summary is None and summary_path.is_file() and not summary_path.is_symlink():
        resolved_summary, _ = _read_hashed_json(summary_path, label="execution summary")

    freeze, freeze_sha, frozen_hashes = _validate_candidate_freeze(root, summary=resolved_summary)
    review, review_sha = _validate_independent_review(
        root,
        campaign,
        freeze=freeze,
        freeze_sha=freeze_sha,
        frozen_hashes=frozen_hashes,
    )
    expected_claim = (
        _build_claim_audit(root, resolved_summary, freeze_sha=freeze_sha, review=review, review_sha=review_sha)
        if resolved_summary is not None
        else None
    )
    claim, claim_sha = _validate_claim_audit(
        root,
        freeze_sha=freeze_sha,
        review_sha=review_sha,
        expected=expected_claim,
    )
    return {
        "freeze": freeze,
        "freeze_sha256": freeze_sha,
        "review": review,
        "independent_review_sha256": review_sha,
        "claim_audit": claim,
        "claim_audit_sha256": claim_sha,
    }


def _write_public_dataset_contract(campaign_root: Path, manifest: dict[str, Any]) -> Path:
    """Bind candidate-visible demo inputs without exposing sealed labels."""

    public_files = []
    manifest_files = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    for relative in ("data/train.csv", "data/test.csv"):
        item = manifest_files.get(relative)
        if not isinstance(item, dict):
            raise ValueError(f"demo dataset manifest is missing {relative}")
        path = campaign_root / relative
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"demo public dataset hash mismatch: {relative}")
        public_files.append(
            {
                "path": Path(relative).name,
                "bytes": path.stat().st_size,
                "sha256": item["sha256"],
            }
        )
    contract = {
        "schema": "evomind.demo.public_dataset_contract.v1",
        "schema_version": DEMO_SCHEMA_VERSION,
        "task_id": DEMO_TASK_ID,
        "split_policy": "fixed synthetic train plus feature-only sealed holdout",
        "public_files": public_files,
        "candidate_private_label_access": False,
        "official_competition_data": False,
    }
    target = campaign_root / "data" / "dataset_contract.json"
    target.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def prepare_demo_dataset(root: str | Path) -> dict[str, Any]:
    """Create or verify the small fixed demo dataset and sealed review labels."""

    campaign_root = Path(root).expanduser().resolve()
    data_dir = campaign_root / "data"
    review_dir = campaign_root / "review"
    manifest_path = campaign_root / "dataset-manifest.json"
    expected_files = [data_dir / "train.csv", data_dir / "test.csv", review_dir / "holdout_labels.csv"]
    if manifest_path.is_file() and all(path.is_file() for path in expected_files):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
        if all(files.get(path.relative_to(campaign_root).as_posix()) == _sha256(path) for path in expected_files):
            _write_public_dataset_contract(campaign_root, manifest)
            return manifest

    campaign_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    features, labels = make_classification(
        n_samples=3_000,
        n_features=12,
        n_informative=7,
        n_redundant=3,
        n_clusters_per_class=3,
        class_sep=0.8,
        flip_y=0.04,
        random_state=42,
    )
    columns = [f"feature_{index:02d}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    frame.insert(0, "customer_id", [f"CUST{index:05d}" for index in range(len(frame))])
    frame["churned"] = labels
    # The final 600 records form the sealed independent-review set.  Candidate
    # processes receive its features but the labels live outside --data-dir.
    train = frame.iloc[:2_400].copy()
    holdout = frame.iloc[2_400:].copy()
    train.to_csv(data_dir / "train.csv", index=False)
    holdout.drop(columns=["churned"]).to_csv(data_dir / "test.csv", index=False)
    holdout[["customer_id", "churned"]].to_csv(review_dir / "holdout_labels.csv", index=False)
    files = [
        {
            "path": path.relative_to(campaign_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in expected_files
    ]
    manifest = {
        "schema": "evomind.demo.dataset_manifest.v1",
        "schema_version": DEMO_SCHEMA_VERSION,
        "task_id": DEMO_TASK_ID,
        "generator": "sklearn.make_classification",
        "random_state": 42,
        "train_rows": len(train),
        "sealed_holdout_rows": len(holdout),
        "target": "churned",
        "files": files,
        "claim_boundary": "Synthetic product-demo data; not MLE-Bench or an official competition result.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_public_dataset_contract(campaign_root, manifest)
    return manifest


def _candidate_code(index: int) -> str:
    model_blocks = {
        0: "model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))",
        1: "model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3, n_jobs=1, random_state=42)",
        2: "raise ValueError('invalid_hyperparameter_detected_by_real_execution')",
        3: "model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)",
        4: "model = DecisionTreeClassifier(max_depth=1, random_state=42)",
        5: "model = GaussianNB()",
        6: "model = ExtraTreesClassifier(n_estimators=80, max_depth=3, n_jobs=1, random_state=42)",
        7: """model = VotingClassifier(estimators=[
    ('linear', make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))),
    ('forest', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3, n_jobs=1, random_state=42)),
    ('boosting', HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)),
], voting='soft', weights=[1, 2, 3])""",
    }
    model_names = [
        "scaled_logistic_regression",
        "random_forest",
        "invalid_hyperparameter_probe",
        "hist_gradient_boosting_repair",
        "shallow_tree_ablation",
        "gaussian_naive_bayes_ablation",
        "extra_trees_ablation",
        "complementary_soft_voting_crossover",
    ]
    block = model_blocks[index]
    name = model_names[index]
    return f'''from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()
data_dir = Path(args.data_dir).resolve()
out_dir = Path(args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
feature_columns = [column for column in train.columns if column.startswith("feature_")]
X_train, X_valid, y_train, y_valid = train_test_split(
    train[feature_columns], train["churned"], test_size=0.30, stratify=train["churned"], random_state=42
)
{block}
model.fit(X_train, y_train)
valid_probability = model.predict_proba(X_valid)[:, 1]
score = float(roc_auc_score(y_valid, valid_probability))
test_probability = model.predict_proba(test[feature_columns])[:, 1]
pd.DataFrame({{"customer_id": test["customer_id"], "churn_probability": test_probability}}).to_csv(out_dir / "submission.csv", index=False)
joblib.dump(model, out_dir / "model.joblib")
digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
environment = {{
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
    "system": platform.system(),
}}
environment_hash = hashlib.sha256(
    json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
metrics = {{
    "schema": "evomind.demo.public_validation.v1",
    "cv_score": score,
    "metric": "roc_auc",
    "metric_direction": "maximize",
    "model": "{name}",
    "split": "stratified_holdout_seed_42",
    "train_sha256": digest,
    "evaluator_version": "evomind.demo.public_validation.v1",
    "environment_hash": environment_hash,
    "official_submission_performed": False,
}}
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"CV_SCORE={{score:.12f}}")
'''


class DemoVariationGenerator:
    """Fixed proposal plan whose candidate metrics still come from execution."""

    STRATEGIES = (
        "linear_baseline",
        "bagged_trees",
        "invalid_hyperparameter_probe",
        "gradient_boosting_repair",
        "shallow_tree_ablation",
        "probabilistic_baseline",
        "randomized_tree_ablation",
        "complementary_soft_voting",
    )

    def propose(self, context: TaskContext, *, exp_id: str, mode: str = "Base", parent_exp_id: str | None = None,
                expansion_type: str = "primary", **_: Any) -> VariationProposal:
        index = int(exp_id.removeprefix("EXP"))
        if context.task_name != DEMO_TASK_ID or index not in range(DEMO_MAX_NODES):
            raise ValueError("deterministic demo plan is bound to the isolated demo task and eight nodes")
        strategy = self.STRATEGIES[index]
        prompt = (
            f"Verified demo proposal {index}: operator mode={mode}; expansion={expansion_type}; "
            f"strategy={strategy}. Execute against the synthetic public-validation split."
        )
        return VariationProposal(
            exp_id=exp_id,
            code=_candidate_code(index),
            hypothesis=f"Round {index + 1} evaluates {strategy} using real local CPU execution.",
            changes_summary=f"{mode}/{expansion_type}: {strategy}",
            applied_strategies=[strategy],
            parent_exp_id=parent_exp_id,
            code_generation_mode=mode,
            provider="deterministic_demo_plan",
            model=DEMO_PLAN_ID,
            llm_input_tokens=0,
            llm_output_tokens=0,
            raw_response="",
            prompt=prompt,
        )


def freeze_and_review_demo_candidate(exp_root: str | Path, summary: dict[str, Any], campaign_root: str | Path) -> dict[str, Any]:
    """Freeze the real best artifacts, then run a label-sealed independent review."""

    root = Path(exp_root).resolve()
    best_exp_id = str(summary.get("best_exp_id") or "")
    if not best_exp_id:
        raise ValueError("demo candidate freeze requires a best experiment")
    out_dir = root / best_exp_id / "out"
    artifact_paths = [out_dir / "submission.csv", out_dir / "metrics.json", out_dir / "model.joblib"]
    if not all(path.is_file() for path in artifact_paths):
        raise ValueError("demo candidate artifacts are incomplete")
    frozen_at = _utc_now()
    freeze = {
        "schema": "evomind.demo.candidate_freeze.v1",
        "schema_version": DEMO_SCHEMA_VERSION,
        "task_id": DEMO_TASK_ID,
        "run_id": root.name,
        "best_exp_id": best_exp_id,
        "status": "frozen_before_independent_review",
        "frozen_at": frozen_at,
        "public_validation_score": summary.get("best_cv_score"),
        "files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_paths
        ],
        "official_submission_performed": False,
        "private_grader_used": False,
    }
    freeze_path = root / "candidate-freeze.json"
    _write_json(freeze_path, freeze)
    freeze, freeze_sha, frozen_hashes = _validate_candidate_freeze(root, summary=summary)

    campaign = Path(campaign_root).resolve()
    labels_path = campaign / "review" / "holdout_labels.csv"
    submission_path = out_dir / "submission.csv"
    labels = pd.read_csv(labels_path)
    submission = pd.read_csv(submission_path)
    merged = labels.merge(submission, on="customer_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise ValueError("independent review submission/label identity mismatch")
    score = float(roc_auc_score(merged["churned"], merged["churn_probability"]))
    # Reloading the exact frozen model is part of the independent smoke check.
    model = joblib.load(out_dir / "model.joblib")
    if not hasattr(model, "predict_proba"):
        raise ValueError("frozen demo model failed reload smoke")
    review = {
        "schema": "evomind.demo.independent_review.v1",
        "schema_version": DEMO_SCHEMA_VERSION,
        "status": "passed",
        "reviewer": "IndependentDemoReviewer",
        "reviewed_at": _utc_now(),
        "candidate_freeze_sha256": freeze_sha,
        "submission_sha256": _sha256(submission_path),
        "sealed_labels_sha256": _sha256(labels_path),
        "metric": "roc_auc",
        "score": score,
        "rows": len(merged),
        "model_reload_passed": True,
        "source": "sealed_synthetic_holdout",
        "official_submission_performed": False,
    }
    review_path = root / "independent-review.json"
    _write_json(review_path, review)
    # Re-verify the freeze immediately before constructing the next link.  A
    # changed candidate cannot be grandfathered into a review or claim.
    freeze, freeze_sha, frozen_hashes = _validate_candidate_freeze(root, summary=summary)
    review, review_sha = _validate_independent_review(
        root,
        campaign,
        freeze=freeze,
        freeze_sha=freeze_sha,
        frozen_hashes=frozen_hashes,
    )
    claim_audit = _build_claim_audit(root, summary, freeze_sha=freeze_sha, review=review, review_sha=review_sha)
    _write_json(root / "claim-audit.json", claim_audit)
    return verify_demo_evidence_chain(root, campaign, summary=summary)
