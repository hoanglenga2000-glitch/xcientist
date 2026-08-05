"""Pinned OpenAI MLE-Bench split75 integrity checks.

This module is intentionally standard-library only so CI and release archive
verification can share the same fail-closed contract without importing any
training runtime or credentials.
"""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


class Split75ContractError(ValueError):
    """Raised when an embedded benchmark list drifts from the pinned split."""


EXPECTED_REPOSITORY = "https://github.com/openai/mle-bench"
EXPECTED_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
EXPECTED_SPLIT_PATH = "experiments/splits/split75.txt"
EXPECTED_LOCAL_MANIFEST = "benchmark/mle_bench_75/openai_split75_507f92e.txt"
EXPECTED_MANIFEST_SHA256 = "aa6a4dbfd19fee0536235be78361968603c84f0d4e06d4ed0ddc9bb212023057"
EXPECTED_TASK_COUNT = 75
EXPECTED_COMPETITIONS_SHA256 = "e209c85a3bbb6d871e7d323a82dd6a1bc555946796aee20d0a0b21af15ae3d0c"
EXPECTED_KAGGLE_SLUGS_SHA256 = "88863f653fdcb7932dbee7a11b77f0b44485d875bf99a3133e456ebe8577f729"
EXPECTED_RUNNER_SOURCE_SHA256 = "29775ac2b483b4d5363c98130913a23d77b22b31018f4f6879fd9ee2d222e74c"
EXPECTED_RULES_SOURCE_SHA256 = "ebf8e2386eea9f81ca74a35a6c17e784c917d20345605205681b09c2f7a06c11"
_MUTATING_METHODS = frozenset(
    {
        "__delitem__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
)


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _binding_aliases(module: ast.Module, variable_name: str) -> set[str]:
    aliases = {variable_name}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(module):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Name) or value.id not in aliases:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return aliases


def _assert_literal_binding_immutable(
    module: ast.Module,
    variable_name: str,
    assignment: ast.Assign,
    *,
    source_name: str,
) -> None:
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise Split75ContractError(
            f"{source_name} {variable_name} must use one direct literal assignment"
        )
    allowed_target = assignment.targets[0]
    aliases = _binding_aliases(module, variable_name)
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Name)
            and node.id == variable_name
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node is not allowed_target
        ):
            raise Split75ContractError(
                f"{source_name} mutates or rebinds {variable_name} after its literal assignment"
            )
        if (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _root_name(node) in aliases
        ):
            raise Split75ContractError(
                f"{source_name} mutates {variable_name} through an item or attribute assignment"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATING_METHODS
            and _root_name(node.func.value) in aliases
        ):
            raise Split75ContractError(
                f"{source_name} mutates {variable_name} through {node.func.attr}()"
            )


def _canonical_sha256(value: Any, *, sort_keys: bool = False) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(source: str) -> str:
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _literal_assignment(source: str, variable_name: str, *, source_name: str) -> Any:
    try:
        module = ast.parse(source, filename=source_name)
    except SyntaxError as exc:
        raise Split75ContractError(f"{source_name} is not valid Python: {exc}") from exc
    bindings: list[tuple[ast.Assign, Any]] = []
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            try:
                bindings.append((node, ast.literal_eval(node.value)))
            except (TypeError, ValueError) as exc:
                raise Split75ContractError(
                    f"{source_name} {variable_name} must be a literal"
                ) from exc
    if len(bindings) != 1:
        raise Split75ContractError(
            f"{source_name} must define exactly one literal {variable_name}, found {len(bindings)}"
        )
    assignment, value = bindings[0]
    _assert_literal_binding_immutable(
        module,
        variable_name,
        assignment,
        source_name=source_name,
    )
    return value


def _string_list(value: Any, *, context: str, expected_count: int) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise Split75ContractError(f"{context} must be a non-empty string list")
    if len(value) != expected_count or len(set(value)) != expected_count:
        raise Split75ContractError(
            f"{context} must contain {expected_count} unique entries, got {len(value)}/{len(set(value))}"
        )
    return value


def verify_split75_contract(
    *,
    manifest_bytes: bytes,
    upstream: Any,
    registry: Any,
    runner_source: str,
    rules_source: str,
) -> dict[str, Any]:
    runner_source_digest = _source_sha256(runner_source)
    if runner_source_digest != EXPECTED_RUNNER_SOURCE_SHA256:
        raise Split75ContractError(
            "scripts/mlebench_server_runner.py differs from the reviewed source pin: "
            f"actual={runner_source_digest}, expected={EXPECTED_RUNNER_SOURCE_SHA256}"
        )
    rules_source_digest = _source_sha256(rules_source)
    if rules_source_digest != EXPECTED_RULES_SOURCE_SHA256:
        raise Split75ContractError(
            "scripts/accept_kaggle_rules_all75.py differs from the reviewed source pin: "
            f"actual={rules_source_digest}, expected={EXPECTED_RULES_SOURCE_SHA256}"
        )
    if not isinstance(upstream, dict):
        raise Split75ContractError("UPSTREAM.json must contain an object")
    if upstream.get("schema") != "academic_research_os.mle_bench_upstream.v1":
        raise Split75ContractError("UPSTREAM.json schema is not supported")
    if upstream.get("repository") != EXPECTED_REPOSITORY:
        raise Split75ContractError("UPSTREAM.json repository is not the official OpenAI MLE-Bench repo")
    if upstream.get("commit") != EXPECTED_COMMIT:
        raise Split75ContractError("UPSTREAM.json commit differs from the pinned OpenAI MLE-Bench commit")
    if upstream.get("split_path") != EXPECTED_SPLIT_PATH:
        raise Split75ContractError("UPSTREAM.json split_path must pin split75.txt")
    if upstream.get("local_manifest") != EXPECTED_LOCAL_MANIFEST:
        raise Split75ContractError("UPSTREAM.json local_manifest is unexpected")
    if upstream.get("task_count") != EXPECTED_TASK_COUNT or isinstance(upstream.get("task_count"), bool):
        raise Split75ContractError("UPSTREAM.json task_count must be the integer 75")
    expected_digest = upstream.get("sha256")
    actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_digest != EXPECTED_MANIFEST_SHA256 or actual_digest != EXPECTED_MANIFEST_SHA256:
        raise Split75ContractError(
            "split75 manifest SHA-256 mismatch: "
            f"actual={actual_digest}, metadata={expected_digest!r}, pinned={EXPECTED_MANIFEST_SHA256}"
        )
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Split75ContractError("split75 manifest is not UTF-8") from exc
    manifest_ids = manifest_text.splitlines()
    if any(not item or item != item.strip() for item in manifest_ids):
        raise Split75ContractError("split75 manifest contains blank or padded identifiers")
    manifest_ids = _string_list(
        manifest_ids,
        context="split75 manifest",
        expected_count=EXPECTED_TASK_COUNT,
    )

    if not isinstance(registry, dict):
        raise Split75ContractError("tasks_template.json must contain an object")
    reference = registry.get("mle_bench_reference")
    if not isinstance(reference, dict):
        raise Split75ContractError("tasks_template.json mle_bench_reference must be an object")
    for field_name in ("repository", "commit", "split_path", "local_manifest", "sha256"):
        if reference.get(field_name) != upstream.get(field_name):
            raise Split75ContractError(
                f"tasks_template.json reference field {field_name!r} disagrees with UPSTREAM.json"
            )
    if reference.get("total_tasks") != EXPECTED_TASK_COUNT or isinstance(reference.get("total_tasks"), bool):
        raise Split75ContractError("tasks_template.json reference total_tasks must be 75")

    local_tasks = registry.get("tasks")
    if not isinstance(local_tasks, list) or any(not isinstance(item, dict) for item in local_tasks):
        raise Split75ContractError("tasks_template.json tasks must be an object list")
    local_competitions = {
        item.get("competition_name")
        for item in local_tasks
        if isinstance(item.get("competition_name"), str)
    }
    expected_overlap = sorted(local_competitions.intersection(manifest_ids))
    declared_overlap = reference.get("locally_registered_official_competitions")
    if not isinstance(declared_overlap, list) or declared_overlap != expected_overlap:
        raise Split75ContractError(
            "tasks_template.json official overlap mismatch: "
            f"declared={declared_overlap!r}, expected={expected_overlap!r}"
        )

    runner_rows = _literal_assignment(
        runner_source,
        "COMPETITIONS",
        source_name="scripts/mlebench_server_runner.py",
    )
    if (
        not isinstance(runner_rows, list)
        or any(
            not isinstance(row, (tuple, list))
            or len(row) != 3
            or any(not isinstance(value, str) or not value for value in row)
            for row in runner_rows
        )
    ):
        raise Split75ContractError("mlebench_server_runner.py COMPETITIONS must contain string triples")
    runner_ids = [row[0] for row in runner_rows]
    _string_list(
        runner_ids,
        context="runner COMPETITIONS",
        expected_count=EXPECTED_TASK_COUNT,
    )
    if set(runner_ids) != set(manifest_ids):
        raise Split75ContractError(
            "runner COMPETITIONS differs from split75: "
            f"extra={sorted(set(runner_ids) - set(manifest_ids))}, "
            f"missing={sorted(set(manifest_ids) - set(runner_ids))}"
        )
    competitions_digest = _canonical_sha256(runner_rows)
    if competitions_digest != EXPECTED_COMPETITIONS_SHA256:
        raise Split75ContractError(
            "runner COMPETITIONS tier/type/order metadata differs from the pinned contract: "
            f"actual={competitions_digest}, expected={EXPECTED_COMPETITIONS_SHA256}"
        )

    slug_mapping = _literal_assignment(
        runner_source,
        "KAGGLE_SLUGS",
        source_name="scripts/mlebench_server_runner.py",
    )
    if (
        not isinstance(slug_mapping, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in slug_mapping.items()
        )
    ):
        raise Split75ContractError("runner KAGGLE_SLUGS must be a non-empty string mapping")
    if set(slug_mapping) != set(manifest_ids):
        raise Split75ContractError(
            "runner KAGGLE_SLUGS differs from split75: "
            f"extra={sorted(set(slug_mapping) - set(manifest_ids))}, "
            f"missing={sorted(set(manifest_ids) - set(slug_mapping))}"
        )
    slug_mapping_digest = _canonical_sha256(slug_mapping, sort_keys=True)
    if slug_mapping_digest != EXPECTED_KAGGLE_SLUGS_SHA256:
        raise Split75ContractError(
            "runner KAGGLE_SLUGS targets differ from the pinned download mapping: "
            f"actual={slug_mapping_digest}, expected={EXPECTED_KAGGLE_SLUGS_SHA256}"
        )

    rules_ids = _string_list(
        _literal_assignment(
            rules_source,
            "SPLIT75",
            source_name="scripts/accept_kaggle_rules_all75.py",
        ),
        context="rules SPLIT75",
        expected_count=EXPECTED_TASK_COUNT,
    )
    if rules_ids != manifest_ids:
        raise Split75ContractError("rules SPLIT75 order/content differs from the pinned manifest")

    return {
        "manifest_sha256": actual_digest,
        "official_tasks": len(manifest_ids),
        "runner_tasks": len(runner_ids),
        "runner_source_sha256": runner_source_digest,
        "runner_competitions_sha256": competitions_digest,
        "runner_slug_mappings": len(slug_mapping),
        "runner_slug_mapping_sha256": slug_mapping_digest,
        "rules_tasks": len(rules_ids),
        "rules_source_sha256": rules_source_digest,
        "local_official_overlap": expected_overlap,
        "official_competition_ids": manifest_ids,
    }
