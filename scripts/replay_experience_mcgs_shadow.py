#!/usr/bin/env python3
"""Replay exactly six frozen public search traces without external calls.

The command accepts a single, repository-contained manifest.  That manifest
binds the fixed screening task order, every graph byte stream, and every source
file consumed by the replay.  Inputs are read twice, public-only material is
checked before card construction, and the complete six-task campaign is
replayed independently twice.  The output is created once with ``O_EXCL``.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_os.experience_mcgs import (  # noqa: E402
    BudgetLedger,
    ExperienceBoard,
    ExperienceCard,
    ExperienceCardBuilder,
    ExperienceCost,
    LazySummaryCache,
    SearchOperator,
    retrieve_experience,
    route_operator,
    select_experience_parent,
    shadow_replay,
    structural_features,
)
from research_os.mle_ab_campaign import SCREEN_TASKS, canonical_bytes, sha256_file  # noqa: E402

SCHEMA = "evomind.experience_mcgs.shadow_replay.v3"
MANIFEST_SCHEMA = "evomind.experience_mcgs.shadow_manifest.v1"
BOUNDARY_AUDIT_SCHEMA = "evomind.shadow_replay.boundary_audit.v1"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FORBIDDEN_KEY = re.compile(r"(?:^|[_\s-])(?:private|leaderboard|official)(?:$|[_\s-])", re.I)
_FORBIDDEN_VALUE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])private(?:[\s_-]*(?:grader|feedback|score|metric|evaluation|"
    r"result|label|labels|data|split|holdout|test|submission))?(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])leaderboard(?:[\s_-]*(?:feedback|score|rank|result|data))?(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])official[\s_-]*(?:grader|feedback|score|rank|medal|result|submission)"
    r"(?![A-Za-z0-9])"
    r")",
    re.I,
)
_MANIFEST_KEYS = frozenset({"schema", "tasks"})
_TASK_KEYS = frozenset({"task_id", "graph", "sha256", "code"})
_CODE_KEYS = frozenset({"node_id", "path", "sha256"})
_BOUNDARY_KINDS = ("llm", "grader", "submission", "network")


class ShadowReplayContractError(RuntimeError):
    """Raised when immutable Shadow Replay evidence violates its contract."""


class BoundaryViolation(ShadowReplayContractError):
    """Raised after an external-call guard records and blocks an attempt."""


@dataclass(frozen=True)
class CodeBinding:
    node_id: str
    reference: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class GraphBinding:
    task_id: str
    reference: str
    path: Path
    sha256: str
    code: tuple[CodeBinding, ...]


@dataclass(frozen=True)
class ManifestBinding:
    reference: str
    path: Path
    sha256: str
    graphs: tuple[GraphBinding, ...]


@dataclass
class PublicScan:
    mapping_fields: int = 0
    sequence_items: int = 0
    string_leaves: int = 0
    empty_restricted_fields_excluded: int = 0

    def merge(self, other: PublicScan) -> None:
        self.mapping_fields += other.mapping_fields
        self.sequence_items += other.sequence_items
        self.string_leaves += other.string_leaves
        self.empty_restricted_fields_excluded += other.empty_restricted_fields_excluded

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "mapping_fields": self.mapping_fields,
            "sequence_items": self.sequence_items,
            "string_leaves": self.string_leaves,
            "empty_restricted_fields_excluded": self.empty_restricted_fields_excluded,
        }


class BoundaryAudit:
    """Deny and deterministically count every external Shadow Replay boundary.

    Network and process/import escape hatches are guarded for the complete
    replay context.  Explicit LLM, grader, submission, and network capability
    methods are also exposed so callers and tests cannot silently bypass the
    same accounting boundary.
    """

    def __init__(self) -> None:
        self.attempted = {kind: 0 for kind in _BOUNDARY_KINDS}
        self.completed = {kind: 0 for kind in _BOUNDARY_KINDS}
        self.blocked = {kind: 0 for kind in _BOUNDARY_KINDS}
        self.events: list[dict[str, Any]] = []
        self.enforcement_cycles = 0
        self.local_file_reads = 0
        self.local_bytes_read = 0

    def intercept(self, boundary: str, operation: str, detail: str = "") -> None:
        if boundary not in self.attempted:
            raise ValueError(f"unknown Shadow Replay boundary: {boundary}")
        self.attempted[boundary] += 1
        self.blocked[boundary] += 1
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "boundary": boundary,
                "operation": str(operation),
                "detail_sha256": hashlib.sha256(str(detail).encode("utf-8")).hexdigest(),
            }
        )
        raise BoundaryViolation(f"Shadow Replay blocked {boundary} boundary: {operation}")

    def llm_call(self, operation: str = "llm.call") -> None:
        self.intercept("llm", operation)

    def grader_call(self, operation: str = "grader.call") -> None:
        self.intercept("grader", operation)

    def submission_call(self, operation: str = "submission.call") -> None:
        self.intercept("submission", operation)

    def network_call(self, operation: str = "network.call") -> None:
        self.intercept("network", operation)

    def record_local_read(self, size: int) -> None:
        self.local_file_reads += 1
        self.local_bytes_read += max(0, int(size))

    @staticmethod
    def _classify_process(command: Any) -> str:
        if isinstance(command, (list, tuple)):
            text = " ".join(str(item) for item in command)
        else:
            text = str(command)
        lowered = text.lower()
        if "kaggle" in lowered or "submit" in lowered:
            return "submission"
        if "grader" in lowered or " grade" in f" {lowered}":
            return "grader"
        if any(marker in lowered for marker in ("openai", "anthropic", " llm", " model-api")):
            return "llm"
        return "network"

    @contextmanager
    def enforce(self) -> Iterator[BoundaryAudit]:
        self.enforcement_cycles += 1
        audit = self
        original_socket = socket.socket
        original_create_connection = socket.create_connection
        original_getaddrinfo = socket.getaddrinfo
        original_gethostbyname = socket.gethostbyname
        original_import = builtins.__import__
        original_popen = subprocess.Popen
        original_system = os.system

        class GuardedSocket(original_socket):
            def connect(self, address: Any) -> None:  # type: ignore[override]
                audit.intercept("network", "socket.connect", repr(address))

            def connect_ex(self, address: Any) -> int:  # type: ignore[override]
                audit.intercept("network", "socket.connect_ex", repr(address))

            def sendto(self, *args: Any, **kwargs: Any) -> int:  # type: ignore[override]
                audit.intercept("network", "socket.sendto", repr(args[:2]))

        def blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
            audit.intercept("network", "socket.create_connection", repr(args[:1]))

        def blocked_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
            audit.intercept("network", "socket.getaddrinfo", repr(args[:2]))

        def blocked_gethostbyname(*args: Any, **kwargs: Any) -> Any:
            audit.intercept("network", "socket.gethostbyname", repr(args[:1]))

        def guarded_import(
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> Any:
            lowered = str(name).lower()
            root_name = lowered.split(".", 1)[0]
            if root_name in {"openai", "anthropic", "cohere"} or lowered.startswith("google.generativeai"):
                audit.intercept("llm", "import", name)
            if root_name == "kaggle":
                audit.intercept("submission", "import", name)
            if "grader" in lowered:
                audit.intercept("grader", "import", name)
            return original_import(name, globals, locals, fromlist, level)

        def blocked_popen(command: Any, *args: Any, **kwargs: Any) -> Any:
            boundary = audit._classify_process(command)
            audit.intercept(boundary, "subprocess.Popen", repr(command))

        def blocked_system(command: str) -> int:
            boundary = audit._classify_process(command)
            audit.intercept(boundary, "os.system", command)

        socket.socket = GuardedSocket
        socket.create_connection = blocked_create_connection
        socket.getaddrinfo = blocked_getaddrinfo
        socket.gethostbyname = blocked_gethostbyname
        builtins.__import__ = guarded_import
        subprocess.Popen = blocked_popen  # type: ignore[assignment]
        os.system = blocked_system
        try:
            yield self
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            socket.getaddrinfo = original_getaddrinfo
            socket.gethostbyname = original_gethostbyname
            builtins.__import__ = original_import
            subprocess.Popen = original_popen  # type: ignore[assignment]
            os.system = original_system

    @property
    def zero_external_calls(self) -> bool:
        return not any(self.attempted.values()) and not any(self.completed.values())

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": BOUNDARY_AUDIT_SCHEMA,
            "policy": "deny_all_external_calls",
            "guards": [
                "explicit_capability_boundary",
                "provider_import_boundary",
                "subprocess_boundary",
                "socket_connect_boundary",
                "dns_boundary",
            ],
            "enforcement_cycles": self.enforcement_cycles,
            "attempted": dict(self.attempted),
            "completed": dict(self.completed),
            "blocked": dict(self.blocked),
            "events": list(self.events),
            "local_file_reads": self.local_file_reads,
            "local_bytes_read": self.local_bytes_read,
            "zero_external_calls": self.zero_external_calls,
        }
        core["audit_sha256"] = hashlib.sha256(canonical_bytes(core)).hexdigest()
        return core


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def infer_operator(node: Mapping[str, Any]) -> SearchOperator:
    explicit = str(node.get("operator") or "")
    if explicit in {item.value for item in SearchOperator}:
        return SearchOperator(explicit)
    if not node.get("parent_id") and not node.get("reference_parent_ids"):
        return SearchOperator.DRAFT
    branch = str(node.get("branch_type") or "").lower()
    if not bool(node.get("run_success", True)):
        return SearchOperator.DEBUG
    if "cross" in branch or "aggreg" in branch or len(node.get("reference_parent_ids") or []) >= 2:
        return SearchOperator.CROSSOVER
    return SearchOperator.IMPROVE


def _usage(node: Mapping[str, Any]) -> tuple[ExperienceCost, bool]:
    metrics = node.get("metrics") if isinstance(node.get("metrics"), Mapping) else {}
    raw = node.get("usage") if isinstance(node.get("usage"), Mapping) else metrics.get("usage", {})
    raw = raw if isinstance(raw, Mapping) else {}

    def pick(name: str, *aliases: str, default: Any = 0) -> Any:
        for key in (name, *aliases):
            if key in node:
                return node[key]
            if key in raw:
                return raw[key]
            if key in metrics:
                return metrics[key]
        return default

    required_groups = (
        ("prompt_tokens", "input_tokens", "llm_input_tokens"),
        ("completion_tokens", "output_tokens", "llm_output_tokens"),
        ("wall_seconds", "runtime_seconds", "elapsed_seconds"),
    )
    recorded = all(any(key in node or key in raw or key in metrics for key in group) for group in required_groups)
    prompt = max(0, int(pick("prompt_tokens", "input_tokens", "llm_input_tokens")))
    completion = max(0, int(pick("completion_tokens", "output_tokens", "llm_output_tokens")))
    total = max(0, int(pick("total_tokens", default=prompt + completion)))
    return ExperienceCost(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total or prompt + completion,
        wall_seconds=max(0.0, float(pick("wall_seconds", "runtime_seconds", "elapsed_seconds"))),
        gpu_seconds=max(0.0, float(pick("gpu_seconds"))),
        estimated_cost_usd=max(0.0, float(pick("estimated_cost_usd", "cost_usd"))),
    ), recorded


def _summary(card: ExperienceCard) -> dict[str, str]:
    return {
        "method_overview": str(card.provenance.get("changes_summary") or card.method_family),
        "parent_comparison_experience": (
            f"parents={','.join(card.parent_ids)}; progress={card.progress:.6g}; "
            f"status={card.status}; error_signature={card.error_signature or 'none'}"
        ),
    }


def _parse_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ShadowReplayContractError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ShadowReplayContractError(f"non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ShadowReplayContractError(f"invalid UTF-8 JSON in {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ShadowReplayContractError(f"{label} must contain a JSON object")
    return payload


def _materially_populated(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set, frozenset)):
        return any(_materially_populated(child) for child in value.values()) if isinstance(value, Mapping) else any(
            _materially_populated(child) for child in value
        )
    return True


def _assert_public_only(value: Any, *, label: str, path: str = "root") -> PublicScan:
    scan = PublicScan()

    def visit(child: Any, child_path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                key_text = str(key)
                scan.mapping_fields += 1
                if _FORBIDDEN_KEY.search(key_text):
                    if _materially_populated(nested):
                        raise ShadowReplayContractError(
                            f"restricted evaluation material in {label}: {child_path}.{key_text}"
                        )
                    scan.empty_restricted_fields_excluded += 1
                    continue
                visit(nested, f"{child_path}.{key_text}")
            return
        if isinstance(child, (list, tuple, set, frozenset)):
            for index, nested in enumerate(child):
                scan.sequence_items += 1
                visit(nested, f"{child_path}[{index}]")
            return
        if isinstance(child, str):
            scan.string_leaves += 1
            if _FORBIDDEN_VALUE.search(child):
                raise ShadowReplayContractError(f"restricted evaluation value in {label}: {child_path}")

    visit(value, path)
    return scan


def _is_link_like(path: Path) -> bool:
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    hard_linked_file = stat.S_ISREG(metadata.st_mode) and int(getattr(metadata, "st_nlink", 1)) != 1
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag) or hard_linked_file


def _relative_reference(value: str, *, label: str) -> Path:
    text = str(value or "")
    if not text or "\x00" in text:
        raise ShadowReplayContractError(f"{label} path is required")
    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise ShadowReplayContractError(f"{label} path must be relative")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ShadowReplayContractError(f"{label} path traversal is forbidden: {text}")
    return candidate


def _regular_file_within(base: Path, reference: str, *, label: str) -> tuple[Path, str]:
    relative = _relative_reference(reference, label=label)
    base_resolved = base.resolve(strict=True)
    lexical = base_resolved.joinpath(relative)
    current = base_resolved
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            raise ShadowReplayContractError(f"{label} does not exist: {reference}")
        if _is_link_like(current):
            raise ShadowReplayContractError(f"{label} link/reparse path is forbidden: {reference}")
    try:
        resolved = lexical.resolve(strict=True)
        canonical = resolved.relative_to(base_resolved).as_posix()
    except (OSError, ValueError) as exc:
        raise ShadowReplayContractError(f"{label} escapes canonical containment: {reference}") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ShadowReplayContractError(f"{label} is not a regular file: {reference}")
    return resolved, canonical


def _manifest_file(path: Path) -> tuple[Path, str]:
    root = ROOT.resolve(strict=True)
    supplied = path.expanduser()
    if supplied.is_absolute():
        try:
            relative = supplied.absolute().relative_to(root)
        except ValueError as exc:
            raise ShadowReplayContractError("shadow manifest must be contained by the repository root") from exc
        return _regular_file_within(root, relative.as_posix(), label="shadow manifest")
    return _regular_file_within(root, supplied.as_posix(), label="shadow manifest")


def _read_bytes(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    audit: BoundaryAudit | None = None,
) -> bytes:
    if _is_link_like(path):
        raise ShadowReplayContractError(f"{label} link/reparse path is forbidden")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ShadowReplayContractError(f"{label} is not a regular file")
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ShadowReplayContractError(f"unable to read {label}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise ShadowReplayContractError(f"{label} changed while it was read")
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ShadowReplayContractError(f"{label} SHA-256 mismatch")
    if audit is not None:
        audit.record_local_read(len(raw))
    return raw


def _required_sha(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not _SHA256.fullmatch(digest):
        raise ShadowReplayContractError(f"{label} must be a required lowercase 64-hex SHA-256")
    return digest


def _graph_identity(payload: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> set[str]:
    identities = {str(payload.get("task_id") or "").strip()}
    identities.update(str(node.get("task_name") or "").strip() for node in nodes)
    identities.discard("")
    return identities


def _graph_nodes(payload: Mapping[str, Any], *, task_id: str) -> list[Mapping[str, Any]]:
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ShadowReplayContractError(f"graph has no nodes: {task_id}")
    if not all(isinstance(node, Mapping) for node in raw_nodes):
        raise ShadowReplayContractError(f"graph nodes must be objects: {task_id}")
    nodes = list(raw_nodes)
    if _graph_identity(payload, nodes) != {task_id}:
        raise ShadowReplayContractError(f"graph task identity mismatch: {task_id}")
    node_ids = [str(node.get("exp_id") or "") for node in nodes]
    if any(not node_id for node_id in node_ids):
        raise ShadowReplayContractError(f"graph node is missing exp_id: {task_id}")
    if len(set(node_ids)) != len(node_ids):
        raise ShadowReplayContractError(f"graph contains duplicate exp_id: {task_id}")
    return nodes


def _load_manifest(path: Path, *, audit: BoundaryAudit | None = None) -> ManifestBinding:
    manifest_path, manifest_reference = _manifest_file(path)
    raw = _read_bytes(manifest_path, label="shadow manifest", audit=audit)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    payload = _parse_json(raw, label="shadow manifest")
    if set(payload) != _MANIFEST_KEYS:
        raise ShadowReplayContractError("shadow manifest contains missing or unsupported top-level fields")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ShadowReplayContractError(f"unsupported shadow manifest schema: {payload.get('schema')!r}")
    rows = payload.get("tasks")
    if not isinstance(rows, list) or len(rows) != len(SCREEN_TASKS):
        raise ShadowReplayContractError("shadow manifest must contain exactly six screening tasks")

    graphs: list[GraphBinding] = []
    seen_tasks: set[str] = set()
    for index, expected_task in enumerate(SCREEN_TASKS):
        row = rows[index]
        if not isinstance(row, Mapping) or set(row) != _TASK_KEYS:
            raise ShadowReplayContractError(f"shadow manifest task {index} has an invalid contract")
        task_id = str(row.get("task_id") or "")
        if task_id in seen_tasks:
            raise ShadowReplayContractError(f"shadow manifest contains duplicate task_id: {task_id}")
        seen_tasks.add(task_id)
        if task_id != expected_task:
            raise ShadowReplayContractError(
                f"shadow manifest task order mismatch at {index}: expected {expected_task}, got {task_id}"
            )
        graph_sha256 = _required_sha(row.get("sha256"), label=f"{task_id} graph")
        graph_path, graph_reference = _regular_file_within(
            ROOT,
            str(row.get("graph") or ""),
            label=f"{task_id} graph",
        )
        graph_raw = _read_bytes(
            graph_path,
            label=f"{task_id} graph",
            expected_sha256=graph_sha256,
            audit=audit,
        )
        graph_payload = _parse_json(graph_raw, label=f"{task_id} graph")
        nodes = _graph_nodes(graph_payload, task_id=task_id)
        code_rows = row.get("code")
        if not isinstance(code_rows, list) or len(code_rows) != len(nodes):
            raise ShadowReplayContractError(f"{task_id} code bindings must match every graph node exactly")

        code_bindings: list[CodeBinding] = []
        for node_index, node in enumerate(nodes):
            code_row = code_rows[node_index]
            if not isinstance(code_row, Mapping) or set(code_row) != _CODE_KEYS:
                raise ShadowReplayContractError(f"{task_id} code binding {node_index} has an invalid contract")
            node_id = str(node.get("exp_id") or "")
            if str(code_row.get("node_id") or "") != node_id:
                raise ShadowReplayContractError(f"{task_id} code binding node order mismatch: {node_id}")
            graph_code_reference = _relative_reference(
                str(node.get("code_path") or ""),
                label=f"{task_id}/{node_id} graph code",
            ).as_posix()
            manifest_code_reference = _relative_reference(
                str(code_row.get("path") or ""),
                label=f"{task_id}/{node_id} manifest code",
            ).as_posix()
            if manifest_code_reference != graph_code_reference:
                raise ShadowReplayContractError(f"{task_id}/{node_id} code path binding mismatch")
            code_sha256 = _required_sha(
                code_row.get("sha256"),
                label=f"{task_id}/{node_id} code",
            )
            code_path, canonical_code_reference = _regular_file_within(
                graph_path.parent,
                manifest_code_reference,
                label=f"{task_id}/{node_id} code",
            )
            _read_bytes(
                code_path,
                label=f"{task_id}/{node_id} code",
                expected_sha256=code_sha256,
                audit=audit,
            )
            code_bindings.append(
                CodeBinding(
                    node_id=node_id,
                    reference=canonical_code_reference,
                    path=code_path,
                    sha256=code_sha256,
                )
            )
        graphs.append(
            GraphBinding(
                task_id=task_id,
                reference=graph_reference,
                path=graph_path,
                sha256=graph_sha256,
                code=tuple(code_bindings),
            )
        )
    return ManifestBinding(
        reference=manifest_reference,
        path=manifest_path,
        sha256=manifest_sha256,
        graphs=tuple(graphs),
    )


def _metric_direction(payload: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> str:
    directions = {str(payload.get("metric_direction") or "").strip().lower()}
    directions.update(str(node.get("metric_direction") or "").strip().lower() for node in nodes)
    directions.discard("")
    if len(directions) > 1:
        raise ShadowReplayContractError("graph contains inconsistent metric directions")
    return next(iter(directions), "maximize")


def _load_graph_snapshot(
    binding: GraphBinding,
    *,
    audit: BoundaryAudit,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], dict[str, str], PublicScan]:
    current_graph, current_reference = _regular_file_within(
        ROOT,
        binding.reference,
        label=f"{binding.task_id} graph",
    )
    if current_graph != binding.path or current_reference != binding.reference:
        raise ShadowReplayContractError(f"{binding.task_id} graph canonical path drifted")
    for code_binding in binding.code:
        current_code, current_code_reference = _regular_file_within(
            current_graph.parent,
            code_binding.reference,
            label=f"{binding.task_id}/{code_binding.node_id} code",
        )
        if current_code != code_binding.path or current_code_reference != code_binding.reference:
            raise ShadowReplayContractError(
                f"{binding.task_id}/{code_binding.node_id} code canonical path drifted"
            )
    graph_raw = _read_bytes(
        binding.path,
        label=f"{binding.task_id} graph",
        expected_sha256=binding.sha256,
        audit=audit,
    )
    payload = _parse_json(graph_raw, label=f"{binding.task_id} graph")
    scan = _assert_public_only(payload, label=f"{binding.task_id} raw graph")
    nodes = _graph_nodes(payload, task_id=binding.task_id)
    if len(nodes) != len(binding.code):
        raise ShadowReplayContractError(f"{binding.task_id} graph/code binding count drifted")
    code_by_node: dict[str, str] = {}
    for node, code_binding in zip(nodes, binding.code, strict=True):
        node_id = str(node.get("exp_id") or "")
        if node_id != code_binding.node_id:
            raise ShadowReplayContractError(f"{binding.task_id} graph/code node binding drifted")
        current_reference = _relative_reference(
            str(node.get("code_path") or ""),
            label=f"{binding.task_id}/{node_id} graph code",
        ).as_posix()
        if current_reference != code_binding.reference:
            raise ShadowReplayContractError(f"{binding.task_id}/{node_id} graph code path drifted")
        code_raw = _read_bytes(
            code_binding.path,
            label=f"{binding.task_id}/{node_id} code",
            expected_sha256=code_binding.sha256,
            audit=audit,
        )
        try:
            code = code_raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ShadowReplayContractError(f"{binding.task_id}/{node_id} code is not UTF-8") from exc
        scan.merge(_assert_public_only(code, label=f"{binding.task_id}/{node_id} raw code"))
        code_by_node[node_id] = code
    return payload, nodes, code_by_node, scan


def _replay_once(binding: GraphBinding, *, audit: BoundaryAudit) -> dict[str, Any]:
    payload, nodes, code_by_node, public_scan = _load_graph_snapshot(binding, audit=audit)
    task_id = binding.task_id
    direction = _metric_direction(payload, nodes)
    board = ExperienceBoard(task_id)
    ledger = BudgetLedger(
        max_nodes=max(1, len(nodes)),
        max_total_tokens=max(1, sum(_usage(node)[0].total_tokens for node in nodes) + 1),
        max_wall_seconds=max(1.0, sum(_usage(node)[0].wall_seconds for node in nodes) + 1.0),
        max_cost_usd=None,
    )
    visits: dict[str, int] = {}
    step_rows: list[dict[str, Any]] = []
    prompt_tokens: list[float] = []
    usage_complete = True
    new_best = 0
    no_new_best = 0
    prior_best: float | None = None
    cache_hits = 0
    cache_misses = 0
    with tempfile.TemporaryDirectory(prefix="evomind-shadow-") as temporary:
        cache = LazySummaryCache(Path(temporary) / "summary-cache.json")
        for index, node in enumerate(nodes):
            board_count_before = len(board.cards)
            route_before = route_operator(board, no_new_best_expansions=no_new_best)
            node_id = str(node.get("exp_id") or "")
            executed_operator = infer_operator(node)
            if index == 0 and (board_count_before != 0 or route_before is not SearchOperator.DRAFT):
                raise ShadowReplayContractError("first Shadow Replay step must route from an empty Board as Draft")
            if index == 0 and executed_operator is not SearchOperator.DRAFT:
                raise ShadowReplayContractError("first historical execution must be Draft")
            code = code_by_node[node_id]
            score = node.get("cv_score")
            score_value = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None
            signed_score = (
                None
                if score_value is None
                else (-score_value if direction in {"minimize", "lower", "lower_is_better"} else score_value)
            )
            if signed_score is not None and (prior_best is None or signed_score > prior_best):
                if prior_best is not None:
                    new_best += 1
                prior_best = signed_score
                no_new_best = 0
            else:
                no_new_best += 1

            parent_ids = [str(item) for item in (node.get("reference_parent_ids") or [])]
            if node.get("parent_id") and str(node["parent_id"]) not in parent_ids:
                parent_ids.insert(0, str(node["parent_id"]))
            parent_score = node.get("parent_score")
            if score_value is not None and isinstance(parent_score, (int, float)) and math.isfinite(float(parent_score)):
                raw_delta = (
                    float(parent_score) - score_value
                    if direction in {"minimize", "lower", "lower_is_better"}
                    else score_value - float(parent_score)
                )
                progress = max(-1.0, min(1.0, raw_delta / max(abs(float(parent_score)), abs(score_value), 1e-12)))
            else:
                progress = 0.0
            fingerprint = structural_features(
                code,
                executed_operator,
                str(node.get("branch_type") or "unknown"),
            )
            novelty = 1.0
            if board.cards:

                def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
                    a, b = set(left), set(right)
                    return len(a & b) / len(a | b) if a or b else 1.0

                novelty = 1.0 - max(
                    jaccard(fingerprint, card.structural_fingerprint)
                    for card in board.cards.values()
                )
            cost, recorded = _usage(node)
            usage_complete = usage_complete and recorded
            prompt_tokens.append(float(cost.prompt_tokens))
            failed = not bool(node.get("run_success", True)) or score_value is None
            error_text = str(node.get("promotion_reason") or "") if failed else ""
            _assert_public_only(
                {
                    "prompt": str(node.get("prompt") or node.get("hypothesis") or ""),
                    "code": code,
                    "error": error_text,
                },
                label=f"{task_id}/{node_id} card inputs",
            )
            card = ExperienceCardBuilder.build(
                task_id=task_id,
                node_id=node_id,
                parent_ids=parent_ids,
                operator=executed_operator,
                method_family=str(node.get("branch_type") or "unknown"),
                code=code,
                data_hashes=tuple(str(item) for item in (node.get("data_hashes") or ())),
                prompt=str(node.get("prompt") or node.get("hypothesis") or ""),
                execution_id=str(node.get("execution_id") or node_id),
                status="failed" if failed else "success",
                public_validation_score=score_value,
                metric_direction=direction,
                error=error_text,
                progress=progress,
                novelty=novelty,
                cost=cost,
                provenance={
                    "source": "historical_search_graph",
                    "decision": str(node.get("decision") or ""),
                    "promoted": bool(node.get("promoted", False)),
                    "changes_summary": str(node.get("implementation_summary") or ""),
                    "graph_sha256": binding.sha256,
                    "code_sha256": binding.code[index].sha256,
                },
            )
            board.append(card)
            ledger.record(cost)
            visits[node_id] = visits.get(node_id, 0) + 1
            trace = select_experience_parent(board, visits=visits, parent_visits=visits)
            next_operator = route_operator(board, no_new_best_expansions=no_new_best)
            bundle = retrieve_experience(
                board,
                next_operator,
                selected_node_id=trace.selected_node_id,
                max_cards=8,
                max_tokens=6000,
            )
            step_cache: list[dict[str, Any]] = []
            for card_id in bundle.card_ids:
                summary, hit = cache.get_or_create(
                    board.cards[card_id],
                    summarizer_model="deterministic-public-summary-v1",
                    prompt_template_hash=hashlib.sha256(
                        b"method_overview,parent_comparison_experience:v1"
                    ).hexdigest(),
                    summarize=_summary,
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
                step_cache.append({"card_id": card_id, "hit": hit, "summary": summary})
            step_rows.append(
                {
                    "index": index,
                    "node_id": node_id,
                    "card_id": card.card_id,
                    "board_card_count_before": board_count_before,
                    "route_before_execution": route_before.value,
                    "executed_operator": executed_operator.value,
                    "next_operator": next_operator.value,
                    "selection_for_next_step": trace.to_dict(),
                    "retrieval_for_next_step": bundle.to_dict(),
                    "summary_cache": step_cache,
                    "budget": ledger.to_dict(),
                }
            )

    final = shadow_replay(board, visits=visits)
    token_total = ledger.total_tokens
    return {
        "task_id": task_id,
        "graph_path": binding.reference,
        "graph_sha256": binding.sha256,
        "code_bindings": [
            {"node_id": item.node_id, "path": item.reference, "sha256": item.sha256}
            for item in binding.code
        ],
        "public_only_scan": public_scan.to_dict(),
        "initial_route": {"board_card_count": 0, "operator": SearchOperator.DRAFT.value},
        "node_count": len(nodes),
        "node_coverage": len(board.cards),
        "all_nodes_covered": len(board.cards) == len(nodes),
        "usage_complete": usage_complete,
        "prompt_tokens_p99": percentile(prompt_tokens, 0.99),
        "new_best_count": new_best,
        "new_best_per_million_total_tokens": (new_best * 1_000_000 / token_total) if token_total else 0.0,
        "summary_cache_hits": cache_hits,
        "summary_cache_misses": cache_misses,
        "steps": step_rows,
        "budget": ledger.to_dict(),
        "replay": final,
        "board_hash": board.to_dict()["board_hash"],
    }


def _campaign_replay(
    manifest: ManifestBinding,
    *,
    audit: BoundaryAudit,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = [_replay_once(binding, audit=audit) for binding in manifest.graphs]
    second = [_replay_once(binding, audit=audit) for binding in manifest.graphs]
    first_bytes = canonical_bytes(first)
    second_bytes = canonical_bytes(second)
    first_hash = hashlib.sha256(first_bytes).hexdigest()
    second_hash = hashlib.sha256(second_bytes).hexdigest()
    results: list[dict[str, Any]] = []
    for first_graph, second_graph in zip(first, second, strict=True):
        first_graph_bytes = canonical_bytes(first_graph)
        second_graph_bytes = canonical_bytes(second_graph)
        results.append(
            {
                **first_graph,
                "first_replay_sha256": hashlib.sha256(first_graph_bytes).hexdigest(),
                "second_replay_sha256": hashlib.sha256(second_graph_bytes).hexdigest(),
                "independent_replay_canonical_bytes_equal": first_graph_bytes == second_graph_bytes,
            }
        )
    return results, {
        "independent_full_replays": 2,
        "first_campaign_replay_sha256": first_hash,
        "second_campaign_replay_sha256": second_hash,
        "canonical_bytes_equal": first_bytes == second_bytes,
    }


def build_report(manifest_path: Path) -> dict[str, Any]:
    audit = BoundaryAudit()
    with audit.enforce():
        manifest = _load_manifest(manifest_path, audit=audit)
        results, reproducibility = _campaign_replay(manifest, audit=audit)
        current_manifest, current_manifest_reference = _manifest_file(Path(manifest.reference))
        if current_manifest != manifest.path or current_manifest_reference != manifest.reference:
            raise ShadowReplayContractError("shadow manifest canonical path drifted")
        _read_bytes(
            manifest.path,
            label="shadow manifest",
            expected_sha256=manifest.sha256,
            audit=audit,
        )
        for binding in manifest.graphs:
            current_graph, current_reference = _regular_file_within(
                ROOT,
                binding.reference,
                label=f"{binding.task_id} graph",
            )
            if current_graph != binding.path or current_reference != binding.reference:
                raise ShadowReplayContractError(f"{binding.task_id} graph canonical path drifted")
            _read_bytes(
                binding.path,
                label=f"{binding.task_id} graph",
                expected_sha256=binding.sha256,
                audit=audit,
            )
            for code in binding.code:
                current_code, current_code_reference = _regular_file_within(
                    current_graph.parent,
                    code.reference,
                    label=f"{binding.task_id}/{code.node_id} code",
                )
                if current_code != code.path or current_code_reference != code.reference:
                    raise ShadowReplayContractError(
                        f"{binding.task_id}/{code.node_id} code canonical path drifted"
                    )
                _read_bytes(
                    code.path,
                    label=f"{binding.task_id}/{code.node_id} code",
                    expected_sha256=code.sha256,
                    audit=audit,
                )

    task_ids = [item["task_id"] for item in results]
    boundary_report = audit.to_dict()
    checks = {
        "manifest_bound_by_sha256": _SHA256.fullmatch(manifest.sha256) is not None,
        "exactly_six_screen_tasks_in_fixed_order": task_ids == list(SCREEN_TASKS),
        "graph_and_code_hashes_bound": all(
            _SHA256.fullmatch(item["graph_sha256"]) is not None
            and all(_SHA256.fullmatch(code["sha256"]) is not None for code in item["code_bindings"])
            for item in results
        ),
        "independent_full_replay_canonical_bytes_equal": reproducibility["canonical_bytes_equal"],
        "public_only_raw_material": all(item["public_only_scan"]["status"] == "passed" for item in results),
        "private_feedback_absent": all(item["replay"]["private_feedback_absent"] for item in results),
        "legacy_graphs_and_code_unchanged": True,
        "all_nodes_covered": all(item["all_nodes_covered"] for item in results),
        "recorded_usage_complete": all(item["usage_complete"] for item in results),
        "router_retriever_budget_replayed_per_node": all(
            len(item["steps"]) == item["node_count"] for item in results
        ),
        "first_step_routes_empty_board_as_draft": all(
            item["steps"][0]["board_card_count_before"] == 0
            and item["steps"][0]["route_before_execution"] == SearchOperator.DRAFT.value
            and item["steps"][0]["executed_operator"] == SearchOperator.DRAFT.value
            for item in results
        ),
        "executed_and_next_operator_are_distinct_fields": all(
            all("executed_operator" in step and "next_operator" in step for step in item["steps"])
            for item in results
        ),
        "summary_cache_observed": all(
            item["summary_cache_hits"] + item["summary_cache_misses"] > 0 for item in results
        ),
        "zero_llm_grader_submission_network_calls": boundary_report["zero_external_calls"],
    }
    report = {
        "schema": SCHEMA,
        "mode": "shadow_replay_no_external_calls",
        "status": "passed" if all(checks.values()) else "failed_closed",
        "checks": checks,
        "manifest": {
            "schema": MANIFEST_SCHEMA,
            "path": manifest.reference,
            "sha256": manifest.sha256,
            "task_order": list(SCREEN_TASKS),
        },
        "reproducibility": reproducibility,
        "graphs": results,
        "boundaries": {
            "llm_calls": boundary_report["completed"]["llm"],
            "grader_calls": boundary_report["completed"]["grader"],
            "kaggle_submissions": boundary_report["completed"]["submission"],
            "network_calls": boundary_report["completed"]["network"],
            "audit": boundary_report,
        },
        "source_hashes": {
            "experience_mcgs.py": sha256_file(Path(sys.modules[ExperienceBoard.__module__].__file__)),
            "shadow_replay.py": sha256_file(Path(__file__)),
        },
    }
    round_trip = json.loads(canonical_bytes(report).decode("utf-8"))
    report["report_reproducible"] = canonical_bytes(report) == canonical_bytes(round_trip)
    return report


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ShadowReplayContractError(f"immutable Shadow Replay output already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _exclusive_write(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(
            json.dumps(
                {"status": "failed_closed", "error": "immutable output already exists"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        report = build_report(args.manifest)
        write_report(args.output, report)
    except (ShadowReplayContractError, OSError, ValueError, TypeError) as exc:
        print(
            json.dumps({"status": "failed_closed", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "graphs": len(report["graphs"]),
                "manifest_sha256": report["manifest"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
