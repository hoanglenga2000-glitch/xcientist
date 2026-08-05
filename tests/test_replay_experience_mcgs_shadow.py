from __future__ import annotations

import builtins
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import replay_experience_mcgs_shadow as replay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _node(task_id: str) -> dict[str, Any]:
    return {
        "exp_id": "EXP000",
        "parent_id": None,
        "branch_type": "Base",
        "task_name": task_id,
        "hypothesis": "draft a public-validation baseline",
        "implementation_summary": "deterministic linear public baseline",
        "code_path": "EXP000/solution.py",
        "cv_score": 0.5,
        "run_success": True,
        "promoted": True,
        "decision": "promote",
        "metric_direction": "maximize",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "wall_seconds": 1.0,
        # Empty historical placeholders are explicitly excluded by the scan.
        "official_rank": None,
        "leaderboard_team_count": None,
        "official_submission_ref": None,
    }


def _build_manifest(root: Path, *, tasks: tuple[str, ...] = replay.SCREEN_TASKS) -> Path:
    rows: list[dict[str, Any]] = []
    for index, task_id in enumerate(tasks):
        graph_dir = root / "frozen" / f"{index:02d}-{task_id}"
        code = graph_dir / "EXP000" / "solution.py"
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text(f"PUBLIC_TASK = {task_id!r}\nVALUE = {index}\n", encoding="utf-8")
        graph = graph_dir / "search_graph.json"
        _write_json(
            graph,
            {
                "task_id": task_id,
                "metric_direction": "maximize",
                "nodes": [_node(task_id)],
            },
        )
        rows.append(
            {
                "task_id": task_id,
                "graph": graph.relative_to(root).as_posix(),
                "sha256": _sha256(graph),
                "code": [
                    {
                        "node_id": "EXP000",
                        "path": "EXP000/solution.py",
                        "sha256": _sha256(code),
                    }
                ],
            }
        )
    manifest = root / "shadow-manifest.json"
    _write_json(manifest, {"schema": replay.MANIFEST_SCHEMA, "tasks": rows})
    return manifest


@pytest.fixture
def frozen_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    return tmp_path, _build_manifest(tmp_path)


def _manifest_row(root: Path, manifest: Path, index: int = 0) -> tuple[dict[str, Any], dict[str, Any], Path]:
    payload = _read_json(manifest)
    row = payload["tasks"][index]
    return payload, row, root / row["graph"]


def _rewrite_graph_and_bind(manifest: Path, root: Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    manifest_payload, row, graph = _manifest_row(root, manifest)
    graph_payload = _read_json(graph)
    mutator(graph_payload)
    _write_json(graph, graph_payload)
    row["sha256"] = _sha256(graph)
    _write_json(manifest, manifest_payload)


def test_exact_six_manifest_replay_is_bound_audited_and_reproducible(
    frozen_contract: tuple[Path, Path],
) -> None:
    _, manifest = frozen_contract
    report = replay.build_report(manifest)

    assert report["status"] == "passed"
    assert report["manifest"]["sha256"] == _sha256(manifest)
    assert report["manifest"]["task_order"] == list(replay.SCREEN_TASKS)
    assert [item["task_id"] for item in report["graphs"]] == list(replay.SCREEN_TASKS)
    assert report["reproducibility"]["independent_full_replays"] == 2
    assert report["reproducibility"]["canonical_bytes_equal"] is True
    assert report["reproducibility"]["first_campaign_replay_sha256"] == report["reproducibility"][
        "second_campaign_replay_sha256"
    ]
    assert all(item["independent_replay_canonical_bytes_equal"] for item in report["graphs"])
    assert report["report_reproducible"] is True

    first_step = report["graphs"][0]["steps"][0]
    assert first_step["board_card_count_before"] == 0
    assert first_step["route_before_execution"] == "Draft"
    assert first_step["executed_operator"] == "Draft"
    assert first_step["next_operator"] == "Improve"
    assert "operator" not in first_step

    boundaries = report["boundaries"]
    assert boundaries["llm_calls"] == 0
    assert boundaries["grader_calls"] == 0
    assert boundaries["kaggle_submissions"] == 0
    assert boundaries["network_calls"] == 0
    assert boundaries["audit"]["zero_external_calls"] is True
    assert boundaries["audit"]["enforcement_cycles"] == 1
    assert boundaries["audit"]["local_file_reads"] > 0
    assert len(boundaries["audit"]["audit_sha256"]) == 64


def test_cli_requires_manifest() -> None:
    with pytest.raises(SystemExit):
        replay.main(["--output", "unused.json"])


def test_cli_accepts_only_exact_six_and_writes_once(
    frozen_contract: tuple[Path, Path],
) -> None:
    root, manifest = frozen_contract
    output = root / "shadow-report.json"
    relative_manifest = manifest.relative_to(root)

    assert replay.main(["--manifest", str(relative_manifest), "--output", str(output)]) == 0
    before = output.read_bytes()
    assert replay.main(["--manifest", str(relative_manifest), "--output", str(output)]) == 2
    assert output.read_bytes() == before


def test_cli_fails_closed_for_five_tasks_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    manifest = _build_manifest(tmp_path, tasks=replay.SCREEN_TASKS[:-1])
    output = tmp_path / "must-not-exist.json"

    assert replay.main(["--manifest", manifest.name, "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["tasks"].pop(), "exactly six"),
        (
            lambda payload: payload["tasks"].__setitem__(5, {**payload["tasks"][5], "task_id": payload["tasks"][0]["task_id"]}),
            "duplicate task_id",
        ),
        (
            lambda payload: payload["tasks"].__setitem__(slice(0, 2), [payload["tasks"][1], payload["tasks"][0]]),
            "task order mismatch",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__("task_id", "wrong-task"),
            "task order mismatch",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__("sha256", ""),
            "required lowercase 64-hex",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__("sha256", "A" * 64),
            "required lowercase 64-hex",
        ),
    ],
    ids=["missing", "duplicate", "order", "task-id", "missing-hash", "invalid-hash"],
)
def test_manifest_rejects_missing_duplicate_order_task_and_hash_contracts(
    frozen_contract: tuple[Path, Path],
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    _, manifest = frozen_contract
    payload = _read_json(manifest)
    mutation(payload)
    _write_json(manifest, payload)

    with pytest.raises(replay.ShadowReplayContractError, match=match):
        replay.build_report(manifest)


def test_graph_hash_drift_fails_closed(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract
    _, _, graph = _manifest_row(root, manifest)
    graph.write_bytes(graph.read_bytes() + b" ")

    with pytest.raises(replay.ShadowReplayContractError, match="SHA-256 mismatch"):
        replay.build_report(manifest)


def test_graph_task_identity_drift_fails_closed(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract

    def mutate(payload: dict[str, Any]) -> None:
        payload["nodes"][0]["task_name"] = "wrong-task"

    _rewrite_graph_and_bind(manifest, root, mutate)
    with pytest.raises(replay.ShadowReplayContractError, match="task identity mismatch"):
        replay.build_report(manifest)


def test_graph_path_traversal_fails_closed(frozen_contract: tuple[Path, Path]) -> None:
    _, manifest = frozen_contract
    payload = _read_json(manifest)
    payload["tasks"][0]["graph"] = "../outside.json"
    _write_json(manifest, payload)

    with pytest.raises(replay.ShadowReplayContractError, match="traversal"):
        replay.build_report(manifest)


def test_code_path_traversal_fails_closed(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract

    def mutate(payload: dict[str, Any]) -> None:
        payload["nodes"][0]["code_path"] = "../escape.py"

    _rewrite_graph_and_bind(manifest, root, mutate)
    payload = _read_json(manifest)
    payload["tasks"][0]["code"][0]["path"] = "../escape.py"
    _write_json(manifest, payload)

    with pytest.raises(replay.ShadowReplayContractError, match="traversal"):
        replay.build_report(manifest)


def test_missing_code_fails_closed_without_summary_fallback(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract
    payload, row, graph = _manifest_row(root, manifest)
    code = graph.parent / row["code"][0]["path"]
    code.unlink()

    with pytest.raises(replay.ShadowReplayContractError, match="does not exist"):
        replay.build_report(manifest)
    assert payload["tasks"][0]["code"][0]["node_id"] == "EXP000"


def test_code_hash_drift_fails_closed(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract
    _, row, graph = _manifest_row(root, manifest)
    code = graph.parent / row["code"][0]["path"]
    code.write_text("VALUE = 999\n", encoding="utf-8")

    with pytest.raises(replay.ShadowReplayContractError, match="SHA-256 mismatch"):
        replay.build_report(manifest)


def _make_hard_link(link: Path, target: Path) -> None:
    os.link(target, link)


@pytest.mark.parametrize("target_kind", ["graph", "code"])
def test_graph_and_code_links_fail_closed(
    frozen_contract: tuple[Path, Path],
    target_kind: str,
) -> None:
    root, manifest = frozen_contract
    payload, row, graph = _manifest_row(root, manifest)
    if target_kind == "graph":
        real = graph.with_name("real-search-graph.json")
        graph.replace(real)
        _make_hard_link(graph, real)
    else:
        code = graph.parent / row["code"][0]["path"]
        real = code.with_name("real-solution.py")
        code.replace(real)
        _make_hard_link(code, real)
    _write_json(manifest, payload)

    with pytest.raises(replay.ShadowReplayContractError, match="link/reparse"):
        replay.build_report(manifest)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_symlink_equivalent_directory_junction_fails_closed(
    frozen_contract: tuple[Path, Path],
) -> None:
    root, manifest = frozen_contract
    _, _, graph = _manifest_row(root, manifest)
    linked_directory = graph.parent
    real_directory = linked_directory.with_name(f"{linked_directory.name}-real")
    linked_directory.replace(real_directory)
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(linked_directory), str(real_directory)],
        check=False,
        capture_output=True,
    )
    assert created.returncode == 0, repr(created.stderr or created.stdout)
    try:
        with pytest.raises(replay.ShadowReplayContractError, match="link/reparse"):
            replay.build_report(manifest)
    finally:
        os.rmdir(linked_directory)


@pytest.mark.parametrize("location", ["prompt", "code", "error", "graph"])
def test_raw_prompt_code_error_and_graph_reject_restricted_material(
    frozen_contract: tuple[Path, Path],
    location: str,
) -> None:
    root, manifest = frozen_contract
    if location == "code":
        payload, row, graph = _manifest_row(root, manifest)
        code = graph.parent / row["code"][0]["path"]
        code.write_text("PRIVATE_GRADER_FEEDBACK = 'secret'\n", encoding="utf-8")
        row["code"][0]["sha256"] = _sha256(code)
        _write_json(manifest, payload)
    else:

        def mutate(payload: dict[str, Any]) -> None:
            if location == "prompt":
                payload["nodes"][0]["hypothesis"] = "reuse private grader feedback"
            elif location == "error":
                payload["nodes"][0]["run_success"] = False
                payload["nodes"][0]["cv_score"] = None
                payload["nodes"][0]["promotion_reason"] = "private score was rejected"
            else:
                payload["private_feedback"] = {"score": 0.9}

        _rewrite_graph_and_bind(manifest, root, mutate)

    with pytest.raises(replay.ShadowReplayContractError, match="restricted evaluation"):
        replay.build_report(manifest)


def test_empty_restricted_historical_placeholders_are_scanned_but_excluded(
    frozen_contract: tuple[Path, Path],
) -> None:
    _, manifest = frozen_contract
    report = replay.build_report(manifest)

    assert all(
        item["public_only_scan"]["empty_restricted_fields_excluded"] == 3
        for item in report["graphs"]
    )
    cards = json.dumps(report["graphs"], ensure_ascii=False).lower()
    assert "official_rank" not in cards
    assert "leaderboard_team_count" not in cards
    assert "official_submission_ref" not in cards


@pytest.mark.parametrize(
    ("boundary", "method"),
    [
        ("llm", "llm_call"),
        ("grader", "grader_call"),
        ("submission", "submission_call"),
        ("network", "network_call"),
    ],
)
def test_explicit_external_call_boundaries_intercept_and_count(
    boundary: str,
    method: str,
) -> None:
    audit = replay.BoundaryAudit()
    with pytest.raises(replay.BoundaryViolation, match=boundary):
        getattr(audit, method)("unit-test")

    evidence = audit.to_dict()
    assert evidence["attempted"][boundary] == 1
    assert evidence["blocked"][boundary] == 1
    assert evidence["completed"][boundary] == 0
    assert evidence["zero_external_calls"] is False


def test_socket_subprocess_and_provider_import_escape_hatches_are_intercepted() -> None:
    network_audit = replay.BoundaryAudit()
    with network_audit.enforce(), pytest.raises(replay.BoundaryViolation, match="network"):
        socket.create_connection(("127.0.0.1", 9))
    assert network_audit.attempted["network"] == 1

    submission_audit = replay.BoundaryAudit()
    with submission_audit.enforce(), pytest.raises(replay.BoundaryViolation, match="submission"):
        subprocess.run(["kaggle", "submit"], check=False)
    assert submission_audit.attempted["submission"] == 1

    llm_audit = replay.BoundaryAudit()
    with llm_audit.enforce(), pytest.raises(replay.BoundaryViolation, match="llm"):
        builtins.__import__("openai")
    assert llm_audit.attempted["llm"] == 1


def test_missing_usage_remains_a_fail_closed_report(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract

    def mutate(payload: dict[str, Any]) -> None:
        node = payload["nodes"][0]
        node.pop("prompt_tokens")
        node.pop("completion_tokens")
        node.pop("wall_seconds")

    _rewrite_graph_and_bind(manifest, root, mutate)
    report = replay.build_report(manifest)

    assert report["status"] == "failed_closed"
    assert report["checks"]["recorded_usage_complete"] is False


def test_first_historical_operator_must_be_draft(frozen_contract: tuple[Path, Path]) -> None:
    root, manifest = frozen_contract

    def mutate(payload: dict[str, Any]) -> None:
        payload["nodes"][0]["operator"] = "Improve"

    _rewrite_graph_and_bind(manifest, root, mutate)
    with pytest.raises(replay.ShadowReplayContractError, match="first historical execution must be Draft"):
        replay.build_report(manifest)
