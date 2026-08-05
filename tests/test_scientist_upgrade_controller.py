from __future__ import annotations

import copy
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from xsci import scientist_upgrade_controller as controller
from xsci.scientist_upgrade_controller import (
    CHAMPION_REF,
    EvaluatorContract,
    UpgradeControllerError,
    promote_upgrade_campaign,
    rollback_upgrade_campaign,
    run_upgrade_campaign,
)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "EvoMind Test")
    _git(root, "config", "user.email", "evomind-test@invalid.local")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "model.py").write_text("SCORE = 1.0\n", encoding="utf-8")
    (root / "tests" / "evaluator.py").write_text("EVALUATOR_VERSION = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(".xsci/\n", encoding="utf-8")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "baseline")
    return root


@pytest.fixture
def evaluator() -> EvaluatorContract:
    return EvaluatorContract(
        evaluator_id="fixture-evaluator-v1",
        evaluator_files=("tests/evaluator.py",),
        commands=("python -m pytest",),
        primary_metric="quality",
        required_metrics=("quality",),
        isolation_level="test_fixture",
        seed=17,
    )


def _score_callback(worktree: Path, _evaluator: EvaluatorContract, _label: str) -> dict[str, object]:
    match = re.search(r"SCORE\s*=\s*([0-9.]+)", (worktree / "src" / "model.py").read_text(encoding="utf-8"))
    assert match is not None
    score = float(match.group(1))
    return {"passed": True, "score": score, "metrics": {"quality": score}}


def _modification_patch(repository: Path, output: Path, relative_path: str, content: str) -> Path:
    target = repository / relative_path
    original = target.read_bytes()
    target.write_text(content, encoding="utf-8", newline="\n")
    patch = _git(repository, "diff", "--binary", "HEAD", "--", relative_path)
    target.write_bytes(original)
    assert not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    output.write_text(patch + "\n", encoding="utf-8", newline="\n")
    return output


def _rename_patch(repository: Path, output: Path, source: str, destination: str) -> Path:
    _git(repository, "mv", "--", source, destination)
    patch = _git(repository, "diff", "--binary", "HEAD")
    _git(repository, "mv", "--", destination, source)
    assert not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    output.write_text(patch + "\n", encoding="utf-8", newline="\n")
    return output


def _campaign(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
    *,
    scores: tuple[float, float] = (1.1, 1.2),
) -> dict[str, object]:
    patches = [
        _modification_patch(
            repository,
            tmp_path / f"candidate-{index}.diff",
            "src/model.py",
            f"SCORE = {score}\nCANDIDATE = {index}\n",
        )
        for index, score in enumerate(scores, start=1)
    ]
    return run_upgrade_campaign(
        repository,
        candidate_patches=patches,
        evaluator=evaluator,
        evaluator_callback=_score_callback,
    )


def _activation_ok(_repository: Path, _commit: str, tree: str) -> dict[str, object]:
    return {"passed": True, "runtime_tree_sha": tree}


def _new_commit(repository: Path, parent: str, message: str) -> str:
    tree = _git(repository, "rev-parse", f"{parent}^{{tree}}")
    return _git(repository, "commit-tree", tree, "-p", parent, "-m", message)


def test_evaluator_digest_excludes_lock_timestamp_and_detects_contract_drift(
    repository: Path,
    evaluator: EvaluatorContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(controller, "_utc_seconds", lambda: "2026-07-15T00:00:00Z")
    first = controller._evaluator_lock(repository, base, evaluator)
    monkeypatch.setattr(controller, "_utc_seconds", lambda: "2026-07-15T00:00:01Z")
    second = controller._evaluator_lock(repository, base, evaluator)

    assert first["locked_at"] != second["locked_at"]
    assert first["evaluator_digest_sha256"] == second["evaluator_digest_sha256"]
    manifest = {"base_commit": base, "evaluator": asdict(evaluator), "evaluator_lock": first}
    controller._verify_locked_evaluator(repository, manifest)

    drifted = copy.deepcopy(manifest)
    drifted["evaluator"]["commands"] = ["python -m compileall"]
    with pytest.raises(UpgradeControllerError, match="frozen evaluator digest changed"):
        controller._verify_locked_evaluator(repository, drifted)


def test_protected_rename_is_rejected_without_touching_main_worktree(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    protected = _rename_patch(
        repository,
        tmp_path / "protected-rename.diff",
        "tests/evaluator.py",
        "src/evaluator.py",
    )
    allowed = _modification_patch(repository, tmp_path / "allowed.diff", "src/model.py", "SCORE = 1.1\n")

    manifest = run_upgrade_campaign(
        repository,
        candidate_patches=(protected, allowed),
        evaluator=evaluator,
        evaluator_callback=_score_callback,
    )

    rejected = next(item for item in manifest["candidates"] if item["status"] == "rejected_scope")
    assert rejected["protected_paths_modified"] == ["tests/evaluator.py"]
    assert rejected["evaluator_files_modified"] is True
    assert set(rejected["changed_paths"]) == {"src/evaluator.py", "tests/evaluator.py"}
    assert not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")


def test_selection_requires_real_strict_improvement_and_keeps_worktree_clean(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator, scores=(1.0, 1.1))

    equal = next(item for item in manifest["candidates"] if item["evaluation"]["score"] == 1.0)
    improved = next(item for item in manifest["candidates"] if item["evaluation"]["score"] == 1.1)
    assert equal["strictly_improves_baseline"] is False
    assert improved["strictly_improves_baseline"] is True
    assert manifest["selection"]["candidate_id"] == improved["candidate_id"]
    assert manifest["main_worktree_modified"] is False
    assert not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")


def test_promotion_cas_refuses_concurrent_champion_advance(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    base = str(manifest["base_commit"])
    _git(repository, "update-ref", CHAMPION_REF, base)
    concurrent = _new_commit(repository, base, "concurrent champion")
    original_rollback_test = controller._rollback_ref_test

    def advance_during_promotion(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_rollback_test(*args, **kwargs)
        _git(repository, "update-ref", CHAMPION_REF, concurrent, base)
        return result

    monkeypatch.setattr(controller, "_rollback_ref_test", advance_during_promotion)
    activation_called = False

    def activation(*_args: object) -> dict[str, object]:
        nonlocal activation_called
        activation_called = True
        return {"passed": True}

    with pytest.raises(UpgradeControllerError, match="compare-and-swap failed"):
        promote_upgrade_campaign(
            repository,
            manifest["manifest_path"],
            human_approved=True,
            activation_callback=activation,
        )

    assert _git(repository, "rev-parse", CHAMPION_REF) == concurrent
    assert activation_called is False


def test_promotion_rejects_tampered_campaign_attestation(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    path = Path(str(manifest["manifest_path"]))
    tampered = controller._read_json(path)
    tampered["selection"]["strictly_improves_baseline"] = False
    controller._write_json(path, tampered)

    with pytest.raises(UpgradeControllerError, match="manifest attestation is invalid"):
        promote_upgrade_campaign(
            repository,
            path,
            human_approved=True,
            activation_callback=_activation_ok,
        )


def test_promotion_rejects_ambiguous_campaign_json(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    path = Path(str(manifest["manifest_path"]))
    path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")

    with pytest.raises(UpgradeControllerError, match="campaign manifest is unreadable"):
        promote_upgrade_campaign(
            repository,
            path,
            human_approved=True,
            activation_callback=_activation_ok,
        )


def test_promotion_recomputes_strict_improvement_from_attested_evaluations(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    path = Path(str(manifest["manifest_path"]))
    tampered = controller._read_json(path)
    selected_id = tampered["selection"]["candidate_id"]
    selected = next(item for item in tampered["candidates"] if item["candidate_id"] == selected_id)
    selected["evaluation"]["score"] = 0.5
    selected["evaluation"]["metrics"]["quality"] = 0.5
    selected["evaluation"]["evaluation_digest_sha256"] = controller._digest({
        key: value
        for key, value in selected["evaluation"].items()
        if key != "evaluation_digest_sha256"
    })
    selected["evaluation_digest_sha256"] = selected["evaluation"]["evaluation_digest_sha256"]
    tampered["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": controller._manifest_attestation(tampered),
    }
    controller._write_json(path, tampered)

    with pytest.raises(UpgradeControllerError, match="strict improvement evidence is invalid"):
        promote_upgrade_campaign(
            repository,
            path,
            human_approved=True,
            activation_callback=_activation_ok,
        )


def test_failed_activation_restores_previous_champion(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    base = str(manifest["base_commit"])
    _git(repository, "update-ref", CHAMPION_REF, base)

    def fail_activation(_repository: Path, commit: str, _tree: str) -> dict[str, object]:
        assert _git(repository, "rev-parse", CHAMPION_REF) == commit
        return {"passed": False, "runtime_tree_sha": "0" * 40}

    with pytest.raises(UpgradeControllerError, match="previous champion restored"):
        promote_upgrade_campaign(
            repository,
            manifest["manifest_path"],
            human_approved=True,
            activation_callback=fail_activation,
        )

    assert _git(repository, "rev-parse", CHAMPION_REF) == base
    failed = controller._read_json(Path(str(manifest["manifest_path"])))
    assert failed["status"] == "rolled_back_after_failed_activation"
    assert failed["rollback"]["passed"] is True


def test_explicit_rollback_is_idempotent(
    repository: Path,
    tmp_path: Path,
    evaluator: EvaluatorContract,
) -> None:
    manifest = _campaign(repository, tmp_path, evaluator)
    base = str(manifest["base_commit"])
    _git(repository, "update-ref", CHAMPION_REF, base)
    promoted = promote_upgrade_campaign(
        repository,
        manifest["manifest_path"],
        human_approved=True,
        activation_callback=_activation_ok,
    )
    rollback_calls = 0

    def activate_previous(_repository: Path, _commit: str, tree: str) -> dict[str, object]:
        nonlocal rollback_calls
        rollback_calls += 1
        return {"passed": True, "runtime_tree_sha": tree}

    first = rollback_upgrade_campaign(
        repository,
        promoted["manifest_path"],
        activation_callback=activate_previous,
    )
    second = rollback_upgrade_campaign(
        repository,
        promoted["manifest_path"],
        activation_callback=activate_previous,
    )

    assert first == second
    assert second["status"] == "rolled_back"
    assert rollback_calls == 1
    assert _git(repository, "rev-parse", CHAMPION_REF) == base
    assert not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
