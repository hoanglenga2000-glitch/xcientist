"""Contracts for the reviewed, frozen, exactly-once SIIM private grader."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import run_siim_private_grader_once as grader_once


def test_remote_grader_environment_must_stay_inside_operator_root(
    monkeypatch, tmp_path: Path
):
    allowed = tmp_path / "operator_remote_root"
    allowed.mkdir()
    for index, name in enumerate(grader_once.DEDICATED_REMOTE_ENVIRONMENT_KEYS):
        path = allowed / "environment" / f"{index:02d}_{name.lower()}"
        path.mkdir(parents=True)
        monkeypatch.setenv(name, str(path))

    validated = grader_once.validate_dedicated_process_environment(allowed)
    assert set(validated) == set(grader_once.DEDICATED_REMOTE_ENVIRONMENT_KEYS)

    monkeypatch.setenv("TMPDIR", str(tmp_path / "outside"))
    with pytest.raises(grader_once.PrivateGraderOnceError, match="TMPDIR"):
        grader_once.validate_dedicated_process_environment(allowed)


def test_remote_grader_pins_cwd_umask_and_manager_identity_gate():
    import inspect

    source = inspect.getsource(grader_once.dispatch_remote)
    assert "manager.connect()" in source
    assert "cd --" in source
    assert "umask 077" in source
    assert "manager.verify_remote_path_chain" in source


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(grader_once.json_bytes(payload))
    return path


def _make_candidate(tmp_path: Path, *, run_id: str = "siim_private_grader_test"):
    project_root = tmp_path / "project"
    run_dir = project_root / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    submission = run_dir / "submission.csv"
    with submission.open("w", encoding="utf-8", newline="") as handle:
        handle.write("image_name,target\n")
        for index in range(grader_once.EXPECTED_TEST_ROWS):
            handle.write(f"ISIC_{index:07d},{(index % 101) / 100:.2f}\n")
    metrics = _write_json(
        run_dir / "metrics.json",
        {"run_id": run_id, "roc_auc": 0.91, "private_grader_executed": False},
    )
    artifact_hashes = {
        "submission.csv": grader_once.sha256_file(submission),
        "metrics.json": grader_once.sha256_file(metrics),
    }
    review = _write_json(
        run_dir / "review.json",
        {
            "schema": "evomind.siim.independent_review.v1",
            "run_id": run_id,
            "status": "review_passed",
            "checks": {name: True for name in grader_once.EXPECTED_REVIEW_CHECKS},
            "artifact_hashes": artifact_hashes,
        },
    )
    freeze = _write_json(
        run_dir / "candidate_freeze.json",
        {
            "schema": grader_once.CANDIDATE_FREEZE_SCHEMA,
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "frozen_at": "2026-07-29T15:00:00+00:00",
            "configuration_sha256": "a" * 64,
            "artifacts": [
                {
                    "path": relative,
                    "sha256": digest,
                    "bytes": (run_dir / relative).stat().st_size,
                }
                for relative, digest in artifact_hashes.items()
            ],
            "private_grader_execution_count_before_freeze": 0,
            "tuning_closed": True,
            "official_submission": "forbidden",
        },
    )
    return project_root, run_dir, freeze, review, submission


def _result_payload(prepared, request, *, score: float = 0.923) -> dict:
    return {
        "schema": grader_once.RESULT_SCHEMA,
        "run_id": prepared.run_id,
        "competition_id": grader_once.COMPETITION_ID,
        "status": "passed",
        "execution_id": request["request_id"],
        "execution_index": 1,
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "executed_after_freeze": True,
        "official_mlebench_grader_executed": True,
        "metric": "roc_auc",
        "direction": "maximize",
        "score": score,
        "mle_private_grader_score": score,
        "private_answers_sha256": "d" * 64,
        "raw_grader_sha256": "e" * 64,
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
        "completed_at": "2026-07-29T16:00:00+00:00",
    }


def _dispatch_evidence(prepared, request, *, score: float = 0.923):
    result_bytes = grader_once.json_bytes(_result_payload(prepared, request, score=score))
    claim_sha256 = "c" * 64
    receipt = {
        "schema": grader_once.RECEIPT_SCHEMA,
        "run_id": prepared.run_id,
        "competition_id": grader_once.COMPETITION_ID,
        "status": "passed",
        "execution_id": request["request_id"],
        "execution_count": 1,
        "candidate_freeze_sha256": prepared.candidate_freeze_sha256,
        "review_sha256": prepared.review_sha256,
        "submission_sha256": prepared.submission_sha256,
        "claim_sha256": claim_sha256,
        "result_sha256": grader_once.sha256_bytes(result_bytes),
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "completed_at": "2026-07-29T16:00:00+00:00",
    }
    return grader_once.RemoteDispatchEvidence(
        result_bytes=result_bytes,
        receipt_bytes=grader_once.json_bytes(receipt),
        claim_sha256=claim_sha256,
        remote_root=f"/fixture/{prepared.run_id}",
        idempotent_reuse=False,
    )


def _stage_remote_request(tmp_path: Path):
    project_root, _run_dir, _freeze, _review, _submission = _make_candidate(tmp_path)
    prepared = grader_once.prepare_candidate(project_root, "siim_private_grader_test")
    allowed = tmp_path / "allowed"
    request_dir = allowed / grader_once.REMOTE_PRIVATE_GRADER_PARENT_RELATIVE / prepared.run_id
    request_dir.mkdir(parents=True)
    runner = request_dir / "runner.py"
    shutil.copy2(Path(grader_once.__file__), runner)
    request = grader_once.build_request(
        run_id=prepared.run_id,
        candidate_freeze_sha256=prepared.candidate_freeze_sha256,
        review_sha256=prepared.review_sha256,
        submission_sha256=prepared.submission_sha256,
        runner_sha256=grader_once.sha256_file(runner),
    )
    _write_json(request_dir / "request.json", request)
    shutil.copy2(prepared.freeze_path, request_dir / "candidate_freeze.json")
    shutil.copy2(prepared.review_path, request_dir / "review.json")
    shutil.copy2(prepared.submission_path, request_dir / "submission.csv")
    data_root = allowed / "mlebench_official_data"
    official_source = allowed / "mle-bench"
    bundle_root = allowed / f"siim_{grader_once.JOB_TAG}" / "bundles" / ("b" * 64)
    for path in (data_root, official_source, bundle_root):
        path.mkdir(parents=True)
    answers = (
        data_root
        / grader_once.COMPETITION_ID
        / "prepared"
        / "private"
        / "test.csv"
    )
    answers.parent.mkdir(parents=True)
    shutil.copy2(prepared.submission_path, answers)
    official_package = official_source / "mlebench"
    official_package.mkdir()
    (official_package / "__init__.py").write_text("\n", encoding="utf-8")
    (official_package / "grade.py").write_text(
        "def grade_csv(*args, **kwargs):\n    raise AssertionError('fixture only')\n",
        encoding="utf-8",
    )
    (official_package / "registry.py").write_text(
        "class Registry:\n    pass\n", encoding="utf-8"
    )
    bundle_grader = bundle_root / "src" / "research_os" / "mlebench_phase_a.py"
    bundle_grader.parent.mkdir(parents=True)
    bundle_grader.write_text("# immutable fixture grader module\n", encoding="utf-8")
    return prepared, request, allowed, request_dir, runner, data_root, official_source, bundle_root


def _assert_readiness_left_no_execution_state(request_dir: Path) -> None:
    for name in (
        "execution_claim.json",
        "execution_receipt.json",
        "execution_failure.json",
        "result.json",
        grader_once.REMOTE_RAW_GRADER_NAME,
        ".grader_lock",
    ):
        assert not (request_dir / name).exists(), name
    assert not list(request_dir.glob("*ledger*"))


def _private_answers_path(data_root: Path) -> Path:
    return (
        data_root
        / grader_once.COMPETITION_ID
        / "prepared"
        / "private"
        / "test.csv"
    )


def test_candidate_requires_passed_review_and_immutable_freeze(tmp_path):
    project_root, run_dir, _freeze, _review, _submission = _make_candidate(tmp_path)
    prepared = grader_once.prepare_candidate(project_root, "siim_private_grader_test")
    assert prepared.submission_sha256
    assert prepared.request["official_submission"] == "forbidden"
    assert prepared.request["grader_feedback_policy"] == "terminal_delivery_only_no_tuning"

    review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
    review["checks"]["private_grader_not_executed"] = False
    _write_json(run_dir / "review.json", review)
    with pytest.raises(grader_once.PrivateGraderOnceError, match="private_grader_not_executed"):
        grader_once.prepare_candidate(project_root, "siim_private_grader_test")


def test_candidate_rejects_frozen_submission_drift(tmp_path):
    project_root, _run_dir, _freeze, _review, submission = _make_candidate(tmp_path)
    with submission.open("a", encoding="utf-8") as handle:
        handle.write("ISIC_extra,0.5\n")
    with pytest.raises(grader_once.PrivateGraderOnceError, match="frozen artifact changed"):
        grader_once.prepare_candidate(project_root, "siim_private_grader_test")


def test_remote_worker_grades_once_and_reuses_immutable_result(tmp_path):
    (
        prepared,
        request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    calls = []

    def fake_grader(submission_path, competition_id, supplied_data_root, **kwargs):
        calls.append((submission_path, competition_id, supplied_data_root, kwargs))
        return {
            "status": "passed",
            "official_mlebench_grader_executed": True,
            "score": 0.927,
            "provenance": {
                "submission_sha256": prepared.submission_sha256,
                "answers_sha256": grader_once.sha256_file(
                    data_root
                    / grader_once.COMPETITION_ID
                    / "prepared"
                    / "private"
                    / "test.csv"
                ),
                "answers_path": str(
                    data_root
                    / grader_once.COMPETITION_ID
                    / "prepared"
                    / "private"
                    / "test.csv"
                ),
            },
        }

    first = grader_once.remote_grade_once(
        request_dir,
        data_root=data_root,
        official_source_root=official_source,
        allowed_root=allowed,
        bundle_root=bundle_root,
        runner_path=runner,
        grader=fake_grader,
    )
    second = grader_once.remote_grade_once(
        request_dir,
        data_root=data_root,
        official_source_root=official_source,
        allowed_root=allowed,
        bundle_root=bundle_root,
        runner_path=runner,
        grader=fake_grader,
    )
    assert first["status"] == "passed"
    assert second["status"] == "reused_existing_result"
    assert len(calls) == 1
    result = json.loads((request_dir / "result.json").read_text(encoding="utf-8"))
    assert result["execution_id"] == request["request_id"]
    assert result["candidate_freeze_sha256"] == prepared.candidate_freeze_sha256
    assert result["submission_sha256"] == prepared.submission_sha256
    assert result["private_labels_exported"] is False
    assert result["feedback_used_for_tuning"] is False
    assert result["official_submission_executed"] is False
    assert "answers_path" not in result
    assert str(data_root) not in json.dumps(result)
    assert (request_dir / grader_once.REMOTE_RAW_GRADER_NAME).is_file()


@pytest.mark.parametrize(
    ("state", "expected_error"),
    [("missing", "missing private answers input"), ("empty", "must not be empty")],
)
def test_remote_readiness_rejects_missing_or_empty_private_answers_without_claim(
    tmp_path, state, expected_error
):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    answers = _private_answers_path(data_root)
    if state == "missing":
        answers.unlink()
    else:
        answers.write_bytes(b"")

    with pytest.raises(grader_once.PrivateGraderOnceError, match=expected_error):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=lambda *_args, **_kwargs: pytest.fail("grader must not execute"),
        )
    _assert_readiness_left_no_execution_state(request_dir)


def test_remote_readiness_rejects_symlinked_private_answers_without_claim(
    tmp_path, monkeypatch
):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    answers = _private_answers_path(data_root)
    real_answers = answers.with_name("real-private-answers.csv")
    answers.replace(real_answers)
    try:
        answers.symlink_to(real_answers.name)
    except (NotImplementedError, OSError):
        real_answers.replace(answers)
        original_lstat = Path.lstat

        def report_answers_as_symlink(path):
            metadata = original_lstat(path)
            if path == answers:
                return type(
                    "SymlinkMetadata",
                    (),
                    {
                        "st_mode": grader_once.stat.S_IFLNK,
                        "st_size": metadata.st_size,
                    },
                )()
            return metadata

        monkeypatch.setattr(Path, "lstat", report_answers_as_symlink)

    with pytest.raises(grader_once.PrivateGraderOnceError, match="must not be a symbolic link"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=lambda *_args, **_kwargs: pytest.fail("grader must not execute"),
        )
    _assert_readiness_left_no_execution_state(request_dir)


@pytest.mark.parametrize("failure", ["row_count", "schema"])
def test_remote_readiness_rejects_private_answer_shape_without_claim(tmp_path, failure):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    answers = _private_answers_path(data_root)
    if failure == "row_count":
        lines = answers.read_text(encoding="utf-8").splitlines()
        answers.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        expected_error = "row count is not 4,142"
    else:
        text = answers.read_text(encoding="utf-8")
        answers.write_text(text.replace("image_name,target", "image_id,target", 1), encoding="utf-8")
        expected_error = "columns are not image_name,target"

    with pytest.raises(grader_once.PrivateGraderOnceError, match=expected_error):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=lambda *_args, **_kwargs: pytest.fail("grader must not execute"),
        )
    _assert_readiness_left_no_execution_state(request_dir)


def test_remote_readiness_rejects_private_answer_id_misalignment_without_claim(tmp_path):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    answers = _private_answers_path(data_root)
    text = answers.read_text(encoding="utf-8")
    answers.write_text(text.replace("ISIC_0000000", "ISIC_mismatch", 1), encoding="utf-8")

    with pytest.raises(grader_once.PrivateGraderOnceError, match="IDs/order do not align"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=lambda *_args, **_kwargs: pytest.fail("grader must not execute"),
        )
    _assert_readiness_left_no_execution_state(request_dir)


def test_remote_readiness_imports_pinned_grader_without_claim(tmp_path, monkeypatch):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    for name in ("mlebench.grade", "mlebench.registry", "mlebench"):
        monkeypatch.delitem(grader_once.sys.modules, name, raising=False)

    grader, answers, answers_sha256 = grader_once._prepare_remote_grader_readiness(
        request_dir,
        data_root=data_root,
        official_source_root=official_source,
        allowed_root=allowed,
        bundle_root=bundle_root,
        runner_path=runner,
        grader=None,
    )
    assert callable(grader)
    assert answers == _private_answers_path(data_root).resolve()
    assert answers_sha256 == grader_once.sha256_file(answers)
    _assert_readiness_left_no_execution_state(request_dir)


def test_remote_readiness_rejects_official_grader_import_failure_without_claim(
    tmp_path, monkeypatch
):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    original_import_module = grader_once.importlib.import_module

    def fail_official_import(name, *args, **kwargs):
        if name == "mlebench.grade":
            raise ImportError("fixture missing upstream grader dependency")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(grader_once.importlib, "import_module", fail_official_import)
    with pytest.raises(grader_once.PrivateGraderOnceError, match="not importable"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
        )
    _assert_readiness_left_no_execution_state(request_dir)


@pytest.mark.parametrize("escaped_path", ["runner", "bundle", "data"])
def test_remote_readiness_rejects_roots_outside_allowed_range_without_claim(
    tmp_path, escaped_path
):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if escaped_path == "runner":
        escaped_runner = outside / "runner.py"
        shutil.copy2(runner, escaped_runner)
        runner = escaped_runner
    elif escaped_path == "bundle":
        bundle_root = outside / ("f" * 64)
        bundle_root.mkdir()
    else:
        data_root = outside / "mlebench_official_data"
        data_root.mkdir()

    with pytest.raises(grader_once.PrivateGraderOnceError, match="escaped"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=lambda *_args, **_kwargs: pytest.fail("grader must not execute"),
        )
    _assert_readiness_left_no_execution_state(request_dir)


def test_remote_incomplete_claim_fails_closed_without_second_grade(tmp_path):
    (
        _prepared,
        _request,
        allowed,
        request_dir,
        runner,
        data_root,
        official_source,
        bundle_root,
    ) = _stage_remote_request(tmp_path)
    calls = []

    def failing_grader(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("fixture grader interruption")

    with pytest.raises(RuntimeError, match="fixture grader interruption"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=failing_grader,
        )
    with pytest.raises(grader_once.PrivateGraderOnceError, match="blocks retry"):
        grader_once.remote_grade_once(
            request_dir,
            data_root=data_root,
            official_source_root=official_source,
            allowed_root=allowed,
            bundle_root=bundle_root,
            runner_path=runner,
            grader=failing_grader,
        )
    assert len(calls) == 1
    assert (request_dir / "execution_claim.json").is_file()
    assert not (request_dir / "result.json").exists()


def test_local_orchestrator_stages_ingress_and_repeated_call_skips_dispatch(tmp_path):
    project_root, _run_dir, _freeze, _review, _submission = _make_candidate(tmp_path)
    calls = []

    def fake_dispatch(prepared, request, control_dir):
        calls.append((prepared, request, control_dir))
        return _dispatch_evidence(prepared, request)

    first = grader_once.run_private_grader_once(
        project_root,
        "siim_private_grader_test",
        dispatcher=fake_dispatch,
    )
    second = grader_once.run_private_grader_once(
        project_root,
        "siim_private_grader_test",
        dispatcher=fake_dispatch,
    )
    assert first["status"] == "passed"
    assert second["status"] == "reused_existing_result"
    assert first["execution_count"] == second["execution_count"] == 1
    assert len(calls) == 1
    ingress = Path(first["ingress_path"])
    ledger = json.loads(Path(first["ledger_path"]).read_text(encoding="utf-8"))
    assert ingress.is_file()
    assert ledger["candidate_freeze_sha256"] == calls[0][0].candidate_freeze_sha256
    assert ledger["submission_sha256"] == calls[0][0].submission_sha256
    assert ledger["feedback_used_for_tuning"] is False
    assert ledger["official_submission_executed"] is False
    assert ledger["signals_sent"] == 0


def test_existing_request_with_different_candidate_fails_closed(tmp_path):
    project_root, run_dir, _freeze, _review, _submission = _make_candidate(tmp_path)
    prepared = grader_once.prepare_candidate(project_root, "siim_private_grader_test")
    control = (
        project_root
        / "workspace"
        / "hpc"
        / f"{grader_once.JOB_TAG}_siim_private_grader"
        / prepared.run_id
    )
    control.mkdir(parents=True)
    conflicting = dict(prepared.request)
    conflicting["submission_sha256"] = "f" * 64
    _write_json(control / "request.json", conflicting)
    with pytest.raises(grader_once.PrivateGraderOnceError, match="conflicts on submission_sha256"):
        grader_once.run_private_grader_once(
            project_root,
            prepared.run_id,
            dispatcher=lambda *_args: pytest.fail("dispatcher must not run"),
        )
    assert not (run_dir / "private_grader.json").exists()


def test_source_has_no_training_submission_or_signal_execution_path():
    source = Path(grader_once.__file__).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "kaggle competitions submit" not in source.lower()
    assert "resume_siim_hpc_research" not in source
    assert '"feedback_used_for_tuning": False' in source
    assert '"official_submission_executed": False' in source
    assert '"signals_sent": 0' in source
