from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_job89941_may2022_cpu_diagnostic.py"


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_job89941_may2022", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(tmp_path: Path) -> tuple[Path, dict]:
    relative_sources = (
        "scripts/diagnose_may2022_cpu_feature_model.py",
        "scripts/mlebench_medal_recovery_adapters.py",
        "scripts/mlebench_wave2_adapters.py",
        "scripts/russian_transliteration.py",
        "scripts/run_mlebench_lite_wave0.py",
        "src/research_os/mlebench_phase_a.py",
    )
    source_records = []
    for relative in relative_sources:
        source = tmp_path.joinpath(*PurePosixPath(relative).parts)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# frozen {relative}\n", encoding="utf-8")
        source_records.append(
            {
                "local_path": str(source.resolve()),
                "relative_path": relative,
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    root = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    payload = {
        "schema": "evomind.mlebench.may2022_cpu_feature_diagnostic_plan.v1",
        "created_at": "2026-07-27T00:00:00+08:00",
        "status": "frozen",
        "job_id": 89941,
        "resource_mode": "cpu_only",
        "competition_id": "tabular-playground-series-may-2022",
        "visibility_mode": "PUBLIC_ONLY",
        "remote_root": root,
        "run_id": "job89941_may2022_test_run",
        "training": {
            "fold": 0,
            "seed": 42,
            "threads": 48,
            "max_rounds": 2500,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "early_stopping_rounds": 180,
        },
        "inputs": {
            "data_root": f"{root}/mlebench_official_data",
            "oof_bundle": f"{root}/runs/may2022_oof_ensemble.npz",
            "oof_sha256": "5" * 64,
            "oof_bytes": 12345,
        },
        "sources": source_records,
        "contracts": {
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_allowed": False,
            "writes_confined_to_remote_root": True,
            "diagnostic_not_medal": True,
        },
    }
    plan_path = tmp_path / "frozen_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return plan_path, payload


def decode_python_command(command: str) -> str:
    marker = "base64.b64decode('"
    encoded = command.split(marker, 1)[1].split("')", 1)[0]
    return base64.b64decode(encoded).decode("utf-8")


def test_validate_frozen_plan_accepts_exact_source_bytes_and_contract(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)

    plan = module.validate_frozen_plan(
        plan_path,
        project_root=tmp_path,
        allowed_root=module.ALLOWED_GPU_REMOTE_ROOT,
    )

    assert plan.run_id == "job89941_may2022_test_run"
    assert plan.bytes == plan_path.stat().st_size
    assert plan.sha256 == sha256(plan_path)
    assert len(plan.sources) == 6
    assert plan.sources[0].sha256 == sha256(
        tmp_path / "scripts" / "diagnose_may2022_cpu_feature_model.py"
    )


def test_validate_frozen_plan_fails_closed_on_source_drift(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    runner = tmp_path / "scripts" / "diagnose_may2022_cpu_feature_model.py"
    runner.write_text("print('changed after freeze')\n", encoding="utf-8")

    with pytest.raises(module.PlanValidationError, match="Frozen source drift"):
        module.validate_frozen_plan(plan_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(job_id=89942), "job_id"),
        (
            lambda payload: payload["contracts"].update(process_signals_allowed=True),
            "process_signals_allowed",
        ),
        (
            lambda payload: payload["inputs"].update(data_root="/tmp/outside"),
            "inputs.data_root",
        ),
        (
            lambda payload: payload["training"].update(threads=47),
            "training.threads",
        ),
    ],
)
def test_validate_frozen_plan_rejects_wrong_job_contract_path_or_threads(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    module = load_module()
    plan_path, payload = make_plan(tmp_path)
    mutator(payload)
    plan_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(module.PlanValidationError, match=message):
        module.validate_frozen_plan(plan_path, project_root=tmp_path)


def test_remote_path_confinement_blocks_escape_and_root_writes() -> None:
    module = load_module()
    root = module.ALLOWED_GPU_REMOTE_ROOT
    child = f"{root}/evomind_mle22/job89941/run.json"

    assert module.ensure_remote_path(child) == child
    assert module.ensure_remote_write_path(child) == child
    with pytest.raises(ValueError):
        module.ensure_remote_path(f"{root}/../someone_else")
    with pytest.raises(ValueError):
        module.ensure_remote_path(f"{root}-lookalike/file")
    with pytest.raises(ValueError):
        module.ensure_remote_path("relative/path")
    with pytest.raises(ValueError):
        module.ensure_remote_write_path(root)


def test_layout_argv_and_environment_are_cpu_only_and_confined(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    layout = module.build_remote_layout(plan)
    argv = module.build_runner_argv(plan, layout)
    environment = module.build_runtime_environment(plan, layout)
    root = PurePosixPath(module.ALLOWED_GPU_REMOTE_ROOT)

    for path in layout.write_paths():
        PurePosixPath(path).relative_to(root)
        assert PurePosixPath(path) != root
    assert argv[argv.index("--threads") + 1] == "48"
    assert argv[argv.index("--output") + 1] == layout.output
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert module.REMOTE_UNIFIED_SITE_PACKAGES in environment["PYTHONPATH"]
    assert environment["OMP_NUM_THREADS"] == "48"
    assert environment["OPENBLAS_NUM_THREADS"] == "48"
    assert environment["MKL_NUM_THREADS"] == "48"
    assert environment["NUMEXPR_NUM_THREADS"] == "48"
    assert environment["LIGHTGBM_NUM_THREADS"] == "48"
    PurePosixPath(environment["TMPDIR"]).relative_to(root)


def test_launch_is_nohup_nonblocking_exclusive_and_signal_free(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    layout = module.build_remote_layout(plan)
    source = decode_python_command(module.render_launch_command(plan, layout))
    lowered = source.lower()

    assert "os.o_excl" in lowered
    assert "['nohup', 'env']" in lowered
    assert "start_new_session=true" in lowered
    assert "stdin=subprocess.devnull" in lowered
    assert "duplicate_processes_started': 0" in lowered
    assert "'/proc'" in lowered
    assert "os.kill" not in lowered
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "taskkill" not in lowered
    assert "stop-process" not in lowered


def test_readonly_preflight_checks_exact_oof_before_remote_writes(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    source = decode_python_command(module.render_readonly_preflight_command(plan))

    assert "expected_oof_bytes = 12345" in source
    assert f"expected_oof_sha256 = {'5' * 64!r}" in source
    assert "resolve(strict=True)" in source
    assert "os.access(root, os.W_OK)" in source
    assert "import lightgbm, numpy, pandas, sklearn" in source
    assert "from sklearn.metrics import roc_auc_score" in source
    assert module.REMOTE_UNIFIED_SITE_PACKAGES in source
    assert "str(sklearn.__version__)" in source
    assert "importlib.metadata" not in source
    assert "(os.cpu_count() or 0) >= 48" in source
    assert ".mkdir(" not in source
    assert "write_text(" not in source


def test_all_rendered_remote_python_sources_compile(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    layout = module.build_remote_layout(plan)
    staged = module.remote_source_records(plan, layout)
    staged_with_plan = [module._plan_remote_record(plan, layout), *staged]
    commands = (
        module.render_readonly_preflight_command(plan),
        module.render_remote_hash_command(staged_with_plan),
        module.render_staged_import_smoke_command(layout),
        module.render_launch_command(plan, layout),
        module.render_status_command(plan, layout, staged),
    )

    for index, command in enumerate(commands):
        source = decode_python_command(command)
        compile(source, f"<remote-{index}>", "exec")


def test_staged_import_smoke_covers_transitive_russian_dependency(tmp_path: Path) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    layout = module.build_remote_layout(plan)
    source = decode_python_command(module.render_staged_import_smoke_command(layout))

    assert "from scripts import diagnose_may2022_cpu_feature_model" in source
    assert "from scripts import russian_transliteration" in source
    assert "from research_os import mlebench_phase_a" in source
    assert "is_relative_to(staged_root)" in source


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {
                "state_valid": False,
                "staged_integrity_passed": True,
                "result_present": False,
                "result_valid": False,
                "process_running": False,
            },
            "state_invalid",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": False,
                "result_present": False,
                "result_valid": False,
                "process_running": True,
            },
            "source_drift",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": True,
                "result_valid": False,
                "process_running": False,
            },
            "artifact_invalid",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": True,
                "result_valid": True,
                "process_running": True,
            },
            "finalizing",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": True,
                "result_valid": True,
                "process_running": False,
            },
            "completed",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": False,
                "result_valid": False,
                "process_running": True,
            },
            "running",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": False,
                "result_valid": False,
                "process_running": False,
                "launch_claimed": True,
            },
            "launching",
        ),
        (
            {
                "state_valid": True,
                "staged_integrity_passed": True,
                "result_present": False,
                "result_valid": False,
                "process_running": False,
            },
            "failed",
        ),
    ],
)
def test_status_derivation_is_fail_closed(arguments: dict, expected: str) -> None:
    module = load_module()
    assert module.derive_status(**arguments) == expected


def test_collected_result_requires_public_only_no_grader_and_exact_model_hash(
    tmp_path: Path,
) -> None:
    module = load_module()
    plan_path, _payload = make_plan(tmp_path)
    plan = module.validate_frozen_plan(plan_path, project_root=tmp_path)
    layout = module.build_remote_layout(plan)
    model_sha = "a" * 64
    payload = {
        "schema": module.RESULT_SCHEMA,
        "status": "completed",
        "competition_id": module.EXPECTED_COMPETITION,
        "visibility_mode": "PUBLIC_ONLY",
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "input_oof_bundle": {"sha256": plan.payload["inputs"]["oof_sha256"]},
        "model": {"path": layout.model, "sha256": model_sha},
        "runtime": {"threads": 48},
    }

    module.validate_collected_result(payload, plan, layout, model_sha256=model_sha)
    payload["official_grader_executed"] = True
    with pytest.raises(module.DeploymentError, match="official_grader_executed"):
        module.validate_collected_result(payload, plan, layout, model_sha256=model_sha)


def test_cli_supports_deploy_wait_collect_and_standalone_polling() -> None:
    module = load_module()
    deploy_args = module.parse_args(["deploy", "--wait", "--collect"])
    wait_args = module.parse_args(["wait", "--collect", "--poll-seconds", "1"])

    assert deploy_args.command == "deploy"
    assert deploy_args.wait is True
    assert deploy_args.collect is True
    assert wait_args.command == "wait"
    assert wait_args.collect is True
    assert wait_args.poll_seconds == 1
